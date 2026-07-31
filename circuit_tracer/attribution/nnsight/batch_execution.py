"""Backward-pass execution for one validated attribution batch."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from circuit_tracer.observability.events import (
    MemoryDelta,
    MemorySnapshot,
    MemorySnapshotAttrs,
    TraceObserver,
)
from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTapeEntry,
    tensor_nbytes,
)
from circuit_tracer.attribution.nnsight.resource_sampling import (
    should_sample_batch_resources,
)


_MEMORY_ATTR_KEYS: tuple[str, ...] = (
    "rss_current_gib",
    "proc_rss_anon_gib",
    "proc_rss_file_gib",
    "cgroup_memory_current_gib",
    "cgroup_memory_anon_gib",
    "cgroup_memory_file_gib",
    "cuda_allocated_gib",
    "cuda_reserved_gib",
)

_PIN_MEMORY_FALLBACK_MARKERS = (
    "pin_memory",
    "pinned memory",
    "pin memory",
    "out of memory",
    "cuda driver",
    "cuda error",
)

def _is_expected_pin_memory_failure(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _PIN_MEMORY_FALLBACK_MARKERS)


def slice_phase3_gradient_replay_batch(
    replay_gradients: torch.Tensor,
    *,
    layer: int,
    column_offset: int,
    batch_size: int,
) -> torch.Tensor:
    replay_grad = replay_gradients[layer, column_offset : column_offset + batch_size]
    if int(replay_grad.shape[0]) != int(batch_size):
        raise ValueError(
            "Phase-3 gradient replay batch slice shape mismatch "
            f"(offset={int(column_offset)}, expected={int(batch_size)}, "
            f"got={int(replay_grad.shape[0])})"
        )
    return replay_grad


@dataclass(frozen=True)
class BatchAttributionRequest:
    layers: torch.Tensor
    positions: torch.Tensor
    inject_values: torch.Tensor
    retain_graph: bool
    phase_label: str
    feature_column_range: tuple[int, int] | None
    include_nonfeature: bool

    def validate(self, *, session_capacity: int, active_features: int) -> int:
        batch_size = int(self.layers.numel())
        if self.layers.ndim != 1 or self.positions.ndim != 1:
            raise ValueError("layers and positions must be rank-1 tensors")
        if int(self.positions.numel()) != batch_size or int(self.inject_values.shape[0]) != batch_size:
            raise ValueError("layers, positions, and injected values must have equal batch size")
        if batch_size <= 0 or batch_size > session_capacity:
            raise ValueError(
                "compute_batch active lanes must be in [1, session capacity] "
                f"(active={batch_size}, capacity={session_capacity})"
            )
        if self.feature_column_range is not None:
            start, end = self.feature_column_range
            if start < 0 or end < start or end > active_features:
                raise ValueError("feature_column_range must lie within active feature columns")
        return batch_size


@dataclass(frozen=True)
class BatchExecutionResult:
    rows: torch.Tensor
    inject_values_nbytes: int
    batch_buffer_nbytes: int
    layers_in_batch: tuple[int, ...]
    chunked_feature_grad_window_peak: int
    feature_vjp_tape_entry: FeatureVjpTapeEntry | None = None


class BatchExecutionHost(Protocol):
    n_layers: int
    activation_matrix: torch.Tensor
    chunked_decoder_state: dict[str, torch.Tensor] | None
    diagnostic_mode: bool
    capture_phase3_gradients: bool
    phase3_gradient_replay_tensor: torch.Tensor | None
    phase3_gradient_replay_column_offset: int
    phase3_gradient_captures: list[dict[str, torch.Tensor | int]]
    _resid_activations: list[torch.Tensor]
    _feature_output_activations: list[torch.Tensor]
    _batch_buffer: torch.Tensor | None
    _row_size: int
    _chunked_feature_replay_window: int
    _produced_feature_range: tuple[int, int] | None
    _produce_nonfeature: bool
    _diagnostic_stats: dict[str, object]
    _trace_observer: TraceObserver | None
    _resource_sample_count_by_phase: dict[str, int]

    def _clear_saved_grads(self) -> None: ...
    def _materialize_tensor(self, tensor, *, device=None, dtype=None): ...
    def _flush_chunked_feature_grad_window(self, gradients, layers, **kwargs) -> None: ...
    def compute_feature_attributions(self, layer, grad, **kwargs) -> None: ...
    def compute_error_attributions(self, layer, grads) -> None: ...
    def compute_token_attributions(self, grads) -> None: ...
    def _add_layer_stat(self, key: str, layer: int, value: float) -> None: ...
    def _add_stat(self, key: str, value: float) -> None: ...
    def _emit_trace(self, event: str, **fields: object) -> None: ...
    def _record_telemetry_event(self, **kwargs) -> None: ...


def execute_backward_batch(
    host: BatchExecutionHost,
    request: BatchAttributionRequest,
    *,
    batch_call_index: int,
    defer_feature_vjps: bool = False,
) -> BatchExecutionResult:
    """Execute gradient injection and direct-effect extraction in canonical order."""
    execution_device = host._resid_activations[0].device
    batch_size = request.validate(
        session_capacity=int(host._resid_activations[0].shape[0]),
        active_features=int(host.activation_matrix._nnz()),
    )
    host._clear_saved_grads()
    layers = request.layers.to(device=execution_device, dtype=torch.long, non_blocking=request.layers.device.type == "cpu" and execution_device.type == "cuda")
    positions = request.positions.to(device=execution_device, dtype=torch.long, non_blocking=request.positions.device.type == "cpu" and execution_device.type == "cuda")
    inject_values = host._materialize_tensor(request.inject_values, device=execution_device, dtype=request.inject_values.dtype)
    feature_width = (
        request.feature_column_range[1] - request.feature_column_range[0]
        if request.feature_column_range is not None
        else int(host.activation_matrix._nnz())
    )
    nonfeature_width = host._row_size - int(host.activation_matrix._nnz())
    produced_row_size = feature_width + (nonfeature_width if request.include_nonfeature else 0)
    host._produced_feature_range = request.feature_column_range
    host._produce_nonfeature = request.include_nonfeature
    host._batch_buffer = torch.zeros(produced_row_size, batch_size, dtype=torch.float32, device=inject_values.device)
    batch_buffer_nbytes = int(host._batch_buffer.numel() * host._batch_buffer.element_size())
    batch_indices = torch.arange(batch_size, device=layers.device)

    def inject(grad_point, *, lane_indices, pos_indices, values) -> None:
        grads_out = grad_point.grad.clone()
        target_device = grads_out.device
        grads_out.index_put_(
            (lane_indices.to(target_device), pos_indices.to(target_device)),
            values.to(device=target_device, dtype=grads_out.dtype),
        )
        grad_point.grad = grads_out

    layers_in_batch = tuple(sorted(layers.unique().tolist(), reverse=True))
    chunked_feature_grads = {} if host.chunked_decoder_state is not None else None
    chunked_feature_grad_layers: list[int] = []
    capture = bool(host.capture_phase3_gradients and request.phase_label == "phase3_logits")
    replay = bool(host.phase3_gradient_replay_tensor is not None and request.phase_label == "phase3_logits")
    captured: list[torch.Tensor | None] | None = [None] * host.n_layers if capture else None
    deferred: list[torch.Tensor | None] | None = (
        [None] * host.n_layers if defer_feature_vjps else None
    )
    pinned_host_nbytes = 0
    pageable_host_nbytes = 0
    pin_fallback_count = 0
    pin_fallback_reason = None
    replay_gradients = host.phase3_gradient_replay_tensor
    peak = 0

    try:
        last_layer = max(layers_in_batch)
        active_residual = host._resid_activations[last_layer][:batch_size]
        with active_residual.backward(gradient=torch.zeros_like(active_residual), retain_graph=request.retain_graph):
            for layer in reversed(range(last_layer + 1)):
                if layer != last_layer:
                    grad = host._feature_output_activations[layer + 1].grad.detach()[:batch_size]
                    if replay:
                        assert replay_gradients is not None
                        donor = slice_phase3_gradient_replay_batch(
                            replay_gradients,
                            layer=layer,
                            column_offset=int(host.phase3_gradient_replay_column_offset),
                            batch_size=batch_size,
                        )
                        grad = donor.to(device=grad.device, dtype=grad.dtype)
                    if captured is not None and 0 <= layer < host.n_layers:
                        captured[layer] = grad.detach().to(device="cpu", dtype=torch.float32).contiguous()
                    feature_start = time.perf_counter()
                    if deferred is not None:
                        if execution_device.type == "cuda":
                            try:
                                staged = torch.empty_like(
                                    grad,
                                    device="cpu",
                                    memory_format=torch.contiguous_format,
                                    pin_memory=True,
                                )
                                staged.copy_(grad.detach(), non_blocking=False)
                                pinned_host_nbytes += tensor_nbytes(staged)
                            except RuntimeError as exc:  # pragma: no cover - host dependent
                                if not _is_expected_pin_memory_failure(exc):
                                    raise
                                staged = grad.detach().to(device="cpu").contiguous()
                                pageable_host_nbytes += tensor_nbytes(staged)
                                pin_fallback_count += 1
                                if pin_fallback_reason is None:
                                    pin_fallback_reason = f"{type(exc).__name__}: {exc}"
                        else:
                            staged = grad.detach().to(device="cpu").contiguous()
                            pageable_host_nbytes += tensor_nbytes(staged)
                        deferred[layer] = staged
                    elif chunked_feature_grads is None:
                        host.compute_feature_attributions(layer, grad, phase_label=request.phase_label, batch_index=batch_call_index)
                        if host.diagnostic_mode:
                            host._add_layer_stat("feature_attr_seconds_by_layer", layer, time.perf_counter() - feature_start)
                    else:
                        chunked_feature_grads[layer] = grad
                        chunked_feature_grad_layers.append(layer)
                        peak = max(peak, len(chunked_feature_grad_layers))
                        if host.diagnostic_mode:
                            old_peak = cast(float, host._diagnostic_stats["chunked_attr_grad_window_peak"])
                            host._diagnostic_stats["chunked_attr_grad_window_peak"] = max(old_peak, float(len(chunked_feature_grad_layers)))
                        if len(chunked_feature_grad_layers) >= host._chunked_feature_replay_window:
                            host._flush_chunked_feature_grad_window(chunked_feature_grads, chunked_feature_grad_layers, phase_label=request.phase_label, batch_index=batch_call_index)
                    error_start = time.perf_counter()
                    host.compute_error_attributions(layer, grad)
                    if host.diagnostic_mode:
                        host._add_layer_stat("error_attr_seconds_by_layer", layer, time.perf_counter() - error_start)
                mask = layers == layer
                if mask.any():
                    inject(grad_point=host._resid_activations[layer], lane_indices=batch_indices[mask], pos_indices=positions[mask], values=inject_values[mask])

            token_start = time.perf_counter()
            host.compute_token_attributions(host._feature_output_activations[0].grad[:batch_size])
            if host.diagnostic_mode:
                host._add_stat("token_attr_seconds", time.perf_counter() - token_start)
            if chunked_feature_grads is not None and deferred is None:
                host._flush_chunked_feature_grad_window(chunked_feature_grads, chunked_feature_grad_layers, phase_label=request.phase_label, batch_index=batch_call_index)
    finally:
        host._clear_saved_grads()

    if captured is not None:
        present = [gradient is not None for gradient in captured]
        if any(present):
            sample = next(gradient for gradient in captured if gradient is not None)
            assert sample is not None
            host.phase3_gradient_captures.append({
                "batch_call_index": int(batch_call_index),
                "layer_mask": torch.tensor(present, dtype=torch.bool),
                "gradients": torch.stack([torch.zeros_like(sample) if gradient is None else gradient for gradient in captured], dim=0),
            })

    buffer, host._batch_buffer = host._batch_buffer, None
    assert buffer is not None
    host._produced_feature_range = None
    host._produce_nonfeature = True
    tape_entry = None
    if deferred is not None:
        gradient_bytes = sum(
            tensor_nbytes(gradient) for gradient in deferred if gradient is not None
        )
        device_gradient_bytes = sum(
            int(gradient.numel() * buffer.element_size())
            for gradient in deferred
            if gradient is not None
        )
        row_nbytes = tensor_nbytes(buffer)
        tape_entry = FeatureVjpTapeEntry(
            batch_call_index=int(batch_call_index),
            gradients=tuple(deferred),
            row_buffer=buffer,
            batch_size=batch_size,
            host_nbytes=gradient_bytes,
            device_nbytes=device_gradient_bytes,
            row_nbytes=row_nbytes,
            total_nbytes=gradient_bytes + device_gradient_bytes + row_nbytes,
            pinned_host_nbytes=pinned_host_nbytes,
            pageable_host_nbytes=pageable_host_nbytes,
            pin_fallback_count=pin_fallback_count,
            pin_fallback_reason=pin_fallback_reason,
        )
    return BatchExecutionResult(
        rows=buffer.T[:batch_size],
        inject_values_nbytes=int(inject_values.numel() * inject_values.element_size()),
        batch_buffer_nbytes=batch_buffer_nbytes,
        layers_in_batch=layers_in_batch,
        chunked_feature_grad_window_peak=peak,
        feature_vjp_tape_entry=tape_entry,
    )


def execute_observed_batch(
    host: BatchExecutionHost,
    request: BatchAttributionRequest,
    *,
    batch_call_index: int,
    defer_feature_vjps: bool = False,
) -> BatchExecutionResult:
    """Execute a batch while owning resource sampling and typed event rendering."""
    batch_size = request.validate(
        session_capacity=int(host._resid_activations[0].shape[0]),
        active_features=int(host.activation_matrix._nnz()),
    )
    execution_device = host._resid_activations[0].device
    observer = host._trace_observer
    phase_batch_index = host._resource_sample_count_by_phase.get(request.phase_label, 0) + 1
    host._resource_sample_count_by_phase[request.phase_label] = phase_batch_index
    resource_sampled = should_sample_batch_resources(
        phase_label=request.phase_label,
        phase_batch_index=phase_batch_index,
        retain_graph=request.retain_graph,
    )
    memory_before = (
        cast(dict[str, object], observer.observe(MemorySnapshot(execution_device)))
        if observer is not None and resource_sampled
        else {}
    )
    started = time.perf_counter()
    unique_layers = int(request.layers.unique().numel())
    input_nbytes = int(request.inject_values.numel() * request.inject_values.element_size())
    feature_width = (
        request.feature_column_range[1] - request.feature_column_range[0]
        if request.feature_column_range is not None
        else int(host.activation_matrix._nnz())
    )
    nonfeature_width = host._row_size - int(host.activation_matrix._nnz())
    produced_width = feature_width + (nonfeature_width if request.include_nonfeature else 0)
    planned_buffer_nbytes = produced_width * batch_size * torch.tensor([], dtype=torch.float32).element_size()
    host._emit_trace(
        "compute_batch.start",
        phase=request.phase_label,
        phase_batch_index=phase_batch_index,
        resource_sampled=resource_sampled,
        batch_nodes=batch_size,
        unique_layers=unique_layers,
        retain_graph=request.retain_graph,
        inject_values_input_nbytes=input_nbytes,
        planned_batch_buffer_nbytes=planned_buffer_nbytes,
        chunked_feature_replay_window=int(host._chunked_feature_replay_window),
        **(
            cast(
                dict[str, object],
                observer.observe(
                    MemorySnapshotAttrs(
                        memory_before, keys=_MEMORY_ATTR_KEYS, prefix="memory_before"
                    )
                ),
            )
            if observer is not None and resource_sampled
            else {}
        ),
    )
    result = execute_backward_batch(
        host,
        request,
        batch_call_index=batch_call_index,
        defer_feature_vjps=defer_feature_vjps,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    memory_after = (
        cast(dict[str, object], observer.observe(MemorySnapshot(execution_device)))
        if observer is not None and resource_sampled
        else {}
    )
    if host.diagnostic_mode:
        host._add_stat("compute_batch_calls", 1)
        elapsed = elapsed_ms / 1000.0
        host._add_stat("compute_batch_seconds", elapsed)
        phase_bucket = cast(
            dict[str, float], host._diagnostic_stats.setdefault("compute_batch_seconds_by_phase", {})
        )
        phase_bucket[request.phase_label] = phase_bucket.get(request.phase_label, 0.0) + elapsed
    memory_attrs = (
        cast(
            dict[str, object],
            observer.observe(
                MemoryDelta(before=memory_before, after=memory_after, keys=_MEMORY_ATTR_KEYS)
            ),
        )
        if observer is not None and resource_sampled
        else {}
    )
    host._record_telemetry_event(
        scope="batch",
        name="context.compute_batch",
        phase=request.phase_label,
        batch_index=batch_call_index,
        elapsed_ms=elapsed_ms,
        attrs={
            "batch_nodes": batch_size,
            "batch_size": batch_size,
            "phase_batch_index": phase_batch_index,
            "resource_sampled": resource_sampled,
            "row_size": int(host._row_size),
            "unique_layers": len(result.layers_in_batch),
            "retain_graph": request.retain_graph,
            "chunked_decoder": host.chunked_decoder_state is not None,
            "inject_values_input_nbytes": input_nbytes,
            "inject_values_nbytes": result.inject_values_nbytes,
            "batch_buffer_nbytes": result.batch_buffer_nbytes,
            "chunked_feature_replay_window": int(host._chunked_feature_replay_window),
            "chunked_feature_grad_window_peak": result.chunked_feature_grad_window_peak,
            "feature_vjp_deferred": defer_feature_vjps,
            **memory_attrs,
        },
    )
    host._emit_trace(
        "compute_batch.done",
        phase=request.phase_label,
        phase_batch_index=phase_batch_index,
        resource_sampled=resource_sampled,
        batch_nodes=batch_size,
        unique_layers=unique_layers,
        retain_graph=request.retain_graph,
        inject_values_nbytes=result.inject_values_nbytes,
        batch_buffer_nbytes=result.batch_buffer_nbytes,
        chunked_feature_replay_window=int(host._chunked_feature_replay_window),
        chunked_feature_grad_window_peak=result.chunked_feature_grad_window_peak,
        elapsed_s=f"{elapsed_ms / 1000.0:.2f}",
        elapsed_ms=elapsed_ms,
        **memory_attrs,
    )
    return result
