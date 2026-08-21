"""Shared attribution-row consumption for independent backward engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import torch

from circuit_tracer.attribution.nnsight.batch_contract import (
    BatchAttributionRequest,
    BatchExecutionHost,
    BatchExecutionResult,
)
from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTapeEntry,
    tensor_nbytes,
)
from circuit_tracer.observability.errors import safe_exception_message


_PIN_MEMORY_FALLBACK_MARKERS = (
    "pin_memory",
    "pinned memory",
    "pin memory",
    "out of memory",
    "cuda driver",
    "cuda error",
)


def _is_expected_pin_memory_failure(error: RuntimeError) -> bool:
    message = safe_exception_message(error).lower()
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


@dataclass
class AttributionBatchWorkspace:
    """Own one row buffer and consume already-computed intermediate VJPs."""

    host: BatchExecutionHost
    request: BatchAttributionRequest
    batch_call_index: int
    defer_feature_vjps: bool
    execution_device: torch.device
    batch_size: int
    layers: torch.Tensor
    positions: torch.Tensor
    inject_values: torch.Tensor
    batch_buffer_nbytes: int
    layers_in_batch: tuple[int, ...]
    chunked_feature_grads: dict[int, torch.Tensor] | None
    chunked_feature_grad_layers: list[int]
    captured: list[torch.Tensor | None] | None
    deferred: list[torch.Tensor | None] | None
    pinned_host_nbytes: int = 0
    pageable_host_nbytes: int = 0
    pin_fallback_count: int = 0
    pin_fallback_reason: str | None = None
    peak: int = 0
    _finished: bool = False

    @classmethod
    def begin(
        cls,
        host: BatchExecutionHost,
        request: BatchAttributionRequest,
        *,
        batch_call_index: int,
        defer_feature_vjps: bool,
        batch_capacity: int,
    ) -> "AttributionBatchWorkspace":
        execution_device = host._resid_activations[0].device
        batch_size = request.validate(
            batch_capacity=batch_capacity,
            active_features=int(host.activation_matrix._nnz()),
        )
        host._clear_saved_grads()
        layers = request.layers.to(
            device=execution_device,
            dtype=torch.long,
            non_blocking=request.layers.device.type == "cpu" and execution_device.type == "cuda",
        )
        positions = request.positions.to(
            device=execution_device,
            dtype=torch.long,
            non_blocking=(
                request.positions.device.type == "cpu" and execution_device.type == "cuda"
            ),
        )
        inject_values = host._materialize_tensor(
            request.inject_values,
            device=execution_device,
            dtype=request.inject_values.dtype,
        )
        feature_width = (
            request.feature_column_range[1] - request.feature_column_range[0]
            if request.feature_column_range is not None
            else int(host.activation_matrix._nnz())
        )
        nonfeature_width = host._row_size - int(host.activation_matrix._nnz())
        produced_row_size = feature_width + (nonfeature_width if request.include_nonfeature else 0)
        host._produced_feature_range = request.feature_column_range
        host._produce_nonfeature = request.include_nonfeature
        host._batch_buffer = torch.zeros(
            produced_row_size,
            batch_size,
            dtype=torch.float32,
            device=inject_values.device,
        )
        batch_buffer_nbytes = int(host._batch_buffer.numel() * host._batch_buffer.element_size())
        capture = bool(host.capture_phase3_gradients and request.phase_label == "phase3_logits")
        return cls(
            host=host,
            request=request,
            batch_call_index=int(batch_call_index),
            defer_feature_vjps=bool(defer_feature_vjps),
            execution_device=execution_device,
            batch_size=batch_size,
            layers=layers,
            positions=positions,
            inject_values=inject_values,
            batch_buffer_nbytes=batch_buffer_nbytes,
            layers_in_batch=tuple(sorted(layers.unique().tolist(), reverse=True)),
            chunked_feature_grads={} if host.chunked_decoder_state is not None else None,
            chunked_feature_grad_layers=[],
            captured=[None] * host.n_layers if capture else None,
            deferred=[None] * host.n_layers if defer_feature_vjps else None,
        )

    def _stage_deferred_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        if self.execution_device.type == "cuda":
            try:
                staged = torch.empty_like(
                    grad,
                    device="cpu",
                    memory_format=torch.contiguous_format,
                    pin_memory=True,
                )
                staged.copy_(grad.detach(), non_blocking=False)
                self.pinned_host_nbytes += tensor_nbytes(staged)
                return staged
            except RuntimeError as exc:  # pragma: no cover - host dependent
                if not _is_expected_pin_memory_failure(exc):
                    raise
                staged = grad.detach().to(device="cpu").contiguous()
                self.pageable_host_nbytes += tensor_nbytes(staged)
                self.pin_fallback_count += 1
                if self.pin_fallback_reason is None:
                    self.pin_fallback_reason = (
                        f"{type(exc).__module__}.{type(exc).__qualname__}: "
                        f"{safe_exception_message(exc)}"
                    )
                return staged
        staged = grad.detach().to(device="cpu").contiguous()
        self.pageable_host_nbytes += tensor_nbytes(staged)
        return staged

    def consume_intermediate_gradient(self, layer: int, grad: torch.Tensor) -> None:
        replay = bool(
            self.host.phase3_gradient_replay_tensor is not None
            and self.request.phase_label == "phase3_logits"
        )
        if replay:
            replay_gradients = self.host.phase3_gradient_replay_tensor
            assert replay_gradients is not None
            donor = slice_phase3_gradient_replay_batch(
                replay_gradients,
                layer=layer,
                column_offset=int(self.host.phase3_gradient_replay_column_offset),
                batch_size=self.batch_size,
            )
            grad = donor.to(device=grad.device, dtype=grad.dtype)
        if self.captured is not None and 0 <= layer < self.host.n_layers:
            self.captured[layer] = grad.detach().to(device="cpu", dtype=torch.float32).contiguous()
        feature_start = time.perf_counter()
        if self.deferred is not None:
            self.deferred[layer] = self._stage_deferred_gradient(grad)
        elif self.chunked_feature_grads is None:
            self.host.compute_feature_attributions(
                layer,
                grad,
                phase_label=self.request.phase_label,
                batch_index=self.batch_call_index,
            )
            if self.host.diagnostic_mode:
                self.host._add_layer_stat(
                    "feature_attr_seconds_by_layer",
                    layer,
                    time.perf_counter() - feature_start,
                )
        else:
            self.chunked_feature_grads[layer] = grad
            self.chunked_feature_grad_layers.append(layer)
            self.peak = max(self.peak, len(self.chunked_feature_grad_layers))
            if self.host.diagnostic_mode:
                old_peak = cast(
                    float,
                    self.host._diagnostic_stats["chunked_attr_grad_window_peak"],
                )
                self.host._diagnostic_stats["chunked_attr_grad_window_peak"] = max(
                    old_peak,
                    float(len(self.chunked_feature_grad_layers)),
                )
            if len(self.chunked_feature_grad_layers) >= self.host._chunked_feature_replay_window:
                self.host._flush_chunked_feature_grad_window(
                    self.chunked_feature_grads,
                    self.chunked_feature_grad_layers,
                    phase_label=self.request.phase_label,
                    batch_index=self.batch_call_index,
                )
        error_start = time.perf_counter()
        self.host.compute_error_attributions(layer, grad)
        if self.host.diagnostic_mode:
            self.host._add_layer_stat(
                "error_attr_seconds_by_layer",
                layer,
                time.perf_counter() - error_start,
            )

    def consume_token_gradient(self, grad: torch.Tensor) -> None:
        token_start = time.perf_counter()
        self.host.compute_token_attributions(grad)
        if self.host.diagnostic_mode:
            self.host._add_stat("token_attr_seconds", time.perf_counter() - token_start)

    def finish(self, *, engine_attrs: dict[str, object]) -> BatchExecutionResult:
        if self._finished:
            raise RuntimeError("attribution batch workspace is already finished")
        if self.chunked_feature_grads is not None and self.deferred is None:
            self.host._flush_chunked_feature_grad_window(
                self.chunked_feature_grads,
                self.chunked_feature_grad_layers,
                phase_label=self.request.phase_label,
                batch_index=self.batch_call_index,
            )
        if self.captured is not None:
            present = [gradient is not None for gradient in self.captured]
            if any(present):
                sample = next(gradient for gradient in self.captured if gradient is not None)
                assert sample is not None
                self.host.phase3_gradient_captures.append(
                    {
                        "batch_call_index": int(self.batch_call_index),
                        "layer_mask": torch.tensor(present, dtype=torch.bool),
                        "gradients": torch.stack(
                            [
                                torch.zeros_like(sample) if gradient is None else gradient
                                for gradient in self.captured
                            ],
                            dim=0,
                        ),
                    }
                )
        buffer, self.host._batch_buffer = self.host._batch_buffer, None
        assert buffer is not None
        self.host._produced_feature_range = None
        self.host._produce_nonfeature = True
        tape_entry = None
        if self.deferred is not None:
            gradient_bytes = sum(
                tensor_nbytes(gradient) for gradient in self.deferred if gradient is not None
            )
            device_gradient_bytes = sum(
                int(gradient.numel() * buffer.element_size())
                for gradient in self.deferred
                if gradient is not None
            )
            row_nbytes = tensor_nbytes(buffer)
            tape_entry = FeatureVjpTapeEntry(
                batch_call_index=self.batch_call_index,
                gradients=tuple(self.deferred),
                row_buffer=buffer,
                batch_size=self.batch_size,
                host_nbytes=gradient_bytes,
                device_nbytes=device_gradient_bytes,
                row_nbytes=row_nbytes,
                total_nbytes=gradient_bytes + device_gradient_bytes + row_nbytes,
                pinned_host_nbytes=self.pinned_host_nbytes,
                pageable_host_nbytes=self.pageable_host_nbytes,
                pin_fallback_count=self.pin_fallback_count,
                pin_fallback_reason=self.pin_fallback_reason,
            )
        self._finished = True
        return BatchExecutionResult(
            rows=buffer.T[: self.batch_size],
            inject_values_nbytes=int(
                self.inject_values.numel() * self.inject_values.element_size()
            ),
            batch_buffer_nbytes=self.batch_buffer_nbytes,
            layers_in_batch=self.layers_in_batch,
            chunked_feature_grad_window_peak=self.peak,
            feature_vjp_tape_entry=tape_entry,
            engine_attrs=dict(engine_attrs),
        )

    def abort(self) -> None:
        if self._finished:
            return
        self.host._batch_buffer = None
        self.host._produced_feature_range = None
        self.host._produce_nonfeature = True
        self._finished = True
