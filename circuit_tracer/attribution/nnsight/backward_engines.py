"""Pluggable backward engines for attribution-row production."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch

from circuit_tracer.attribution.nnsight.batch_contract import (
    BatchAttributionRequest,
    BatchExecutionHost,
    BatchExecutionResult,
)
from circuit_tracer.attribution.nnsight.batch_workspace import AttributionBatchWorkspace
from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTapeEntry,
    tensor_nbytes,
)
from circuit_tracer.observability.errors import safe_exception_message
from circuit_tracer.tracing.plan import (
    BackwardEngineMode,
    BackwardExecutionTopology,
    ForwardGraphMode,
    VjpKernelMode,
)


class BackwardEngineExecutionError(RuntimeError):
    """Stage-bearing failure that preserves the original engine exception as cause."""

    def __init__(
        self,
        *,
        mode: BackwardEngineMode,
        stage: str,
        cause: BaseException,
        source_layer: int | None = None,
    ) -> None:
        self.mode = mode
        self.stage = stage
        self.source_layer = source_layer
        self.cause_type = f"{type(cause).__module__}.{type(cause).__qualname__}"
        location = "" if source_layer is None else f", source_layer={source_layer}"
        super().__init__(
            f"backward engine {mode!r} failed at stage {stage!r}{location}: "
            f"{safe_exception_message(cause)}"
        )


class BackwardBatchEngine(Protocol):
    @property
    def mode(self) -> BackwardEngineMode: ...

    @property
    def batch_capacity(self) -> int: ...

    @property
    def forward_lane_count(self) -> int: ...

    @property
    def forward_graph_mode(self) -> ForwardGraphMode: ...

    @property
    def vjp_kernel_mode(self) -> VjpKernelMode: ...

    def execute(
        self,
        host: BatchExecutionHost,
        request: BatchAttributionRequest,
        *,
        batch_call_index: int,
        defer_feature_vjps: bool = False,
    ) -> BatchExecutionResult: ...


def _build_vjp_execution_evidence(
    *,
    kernel_mode: VjpKernelMode,
    invocation_count: int,
    source_layer_group_count: int,
) -> dict[str, object]:
    """Describe the VJP API that actually completed without overstating fallback visibility."""
    is_batched = kernel_mode == "autograd_batched"
    return {
        "schema_version": 1,
        "requested_path": str(kernel_mode),
        "effective_invocation": "torch.autograd.grad",
        "is_grads_batched": bool(is_batched),
        "successful_invocation_count": int(invocation_count),
        "source_layer_group_count": int(source_layer_group_count),
        "observation_method": "direct_call_contract_and_success",
        "fallback_state": "unknown" if is_batched else "not_applicable",
        "fallback_state_reason": (
            "pytorch_has_no_programmatic_per_invocation_vmap_fallback_signal"
            if is_batched
            else "serial_autograd_does_not_use_vmap"
        ),
    }


def _require_saved_gradient(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    gradient = tensor.grad
    if gradient is None:
        raise RuntimeError(f"NNSight backward did not provide required gradient: {label}")
    return gradient


@dataclass(frozen=True)
class DuplicatedLaneBackwardEngine:
    """Established NNSight backward path with one graph lane per VJP row."""

    batch_capacity: int
    mode: BackwardEngineMode = field(default="duplicated_lanes", init=False)

    def __post_init__(self) -> None:
        if self.batch_capacity <= 0:
            raise ValueError("backward batch capacity must be positive")

    @property
    def forward_graph_mode(self) -> ForwardGraphMode:
        return "logical_capacity"

    @property
    def vjp_kernel_mode(self) -> VjpKernelMode:
        return "nnsight_injected"

    @property
    def forward_lane_count(self) -> int:
        return self.batch_capacity

    def execute(
        self,
        host: BatchExecutionHost,
        request: BatchAttributionRequest,
        *,
        batch_call_index: int,
        defer_feature_vjps: bool = False,
    ) -> BatchExecutionResult:
        physical_lanes = int(host._resid_activations[0].shape[0])
        if physical_lanes != self.forward_lane_count:
            raise RuntimeError(
                "duplicated-lane graph width does not match configured backward capacity "
                f"(physical={physical_lanes}, configured={self.forward_lane_count})"
            )
        workspace = AttributionBatchWorkspace.begin(
            host,
            request,
            batch_call_index=batch_call_index,
            defer_feature_vjps=defer_feature_vjps,
            batch_capacity=self.batch_capacity,
        )
        layers = workspace.layers
        positions = workspace.positions
        inject_values = workspace.inject_values
        batch_indices = torch.arange(workspace.batch_size, device=layers.device)

        def inject(grad_point, *, lane_indices, pos_indices, values) -> None:
            grads_out = grad_point.grad.clone()
            target_device = grads_out.device
            grads_out.index_put_(
                (lane_indices.to(target_device), pos_indices.to(target_device)),
                values.to(device=target_device, dtype=grads_out.dtype),
            )
            grad_point.grad = grads_out

        try:
            last_layer = max(workspace.layers_in_batch)
            active_residual = host._resid_activations[last_layer][: workspace.batch_size]
            backward_started = time.perf_counter()
            with active_residual.backward(
                gradient=torch.zeros_like(active_residual),
                retain_graph=request.retain_graph,
            ):
                for layer in reversed(range(last_layer + 1)):
                    if layer != last_layer:
                        grad = _require_saved_gradient(
                            host._feature_output_activations[layer + 1],
                            label=f"feature_output[{layer + 1}]",
                        ).detach()[: workspace.batch_size]
                        workspace.consume_intermediate_gradient(layer, grad)
                    mask = layers == layer
                    if mask.any():
                        inject(
                            host._resid_activations[layer],
                            lane_indices=batch_indices[mask],
                            pos_indices=positions[mask],
                            values=inject_values[mask],
                        )
                workspace.consume_token_gradient(
                    _require_saved_gradient(
                        host._feature_output_activations[0],
                        label="feature_output[0]",
                    )[: workspace.batch_size]
                )
            return workspace.finish(
                engine_attrs={
                    "backward_engine_mode": self.mode,
                    "forward_graph_mode": self.forward_graph_mode,
                    "vjp_kernel_mode": self.vjp_kernel_mode,
                    "forward_lane_count": self.forward_lane_count,
                    "backward_batch_capacity": self.batch_capacity,
                    "autograd_call_count": 1,
                    "autograd_elapsed_ms": (time.perf_counter() - backward_started) * 1000.0,
                }
            )
        except BaseException:
            workspace.abort()
            raise
        finally:
            host._clear_saved_grads()


def _select_rows(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, indices.to(device=tensor.device, dtype=torch.long))


def _normalize_batched_gradient(
    gradient: torch.Tensor | None,
    *,
    input_tensor: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if gradient is None:
        raise RuntimeError(
            "autograd did not provide a required attribution gradient "
            f"(input_shape={tuple(input_tensor.shape)}, group_size={group_size})"
        )
    expected_shape = (group_size, *tuple(input_tensor.shape))
    if tuple(gradient.shape) != expected_shape:
        raise RuntimeError(
            "batched VJP gradient shape mismatch "
            f"(expected={expected_shape}, actual={tuple(gradient.shape)})"
        )
    if int(input_tensor.shape[0]) != 1:
        raise RuntimeError(
            "single-forward batched VJP requires exactly one physical forward lane "
            f"(got {int(input_tensor.shape[0])})"
        )
    return gradient[:, 0]


def _merge_feature_vjp_tape_entries(
    *,
    entries: list[tuple[torch.Tensor, FeatureVjpTapeEntry]],
    batch_call_index: int,
    batch_size: int,
    combined_row_buffer: torch.Tensor,
) -> FeatureVjpTapeEntry:
    max_layers = max(len(entry.gradients) for _, entry in entries)
    gradients: list[torch.Tensor | None] = []
    for layer in range(max_layers):
        samples = [
            entry.gradients[layer]
            for _, entry in entries
            if layer < len(entry.gradients) and entry.gradients[layer] is not None
        ]
        if not samples:
            gradients.append(None)
            continue
        sample = cast(torch.Tensor, samples[0])
        combined = torch.zeros(
            (batch_size, *sample.shape[1:]),
            device="cpu",
            dtype=sample.dtype,
        )
        for row_indices, entry in entries:
            gradient = entry.gradients[layer]
            if gradient is not None:
                combined.index_copy_(
                    0,
                    row_indices.to(device="cpu", dtype=torch.long),
                    gradient,
                )
        gradients.append(combined)
    host_nbytes = sum(tensor_nbytes(gradient) for gradient in gradients if gradient is not None)
    device_nbytes = sum(
        int(gradient.numel() * combined_row_buffer.element_size())
        for gradient in gradients
        if gradient is not None
    )
    row_nbytes = tensor_nbytes(combined_row_buffer)
    fallback_count = sum(entry.pin_fallback_count for _, entry in entries)
    fallback_reason = next(
        (
            entry.pin_fallback_reason
            for _, entry in entries
            if entry.pin_fallback_reason is not None
        ),
        None,
    )
    for _, entry in entries:
        entry.clear()
    return FeatureVjpTapeEntry(
        batch_call_index=int(batch_call_index),
        gradients=tuple(gradients),
        row_buffer=combined_row_buffer,
        batch_size=int(batch_size),
        host_nbytes=host_nbytes,
        device_nbytes=device_nbytes,
        row_nbytes=row_nbytes,
        total_nbytes=host_nbytes + device_nbytes + row_nbytes,
        pinned_host_nbytes=0,
        pageable_host_nbytes=host_nbytes,
        pin_fallback_count=fallback_count,
        pin_fallback_reason=fallback_reason,
    )


def _clear_feature_vjp_tape_entries(
    entries: list[tuple[torch.Tensor, FeatureVjpTapeEntry]],
) -> None:
    for _, entry in entries:
        entry.clear()


@dataclass(frozen=True)
class SingleForwardBatchedVjpEngine:
    """One forward graph lane with source-layer-grouped autograd VJPs."""

    batch_capacity: int
    mode: BackwardEngineMode = field(
        default="single_forward_batched_vjp",
        init=False,
    )
    forward_lane_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.batch_capacity <= 0:
            raise ValueError("backward batch capacity must be positive")

    @property
    def forward_graph_mode(self) -> ForwardGraphMode:
        return "single_lane"

    @property
    def vjp_kernel_mode(self) -> VjpKernelMode:
        return "autograd_batched"

    def _compute_gradients(
        self,
        *,
        output: torch.Tensor,
        inputs: tuple[torch.Tensor, ...],
        cotangent: torch.Tensor,
        retain_graph: bool,
    ) -> tuple[tuple[torch.Tensor | None, ...], int]:
        gradients = torch.autograd.grad(
            outputs=(output,),
            inputs=inputs,
            grad_outputs=(cotangent,),
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
            is_grads_batched=True,
            materialize_grads=False,
        )
        return gradients, 1

    def execute(
        self,
        host: BatchExecutionHost,
        request: BatchAttributionRequest,
        *,
        batch_call_index: int,
        defer_feature_vjps: bool = False,
    ) -> BatchExecutionResult:
        batch_size = request.validate(
            batch_capacity=self.batch_capacity,
            active_features=int(host.activation_matrix._nnz()),
        )
        physical_lanes = int(host._resid_activations[0].shape[0])
        if physical_lanes != self.forward_lane_count:
            raise RuntimeError(
                "single-forward VJP graph width mismatch "
                f"(physical={physical_lanes}, required={self.forward_lane_count})"
            )
        if request.phase_label == "phase3_logits" and (
            host.phase3_gradient_replay_tensor is not None
        ):
            raise RuntimeError(
                "single-forward VJP does not accept Phase-3 gradient replay; "
                "select a native backward run or duplicated_lanes"
            )

        source_layers = tuple(
            sorted((int(value) for value in request.layers.unique().tolist()), reverse=True)
        )
        if request.phase_label == "phase3_logits" and host.capture_phase3_gradients:
            if len(source_layers) != 1:
                raise RuntimeError(
                    "Phase-3 gradient capture requires one source layer per batched-VJP call"
                )

        combined_row_buffer: torch.Tensor | None = None
        tape_entries: list[tuple[torch.Tensor, FeatureVjpTapeEntry]] = []
        group_results: list[BatchExecutionResult] = []
        cotangent_build_ms = 0.0
        autograd_ms = 0.0
        autograd_call_count = 0
        contraction_ms = 0.0
        cotangent_total_nbytes = 0
        cotangent_peak_nbytes = 0
        max_group_size = 0

        for group_number, source_layer in enumerate(source_layers):
            row_indices = (request.layers == source_layer).nonzero(as_tuple=False).flatten()
            group_size = int(row_indices.numel())
            max_group_size = max(max_group_size, group_size)
            group_request = BatchAttributionRequest(
                layers=_select_rows(request.layers, row_indices),
                positions=_select_rows(request.positions, row_indices),
                inject_values=_select_rows(request.inject_values, row_indices),
                retain_graph=request.retain_graph,
                phase_label=request.phase_label,
                feature_column_range=request.feature_column_range,
                include_nonfeature=request.include_nonfeature,
            )
            workspace: AttributionBatchWorkspace | None = None
            stage = "workspace_prepare"
            try:
                workspace = AttributionBatchWorkspace.begin(
                    host,
                    group_request,
                    batch_call_index=batch_call_index,
                    defer_feature_vjps=defer_feature_vjps,
                    batch_capacity=self.batch_capacity,
                )
                if source_layer < 0 or source_layer >= len(host._resid_activations):
                    raise ValueError(
                        f"source layer {source_layer} is outside cached residual range"
                    )
                output = host._resid_activations[source_layer]
                if output.ndim != 3 or int(output.shape[0]) != 1:
                    raise RuntimeError(
                        "single-forward batched VJP expects residual shape (1, positions, d_model) "
                        f"(got {tuple(output.shape)})"
                    )
                stage = "cotangent_build"
                build_started = time.perf_counter()
                cotangent = torch.zeros(
                    (group_size, *output.shape),
                    device=output.device,
                    dtype=output.dtype,
                )
                group_positions = workspace.positions.to(device=output.device, dtype=torch.long)
                group_inject = workspace.inject_values.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                cotangent[
                    torch.arange(group_size, device=output.device),
                    0,
                    group_positions,
                ] = group_inject
                build_elapsed = (time.perf_counter() - build_started) * 1000.0
                cotangent_build_ms += build_elapsed
                cotangent_nbytes = tensor_nbytes(cotangent)
                cotangent_total_nbytes += cotangent_nbytes
                cotangent_peak_nbytes = max(cotangent_peak_nbytes, cotangent_nbytes)

                inputs = tuple(host._feature_output_activations[: source_layer + 1])
                stage = "autograd_grad"
                autograd_started = time.perf_counter()
                gradients, group_autograd_call_count = self._compute_gradients(
                    output=output,
                    inputs=inputs,
                    cotangent=cotangent,
                    retain_graph=(request.retain_graph or group_number < len(source_layers) - 1),
                )
                autograd_call_count += group_autograd_call_count
                autograd_ms += (time.perf_counter() - autograd_started) * 1000.0

                stage = "contraction"
                contraction_started = time.perf_counter()
                for layer in range(source_layer):
                    gradient = _normalize_batched_gradient(
                        gradients[layer + 1],
                        input_tensor=inputs[layer + 1],
                        group_size=group_size,
                    )
                    workspace.consume_intermediate_gradient(layer, gradient)
                token_gradient = _normalize_batched_gradient(
                    gradients[0],
                    input_tensor=inputs[0],
                    group_size=group_size,
                )
                workspace.consume_token_gradient(token_gradient)
                group_result = workspace.finish(
                    engine_attrs={
                        "backward_engine_mode": self.mode,
                        "forward_graph_mode": self.forward_graph_mode,
                        "vjp_kernel_mode": self.vjp_kernel_mode,
                        "source_layer": source_layer,
                        "source_layer_group_size": group_size,
                        "cotangent_nbytes": cotangent_nbytes,
                    }
                )
                contraction_ms += (time.perf_counter() - contraction_started) * 1000.0
            except BackwardEngineExecutionError:
                if workspace is not None:
                    workspace.abort()
                _clear_feature_vjp_tape_entries(tape_entries)
                raise
            except BaseException as cause:
                if workspace is not None:
                    workspace.abort()
                _clear_feature_vjp_tape_entries(tape_entries)
                raise BackwardEngineExecutionError(
                    mode=self.mode,
                    stage=stage,
                    source_layer=source_layer,
                    cause=cause,
                ) from cause
            finally:
                host._clear_saved_grads()
            try:
                group_results.append(group_result)
                if combined_row_buffer is None:
                    combined_row_buffer = torch.zeros(
                        (int(group_result.rows.shape[1]), batch_size),
                        device=group_result.rows.device,
                        dtype=group_result.rows.dtype,
                    )
                combined_row_buffer.index_copy_(
                    1,
                    row_indices.to(device=combined_row_buffer.device, dtype=torch.long),
                    group_result.rows.T,
                )
                if group_result.feature_vjp_tape_entry is not None:
                    tape_entries.append(
                        (
                            row_indices.detach().cpu(),
                            group_result.feature_vjp_tape_entry,
                        )
                    )
            except BaseException as cause:
                if group_result.feature_vjp_tape_entry is not None:
                    group_result.feature_vjp_tape_entry.clear()
                _clear_feature_vjp_tape_entries(tape_entries)
                raise BackwardEngineExecutionError(
                    mode=self.mode,
                    stage="row_reassembly",
                    source_layer=source_layer,
                    cause=cause,
                ) from cause

        assert combined_row_buffer is not None
        tape_entry = None
        if defer_feature_vjps:
            try:
                tape_entry = _merge_feature_vjp_tape_entries(
                    entries=tape_entries,
                    batch_call_index=batch_call_index,
                    batch_size=batch_size,
                    combined_row_buffer=combined_row_buffer,
                )
            except BaseException as cause:
                _clear_feature_vjp_tape_entries(tape_entries)
                raise BackwardEngineExecutionError(
                    mode=self.mode,
                    stage="tape_reassembly",
                    source_layer=None,
                    cause=cause,
                ) from cause
        vjp_execution_evidence = _build_vjp_execution_evidence(
            kernel_mode=self.vjp_kernel_mode,
            invocation_count=autograd_call_count,
            source_layer_group_count=len(source_layers),
        )
        engine_attrs: dict[str, object] = {
            "backward_engine_mode": self.mode,
            "forward_graph_mode": self.forward_graph_mode,
            "vjp_kernel_mode": self.vjp_kernel_mode,
            "forward_lane_count": self.forward_lane_count,
            "backward_batch_capacity": self.batch_capacity,
            "source_layer_group_count": len(source_layers),
            "source_layer_group_max_rows": max_group_size,
            "autograd_call_count": autograd_call_count,
            "cotangent_build_elapsed_ms": cotangent_build_ms,
            "autograd_elapsed_ms": autograd_ms,
            "contraction_elapsed_ms": contraction_ms,
            "cotangent_total_nbytes": cotangent_total_nbytes,
            "cotangent_peak_nbytes": cotangent_peak_nbytes,
            "vjp_execution_evidence": vjp_execution_evidence,
            "vjp_requested_path": vjp_execution_evidence["requested_path"],
            "vjp_effective_invocation": vjp_execution_evidence["effective_invocation"],
            "vjp_is_grads_batched": vjp_execution_evidence["is_grads_batched"],
            "vjp_fallback_state": vjp_execution_evidence["fallback_state"],
            "vjp_fallback_observation_method": vjp_execution_evidence["observation_method"],
            "vjp_fallback_state_reason": vjp_execution_evidence["fallback_state_reason"],
        }
        return BatchExecutionResult(
            rows=combined_row_buffer.T,
            inject_values_nbytes=int(
                request.inject_values.numel() * request.inject_values.element_size()
            ),
            batch_buffer_nbytes=tensor_nbytes(combined_row_buffer),
            layers_in_batch=source_layers,
            chunked_feature_grad_window_peak=max(
                result.chunked_feature_grad_window_peak for result in group_results
            ),
            feature_vjp_tape_entry=tape_entry,
            engine_attrs=engine_attrs,
        )


@dataclass(frozen=True)
class SingleForwardSerialVjpEngine(SingleForwardBatchedVjpEngine):
    """One forward graph lane with one ordinary autograd VJP per logical row."""

    mode: BackwardEngineMode = field(
        default="single_forward_serial_vjp",
        init=False,
    )

    @property
    def vjp_kernel_mode(self) -> VjpKernelMode:
        return "autograd_serial"

    def _compute_gradients(
        self,
        *,
        output: torch.Tensor,
        inputs: tuple[torch.Tensor, ...],
        cotangent: torch.Tensor,
        retain_graph: bool,
    ) -> tuple[tuple[torch.Tensor | None, ...], int]:
        group_size = int(cotangent.shape[0])
        combined: list[torch.Tensor | None] = [None for _ in inputs]
        for row_index in range(group_size):
            row_gradients = torch.autograd.grad(
                outputs=(output,),
                inputs=inputs,
                grad_outputs=(cotangent[row_index],),
                retain_graph=(retain_graph or row_index < group_size - 1),
                create_graph=False,
                allow_unused=True,
                materialize_grads=False,
            )
            for input_index, gradient in enumerate(row_gradients):
                if gradient is None:
                    continue
                buffer = combined[input_index]
                if buffer is None:
                    buffer = gradient.new_zeros((group_size, *gradient.shape))
                    combined[input_index] = buffer
                buffer[row_index].copy_(gradient)

        return tuple(combined), group_size


def resolve_backward_batch_engine(
    *,
    mode: str,
    batch_capacity: int,
) -> BackwardBatchEngine:
    """Resolve the only mode branch before any attribution batch executes."""

    topology = BackwardExecutionTopology.resolve(
        mode=cast(BackwardEngineMode, mode),
        batch_capacity=batch_capacity,
    )
    if topology.mode == "duplicated_lanes":
        return DuplicatedLaneBackwardEngine(batch_capacity=topology.batch_capacity)
    if topology.mode == "single_forward_batched_vjp":
        return SingleForwardBatchedVjpEngine(batch_capacity=topology.batch_capacity)
    return SingleForwardSerialVjpEngine(batch_capacity=topology.batch_capacity)
