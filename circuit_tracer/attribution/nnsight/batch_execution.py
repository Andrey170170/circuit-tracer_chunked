"""Observed execution wrapper for one backward-engine attribution batch."""

from __future__ import annotations

import time
from typing import cast

import torch

from circuit_tracer.attribution.nnsight.backward_engines import (
    BackwardBatchEngine,
    BackwardEngineExecutionError,
    DuplicatedLaneBackwardEngine,
)
from circuit_tracer.attribution.nnsight.batch_contract import (
    BatchAttributionRequest,
    BatchExecutionHost,
    BatchExecutionResult,
)
from circuit_tracer.attribution.nnsight.batch_workspace import (
    slice_phase3_gradient_replay_batch,
)
from circuit_tracer.attribution.nnsight.resource_sampling import (
    should_sample_batch_resources,
)
from circuit_tracer.observability.errors import safe_exception_attrs
from circuit_tracer.observability.events import (
    MemoryDelta,
    MemorySnapshot,
    MemorySnapshotAttrs,
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


def execute_backward_batch(
    host: BatchExecutionHost,
    request: BatchAttributionRequest,
    *,
    batch_call_index: int,
    defer_feature_vjps: bool = False,
) -> BatchExecutionResult:
    """Compatibility entrypoint for the established duplicated-lane engine."""

    return DuplicatedLaneBackwardEngine(
        batch_capacity=int(host._resid_activations[0].shape[0])
    ).execute(
        host,
        request,
        batch_call_index=batch_call_index,
        defer_feature_vjps=defer_feature_vjps,
    )


def _memory_snapshot(host: BatchExecutionHost, execution_device: torch.device) -> dict[str, object]:
    observer = host._trace_observer
    if observer is None:
        return {}
    try:
        return cast(dict[str, object], observer.observe(MemorySnapshot(execution_device)))
    except BaseException:
        return {}


def execute_observed_batch(
    host: BatchExecutionHost,
    request: BatchAttributionRequest,
    *,
    strategy: BackwardBatchEngine,
    batch_call_index: int,
    defer_feature_vjps: bool = False,
) -> BatchExecutionResult:
    """Execute a selected engine while owning resource and failure evidence."""

    batch_size = request.validate(
        batch_capacity=strategy.batch_capacity,
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
    memory_before = _memory_snapshot(host, execution_device) if resource_sampled else {}
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
    planned_buffer_nbytes = produced_width * batch_size * torch.float32.itemsize
    common_attrs = {
        "backward_engine_mode": strategy.mode,
        "forward_graph_mode": strategy.forward_graph_mode,
        "vjp_kernel_mode": strategy.vjp_kernel_mode,
        "forward_lane_count": strategy.forward_lane_count,
        "backward_batch_capacity": strategy.batch_capacity,
        "phase_batch_index": phase_batch_index,
        "resource_sampled": resource_sampled,
        "batch_nodes": batch_size,
        "unique_layers": unique_layers,
        "retain_graph": request.retain_graph,
        "feature_vjp_deferred": defer_feature_vjps,
    }
    host._emit_trace(
        "compute_batch.start",
        phase=request.phase_label,
        inject_values_input_nbytes=input_nbytes,
        planned_batch_buffer_nbytes=planned_buffer_nbytes,
        chunked_feature_replay_window=int(host._chunked_feature_replay_window),
        **common_attrs,
        **(
            cast(
                dict[str, object],
                observer.observe(
                    MemorySnapshotAttrs(
                        memory_before,
                        keys=_MEMORY_ATTR_KEYS,
                        prefix="memory_before",
                    )
                ),
            )
            if observer is not None and resource_sampled
            else {}
        ),
    )
    try:
        result = strategy.execute(
            host,
            request,
            batch_call_index=batch_call_index,
            defer_feature_vjps=defer_feature_vjps,
        )
    except BaseException as error:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        failure_attrs: dict[str, object] = {
            **common_attrs,
            **safe_exception_attrs(error),
        }
        if isinstance(error, BackwardEngineExecutionError):
            failure_attrs.update(
                failure_stage=error.stage,
                failure_source_layer=error.source_layer,
                failure_cause_type=error.cause_type,
            )
        memory_after = _memory_snapshot(host, execution_device) if resource_sampled else {}
        if observer is not None and resource_sampled:
            try:
                failure_attrs.update(
                    cast(
                        dict[str, object],
                        observer.observe(
                            MemoryDelta(
                                before=memory_before,
                                after=memory_after,
                                keys=_MEMORY_ATTR_KEYS,
                            )
                        ),
                    )
                )
            except BaseException:
                pass
        try:
            host._record_telemetry_event(
                scope="batch",
                name="context.compute_batch.failed",
                phase=request.phase_label,
                batch_index=batch_call_index,
                elapsed_ms=elapsed_ms,
                attrs=failure_attrs,
            )
            host._emit_trace(
                "compute_batch.failed",
                phase=request.phase_label,
                elapsed_ms=elapsed_ms,
                **failure_attrs,
            )
        except BaseException as telemetry_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "batch failure telemetry also failed: "
                    f"{safe_exception_attrs(telemetry_error)['error_repr']}"
                )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    memory_after = _memory_snapshot(host, execution_device) if resource_sampled else {}
    if host.diagnostic_mode:
        host._add_stat("compute_batch_calls", 1)
        elapsed = elapsed_ms / 1000.0
        host._add_stat("compute_batch_seconds", elapsed)
        phase_bucket = cast(
            dict[str, float],
            host._diagnostic_stats.setdefault("compute_batch_seconds_by_phase", {}),
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
    done_attrs = {
        **common_attrs,
        "batch_size": batch_size,
        "row_size": int(host._row_size),
        "unique_layers": len(result.layers_in_batch),
        "chunked_decoder": host.chunked_decoder_state is not None,
        "inject_values_input_nbytes": input_nbytes,
        "inject_values_nbytes": result.inject_values_nbytes,
        "batch_buffer_nbytes": result.batch_buffer_nbytes,
        "chunked_feature_replay_window": int(host._chunked_feature_replay_window),
        "chunked_feature_grad_window_peak": result.chunked_feature_grad_window_peak,
        **result.engine_attrs,
        **memory_attrs,
    }
    host._record_telemetry_event(
        scope="batch",
        name="context.compute_batch",
        phase=request.phase_label,
        batch_index=batch_call_index,
        elapsed_ms=elapsed_ms,
        attrs=done_attrs,
    )
    host._emit_trace(
        "compute_batch.done",
        phase=request.phase_label,
        elapsed_s=f"{elapsed_ms / 1000.0:.2f}",
        elapsed_ms=elapsed_ms,
        **done_attrs,
    )
    return result


__all__ = [
    "BatchAttributionRequest",
    "BatchExecutionHost",
    "BatchExecutionResult",
    "execute_backward_batch",
    "execute_observed_batch",
    "slice_phase3_gradient_replay_batch",
]
