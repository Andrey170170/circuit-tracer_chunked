"""Ordered logit-batch execution for NNSight Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase4_policy import _PHASE4_REFRESH_MEMORY_ATTR_KEYS
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_matrix_abs_stats,
    _build_phase4_normalization_stats,
)
from circuit_tracer.attribution.nnsight.session_controls import ordered_physical_ranges
from circuit_tracer.attribution.nnsight.telemetry import (
    _record_cross_cluster_batch_event,
    _safe_float,
)
from circuit_tracer.observability.events import (
    BatchProfile,
    DiagnosticSnapshot,
    MemoryDelta,
    MemorySnapshot,
    PhaseMetrics,
    TraceEvent,
    TraceObserver,
)

from .phase3_rows import (
    EffectiveRows,
    Phase3ReplayRows,
    Phase3TransferMetrics,
    account_row_transfer,
    capture_replay_rows,
    commit_effective_rows,
    produce_logit_rows,
    resolve_effective_rows,
)


@dataclass(frozen=True)
class Phase3BatchResult:
    """State produced by the complete ordered logit-batch pass."""

    rows_cpu_staging: torch.Tensor | None
    replay_rows: Phase3ReplayRows
    metrics: Phase3TransferMetrics
    logical_batch_count: int
    physical_batch_count: int
    physical_batch_peak_rows: int
    last_row_start: int


def run_logit_batches(
    *, inputs: Any, config: Any, phase_start: float
) -> Phase3BatchResult:
    """Execute Phase 3 batches in canonical logical/physical order."""
    targets = inputs.targets
    ranges = ordered_physical_ranges(
        total_rows=len(targets),
        logical_batch_rows=config.effective_logit_batch_size,
        physical_batch_rows=config.compute_microbatch_max_rows,
    )
    logical_count = max(
        (len(targets) + config.effective_logit_batch_size - 1)
        // config.effective_logit_batch_size,
        1,
    )
    total_metrics = Phase3TransferMetrics()
    replay_rows = Phase3ReplayRows()
    staging = None
    peak_rows = 0
    last_start = -1
    for physical_index, (logical_index, row_start, row_end) in enumerate(ranges, 1):
        last_start = row_start
        batch = targets.logit_vectors[row_start:row_end]
        peak_rows = max(peak_rows, int(batch.shape[0]))
        staging, batch_metrics = _run_one_batch(
            inputs=inputs,
            config=config,
            batch=batch,
            row_start=row_start,
            logical_index=logical_index,
            physical_index=physical_index,
            logical_count=logical_count,
            physical_count=len(ranges),
            staging=staging,
            replay_rows=replay_rows,
        )
        total_metrics.absorb(batch_metrics)
    _finish_batch_phase(
        inputs=inputs,
        phase_start=phase_start,
        logical_count=logical_count,
        physical_count=len(ranges),
        peak_rows=peak_rows,
        metrics=total_metrics,
        last_start=last_start,
    )
    return Phase3BatchResult(
        staging, replay_rows, total_metrics, logical_count, len(ranges), peak_rows, last_start
    )


def _run_one_batch(
    *, inputs: Any, config: Any, batch: torch.Tensor, row_start: int,
    logical_index: int, physical_index: int, logical_count: int, physical_count: int,
    staging: torch.Tensor | None, replay_rows: Phase3ReplayRows,
) -> tuple[torch.Tensor | None, Phase3TransferMetrics]:
    observer: TraceObserver = inputs.telemetry_observer
    ctx_before = _diagnostic(observer, inputs.ctx) if config.profile else None
    transcoder_before = _diagnostic(observer, inputs.model.transcoders) if config.profile else None
    batch_start = time.perf_counter()
    memory_before = _memory_snapshot(observer, inputs.model.device)
    if config.phase3_gradient_replay_mode_resolved == "donor":
        setattr(inputs.ctx, "phase3_gradient_replay_column_offset", int(row_start))
    produced = produce_logit_rows(
        ctx=inputs.ctx,
        model_device=inputs.model.device,
        batch=batch,
        row_start=row_start,
        n_layers=config.n_layers,
        n_pos=config.n_pos,
        output_position=config.output_position,
        dtype=config.exact_trace_internal_dtype_resolved,
        full_retention_backend=config.full_retention_backend,
        feature_row_retention=config.feature_row_retention,
        feature_column_tile_size=config.feature_row_column_tile_size,
        feature_row_store=inputs.feature_row_store,
        nonfeature_row_store=inputs.nonfeature_row_store,
        rows_cpu_staging=staging,
    )
    rows = resolve_effective_rows(
        produced=produced,
        donor_bundle=inputs.loaded_phase3_row_donor_bundle,
        row_start=row_start,
        row_count=int(batch.shape[0]),
        logit_offset=config.logit_offset,
        total_active_features=config.total_active_feats,
        dtype=config.exact_trace_internal_dtype_resolved,
    )
    produced.metrics.denominator_elapsed_ms += rows.denominator_elapsed_ms
    account_row_transfer(produced.metrics, rows)
    if config.capture_phase3_row_bundle_enabled:
        capture_replay_rows(
            captures=replay_rows,
            rows=rows,
            total_active_features=config.total_active_feats,
            n_layers=config.n_layers,
            n_pos=config.n_pos,
            logit_offset=config.logit_offset,
        )
    _record_anomaly_batch(
        inputs.anomaly_debug_result, rows, batch, logical_index, physical_index,
        logical_count, physical_count
    )
    write_ms = commit_effective_rows(
        produced=produced,
        rows=rows,
        batch=batch,
        row_start=row_start,
        n_layers=config.n_layers,
        n_pos=config.n_pos,
        output_position=config.output_position,
        logit_offset=config.logit_offset,
        total_active_features=config.total_active_feats,
        use_compact_store=config.use_compact_feature_row_store,
        feature_row_store=inputs.feature_row_store,
        nonfeature_row_store=inputs.nonfeature_row_store,
        edge_matrix=inputs.edge_matrix,
        row_to_node_index=inputs.row_to_node_index,
    )
    produced.metrics.row_store_write_elapsed_ms += write_ms
    elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
    _observe_batch(
        inputs, config, rows, produced.inject_transfer, produced.metrics, memory_before,
        batch, row_start, logical_index, physical_index, logical_count, physical_count,
        elapsed_ms, ctx_before, transcoder_before
    )
    return produced.rows_cpu_staging, produced.metrics


def _observe_batch(
    inputs: Any, config: Any, rows: EffectiveRows, inject_transfer: dict[str, object],
    metrics: Phase3TransferMetrics, memory_before: dict[str, object], batch: torch.Tensor,
    row_start: int, logical_index: int, physical_index: int, logical_count: int,
    physical_count: int, elapsed_ms: float, ctx_before: dict[str, object] | None,
    transcoder_before: dict[str, object] | None,
) -> None:
    observer: TraceObserver = inputs.telemetry_observer
    memory_after = _memory_snapshot(observer, inputs.model.device)
    memory_delta = cast(
        dict[str, object],
        observer.observe(MemoryDelta(memory_before, memory_after, _PHASE4_REFRESH_MEMORY_ATTR_KEYS)),
    )
    attrs = {
        "batch_rows": int(batch.shape[0]), "batch_start_index": int(row_start),
        "logical_batch_index": int(logical_index + 1), "physical_batch_index": physical_index,
        "total_logit_batches": logical_count, "total_logical_batches": logical_count,
        "total_physical_batches": physical_count,
        "compute_batch_elapsed_ms": float(metrics.compute_batch_elapsed_ms),
        "cpu_staging_elapsed_ms": float(metrics.cpu_staging_elapsed_ms),
        "denominator_elapsed_ms": float(metrics.denominator_elapsed_ms),
        "row_store_write_elapsed_ms": float(metrics.row_store_write_elapsed_ms),
        **inject_transfer, **rows.row_transfer, **memory_delta,
    }
    observer.observe(TraceEvent(
        scope="batch", name="phase3.logit_batch", phase="phase3",
        batch_index=physical_index, elapsed_ms=elapsed_ms, attrs=attrs, wall_clock=True,
    ))
    _record_debug_batch(
        inputs, rows, batch, row_start, logical_index, physical_index,
        logical_count, physical_count
    )
    if config.profile and physical_index % config.profile_log_interval == 0:
        observer.observe(BatchProfile(
            "Phase 3", physical_index, physical_count, elapsed_ms / 1000.0,
            ctx_before, _diagnostic(observer, inputs.ctx), transcoder_before,
            _diagnostic(observer, inputs.model.transcoders),
        ))


def _finish_batch_phase(
    *, inputs: Any, phase_start: float, logical_count: int, physical_count: int,
    peak_rows: int, metrics: Phase3TransferMetrics, last_start: int,
) -> None:
    observer: TraceObserver = inputs.telemetry_observer
    observer.observe(PhaseMetrics(
        f"{last_start + 1} logit attribution(s)", phase_start, inputs.model.device
    ))
    attrs = {
        "logit_count": int(len(inputs.targets)), "batches": logical_count,
        "logical_batch_count": logical_count, "physical_batch_count": physical_count,
        "physical_batch_peak_rows": peak_rows,
        "phase3_compute_batch_elapsed_ms_total": float(metrics.compute_batch_elapsed_ms),
        "phase3_cpu_staging_elapsed_ms_total": float(metrics.cpu_staging_elapsed_ms),
        "phase3_denominator_elapsed_ms_total": float(metrics.denominator_elapsed_ms),
        "phase3_denominator_global_max_elapsed_ms_total": float(metrics.denominator_global_max_elapsed_ms),
        "phase3_denominator_scaled_sum_elapsed_ms_total": float(metrics.denominator_scaled_sum_elapsed_ms),
        "phase3_row_store_write_elapsed_ms_total": float(metrics.row_store_write_elapsed_ms),
        "phase3_gpu_to_cpu_bytes_total": metrics.gpu_to_cpu_bytes,
        "phase3_cpu_to_gpu_bytes_total": metrics.cpu_to_gpu_bytes,
        "phase3_copy_count": metrics.copy_count,
        "phase3_feature_backward_count_total": metrics.feature_backward_count,
        "phase3_feature_produced_tile_count_total": metrics.feature_produced_tile_count,
        "phase3_feature_backward_tile_count_total": metrics.feature_backward_tile_count,
        "phase3_feature_transient_peak_bytes": metrics.feature_transient_peak_bytes,
    }
    observer.observe(TraceEvent(
        scope="phase", name="phase3.logit_attribution", phase="phase3",
        elapsed_ms=(time.perf_counter() - phase_start) * 1000.0, attrs=attrs, wall_clock=True,
    ))
    reset = getattr(inputs.ctx, "reset_decoder_cache", None)
    if callable(reset):
        reset()


def _record_anomaly_batch(
    anomaly: dict[str, object] | None, rows: EffectiveRows, batch: torch.Tensor,
    logical_index: int, physical_index: int, logical_count: int, physical_count: int,
) -> None:
    if anomaly is None:
        return
    batches = anomaly.setdefault("phase3_logit_row_batches", [])
    assert isinstance(batches, list)
    batches.append({
        "batch_index": physical_index, "logical_batch_index": logical_index + 1,
        "physical_batch_index": physical_index, "total_logical_batches": logical_count,
        "total_physical_batches": physical_count, "batch_row_count": int(batch.shape[0]),
        "row_input_stats": _build_matrix_abs_stats(rows.row_input, epsilon=1e-12, top_k=8),
        "row_abs_sum_stats": _build_phase4_normalization_stats(rows.denominator, clamp_epsilon=1e-8),
    })


def _record_debug_batch(
    inputs: Any, rows: EffectiveRows, batch: torch.Tensor, row_start: int,
    logical_index: int, physical_index: int, logical_count: int, physical_count: int,
) -> None:
    if inputs.cross_cluster_debug_batches is None:
        return
    row_stats = _build_matrix_abs_stats(rows.row_input, epsilon=1e-12, top_k=0)
    l1_stats = _build_phase4_normalization_stats(rows.denominator, clamp_epsilon=1e-8)
    _record_cross_cluster_batch_event(
        cross_cluster_debug_batches=inputs.cross_cluster_debug_batches,
        event_name="phase3.logit_batch", phase="phase3", event_index=physical_index,
        payload={
            "batch_rows": int(batch.shape[0]), "batch_start_index": row_start,
            "total_logit_batches": logical_count, "logical_batch_index": logical_index + 1,
            "physical_batch_index": physical_index, "total_logical_batches": logical_count,
            "total_physical_batches": physical_count,
            "row_input_nonfinite_count": int(row_stats["nonfinite_count"]),
            "row_input_finite_max_abs": _safe_float(row_stats.get("finite_max_abs")),
            "row_l1_abs_sum": _safe_float(l1_stats.get("abs_sum")),
            "row_l1_max": _safe_float(l1_stats.get("max")),
            "row_l1_nonfinite_count": int(l1_stats["nonfinite_count"]),
            "row_l1_effectively_all_zero": bool(l1_stats["effectively_all_zero"]),
            **_memory_snapshot(inputs.telemetry_observer, inputs.model.device),
        },
    )


def _memory_snapshot(observer: TraceObserver, device: torch.device) -> dict[str, object]:
    return cast(dict[str, object], observer.observe(MemorySnapshot(device)))


def _diagnostic(observer: TraceObserver, value: object) -> dict[str, object] | None:
    return cast(dict[str, object] | None, observer.observe(DiagnosticSnapshot(value)))
