"""Feature batch production and evidence operations for NNSight Phase 4."""

from __future__ import annotations
import time
import torch
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _build_phase4_batch_locality_summary,
    _compute_phase4_locality_shaped_batch_end,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_matrix_abs_stats,
    _build_phase4_normalization_stats,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_phase4_executor_batch_telemetry,
    _build_phase4_executor_substage_telemetry,
    _record_cross_cluster_batch_event,
    _safe_float,
    _tensor_nbytes_estimate,
)
from circuit_tracer.attribution.nnsight.tiled_rows import (
    produce_and_store_tiled_rows,
    produce_tiled_rows_no_retention,
)
from circuit_tracer.observability.events import BatchProfile, TraceEvent
from circuit_tracer.attribution.nnsight.phases.phase4_storage import (
    commit_feature_rows,
    reduce_feature_rows,
)


def produce_feature_batch(state):
    """Materialize encoder vectors and produce one feature-row microbatch."""
    state.chunk_pending_end = state.chunk_pending_start + int(state.idx_batch.numel())
    state.n_visited += len(state.idx_batch)
    state.phase4_executor_microbatch_count += 1
    state.executor_microbatch_index = int(state.phase4_executor_microbatch_count)
    state.ctx_before = state.diagnostic_snapshot(state.ctx) if state.profile else None
    state.transcoder_before = (
        state.diagnostic_snapshot(state.model.transcoders) if state.profile else None
    )
    state.batch_start = time.perf_counter()
    state.batch_memory_before = state.memory_snapshot()
    state.encoder_vectors_source_device = None
    state.encoder_vectors_source_dtype = None
    if getattr(state.ctx, "encoder_vecs", None) is not None and state.ctx.encoder_vecs.numel() > 0:
        state.encoder_vectors_source_device = str(state.ctx.encoder_vecs.device.type)
        state.encoder_vectors_source_dtype = state.ctx.encoder_vecs.dtype
    state.encoder_materialize_start = time.perf_counter()
    state.encoder_vectors = state.ctx.materialize_encoder_vectors(state.idx_batch)
    state.executor_encoder_materialize_elapsed_ms = (
        time.perf_counter() - state.encoder_materialize_start
    ) * 1000.0
    state.encoder_vectors_transfer_bytes = (
        _tensor_nbytes_estimate(state.encoder_vectors)
        if state.encoder_vectors_source_device is not None
        and (
            state.encoder_vectors_source_device != state.encoder_vectors.device.type
            or state.encoder_vectors_source_dtype != state.encoder_vectors.dtype
        )
        else 0
    )
    state.encoder_vectors_transfer_telemetry = {
        "encoder_vectors_source": state.encoder_vectors_source_device,
        "encoder_vectors_destination": str(state.encoder_vectors.device.type),
        "encoder_vectors_dtype_source": str(state.encoder_vectors_source_dtype)
        if state.encoder_vectors_source_dtype is not None
        else None,
        "encoder_vectors_dtype_destination": str(state.encoder_vectors.dtype),
        "encoder_vectors_bytes": int(_tensor_nbytes_estimate(state.encoder_vectors)),
        "encoder_vectors_transfer_bytes": int(state.encoder_vectors_transfer_bytes),
        "encoder_vectors_materialize_elapsed_ms": float(
            state.executor_encoder_materialize_elapsed_ms
        ),
    }
    if state.encoder_vectors_source_device == "cpu" and state.encoder_vectors.device.type == "cuda":
        state.phase4_cpu_to_gpu_bytes_total += int(state.encoder_vectors_transfer_bytes)
    state.compute_batch_start = time.perf_counter()
    state.no_retention = state.config.feature_row_retention == "none_recompute"
    state.tiled_production = (
        state.config.full_retention_backend == "column_tiled_v1" or state.no_retention
    )
    state.tiled_feature_telemetry: dict[str, int | float] = {}
    if state.no_retention:
        state.ctx.reset_saved_graph_handles()
        state.ctx.rebuild_saved_graph_handles()
        state.rows, state.tiled_denominator = produce_tiled_rows_no_retention(
            ctx=state.ctx,
            layers=state.feat_layers[state.idx_batch],
            positions=state.feat_pos[state.idx_batch],
            inject_values=state.encoder_vectors,
            feature_column_tile_size=state.config.feature_row_column_tile_size,
            dtype=state.exact_trace_internal_dtype_resolved,
            phase_label="phase4_features",
            telemetry=state.tiled_feature_telemetry,
        )
    elif state.tiled_production:
        assert state.feature_row_store is not None and state.nonfeature_row_store is not None
        state.rows, state.tiled_denominator = produce_and_store_tiled_rows(
            ctx=state.ctx,
            layers=state.feat_layers[state.idx_batch],
            positions=state.feat_pos[state.idx_batch],
            inject_values=state.encoder_vectors,
            row_start=state.st,
            feature_row_store=state.feature_row_store,
            nonfeature_row_store=state.nonfeature_row_store,
            feature_column_tile_size=state.config.feature_row_column_tile_size,
            dtype=state.exact_trace_internal_dtype_resolved,
            phase_label="phase4_features",
            retain_graph=state.n_visited < state.actual_max_feature_nodes,
            telemetry=state.tiled_feature_telemetry,
        )
    else:
        state.rows = state.ctx.compute_batch(
            layers=state.feat_layers[state.idx_batch],
            positions=state.feat_pos[state.idx_batch],
            inject_values=state.encoder_vectors,
            retain_graph=state.n_visited < state.actual_max_feature_nodes,
            phase_label="phase4_features",
        )
    state.executor_compute_batch_elapsed_ms = (
        time.perf_counter() - state.compute_batch_start
    ) * 1000.0


def record_feature_batch_evidence(state):
    """Record profile, transfer, locality, and debug batch evidence."""
    if state.profile:
        state.batch_number = state.executor_microbatch_index
        if state.batch_number % state.profile_log_interval == 0:
            state.batch_elapsed_ms = (time.perf_counter() - state.batch_start) * 1000.0
            state.telemetry_observer.observe(
                BatchProfile(
                    "Phase 4",
                    state.batch_number,
                    None,
                    state.batch_elapsed_ms / 1000.0,
                    state.ctx_before,
                    state.diagnostic_snapshot(state.ctx),
                    state.transcoder_before,
                    state.diagnostic_snapshot(state.model.transcoders),
                )
            )
    state.batch_number = state.executor_microbatch_index
    state.batch_elapsed_ms = (time.perf_counter() - state.batch_start) * 1000.0
    state.batch_memory_after = state.memory_snapshot()
    state.phase4_feature_batch_elapsed_ms_total += state.batch_elapsed_ms
    state.phase4_executor_encoder_materialize_elapsed_ms_total += (
        state.executor_encoder_materialize_elapsed_ms
    )
    state.phase4_executor_compute_batch_elapsed_ms_total += state.executor_compute_batch_elapsed_ms
    state.phase4_executor_cpu_staging_elapsed_ms_total += state.executor_cpu_staging_elapsed_ms
    state.phase4_executor_denominator_elapsed_ms_total += state.executor_denominator_elapsed_ms
    state.phase4_executor_row_store_write_elapsed_ms_total += (
        state.executor_row_store_write_elapsed_ms
    )
    state.executor_batch_telemetry = _build_phase4_executor_batch_telemetry(
        scheduler_reference_batch_index=state.scheduler_reference_batch_index,
        scheduler_reference_batch_count=state.phase4_scheduler_reference_batch_count,
        scheduler_reference_batch_rows=int(state.reference_idx_batch.numel()),
        executor_microbatch_index=state.executor_microbatch_index,
        executor_microbatch_count=state.phase4_executor_microbatch_count,
        executor_configured_reference_batch_size=state.phase4_executor_reference_batch_size,
        executor_microbatch_rows=int(state.idx_batch.numel()),
        executor_microbatch_size=state.phase4_executor_microbatch_size,
    )
    state.executor_substage_telemetry = _build_phase4_executor_substage_telemetry(
        telemetry_detail=state.phase4_scheduler_config.telemetry_detail,
        encoder_materialize_elapsed_ms=state.executor_encoder_materialize_elapsed_ms,
        compute_batch_elapsed_ms=state.executor_compute_batch_elapsed_ms,
        cpu_staging_elapsed_ms=state.executor_cpu_staging_elapsed_ms,
        denominator_elapsed_ms=state.executor_denominator_elapsed_ms,
        row_store_write_elapsed_ms=state.executor_row_store_write_elapsed_ms,
        batch_elapsed_ms=state.batch_elapsed_ms,
    )
    if (
        state.row_store_append_telemetry is not None
        and state.phase4_scheduler_config.telemetry_detail in {"normal", "debug"}
    ):
        state.executor_substage_telemetry.update(state.row_store_append_telemetry)
    state.executor_streaming_telemetry = {
        "executor_reference_batch_size": int(state.reference_idx_batch.numel()),
        "executor_microbatch_size": int(state.phase4_executor_microbatch_size),
        "executor_streaming_chunk_index": int(state.streaming_chunk_index)
        if state.executor_physically_split
        else None,
        "executor_streaming_chunk_count": int(state.streaming_chunk_count)
        if state.executor_physically_split
        else None,
        "executor_physically_split": bool(state.executor_physically_split),
        "scheduler_pending_start_index": int(state.chunk_pending_start),
        "scheduler_pending_end_index": int(state.chunk_pending_end),
        "scheduler_reference_pending_start_index": int(state.reference_pending_start),
        "scheduler_reference_pending_end_index": int(state.reference_pending_end),
    }
    state.batch_locality_summary = _build_phase4_batch_locality_summary(
        state.idx_batch,
        feat_layers=state.feat_layers,
        feat_ids=state.feat_ids,
        exact_chunked_decoder=state.exact_chunked_decoder,
        decoder_chunk_size=state.decoder_chunk_size,
    )
    state.telemetry_observer.observe(
        TraceEvent(
            scope="batch",
            name="phase4.feature_batch",
            phase="phase4",
            batch_index=state.batch_number,
            elapsed_ms=state.batch_elapsed_ms,
            attrs={
                "batch_rows": int(state.row_count),
                "visited_features": int(state.n_visited),
                "target_feature_count": int(state.actual_max_feature_nodes),
                **state.phase4_execution_metadata,
                **state.executor_batch_telemetry,
                "scheduler_refresh_index": state.pending_refresh_index,
                **state.executor_streaming_telemetry,
                **state.batch_locality_summary,
                **state.executor_substage_telemetry,
                **state.encoder_vectors_transfer_telemetry,
                **state.executor_row_transfer_telemetry,
                **state.memory_delta(state.batch_memory_before, state.batch_memory_after),
            },
            wall_clock=True,
        )
    )
    if state.cross_cluster_debug_batches is not None:
        state.row_input_stats = _build_matrix_abs_stats(
            state.row_input_slice, epsilon=1e-12, top_k=0
        )
        state.row_abs_sum_stats = _build_phase4_normalization_stats(
            state.row_denominator_scaled_l1, clamp_epsilon=1e-08
        )
        _record_cross_cluster_batch_event(
            cross_cluster_debug_batches=state.cross_cluster_debug_batches,
            event_name="phase4.feature_batch",
            phase="phase4",
            event_index=state.batch_number,
            payload={
                "batch_rows": int(state.row_count),
                "visited_features": int(state.n_visited),
                "target_feature_count": int(state.actual_max_feature_nodes),
                **state.phase4_execution_metadata,
                **state.executor_batch_telemetry,
                "scheduler_refresh_index": state.pending_refresh_index,
                **state.executor_streaming_telemetry,
                **state.batch_locality_summary,
                **state.executor_substage_telemetry,
                "idx_batch_hash": state.batch_locality_summary.get("scheduler_batch_hash"),
                "row_input_nonfinite_count": int(state.row_input_stats["nonfinite_count"]),
                "row_input_finite_max_abs": _safe_float(
                    state.row_input_stats.get("finite_max_abs")
                ),
                "row_l1_abs_sum": _safe_float(state.row_abs_sum_stats.get("abs_sum")),
                "row_l1_max": _safe_float(state.row_abs_sum_stats.get("max")),
                "row_l1_nonfinite_count": int(state.row_abs_sum_stats["nonfinite_count"]),
                "row_l1_effectively_all_zero": bool(
                    state.row_abs_sum_stats["effectively_all_zero"]
                ),
                "batch_elapsed_ms": float(state.batch_elapsed_ms),
                **state.memory_snapshot(),
            },
        )
    state.chunk_pending_start = state.chunk_pending_end


def execute_pending_frontier(state):
    """Execute the selected frontier in reference batches and physical microbatches."""
    state.pending_offset = 0
    state.planned_boundaries = (
        state.phase4_frontier_plan.batch_boundaries
        if state.scheduler_uses_reference_planner and state.phase4_frontier_plan is not None
        else None
    )
    state.planned_boundary_offset = 0
    while state.pending_offset < len(state.pending):
        if state.planned_boundaries is not None:
            if state.planned_boundary_offset >= len(state.planned_boundaries):
                raise RuntimeError(
                    "Planner v1 exhausted planned boundaries before pending frontier completion"
                )
            state.boundary_start, state.batch_end = state.planned_boundaries[
                state.planned_boundary_offset
            ]
            if state.boundary_start != state.pending_offset:
                raise RuntimeError(
                    f"Planner v1 planned boundary start mismatch (expected={state.pending_offset}, got={state.boundary_start})"
                )
            state.planned_boundary_offset += 1
        else:
            state.batch_end = _compute_phase4_locality_shaped_batch_end(
                state.pending,
                pending_offset=state.pending_offset,
                max_batch_size=state.phase4_feature_batch_size,
                feat_layers=state.feat_layers,
                feat_ids=state.feat_ids,
                exact_chunked_decoder=state.exact_chunked_decoder,
                decoder_chunk_size=state.decoder_chunk_size,
            )
        if state.batch_end <= state.pending_offset:
            raise RuntimeError(
                f"Phase 4 scheduling produced a non-advancing batch boundary (offset={state.pending_offset}, batch_end={state.batch_end})"
            )
        state.reference_pending_start = state.pending_offset
        state.reference_pending_end = state.batch_end
        state.reference_idx_batch = state.pending[
            state.reference_pending_start : state.reference_pending_end
        ]
        state.pending_offset = state.batch_end
        state.scheduler_reference_batch_index = int(state.phase4_scheduler_reference_batch_count)
        state.phase4_scheduler_reference_batch_count += 1
        if state.phase4_executor_microbatch_size < int(state.reference_idx_batch.numel()):
            state.executor_batches: list[torch.Tensor] = []
            state.streaming_pending_offset = 0
            while state.streaming_pending_offset < int(state.reference_idx_batch.numel()):
                state.streaming_end = min(
                    state.streaming_pending_offset + state.phase4_executor_microbatch_size,
                    int(state.reference_idx_batch.numel()),
                )
                state.executor_batches.append(
                    state.reference_idx_batch[state.streaming_pending_offset : state.streaming_end]
                )
                state.streaming_pending_offset = state.streaming_end
        else:
            state.executor_batches = [state.reference_idx_batch]
        state.streaming_chunk_count = int(len(state.executor_batches))
        state.chunk_pending_start = state.reference_pending_start
        for state.streaming_chunk_index, state.idx_batch in enumerate(
            state.executor_batches, start=1
        ):
            produce_feature_batch(state)
            reduce_feature_rows(state)
            commit_feature_rows(state)
            record_feature_batch_evidence(state)
    if state.planned_boundaries is not None and state.planned_boundary_offset != len(
        state.planned_boundaries
    ):
        raise RuntimeError(
            f"Planner v1 produced unused planned boundaries (used={state.planned_boundary_offset}, planned={len(state.planned_boundaries)})"
        )
