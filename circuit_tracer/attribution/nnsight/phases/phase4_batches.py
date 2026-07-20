"""Feature batch production and evidence operations for NNSight Phase 4."""

from __future__ import annotations
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _SemanticBatch:
    index: int
    start: int
    end: int
    rows: torch.Tensor


@dataclass(frozen=True)
class _ExecutionBatch:
    start: int
    end: int
    rows: torch.Tensor
    semantic_batches: tuple[_SemanticBatch, ...]
    split_chunk_index: int | None = None
    split_chunk_count: int | None = None

    @property
    def coalesced(self) -> bool:
        return len(self.semantic_batches) > 1

    @property
    def split(self) -> bool:
        return self.split_chunk_count is not None


def _pack_semantic_batches(
    semantic_batches: list[_SemanticBatch], *, execution_batch_max_rows: int
) -> list[_ExecutionBatch]:
    """Pack whole semantic batches in order, splitting only oversized batches."""
    execution_batches: list[_ExecutionBatch] = []
    pending_group: list[_SemanticBatch] = []

    def flush_group() -> None:
        if not pending_group:
            return
        first, last = pending_group[0], pending_group[-1]
        rows = (
            first.rows
            if len(pending_group) == 1
            else torch.cat(tuple(batch.rows for batch in pending_group))
        )
        execution_batches.append(
            _ExecutionBatch(
                start=first.start,
                end=last.end,
                rows=rows,
                semantic_batches=tuple(pending_group),
            )
        )
        pending_group.clear()

    for semantic in semantic_batches:
        semantic_rows = int(semantic.rows.numel())
        if semantic_rows > execution_batch_max_rows:
            flush_group()
            split_count = (semantic_rows + execution_batch_max_rows - 1) // execution_batch_max_rows
            for split_index, offset in enumerate(
                range(0, semantic_rows, execution_batch_max_rows), start=1
            ):
                end_offset = min(offset + execution_batch_max_rows, semantic_rows)
                execution_batches.append(
                    _ExecutionBatch(
                        start=semantic.start + offset,
                        end=semantic.start + end_offset,
                        rows=semantic.rows[offset:end_offset],
                        semantic_batches=(semantic,),
                        split_chunk_index=split_index,
                        split_chunk_count=split_count,
                    )
                )
            continue
        grouped_rows = sum(int(batch.rows.numel()) for batch in pending_group)
        if pending_group and grouped_rows + semantic_rows > execution_batch_max_rows:
            flush_group()
        pending_group.append(semantic)
    flush_group()
    return execution_batches


def produce_feature_batch(state):
    """Materialize encoder vectors and produce one feature-row microbatch."""
    state.chunk_pending_end = state.chunk_pending_start + int(state.idx_batch.numel())
    state.n_visited += len(state.idx_batch)
    state.phase4_execution_batch_count += 1
    state.execution_batch_index = int(state.phase4_execution_batch_count)
    if state.executor_physically_coalesced:
        state.phase4_coalesced_execution_batch_count += 1
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
        state.batch_number = state.execution_batch_index
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
    state.batch_number = state.execution_batch_index
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
        semantic_batch_count=state.phase4_scheduler_reference_batch_count,
        semantic_batch_max_rows=state.phase4_semantic_batch_max_rows,
        semantic_batch_index_start=state.semantic_batch_index_start,
        semantic_batch_index_end=state.semantic_batch_index_end,
        semantic_batch_rows=state.semantic_batch_rows,
        execution_batch_index=state.execution_batch_index,
        execution_batch_count=state.phase4_execution_batch_count,
        execution_batch_rows=int(state.idx_batch.numel()),
        execution_batch_max_rows=state.phase4_execution_batch_max_rows,
        execution_batch_coalesced=state.executor_physically_coalesced,
        execution_batch_split=state.executor_physically_split,
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
        "phase4_execution_batch_max_rows": int(state.phase4_execution_batch_max_rows),
        "phase4_execution_batch_rows": int(state.idx_batch.numel()),
        "phase4_execution_batch_coalesced": bool(state.executor_physically_coalesced),
        "phase4_execution_split_chunk_index": int(state.streaming_chunk_index)
        if state.executor_physically_split
        else None,
        "phase4_execution_split_chunk_count": int(state.streaming_chunk_count)
        if state.executor_physically_split
        else None,
        "phase4_execution_batch_split": bool(state.executor_physically_split),
        "phase4_execution_pending_start_index": int(state.chunk_pending_start),
        "phase4_execution_pending_end_index": int(state.chunk_pending_end),
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
    """Execute one refresh-cycle frontier with static physical coalescing."""
    state.pending_offset = 0
    state.planned_boundaries = (
        state.phase4_frontier_plan.batch_boundaries
        if state.scheduler_uses_reference_planner and state.phase4_frontier_plan is not None
        else None
    )
    state.planned_boundary_offset = 0
    semantic_batches: list[_SemanticBatch] = []
    semantic_index_base = int(state.phase4_scheduler_reference_batch_count)
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
        semantic_pending_start = state.pending_offset
        semantic_pending_end = state.batch_end
        semantic_idx_batch = state.pending[semantic_pending_start:semantic_pending_end]
        state.pending_offset = state.batch_end
        semantic_batches.append(
            _SemanticBatch(
                index=semantic_index_base + len(semantic_batches),
                start=semantic_pending_start,
                end=semantic_pending_end,
                rows=semantic_idx_batch,
            )
        )
    if state.planned_boundaries is not None and state.planned_boundary_offset != len(
        state.planned_boundaries
    ):
        raise RuntimeError(
            f"Planner v1 produced unused planned boundaries (used={state.planned_boundary_offset}, planned={len(state.planned_boundaries)})"
        )
    execution_batches = _pack_semantic_batches(
        semantic_batches,
        execution_batch_max_rows=state.phase4_execution_batch_max_rows,
    )
    for execution_batch in execution_batches:
        first_semantic = execution_batch.semantic_batches[0]
        last_semantic = execution_batch.semantic_batches[-1]
        state.phase4_scheduler_reference_batch_count = last_semantic.index + 1
        state.semantic_batch_index_start = first_semantic.index
        state.semantic_batch_index_end = last_semantic.index
        state.semantic_batch_rows = tuple(
            int(batch.rows.numel()) for batch in execution_batch.semantic_batches
        )
        state.chunk_pending_start = execution_batch.start
        state.idx_batch = execution_batch.rows
        state.executor_physically_split = execution_batch.split
        state.executor_physically_coalesced = execution_batch.coalesced
        state.streaming_chunk_index = execution_batch.split_chunk_index
        state.streaming_chunk_count = execution_batch.split_chunk_count
        produce_feature_batch(state)
        reduce_feature_rows(state)
        commit_feature_rows(state)
        record_feature_batch_evidence(state)
