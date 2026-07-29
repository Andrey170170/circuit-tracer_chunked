"""Row replay, reduction, and storage commit operations for NNSight Phase 4."""

from __future__ import annotations
import time
import torch
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_matrix_abs_stats,
    _build_phase4_normalization_stats,
    _copy_feature_rows_to_cpu_staging,
    _copy_rows_to_cpu_staging,
)
from circuit_tracer.attribution.nnsight.replay import _compute_row_denominator_scaled_l1
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_phase4_gpu_row_reduction_transfer_telemetry,
    _build_row_transfer_telemetry,
)
from circuit_tracer.attribution.nnsight.row_replay import RowRecipe, RowRecipeLedger


def reduce_feature_rows(state):
    """Reduce produced rows while preserving exact row-L1 numerics."""
    if state.tiled_production:
        state.phase4_gpu_to_cpu_bytes_total += int(
            state.tiled_feature_telemetry.get("feature_transfer_bytes", 0)
        )
        state.phase4_copy_count += int(state.tiled_feature_telemetry.get("feature_copy_count", 0))
        state.phase4_feature_backward_count_total += int(
            state.tiled_feature_telemetry.get("feature_backward_count", 0)
        )
        state.phase4_feature_produced_tile_count_total += int(
            state.tiled_feature_telemetry.get("feature_produced_tile_count", 0)
        )
        state.phase4_feature_backward_tile_count_total += int(
            state.tiled_feature_telemetry.get("feature_backward_tile_count", 0)
        )
        state.phase4_feature_transient_peak_bytes = max(
            state.phase4_feature_transient_peak_bytes,
            int(state.tiled_feature_telemetry.get("feature_transient_peak_bytes", 0)),
        )
        state.phase4_executor_cpu_staging_elapsed_ms_total += float(
            state.tiled_feature_telemetry.get("feature_cpu_copy_elapsed_ms", 0.0)
        )
        state.phase4_executor_denominator_elapsed_ms_total += float(
            state.tiled_feature_telemetry.get("feature_denominator_elapsed_ms", 0.0)
        )
        state.phase4_executor_denominator_global_max_elapsed_ms_total += float(
            state.tiled_feature_telemetry.get("feature_denominator_global_max_elapsed_ms", 0.0)
        )
        state.phase4_executor_denominator_scaled_sum_elapsed_ms_total += float(
            state.tiled_feature_telemetry.get("feature_denominator_scaled_sum_elapsed_ms", 0.0)
        )
        state.phase4_executor_row_store_write_elapsed_ms_total += float(
            state.tiled_feature_telemetry.get("feature_store_write_elapsed_ms", 0.0)
        )
    state.row_count = state.rows.shape[0]
    state.end = state.st + state.row_count
    if state.tiled_production:
        state.cpu_staging_start = time.perf_counter()
        state.rows_cpu = state.rows
        state.row_input_slice = state.rows_cpu
        state.feature_row_slice = torch.empty((state.row_count, 0), dtype=state.rows_cpu.dtype)
        state.nonfeature_row_slice = state.rows_cpu
        state.executor_cpu_staging_elapsed_ms = 0.0
        state.denominator_start = time.perf_counter()
        state.row_denominator_scaled_l1 = state.tiled_denominator
        state.executor_denominator_elapsed_ms = 0.0
        state.executor_row_transfer_telemetry = _build_row_transfer_telemetry(
            rows=state.rows,
            rows_cpu=state.rows_cpu,
            row_input_slice=state.row_input_slice,
            feature_row_slice=state.feature_row_slice,
        )
    elif state.phase4_row_reduction_config.effective_mode == "gpu_v1":
        if not state.use_compact_feature_row_store:
            raise RuntimeError("phase4_row_reduction='gpu_v1' requires compact Phase-4 row store")
        state.cpu_staging_start = time.perf_counter()
        state.feature_row_slice, state.feature_rows_cpu_staging = _copy_feature_rows_to_cpu_staging(
            state.rows,
            total_active_feats=state.total_active_feats,
            staging_buffer=state.feature_rows_cpu_staging,
        )
        state.executor_cpu_staging_elapsed_ms = (
            time.perf_counter() - state.cpu_staging_start
        ) * 1000.0
        state.row_input_slice = state.rows[:, : state.logit_offset]
        state.denominator_start = time.perf_counter()
        state.row_abs_max_gpu, state.row_l1_scaled_gpu = _compute_row_denominator_scaled_l1(
            state.row_input_slice,
            dtype=state.exact_trace_internal_dtype_resolved,
            preserve_device=True,
        )
        state.executor_denominator_elapsed_ms = (
            time.perf_counter() - state.denominator_start
        ) * 1000.0
        state.executor_row_transfer_telemetry = _build_phase4_gpu_row_reduction_transfer_telemetry(
            rows=state.rows,
            feature_row_slice=state.feature_row_slice,
            row_abs_max=state.row_abs_max_gpu,
            row_l1_scaled=state.row_l1_scaled_gpu,
        )
        state.row_denominator_scaled_l1 = (state.row_abs_max_gpu, state.row_l1_scaled_gpu)
        state.nonfeature_row_slice = state.rows[:, state.total_active_feats : state.logit_offset]
    else:
        state.cpu_staging_start = time.perf_counter()
        state.rows_cpu, state.rows_cpu_staging = _copy_rows_to_cpu_staging(
            state.rows, staging_buffer=state.rows_cpu_staging
        )
        state.executor_cpu_staging_elapsed_ms = (
            time.perf_counter() - state.cpu_staging_start
        ) * 1000.0
        state.row_input_slice = state.rows_cpu[:, : state.logit_offset]
        state.feature_row_slice = state.rows_cpu[:, : state.total_active_feats]
        state.nonfeature_row_slice = state.rows_cpu[
            :, state.total_active_feats : state.logit_offset
        ]
        state.executor_row_transfer_telemetry = _build_row_transfer_telemetry(
            rows=state.rows,
            rows_cpu=state.rows_cpu,
            row_input_slice=state.row_input_slice,
            feature_row_slice=state.feature_row_slice,
        )
        state.denominator_start = time.perf_counter()
        state.row_abs_max_cpu, state.row_l1_scaled_cpu = _compute_row_denominator_scaled_l1(
            state.row_input_slice, dtype=state.exact_trace_internal_dtype_resolved
        )
        state.row_denominator_scaled_l1 = (state.row_abs_max_cpu, state.row_l1_scaled_cpu)
        state.executor_denominator_elapsed_ms = (
            time.perf_counter() - state.denominator_start
        ) * 1000.0
    if state.executor_row_transfer_telemetry["row_transfer_source"] == "cuda":
        state.phase4_gpu_to_cpu_bytes_total += int(
            state.executor_row_transfer_telemetry["row_transfer_bytes"]
        )
    state.phase4_row_reduction_gpu_to_cpu_bytes_saved_total += int(
        state.executor_row_transfer_telemetry.get("row_reduction_gpu_to_cpu_bytes_saved", 0)
    )
    if state.executor_row_transfer_telemetry["row_transfer_destination"] == "cuda":
        state.phase4_cpu_to_gpu_bytes_total += int(
            state.executor_row_transfer_telemetry["row_transfer_bytes"]
        )
    if int(state.executor_row_transfer_telemetry["row_transfer_bytes"]) > 0:
        state.phase4_copy_count += 1


def commit_feature_rows(state):
    """Commit feature and nonfeature rows to owned storage."""
    if state.anomaly_debug_result is not None and state.phase4_execution_batch_count <= 2:
        state.feature_row_batches = state.anomaly_debug_result.setdefault(
            "phase4_feature_row_batches", []
        )
        assert isinstance(state.feature_row_batches, list)
        state.feature_row_batches.append(
            {
                "batch_index": int(state.execution_batch_index),
                "batch_row_count": int(state.row_count),
                "row_input_stats": _build_matrix_abs_stats(
                    state.row_input_slice, epsilon=1e-12, top_k=8
                ),
                "row_abs_sum_stats": _build_phase4_normalization_stats(
                    state.row_denominator_scaled_l1, clamp_epsilon=1e-08
                ),
            }
        )
    if state.no_retention:
        assert isinstance(state.feature_row_store, RowRecipeLedger)
        assert isinstance(state.nonfeature_row_store, RowRecipeLedger)
        for state.local_index in range(state.row_count):
            state.ordinal = state.st + state.local_index
            state.recipe = RowRecipe(
                ordinal=state.ordinal,
                source_kind="feature",
                layer=int(state.feat_layers[state.idx_batch[state.local_index]]),
                position=int(state.feat_pos[state.idx_batch[state.local_index]]),
                injection=state.encoder_vectors[state.local_index],
            )
            state.denominator = (
                state.row_denominator_scaled_l1[0][state.local_index : state.local_index + 1],
                state.row_denominator_scaled_l1[1][state.local_index : state.local_index + 1],
            )
            state.node_index = int(state.idx_batch[state.local_index])
            state.feature_row_store.append_recipe(
                state.recipe, node_index=state.node_index, denominator=state.denominator
            )
            state.nonfeature_row_store.append_recipe(
                state.recipe, node_index=state.node_index, denominator=state.denominator
            )
        state.executor_row_store_write_elapsed_ms = 0.0
        state.row_store_append_telemetry = {}
    elif state.use_compact_feature_row_store and (not state.tiled_production):
        assert state.feature_row_store is not None
        assert state.nonfeature_row_store is not None
        state.row_store_write_start = time.perf_counter()
        feature_append_kwargs = {}
        admission = getattr(state.feature_row_store, "admission", None)
        if admission is not None and bool(getattr(admission, "admitted", False)):
            feature_append_kwargs["resident_feature_rows"] = state.rows[
                :, : state.total_active_feats
            ]
        state.row_store_append_telemetry = state.feature_row_store.append_rows(
            row_start=state.st,
            feature_rows=state.feature_row_slice,
            row_denominator_scaled_l1=state.row_denominator_scaled_l1,
            phase="phase4",
            **feature_append_kwargs,
        )
        state.nonfeature_row_store.append_rows(
            row_start=state.st,
            feature_rows=state.nonfeature_row_slice,
            row_denominator_scaled_l1=state.row_denominator_scaled_l1,
            phase="phase4",
        )
        state.executor_row_store_write_elapsed_ms = (
            time.perf_counter() - state.row_store_write_start
        ) * 1000.0
    elif not state.use_compact_feature_row_store:
        assert state.phase4_row_reduction_config.effective_mode == "off"
        state.row_store_write_start = time.perf_counter()
        state.edge_matrix[state.st : state.end, : state.logit_offset] = state.rows_cpu
        state.executor_row_store_write_elapsed_ms = (
            time.perf_counter() - state.row_store_write_start
        ) * 1000.0
        state.row_store_append_telemetry = None
    else:
        state.executor_row_store_write_elapsed_ms = 0.0
        state.row_store_append_telemetry = {}
    state.row_to_node_index[state.st : state.end] = state.idx_batch
    state.visited[state.idx_batch] = True
    state.st = state.end
    state.pbar.update(len(state.idx_batch))
