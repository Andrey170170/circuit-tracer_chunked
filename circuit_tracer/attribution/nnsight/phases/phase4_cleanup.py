"""Completion and cleanup operations for NNSight Phase 4."""

from __future__ import annotations
import time
from circuit_tracer.observability.events import PhaseMetrics, TraceEvent


def finish_phase4(state):
    """Close progress and emit final Phase 4 metrics."""
    state.pbar.close()
    state.telemetry_observer.observe(
        PhaseMetrics(
            "Feature attributions",
            state.phase4_start,
            state.model.device,
            {
                "selected_features": int(state.visited.sum().item()),
                "final_feature_batch_size": state.phase4_feature_batch_size,
                "phase4_batches": state.phase4_scheduler_reference_batch_count,
                "phase4_executor_microbatch_count": state.phase4_executor_microbatch_count,
            },
        )
    )
    state.phase4_elapsed_ms = (time.perf_counter() - state.phase4_start) * 1000.0
    state.telemetry_observer.observe(
        TraceEvent(
            scope="phase",
            name="phase4.feature_attribution",
            phase="phase4",
            elapsed_ms=state.phase4_elapsed_ms,
            attrs={
                "selected_features": int(state.visited.sum().item()),
                "feature_batch_size": int(state.phase4_feature_batch_size),
                "phase4_batches": int(state.phase4_scheduler_reference_batch_count),
                "phase4_executor_microbatch_count": int(state.phase4_executor_microbatch_count),
                "phase4_refreshes": int(state.phase4_refresh_count),
                "phase4_refresh_elapsed_ms_total": float(state.phase4_refresh_elapsed_ms_total),
                "phase4_feature_batch_elapsed_ms_total": float(
                    state.phase4_feature_batch_elapsed_ms_total
                ),
                "phase4_refresh_partial_influence_elapsed_ms_total": float(
                    state.phase4_refresh_partial_influence_elapsed_ms_total
                ),
                "phase4_refresh_rank_topk_elapsed_ms_total": float(
                    state.phase4_refresh_rank_topk_elapsed_ms_total
                ),
                "phase4_refresh_frontier_plan_elapsed_ms_total": float(
                    state.phase4_refresh_frontier_plan_elapsed_ms_total
                ),
                "phase4_refresh_row_store_read_elapsed_ms_total": float(
                    state.phase4_refresh_row_store_read_elapsed_ms_total
                ),
                "phase4_refresh_influence_normalization_elapsed_ms_total": float(
                    state.phase4_refresh_influence_normalization_elapsed_ms_total
                ),
                "phase4_refresh_influence_matmul_elapsed_ms_total": float(
                    state.phase4_refresh_influence_matmul_elapsed_ms_total
                ),
                "phase4_executor_encoder_materialize_elapsed_ms_total": float(
                    state.phase4_executor_encoder_materialize_elapsed_ms_total
                ),
                "phase4_executor_compute_batch_elapsed_ms_total": float(
                    state.phase4_executor_compute_batch_elapsed_ms_total
                ),
                "phase4_executor_cpu_staging_elapsed_ms_total": float(
                    state.phase4_executor_cpu_staging_elapsed_ms_total
                ),
                "phase4_executor_denominator_elapsed_ms_total": float(
                    state.phase4_executor_denominator_elapsed_ms_total
                ),
                "phase4_executor_denominator_global_max_elapsed_ms_total": float(
                    state.phase4_executor_denominator_global_max_elapsed_ms_total
                ),
                "phase4_executor_denominator_scaled_sum_elapsed_ms_total": float(
                    state.phase4_executor_denominator_scaled_sum_elapsed_ms_total
                ),
                "phase4_executor_row_store_write_elapsed_ms_total": float(
                    state.phase4_executor_row_store_write_elapsed_ms_total
                ),
                "phase4_gpu_to_cpu_bytes_total": int(state.phase4_gpu_to_cpu_bytes_total),
                "phase4_row_reduction_gpu_to_cpu_bytes_saved_total": int(
                    state.phase4_row_reduction_gpu_to_cpu_bytes_saved_total
                ),
                "phase4_cpu_to_gpu_bytes_total": int(state.phase4_cpu_to_gpu_bytes_total),
                "phase4_copy_count": int(state.phase4_copy_count),
                "phase4_feature_backward_count_total": int(
                    state.phase4_feature_backward_count_total
                ),
                "phase4_feature_produced_tile_count_total": int(
                    state.phase4_feature_produced_tile_count_total
                ),
                "phase4_feature_backward_tile_count_total": int(
                    state.phase4_feature_backward_tile_count_total
                ),
                "phase4_feature_transient_peak_bytes": int(
                    state.phase4_feature_transient_peak_bytes
                ),
                **state.phase4_execution_metadata,
                **(state.phase4_no_refresh_plan_telemetry or {}),
            },
            wall_clock=True,
        )
    )
