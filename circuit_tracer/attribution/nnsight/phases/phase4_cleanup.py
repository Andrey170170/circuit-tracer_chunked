"""Completion and cleanup operations for NNSight Phase 4."""

from __future__ import annotations
import time
from circuit_tracer.observability.events import PhaseMetrics, TraceEvent


def finish_phase4(state):
    """Close progress and emit final Phase 4 metrics."""
    state.pbar.close()
    state.phase4_execution_metadata.update(
        {
            "phase4_feature_vjp_actual_decoder_page_load_count_total": int(
                state.phase4_feature_vjp_actual_decoder_counters["decoder_load_count"]
            ),
            "phase4_feature_vjp_actual_decoder_load_bytes_total": int(
                state.phase4_feature_vjp_actual_decoder_counters["decoder_load_bytes"]
            ),
        }
    )
    state.telemetry_observer.observe(
        PhaseMetrics(
            "Feature attributions",
            state.phase4_start,
            state.model.device,
            {
                "selected_features": int(state.visited.sum().item()),
                "final_feature_batch_size": state.phase4_feature_batch_size,
                "phase4_semantic_batch_count": state.phase4_scheduler_reference_batch_count,
                "phase4_semantic_rows": state.n_visited,
                "phase4_execution_batch_count": state.phase4_execution_batch_count,
                "phase4_execution_rows": state.n_visited,
                "phase4_coalesced_execution_batch_count": (
                    state.phase4_coalesced_execution_batch_count
                ),
            },
        )
    )
    state.phase4_elapsed_ms = (time.perf_counter() - state.phase4_start) * 1000.0
    state.phase4_device_timing_summary = state.phase4_device_timing.resolve()
    state.phase4_device_timing_attrs = state.phase4_device_timing_summary.as_attrs(prefix="phase4")
    state.phase4_execution_metadata.update(state.phase4_device_timing_attrs)
    state.telemetry_observer.observe(
        TraceEvent(
            scope="phase",
            name="phase4.feature_attribution",
            phase="phase4",
            elapsed_ms=state.phase4_elapsed_ms,
            attrs={
                "selected_features": int(state.visited.sum().item()),
                "feature_batch_size": int(state.phase4_feature_batch_size),
                "phase4_semantic_batch_count": int(state.phase4_scheduler_reference_batch_count),
                "phase4_semantic_rows": int(state.n_visited),
                "phase4_execution_batch_count": int(state.phase4_execution_batch_count),
                "phase4_execution_rows": int(state.n_visited),
                "phase4_execution_batch_max_rows": int(state.phase4_execution_batch_max_rows),
                "phase4_coalesced_execution_batch_count": int(
                    state.phase4_coalesced_execution_batch_count
                ),
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
                **state.phase4_device_timing_attrs,
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
                "phase4_feature_vjp_tape_window_count": int(
                    state.phase4_feature_vjp_tape_window_count
                ),
                "phase4_feature_vjp_tape_batch_count": int(
                    state.phase4_feature_vjp_tape_batch_count
                ),
                "phase4_feature_vjp_tape_bytes_total": int(
                    state.phase4_feature_vjp_tape_bytes_total
                ),
                "phase4_feature_vjp_tape_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_high_watermark_bytes
                ),
                "phase4_feature_vjp_tape_host_bytes_total": int(
                    state.phase4_feature_vjp_tape_host_bytes_total
                ),
                "phase4_feature_vjp_tape_device_bytes_total": int(
                    state.phase4_feature_vjp_tape_device_bytes_total
                ),
                "phase4_feature_vjp_tape_row_bytes_total": int(
                    state.phase4_feature_vjp_tape_row_bytes_total
                ),
                "phase4_feature_vjp_tape_pinned_host_bytes_total": int(
                    state.phase4_feature_vjp_tape_pinned_host_bytes_total
                ),
                "phase4_feature_vjp_tape_pageable_host_bytes_total": int(
                    state.phase4_feature_vjp_tape_pageable_host_bytes_total
                ),
                "phase4_feature_vjp_tape_host_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_host_high_watermark_bytes
                ),
                "phase4_feature_vjp_tape_device_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_device_high_watermark_bytes
                ),
                "phase4_feature_vjp_tape_row_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_row_high_watermark_bytes
                ),
                "phase4_feature_vjp_tape_pinned_host_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_pinned_host_high_watermark_bytes
                ),
                "phase4_feature_vjp_tape_pageable_host_high_watermark_bytes": int(
                    state.phase4_feature_vjp_tape_pageable_host_high_watermark_bytes
                ),
                "phase4_feature_vjp_pin_fallback_count": int(
                    state.phase4_feature_vjp_pin_fallback_count
                ),
                "phase4_feature_vjp_pin_fallback_reasons": tuple(
                    sorted(state.phase4_feature_vjp_pin_fallback_reasons)
                ),
                "phase4_feature_vjp_effective_host_placements": tuple(
                    sorted(state.phase4_feature_vjp_effective_host_placements)
                ),
                "phase4_feature_vjp_tape_oversize_fallback_batches": int(
                    state.phase4_feature_vjp_tape_oversize_fallback_batches
                ),
                "phase4_feature_vjp_decoder_replay_count": int(
                    state.phase4_feature_vjp_decoder_replay_count
                ),
                "phase4_feature_vjp_planned_decoder_traversal_numerator": int(
                    state.phase4_feature_vjp_planned_decoder_traversal_numerator
                ),
                "phase4_feature_vjp_planned_decoder_traversal_denominator": int(
                    state.phase4_feature_vjp_planned_decoder_traversal_denominator
                ),
                "phase4_feature_vjp_actual_decoder_page_load_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_load_count"]
                ),
                "phase4_feature_vjp_actual_decoder_load_bytes_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_load_bytes"]
                ),
                "phase4_feature_vjp_actual_decoder_request_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_chunk_request_count"]
                ),
                "phase4_feature_vjp_actual_decoder_request_bytes_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_chunk_request_bytes"]
                ),
                "phase4_feature_vjp_actual_decoder_cache_hit_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_cache_hit_count"]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_request_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_request_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_load_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_prefetch_load_count"]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_load_bytes_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters["decoder_prefetch_load_bytes"]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_cache_hit_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_cache_hit_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consume_hit_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_consume_hit_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_host_wait_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_host_wait_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_host_wait_seconds_total": float(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_host_wait_seconds"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_in_flight_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_in_flight_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_in_flight_bytes_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_in_flight_bytes_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_in_flight_count_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_in_flight_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_in_flight_bytes_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_in_flight_bytes"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_retirement_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_consumer_retirement_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_backpressure_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_consumer_backpressure_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_backpressure_seconds_total": float(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_consumer_backpressure_seconds"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_owner_open_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_owner_open_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_owner_close_count_total": int(
                    state.phase4_feature_vjp_actual_decoder_counters[
                        "decoder_prefetch_owner_close_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_retained_bytes_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_consumer_retained_bytes_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_pipeline_owned_final_page_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_owner_high_watermark": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks[
                        "decoder_prefetch_owner_high_watermark"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_active_count_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_consumer_active_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_active_bytes_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_consumer_active_bytes"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_retained_count_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_consumer_retained_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_consumer_retained_bytes_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_consumer_retained_bytes"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_count_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_pipeline_owned_final_page_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_bytes_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_pipeline_owned_final_page_bytes"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_prefetch_owner_count_final": int(
                    state.phase4_feature_vjp_actual_decoder_prefetch_current[
                        "decoder_prefetch_owner_count"
                    ]
                ),
                "phase4_feature_vjp_actual_decoder_page_load_windows": int(
                    state.phase4_feature_vjp_actual_decoder_page_load_windows
                ),
                "phase4_feature_vjp_capture_elapsed_ms_total": float(
                    state.phase4_feature_vjp_capture_elapsed_ms_total
                ),
                "phase4_feature_vjp_replay_elapsed_ms_total": float(
                    state.phase4_feature_vjp_replay_elapsed_ms_total
                ),
                "phase4_feature_vjp_commit_elapsed_ms_total": float(
                    state.phase4_feature_vjp_commit_elapsed_ms_total
                ),
                **state.phase4_execution_metadata,
                **(state.phase4_no_refresh_plan_telemetry or {}),
            },
            wall_clock=True,
        )
    )
