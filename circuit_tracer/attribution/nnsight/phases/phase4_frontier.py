"""Frontier state and refresh scheduling operations for NNSight Phase 4."""

from __future__ import annotations
import time
import torch
from tqdm import tqdm
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _Phase4FrontierPlan,
    _apply_phase4_planner_v2_refresh_plan,
    _build_phase4_planner_v2_refresh_telemetry_disabled,
    _build_phase4_scheduler_plan_telemetry,
    _compute_phase4_locality_shaped_frontier_size,
    _compute_phase4_refresh_cycle_batches,
    _compute_phase4_refresh_queue_window_size,
    _plan_phase4_frontier_membership_preserving_v1,
    _reorder_pending_for_phase4_locality,
)
from circuit_tracer.attribution.nnsight.phase_support import _build_vector_stats
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_phase4_refresh_substage_telemetry,
    _record_cross_cluster_checkpoint,
)
from circuit_tracer.observability.events import MemoryBoundary
from circuit_tracer.attribution.nnsight.phases.phase4_influence import (
    rank_feature_frontier,
    recompute_feature_influences,
)
from circuit_tracer.attribution.nnsight.phases.phase4_diagnostics import (
    record_refresh_debug,
    record_refresh_trace,
    _bind_phase4_contract,
)


def _configure_phase4_schedule(state):
    """Resolve scheduling cadence and entry evidence."""
    state.logger.info("Phase 4: Computing feature attributions")
    state.phase4_start = time.perf_counter()
    state.phase4_frontier_buffer_metadata["initial_target_feature_nodes"] = int(
        state.actual_max_feature_nodes
    )
    state.phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] = int(
        state.actual_max_feature_nodes
    )
    state.feature_rows_cpu_staging: torch.Tensor | None = None
    state.telemetry_observer.observe(MemoryBoundary("Phase 4 start", state.model.device))
    state.decoder_chunk_size = getattr(state.model.transcoders, "decoder_chunk_size", None)
    state.phase4_feature_batch_size = state.effective_feature_batch_size
    state.phase4_refresh_queue_multiplier = int(
        state.phase4_refresh_policy_config.effective_queue_multiplier
    )
    state.phase4_refresh_cycle_batches = _compute_phase4_refresh_cycle_batches(
        update_interval=state.update_interval,
        queue_multiplier=state.phase4_refresh_queue_multiplier,
    )
    state.phase4_refresh_reference_cycle_batches = _compute_phase4_refresh_cycle_batches(
        update_interval=state.update_interval, queue_multiplier=1
    )
    state.phase4_refresh_reference_queue_size = _compute_phase4_refresh_queue_window_size(
        update_interval=state.update_interval,
        phase4_feature_batch_size=state.phase4_feature_batch_size,
        queue_multiplier=1,
    )
    state.phase4_refresh_effective_queue_size = _compute_phase4_refresh_queue_window_size(
        update_interval=state.update_interval,
        phase4_feature_batch_size=state.phase4_feature_batch_size,
        queue_multiplier=state.phase4_refresh_queue_multiplier,
    )
    state.phase4_row_executor_effective_mode = state.phase4_row_executor_config.effective_mode
    state.phase4_semantic_batch_max_rows = int(state.phase4_feature_batch_size)
    state.phase4_execution_batch_max_rows = int(state.config.execution_batch_max_rows)
    execution_may_split = (
        state.phase4_execution_batch_max_rows < state.phase4_semantic_batch_max_rows
    )
    execution_may_coalesce = (
        state.phase4_execution_batch_max_rows > state.phase4_semantic_batch_max_rows
    )
    state.phase4_execution_metadata.update(
        {
            "phase4_semantic_batch_max_rows": int(state.phase4_semantic_batch_max_rows),
            "phase4_execution_batch_max_rows": int(state.phase4_execution_batch_max_rows),
            "phase4_execution_may_split": bool(execution_may_split),
            "phase4_execution_may_coalesce": bool(execution_may_coalesce),
            "phase4_execution_policy": "static_semantic_preserving_v1"
            if execution_may_split or execution_may_coalesce
            else state.phase4_row_executor_effective_mode,
            "refresh_cycle_batches_reference": int(state.phase4_refresh_reference_cycle_batches),
            "refresh_cycle_batches_effective": int(state.phase4_refresh_cycle_batches),
            "refresh_queue_size_reference": int(state.phase4_refresh_reference_queue_size),
            "refresh_queue_size_effective": int(state.phase4_refresh_effective_queue_size),
            "feature_vjp_tape_enabled": bool(state.config.feature_vjp_tape_enabled),
            "feature_vjp_tape_batch_window_effective": int(
                state.config.feature_vjp_tape_batch_window
            ),
            "feature_vjp_tape_max_bytes_effective": int(
                state.config.feature_vjp_tape_max_bytes
            ),
            "feature_vjp_tape_fallback_reason": (
                state.config.feature_vjp_tape_fallback_reason
            ),
            "feature_vjp_tape_byte_cap_scope": (
                "simultaneous_host_device_and_row_ownership"
            ),
        }
    )
    state.logger.info(
        f"Phase 4 frontier scheduler | mode={state.phase4_scheduler_config.requested_mode} | version={state.phase4_scheduler_config.version} | policy={state.phase4_scheduler_config.policy} | effective_mode={state.phase4_scheduler_config.effective_mode} | effective_version={state.phase4_scheduler_config.effective_version} | effective_policy={state.phase4_scheduler_config.effective_policy} | effective_behavior={state.phase4_scheduler_config.effective_behavior} | debug={state.phase4_scheduler_config.debug} | telemetry_detail={state.phase4_scheduler_config.telemetry_detail} | exact_chunked_decoder={state.exact_chunked_decoder} | decoder_chunk_size={state.decoder_chunk_size}"
    )
    state.logger.info(
        f"Phase 4 feature batch mode | planner_enabled={state.planner_enabled} | planner_status={state.planner_status} | fixed_feature_batch_size={state.phase4_feature_batch_size} | max_feature_batch_size={state.max_phase4_feature_batch_size}"
        + (
            f" | planner_skip_reason={state.planner_skip_reason}"
            if state.planner_skip_reason is not None
            else ""
        )
    )
    state.logger.info(
        f"Phase 4 execution flags | refresh_optimization={state.phase4_refresh_optimization_config.requested_mode} (effective={state.phase4_refresh_optimization_config.effective_mode}, behavior={state.phase4_refresh_optimization_config.effective_behavior}) | refresh_policy={state.phase4_refresh_policy_config.requested_policy} (effective={state.phase4_refresh_policy_config.effective_policy}, interval_multiplier={state.phase4_refresh_policy_config.requested_interval_multiplier}, interval_multiplier_effective={state.phase4_refresh_policy_config.effective_interval_multiplier}, queue_multiplier_effective={state.phase4_refresh_policy_config.effective_queue_multiplier}, queue_size_reference={state.phase4_refresh_reference_queue_size}, queue_size_effective={state.phase4_refresh_effective_queue_size}, behavior={state.phase4_refresh_policy_config.effective_behavior}) | ranker={state.phase4_ranker_config.requested_mode} (effective={state.phase4_ranker_config.effective_mode}, behavior={state.phase4_ranker_config.effective_behavior}) | row_executor={state.phase4_row_executor_config.requested_mode} (effective={state.phase4_row_executor_config.effective_mode}, behavior={state.phase4_row_executor_config.effective_behavior}) | row_store_cache_control={state.row_store_cache_control_config.requested_mode} (effective={state.row_store_cache_control_config.effective_mode}, behavior={state.row_store_cache_control_config.effective_behavior}) | exact_encoder_residency={state.exact_encoder_residency_config.requested_mode} (effective={state.exact_encoder_residency_config.effective_mode}, behavior={state.exact_encoder_residency_config.effective_behavior}) | semantic_batch_max_rows={state.phase4_semantic_batch_max_rows} | execution_batch_max_rows={state.phase4_execution_batch_max_rows}"
    )
    state.scheduler_uses_reference_planner = state.phase4_scheduler_config.effective_mode in {
        "planner_v1",
        "planner_v2",
    }
    if state.cross_cluster_debug_summary is not None:
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=state.cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=state.cross_cluster_debug_checkpoints,
            checkpoint_name="phase4_entry",
            phase="phase4",
            summary_payload=None,
            stream_payload={
                "checkpoint_stage": "entry",
                "phase4_feature_batch_size": int(state.phase4_feature_batch_size),
                "planner_enabled": bool(state.planner_enabled),
                "planner_status": state.planner_status,
                "planner_skip_reason": state.planner_skip_reason,
                **state.phase4_execution_metadata,
                "actual_max_feature_nodes": int(state.actual_max_feature_nodes),
                "total_active_features": int(state.total_active_feats),
                "update_interval": int(state.update_interval),
            },
        )


def _initialize_phase4_counters(state):
    """Initialize frontier, metrics, and progress state."""
    state.st = state.n_logits
    state.visited = torch.zeros(state.total_active_feats, dtype=torch.bool)
    state.n_visited = 0
    state.phase4_scheduler_reference_batch_count = 0
    state.phase4_execution_batch_count = 0
    state.phase4_coalesced_execution_batch_count = 0
    state.phase4_refresh_count = 0
    state.phase4_frontier_buffer_extra_used_total = 0
    state.phase4_refresh_elapsed_ms_total = 0.0
    state.phase4_feature_batch_elapsed_ms_total = 0.0
    state.phase4_refresh_partial_influence_elapsed_ms_total = 0.0
    state.phase4_refresh_row_store_read_elapsed_ms_total = 0.0
    state.phase4_refresh_rank_topk_elapsed_ms_total = 0.0
    state.phase4_refresh_frontier_plan_elapsed_ms_total = 0.0
    state.phase4_refresh_influence_normalization_elapsed_ms_total = 0.0
    state.phase4_refresh_influence_matmul_elapsed_ms_total = 0.0
    state.phase4_executor_encoder_materialize_elapsed_ms_total = 0.0
    state.phase4_executor_compute_batch_elapsed_ms_total = 0.0
    state.phase4_executor_cpu_staging_elapsed_ms_total = 0.0
    state.phase4_executor_denominator_elapsed_ms_total = 0.0
    state.phase4_executor_denominator_global_max_elapsed_ms_total = 0.0
    state.phase4_executor_denominator_scaled_sum_elapsed_ms_total = 0.0
    state.phase4_executor_row_store_write_elapsed_ms_total = 0.0
    state.phase4_gpu_to_cpu_bytes_total = 0
    state.phase4_row_reduction_gpu_to_cpu_bytes_saved_total = 0
    state.phase4_cpu_to_gpu_bytes_total = 0
    state.phase4_copy_count = 0
    state.phase4_feature_backward_count_total = 0
    state.phase4_feature_produced_tile_count_total = 0
    state.phase4_feature_backward_tile_count_total = 0
    state.phase4_feature_transient_peak_bytes = 0
    state.phase4_feature_vjp_tape_window_count = 0
    state.phase4_feature_vjp_tape_batch_count = 0
    state.phase4_feature_vjp_tape_bytes_total = 0
    state.phase4_feature_vjp_tape_high_watermark_bytes = 0
    state.phase4_feature_vjp_tape_host_bytes_total = 0
    state.phase4_feature_vjp_tape_device_bytes_total = 0
    state.phase4_feature_vjp_tape_row_bytes_total = 0
    state.phase4_feature_vjp_tape_pinned_host_bytes_total = 0
    state.phase4_feature_vjp_tape_pageable_host_bytes_total = 0
    state.phase4_feature_vjp_tape_host_high_watermark_bytes = 0
    state.phase4_feature_vjp_tape_device_high_watermark_bytes = 0
    state.phase4_feature_vjp_tape_row_high_watermark_bytes = 0
    state.phase4_feature_vjp_tape_pinned_host_high_watermark_bytes = 0
    state.phase4_feature_vjp_tape_pageable_host_high_watermark_bytes = 0
    state.phase4_feature_vjp_pin_fallback_count = 0
    state.phase4_feature_vjp_pin_fallback_reasons: set[str] = set()
    state.phase4_feature_vjp_effective_host_placements: set[str] = set()
    state.phase4_feature_vjp_tape_oversize_fallback_batches = 0
    state.phase4_feature_vjp_decoder_replay_count = 0
    state.phase4_feature_vjp_planned_decoder_traversal_numerator = 0
    state.phase4_feature_vjp_planned_decoder_traversal_denominator = 0
    state.phase4_feature_vjp_actual_decoder_counters = {
        "decoder_chunk_request_count": 0,
        "decoder_chunk_request_bytes": 0,
        "decoder_load_count": 0,
        "decoder_load_bytes": 0,
        "decoder_cache_hit_count": 0,
        "decoder_prefetch_request_count": 0,
        "decoder_prefetch_load_count": 0,
        "decoder_prefetch_load_bytes": 0,
        "decoder_prefetch_cache_hit_count": 0,
        "decoder_prefetch_consume_hit_count": 0,
        "decoder_prefetch_host_wait_count": 0,
        "decoder_prefetch_host_wait_seconds": 0.0,
        "decoder_prefetch_consumer_retirement_count": 0,
        "decoder_prefetch_consumer_backpressure_count": 0,
        "decoder_prefetch_consumer_backpressure_seconds": 0.0,
        "decoder_prefetch_owner_open_count": 0,
        "decoder_prefetch_owner_close_count": 0,
    }
    state.phase4_feature_vjp_actual_decoder_prefetch_high_watermarks = {
        "decoder_prefetch_in_flight_high_watermark": 0,
        "decoder_prefetch_in_flight_bytes_high_watermark": 0,
        "decoder_prefetch_consumer_retained_bytes_high_watermark": 0,
        "decoder_prefetch_pipeline_owned_final_page_high_watermark": 0,
        "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": 0,
        "decoder_prefetch_owner_high_watermark": 0,
    }
    state.phase4_feature_vjp_actual_decoder_prefetch_current = {
        "decoder_prefetch_in_flight_count": 0,
        "decoder_prefetch_in_flight_bytes": 0,
        "decoder_prefetch_consumer_active_count": 0,
        "decoder_prefetch_consumer_active_bytes": 0,
        "decoder_prefetch_consumer_retained_count": 0,
        "decoder_prefetch_consumer_retained_bytes": 0,
        "decoder_prefetch_pipeline_owned_final_page_count": 0,
        "decoder_prefetch_pipeline_owned_final_page_bytes": 0,
        "decoder_prefetch_owner_count": 0,
    }
    state.phase4_feature_vjp_actual_decoder_page_load_windows = 0
    state.phase4_feature_vjp_capture_elapsed_ms_total = 0.0
    state.phase4_feature_vjp_replay_elapsed_ms_total = 0.0
    state.phase4_feature_vjp_commit_elapsed_ms_total = 0.0
    state.phase4_no_refresh_plan_telemetry: dict[str, object] | None = None
    state.previous_phase4_pending: torch.Tensor | None = None
    state.first_phase4_pending: torch.Tensor | None = None
    state.phase4_logit_probability_stats: dict[str, object] | None = None
    state.phase4_logit_probabilities = state.targets.logit_probabilities.detach().to(
        device="cpu", dtype=state.exact_trace_internal_dtype_resolved
    )
    if state.anomaly_debug_result is not None:
        state.phase4_logit_probability_stats = _build_vector_stats(
            state.phase4_logit_probabilities, epsilon=1e-12, top_k=8
        )
        state.anomaly_debug_result["logit_probability_stats"] = state.phase4_logit_probability_stats
    state.pbar = tqdm(
        total=state.actual_max_feature_nodes,
        desc="Feature influence computation",
        disable=not state.verbose,
    )


def initialize_phase4(state):
    _bind_phase4_contract(state)
    _configure_phase4_schedule(state)
    _initialize_phase4_counters(state)


def prepare_feature_frontier(state):
    """Select the next pending feature frontier."""
    state.phase4_frontier_plan: _Phase4FrontierPlan | None = None
    state.pending_refresh_index: int | None = None
    if state.actual_max_feature_nodes == state.total_active_feats:
        state.pending = torch.arange(state.total_active_feats)
        if state.scheduler_uses_reference_planner:
            state.phase4_frontier_plan = _plan_phase4_frontier_membership_preserving_v1(
                state.pending,
                max_batch_size=state.phase4_feature_batch_size,
                max_batches=None,
                feat_layers=state.feat_layers,
                feat_positions=state.feat_pos,
                feat_ids=state.feat_ids,
                exact_chunked_decoder=state.exact_chunked_decoder,
                decoder_chunk_size=state.decoder_chunk_size,
                apply_locality_reorder=False,
            )
            state.pending = state.phase4_frontier_plan.selected_frontier
            state.phase4_no_refresh_plan_telemetry = _build_phase4_scheduler_plan_telemetry(
                phase4_frontier_plan=state.phase4_frontier_plan,
                telemetry_detail=state.phase4_scheduler_config.telemetry_detail,
            )
            if state.phase4_scheduler_config.debug:
                state.logger.info(
                    f"Phase 4 scheduler plan | selected_count={state.phase4_frontier_plan.invariant_summary.get('selected_count')} | batch_count={state.phase4_frontier_plan.invariant_summary.get('batch_count')} | boundary_reasons={state.phase4_frontier_plan.boundary_reason_counts}"
                )
    else:
        refresh_frontier(state)


def plan_feature_frontier(state):
    """Plan locality-preserving frontier batches."""
    state.frontier_plan_start = time.perf_counter()
    if state.scheduler_uses_reference_planner:
        state.phase4_frontier_plan = _plan_phase4_frontier_membership_preserving_v1(
            state.pending_candidates,
            max_batch_size=state.phase4_feature_batch_size,
            max_batches=state.phase4_refresh_cycle_batches,
            feat_layers=state.feat_layers,
            feat_positions=state.feat_pos,
            feat_ids=state.feat_ids,
            exact_chunked_decoder=state.exact_chunked_decoder,
            decoder_chunk_size=state.decoder_chunk_size,
            apply_locality_reorder=True,
        )
        state.pending = state.phase4_frontier_plan.selected_frontier
        state.queue_size = int(state.pending.numel())
        if state.phase4_scheduler_config.debug:
            state.logger.info(
                f"Phase 4 scheduler plan | membership_hash={state.phase4_frontier_plan.selected_membership_hash} | order_hash={state.phase4_frontier_plan.selected_order_hash} | fragmentation={state.phase4_frontier_plan.locality_fragmentation_summary} | boundary_reasons={state.phase4_frontier_plan.boundary_reason_counts} | invariants={state.phase4_frontier_plan.invariant_summary}"
            )
    else:
        state.pending = _reorder_pending_for_phase4_locality(
            state.pending_candidates,
            feat_layers=state.feat_layers,
            feat_positions=state.feat_pos,
            feat_ids=state.feat_ids,
            exact_chunked_decoder=state.exact_chunked_decoder,
            decoder_chunk_size=state.decoder_chunk_size,
        )
        state.queue_size = _compute_phase4_locality_shaped_frontier_size(
            state.pending,
            max_batch_size=state.phase4_feature_batch_size,
            max_batches=state.phase4_refresh_cycle_batches,
            feat_layers=state.feat_layers,
            feat_ids=state.feat_ids,
            exact_chunked_decoder=state.exact_chunked_decoder,
            decoder_chunk_size=state.decoder_chunk_size,
        )
        state.pending = state.pending[: state.queue_size]
    state.planner_v2_candidate_window = torch.empty(0, dtype=torch.long)
    state.planner_v2_refresh_telemetry = _build_phase4_planner_v2_refresh_telemetry_disabled()
    if (
        state.phase4_scheduler_config.requested_mode == "planner_v2"
        and state.phase4_frontier_plan is not None
    ):
        assert state.unvisited_feature_rank is not None
        state.planner_v2_candidate_scores = state.feature_influences[state.unvisited_feature_rank]
        (
            state.phase4_frontier_plan,
            state.planner_v2_candidate_window,
            state.planner_v2_refresh_telemetry,
        ) = _apply_phase4_planner_v2_refresh_plan(
            reference_plan=state.phase4_frontier_plan,
            unvisited_feature_rank=state.unvisited_feature_rank,
            candidate_scores=state.planner_v2_candidate_scores,
            visited=state.visited,
            max_batch_size=state.phase4_feature_batch_size,
            max_batches=state.phase4_refresh_cycle_batches,
            feat_layers=state.feat_layers,
            feat_positions=state.feat_pos,
            feat_ids=state.feat_ids,
            exact_chunked_decoder=state.exact_chunked_decoder,
            decoder_chunk_size=state.decoder_chunk_size,
        )
        state.pending = state.phase4_frontier_plan.selected_frontier
        state.queue_size = int(state.pending.numel())
        if state.phase4_scheduler_config.debug:
            state.logger.info(
                f"Phase 4 planner_v2 refresh | reference_frontier_size={state.planner_v2_refresh_telemetry.get('scheduler_planner_v2_reference_frontier_size')} | candidate_window_size={state.planner_v2_refresh_telemetry.get('scheduler_planner_v2_candidate_window_size')} | changed_membership={state.planner_v2_refresh_telemetry.get('scheduler_planner_v2_selection_changed_membership')} | fallback={state.planner_v2_refresh_telemetry.get('scheduler_planner_v2_fallback_to_reference')} | fallback_reason={state.planner_v2_refresh_telemetry.get('scheduler_planner_v2_fallback_reason')}"
            )
    state.phase4_plan_telemetry = _build_phase4_scheduler_plan_telemetry(
        phase4_frontier_plan=state.phase4_frontier_plan,
        telemetry_detail=state.phase4_scheduler_config.telemetry_detail,
    )
    state.refresh_frontier_plan_elapsed_ms = (
        time.perf_counter() - state.frontier_plan_start
    ) * 1000.0
    state.phase4_refresh_frontier_plan_elapsed_ms_total += state.refresh_frontier_plan_elapsed_ms
    state.refresh_substage_telemetry = _build_phase4_refresh_substage_telemetry(
        telemetry_detail=state.phase4_scheduler_config.telemetry_detail,
        partial_influence_elapsed_ms=state.refresh_partial_influence_elapsed_ms,
        rank_topk_elapsed_ms=state.refresh_rank_topk_elapsed_ms,
        frontier_plan_elapsed_ms=state.refresh_frontier_plan_elapsed_ms,
        row_store_read_elapsed_ms=state.refresh_row_store_read_elapsed_ms,
        influence_normalization_elapsed_ms=state.refresh_influence_normalization_elapsed_ms,
        influence_matmul_elapsed_ms=state.refresh_influence_matmul_elapsed_ms,
        chunk_request_count=state.refresh_chunk_request_count,
        active_row_chunk_count=state.refresh_active_row_chunk_count,
        row_reader_row_count=state.refresh_rows_touched,
        solver_iteration_count=state.refresh_solver_iteration_count,
        row_chunk_strategy=state.refresh_row_chunk_strategy,
        row_weight_nonzero_row_count=state.refresh_row_weight_nonzero_rows,
        row_weight_zero_row_count=state.refresh_row_weight_zero_rows,
        row_reader_overread_zero_row_count=state.refresh_row_reader_overread_zero_rows,
        active_row_range_count=state.refresh_active_row_range_count,
        streaming_chunk_reuse_stats=state.streaming_chunk_reuse_stats,
        feature_row_store_read_stats=state.feature_row_store_read_stats,
    )
    state.refresh_memory_after = (
        state.memory_snapshot() if state.refresh_resource_sampled else {}
    )
    state.refresh_elapsed_ms = (time.perf_counter() - state.refresh_start) * 1000.0
    state.phase4_refresh_elapsed_ms_total += state.refresh_elapsed_ms


def refresh_frontier(state):
    recompute_feature_influences(state)
    rank_feature_frontier(state)
    plan_feature_frontier(state)
    record_refresh_trace(state)
    record_refresh_debug(state)
