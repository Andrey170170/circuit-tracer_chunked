"""Phase 4 feature attribution execution for the NNSight backend."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _PHASE4_REFRESH_MEMORY_ATTR_KEYS,
    _Phase4FrontierPlan,
    _apply_phase4_planner_v2_refresh_plan,
    _build_phase4_batch_locality_summary,
    _build_phase4_planner_v2_refresh_telemetry_disabled,
    _build_phase4_scheduler_plan_telemetry,
    _compute_phase4_locality_shaped_batch_end,
    _compute_phase4_locality_shaped_frontier_size,
    _compute_phase4_rank_selection_max_feature_nodes_cap_bound,
    _compute_phase4_refresh_cycle_batches,
    _compute_phase4_refresh_queue_window_size,
    _plan_phase4_frontier_membership_preserving_v1,
    _rank_phase4_unvisited_features_argsort,
    _reorder_pending_for_phase4_locality,
    _select_phase4_frontier_rank_selection,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_matrix_abs_stats,
    _build_phase4_deterministic_shadow_pending,
    _build_phase4_frontier_buffer_decision,
    _build_phase4_normalization_stats,
    _build_vector_stats,
    _compare_phase4_frontiers,
    _copy_feature_rows_to_cpu_staging,
    _copy_rows_to_cpu_staging,
    _record_phase4_refresh_debug,
)
from circuit_tracer.attribution.nnsight.replay import (
    _compute_row_denominator_scaled_l1,
    _hash_index_tensor,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_phase4_executor_batch_telemetry,
    _build_phase4_executor_substage_telemetry,
    _build_phase4_gpu_row_reduction_transfer_telemetry,
    _build_phase4_refresh_substage_telemetry,
    _build_row_transfer_telemetry,
    _record_cross_cluster_batch_event,
    _record_cross_cluster_checkpoint,
    _safe_float,
    _safe_int,
    _tensor_nbytes_estimate,
)
from circuit_tracer.attribution.nnsight.tiled_rows import produce_and_store_tiled_rows
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_feature_influences_tiled,
    compute_partial_influences,
)
from circuit_tracer.observability.human_logs import (
    _log_batch_profile,
    _log_memory_boundary,
    _log_phase_metrics,
    _snapshot_diagnostics,
)
from circuit_tracer.utils.telemetry import (
    build_memory_before_after_attrs,
    diff_numeric_metrics,
    get_memory_snapshot,
)


@dataclass(frozen=True)
class Phase4Inputs:
    logger: Any
    model: Any
    ctx: Any
    targets: AttributionTargets
    edge_matrix: torch.Tensor | None
    feat_ids: torch.Tensor
    feat_layers: torch.Tensor
    feat_pos: torch.Tensor
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
    row_to_node_index: torch.Tensor
    telemetry_observer: Any
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    anomaly_debug_result: dict[str, object] | None
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    rows_cpu_staging: torch.Tensor | None


@dataclass(frozen=True)
class Phase4Config:
    actual_max_feature_nodes: int
    total_active_feats: int
    n_logits: int
    logit_offset: int
    effective_feature_batch_size: int
    compute_microbatch_max_rows: int
    max_phase4_feature_batch_size: int
    update_interval: int
    row_store_capacity_feature_nodes: int
    exact_trace_internal_dtype_resolved: torch.dtype
    influence_compute_dtype: torch.dtype
    shadow_debug_compute_dtype: torch.dtype
    exact_chunked_decoder: bool
    use_compact_feature_row_store: bool
    planner_enabled: bool
    planner_status: str
    planner_skip_reason: str | None
    phase4_debug_summary_enabled: bool
    cross_cluster_debug_enabled: bool
    phase4_frontier_buffer_relative_epsilon: float | None
    phase4_frontier_buffer_max_extra_per_refresh: int
    phase4_frontier_buffer_max_extra_total: int
    phase4_refresh_prepared_chunk_cache_bytes_effective: int
    phase4_refresh_active_row_accumulation_effective: str
    phase4_scheduler_config: Any
    phase4_refresh_optimization_config: Any
    phase4_refresh_policy_config: Any
    phase4_ranker_config: Any
    phase4_row_executor_config: Any
    phase4_row_reduction_config: Any
    row_store_cache_control_config: Any
    exact_encoder_residency_config: Any
    profile: bool
    profile_log_interval: int
    verbose: bool
    full_retention_backend: str = "full_file"
    influence_row_tile_size: int = 4096
    influence_column_tile_size: int = 2048
    feature_row_column_tile_size: int = 2048


@dataclass(frozen=True)
class Phase4Result:
    visited: torch.Tensor
    actual_max_feature_nodes: int
    edge_matrix: torch.Tensor | None
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
    row_to_node_index: torch.Tensor
    rows_cpu_staging: torch.Tensor | None
    st: int
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    anomaly_debug_result: dict[str, object] | None
    phase4_elapsed_ms: float
    phase4_feature_batch_size: int
    phase4_executor_reference_batch_size: int
    phase4_executor_microbatch_size: int
    phase4_refresh_count: int
    phase4_scheduler_reference_batch_count: int
    phase4_executor_microbatch_count: int
    phase4_refresh_elapsed_ms_total: float
    phase4_feature_batch_elapsed_ms_total: float
    phase4_refresh_partial_influence_elapsed_ms_total: float
    phase4_refresh_rank_topk_elapsed_ms_total: float
    phase4_refresh_frontier_plan_elapsed_ms_total: float
    phase4_refresh_row_store_read_elapsed_ms_total: float


def run_phase4(*, inputs: Phase4Inputs, config: Phase4Config) -> Phase4Result:
    """Run feature attribution and return all state consumed by Phase 5."""
    logger = inputs.logger
    model = inputs.model
    ctx = inputs.ctx
    targets = inputs.targets
    edge_matrix = inputs.edge_matrix
    feat_ids = inputs.feat_ids
    feat_layers = inputs.feat_layers
    feat_pos = inputs.feat_pos
    feature_row_store = inputs.feature_row_store
    nonfeature_row_store = inputs.nonfeature_row_store
    row_to_node_index = inputs.row_to_node_index
    telemetry_observer = inputs.telemetry_observer
    cross_cluster_debug_summary = inputs.cross_cluster_debug_summary
    cross_cluster_debug_checkpoints = inputs.cross_cluster_debug_checkpoints
    cross_cluster_debug_batches = inputs.cross_cluster_debug_batches
    anomaly_debug_result = inputs.anomaly_debug_result
    phase4_frontier_buffer_metadata = inputs.phase4_frontier_buffer_metadata
    phase4_execution_metadata = inputs.phase4_execution_metadata
    rows_cpu_staging = inputs.rows_cpu_staging
    actual_max_feature_nodes = config.actual_max_feature_nodes
    total_active_feats = config.total_active_feats
    n_logits = config.n_logits
    logit_offset = config.logit_offset
    effective_feature_batch_size = config.effective_feature_batch_size
    max_phase4_feature_batch_size = config.max_phase4_feature_batch_size
    update_interval = config.update_interval
    row_store_capacity_feature_nodes = config.row_store_capacity_feature_nodes
    exact_trace_internal_dtype_resolved = config.exact_trace_internal_dtype_resolved
    influence_compute_dtype = config.influence_compute_dtype
    shadow_debug_compute_dtype = config.shadow_debug_compute_dtype
    exact_chunked_decoder = config.exact_chunked_decoder
    use_compact_feature_row_store = config.use_compact_feature_row_store
    planner_enabled = config.planner_enabled
    planner_status = config.planner_status
    planner_skip_reason = config.planner_skip_reason
    phase4_debug_summary_enabled = config.phase4_debug_summary_enabled
    cross_cluster_debug_enabled = config.cross_cluster_debug_enabled
    phase4_frontier_buffer_relative_epsilon = config.phase4_frontier_buffer_relative_epsilon
    phase4_frontier_buffer_max_extra_per_refresh = (
        config.phase4_frontier_buffer_max_extra_per_refresh
    )
    phase4_frontier_buffer_max_extra_total = config.phase4_frontier_buffer_max_extra_total
    phase4_refresh_prepared_chunk_cache_bytes_effective = (
        config.phase4_refresh_prepared_chunk_cache_bytes_effective
    )
    phase4_refresh_active_row_accumulation_effective = (
        config.phase4_refresh_active_row_accumulation_effective
    )
    phase4_scheduler_config = config.phase4_scheduler_config
    phase4_refresh_optimization_config = config.phase4_refresh_optimization_config
    phase4_refresh_policy_config = config.phase4_refresh_policy_config
    phase4_ranker_config = config.phase4_ranker_config
    phase4_row_executor_config = config.phase4_row_executor_config
    phase4_row_reduction_config = config.phase4_row_reduction_config
    row_store_cache_control_config = config.row_store_cache_control_config
    exact_encoder_residency_config = config.exact_encoder_residency_config
    profile = config.profile
    profile_log_interval = config.profile_log_interval
    verbose = config.verbose
    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase4_start = time.perf_counter()
    phase4_frontier_buffer_metadata["initial_target_feature_nodes"] = int(actual_max_feature_nodes)
    phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] = int(
        actual_max_feature_nodes
    )
    feature_rows_cpu_staging: torch.Tensor | None = None
    _log_memory_boundary(logger, "Phase 4 start", model.device)
    decoder_chunk_size = getattr(model.transcoders, "decoder_chunk_size", None)
    phase4_feature_batch_size = effective_feature_batch_size
    phase4_refresh_queue_multiplier = int(phase4_refresh_policy_config.effective_queue_multiplier)
    phase4_refresh_cycle_batches = _compute_phase4_refresh_cycle_batches(
        update_interval=update_interval,
        queue_multiplier=phase4_refresh_queue_multiplier,
    )
    phase4_refresh_reference_cycle_batches = _compute_phase4_refresh_cycle_batches(
        update_interval=update_interval,
        queue_multiplier=1,
    )
    phase4_refresh_reference_queue_size = _compute_phase4_refresh_queue_window_size(
        update_interval=update_interval,
        phase4_feature_batch_size=phase4_feature_batch_size,
        queue_multiplier=1,
    )
    phase4_refresh_effective_queue_size = _compute_phase4_refresh_queue_window_size(
        update_interval=update_interval,
        phase4_feature_batch_size=phase4_feature_batch_size,
        queue_multiplier=phase4_refresh_queue_multiplier,
    )
    phase4_row_executor_effective_mode = phase4_row_executor_config.effective_mode
    phase4_executor_reference_batch_size = int(phase4_feature_batch_size)
    phase4_executor_microbatch_size = min(
        phase4_executor_reference_batch_size, config.compute_microbatch_max_rows
    )
    executor_physically_split = (
        phase4_executor_microbatch_size < phase4_executor_reference_batch_size
    )
    phase4_execution_metadata.update(
        {
            "executor_configured_reference_batch_size": int(phase4_executor_reference_batch_size),
            "executor_reference_batch_size": int(phase4_executor_reference_batch_size),
            "executor_microbatch_size": int(phase4_executor_microbatch_size),
            "executor_physically_split": bool(executor_physically_split),
            "executor_effective_execution": (
                "physical_microbatching_v1"
                if executor_physically_split
                else phase4_row_executor_effective_mode
            ),
            "refresh_cycle_batches_reference": int(phase4_refresh_reference_cycle_batches),
            "refresh_cycle_batches_effective": int(phase4_refresh_cycle_batches),
            "refresh_queue_size_reference": int(phase4_refresh_reference_queue_size),
            "refresh_queue_size_effective": int(phase4_refresh_effective_queue_size),
        }
    )
    logger.info(
        "Phase 4 frontier scheduler | "
        f"mode={phase4_scheduler_config.requested_mode} | "
        f"version={phase4_scheduler_config.version} | "
        f"policy={phase4_scheduler_config.policy} | "
        f"effective_mode={phase4_scheduler_config.effective_mode} | "
        f"effective_version={phase4_scheduler_config.effective_version} | "
        f"effective_policy={phase4_scheduler_config.effective_policy} | "
        f"effective_behavior={phase4_scheduler_config.effective_behavior} | "
        f"debug={phase4_scheduler_config.debug} | "
        f"telemetry_detail={phase4_scheduler_config.telemetry_detail} | "
        f"exact_chunked_decoder={exact_chunked_decoder} | "
        f"decoder_chunk_size={decoder_chunk_size}"
    )
    logger.info(
        "Phase 4 feature batch mode | "
        f"planner_enabled={planner_enabled} | "
        f"planner_status={planner_status} | "
        f"fixed_feature_batch_size={phase4_feature_batch_size} | "
        f"max_feature_batch_size={max_phase4_feature_batch_size}"
        + (
            f" | planner_skip_reason={planner_skip_reason}"
            if planner_skip_reason is not None
            else ""
        )
    )
    logger.info(
        "Phase 4 execution flags | "
        f"refresh_optimization={phase4_refresh_optimization_config.requested_mode}"
        f" (effective={phase4_refresh_optimization_config.effective_mode}, "
        f"behavior={phase4_refresh_optimization_config.effective_behavior}) | "
        f"refresh_policy={phase4_refresh_policy_config.requested_policy}"
        f" (effective={phase4_refresh_policy_config.effective_policy}, "
        f"interval_multiplier={phase4_refresh_policy_config.requested_interval_multiplier}, "
        f"interval_multiplier_effective={phase4_refresh_policy_config.effective_interval_multiplier}, "
        f"queue_multiplier_effective={phase4_refresh_policy_config.effective_queue_multiplier}, "
        f"queue_size_reference={phase4_refresh_reference_queue_size}, "
        f"queue_size_effective={phase4_refresh_effective_queue_size}, "
        f"behavior={phase4_refresh_policy_config.effective_behavior}) | "
        f"ranker={phase4_ranker_config.requested_mode}"
        f" (effective={phase4_ranker_config.effective_mode}, "
        f"behavior={phase4_ranker_config.effective_behavior}) | "
        f"row_executor={phase4_row_executor_config.requested_mode}"
        f" (effective={phase4_row_executor_config.effective_mode}, "
        f"behavior={phase4_row_executor_config.effective_behavior}) | "
        f"row_store_cache_control={row_store_cache_control_config.requested_mode}"
        f" (effective={row_store_cache_control_config.effective_mode}, "
        f"behavior={row_store_cache_control_config.effective_behavior}) | "
        f"exact_encoder_residency={exact_encoder_residency_config.requested_mode}"
        f" (effective={exact_encoder_residency_config.effective_mode}, "
        f"behavior={exact_encoder_residency_config.effective_behavior}) | "
        f"executor_reference_batch_size={phase4_executor_reference_batch_size} | "
        f"executor_microbatch_size={phase4_executor_microbatch_size}"
    )
    scheduler_uses_reference_planner = phase4_scheduler_config.effective_mode in {
        "planner_v1",
        "planner_v2",
    }
    if cross_cluster_debug_summary is not None:
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
            checkpoint_name="phase4_entry",
            phase="phase4",
            summary_payload=None,
            stream_payload={
                "checkpoint_stage": "entry",
                "phase4_feature_batch_size": int(phase4_feature_batch_size),
                "planner_enabled": bool(planner_enabled),
                "planner_status": planner_status,
                "planner_skip_reason": planner_skip_reason,
                **phase4_execution_metadata,
                "actual_max_feature_nodes": int(actual_max_feature_nodes),
                "total_active_features": int(total_active_feats),
                "update_interval": int(update_interval),
            },
        )
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0
    phase4_scheduler_reference_batch_count = 0
    phase4_executor_microbatch_count = 0
    phase4_refresh_count = 0
    phase4_frontier_buffer_extra_used_total = 0
    phase4_refresh_elapsed_ms_total = 0.0
    phase4_feature_batch_elapsed_ms_total = 0.0
    phase4_refresh_partial_influence_elapsed_ms_total = 0.0
    phase4_refresh_row_store_read_elapsed_ms_total = 0.0
    phase4_refresh_rank_topk_elapsed_ms_total = 0.0
    phase4_refresh_frontier_plan_elapsed_ms_total = 0.0
    phase4_refresh_influence_normalization_elapsed_ms_total = 0.0
    phase4_refresh_influence_matmul_elapsed_ms_total = 0.0
    phase4_executor_encoder_materialize_elapsed_ms_total = 0.0
    phase4_executor_compute_batch_elapsed_ms_total = 0.0
    phase4_executor_cpu_staging_elapsed_ms_total = 0.0
    phase4_executor_denominator_elapsed_ms_total = 0.0
    phase4_executor_row_store_write_elapsed_ms_total = 0.0
    phase4_gpu_to_cpu_bytes_total = 0
    phase4_row_reduction_gpu_to_cpu_bytes_saved_total = 0
    phase4_cpu_to_gpu_bytes_total = 0
    phase4_copy_count = 0
    phase4_no_refresh_plan_telemetry: dict[str, object] | None = None
    previous_phase4_pending: torch.Tensor | None = None
    first_phase4_pending: torch.Tensor | None = None
    phase4_logit_probability_stats: dict[str, object] | None = None
    phase4_logit_probabilities = targets.logit_probabilities.detach().to(
        device="cpu",
        dtype=exact_trace_internal_dtype_resolved,
    )
    if anomaly_debug_result is not None:
        phase4_logit_probability_stats = _build_vector_stats(
            phase4_logit_probabilities,
            epsilon=1e-12,
            top_k=8,
        )
        anomaly_debug_result["logit_probability_stats"] = phase4_logit_probability_stats

    pbar = tqdm(
        total=actual_max_feature_nodes,
        desc="Feature influence computation",
        disable=not verbose,
    )

    while n_visited < actual_max_feature_nodes:
        phase4_frontier_plan: _Phase4FrontierPlan | None = None
        pending_refresh_index: int | None = None
        if actual_max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
            if scheduler_uses_reference_planner:
                phase4_frontier_plan = _plan_phase4_frontier_membership_preserving_v1(
                    pending,
                    max_batch_size=phase4_feature_batch_size,
                    max_batches=None,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                    apply_locality_reorder=False,
                )
                pending = phase4_frontier_plan.selected_frontier
                phase4_no_refresh_plan_telemetry = _build_phase4_scheduler_plan_telemetry(
                    phase4_frontier_plan=phase4_frontier_plan,
                    telemetry_detail=phase4_scheduler_config.telemetry_detail,
                )
                if phase4_scheduler_config.debug:
                    logger.info(
                        "Phase 4 scheduler plan | "
                        f"selected_count={phase4_frontier_plan.invariant_summary.get('selected_count')} | "
                        f"batch_count={phase4_frontier_plan.invariant_summary.get('batch_count')} | "
                        f"boundary_reasons={phase4_frontier_plan.boundary_reason_counts}"
                    )
        else:
            refresh_index = int(phase4_refresh_count)
            pending_refresh_index = refresh_index
            refresh_start = time.perf_counter()
            refresh_memory_before = get_memory_snapshot(model.device)
            feature_row_store_snapshot_before = (
                feature_row_store.get_diagnostic_snapshot()
                if use_compact_feature_row_store and feature_row_store is not None
                else None
            )
            streaming_chunk_reuse_stats: dict[str, int | float | str] | None = None
            refresh_row_store_read_elapsed_ms: float | None = None
            refresh_influence_normalization_elapsed_ms: float | None = None
            refresh_influence_matmul_elapsed_ms: float | None = None
            refresh_chunk_request_count: int | None = None
            refresh_active_row_chunk_count: int | None = None
            refresh_rows_touched: int | None = None
            refresh_solver_iteration_count: int | None = None
            refresh_row_chunk_strategy: str | None = None
            refresh_row_weight_nonzero_rows: int | None = None
            refresh_row_weight_zero_rows: int | None = None
            refresh_row_reader_overread_zero_rows: int | None = None
            refresh_active_row_range_count: int | None = None
            partial_influence_start = time.perf_counter()
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                streaming_chunk_reuse_stats = {}
                refresh_active_row_only_chunks = (
                    phase4_refresh_optimization_config.effective_mode == "v1"
                )
                row_denominator_prefix = (
                    feature_row_store.row_abs_max[:st],
                    feature_row_store.row_l1_scaled[:st],
                )
                refresh_prepared_row_reader = bool(
                    phase4_refresh_prepared_chunk_cache_bytes_effective > 0
                )
                if refresh_prepared_row_reader:

                    def refresh_row_reader(row_start: int, row_end: int) -> torch.Tensor:
                        return feature_row_store.read_prepared_feature_rows(
                            row_start,
                            row_end,
                            device=feature_row_store.row_abs_max.device,
                            dtype=influence_compute_dtype,
                            phase="phase4",
                        )
                else:

                    def refresh_row_reader(row_start: int, row_end: int) -> torch.Tensor:
                        return feature_row_store.read_feature_rows(
                            row_start,
                            row_end,
                            phase="phase4",
                        )

                if config.full_retention_backend == "column_tiled_v1":
                    feature_influences = compute_partial_feature_influences_tiled(
                        feature_row_store.read_tile,
                        row_denominator_prefix,
                        phase4_logit_probabilities,
                        row_to_node_index[:st],
                        n_feature_nodes=total_active_feats,
                        n_logits=n_logits,
                        row_tile_size=config.influence_row_tile_size,
                        column_tile_size=config.influence_column_tile_size,
                        device=feature_row_store.row_abs_max.device,
                        compute_dtype=influence_compute_dtype,
                        telemetry=streaming_chunk_reuse_stats,
                    )
                else:
                    feature_influences = compute_partial_feature_influences_streaming(
                        refresh_row_reader,
                        row_denominator_prefix,
                        phase4_logit_probabilities,
                        row_to_node_index[:st],
                        n_feature_nodes=total_active_feats,
                        n_logits=n_logits,
                        device=feature_row_store.row_abs_max.device,
                        chunk_reuse_stats=streaming_chunk_reuse_stats,
                        compute_dtype=influence_compute_dtype,
                        active_row_only_chunks=refresh_active_row_only_chunks,
                        row_reader_returns_prepared=refresh_prepared_row_reader,
                        active_row_accumulation=phase4_refresh_active_row_accumulation_effective,
                    )
                refresh_row_store_read_elapsed_ms = _safe_float(
                    streaming_chunk_reuse_stats.get("row_reader_elapsed_ms_total")
                )
                refresh_influence_normalization_elapsed_ms = _safe_float(
                    streaming_chunk_reuse_stats.get("normalization_elapsed_ms_total")
                )
                refresh_influence_matmul_elapsed_ms = _safe_float(
                    streaming_chunk_reuse_stats.get("matmul_elapsed_ms_total")
                )
                refresh_direct_accumulation_elapsed_ms = _safe_float(
                    streaming_chunk_reuse_stats.get("direct_accumulation_elapsed_ms_total")
                )
                if refresh_direct_accumulation_elapsed_ms is not None:
                    refresh_influence_matmul_elapsed_ms = float(
                        refresh_influence_matmul_elapsed_ms or 0.0
                    ) + float(refresh_direct_accumulation_elapsed_ms)
                refresh_chunk_request_count = _safe_int(
                    streaming_chunk_reuse_stats.get("chunk_request_count")
                )
                refresh_active_row_chunk_count = _safe_int(
                    streaming_chunk_reuse_stats.get("active_row_chunk_count")
                )
                refresh_rows_touched = _safe_int(
                    streaming_chunk_reuse_stats.get("row_reader_row_count")
                )
                refresh_solver_iteration_count = _safe_int(
                    streaming_chunk_reuse_stats.get("iteration_count")
                )
                row_chunk_strategy_value = streaming_chunk_reuse_stats.get("row_chunk_strategy")
                if isinstance(row_chunk_strategy_value, str):
                    refresh_row_chunk_strategy = row_chunk_strategy_value
                refresh_row_weight_nonzero_rows = _safe_int(
                    streaming_chunk_reuse_stats.get("row_weight_nonzero_row_count")
                )
                refresh_row_weight_zero_rows = _safe_int(
                    streaming_chunk_reuse_stats.get("row_weight_zero_row_count")
                )
                refresh_row_reader_overread_zero_rows = _safe_int(
                    streaming_chunk_reuse_stats.get("row_reader_overread_zero_row_count")
                )
                refresh_active_row_range_count = _safe_int(
                    streaming_chunk_reuse_stats.get("active_row_range_count")
                )
            else:
                influences = compute_partial_influences(
                    edge_matrix[:st],
                    phase4_logit_probabilities,
                    row_to_node_index[:st],
                    device=edge_matrix.device,
                )
                feature_influences = influences[:total_active_feats]

            refresh_partial_influence_elapsed_ms = (
                time.perf_counter() - partial_influence_start
            ) * 1000.0
            phase4_refresh_partial_influence_elapsed_ms_total += (
                refresh_partial_influence_elapsed_ms
            )
            if refresh_row_store_read_elapsed_ms is not None:
                phase4_refresh_row_store_read_elapsed_ms_total += refresh_row_store_read_elapsed_ms
            if refresh_influence_normalization_elapsed_ms is not None:
                phase4_refresh_influence_normalization_elapsed_ms_total += (
                    refresh_influence_normalization_elapsed_ms
                )
            if refresh_influence_matmul_elapsed_ms is not None:
                phase4_refresh_influence_matmul_elapsed_ms_total += (
                    refresh_influence_matmul_elapsed_ms
                )

            max_frontier_size = min(
                _compute_phase4_refresh_queue_window_size(
                    update_interval=update_interval,
                    phase4_feature_batch_size=phase4_feature_batch_size,
                    queue_multiplier=phase4_refresh_queue_multiplier,
                ),
                int(actual_max_feature_nodes - n_visited),
            )

            phase4_frontier_buffer_event: dict[str, object] | None = None
            if bool(phase4_frontier_buffer_metadata["enabled"]):
                unvisited_scores_for_buffer = feature_influences[
                    _rank_phase4_unvisited_features_argsort(feature_influences, visited)
                ]
                buffer_decision = _build_phase4_frontier_buffer_decision(
                    candidate_scores=unvisited_scores_for_buffer,
                    base_frontier_size=int(max_frontier_size),
                    actual_max_feature_nodes=int(actual_max_feature_nodes),
                    capacity_feature_nodes=int(row_store_capacity_feature_nodes),
                    total_active_features=int(total_active_feats),
                    used_total=int(phase4_frontier_buffer_extra_used_total),
                    epsilon=phase4_frontier_buffer_relative_epsilon,
                    max_per_refresh=int(phase4_frontier_buffer_max_extra_per_refresh),
                    max_total=int(phase4_frontier_buffer_max_extra_total),
                    refresh_index=refresh_index,
                    visited_before=int(n_visited),
                )
                extra = int(buffer_decision["extra_feature_count"])
                phase4_frontier_buffer_event = cast(dict[str, object], buffer_decision["event"])
                cast(list[dict[str, object]], phase4_frontier_buffer_metadata["events"]).append(
                    phase4_frontier_buffer_event
                )
                if extra > 0:
                    phase4_frontier_buffer_extra_used_total += extra
                    actual_max_feature_nodes += extra
                    max_frontier_size = int(buffer_decision["expanded_frontier_size"])
                    phase4_frontier_buffer_metadata["extra_feature_count_total"] = int(
                        phase4_frontier_buffer_extra_used_total
                    )
                    phase4_frontier_buffer_metadata["expanded_refresh_count"] = (
                        int(phase4_frontier_buffer_metadata["expanded_refresh_count"]) + 1
                    )
                    phase4_frontier_buffer_metadata["effective"] = True
                    phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] = int(
                        actual_max_feature_nodes
                    )
                    if getattr(pbar, "total", None) is not None:
                        pbar.total = int(actual_max_feature_nodes)
                        pbar.refresh()
                elif phase4_frontier_buffer_event.get("fallback_reason") is not None:
                    phase4_frontier_buffer_metadata["fallback_count"] = (
                        int(phase4_frontier_buffer_metadata["fallback_count"]) + 1
                    )

            rank_topk_start = time.perf_counter()
            rank_selection = _select_phase4_frontier_rank_selection(
                feature_influences=feature_influences,
                visited=visited,
                frontier_size=max_frontier_size,
                ranker_mode=phase4_ranker_config.effective_mode,
            )
            pending_candidates = rank_selection.selected_frontier
            unvisited_feature_rank: torch.Tensor | None = None
            if (
                phase4_scheduler_config.requested_mode == "planner_v2"
                or phase4_debug_summary_enabled
            ):
                unvisited_feature_rank = _rank_phase4_unvisited_features_argsort(
                    feature_influences,
                    visited,
                )

            max_feature_nodes_cap_bound = (
                _compute_phase4_rank_selection_max_feature_nodes_cap_bound(
                    candidate_count=int(rank_selection.candidate_count),
                    actual_max_feature_nodes=int(actual_max_feature_nodes),
                    n_visited=int(n_visited),
                    max_frontier_size=int(max_frontier_size),
                )
            )

            ranker_refresh_telemetry = {
                "ranker_frontier_candidate_count": int(rank_selection.candidate_count),
                "ranker_frontier_selected_count": int(rank_selection.selected_count),
                "ranker_frontier_selected_hash": rank_selection.selected_order_hash,
                "ranker_frontier_selected_order_hash": rank_selection.selected_order_hash,
                "ranker_frontier_selected_membership_hash": (
                    rank_selection.selected_membership_hash
                ),
                "ranker_frontier_cutoff_score": rank_selection.cutoff_score,
                "ranker_frontier_cutoff_gap": rank_selection.cutoff_gap,
                "ranker_frontier_relative_cutoff_gap": rank_selection.relative_cutoff_gap,
                "ranker_frontier_near_cutoff_epsilon": rank_selection.near_cutoff_epsilon,
                "ranker_frontier_near_cutoff_count": int(rank_selection.near_cutoff_count),
                "ranker_frontier_max_feature_nodes_cap_bound": bool(max_feature_nodes_cap_bound),
                "ranker_frontier_tie_count_at_cutoff": int(rank_selection.tie_count_at_cutoff),
                "ranker_frontier_tie_at_cutoff": bool(rank_selection.tie_at_cutoff),
                "ranker_frontier_tie_behavior": rank_selection.tie_behavior,
            }
            if (
                (cross_cluster_debug_enabled or phase4_scheduler_config.telemetry_detail == "debug")
                and rank_selection.cutoff_score is not None
                and rank_selection.cutoff_score > 0
            ):
                unvisited_scores_for_cutoff = (
                    feature_influences[
                        _rank_phase4_unvisited_features_argsort(feature_influences, visited)
                    ]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                )
                cutoff_score_for_profile = float(rank_selection.cutoff_score)
                below_cutoff_scores = unvisited_scores_for_cutoff[
                    int(rank_selection.selected_count) :
                ]
                near_cutoff_counts = {
                    str(eps): int(
                        (below_cutoff_scores >= cutoff_score_for_profile * (1.0 - float(eps)))
                        .sum()
                        .item()
                    )
                    for eps in (0.001, 0.01, 0.05)
                }
                ranker_refresh_telemetry["ranker_frontier_near_cutoff_counts"] = near_cutoff_counts
            candidate_scores: torch.Tensor | None = None
            rank_signal_stats: dict[str, object] | None = None
            normalization_input_stats: dict[str, object] | None = None
            if phase4_debug_summary_enabled:
                if unvisited_feature_rank is not None:
                    candidate_scores = feature_influences[unvisited_feature_rank].detach().cpu()
                else:
                    candidate_scores = rank_selection.selected_scores
                rank_signal_stats = _build_vector_stats(
                    candidate_scores,
                    epsilon=1e-12,
                    top_k=8,
                )
                if use_compact_feature_row_store:
                    assert feature_row_store is not None
                    normalization_input_stats = _build_phase4_normalization_stats(
                        (
                            feature_row_store.row_abs_max[:st].detach().cpu(),
                            feature_row_store.row_l1_scaled[:st].detach().cpu(),
                        ),
                    )
                else:
                    normalization_input_stats = _build_phase4_normalization_stats(
                        edge_matrix[:st, :logit_offset].abs().sum(dim=1).detach().cpu(),
                    )
            feature_row_store_snapshot_after = (
                feature_row_store.get_diagnostic_snapshot()
                if use_compact_feature_row_store and feature_row_store is not None
                else None
            )
            feature_row_store_read_stats = (
                diff_numeric_metrics(
                    feature_row_store_snapshot_before,
                    feature_row_store_snapshot_after,
                )
                if feature_row_store_snapshot_after is not None
                else None
            )
            refresh_rank_topk_elapsed_ms = (time.perf_counter() - rank_topk_start) * 1000.0
            phase4_refresh_rank_topk_elapsed_ms_total += refresh_rank_topk_elapsed_ms

            frontier_plan_start = time.perf_counter()
            if scheduler_uses_reference_planner:
                phase4_frontier_plan = _plan_phase4_frontier_membership_preserving_v1(
                    pending_candidates,
                    max_batch_size=phase4_feature_batch_size,
                    max_batches=phase4_refresh_cycle_batches,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                    apply_locality_reorder=True,
                )
                pending = phase4_frontier_plan.selected_frontier
                queue_size = int(pending.numel())
                if phase4_scheduler_config.debug:
                    logger.info(
                        "Phase 4 scheduler plan | "
                        f"membership_hash={phase4_frontier_plan.selected_membership_hash} | "
                        f"order_hash={phase4_frontier_plan.selected_order_hash} | "
                        f"fragmentation={phase4_frontier_plan.locality_fragmentation_summary} | "
                        f"boundary_reasons={phase4_frontier_plan.boundary_reason_counts} | "
                        f"invariants={phase4_frontier_plan.invariant_summary}"
                    )
            else:
                pending = _reorder_pending_for_phase4_locality(
                    pending_candidates,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                queue_size = _compute_phase4_locality_shaped_frontier_size(
                    pending,
                    max_batch_size=phase4_feature_batch_size,
                    max_batches=phase4_refresh_cycle_batches,
                    feat_layers=feat_layers,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                pending = pending[:queue_size]

            planner_v2_candidate_window = torch.empty(0, dtype=torch.long)
            planner_v2_refresh_telemetry = _build_phase4_planner_v2_refresh_telemetry_disabled()
            if (
                phase4_scheduler_config.requested_mode == "planner_v2"
                and phase4_frontier_plan is not None
            ):
                assert unvisited_feature_rank is not None
                planner_v2_candidate_scores = feature_influences[unvisited_feature_rank]
                (
                    phase4_frontier_plan,
                    planner_v2_candidate_window,
                    planner_v2_refresh_telemetry,
                ) = _apply_phase4_planner_v2_refresh_plan(
                    reference_plan=phase4_frontier_plan,
                    unvisited_feature_rank=unvisited_feature_rank,
                    candidate_scores=planner_v2_candidate_scores,
                    visited=visited,
                    max_batch_size=phase4_feature_batch_size,
                    max_batches=phase4_refresh_cycle_batches,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                pending = phase4_frontier_plan.selected_frontier
                queue_size = int(pending.numel())
                if phase4_scheduler_config.debug:
                    logger.info(
                        "Phase 4 planner_v2 refresh | "
                        f"reference_frontier_size={planner_v2_refresh_telemetry.get('scheduler_planner_v2_reference_frontier_size')} | "
                        f"candidate_window_size={planner_v2_refresh_telemetry.get('scheduler_planner_v2_candidate_window_size')} | "
                        f"changed_membership={planner_v2_refresh_telemetry.get('scheduler_planner_v2_selection_changed_membership')} | "
                        f"fallback={planner_v2_refresh_telemetry.get('scheduler_planner_v2_fallback_to_reference')} | "
                        f"fallback_reason={planner_v2_refresh_telemetry.get('scheduler_planner_v2_fallback_reason')}"
                    )
            phase4_plan_telemetry = _build_phase4_scheduler_plan_telemetry(
                phase4_frontier_plan=phase4_frontier_plan,
                telemetry_detail=phase4_scheduler_config.telemetry_detail,
            )
            refresh_frontier_plan_elapsed_ms = (time.perf_counter() - frontier_plan_start) * 1000.0
            phase4_refresh_frontier_plan_elapsed_ms_total += refresh_frontier_plan_elapsed_ms
            refresh_substage_telemetry = _build_phase4_refresh_substage_telemetry(
                telemetry_detail=phase4_scheduler_config.telemetry_detail,
                partial_influence_elapsed_ms=refresh_partial_influence_elapsed_ms,
                rank_topk_elapsed_ms=refresh_rank_topk_elapsed_ms,
                frontier_plan_elapsed_ms=refresh_frontier_plan_elapsed_ms,
                row_store_read_elapsed_ms=refresh_row_store_read_elapsed_ms,
                influence_normalization_elapsed_ms=refresh_influence_normalization_elapsed_ms,
                influence_matmul_elapsed_ms=refresh_influence_matmul_elapsed_ms,
                chunk_request_count=refresh_chunk_request_count,
                active_row_chunk_count=refresh_active_row_chunk_count,
                row_reader_row_count=refresh_rows_touched,
                solver_iteration_count=refresh_solver_iteration_count,
                row_chunk_strategy=refresh_row_chunk_strategy,
                row_weight_nonzero_row_count=refresh_row_weight_nonzero_rows,
                row_weight_zero_row_count=refresh_row_weight_zero_rows,
                row_reader_overread_zero_row_count=refresh_row_reader_overread_zero_rows,
                active_row_range_count=refresh_active_row_range_count,
                streaming_chunk_reuse_stats=streaming_chunk_reuse_stats,
                feature_row_store_read_stats=feature_row_store_read_stats,
            )
            refresh_memory_after = get_memory_snapshot(model.device)
            refresh_elapsed_ms = (time.perf_counter() - refresh_start) * 1000.0
            phase4_refresh_elapsed_ms_total += refresh_elapsed_ms
            telemetry_observer.batch(
                name="phase4.refresh",
                phase="phase4",
                batch_index=phase4_refresh_count + 1,
                elapsed_ms=refresh_elapsed_ms,
                attrs={
                    "refresh_index": refresh_index,
                    "stored_rows": int(st),
                    "visited_features": int(n_visited),
                    "frontier_candidate_count": int(rank_selection.candidate_count),
                    "queue_size": int(queue_size),
                    "phase4_frontier_buffer_extra_count": int(
                        0
                        if phase4_frontier_buffer_event is None
                        else phase4_frontier_buffer_event.get("extra_feature_count", 0)
                    ),
                    "phase4_frontier_buffer_extra_used_total": int(
                        phase4_frontier_buffer_extra_used_total
                    ),
                    "phase4_frontier_buffer_expanded_frontier_size": int(max_frontier_size),
                    "pending_count": int(pending.numel()),
                    "pending_hash": _hash_index_tensor(pending) if pending.numel() > 0 else None,
                    **phase4_execution_metadata,
                    **ranker_refresh_telemetry,
                    **planner_v2_refresh_telemetry,
                    **phase4_plan_telemetry,
                    **refresh_substage_telemetry,
                    "rank_nonzero_count": (
                        int(rank_signal_stats["nonzero_count"])
                        if rank_signal_stats is not None
                        else None
                    ),
                    "rank_effective_nonzero_count": (
                        int(rank_signal_stats["effective_nonzero_count"])
                        if rank_signal_stats is not None
                        else None
                    ),
                    "rank_max": (
                        _safe_float(rank_signal_stats.get("max"))
                        if rank_signal_stats is not None
                        else None
                    ),
                    "rank_abs_sum": (
                        _safe_float(rank_signal_stats.get("abs_sum"))
                        if rank_signal_stats is not None
                        else None
                    ),
                    "rank_all_zero": (
                        bool(rank_signal_stats["all_zero"])
                        if rank_signal_stats is not None
                        else None
                    ),
                    "rank_effectively_all_zero": (
                        bool(rank_signal_stats["effectively_all_zero"])
                        if rank_signal_stats is not None
                        else None
                    ),
                    "normalization_clamped_row_count": (
                        int(normalization_input_stats["clamped_row_count"])
                        if normalization_input_stats is not None
                        else None
                    ),
                    "normalization_clamped_row_fraction": (
                        _safe_float(normalization_input_stats.get("clamped_row_fraction"))
                        if normalization_input_stats is not None
                        else None
                    ),
                    "feature_row_store_read_calls": _safe_float(
                        (feature_row_store_read_stats or {}).get("read_call_count")
                    ),
                    "feature_row_store_read_rows": _safe_float(
                        (feature_row_store_read_stats or {}).get("read_row_count")
                    ),
                    "feature_row_store_read_bytes": (
                        int(
                            float((feature_row_store_read_stats or {}).get("read_row_count") or 0)
                            * int(total_active_feats)
                            * torch.empty(
                                (), dtype=exact_trace_internal_dtype_resolved
                            ).element_size()
                        )
                        if use_compact_feature_row_store
                        else None
                    ),
                    "feature_row_store_read_cache_hits": _safe_float(
                        (feature_row_store_read_stats or {}).get("read_cache_hit_count")
                    ),
                    "feature_row_store_read_cache_misses": _safe_float(
                        (feature_row_store_read_stats or {}).get("read_cache_miss_count")
                    ),
                    "feature_row_store_read_cache_store_success": _safe_float(
                        (feature_row_store_read_stats or {}).get("read_cache_store_success_count")
                    ),
                    "feature_row_store_read_cache_store_skip_disabled": _safe_float(
                        (feature_row_store_read_stats or {}).get(
                            "read_cache_store_skip_disabled_count"
                        )
                    ),
                    "feature_row_store_read_cache_store_skip_too_large": _safe_float(
                        (feature_row_store_read_stats or {}).get(
                            "read_cache_store_skip_too_large_count"
                        )
                    ),
                    "streaming_chunk_cache_requests": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_request_count")
                    ),
                    "streaming_chunk_cache_enabled": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_cache_enabled")
                    ),
                    "streaming_chunk_cache_max_bytes": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_cache_max_bytes")
                    ),
                    "streaming_chunk_cache_hits": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_cache_hit_count")
                    ),
                    "streaming_chunk_cache_misses": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_cache_miss_count")
                    ),
                    "streaming_row_reader_calls": _safe_int(
                        (streaming_chunk_reuse_stats or {}).get("row_reader_call_count")
                    ),
                    "streaming_row_reader_rows": _safe_int(
                        (streaming_chunk_reuse_stats or {}).get("row_reader_row_count")
                    ),
                    "streaming_row_reader_estimated_bytes": (
                        int(
                            float(
                                (streaming_chunk_reuse_stats or {}).get("row_reader_row_count") or 0
                            )
                            * int(total_active_feats)
                            * torch.empty(
                                (), dtype=exact_trace_internal_dtype_resolved
                            ).element_size()
                        )
                        if streaming_chunk_reuse_stats is not None
                        else None
                    ),
                    "streaming_chunk_cache_store_success": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get("chunk_cache_store_success_count")
                    ),
                    "streaming_chunk_cache_store_skip_disabled": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get(
                            "chunk_cache_store_skip_disabled_count"
                        )
                    ),
                    "streaming_chunk_cache_store_skip_too_large": _safe_float(
                        (streaming_chunk_reuse_stats or {}).get(
                            "chunk_cache_store_skip_too_large_count"
                        )
                    ),
                    "feature_row_store_materialize_calls": _safe_float(
                        (feature_row_store_read_stats or {}).get("materialize_call_count")
                    ),
                    "feature_row_store_materialize_rows": _safe_float(
                        (feature_row_store_read_stats or {}).get("materialize_row_count")
                    ),
                    "feature_row_store_materialize_columns": _safe_float(
                        (feature_row_store_read_stats or {}).get("materialize_column_count")
                    ),
                    **build_memory_before_after_attrs(
                        before=refresh_memory_before,
                        after=refresh_memory_after,
                        keys=_PHASE4_REFRESH_MEMORY_ATTR_KEYS,
                    ),
                },
                wall_clock=True,
            )
            if cross_cluster_debug_batches is not None:
                assert rank_signal_stats is not None
                assert normalization_input_stats is not None
                _record_cross_cluster_batch_event(
                    cross_cluster_debug_batches=cross_cluster_debug_batches,
                    event_name="phase4.refresh",
                    phase="phase4",
                    event_index=phase4_refresh_count + 1,
                    payload={
                        "refresh_index": refresh_index,
                        "stored_rows": int(st),
                        "visited_features": int(n_visited),
                        "frontier_candidate_count": int(rank_selection.candidate_count),
                        "queue_size": int(queue_size),
                        "pending_count": int(pending.numel()),
                        "pending_hash": (
                            _hash_index_tensor(pending) if pending.numel() > 0 else None
                        ),
                        "pending_sample": [
                            int(value) for value in pending.detach().cpu()[:16].tolist()
                        ],
                        "pending_full": (
                            [int(value) for value in pending.detach().cpu().tolist()]
                            if phase4_scheduler_config.telemetry_detail == "debug"
                            else None
                        ),
                        "planner_v2_candidate_window_size": int(
                            planner_v2_candidate_window.numel()
                        ),
                        "planner_v2_candidate_window_hash": (
                            _hash_index_tensor(planner_v2_candidate_window)
                            if planner_v2_candidate_window.numel() > 0
                            else None
                        ),
                        **phase4_execution_metadata,
                        **ranker_refresh_telemetry,
                        **planner_v2_refresh_telemetry,
                        **phase4_plan_telemetry,
                        **refresh_substage_telemetry,
                        "rank_nonzero_count": int(rank_signal_stats["nonzero_count"]),
                        "rank_effective_nonzero_count": int(
                            rank_signal_stats["effective_nonzero_count"]
                        ),
                        "rank_nonfinite_count": int(rank_signal_stats["nonfinite_count"]),
                        "rank_max": _safe_float(rank_signal_stats.get("max")),
                        "rank_abs_sum": _safe_float(rank_signal_stats.get("abs_sum")),
                        "rank_effectively_all_zero": bool(
                            rank_signal_stats["effectively_all_zero"]
                        ),
                        "normalization_clamped_row_count": int(
                            normalization_input_stats["clamped_row_count"]
                        ),
                        "normalization_clamped_row_fraction": _safe_float(
                            normalization_input_stats.get("clamped_row_fraction")
                        ),
                        "feature_row_store_read_calls": _safe_float(
                            (feature_row_store_read_stats or {}).get("read_call_count")
                        ),
                        "feature_row_store_read_rows": _safe_float(
                            (feature_row_store_read_stats or {}).get("read_row_count")
                        ),
                        "refresh_elapsed_ms": float(refresh_elapsed_ms),
                        **get_memory_snapshot(model.device),
                    },
                )
            if anomaly_debug_result is not None:
                assert candidate_scores is not None
                assert phase4_logit_probability_stats is not None
                _record_phase4_refresh_debug(
                    anomaly_debug_result,
                    refresh_index=refresh_index,
                    n_visited=n_visited,
                    queue_size=queue_size,
                    pending=pending,
                    previous_pending=previous_phase4_pending,
                    first_pending=first_phase4_pending,
                    candidate_scores=candidate_scores,
                    refresh_elapsed_ms=refresh_elapsed_ms,
                    rank_signal_stats=rank_signal_stats,
                    logit_probability_stats=phase4_logit_probability_stats,
                    normalization_input_stats=normalization_input_stats,
                    feature_row_store_read_stats=feature_row_store_read_stats,
                    streaming_chunk_reuse_stats=streaming_chunk_reuse_stats,
                )
                debug_records = anomaly_debug_result.get("records", [])
                assert isinstance(debug_records, list) and debug_records
                current_debug_record = debug_records[-1]
                assert isinstance(current_debug_record, dict)
                assert unvisited_feature_rank is not None
                deterministic_pending = _build_phase4_deterministic_shadow_pending(
                    unvisited_feature_rank,
                    feature_influences.detach().cpu(),
                    queue_size=queue_size,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                current_debug_record["deterministic_shadow"] = _compare_phase4_frontiers(
                    pending,
                    deterministic_pending,
                )
                if phase4_refresh_count == 0:
                    if use_compact_feature_row_store:
                        assert feature_row_store is not None
                        shadow_row_denominator = (
                            feature_row_store.row_abs_max[:st].to(dtype=shadow_debug_compute_dtype),
                            feature_row_store.row_l1_scaled[:st].to(
                                dtype=shadow_debug_compute_dtype
                            ),
                        )
                        float64_feature_influences = compute_partial_feature_influences_streaming(
                            lambda row_start, row_end: feature_row_store.read_feature_rows(
                                row_start,
                                row_end,
                                phase="phase4_anomaly_debug",
                            ),
                            shadow_row_denominator,
                            phase4_logit_probabilities.to(dtype=shadow_debug_compute_dtype),
                            row_to_node_index[:st],
                            n_feature_nodes=total_active_feats,
                            n_logits=n_logits,
                            device=torch.device("cpu"),
                            compute_dtype=shadow_debug_compute_dtype,
                        )
                    else:
                        float64_influences = compute_partial_influences(
                            edge_matrix[:st].to(dtype=shadow_debug_compute_dtype),
                            phase4_logit_probabilities.to(dtype=shadow_debug_compute_dtype),
                            row_to_node_index[:st],
                            device=torch.device("cpu"),
                        )
                        float64_feature_influences = float64_influences[:total_active_feats]
                    if exact_trace_internal_dtype_resolved == torch.float32:
                        float32_feature_influences = feature_influences
                    elif use_compact_feature_row_store:
                        assert feature_row_store is not None
                        float32_row_denominator = (
                            feature_row_store.row_abs_max[:st].to(dtype=torch.float32),
                            feature_row_store.row_l1_scaled[:st].to(dtype=torch.float32),
                        )
                        float32_feature_influences = compute_partial_feature_influences_streaming(
                            lambda row_start, row_end: feature_row_store.read_feature_rows(
                                row_start,
                                row_end,
                                phase="phase4_anomaly_debug",
                            ),
                            float32_row_denominator,
                            phase4_logit_probabilities.to(dtype=torch.float32),
                            row_to_node_index[:st],
                            n_feature_nodes=total_active_feats,
                            n_logits=n_logits,
                            device=torch.device("cpu"),
                        )
                    else:
                        float32_influences = compute_partial_influences(
                            edge_matrix[:st].to(dtype=torch.float32),
                            phase4_logit_probabilities.to(dtype=torch.float32),
                            row_to_node_index[:st],
                            device=torch.device("cpu"),
                        )
                        float32_feature_influences = float32_influences[:total_active_feats]
                    float32_signal_stats = _build_vector_stats(
                        float32_feature_influences.detach().cpu(),
                        epsilon=1e-12,
                        top_k=8,
                    )
                    float64_signal_stats = _build_vector_stats(
                        float64_feature_influences.detach().cpu(),
                        epsilon=1e-12,
                        top_k=8,
                    )
                    float64_feature_rank = torch.argsort(
                        float64_feature_influences,
                        descending=True,
                    ).cpu()
                    float64_pending = float64_feature_rank[~visited[float64_feature_rank]][
                        :queue_size
                    ]
                    float64_pending = _reorder_pending_for_phase4_locality(
                        float64_pending,
                        feat_layers=feat_layers,
                        feat_positions=feat_pos,
                        feat_ids=feat_ids,
                        exact_chunked_decoder=exact_chunked_decoder,
                        decoder_chunk_size=decoder_chunk_size,
                    )
                    current_debug_record["float64_shadow"] = _compare_phase4_frontiers(
                        pending,
                        float64_pending,
                    )
                    current_debug_record["float_precision_signal_compare"] = {
                        "float32": float32_signal_stats,
                        "float64": float64_signal_stats,
                        "float32_all_zero": bool(float32_signal_stats["all_zero"]),
                        "float64_all_zero": bool(float64_signal_stats["all_zero"]),
                        "float32_effectively_all_zero": bool(
                            float32_signal_stats["effectively_all_zero"]
                        ),
                        "float64_effectively_all_zero": bool(
                            float64_signal_stats["effectively_all_zero"]
                        ),
                    }
                current_pending_cpu = pending.detach().to(device="cpu", dtype=torch.int64)
                if first_phase4_pending is None:
                    first_phase4_pending = current_pending_cpu.clone()
                previous_phase4_pending = current_pending_cpu
            phase4_refresh_count += 1

        pending_offset = 0
        planned_boundaries = (
            phase4_frontier_plan.batch_boundaries
            if scheduler_uses_reference_planner and phase4_frontier_plan is not None
            else None
        )
        planned_boundary_offset = 0
        while pending_offset < len(pending):
            if planned_boundaries is not None:
                if planned_boundary_offset >= len(planned_boundaries):
                    raise RuntimeError(
                        "Planner v1 exhausted planned boundaries before pending frontier completion"
                    )
                boundary_start, batch_end = planned_boundaries[planned_boundary_offset]
                if boundary_start != pending_offset:
                    raise RuntimeError(
                        "Planner v1 planned boundary start mismatch "
                        f"(expected={pending_offset}, got={boundary_start})"
                    )
                planned_boundary_offset += 1
            else:
                batch_end = _compute_phase4_locality_shaped_batch_end(
                    pending,
                    pending_offset=pending_offset,
                    max_batch_size=phase4_feature_batch_size,
                    feat_layers=feat_layers,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
            if batch_end <= pending_offset:
                raise RuntimeError(
                    "Phase 4 scheduling produced a non-advancing batch boundary "
                    f"(offset={pending_offset}, batch_end={batch_end})"
                )
            reference_pending_start = pending_offset
            reference_pending_end = batch_end
            reference_idx_batch = pending[reference_pending_start:reference_pending_end]
            pending_offset = batch_end
            scheduler_reference_batch_index = int(phase4_scheduler_reference_batch_count)
            phase4_scheduler_reference_batch_count += 1

            if phase4_executor_microbatch_size < int(reference_idx_batch.numel()):
                executor_batches: list[torch.Tensor] = []
                streaming_pending_offset = 0
                while streaming_pending_offset < int(reference_idx_batch.numel()):
                    streaming_end = min(
                        streaming_pending_offset + phase4_executor_microbatch_size,
                        int(reference_idx_batch.numel()),
                    )
                    executor_batches.append(
                        reference_idx_batch[streaming_pending_offset:streaming_end]
                    )
                    streaming_pending_offset = streaming_end
            else:
                executor_batches = [reference_idx_batch]

            streaming_chunk_count = int(len(executor_batches))
            chunk_pending_start = reference_pending_start
            for streaming_chunk_index, idx_batch in enumerate(executor_batches, start=1):
                chunk_pending_end = chunk_pending_start + int(idx_batch.numel())
                n_visited += len(idx_batch)
                phase4_executor_microbatch_count += 1
                executor_microbatch_index = int(phase4_executor_microbatch_count)

                ctx_before = _snapshot_diagnostics(ctx) if profile else None
                transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
                batch_start = time.perf_counter()
                batch_memory_before = get_memory_snapshot(model.device)
                encoder_vectors_source_device = None
                encoder_vectors_source_dtype = None
                if getattr(ctx, "encoder_vecs", None) is not None and ctx.encoder_vecs.numel() > 0:
                    encoder_vectors_source_device = str(ctx.encoder_vecs.device.type)
                    encoder_vectors_source_dtype = ctx.encoder_vecs.dtype
                encoder_materialize_start = time.perf_counter()
                encoder_vectors = ctx.materialize_encoder_vectors(idx_batch)
                executor_encoder_materialize_elapsed_ms = (
                    time.perf_counter() - encoder_materialize_start
                ) * 1000.0
                encoder_vectors_transfer_bytes = (
                    _tensor_nbytes_estimate(encoder_vectors)
                    if encoder_vectors_source_device is not None
                    and (
                        encoder_vectors_source_device != encoder_vectors.device.type
                        or encoder_vectors_source_dtype != encoder_vectors.dtype
                    )
                    else 0
                )
                encoder_vectors_transfer_telemetry = {
                    "encoder_vectors_source": encoder_vectors_source_device,
                    "encoder_vectors_destination": str(encoder_vectors.device.type),
                    "encoder_vectors_dtype_source": str(encoder_vectors_source_dtype)
                    if encoder_vectors_source_dtype is not None
                    else None,
                    "encoder_vectors_dtype_destination": str(encoder_vectors.dtype),
                    "encoder_vectors_bytes": int(_tensor_nbytes_estimate(encoder_vectors)),
                    "encoder_vectors_transfer_bytes": int(encoder_vectors_transfer_bytes),
                    "encoder_vectors_materialize_elapsed_ms": float(
                        executor_encoder_materialize_elapsed_ms
                    ),
                }
                if encoder_vectors_source_device == "cpu" and encoder_vectors.device.type == "cuda":
                    phase4_cpu_to_gpu_bytes_total += int(encoder_vectors_transfer_bytes)
                compute_batch_start = time.perf_counter()
                tiled_production = config.full_retention_backend == "column_tiled_v1"
                if tiled_production:
                    assert feature_row_store is not None and nonfeature_row_store is not None
                    rows, tiled_denominator = produce_and_store_tiled_rows(
                        ctx=ctx,
                        layers=feat_layers[idx_batch],
                        positions=feat_pos[idx_batch],
                        inject_values=encoder_vectors,
                        row_start=st,
                        feature_row_store=feature_row_store,
                        nonfeature_row_store=nonfeature_row_store,
                        feature_column_tile_size=config.feature_row_column_tile_size,
                        dtype=exact_trace_internal_dtype_resolved,
                        phase_label="phase4_features",
                        retain_graph=n_visited < actual_max_feature_nodes,
                    )
                else:
                    rows = ctx.compute_batch(
                        layers=feat_layers[idx_batch],
                        positions=feat_pos[idx_batch],
                        inject_values=encoder_vectors,
                        retain_graph=n_visited < actual_max_feature_nodes,
                        phase_label="phase4_features",
                    )
                executor_compute_batch_elapsed_ms = (
                    time.perf_counter() - compute_batch_start
                ) * 1000.0

                row_count = rows.shape[0]
                end = st + row_count
                if tiled_production:
                    cpu_staging_start = time.perf_counter()
                    rows_cpu = rows
                    row_input_slice = rows_cpu
                    feature_row_slice = torch.empty((row_count, 0), dtype=rows_cpu.dtype)
                    nonfeature_row_slice = rows_cpu
                    executor_cpu_staging_elapsed_ms = 0.0
                    denominator_start = time.perf_counter()
                    row_denominator_scaled_l1 = tiled_denominator
                    executor_denominator_elapsed_ms = 0.0
                    executor_row_transfer_telemetry = _build_row_transfer_telemetry(
                        rows=rows,
                        rows_cpu=rows_cpu,
                        row_input_slice=row_input_slice,
                        feature_row_slice=feature_row_slice,
                    )
                elif phase4_row_reduction_config.effective_mode == "gpu_v1":
                    if not use_compact_feature_row_store:
                        raise RuntimeError(
                            "phase4_row_reduction='gpu_v1' requires compact Phase-4 row store"
                        )
                    cpu_staging_start = time.perf_counter()
                    feature_row_slice, feature_rows_cpu_staging = _copy_feature_rows_to_cpu_staging(
                        rows,
                        total_active_feats=total_active_feats,
                        staging_buffer=feature_rows_cpu_staging,
                    )
                    executor_cpu_staging_elapsed_ms = (
                        time.perf_counter() - cpu_staging_start
                    ) * 1000.0
                    row_input_slice = rows[:, :logit_offset]
                    denominator_start = time.perf_counter()
                    row_abs_max_gpu, row_l1_scaled_gpu = _compute_row_denominator_scaled_l1(
                        row_input_slice,
                        dtype=exact_trace_internal_dtype_resolved,
                        preserve_device=True,
                    )
                    executor_denominator_elapsed_ms = (
                        time.perf_counter() - denominator_start
                    ) * 1000.0
                    executor_row_transfer_telemetry = (
                        _build_phase4_gpu_row_reduction_transfer_telemetry(
                            rows=rows,
                            feature_row_slice=feature_row_slice,
                            row_abs_max=row_abs_max_gpu,
                            row_l1_scaled=row_l1_scaled_gpu,
                        )
                    )
                    row_denominator_scaled_l1 = (row_abs_max_gpu, row_l1_scaled_gpu)
                    nonfeature_row_slice = rows[:, total_active_feats:logit_offset]
                else:
                    cpu_staging_start = time.perf_counter()
                    rows_cpu, rows_cpu_staging = _copy_rows_to_cpu_staging(
                        rows,
                        staging_buffer=rows_cpu_staging,
                    )
                    executor_cpu_staging_elapsed_ms = (
                        time.perf_counter() - cpu_staging_start
                    ) * 1000.0
                    row_input_slice = rows_cpu[:, :logit_offset]
                    feature_row_slice = rows_cpu[:, :total_active_feats]
                    nonfeature_row_slice = rows_cpu[:, total_active_feats:logit_offset]
                    executor_row_transfer_telemetry = _build_row_transfer_telemetry(
                        rows=rows,
                        rows_cpu=rows_cpu,
                        row_input_slice=row_input_slice,
                        feature_row_slice=feature_row_slice,
                    )
                    denominator_start = time.perf_counter()
                    row_abs_max_cpu, row_l1_scaled_cpu = _compute_row_denominator_scaled_l1(
                        row_input_slice,
                        dtype=exact_trace_internal_dtype_resolved,
                    )
                    row_denominator_scaled_l1 = (row_abs_max_cpu, row_l1_scaled_cpu)
                    executor_denominator_elapsed_ms = (
                        time.perf_counter() - denominator_start
                    ) * 1000.0
                if executor_row_transfer_telemetry["row_transfer_source"] == "cuda":
                    phase4_gpu_to_cpu_bytes_total += int(
                        executor_row_transfer_telemetry["row_transfer_bytes"]
                    )
                phase4_row_reduction_gpu_to_cpu_bytes_saved_total += int(
                    executor_row_transfer_telemetry.get(
                        "row_reduction_gpu_to_cpu_bytes_saved",
                        0,
                    )
                )
                if executor_row_transfer_telemetry["row_transfer_destination"] == "cuda":
                    phase4_cpu_to_gpu_bytes_total += int(
                        executor_row_transfer_telemetry["row_transfer_bytes"]
                    )
                if int(executor_row_transfer_telemetry["row_transfer_bytes"]) > 0:
                    phase4_copy_count += 1
                if anomaly_debug_result is not None and phase4_executor_microbatch_count <= 2:
                    feature_row_batches = anomaly_debug_result.setdefault(
                        "phase4_feature_row_batches",
                        [],
                    )
                    assert isinstance(feature_row_batches, list)
                    feature_row_batches.append(
                        {
                            "batch_index": int(executor_microbatch_index),
                            "batch_row_count": int(row_count),
                            "row_input_stats": _build_matrix_abs_stats(
                                row_input_slice,
                                epsilon=1e-12,
                                top_k=8,
                            ),
                            "row_abs_sum_stats": _build_phase4_normalization_stats(
                                row_denominator_scaled_l1,
                                clamp_epsilon=1e-8,
                            ),
                        }
                    )
                if use_compact_feature_row_store and not tiled_production:
                    assert feature_row_store is not None
                    assert nonfeature_row_store is not None
                    row_store_write_start = time.perf_counter()
                    row_store_append_telemetry = feature_row_store.append_rows(
                        row_start=st,
                        feature_rows=feature_row_slice,
                        row_denominator_scaled_l1=row_denominator_scaled_l1,
                        phase="phase4",
                    )
                    nonfeature_row_store.append_rows(
                        row_start=st,
                        feature_rows=nonfeature_row_slice,
                        row_denominator_scaled_l1=row_denominator_scaled_l1,
                        phase="phase4",
                    )
                    executor_row_store_write_elapsed_ms = (
                        time.perf_counter() - row_store_write_start
                    ) * 1000.0
                elif not use_compact_feature_row_store:
                    assert phase4_row_reduction_config.effective_mode == "off"
                    row_store_write_start = time.perf_counter()
                    edge_matrix[st:end, :logit_offset] = rows_cpu
                    executor_row_store_write_elapsed_ms = (
                        time.perf_counter() - row_store_write_start
                    ) * 1000.0
                    row_store_append_telemetry = None
                else:
                    executor_row_store_write_elapsed_ms = 0.0
                    row_store_append_telemetry = {}
                row_to_node_index[st:end] = idx_batch
                visited[idx_batch] = True
                st = end
                pbar.update(len(idx_batch))

                if profile:
                    batch_number = executor_microbatch_index
                    if batch_number % profile_log_interval == 0:
                        batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
                        _log_batch_profile(
                            logger,
                            "Phase 4",
                            batch_number,
                            None,
                            batch_elapsed_ms / 1000.0,
                            ctx_before,
                            _snapshot_diagnostics(ctx),
                            transcoder_before,
                            _snapshot_diagnostics(model.transcoders),
                        )
                batch_number = executor_microbatch_index
                batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
                batch_memory_after = get_memory_snapshot(model.device)
                phase4_feature_batch_elapsed_ms_total += batch_elapsed_ms
                phase4_executor_encoder_materialize_elapsed_ms_total += (
                    executor_encoder_materialize_elapsed_ms
                )
                phase4_executor_compute_batch_elapsed_ms_total += executor_compute_batch_elapsed_ms
                phase4_executor_cpu_staging_elapsed_ms_total += executor_cpu_staging_elapsed_ms
                phase4_executor_denominator_elapsed_ms_total += executor_denominator_elapsed_ms
                phase4_executor_row_store_write_elapsed_ms_total += (
                    executor_row_store_write_elapsed_ms
                )
                executor_batch_telemetry = _build_phase4_executor_batch_telemetry(
                    scheduler_reference_batch_index=scheduler_reference_batch_index,
                    scheduler_reference_batch_count=phase4_scheduler_reference_batch_count,
                    scheduler_reference_batch_rows=int(reference_idx_batch.numel()),
                    executor_microbatch_index=executor_microbatch_index,
                    executor_microbatch_count=phase4_executor_microbatch_count,
                    executor_configured_reference_batch_size=phase4_executor_reference_batch_size,
                    executor_microbatch_rows=int(idx_batch.numel()),
                    executor_microbatch_size=phase4_executor_microbatch_size,
                )
                executor_substage_telemetry = _build_phase4_executor_substage_telemetry(
                    telemetry_detail=phase4_scheduler_config.telemetry_detail,
                    encoder_materialize_elapsed_ms=executor_encoder_materialize_elapsed_ms,
                    compute_batch_elapsed_ms=executor_compute_batch_elapsed_ms,
                    cpu_staging_elapsed_ms=executor_cpu_staging_elapsed_ms,
                    denominator_elapsed_ms=executor_denominator_elapsed_ms,
                    row_store_write_elapsed_ms=executor_row_store_write_elapsed_ms,
                    batch_elapsed_ms=batch_elapsed_ms,
                )
                if (
                    row_store_append_telemetry is not None
                    and phase4_scheduler_config.telemetry_detail in {"normal", "debug"}
                ):
                    executor_substage_telemetry.update(row_store_append_telemetry)
                executor_streaming_telemetry = {
                    "executor_reference_batch_size": int(reference_idx_batch.numel()),
                    "executor_microbatch_size": int(phase4_executor_microbatch_size),
                    "executor_streaming_chunk_index": int(streaming_chunk_index)
                    if executor_physically_split
                    else None,
                    "executor_streaming_chunk_count": int(streaming_chunk_count)
                    if executor_physically_split
                    else None,
                    "executor_physically_split": bool(executor_physically_split),
                    "scheduler_pending_start_index": int(chunk_pending_start),
                    "scheduler_pending_end_index": int(chunk_pending_end),
                    "scheduler_reference_pending_start_index": int(reference_pending_start),
                    "scheduler_reference_pending_end_index": int(reference_pending_end),
                }
                batch_locality_summary = _build_phase4_batch_locality_summary(
                    idx_batch,
                    feat_layers=feat_layers,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                telemetry_observer.batch(
                    name="phase4.feature_batch",
                    phase="phase4",
                    batch_index=batch_number,
                    elapsed_ms=batch_elapsed_ms,
                    attrs={
                        "batch_rows": int(row_count),
                        "visited_features": int(n_visited),
                        "target_feature_count": int(actual_max_feature_nodes),
                        **phase4_execution_metadata,
                        **executor_batch_telemetry,
                        "scheduler_refresh_index": pending_refresh_index,
                        **executor_streaming_telemetry,
                        **batch_locality_summary,
                        **executor_substage_telemetry,
                        **encoder_vectors_transfer_telemetry,
                        **executor_row_transfer_telemetry,
                        **build_memory_before_after_attrs(
                            before=batch_memory_before,
                            after=batch_memory_after,
                            keys=_PHASE4_REFRESH_MEMORY_ATTR_KEYS,
                        ),
                    },
                    wall_clock=True,
                )
                if cross_cluster_debug_batches is not None:
                    row_input_stats = _build_matrix_abs_stats(
                        row_input_slice,
                        epsilon=1e-12,
                        top_k=0,
                    )
                    row_abs_sum_stats = _build_phase4_normalization_stats(
                        row_denominator_scaled_l1,
                        clamp_epsilon=1e-8,
                    )
                    _record_cross_cluster_batch_event(
                        cross_cluster_debug_batches=cross_cluster_debug_batches,
                        event_name="phase4.feature_batch",
                        phase="phase4",
                        event_index=batch_number,
                        payload={
                            "batch_rows": int(row_count),
                            "visited_features": int(n_visited),
                            "target_feature_count": int(actual_max_feature_nodes),
                            **phase4_execution_metadata,
                            **executor_batch_telemetry,
                            "scheduler_refresh_index": pending_refresh_index,
                            **executor_streaming_telemetry,
                            **batch_locality_summary,
                            **executor_substage_telemetry,
                            "idx_batch_hash": batch_locality_summary.get("scheduler_batch_hash"),
                            "row_input_nonfinite_count": int(row_input_stats["nonfinite_count"]),
                            "row_input_finite_max_abs": _safe_float(
                                row_input_stats.get("finite_max_abs")
                            ),
                            "row_l1_abs_sum": _safe_float(row_abs_sum_stats.get("abs_sum")),
                            "row_l1_max": _safe_float(row_abs_sum_stats.get("max")),
                            "row_l1_nonfinite_count": int(row_abs_sum_stats["nonfinite_count"]),
                            "row_l1_effectively_all_zero": bool(
                                row_abs_sum_stats["effectively_all_zero"]
                            ),
                            "batch_elapsed_ms": float(batch_elapsed_ms),
                            **get_memory_snapshot(model.device),
                        },
                    )
                chunk_pending_start = chunk_pending_end
        if planned_boundaries is not None and planned_boundary_offset != len(planned_boundaries):
            raise RuntimeError(
                "Planner v1 produced unused planned boundaries "
                f"(used={planned_boundary_offset}, planned={len(planned_boundaries)})"
            )

    pbar.close()
    _log_phase_metrics(
        logger,
        "Feature attributions",
        phase4_start,
        model.device,
        selected_features=int(visited.sum().item()),
        final_feature_batch_size=phase4_feature_batch_size,
        phase4_batches=phase4_scheduler_reference_batch_count,
        phase4_executor_microbatch_count=phase4_executor_microbatch_count,
    )
    phase4_elapsed_ms = (time.perf_counter() - phase4_start) * 1000.0
    telemetry_observer.phase(
        name="phase4.feature_attribution",
        phase="phase4",
        elapsed_ms=phase4_elapsed_ms,
        attrs={
            "selected_features": int(visited.sum().item()),
            "feature_batch_size": int(phase4_feature_batch_size),
            "phase4_batches": int(phase4_scheduler_reference_batch_count),
            "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
            "phase4_refreshes": int(phase4_refresh_count),
            "phase4_refresh_elapsed_ms_total": float(phase4_refresh_elapsed_ms_total),
            "phase4_feature_batch_elapsed_ms_total": float(phase4_feature_batch_elapsed_ms_total),
            "phase4_refresh_partial_influence_elapsed_ms_total": float(
                phase4_refresh_partial_influence_elapsed_ms_total
            ),
            "phase4_refresh_rank_topk_elapsed_ms_total": float(
                phase4_refresh_rank_topk_elapsed_ms_total
            ),
            "phase4_refresh_frontier_plan_elapsed_ms_total": float(
                phase4_refresh_frontier_plan_elapsed_ms_total
            ),
            "phase4_refresh_row_store_read_elapsed_ms_total": float(
                phase4_refresh_row_store_read_elapsed_ms_total
            ),
            "phase4_refresh_influence_normalization_elapsed_ms_total": float(
                phase4_refresh_influence_normalization_elapsed_ms_total
            ),
            "phase4_refresh_influence_matmul_elapsed_ms_total": float(
                phase4_refresh_influence_matmul_elapsed_ms_total
            ),
            "phase4_executor_encoder_materialize_elapsed_ms_total": float(
                phase4_executor_encoder_materialize_elapsed_ms_total
            ),
            "phase4_executor_compute_batch_elapsed_ms_total": float(
                phase4_executor_compute_batch_elapsed_ms_total
            ),
            "phase4_executor_cpu_staging_elapsed_ms_total": float(
                phase4_executor_cpu_staging_elapsed_ms_total
            ),
            "phase4_executor_denominator_elapsed_ms_total": float(
                phase4_executor_denominator_elapsed_ms_total
            ),
            "phase4_executor_row_store_write_elapsed_ms_total": float(
                phase4_executor_row_store_write_elapsed_ms_total
            ),
            "phase4_gpu_to_cpu_bytes_total": int(phase4_gpu_to_cpu_bytes_total),
            "phase4_row_reduction_gpu_to_cpu_bytes_saved_total": int(
                phase4_row_reduction_gpu_to_cpu_bytes_saved_total
            ),
            "phase4_cpu_to_gpu_bytes_total": int(phase4_cpu_to_gpu_bytes_total),
            "phase4_copy_count": int(phase4_copy_count),
            **phase4_execution_metadata,
            **(phase4_no_refresh_plan_telemetry or {}),
        },
        wall_clock=True,
    )
    if anomaly_debug_result is not None:
        records = anomaly_debug_result.get("records", [])
        cutoff_margins = [
            float(record["cutoff"]["cutoff_margin"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("cutoff"), dict)
            and record["cutoff"].get("cutoff_margin") is not None
        ]
        previous_overlaps = [
            float(record["overlap_with_previous"])
            for record in records
            if isinstance(record, dict) and record.get("overlap_with_previous") is not None
        ]
        first_overlaps = [
            float(record["overlap_with_first"])
            for record in records
            if isinstance(record, dict) and record.get("overlap_with_first") is not None
        ]
        deterministic_overlaps = [
            float(record["deterministic_shadow"]["overlap_fraction"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("deterministic_shadow"), dict)
            and record["deterministic_shadow"].get("overlap_fraction") is not None
        ]
        float64_overlaps = [
            float(record["float64_shadow"]["overlap_fraction"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("float64_shadow"), dict)
            and record["float64_shadow"].get("overlap_fraction") is not None
        ]
        refresh_elapsed_values = [
            float(record["refresh_elapsed_ms"])
            for record in records
            if isinstance(record, dict) and record.get("refresh_elapsed_ms") is not None
        ]
        rank_nonzero_counts = [
            int(record["rank_signal_stats"]["nonzero_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("rank_signal_stats"), dict)
            and record["rank_signal_stats"].get("nonzero_count") is not None
        ]
        rank_effective_nonzero_counts = [
            int(record["rank_signal_stats"]["effective_nonzero_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("rank_signal_stats"), dict)
            and record["rank_signal_stats"].get("effective_nonzero_count") is not None
        ]
        rank_abs_sums = [
            float(record["rank_signal_stats"]["abs_sum"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("rank_signal_stats"), dict)
            and record["rank_signal_stats"].get("abs_sum") is not None
        ]
        rank_max_values = [
            float(record["rank_signal_stats"]["max"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("rank_signal_stats"), dict)
            and record["rank_signal_stats"].get("max") is not None
        ]
        rank_all_zero_count = sum(
            1
            for record in records
            if isinstance(record, dict) and bool(record.get("rank_signal_all_zero"))
        )
        rank_effectively_all_zero_count = sum(
            1
            for record in records
            if isinstance(record, dict) and bool(record.get("rank_signal_effectively_all_zero"))
        )
        normalization_clamped_counts = [
            int(record["normalization_input_stats"]["clamped_row_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("normalization_input_stats"), dict)
            and record["normalization_input_stats"].get("clamped_row_count") is not None
        ]
        normalization_clamped_fractions = [
            float(record["normalization_input_stats"]["clamped_row_fraction"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("normalization_input_stats"), dict)
            and record["normalization_input_stats"].get("clamped_row_fraction") is not None
        ]
        feature_row_store_read_calls = [
            float(record["feature_row_store_read_stats"]["read_call_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("feature_row_store_read_stats"), dict)
            and record["feature_row_store_read_stats"].get("read_call_count") is not None
        ]
        feature_row_store_read_rows = [
            float(record["feature_row_store_read_stats"]["read_row_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("feature_row_store_read_stats"), dict)
            and record["feature_row_store_read_stats"].get("read_row_count") is not None
        ]
        feature_row_store_cache_store_success = [
            float(record["feature_row_store_read_stats"]["read_cache_store_success_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("feature_row_store_read_stats"), dict)
            and record["feature_row_store_read_stats"].get("read_cache_store_success_count")
            is not None
        ]
        feature_row_store_cache_skip_disabled = [
            float(record["feature_row_store_read_stats"]["read_cache_store_skip_disabled_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("feature_row_store_read_stats"), dict)
            and record["feature_row_store_read_stats"].get("read_cache_store_skip_disabled_count")
            is not None
        ]
        feature_row_store_cache_skip_too_large = [
            float(record["feature_row_store_read_stats"]["read_cache_store_skip_too_large_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("feature_row_store_read_stats"), dict)
            and record["feature_row_store_read_stats"].get("read_cache_store_skip_too_large_count")
            is not None
        ]
        streaming_chunk_cache_hits = [
            float(record["streaming_chunk_reuse_stats"]["chunk_cache_hit_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
            and record["streaming_chunk_reuse_stats"].get("chunk_cache_hit_count") is not None
        ]
        streaming_chunk_cache_misses = [
            float(record["streaming_chunk_reuse_stats"]["chunk_cache_miss_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
            and record["streaming_chunk_reuse_stats"].get("chunk_cache_miss_count") is not None
        ]
        streaming_chunk_cache_store_success = [
            float(record["streaming_chunk_reuse_stats"]["chunk_cache_store_success_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
            and record["streaming_chunk_reuse_stats"].get("chunk_cache_store_success_count")
            is not None
        ]
        streaming_chunk_cache_skip_disabled = [
            float(record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_disabled_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
            and record["streaming_chunk_reuse_stats"].get("chunk_cache_store_skip_disabled_count")
            is not None
        ]
        streaming_chunk_cache_skip_too_large = [
            float(record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_too_large_count"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
            and record["streaming_chunk_reuse_stats"].get("chunk_cache_store_skip_too_large_count")
            is not None
        ]
        first_float_precision = None
        if records and isinstance(records[0], dict):
            precision_compare = records[0].get("float_precision_signal_compare")
            if isinstance(precision_compare, dict):
                first_float_precision = precision_compare
        phase3_logit_row_batches = anomaly_debug_result.get("phase3_logit_row_batches", [])
        first_phase3_logit_batch = (
            phase3_logit_row_batches[0]
            if isinstance(phase3_logit_row_batches, list) and phase3_logit_row_batches
            else None
        )
        phase4_feature_row_batches = anomaly_debug_result.get("phase4_feature_row_batches", [])
        anomaly_debug_result["refresh_count"] = int(len(records))
        anomaly_debug_result["status"] = "captured_refresh_debug"
        anomaly_debug_result["summary"] = {
            "refresh_count": int(len(records)),
            "pending_size_first": (
                int(records[0]["pending_size"]) if records and isinstance(records[0], dict) else 0
            ),
            "cutoff_margin_min": min(cutoff_margins) if cutoff_margins else None,
            "cutoff_margin_mean": (
                sum(cutoff_margins) / len(cutoff_margins) if cutoff_margins else None
            ),
            "overlap_with_previous_mean": (
                sum(previous_overlaps) / len(previous_overlaps) if previous_overlaps else None
            ),
            "overlap_with_first_mean": (
                sum(first_overlaps) / len(first_overlaps) if first_overlaps else None
            ),
            "deterministic_shadow_overlap_mean": (
                sum(deterministic_overlaps) / len(deterministic_overlaps)
                if deterministic_overlaps
                else None
            ),
            "float64_shadow_overlap_mean": (
                sum(float64_overlaps) / len(float64_overlaps) if float64_overlaps else None
            ),
            "refresh_elapsed_ms_total": (
                sum(refresh_elapsed_values) if refresh_elapsed_values else None
            ),
            "refresh_elapsed_ms_mean": (
                (sum(refresh_elapsed_values) / len(refresh_elapsed_values))
                if refresh_elapsed_values
                else None
            ),
            "rank_signal_all_zero_refresh_count": int(rank_all_zero_count),
            "rank_signal_effectively_all_zero_refresh_count": int(rank_effectively_all_zero_count),
            "rank_signal_nonzero_count_min": (
                min(rank_nonzero_counts) if rank_nonzero_counts else None
            ),
            "rank_signal_nonzero_count_mean": (
                (sum(rank_nonzero_counts) / len(rank_nonzero_counts))
                if rank_nonzero_counts
                else None
            ),
            "rank_signal_effective_nonzero_count_min": (
                min(rank_effective_nonzero_counts) if rank_effective_nonzero_counts else None
            ),
            "rank_signal_effective_nonzero_count_mean": (
                (sum(rank_effective_nonzero_counts) / len(rank_effective_nonzero_counts))
                if rank_effective_nonzero_counts
                else None
            ),
            "rank_signal_abs_sum_mean": (
                (sum(rank_abs_sums) / len(rank_abs_sums)) if rank_abs_sums else None
            ),
            "rank_signal_max_max": max(rank_max_values) if rank_max_values else None,
            "normalization_clamped_row_count_max": (
                max(normalization_clamped_counts) if normalization_clamped_counts else None
            ),
            "normalization_clamped_row_fraction_mean": (
                (sum(normalization_clamped_fractions) / len(normalization_clamped_fractions))
                if normalization_clamped_fractions
                else None
            ),
            "feature_row_store_read_calls_per_refresh_mean": (
                (sum(feature_row_store_read_calls) / len(feature_row_store_read_calls))
                if feature_row_store_read_calls
                else None
            ),
            "feature_row_store_read_rows_per_refresh_mean": (
                (sum(feature_row_store_read_rows) / len(feature_row_store_read_rows))
                if feature_row_store_read_rows
                else None
            ),
            "feature_row_store_cache_store_success_per_refresh_mean": (
                (
                    sum(feature_row_store_cache_store_success)
                    / len(feature_row_store_cache_store_success)
                )
                if feature_row_store_cache_store_success
                else None
            ),
            "feature_row_store_cache_skip_disabled_per_refresh_mean": (
                (
                    sum(feature_row_store_cache_skip_disabled)
                    / len(feature_row_store_cache_skip_disabled)
                )
                if feature_row_store_cache_skip_disabled
                else None
            ),
            "feature_row_store_cache_skip_too_large_per_refresh_mean": (
                (
                    sum(feature_row_store_cache_skip_too_large)
                    / len(feature_row_store_cache_skip_too_large)
                )
                if feature_row_store_cache_skip_too_large
                else None
            ),
            "streaming_chunk_cache_hits_per_refresh_mean": (
                (sum(streaming_chunk_cache_hits) / len(streaming_chunk_cache_hits))
                if streaming_chunk_cache_hits
                else None
            ),
            "streaming_chunk_cache_misses_per_refresh_mean": (
                (sum(streaming_chunk_cache_misses) / len(streaming_chunk_cache_misses))
                if streaming_chunk_cache_misses
                else None
            ),
            "streaming_chunk_cache_store_success_per_refresh_mean": (
                (
                    sum(streaming_chunk_cache_store_success)
                    / len(streaming_chunk_cache_store_success)
                )
                if streaming_chunk_cache_store_success
                else None
            ),
            "streaming_chunk_cache_skip_disabled_per_refresh_mean": (
                (
                    sum(streaming_chunk_cache_skip_disabled)
                    / len(streaming_chunk_cache_skip_disabled)
                )
                if streaming_chunk_cache_skip_disabled
                else None
            ),
            "streaming_chunk_cache_skip_too_large_per_refresh_mean": (
                (
                    sum(streaming_chunk_cache_skip_too_large)
                    / len(streaming_chunk_cache_skip_too_large)
                )
                if streaming_chunk_cache_skip_too_large
                else None
            ),
            "phase3_logit_row_batch_count": int(
                len(phase3_logit_row_batches) if isinstance(phase3_logit_row_batches, list) else 0
            ),
            "phase4_feature_row_batch_count": int(
                len(phase4_feature_row_batches)
                if isinstance(phase4_feature_row_batches, list)
                else 0
            ),
            "first_refresh_float32_effectively_all_zero": (
                bool(first_float_precision.get("float32_effectively_all_zero"))
                if isinstance(first_float_precision, dict)
                else None
            ),
            "first_refresh_float64_effectively_all_zero": (
                bool(first_float_precision.get("float64_effectively_all_zero"))
                if isinstance(first_float_precision, dict)
                else None
            ),
            "phase3_logit_row_batch_0_abs_sum": (
                first_phase3_logit_batch.get("row_abs_sum_stats", {}).get("abs_sum")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
            "phase3_logit_row_batch_0_max_abs": (
                first_phase3_logit_batch.get("row_input_stats", {}).get("finite_max_abs")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
            "phase3_logit_row_batch_0_nonfinite_count": (
                first_phase3_logit_batch.get("row_input_stats", {}).get("nonfinite_count")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
            "phase3_logit_row_batch_0_row_l1_max": (
                first_phase3_logit_batch.get("row_abs_sum_stats", {}).get("max")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
            "phase3_logit_row_batch_0_row_l1_effectively_all_zero": (
                first_phase3_logit_batch.get("row_abs_sum_stats", {}).get("effectively_all_zero")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
            "phase3_logit_row_batch_0_row_l1_nonfinite_count": (
                first_phase3_logit_batch.get("row_abs_sum_stats", {}).get("nonfinite_count")
                if isinstance(first_phase3_logit_batch, dict)
                else None
            ),
        }

    return Phase4Result(
        visited=visited,
        actual_max_feature_nodes=actual_max_feature_nodes,
        edge_matrix=edge_matrix,
        feature_row_store=feature_row_store,
        nonfeature_row_store=nonfeature_row_store,
        row_to_node_index=row_to_node_index,
        rows_cpu_staging=rows_cpu_staging,
        st=st,
        phase4_frontier_buffer_metadata=phase4_frontier_buffer_metadata,
        phase4_execution_metadata=phase4_execution_metadata,
        cross_cluster_debug_summary=cross_cluster_debug_summary,
        cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
        cross_cluster_debug_batches=cross_cluster_debug_batches,
        anomaly_debug_result=anomaly_debug_result,
        phase4_elapsed_ms=phase4_elapsed_ms,
        phase4_feature_batch_size=phase4_feature_batch_size,
        phase4_executor_reference_batch_size=phase4_executor_reference_batch_size,
        phase4_executor_microbatch_size=phase4_executor_microbatch_size,
        phase4_refresh_count=phase4_refresh_count,
        phase4_scheduler_reference_batch_count=phase4_scheduler_reference_batch_count,
        phase4_executor_microbatch_count=phase4_executor_microbatch_count,
        phase4_refresh_elapsed_ms_total=phase4_refresh_elapsed_ms_total,
        phase4_feature_batch_elapsed_ms_total=phase4_feature_batch_elapsed_ms_total,
        phase4_refresh_partial_influence_elapsed_ms_total=phase4_refresh_partial_influence_elapsed_ms_total,
        phase4_refresh_rank_topk_elapsed_ms_total=phase4_refresh_rank_topk_elapsed_ms_total,
        phase4_refresh_frontier_plan_elapsed_ms_total=phase4_refresh_frontier_plan_elapsed_ms_total,
        phase4_refresh_row_store_read_elapsed_ms_total=phase4_refresh_row_store_read_elapsed_ms_total,
    )
