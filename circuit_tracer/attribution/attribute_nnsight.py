"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import logging
import os
import sys
import time
from collections.abc import Sequence
from typing import Any, Literal, Mapping, cast

import torch

from circuit_tracer.attribution.targets import (
    TargetSpec,
)
from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.graph import (
    Graph,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    require_exact_chunked_provider,
)
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.utils.telemetry import (
    format_memory_snapshot,
)

from circuit_tracer.observability.exception_export import (
    _TELEMETRY_EXCEPTION_EVENTS_ATTR as _TELEMETRY_EXCEPTION_EVENTS_ATTR,
    _TELEMETRY_EXCEPTION_SUMMARY_ATTR as _TELEMETRY_EXCEPTION_SUMMARY_ATTR,
    _attach_telemetry_export_to_exception as _attach_telemetry_export_to_exception,
)
from circuit_tracer.observability.human_logs import (
    _log_batch_profile as _log_batch_profile,
    _log_memory_boundary as _log_memory_boundary,
    _log_phase_metrics as _log_phase_metrics,
    _snapshot_diagnostics as _snapshot_diagnostics,
)
from circuit_tracer.observability.lifecycle import (
    TelemetryObserver,
    _TelemetryObserver as _TelemetryObserver,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _RowTransferTelemetry as _RowTransferTelemetry,
    _build_cross_cluster_runtime_snapshot,
    _build_phase4_executor_batch_telemetry as _build_phase4_executor_batch_telemetry,
    _build_phase4_executor_substage_telemetry as _build_phase4_executor_substage_telemetry,
    _build_phase4_gpu_row_reduction_transfer_telemetry as _build_phase4_gpu_row_reduction_transfer_telemetry,
    _build_phase4_refresh_substage_telemetry as _build_phase4_refresh_substage_telemetry,
    _build_row_transfer_telemetry as _build_row_transfer_telemetry,
    _dtype_element_size as _dtype_element_size,
    _record_cross_cluster_batch_event as _record_cross_cluster_batch_event,
    _record_cross_cluster_checkpoint,
)
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _EXACT_ENCODER_RESIDENCY_DEFAULT as _EXACT_ENCODER_RESIDENCY_DEFAULT,
    _EXACT_ENCODER_RESIDENCY_EFFECTIVE_MODE_BY_MODE as _EXACT_ENCODER_RESIDENCY_EFFECTIVE_MODE_BY_MODE,
    _ExactEncoderResidencyConfig as _ExactEncoderResidencyConfig,
    _PHASE4_PLANNER_V2_CANDIDATE_WINDOW_MULTIPLIER as _PHASE4_PLANNER_V2_CANDIDATE_WINDOW_MULTIPLIER,
    _PHASE4_PLANNER_V2_LOCKED_PREFIX_FRACTION as _PHASE4_PLANNER_V2_LOCKED_PREFIX_FRACTION,
    _PHASE4_PLANNER_V2_MAX_REPLACEMENT_FRACTION as _PHASE4_PLANNER_V2_MAX_REPLACEMENT_FRACTION,
    _PHASE4_PLANNER_V2_MIN_SCORE_RATIO as _PHASE4_PLANNER_V2_MIN_SCORE_RATIO,
    _PHASE4_PLANNER_V2_POLICY_VERSION as _PHASE4_PLANNER_V2_POLICY_VERSION,
    _PHASE4_RANKER_DEFAULT as _PHASE4_RANKER_DEFAULT,
    _PHASE4_RANKER_EFFECTIVE_MODE_BY_MODE as _PHASE4_RANKER_EFFECTIVE_MODE_BY_MODE,
    _PHASE4_RANKER_TIE_BEHAVIOR_BY_MODE as _PHASE4_RANKER_TIE_BEHAVIOR_BY_MODE,
    _PHASE4_RANK_SELECTION_NEAR_CUTOFF_EPSILON as _PHASE4_RANK_SELECTION_NEAR_CUTOFF_EPSILON,
    _PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT as _PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT,
    _PHASE4_REFRESH_MEMORY_ATTR_KEYS as _PHASE4_REFRESH_MEMORY_ATTR_KEYS,
    _PHASE4_REFRESH_OPTIMIZATION_EFFECTIVE_MODE_BY_MODE as _PHASE4_REFRESH_OPTIMIZATION_EFFECTIVE_MODE_BY_MODE,
    _PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE as _PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE,
    _PHASE4_REFRESH_POLICY_DEFAULT as _PHASE4_REFRESH_POLICY_DEFAULT,
    _PHASE4_REFRESH_POLICY_EFFECTIVE_POLICY_BY_POLICY as _PHASE4_REFRESH_POLICY_EFFECTIVE_POLICY_BY_POLICY,
    _PHASE4_ROW_EXECUTOR_EFFECTIVE_MODE_BY_MODE as _PHASE4_ROW_EXECUTOR_EFFECTIVE_MODE_BY_MODE,
    _PHASE4_ROW_EXECUTOR_VERSION_BY_MODE as _PHASE4_ROW_EXECUTOR_VERSION_BY_MODE,
    _PHASE4_ROW_REDUCTION_VERSION_BY_MODE as _PHASE4_ROW_REDUCTION_VERSION_BY_MODE,
    _PHASE4_SCHEDULER_EFFECTIVE_MODE_BY_MODE as _PHASE4_SCHEDULER_EFFECTIVE_MODE_BY_MODE,
    _PHASE4_SCHEDULER_MODE_ALIAS as _PHASE4_SCHEDULER_MODE_ALIAS,
    _PHASE4_SCHEDULER_POLICY_BY_MODE as _PHASE4_SCHEDULER_POLICY_BY_MODE,
    _PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS as _PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS,
    _PHASE4_SCHEDULER_VERSION_BY_MODE as _PHASE4_SCHEDULER_VERSION_BY_MODE,
    _PHASE4_STREAMING_V1_MAX_MICROBATCH_SIZE as _PHASE4_STREAMING_V1_MAX_MICROBATCH_SIZE,
    _Phase4FrontierPlan as _Phase4FrontierPlan,
    _Phase4FrontierRankSelection as _Phase4FrontierRankSelection,
    _Phase4RankerConfig as _Phase4RankerConfig,
    _Phase4RefreshOptimizationConfig as _Phase4RefreshOptimizationConfig,
    _Phase4RefreshPolicyConfig as _Phase4RefreshPolicyConfig,
    _Phase4RowExecutorConfig as _Phase4RowExecutorConfig,
    _Phase4RowReductionConfig as _Phase4RowReductionConfig,
    _Phase4SchedulerConfig as _Phase4SchedulerConfig,
    _apply_phase4_planner_v2_refresh_plan as _apply_phase4_planner_v2_refresh_plan,
    _build_exact_encoder_residency_metadata as _build_exact_encoder_residency_metadata,
    _build_phase4_batch_locality_summary as _build_phase4_batch_locality_summary,
    _build_phase4_frontier_locality_fragmentation_summary as _build_phase4_frontier_locality_fragmentation_summary,
    _build_phase4_planner_v2_candidate_window as _build_phase4_planner_v2_candidate_window,
    _build_phase4_planner_v2_refresh_telemetry_disabled as _build_phase4_planner_v2_refresh_telemetry_disabled,
    _build_phase4_probe_pending_frontier as _build_phase4_probe_pending_frontier,
    _build_phase4_ranker_metadata as _build_phase4_ranker_metadata,
    _build_phase4_refresh_optimization_metadata as _build_phase4_refresh_optimization_metadata,
    _build_phase4_refresh_policy_metadata as _build_phase4_refresh_policy_metadata,
    _build_phase4_row_executor_metadata as _build_phase4_row_executor_metadata,
    _build_phase4_row_reduction_metadata as _build_phase4_row_reduction_metadata,
    _build_phase4_scheduler_metadata as _build_phase4_scheduler_metadata,
    _build_phase4_scheduler_plan_telemetry as _build_phase4_scheduler_plan_telemetry,
    _compute_phase4_locality_shaped_batch_end as _compute_phase4_locality_shaped_batch_end,
    _compute_phase4_locality_shaped_batch_end_with_reason as _compute_phase4_locality_shaped_batch_end_with_reason,
    _compute_phase4_locality_shaped_frontier_size as _compute_phase4_locality_shaped_frontier_size,
    _compute_phase4_planned_feature_batch_size as _compute_phase4_planned_feature_batch_size,
    _compute_phase4_rank_selection_cutoff_metadata as _compute_phase4_rank_selection_cutoff_metadata,
    _compute_phase4_rank_selection_max_feature_nodes_cap_bound as _compute_phase4_rank_selection_max_feature_nodes_cap_bound,
    _compute_phase4_refresh_cycle_batches as _compute_phase4_refresh_cycle_batches,
    _compute_phase4_refresh_queue_window_size as _compute_phase4_refresh_queue_window_size,
    _get_cuda_reserved_snapshot as _get_cuda_reserved_snapshot,
    _phase4_planner_v2_group_key as _phase4_planner_v2_group_key,
    _plan_phase4_feature_batch_size_preflight as _plan_phase4_feature_batch_size_preflight,
    _plan_phase4_frontier_membership_preserving_v1 as _plan_phase4_frontier_membership_preserving_v1,
    _rank_phase4_unvisited_features_argsort as _rank_phase4_unvisited_features_argsort,
    _reorder_pending_for_phase4_locality as _reorder_pending_for_phase4_locality,
    _resolve_exact_encoder_residency as _resolve_exact_encoder_residency,
    _resolve_exact_encoder_residency_config as _resolve_exact_encoder_residency_config,
    _resolve_phase4_feature_batch_planner_enabled as _resolve_phase4_feature_batch_planner_enabled,
    _resolve_phase4_feature_batch_planner_status as _resolve_phase4_feature_batch_planner_status,
    _resolve_phase4_ranker as _resolve_phase4_ranker,
    _resolve_phase4_ranker_config as _resolve_phase4_ranker_config,
    _resolve_phase4_refresh_interval_multiplier as _resolve_phase4_refresh_interval_multiplier,
    _resolve_phase4_refresh_optimization_config as _resolve_phase4_refresh_optimization_config,
    _resolve_phase4_refresh_optimization_mode as _resolve_phase4_refresh_optimization_mode,
    _resolve_phase4_refresh_policy as _resolve_phase4_refresh_policy,
    _resolve_phase4_refresh_policy_config as _resolve_phase4_refresh_policy_config,
    _resolve_phase4_row_executor_config as _resolve_phase4_row_executor_config,
    _resolve_phase4_row_executor_mode as _resolve_phase4_row_executor_mode,
    _resolve_phase4_row_reduction_config as _resolve_phase4_row_reduction_config,
    _resolve_phase4_row_reduction_mode as _resolve_phase4_row_reduction_mode,
    _resolve_phase4_scheduler_config as _resolve_phase4_scheduler_config,
    _resolve_phase4_scheduler_mode as _resolve_phase4_scheduler_mode,
    _resolve_phase4_scheduler_telemetry_detail as _resolve_phase4_scheduler_telemetry_detail,
    _resolve_phase4_streaming_v1_microbatch_size as _resolve_phase4_streaming_v1_microbatch_size,
    _select_phase4_frontier_rank_selection as _select_phase4_frontier_rank_selection,
    _select_phase4_planner_v2_membership as _select_phase4_planner_v2_membership,
)
from circuit_tracer.attribution.nnsight.phase1_policy import (
    _PHASE1_TRACE_BATCH_POLICY_DEFAULT as _PHASE1_TRACE_BATCH_POLICY_DEFAULT,
    _PHASE1_TRACE_BATCH_POLICY_EFFECTIVE_POLICY_BY_POLICY as _PHASE1_TRACE_BATCH_POLICY_EFFECTIVE_POLICY_BY_POLICY,
    _PHASE1_TRACE_BATCH_SIZE_MAX_DEFAULT as _PHASE1_TRACE_BATCH_SIZE_MAX_DEFAULT,
    _Phase1TraceBatchConfig as _Phase1TraceBatchConfig,
    _Phase1TraceBatchSizing as _Phase1TraceBatchSizing,
    _build_phase1_trace_batch_metadata as _build_phase1_trace_batch_metadata,
    _build_phase1_trace_batch_sizing_metadata as _build_phase1_trace_batch_sizing_metadata,
    _resolve_phase1_trace_batch_config as _resolve_phase1_trace_batch_config,
    _resolve_phase1_trace_batch_policy as _resolve_phase1_trace_batch_policy,
    _resolve_phase1_trace_batch_size_max as _resolve_phase1_trace_batch_size_max,
    _resolve_phase1_trace_batch_sizing as _resolve_phase1_trace_batch_sizing,
)
from circuit_tracer.attribution.nnsight.phases.phase0 import (
    Phase0CleanupOwner,
    Phase0Config,
    Phase0ExecutionError,
    Phase0Inputs,
    run_phase0,
)
from circuit_tracer.attribution.nnsight.phases.phase1 import (
    _run_phase1_forward_pass as _run_phase1_forward_pass,
)
from circuit_tracer.attribution.nnsight.phases.phase2 import (
    Phase2Config,
    Phase2Inputs,
    Phase2ResourceOwner,
    run_phase2,
)
from circuit_tracer.attribution.nnsight.phases.phase3 import (
    Phase3Config,
    Phase3Inputs,
    run_phase3,
)
from circuit_tracer.attribution.nnsight.phases.phase4 import (
    Phase4Config,
    Phase4Inputs,
    run_phase4,
)
from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata as PrefixViewMetadata,
    _apply_prefix_view_activation_mask as _apply_prefix_view_activation_mask,
    _compact_nonfeature_column_counts as _compact_nonfeature_column_counts,
    _compact_selected_feature_columns as _compact_selected_feature_columns,
    _hash_token_ids as _hash_token_ids,
    _resolve_prefix_view_output_position as _resolve_prefix_view_output_position,
    _resolve_prefix_view_trace_input_ids as _resolve_prefix_view_trace_input_ids,
    _token_ids_from_attribution_targets as _token_ids_from_attribution_targets,
    _tokens_from_prompt_for_prefix_view as _tokens_from_prompt_for_prefix_view,
    validate_compact_prefix_view_output as validate_compact_prefix_view_output,
    validate_prefix_view_metadata as validate_prefix_view_metadata,
)
from circuit_tracer.attribution.nnsight.numerics import (
    _EXACT_TRACE_INTERNAL_DTYPE_BY_NAME as _EXACT_TRACE_INTERNAL_DTYPE_BY_NAME,
    _exact_trace_internal_dtype_name as _exact_trace_internal_dtype_name,
    _resolve_exact_trace_internal_dtype as _resolve_exact_trace_internal_dtype,
    _row_abs_sums_to_scaled_l1 as _row_abs_sums_to_scaled_l1,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _resolve_internal_precision_requested as _resolve_internal_precision_requested,
    _warn_internal_precision_deprecated as _warn_internal_precision_deprecated,
    _dtype_to_name as _dtype_to_name,
    _resolve_internal_dtype_map as _resolve_internal_dtype_map,
    _dtype_from_name as _dtype_from_name,
    _build_vector_stats as _build_vector_stats,
    _resolve_phase3_effective_row_state as _resolve_phase3_effective_row_state,
    _copy_rows_to_cpu_staging as _copy_rows_to_cpu_staging,
    _copy_feature_rows_to_cpu_staging as _copy_feature_rows_to_cpu_staging,
    _row_denominator_to_row_abs_sums as _row_denominator_to_row_abs_sums,
    _build_matrix_abs_stats as _build_matrix_abs_stats,
    _build_phase4_normalization_stats as _build_phase4_normalization_stats,
    _build_phase4_cutoff_debug as _build_phase4_cutoff_debug,
    _build_phase3_frontier_buffer_metadata as _build_phase3_frontier_buffer_metadata,
    _build_phase4_frontier_buffer_decision as _build_phase4_frontier_buffer_decision,
    _build_semantic_sketch_fallback as _build_semantic_sketch_fallback,
    _build_feature_semantic_descriptors_payload as _build_feature_semantic_descriptors_payload,
    _annotate_phase4_selection_on_feature_semantic_descriptors as _annotate_phase4_selection_on_feature_semantic_descriptors,
    _record_phase4_refresh_debug as _record_phase4_refresh_debug,
    _compare_phase4_frontiers as _compare_phase4_frontiers,
    _build_phase4_deterministic_shadow_pending as _build_phase4_deterministic_shadow_pending,
    _build_phase4_environment_fingerprint as _build_phase4_environment_fingerprint,
)
from circuit_tracer.attribution.nnsight.replay import (
    _PHASE0_DONOR_CONTEXT_POLICY_BY_NAME as _PHASE0_DONOR_CONTEXT_POLICY_BY_NAME,
    _PHASE0_REPLAY_MODE_BY_NAME as _PHASE0_REPLAY_MODE_BY_NAME,
    _PHASE3_REPLAY_MODE_BY_NAME as _PHASE3_REPLAY_MODE_BY_NAME,
    _PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME as _PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME,
    _build_phase0_activation_matrix_from_loaded_bundle as _build_phase0_activation_matrix_from_loaded_bundle,
    _compute_row_abs_sums as _compute_row_abs_sums,
    _compute_row_denominator_scaled_l1 as _compute_row_denominator_scaled_l1,
    _build_phase0_donor_bundle_payload as _build_phase0_donor_bundle_payload,
    _build_phase0_replay_metadata as _build_phase0_replay_metadata,
    _build_phase0_replay_validation_context as _build_phase0_replay_validation_context,
    _build_phase3_gradient_bundle_payload as _build_phase3_gradient_bundle_payload,
    _build_phase3_replay_metadata as _build_phase3_replay_metadata,
    _build_phase3_row_bundle_payload as _build_phase3_row_bundle_payload,
    _build_phase3_seed_bundle_payload as _build_phase3_seed_bundle_payload,
    _build_phase3_seed_influence_topk as _build_phase3_seed_influence_topk,
    _extract_clt_constants_hash_from_snapshot as _extract_clt_constants_hash_from_snapshot,
    _hash_float_tensor as _hash_float_tensor,
    _hash_index_tensor as _hash_index_tensor,
    _hash_sparse_membership_indices as _hash_sparse_membership_indices,
    _hash_tensor_raw_bytes as _hash_tensor_raw_bytes,
    _load_phase0_donor_bundle_npz as _load_phase0_donor_bundle_npz,
    _load_phase3_gradient_donor_bundle_npz as _load_phase3_gradient_donor_bundle_npz,
    _load_phase3_row_donor_bundle_npz as _load_phase3_row_donor_bundle_npz,
    _phase0_npz_int as _phase0_npz_int,
    _phase0_npz_optional_str as _phase0_npz_optional_str,
    _phase0_npz_scalar as _phase0_npz_scalar,
    _phase0_to_int64_tensor as _phase0_to_int64_tensor,
    _phase3_npz_int as _phase3_npz_int,
    _phase3_npz_optional_str as _phase3_npz_optional_str,
    _resolve_phase0_donor_context_policy as _resolve_phase0_donor_context_policy,
    _resolve_phase0_replay_mode as _resolve_phase0_replay_mode,
    _resolve_phase3_replay_mode as _resolve_phase3_replay_mode,
    _resolve_phase3_replay_validation_policy as _resolve_phase3_replay_validation_policy,
)
from circuit_tracer.attribution.nnsight.row_store import (
    _FileBackedFeatureRowStore as _FileBackedFeatureRowStore,
    _ROW_STORE_CACHE_CONTROL_DEFAULT as _ROW_STORE_CACHE_CONTROL_DEFAULT,
    _ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE as _ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE,
    _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_AND_READ_V1 as _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_AND_READ_V1,
    _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_V1 as _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_V1,
    _ROW_STORE_TEMP_ROOT_POLICY_BY_NAME as _ROW_STORE_TEMP_ROOT_POLICY_BY_NAME,
    _ROW_STORE_TEMP_ROOT_POLICY_DEFAULT as _ROW_STORE_TEMP_ROOT_POLICY_DEFAULT,
    _ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL as _ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL,
    _RowStoreCacheControlConfig as _RowStoreCacheControlConfig,
    _RowStoreCacheControlMode as _RowStoreCacheControlMode,
    _RowStoreTempRootSelection as _RowStoreTempRootSelection,
    _build_row_store_cache_control_metadata as _build_row_store_cache_control_metadata,
    _is_existing_writable_dir as _is_existing_writable_dir,
    _resolve_row_store_cache_control as _resolve_row_store_cache_control,
    _resolve_row_store_cache_control_config as _resolve_row_store_cache_control_config,
    _resolve_row_store_temp_root_policy as _resolve_row_store_temp_root_policy,
    _select_row_store_temp_root as _select_row_store_temp_root,
)


_PHASE0_ACTIVATION_THRESHOLD_COMPARE_MODE_BY_NAME: dict[str, str] = {
    "baseline": "baseline",
    "default": "baseline",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp64": "fp64",
    "float64": "fp64",
    "torch.float64": "fp64",
}


def _resolve_phase0_activation_threshold_compare_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _PHASE0_ACTIVATION_THRESHOLD_COMPARE_MODE_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_PHASE0_ACTIVATION_THRESHOLD_COMPARE_MODE_BY_NAME))
        raise ValueError(
            f"phase0_activation_threshold_compare_mode must be one of: {allowed} (got {value!r})"
        )
    return resolved


def _resolve_telemetry_max_events(
    *,
    telemetry_max_events: int | None,
    compact_output: bool,
    exact_chunked_decoder: bool,
    profile: bool,
    phase4_anomaly_debug_enabled: bool,
) -> int:
    if telemetry_max_events is not None and telemetry_max_events > 0:
        return int(telemetry_max_events)

    if compact_output and exact_chunked_decoder:
        return 120_000
    if profile or phase4_anomaly_debug_enabled:
        return 60_000
    return 20_000


def _resolve_phase4_anomaly_debug_enabled(phase4_anomaly_debug: bool) -> bool:
    return bool(phase4_anomaly_debug)


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: NNSightReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    plan_feature_batch_size: bool = False,
    auto_scale_feature_batch_size: bool = False,
    feature_batch_size_max: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    internal_precision: Literal["float32", "float64"] | None = None,
    phase4_anomaly_debug: bool = False,
    cross_cluster_debug: bool = False,
    capture_phase0_donor_bundle: bool = False,
    capture_phase3_seed_bundle: bool = False,
    capture_phase3_gradient_bundle: bool = False,
    capture_phase3_row_bundle: bool = False,
    capture_feature_semantic_descriptors: bool = False,
    semantic_descriptor_top_k: int = 2048,
    semantic_descriptor_dim: int = 64,
    telemetry_max_events: int | None = None,
    telemetry_jsonl_path: str | os.PathLike[str] | None = None,
    telemetry_context: Mapping[str, object] | None = None,
    compact_output: bool = False,
    phase0_donor_bundle: str | os.PathLike[str] | None = None,
    phase0_replay_mode: Literal["disabled", "donor_phase0"] = "disabled",
    phase0_donor_context_policy: Literal["strict", "warn"] = "strict",
    phase3_gradient_donor_bundle: str | os.PathLike[str] | None = None,
    phase3_gradient_replay_mode: Literal["disabled", "donor"] = "disabled",
    phase3_row_donor_bundle: str | os.PathLike[str] | None = None,
    phase3_row_replay_mode: Literal["disabled", "donor"] = "disabled",
    phase3_replay_validation_policy: Literal["strict"] = "strict",
    phase4_scheduler_mode: Literal["locality", "planner_v1", "planner_v2", "legacy"] = "locality",
    phase4_scheduler_debug: bool = False,
    phase4_scheduler_telemetry_detail: Literal["summary", "normal", "debug"] = "normal",
    phase4_refresh_optimization: Literal["off", "v1"] = "v1",
    phase4_refresh_prepared_chunk_cache_bytes: int = 0,
    phase4_refresh_active_row_accumulation: Literal["zero_fill", "direct_v1"] = "direct_v1",
    phase4_row_executor: Literal["batched", "streaming_v1"] = "batched",
    phase4_row_reduction: Literal["off", "gpu_v1"] = "gpu_v1",
    phase1_trace_batch_policy: Literal["legacy", "cap_effective_batches"] = "legacy",
    phase1_trace_batch_size_max: int | None = None,
    phase4_refresh_policy: Literal["standard", "deferred_v1"] = "standard",
    phase4_refresh_interval_multiplier: int = 1,
    phase4_ranker: Literal["argsort", "topk_v1"] = "argsort",
    row_store_cache_control: _RowStoreCacheControlMode = "off",
    row_store_temp_root_policy: Literal["default", "env_node_local"] = "default",
    row_store_temp_root: str | os.PathLike[str] | None = None,
    row_store_preallocate: bool = True,
    exact_encoder_residency: Literal["lazy", "active_cpu", "active_pinned_cpu"] = "lazy",
    exact_trace_internal_dtype: Literal["fp32", "fp64"] = "fp32",
    phase3_frontier_buffer_relative_epsilon: float | None = None,
    phase3_frontier_buffer_max_extra: int = 0,
    phase4_frontier_buffer_relative_epsilon: float | None = None,
    phase4_frontier_buffer_max_extra_per_refresh: int = 0,
    phase4_frontier_buffer_max_extra_total: int = 0,
    phase0_activation_threshold_compare_mode: Literal[
        "baseline", "bf16", "fp32", "fp64"
    ] = "baseline",
    prefix_view_metadata: Mapping[str, Any] | None = None,
    output_position: int | None = None,
    decoder_chunk_cache: object | None = None,
    decoder_cache_fingerprint: object | None = None,
    _phase0_context_override: object | None = None,
    _target_logits_override: torch.Tensor | None = None,
    _target_logit_source: str | None = None,
) -> Graph:
    """Compute an attribution graph for *prompt* using NNSight backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``NNSightReplacementModel``
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or tuple)
                            with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        feature_batch_size: Optional override for feature-attribution batches.
            Defaults to ``batch_size`` when omitted.
        logit_batch_size: Optional override for logit-attribution batches.
            Defaults to ``batch_size`` when omitted.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        profile: Whether to emit batch-level diagnostic profiling logs.
        profile_log_interval: Log every N batches when profiling.
        diagnostic_feature_cap: Optional debug-only early cap on active features.
            This changes attribution semantics and should only be used for profiling.
        sparsification: Optional candidate-screening config applied before
            reconstruction and reused by later attribution phases.
        chunked_feature_replay_window: Exact-mode knob controlling how many
            layer grads are buffered before chunked feature replay flush.
        error_vector_prefetch_lookahead: Exact-mode knob controlling staged
            error-vector lookahead window size.
        stage_encoder_vecs_on_cpu: Exact-mode knob to force/disable CPU staging
            of encoder vectors. ``None`` preserves backend default behavior.
        stage_error_vectors_on_cpu: Exact-mode knob to force/disable CPU
            staging of error vectors. ``None`` preserves backend defaults.
        row_subchunk_size: Optional exact-mode knob controlling inner replay
            row subchunk size. ``None`` preserves current behavior (equal to
            decoder chunk size).
        plan_feature_batch_size: Whether to run a probe-based preflight and
            choose a single fixed Phase-4 feature microbatch size for the run.
        auto_scale_feature_batch_size: Legacy alias for
            ``plan_feature_batch_size`` (kept for backward compatibility).
        feature_batch_size_max: Optional upper bound for the preflight-planned
            Phase-4 feature microbatch size.
        feature_batch_target_reserved_fraction: Reserved-memory utilization
            target used by the planner (0-1).
        feature_batch_min_free_fraction: Minimum free-memory fraction to keep
            unused (0-1), applied as a stricter cap than target utilization.
        feature_batch_probe_batches: Number of preflight Phase-4 probe batches
            to run before the real attribution pass.
        internal_precision: Deprecated compatibility override for exact chunked
            internals. Prefer ``exact_trace_internal_dtype``; when omitted,
            internal precision is derived from that canonical contract.
        phase4_anomaly_debug: Enable opt-in Phase-4 anomaly debug scaffolding.
            Environment-variable activation is intentionally unsupported so run
            provenance stays explicit in scenario/config inputs.
        cross_cluster_debug: Enable broad scalar-only cross-cluster debug summary
            scaffolding (Phase 0 through pre-Phase-4 checkpoints).
        capture_phase0_donor_bundle: Enable opt-in Phase-0 donor bundle payload
            capture for compact exact-chunked runs.
        capture_phase3_seed_bundle: Enable opt-in Phase-3 seed-bundle payload
            capture for compact exact-chunked runs.
        capture_phase3_gradient_bundle: Enable opt-in Phase-3 backward-gradient
            payload capture for compact exact-chunked runs.
        capture_phase3_row_bundle: Enable opt-in Phase-3 direct-effect row
            payload capture for compact exact-chunked runs.
        capture_feature_semantic_descriptors: Enable opt-in bounded semantic
            descriptor payload for Phase-3 candidate features.
        semantic_descriptor_top_k: Maximum number of candidate features to
            include in semantic descriptor payloads.
        semantic_descriptor_dim: Descriptor width (number of float values)
            for each candidate feature sketch.
        telemetry_max_events: Optional cap for in-memory telemetry event storage.
            If omitted, a deterministic in-code default policy is used.
        phase0_donor_bundle: Optional path to a saved Phase-0 donor bundle
            (``*.npz``) used for replay in compact exact-chunked runs.
        phase0_replay_mode: Phase-0 donor replay mode. ``"disabled"`` keeps
            host Phase-0 active features. ``"donor_phase0"`` replaces host
            Phase-0 active features with the donor bundle.
        phase0_donor_context_policy: Validation policy for donor/host context
            mismatches (``"strict"`` raises, ``"warn"`` records warnings).
        exact_trace_internal_dtype: Internal dtype for compact exact-trace
            normalization/influence ranking path. ``"fp64"`` uses float64
            internals; ``"fp32"`` uses float32 internals.
        phase0_activation_threshold_compare_mode: Phase-0-only activation/
            threshold compare mode for JumpReLU membership decisions.
            ``"baseline"`` preserves default compare behavior.
        phase4_scheduler_mode: Phase-4 frontier scheduler mode. ``"locality"``
            keeps current behavior. ``"planner_v1"`` routes frontier selection and
            intra-frontier batching through the membership-preserving planner core.
            ``"planner_v2"`` enables bounded-membership selection (with explicit
            per-refresh fallback telemetry when the planner-v1 reference plan is
            reused). ``"legacy"`` is accepted as an alias for ``"locality"``.
        phase4_scheduler_debug: Emit additional planner-v1 scheduler diagnostics in
            Phase 4 logs.
        phase4_scheduler_telemetry_detail: Scheduler telemetry verbosity for
            Phase-4 refresh/batch events. ``"summary"`` keeps compact planner
            metadata, ``"normal"`` adds full plan aggregates, and ``"debug"``
            additionally includes bounded samples.
        phase4_refresh_optimization: Requested Phase-4 refresh optimization mode.
            ``"off"`` keeps current behavior. ``"v1"`` enables the compact
            refresh row-range reader optimization while preserving exact math.
        phase4_row_executor: Requested Phase-4 row execution mode.
        phase4_row_reduction: Requested Phase-4 row-reduction backend.
            ``"gpu_v1"`` enables the compact exact Phase-4 staged GPU-denominator
            path and is the default; ``"off"`` keeps the CPU reference path.
            ``"batched"`` keeps current behavior. ``"streaming_v1"`` executes
            compact exact-trace Phase-4 feature rows in smaller compute micro-batches
            while preserving scheduler frontier membership/order semantics.
        phase1_trace_batch_policy: Requested Phase-1 trace-batch sizing policy.
            ``"legacy"`` keeps current behavior. ``"cap_effective_batches"``
            caps only the effective Phase-1 source/invoke trace batch size to
            ``phase1_trace_batch_size_max``.
        phase1_trace_batch_size_max: Optional cap paired with
            ``phase1_trace_batch_policy``. Required to activate
            ``"cap_effective_batches"``; when omitted under that policy, execution
            falls back to ``"legacy"`` with explicit metadata.
        phase4_refresh_policy: Requested Phase-4 refresh cadence policy.
            ``"standard"`` keeps current behavior; ``"deferred_v1"`` expands
            the per-refresh pending/frontier queue for compact exact-trace Phase 4
            (``compact_output=True`` + exact chunked decoder). Outside that path,
            execution falls back to ``"standard"`` with explicit metadata.
        phase4_refresh_interval_multiplier: Positive integer cadence multiplier
            used by ``"deferred_v1"`` to scale the Phase-4 pending/frontier
            queue window while keeping executor microbatch sizing unchanged.
        phase4_ranker: Requested Phase-4 ranking implementation.
            ``"argsort"`` keeps current behavior; ``"topk_v1"`` uses
            ``torch.topk`` for Phase-4 frontier membership selection (next-K
            pending window), then orders selected entries by descending score.
            For equal scores at the cutoff, membership may differ from argsort.
        row_store_cache_control: Requested compact row-store cache control mode.
            ``"off"`` keeps current behavior;
            ``"fadvise_dontneed_after_append_v1"`` asks the file-backed dense
            row store to drop appended byte ranges from the page cache after
            writes; ``"fadvise_dontneed_after_append_and_read_v1"`` also drops
            safely materialized read ranges after copying them out of the
            memmap-backed store.
        exact_encoder_residency: Requested exact encoder residency mode.
            ``"lazy"`` keeps current behavior. ``"active_cpu"`` and
            ``"active_pinned_cpu"`` materialize active encoder rows during
            Phase 0 and stage them on CPU for exact chunked decoder runs.
            Outside exact chunked decoder execution, active modes fall back to
            ``"lazy"`` with explicit metadata.
        exact_trace_internal_dtype: Internal dtype for compact exact-trace
            normalization/influence ranking path. ``"fp32"`` uses float32
            internals and is the post-fix default; ``"fp64"`` uses float64
            internals.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if (verbose or profile) and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    normalized_prefix_view_metadata = validate_prefix_view_metadata(
        prompt=prompt,
        attribution_targets=attribution_targets,
        prefix_view_metadata=prefix_view_metadata,
    )
    output_position = _resolve_prefix_view_output_position(
        normalized_prefix_view_metadata,
        output_position,
    )
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            feature_batch_size=feature_batch_size,
            logit_batch_size=logit_batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            profile=profile,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            sparsification=sparsification,
            chunked_feature_replay_window=chunked_feature_replay_window,
            error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
            row_subchunk_size=row_subchunk_size,
            plan_feature_batch_size=plan_feature_batch_size,
            auto_scale_feature_batch_size=auto_scale_feature_batch_size,
            feature_batch_size_max=feature_batch_size_max,
            feature_batch_target_reserved_fraction=feature_batch_target_reserved_fraction,
            feature_batch_min_free_fraction=feature_batch_min_free_fraction,
            feature_batch_probe_batches=feature_batch_probe_batches,
            internal_precision=internal_precision,
            phase4_anomaly_debug=phase4_anomaly_debug,
            cross_cluster_debug=cross_cluster_debug,
            capture_phase0_donor_bundle=capture_phase0_donor_bundle,
            capture_phase3_seed_bundle=capture_phase3_seed_bundle,
            capture_phase3_gradient_bundle=capture_phase3_gradient_bundle,
            capture_phase3_row_bundle=capture_phase3_row_bundle,
            capture_feature_semantic_descriptors=capture_feature_semantic_descriptors,
            semantic_descriptor_top_k=semantic_descriptor_top_k,
            semantic_descriptor_dim=semantic_descriptor_dim,
            telemetry_max_events=telemetry_max_events,
            telemetry_jsonl_path=telemetry_jsonl_path,
            telemetry_context=telemetry_context,
            compact_output=compact_output,
            phase0_donor_bundle=phase0_donor_bundle,
            phase0_replay_mode=phase0_replay_mode,
            phase0_donor_context_policy=phase0_donor_context_policy,
            phase3_gradient_donor_bundle=phase3_gradient_donor_bundle,
            phase3_gradient_replay_mode=phase3_gradient_replay_mode,
            phase3_row_donor_bundle=phase3_row_donor_bundle,
            phase3_row_replay_mode=phase3_row_replay_mode,
            phase3_replay_validation_policy=phase3_replay_validation_policy,
            phase4_scheduler_mode=phase4_scheduler_mode,
            phase4_scheduler_debug=phase4_scheduler_debug,
            phase4_scheduler_telemetry_detail=phase4_scheduler_telemetry_detail,
            phase4_refresh_optimization=phase4_refresh_optimization,
            phase4_refresh_prepared_chunk_cache_bytes=phase4_refresh_prepared_chunk_cache_bytes,
            phase4_refresh_active_row_accumulation=phase4_refresh_active_row_accumulation,
            phase4_row_executor=phase4_row_executor,
            phase4_row_reduction=phase4_row_reduction,
            phase1_trace_batch_policy=phase1_trace_batch_policy,
            phase1_trace_batch_size_max=phase1_trace_batch_size_max,
            phase4_refresh_policy=phase4_refresh_policy,
            phase4_refresh_interval_multiplier=phase4_refresh_interval_multiplier,
            phase4_ranker=phase4_ranker,
            row_store_cache_control=row_store_cache_control,
            row_store_temp_root_policy=row_store_temp_root_policy,
            row_store_temp_root=row_store_temp_root,
            row_store_preallocate=row_store_preallocate,
            exact_encoder_residency=exact_encoder_residency,
            exact_trace_internal_dtype=exact_trace_internal_dtype,
            phase3_frontier_buffer_relative_epsilon=phase3_frontier_buffer_relative_epsilon,
            phase3_frontier_buffer_max_extra=phase3_frontier_buffer_max_extra,
            phase4_frontier_buffer_relative_epsilon=phase4_frontier_buffer_relative_epsilon,
            phase4_frontier_buffer_max_extra_per_refresh=phase4_frontier_buffer_max_extra_per_refresh,
            phase4_frontier_buffer_max_extra_total=phase4_frontier_buffer_max_extra_total,
            phase0_activation_threshold_compare_mode=phase0_activation_threshold_compare_mode,
            prefix_view_metadata=normalized_prefix_view_metadata,
            output_position=output_position,
            decoder_chunk_cache=decoder_chunk_cache,
            decoder_cache_fingerprint=decoder_cache_fingerprint,
            phase0_context_override=_phase0_context_override,
            target_logits_override=_target_logits_override,
            target_logit_source=_target_logit_source,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


class FullSequenceWindowAttributionSession:
    """Small resource session for per-target full-sequence window attribution.

    When both reuse toggles are disabled, calls delegate to ``attribute()`` so
    existing behavior/resource handling remains the fallback.  Otherwise a
    max-prefix Phase-0 context is built lazily and per-target calls use causal
    prefix-view contexts and/or cached window logits.
    """

    def __init__(
        self,
        *,
        model: NNSightReplacementModel,
        full_token_ids: torch.Tensor | list[int],
        window_max_prefix_len: int,
        decoder_chunk_cache: object | None = None,
        decoder_cache_fingerprint: object | None = None,
        reuse_phase0_window_state: bool = False,
        reuse_target_logits: bool = False,
        reference_check_metadata: Mapping[str, Any] | None = None,
        **setup_kwargs: Any,
    ) -> None:
        self.model = model
        self.full_token_ids = (
            full_token_ids.detach().clone().to(dtype=torch.long).reshape(-1)
            if isinstance(full_token_ids, torch.Tensor)
            else torch.tensor([int(v) for v in full_token_ids], dtype=torch.long)
        )
        self.window_max_prefix_len = int(window_max_prefix_len)
        if self.window_max_prefix_len <= 0:
            raise ValueError("window_max_prefix_len must be > 0")
        if self.window_max_prefix_len > int(self.full_token_ids.numel()):
            raise ValueError("window_max_prefix_len exceeds full_token_ids length")
        self.decoder_chunk_cache = decoder_chunk_cache
        self.decoder_cache_fingerprint = decoder_cache_fingerprint
        self.reuse_phase0_window_state = bool(reuse_phase0_window_state)
        self.reuse_target_logits = bool(reuse_target_logits)
        self.reference_check_metadata = dict(reference_check_metadata or {})
        self.setup_kwargs = dict(setup_kwargs)
        self._window_context = None

    def _get_window_context(self):
        if self._window_context is None:
            self._window_context = self.model.setup_attribution(
                self.full_token_ids[: self.window_max_prefix_len],
                retain_full_logits=True,
                decoder_chunk_cache=self.decoder_chunk_cache,
                decoder_cache_fingerprint=self.decoder_cache_fingerprint,
                **self.setup_kwargs,
            )
        return self._window_context

    def attribute_target_position(
        self,
        target_position: int,
        *,
        attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
        prefix_view_metadata: Mapping[str, Any] | None = None,
        **attribute_kwargs: Any,
    ) -> Graph:
        target_position = int(target_position)
        if target_position <= 0 or target_position >= int(self.full_token_ids.numel()):
            raise ValueError("target_position must select a non-initial token in full_token_ids")
        if target_position > self.window_max_prefix_len:
            raise ValueError("target_position exceeds window_max_prefix_len")
        supplied_output_position = attribute_kwargs.pop("output_position", None)
        if (
            supplied_output_position is not None
            and int(supplied_output_position) != target_position - 1
        ):
            raise ValueError("output_position must equal target_position - 1")
        if prefix_view_metadata is not None:
            metadata_target_position = int(prefix_view_metadata.get("target_position", -1))
            if metadata_target_position != target_position:
                raise ValueError(
                    "prefix_view_metadata target_position must match session target_position"
                )
            if (
                self.reuse_phase0_window_state or self.reuse_target_logits
            ) and prefix_view_metadata.get("mode") != "full_sequence_target_position":
                raise ValueError(
                    "window reuse requires full_sequence_target_position prefix metadata"
                )
        if self.decoder_chunk_cache is not None:
            attribute_kwargs.setdefault("decoder_chunk_cache", self.decoder_chunk_cache)
            attribute_kwargs.setdefault("decoder_cache_fingerprint", self.decoder_cache_fingerprint)

        if not self.reuse_phase0_window_state and not self.reuse_target_logits:
            prompt_tokens = (
                self.full_token_ids
                if isinstance(prefix_view_metadata, Mapping)
                and prefix_view_metadata.get("mode") == "full_sequence_target_position"
                else self.full_token_ids[:target_position]
            )
            return attribute(
                prompt_tokens,
                self.model,
                attribution_targets=attribution_targets,
                prefix_view_metadata=prefix_view_metadata,
                output_position=target_position - 1,
                **attribute_kwargs,
            )

        window_ctx = self._get_window_context()
        phase0_override = None
        if self.reuse_phase0_window_state:
            derive = getattr(window_ctx, "derive_prefix_view_context", None)
            if not callable(derive):
                raise RuntimeError("AttributionContext does not support prefix-view derivation")
            phase0_override = derive(target_position)

        target_logits_override = None
        target_logit_source = None
        if self.reuse_target_logits:
            target_logits_override = window_ctx.get_logits_at_position(target_position - 1)[
                0
            ].detach()
            target_logit_source = "full_sequence_window_logits"

        return attribute(
            self.full_token_ids,
            self.model,
            attribution_targets=attribution_targets,
            prefix_view_metadata=prefix_view_metadata,
            output_position=target_position - 1,
            _phase0_context_override=phase0_override,
            _target_logits_override=target_logits_override,
            _target_logit_source=target_logit_source,
            **attribute_kwargs,
        )

    def cleanup(self) -> None:
        if self._window_context is not None:
            cleanup = getattr(self._window_context, "cleanup", None)
            if callable(cleanup):
                cleanup()
            self._window_context = None


def _run_attribution(
    model: NNSightReplacementModel,
    prompt,
    attribution_targets,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
    feature_batch_size: int | None,
    logit_batch_size: int | None,
    max_feature_nodes: int | None,
    offload: Literal["cpu", "disk", None],
    verbose: bool,
    offload_handles,
    logger,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    plan_feature_batch_size: bool = False,
    auto_scale_feature_batch_size: bool = False,
    feature_batch_size_max: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    internal_precision: Literal["float32", "float64"] | None = None,
    phase4_anomaly_debug: bool = False,
    cross_cluster_debug: bool = False,
    capture_phase0_donor_bundle: bool = False,
    capture_phase3_seed_bundle: bool = False,
    capture_phase3_gradient_bundle: bool = False,
    capture_phase3_row_bundle: bool = False,
    capture_feature_semantic_descriptors: bool = False,
    semantic_descriptor_top_k: int = 2048,
    semantic_descriptor_dim: int = 64,
    telemetry_max_events: int | None = None,
    telemetry_jsonl_path: str | os.PathLike[str] | None = None,
    telemetry_context: Mapping[str, object] | None = None,
    compact_output: bool = False,
    phase0_donor_bundle: str | os.PathLike[str] | None = None,
    phase0_replay_mode: Literal["disabled", "donor_phase0"] = "disabled",
    phase0_donor_context_policy: Literal["strict", "warn"] = "strict",
    phase3_gradient_donor_bundle: str | os.PathLike[str] | None = None,
    phase3_gradient_replay_mode: Literal["disabled", "donor"] = "disabled",
    phase3_row_donor_bundle: str | os.PathLike[str] | None = None,
    phase3_row_replay_mode: Literal["disabled", "donor"] = "disabled",
    phase3_replay_validation_policy: Literal["strict"] = "strict",
    phase4_scheduler_mode: Literal["locality", "planner_v1", "planner_v2", "legacy"] = "locality",
    phase4_scheduler_debug: bool = False,
    phase4_scheduler_telemetry_detail: Literal["summary", "normal", "debug"] = "normal",
    phase4_refresh_optimization: Literal["off", "v1"] = "v1",
    phase4_refresh_prepared_chunk_cache_bytes: int = 0,
    phase4_refresh_active_row_accumulation: Literal["zero_fill", "direct_v1"] = "direct_v1",
    phase4_row_executor: Literal["batched", "streaming_v1"] = "batched",
    phase4_row_reduction: Literal["off", "gpu_v1"] = "gpu_v1",
    phase1_trace_batch_policy: Literal["legacy", "cap_effective_batches"] = "legacy",
    phase1_trace_batch_size_max: int | None = None,
    phase4_refresh_policy: Literal["standard", "deferred_v1"] = "standard",
    phase4_refresh_interval_multiplier: int = 1,
    phase4_ranker: Literal["argsort", "topk_v1"] = "argsort",
    row_store_cache_control: _RowStoreCacheControlMode = "off",
    row_store_temp_root_policy: Literal["default", "env_node_local"] = "default",
    row_store_temp_root: str | os.PathLike[str] | None = None,
    row_store_preallocate: bool = True,
    exact_encoder_residency: Literal["lazy", "active_cpu", "active_pinned_cpu"] = "lazy",
    exact_trace_internal_dtype: Literal["fp32", "fp64"] = "fp32",
    phase3_frontier_buffer_relative_epsilon: float | None = None,
    phase3_frontier_buffer_max_extra: int = 0,
    phase4_frontier_buffer_relative_epsilon: float | None = None,
    phase4_frontier_buffer_max_extra_per_refresh: int = 0,
    phase4_frontier_buffer_max_extra_total: int = 0,
    phase0_activation_threshold_compare_mode: Literal[
        "baseline", "bf16", "fp32", "fp64"
    ] = "baseline",
    prefix_view_metadata: PrefixViewMetadata | None = None,
    output_position: int | None = None,
    decoder_chunk_cache: object | None = None,
    decoder_cache_fingerprint: object | None = None,
    phase0_context_override: object | None = None,
    target_logits_override: torch.Tensor | None = None,
    target_logit_source: str | None = None,
):
    start_time = time.time()
    run_start = time.perf_counter()
    exact_trace_internal_dtype_resolved = _resolve_exact_trace_internal_dtype(
        exact_trace_internal_dtype
    )
    exact_trace_internal_dtype_name = _exact_trace_internal_dtype_name(
        exact_trace_internal_dtype_resolved
    )
    phase0_activation_threshold_compare_mode_resolved = (
        _resolve_phase0_activation_threshold_compare_mode(phase0_activation_threshold_compare_mode)
    )
    row_store_temp_root_policy_resolved = _resolve_row_store_temp_root_policy(
        row_store_temp_root_policy
    )
    phase0_replay_mode_resolved = _resolve_phase0_replay_mode(phase0_replay_mode)
    phase0_donor_context_policy_resolved = _resolve_phase0_donor_context_policy(
        phase0_donor_context_policy
    )
    phase3_gradient_replay_mode_resolved = _resolve_phase3_replay_mode(phase3_gradient_replay_mode)
    phase3_row_replay_mode_resolved = _resolve_phase3_replay_mode(phase3_row_replay_mode)
    phase3_replay_validation_policy_resolved = _resolve_phase3_replay_validation_policy(
        phase3_replay_validation_policy
    )
    phase0_donor_bundle_path = (
        os.fspath(phase0_donor_bundle) if phase0_donor_bundle is not None else None
    )
    phase3_gradient_donor_bundle_path = (
        os.fspath(phase3_gradient_donor_bundle)
        if phase3_gradient_donor_bundle is not None
        else None
    )
    phase3_row_donor_bundle_path = (
        os.fspath(phase3_row_donor_bundle) if phase3_row_donor_bundle is not None else None
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if feature_batch_size is not None and feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be > 0 when provided")
    if logit_batch_size is not None and logit_batch_size <= 0:
        raise ValueError("logit_batch_size must be > 0 when provided")
    if chunked_feature_replay_window <= 0:
        raise ValueError("chunked_feature_replay_window must be > 0")
    if error_vector_prefetch_lookahead <= 0:
        raise ValueError("error_vector_prefetch_lookahead must be > 0")
    if row_subchunk_size is not None and row_subchunk_size <= 0:
        raise ValueError("row_subchunk_size must be > 0 when provided")
    if feature_batch_size_max is not None and feature_batch_size_max <= 0:
        raise ValueError("feature_batch_size_max must be > 0 when provided")
    if not 0.0 < feature_batch_target_reserved_fraction < 1.0:
        raise ValueError("feature_batch_target_reserved_fraction must be in (0, 1)")
    if not 0.0 <= feature_batch_min_free_fraction < 1.0:
        raise ValueError("feature_batch_min_free_fraction must be in [0, 1)")
    if feature_batch_probe_batches <= 0:
        raise ValueError("feature_batch_probe_batches must be > 0")
    if phase4_refresh_prepared_chunk_cache_bytes < 0:
        raise ValueError("phase4_refresh_prepared_chunk_cache_bytes must be >= 0")
    if phase3_frontier_buffer_max_extra < 0:
        raise ValueError("phase3_frontier_buffer_max_extra must be >= 0")
    if (
        phase3_frontier_buffer_relative_epsilon is not None
        and phase3_frontier_buffer_relative_epsilon < 0
    ):
        raise ValueError("phase3_frontier_buffer_relative_epsilon must be >= 0 when provided")
    if phase4_frontier_buffer_max_extra_per_refresh < 0:
        raise ValueError("phase4_frontier_buffer_max_extra_per_refresh must be >= 0")
    if phase4_frontier_buffer_max_extra_total < 0:
        raise ValueError("phase4_frontier_buffer_max_extra_total must be >= 0")
    if (
        phase4_frontier_buffer_relative_epsilon is not None
        and phase4_frontier_buffer_relative_epsilon < 0
    ):
        raise ValueError("phase4_frontier_buffer_relative_epsilon must be >= 0 when provided")
    if phase4_refresh_active_row_accumulation not in ("zero_fill", "direct_v1"):
        raise ValueError(
            "phase4_refresh_active_row_accumulation must be 'zero_fill' or 'direct_v1'"
        )
    if phase0_replay_mode_resolved == "disabled" and phase0_donor_bundle_path is not None:
        raise ValueError(
            "phase0_donor_bundle was provided but phase0_replay_mode is disabled; "
            "refusing potentially accidental replay configuration"
        )
    if phase0_replay_mode_resolved != "disabled" and not phase0_donor_bundle_path:
        raise ValueError("phase0_replay_mode requires a phase0_donor_bundle path")
    if phase3_gradient_replay_mode_resolved == "disabled" and phase3_gradient_donor_bundle_path:
        raise ValueError(
            "phase3_gradient_donor_bundle was provided but phase3_gradient_replay_mode is disabled"
        )
    if phase3_gradient_replay_mode_resolved != "disabled" and not phase3_gradient_donor_bundle_path:
        raise ValueError("phase3_gradient_replay_mode requires a phase3_gradient_donor_bundle path")
    if phase3_row_replay_mode_resolved == "disabled" and phase3_row_donor_bundle_path:
        raise ValueError(
            "phase3_row_donor_bundle was provided but phase3_row_replay_mode is disabled"
        )
    if phase3_row_replay_mode_resolved != "disabled" and not phase3_row_donor_bundle_path:
        raise ValueError("phase3_row_replay_mode requires a phase3_row_donor_bundle path")

    phase4_anomaly_debug_enabled = _resolve_phase4_anomaly_debug_enabled(phase4_anomaly_debug)
    if internal_precision is not None:
        _warn_internal_precision_deprecated()
    internal_precision_requested = _resolve_internal_precision_requested(
        internal_precision,
        exact_trace_internal_dtype=exact_trace_internal_dtype_resolved,
    )
    resolved_dtype_map = _resolve_internal_dtype_map(
        internal_precision_requested=internal_precision_requested,
        phase4_anomaly_debug_enabled=phase4_anomaly_debug_enabled,
    )
    transcoder_capabilities = get_transcoder_capabilities(model.transcoders)
    exact_chunked_provider_enabled = require_exact_chunked_provider(model.transcoders)
    supports_compact_row_store = bool(
        exact_chunked_provider_enabled and transcoder_capabilities.supports_compact_row_store
    )
    supports_decoder_chunk_cache = bool(
        exact_chunked_provider_enabled and transcoder_capabilities.supports_decoder_chunk_cache
    )
    supports_exact_encoder_residency = bool(
        exact_chunked_provider_enabled and transcoder_capabilities.supports_exact_encoder_residency
    )
    # Compatibility/debug breadcrumb for historical metadata and helper plumbing.
    exact_chunked_decoder = exact_chunked_provider_enabled
    use_compact_feature_row_store = compact_output and supports_compact_row_store
    feature_row_storage_dtype = _dtype_from_name(resolved_dtype_map["feature_row_storage_dtype"])
    row_abs_sum_dtype = _dtype_from_name(resolved_dtype_map["row_abs_sum_dtype"])
    influence_compute_dtype = _dtype_from_name(resolved_dtype_map["influence_compute_dtype"])
    planner_compute_dtype = _dtype_from_name(resolved_dtype_map["planner_compute_dtype"])
    shadow_debug_compute_dtype = _dtype_from_name(resolved_dtype_map["shadow_debug_compute_dtype"])
    cross_cluster_debug_enabled = bool(cross_cluster_debug)
    capture_phase0_donor_bundle_enabled = bool(capture_phase0_donor_bundle)
    capture_phase3_seed_bundle_enabled = bool(capture_phase3_seed_bundle)
    capture_phase3_gradient_bundle_enabled = bool(capture_phase3_gradient_bundle)
    capture_phase3_row_bundle_enabled = bool(capture_phase3_row_bundle)
    capture_feature_semantic_descriptors_enabled = bool(capture_feature_semantic_descriptors)
    semantic_descriptor_top_k = int(semantic_descriptor_top_k)
    semantic_descriptor_dim = int(semantic_descriptor_dim)
    if semantic_descriptor_top_k <= 0:
        raise ValueError("semantic_descriptor_top_k must be > 0")
    if semantic_descriptor_dim <= 0:
        raise ValueError("semantic_descriptor_dim must be > 0")
    phase4_scheduler_config = _resolve_phase4_scheduler_config(
        phase4_scheduler_mode=phase4_scheduler_mode,
        phase4_scheduler_debug=phase4_scheduler_debug,
        phase4_scheduler_telemetry_detail=phase4_scheduler_telemetry_detail,
    )
    phase4_scheduler_metadata = _build_phase4_scheduler_metadata(phase4_scheduler_config)
    phase4_refresh_optimization_config = _resolve_phase4_refresh_optimization_config(
        phase4_refresh_optimization,
        compact_output=compact_output,
        exact_chunked_provider_enabled=supports_compact_row_store,
    )
    phase4_refresh_optimization_metadata = _build_phase4_refresh_optimization_metadata(
        phase4_refresh_optimization_config
    )
    phase4_refresh_aux_applicable = bool(
        use_compact_feature_row_store and phase4_refresh_optimization_config.effective_mode == "v1"
    )
    phase4_refresh_prepared_chunk_cache_bytes_effective = (
        int(phase4_refresh_prepared_chunk_cache_bytes)
        if phase4_refresh_aux_applicable and supports_decoder_chunk_cache
        else 0
    )
    phase4_refresh_active_row_accumulation_effective = (
        phase4_refresh_active_row_accumulation if phase4_refresh_aux_applicable else "zero_fill"
    )
    phase4_refresh_aux_fallback_reason = None if phase4_refresh_aux_applicable else "not_applicable"
    phase4_refresh_optimization_metadata.update(
        {
            "refresh_prepared_chunk_cache_bytes_requested": int(
                phase4_refresh_prepared_chunk_cache_bytes
            ),
            "refresh_prepared_chunk_cache_bytes_effective": int(
                phase4_refresh_prepared_chunk_cache_bytes_effective
            ),
            "refresh_prepared_chunk_cache_enabled": bool(
                phase4_refresh_prepared_chunk_cache_bytes_effective > 0
            ),
            "refresh_active_row_accumulation_requested": phase4_refresh_active_row_accumulation,
            "refresh_active_row_accumulation_effective": (
                phase4_refresh_active_row_accumulation_effective
            ),
            "refresh_active_row_accumulation_fallback_reason": phase4_refresh_aux_fallback_reason,
            "refresh_active_row_accumulation_applicable": bool(phase4_refresh_aux_applicable),
        }
    )
    phase4_row_executor_config = _resolve_phase4_row_executor_config(
        phase4_row_executor,
        compact_output=compact_output,
        exact_chunked_provider_enabled=supports_compact_row_store,
    )
    phase4_row_executor_metadata = _build_phase4_row_executor_metadata(phase4_row_executor_config)
    phase4_row_reduction_config = _resolve_phase4_row_reduction_config(
        phase4_row_reduction,
        compact_output=compact_output,
        exact_chunked_provider_enabled=supports_compact_row_store,
    )
    phase4_row_reduction_metadata = _build_phase4_row_reduction_metadata(
        phase4_row_reduction_config
    )
    phase1_trace_batch_config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy=phase1_trace_batch_policy,
        phase1_trace_batch_size_max=phase1_trace_batch_size_max,
    )
    phase1_trace_batch_metadata = _build_phase1_trace_batch_metadata(phase1_trace_batch_config)
    phase1_trace_batch_sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=batch_size,
        feature_batch_size=feature_batch_size,
        logit_batch_size=logit_batch_size,
        feature_batch_size_max=feature_batch_size_max,
        phase1_trace_batch_config=phase1_trace_batch_config,
    )
    phase1_trace_batch_metadata.update(
        _build_phase1_trace_batch_sizing_metadata(phase1_trace_batch_sizing)
    )
    effective_source_batch_size = phase1_trace_batch_sizing.effective_source_batch_size
    effective_feature_batch_size = phase1_trace_batch_sizing.effective_feature_batch_size
    effective_logit_batch_size = phase1_trace_batch_sizing.effective_logit_batch_size
    max_phase4_feature_batch_size = (
        phase1_trace_batch_sizing.effective_phase4_max_feature_batch_size
    )
    phase4_refresh_policy_config = _resolve_phase4_refresh_policy_config(
        phase4_refresh_policy=phase4_refresh_policy,
        phase4_refresh_interval_multiplier=phase4_refresh_interval_multiplier,
        compact_output=compact_output,
        exact_chunked_provider_enabled=supports_compact_row_store,
    )
    phase4_refresh_policy_metadata = _build_phase4_refresh_policy_metadata(
        phase4_refresh_policy_config
    )
    phase4_ranker_config = _resolve_phase4_ranker_config(phase4_ranker)
    phase4_ranker_metadata = _build_phase4_ranker_metadata(phase4_ranker_config)
    row_store_cache_control_config = _resolve_row_store_cache_control_config(
        row_store_cache_control,
        compact_output=compact_output,
        supports_compact_row_store=supports_compact_row_store,
    )
    row_store_cache_control_metadata = _build_row_store_cache_control_metadata(
        row_store_cache_control_config
    )
    exact_encoder_residency_config = _resolve_exact_encoder_residency_config(
        exact_encoder_residency,
        supports_exact_encoder_residency=supports_exact_encoder_residency,
    )
    exact_encoder_residency_metadata = _build_exact_encoder_residency_metadata(
        exact_encoder_residency_config
    )
    phase4_execution_metadata: dict[str, object] = {
        **phase4_scheduler_metadata,
        **phase4_refresh_optimization_metadata,
        **phase4_row_executor_metadata,
        **phase4_row_reduction_metadata,
        **phase4_refresh_policy_metadata,
        **phase4_ranker_metadata,
        **row_store_cache_control_metadata,
        **exact_encoder_residency_metadata,
    }
    phase4_debug_summary_enabled = phase4_anomaly_debug_enabled or cross_cluster_debug_enabled
    telemetry_max_events_resolved = _resolve_telemetry_max_events(
        telemetry_max_events=telemetry_max_events,
        compact_output=compact_output,
        exact_chunked_decoder=exact_chunked_decoder,
        profile=profile,
        phase4_anomaly_debug_enabled=phase4_anomaly_debug_enabled,
    )
    telemetry_observer = TelemetryObserver.create(
        enabled=(profile or compact_output or phase4_anomaly_debug_enabled),
        max_events=telemetry_max_events_resolved,
        jsonl_path=telemetry_jsonl_path,
        static_context=telemetry_context,
    )
    telemetry_recorder = telemetry_observer.recorder
    telemetry_observer.run(
        name="attribute.start",
        attrs={
            "profile": profile,
            "compact_output": compact_output,
            "transcoder_architecture": transcoder_capabilities.architecture,
            "transcoder_checkpoint_format": transcoder_capabilities.checkpoint_format,
            "exact_chunked_provider_enabled": exact_chunked_provider_enabled,
            "supports_compact_row_store": supports_compact_row_store,
            "supports_decoder_chunk_cache": supports_decoder_chunk_cache,
            "supports_exact_encoder_residency": supports_exact_encoder_residency,
            "decoder_output_topology": transcoder_capabilities.decoder_output_topology,
            "batch_size": batch_size,
            "feature_batch_size": feature_batch_size,
            "logit_batch_size": logit_batch_size,
            "telemetry_max_events": telemetry_max_events_resolved,
            "exact_trace_internal_dtype": exact_trace_internal_dtype_name,
            "phase0_activation_threshold_compare_mode": (
                phase0_activation_threshold_compare_mode_resolved
            ),
            "phase0_replay_mode": phase0_replay_mode_resolved,
            "phase0_donor_bundle_supplied": bool(phase0_donor_bundle_path is not None),
            "phase0_donor_context_policy": phase0_donor_context_policy_resolved,
            "phase3_gradient_replay_mode": phase3_gradient_replay_mode_resolved,
            "phase3_gradient_donor_bundle_supplied": bool(
                phase3_gradient_donor_bundle_path is not None
            ),
            "phase3_row_replay_mode": phase3_row_replay_mode_resolved,
            "phase3_row_donor_bundle_supplied": bool(phase3_row_donor_bundle_path is not None),
            "phase3_replay_validation_policy": phase3_replay_validation_policy_resolved,
            "internal_precision_requested": internal_precision_requested,
            "resolved_dtype_map": resolved_dtype_map,
            "cross_cluster_debug_enabled": cross_cluster_debug_enabled,
            "capture_phase0_donor_bundle_enabled": capture_phase0_donor_bundle_enabled,
            "capture_phase3_seed_bundle_enabled": capture_phase3_seed_bundle_enabled,
            "capture_phase3_gradient_bundle_enabled": capture_phase3_gradient_bundle_enabled,
            "capture_phase3_row_bundle_enabled": capture_phase3_row_bundle_enabled,
            "capture_feature_semantic_descriptors_enabled": (
                capture_feature_semantic_descriptors_enabled
            ),
            "phase0_donor_bundle_schema_version": 1,
            "phase0_donor_bundle_replay_kind": "phase0_active_features_v1",
            "semantic_descriptor_top_k": semantic_descriptor_top_k,
            "semantic_descriptor_dim": semantic_descriptor_dim,
            "prefix_view_validation_applied": prefix_view_metadata is not None,
            "prefix_view_metadata": dict(prefix_view_metadata) if prefix_view_metadata else None,
            "phase0_window_state_reuse_requested": phase0_context_override is not None,
            "phase0_window_state_reuse_effective": phase0_context_override is not None,
            "target_logit_source": target_logit_source
            or ("override" if target_logits_override is not None else "context"),
            **{f"phase1_{key}": value for key, value in phase1_trace_batch_metadata.items()},
            **{f"phase4_{key}": value for key, value in phase4_execution_metadata.items()},
        },
    )

    planner_enabled = _resolve_phase4_feature_batch_planner_enabled(
        plan_feature_batch_size=plan_feature_batch_size,
        auto_scale_feature_batch_size=auto_scale_feature_batch_size,
    )
    if auto_scale_feature_batch_size and not plan_feature_batch_size:
        logger.info(
            "Phase-4 feature batch planning | "
            "legacy auto_scale_feature_batch_size flag detected; "
            "using fixed preflight planner semantics"
        )
    if (not planner_enabled) and max_phase4_feature_batch_size < effective_feature_batch_size:
        raise ValueError("feature_batch_size_max must be >= the effective feature batch size")

    planner_status, planner_skip_reason = _resolve_phase4_feature_batch_planner_status(
        planner_enabled=planner_enabled,
        effective_feature_batch_size=effective_feature_batch_size,
        max_feature_batch_size=max_phase4_feature_batch_size,
    )
    anomaly_debug_result: dict[str, object] | None = None
    cross_cluster_debug_summary: dict[str, object] | None = None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None = None
    cross_cluster_debug_batches: list[dict[str, object]] | None = None
    if phase4_anomaly_debug_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "Phase-4 anomaly debug requires compact_output=True and exact chunked provider support"
        )
    if cross_cluster_debug_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "cross_cluster_debug requires compact_output=True and exact chunked provider support"
        )
    if capture_phase0_donor_bundle_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "capture_phase0_donor_bundle requires compact_output=True and "
            "exact chunked provider support"
        )
    if capture_phase3_seed_bundle_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "capture_phase3_seed_bundle requires compact_output=True and exact chunked provider support"
        )
    if capture_phase3_gradient_bundle_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "capture_phase3_gradient_bundle requires compact_output=True and "
            "exact chunked provider support"
        )
    if capture_phase3_row_bundle_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "capture_phase3_row_bundle requires compact_output=True and exact chunked provider support"
        )
    if capture_feature_semantic_descriptors_enabled and not (
        compact_output and exact_chunked_decoder
    ):
        raise ValueError(
            "capture_feature_semantic_descriptors requires compact_output=True and "
            "exact chunked provider support"
        )
    if phase0_replay_mode_resolved != "disabled" and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "phase0 donor replay requires compact_output=True and exact chunked provider support"
        )
    if phase3_gradient_replay_mode_resolved != "disabled" and not (
        compact_output and exact_chunked_decoder
    ):
        raise ValueError(
            "Phase-3 gradient replay requires compact_output=True and exact chunked provider support"
        )
    if phase3_row_replay_mode_resolved != "disabled" and not (
        compact_output and exact_chunked_decoder
    ):
        raise ValueError(
            "Phase-3 row replay requires compact_output=True and exact chunked provider support"
        )
    if phase4_anomaly_debug_enabled:
        anomaly_debug_result = {
            "schema_version": 2,
            "enabled": True,
            "mode": "phase4_shadow_debug",
            "status": "scaffold",
            "shadow_execution": False,
            "refresh_count": 0,
            "environment": _build_phase4_environment_fingerprint(),
            "summary": {},
            "records": [],
        }
    if cross_cluster_debug_enabled:
        cross_cluster_debug_summary = {
            "schema_version": 1,
            "enabled": True,
            "status": "collecting",
            "mode": "early_phase_scalar_summary",
            "phase0_replay_mode": phase0_replay_mode_resolved,
            "phase0_donor_context_policy": phase0_donor_context_policy_resolved,
            "phase0_donor_bundle_basename": (
                os.path.basename(phase0_donor_bundle_path)
                if isinstance(phase0_donor_bundle_path, str)
                else None
            ),
            "phase3_gradient_replay_mode": phase3_gradient_replay_mode_resolved,
            "phase3_gradient_donor_bundle_basename": (
                os.path.basename(phase3_gradient_donor_bundle_path)
                if isinstance(phase3_gradient_donor_bundle_path, str)
                else None
            ),
            "phase3_row_replay_mode": phase3_row_replay_mode_resolved,
            "phase3_row_donor_bundle_basename": (
                os.path.basename(phase3_row_donor_bundle_path)
                if isinstance(phase3_row_donor_bundle_path, str)
                else None
            ),
            "phase3_replay_validation_policy": phase3_replay_validation_policy_resolved,
            "phase0_activation_threshold_compare_mode": (
                phase0_activation_threshold_compare_mode_resolved
            ),
            "internal_precision_requested": internal_precision_requested,
            "resolved_dtype_map": resolved_dtype_map,
            "phase1_trace_batch": phase1_trace_batch_metadata,
            "phase4_scheduler": phase4_scheduler_metadata,
            "phase4_execution": phase4_execution_metadata,
            "environment": _build_phase4_environment_fingerprint(),
            "checkpoints": {},
        }
        cross_cluster_debug_checkpoints = []
        cross_cluster_debug_batches = []
    if planner_enabled and not (compact_output and supports_compact_row_store):
        raise ValueError(
            "Phase-4 feature batch planner requires compact_output=True and compact row-store provider support"
        )
    if planner_enabled:
        if planner_status == "skipped_no_headroom":
            logger.info(
                "Phase-4 feature batch planner skipped | "
                f"status={planner_status} | "
                f"initial_feature_batch_size={effective_feature_batch_size} | "
                f"feature_batch_size_max={max_phase4_feature_batch_size} | "
                f"reason={planner_skip_reason}"
            )
            telemetry_recorder.record_event(
                scope="phase",
                name="phase4.planner.preflight",
                phase="phase4",
                attrs={
                    "planner_status": planner_status,
                    "planned_feature_batch_size": effective_feature_batch_size,
                    "planner_skip_reason": planner_skip_reason,
                },
            )
        else:
            planner_probe_feature_batch_size = min(
                effective_feature_batch_size,
                max_phase4_feature_batch_size,
            )
            effective_feature_batch_size = _plan_phase4_feature_batch_size_preflight(
                model=model,
                prompt=prompt,
                attribution_targets=attribution_targets,
                batch_size=effective_source_batch_size,
                initial_feature_batch_size=planner_probe_feature_batch_size,
                effective_logit_batch_size=effective_logit_batch_size,
                max_feature_batch_size=max_phase4_feature_batch_size,
                max_feature_nodes=max_feature_nodes,
                update_interval=update_interval,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                logger=logger,
                sparsification=sparsification,
                chunked_feature_replay_window=chunked_feature_replay_window,
                error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
                stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
                stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
                row_subchunk_size=row_subchunk_size,
                exact_encoder_residency=exact_encoder_residency_config.effective_mode,
                diagnostic_feature_cap=diagnostic_feature_cap,
                feature_batch_target_reserved_fraction=feature_batch_target_reserved_fraction,
                feature_batch_min_free_fraction=feature_batch_min_free_fraction,
                feature_batch_probe_batches=feature_batch_probe_batches,
                exact_trace_internal_dtype=exact_trace_internal_dtype_resolved,
                internal_precision_requested=internal_precision_requested,
                resolved_dtype_map=resolved_dtype_map,
                row_abs_sum_dtype=row_abs_sum_dtype,
                planner_compute_dtype=planner_compute_dtype,
                telemetry_recorder=telemetry_recorder,
                prefix_view_metadata=prefix_view_metadata,
            )
            planner_status = "executed"

    trace_batch_size = max(
        effective_source_batch_size,
        effective_feature_batch_size,
        effective_logit_batch_size,
    )
    phase1_trace_batch_metadata.update(
        trace_batch_size_legacy=int(phase1_trace_batch_sizing.trace_batch_size_legacy),
        trace_batch_size_effective=int(trace_batch_size),
    )
    ctx = None
    feature_row_store: _FileBackedFeatureRowStore | None = None
    nonfeature_row_store: _FileBackedFeatureRowStore | None = None
    phase2_resource_owner = Phase2ResourceOwner()
    compact_output_result: dict[str, object] | None = None
    phase0_donor_bundle_payload: dict[str, object] | None = None
    phase3_seed_bundle_payload: dict[str, object] | None = None
    phase3_gradient_bundle_payload: dict[str, object] | None = None
    phase3_row_bundle_payload: dict[str, object] | None = None
    feature_semantic_descriptors_payload: dict[str, object] | None = None
    phase0_replay_metadata: dict[str, object] = _build_phase0_replay_metadata(
        mode=phase0_replay_mode_resolved,
        status="disabled" if phase0_replay_mode_resolved == "disabled" else "pending",
        donor_bundle_path=phase0_donor_bundle_path,
        context_policy=phase0_donor_context_policy_resolved,
        replay_single_step_intended=True,
        note="single-step intended replay mode",
    )
    phase3_gradient_replay_metadata: dict[str, object] = _build_phase3_replay_metadata(
        replay_kind="phase3_gradient_replay_v1",
        mode=phase3_gradient_replay_mode_resolved,
        status="disabled" if phase3_gradient_replay_mode_resolved == "disabled" else "pending",
        donor_bundle_path=phase3_gradient_donor_bundle_path,
        validation_policy=phase3_replay_validation_policy_resolved,
        source="host_computed" if phase3_gradient_replay_mode_resolved == "disabled" else None,
    )
    phase3_row_replay_metadata: dict[str, object] = _build_phase3_replay_metadata(
        replay_kind="phase3_row_replay_v1",
        mode=phase3_row_replay_mode_resolved,
        status="disabled" if phase3_row_replay_mode_resolved == "disabled" else "pending",
        donor_bundle_path=phase3_row_donor_bundle_path,
        validation_policy=phase3_replay_validation_policy_resolved,
        source="host_computed" if phase3_row_replay_mode_resolved == "disabled" else None,
    )
    loaded_phase3_row_donor_bundle: dict[str, object] | None = None

    phase0_failure: BaseException | None = None
    try:
        phase0_result = run_phase0(
            inputs=Phase0Inputs(
                logger=logger,
                model=model,
                prompt=prompt,
                sparsification=sparsification,
                telemetry_observer=telemetry_observer,
                telemetry_recorder=telemetry_recorder,
                phase0_context_override=phase0_context_override,
                prefix_view_metadata=prefix_view_metadata,
                exact_encoder_residency_metadata=exact_encoder_residency_metadata,
                phase4_execution_metadata=phase4_execution_metadata,
                cross_cluster_debug_summary=(
                    cross_cluster_debug_summary if cross_cluster_debug_enabled else None
                ),
                cross_cluster_debug_checkpoints=(
                    cross_cluster_debug_checkpoints if cross_cluster_debug_enabled else []
                ),
                cleanup_owner=Phase0CleanupOwner(),
            ),
            config=Phase0Config(
                output_position=output_position,
                profile=profile,
                phase0_activation_threshold_compare_mode=phase0_activation_threshold_compare_mode_resolved,
                cross_cluster_debug_enabled=cross_cluster_debug_enabled,
                exact_chunked_provider_enabled=exact_chunked_provider_enabled,
                exact_chunked_decoder=exact_chunked_decoder,
                chunked_feature_replay_window=chunked_feature_replay_window,
                error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
                stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
                stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
                row_subchunk_size=row_subchunk_size,
                planner_enabled=planner_enabled,
                max_phase4_feature_batch_size=max_phase4_feature_batch_size,
                phase1_trace_batch_config=phase1_trace_batch_config,
                phase1_trace_batch_metadata=phase1_trace_batch_metadata,
                phase4_refresh_policy_config=phase4_refresh_policy_config,
                phase4_ranker_config=phase4_ranker_config,
                row_store_cache_control_config=row_store_cache_control_config,
                exact_encoder_residency_config=exact_encoder_residency_config,
                exact_trace_internal_dtype_name=exact_trace_internal_dtype_name,
                effective_source_batch_size=effective_source_batch_size,
                effective_feature_batch_size=effective_feature_batch_size,
                effective_logit_batch_size=effective_logit_batch_size,
                internal_precision_requested=internal_precision_requested,
                resolved_dtype_map=resolved_dtype_map,
                decoder_chunk_cache=decoder_chunk_cache,
                decoder_cache_fingerprint=decoder_cache_fingerprint,
                capture_phase3_gradient_bundle_enabled=capture_phase3_gradient_bundle_enabled,
                diagnostic_feature_cap=diagnostic_feature_cap,
            ),
        )
    except Phase0ExecutionError as exc:
        ctx = exc.ctx
        phase0_failure = exc.cause
    else:
        ctx = phase0_result.ctx
        input_ids = phase0_result.input_ids
        n_input_pos = phase0_result.n_input_pos
        output_position = phase0_result.output_position
        trace_input_ids = phase0_result.trace_input_ids
        activation_matrix = phase0_result.activation_matrix
    try:
        if phase0_failure is not None:
            raise phase0_failure
        if offload and not model.skip_transcoder and not exact_chunked_decoder:
            offload_handles += offload_modules(model.transcoders, offload)

        # Phase 1: forward pass
        _run_phase1_forward_pass(
            logger=logger,
            model=model,
            ctx=ctx,
            trace_input_ids=trace_input_ids,
            trace_batch_size=trace_batch_size,
            trace_batch_config=phase1_trace_batch_config,
            trace_batch_metadata=phase1_trace_batch_metadata,
            effective_source_batch_size=effective_source_batch_size,
            effective_feature_batch_size=effective_feature_batch_size,
            effective_logit_batch_size=effective_logit_batch_size,
            telemetry_observer=telemetry_observer,
        )

        if offload:
            offload_handles += offload_modules(
                [layer.mlp for layer in getattr(model.pre_logit_location, "layers")], offload
            )
            if model.skip_transcoder and not exact_chunked_decoder:
                offload_handles += offload_modules(model.transcoders, offload)

        phase2_result = run_phase2(
            inputs=Phase2Inputs(
                logger=logger,
                model=model,
                ctx=ctx,
                input_ids=input_ids,
                activation_matrix=activation_matrix,
                telemetry_observer=telemetry_observer,
                telemetry_recorder=telemetry_recorder,
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                offload_handles=offload_handles,
                attribution_targets=attribution_targets,
                target_logits_override=target_logits_override,
                resource_owner=phase2_resource_owner,
            ),
            config=Phase2Config(
                output_position=output_position,
                n_input_pos=n_input_pos,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                phase0_replay_mode_resolved=phase0_replay_mode_resolved,
                phase0_donor_bundle_path=phase0_donor_bundle_path,
                phase0_donor_context_policy_resolved=phase0_donor_context_policy_resolved,
                capture_phase0_donor_bundle_enabled=capture_phase0_donor_bundle_enabled,
                offload=offload,
                max_feature_nodes=max_feature_nodes,
                phase3_frontier_buffer_relative_epsilon=phase3_frontier_buffer_relative_epsilon,
                phase3_frontier_buffer_max_extra=phase3_frontier_buffer_max_extra,
                phase4_frontier_buffer_relative_epsilon=phase4_frontier_buffer_relative_epsilon,
                phase4_frontier_buffer_max_extra_per_refresh=phase4_frontier_buffer_max_extra_per_refresh,
                phase4_frontier_buffer_max_extra_total=phase4_frontier_buffer_max_extra_total,
                compact_output=compact_output,
                exact_chunked_decoder=exact_chunked_decoder,
                use_compact_feature_row_store=use_compact_feature_row_store,
                exact_trace_internal_dtype_resolved=exact_trace_internal_dtype_resolved,
                phase4_refresh_prepared_chunk_cache_bytes_effective=phase4_refresh_prepared_chunk_cache_bytes_effective,
                row_store_cache_control_config=row_store_cache_control_config,
                row_store_temp_root_policy_resolved=row_store_temp_root_policy_resolved,
                row_store_temp_root=row_store_temp_root,
                row_store_preallocate=row_store_preallocate,
                feature_row_storage_dtype=feature_row_storage_dtype,
                row_abs_sum_dtype=row_abs_sum_dtype,
                effective_feature_batch_size=effective_feature_batch_size,
                phase3_gradient_replay_mode_resolved=phase3_gradient_replay_mode_resolved,
                phase3_gradient_donor_bundle_path=phase3_gradient_donor_bundle_path,
                phase3_replay_validation_policy_resolved=phase3_replay_validation_policy_resolved,
                trace_batch_size=trace_batch_size,
                phase3_row_replay_mode_resolved=phase3_row_replay_mode_resolved,
                phase3_row_donor_bundle_path=phase3_row_donor_bundle_path,
            ),
        )
        targets = phase2_result.targets
        activation_matrix = phase2_result.activation_matrix
        feat_layers = phase2_result.feat_layers
        feat_pos = phase2_result.feat_pos
        feat_ids = phase2_result.feat_ids
        n_layers = phase2_result.n_layers
        n_pos = phase2_result.n_pos
        total_active_feats = phase2_result.total_active_feats
        logit_offset = phase2_result.logit_offset
        n_logits = phase2_result.n_logits
        total_nodes = phase2_result.total_nodes
        base_max_feature_nodes = phase2_result.base_max_feature_nodes
        actual_max_feature_nodes = phase2_result.actual_max_feature_nodes
        row_store_capacity_feature_nodes = phase2_result.row_store_capacity_feature_nodes
        feature_row_store = phase2_result.feature_row_store
        nonfeature_row_store = phase2_result.nonfeature_row_store
        edge_matrix = phase2_result.edge_matrix
        row_to_node_index = phase2_result.row_to_node_index
        phase0_donor_bundle_payload = phase2_result.phase0_donor_bundle_payload
        phase0_replay_metadata = phase2_result.phase0_replay_metadata
        phase3_frontier_buffer_metadata = phase2_result.phase3_frontier_buffer_metadata
        phase4_frontier_buffer_metadata = phase2_result.phase4_frontier_buffer_metadata
        phase3_gradient_replay_metadata = phase2_result.phase3_gradient_replay_metadata
        phase3_row_replay_metadata = phase2_result.phase3_row_replay_metadata
        loaded_phase3_row_donor_bundle = phase2_result.loaded_phase3_row_donor_bundle

        phase3_result = run_phase3(
            inputs=Phase3Inputs(
                logger=logger,
                model=model,
                ctx=ctx,
                targets=targets,
                activation_matrix=activation_matrix,
                feat_layers=feat_layers,
                feat_pos=feat_pos,
                feat_ids=feat_ids,
                feature_row_store=feature_row_store,
                nonfeature_row_store=nonfeature_row_store,
                edge_matrix=edge_matrix,
                row_to_node_index=row_to_node_index,
                telemetry_observer=telemetry_observer,
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                cross_cluster_debug_batches=cross_cluster_debug_batches,
                anomaly_debug_result=anomaly_debug_result,
                loaded_phase3_row_donor_bundle=loaded_phase3_row_donor_bundle,
                phase3_frontier_buffer_metadata=phase3_frontier_buffer_metadata,
                phase3_gradient_bundle_payload=phase3_gradient_bundle_payload,
                phase3_row_bundle_payload=phase3_row_bundle_payload,
                phase3_seed_bundle_payload=phase3_seed_bundle_payload,
                feature_semantic_descriptors_payload=feature_semantic_descriptors_payload,
            ),
            config=Phase3Config(
                effective_logit_batch_size=effective_logit_batch_size,
                effective_feature_batch_size=effective_feature_batch_size,
                output_position=output_position,
                n_layers=n_layers,
                n_pos=n_pos,
                n_logits=n_logits,
                logit_offset=logit_offset,
                total_active_feats=total_active_feats,
                base_max_feature_nodes=base_max_feature_nodes,
                actual_max_feature_nodes=actual_max_feature_nodes,
                exact_trace_internal_dtype_resolved=exact_trace_internal_dtype_resolved,
                phase3_gradient_replay_mode_resolved=phase3_gradient_replay_mode_resolved,
                phase3_row_replay_mode_resolved=phase3_row_replay_mode_resolved,
                capture_phase3_gradient_bundle_enabled=capture_phase3_gradient_bundle_enabled,
                capture_phase3_row_bundle_enabled=capture_phase3_row_bundle_enabled,
                capture_phase3_seed_bundle_enabled=capture_phase3_seed_bundle_enabled,
                capture_feature_semantic_descriptors_enabled=(
                    capture_feature_semantic_descriptors_enabled
                ),
                phase3_frontier_buffer_relative_epsilon=phase3_frontier_buffer_relative_epsilon,
                phase3_frontier_buffer_max_extra=phase3_frontier_buffer_max_extra,
                update_interval=update_interval,
                planner_compute_dtype=planner_compute_dtype,
                influence_compute_dtype=influence_compute_dtype,
                shadow_debug_compute_dtype=shadow_debug_compute_dtype,
                phase4_refresh_policy_config=phase4_refresh_policy_config,
                exact_chunked_decoder=exact_chunked_decoder,
                use_compact_feature_row_store=use_compact_feature_row_store,
                semantic_descriptor_top_k=semantic_descriptor_top_k,
                semantic_descriptor_dim=semantic_descriptor_dim,
                profile=profile,
                profile_log_interval=profile_log_interval,
            ),
        )
        row_to_node_index = phase3_result.row_to_node_index
        rows_cpu_staging = phase3_result.rows_cpu_staging
        actual_max_feature_nodes = phase3_result.actual_max_feature_nodes
        phase3_frontier_buffer_metadata = phase3_result.phase3_frontier_buffer_metadata
        phase3_gradient_bundle_payload = phase3_result.phase3_gradient_bundle_payload
        phase3_row_bundle_payload = phase3_result.phase3_row_bundle_payload
        phase3_seed_bundle_payload = phase3_result.phase3_seed_bundle_payload
        feature_semantic_descriptors_payload = (
            phase3_result.feature_semantic_descriptors_payload
        )
        anomaly_debug_result = phase3_result.anomaly_debug_result
        phase4_result = run_phase4(
            inputs=Phase4Inputs(
                logger=logger,
                model=model,
                ctx=ctx,
                targets=targets,
                edge_matrix=edge_matrix,
                feat_ids=feat_ids,
                feat_layers=feat_layers,
                feat_pos=feat_pos,
                feature_row_store=feature_row_store,
                nonfeature_row_store=nonfeature_row_store,
                row_to_node_index=row_to_node_index,
                telemetry_observer=telemetry_observer,
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                cross_cluster_debug_batches=cross_cluster_debug_batches,
                anomaly_debug_result=anomaly_debug_result,
                phase4_frontier_buffer_metadata=phase4_frontier_buffer_metadata,
                phase4_execution_metadata=phase4_execution_metadata,
                rows_cpu_staging=rows_cpu_staging,
            ),
            config=Phase4Config(
                actual_max_feature_nodes=actual_max_feature_nodes,
                total_active_feats=total_active_feats,
                n_logits=n_logits,
                logit_offset=logit_offset,
                effective_feature_batch_size=effective_feature_batch_size,
                max_phase4_feature_batch_size=max_phase4_feature_batch_size,
                update_interval=update_interval,
                row_store_capacity_feature_nodes=row_store_capacity_feature_nodes,
                exact_trace_internal_dtype_resolved=exact_trace_internal_dtype_resolved,
                influence_compute_dtype=influence_compute_dtype,
                shadow_debug_compute_dtype=shadow_debug_compute_dtype,
                exact_chunked_decoder=exact_chunked_decoder,
                use_compact_feature_row_store=use_compact_feature_row_store,
                planner_enabled=planner_enabled,
                planner_status=planner_status,
                planner_skip_reason=planner_skip_reason,
                phase4_debug_summary_enabled=phase4_debug_summary_enabled,
                cross_cluster_debug_enabled=cross_cluster_debug_enabled,
                phase4_frontier_buffer_relative_epsilon=phase4_frontier_buffer_relative_epsilon,
                phase4_frontier_buffer_max_extra_per_refresh=phase4_frontier_buffer_max_extra_per_refresh,
                phase4_frontier_buffer_max_extra_total=phase4_frontier_buffer_max_extra_total,
                phase4_refresh_prepared_chunk_cache_bytes_effective=phase4_refresh_prepared_chunk_cache_bytes_effective,
                phase4_refresh_active_row_accumulation_effective=phase4_refresh_active_row_accumulation_effective,
                phase4_scheduler_config=phase4_scheduler_config,
                phase4_refresh_optimization_config=phase4_refresh_optimization_config,
                phase4_refresh_policy_config=phase4_refresh_policy_config,
                phase4_ranker_config=phase4_ranker_config,
                phase4_row_executor_config=phase4_row_executor_config,
                phase4_row_reduction_config=phase4_row_reduction_config,
                row_store_cache_control_config=row_store_cache_control_config,
                exact_encoder_residency_config=exact_encoder_residency_config,
                profile=profile,
                profile_log_interval=profile_log_interval,
                verbose=verbose,
            ),
        )
        visited = phase4_result.visited
        actual_max_feature_nodes = phase4_result.actual_max_feature_nodes
        edge_matrix = phase4_result.edge_matrix
        feature_row_store = phase4_result.feature_row_store
        nonfeature_row_store = phase4_result.nonfeature_row_store
        row_to_node_index = phase4_result.row_to_node_index
        rows_cpu_staging = phase4_result.rows_cpu_staging
        st = phase4_result.st
        phase4_frontier_buffer_metadata = phase4_result.phase4_frontier_buffer_metadata
        phase4_execution_metadata = phase4_result.phase4_execution_metadata
        cross_cluster_debug_summary = phase4_result.cross_cluster_debug_summary
        cross_cluster_debug_checkpoints = phase4_result.cross_cluster_debug_checkpoints
        cross_cluster_debug_batches = phase4_result.cross_cluster_debug_batches
        anomaly_debug_result = phase4_result.anomaly_debug_result
        phase4_elapsed_ms = phase4_result.phase4_elapsed_ms
        phase4_feature_batch_size = phase4_result.phase4_feature_batch_size
        phase4_executor_reference_batch_size = phase4_result.phase4_executor_reference_batch_size
        phase4_executor_microbatch_size = phase4_result.phase4_executor_microbatch_size
        phase4_refresh_count = phase4_result.phase4_refresh_count
        phase4_scheduler_reference_batch_count = phase4_result.phase4_scheduler_reference_batch_count
        phase4_executor_microbatch_count = phase4_result.phase4_executor_microbatch_count
        phase4_refresh_elapsed_ms_total = phase4_result.phase4_refresh_elapsed_ms_total
        phase4_feature_batch_elapsed_ms_total = phase4_result.phase4_feature_batch_elapsed_ms_total
        phase4_refresh_partial_influence_elapsed_ms_total = phase4_result.phase4_refresh_partial_influence_elapsed_ms_total
        phase4_refresh_rank_topk_elapsed_ms_total = phase4_result.phase4_refresh_rank_topk_elapsed_ms_total
        phase4_refresh_frontier_plan_elapsed_ms_total = phase4_result.phase4_refresh_frontier_plan_elapsed_ms_total
        phase4_refresh_row_store_read_elapsed_ms_total = phase4_result.phase4_refresh_row_store_read_elapsed_ms_total
        # Phase 5: packaging graph / compact output
        phase5_start = time.perf_counter()
        selected_features = torch.where(visited)[0]
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            selected_features = _compact_selected_feature_columns(
                selected_features,
                n_feature_columns=feature_row_store.n_feature_columns,
            )
        if capture_feature_semantic_descriptors_enabled and isinstance(
            feature_semantic_descriptors_payload, dict
        ):
            _annotate_phase4_selection_on_feature_semantic_descriptors(
                feature_semantic_descriptors_payload,
                selected_features=selected_features,
            )
        selected_features_cpu = (
            selected_features.detach().to(device="cpu", dtype=torch.long)
            if compact_output
            else None
        )
        if compact_output:
            active_features_cpu = activation_matrix.indices().T.detach().cpu()
            n_active_features = int(active_features_cpu.shape[0])
            compact_token_count = int(n_pos)
            n_error_nodes, _n_token_nodes, n_nonfeature_nodes = _compact_nonfeature_column_counts(
                n_layers=int(model.cfg.n_layers),
                compact_token_count=compact_token_count,
            )
            error_col_start = n_active_features
            token_col_start = error_col_start + n_error_nodes
            logit_col_start = token_col_start + compact_token_count
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                assert nonfeature_row_store is not None
                assert selected_features_cpu is not None
                feature_feature_edges = feature_row_store.materialize_dense_feature_slice(
                    row_start=n_logits,
                    row_end=st,
                    selected_feature_columns=selected_features_cpu,
                    phase="phase5",
                )
                logit_feature_edges = feature_row_store.materialize_dense_feature_slice(
                    row_start=0,
                    row_end=n_logits,
                    selected_feature_columns=selected_features_cpu,
                    phase="phase5",
                )
                if int(nonfeature_row_store.n_feature_columns) != int(n_nonfeature_nodes):
                    raise ValueError(
                        "compact nonfeature row-store width does not match "
                        "prefix-visible error/token column count"
                    )
                nonfeature_columns = torch.arange(n_nonfeature_nodes, dtype=torch.long)
                feature_nonfeature_edges = nonfeature_row_store.materialize_dense_feature_slice(
                    row_start=n_logits,
                    row_end=st,
                    selected_feature_columns=nonfeature_columns,
                    phase="phase5",
                )
                logit_nonfeature_edges = nonfeature_row_store.materialize_dense_feature_slice(
                    row_start=0,
                    row_end=n_logits,
                    selected_feature_columns=nonfeature_columns,
                    phase="phase5",
                )
                n_error_columns = int(token_col_start - error_col_start)
                feature_error_edges = feature_nonfeature_edges[:, :n_error_columns]
                feature_token_edges = feature_nonfeature_edges[:, n_error_columns:]
                logit_error_edges = logit_nonfeature_edges[:, :n_error_columns]
                logit_token_edges = logit_nonfeature_edges[:, n_error_columns:]
            else:
                feature_feature_edges = edge_matrix[n_logits:st, selected_features].detach().cpu()
                logit_feature_edges = edge_matrix[:n_logits, selected_features].detach().cpu()
                feature_error_edges = (
                    edge_matrix[n_logits:st, error_col_start:token_col_start].detach().cpu()
                )
                feature_token_edges = (
                    edge_matrix[n_logits:st, token_col_start:logit_col_start].detach().cpu()
                )
                logit_error_edges = (
                    edge_matrix[:n_logits, error_col_start:token_col_start].detach().cpu()
                )
                logit_token_edges = (
                    edge_matrix[:n_logits, token_col_start:logit_col_start].detach().cpu()
                )

            assert selected_features_cpu is not None
            compact_output_result = {
                "input_string": model.tokenizer.decode(input_ids),
                "input_tokens": input_ids[:n_pos].detach().cpu(),
                "full_input_tokens": input_ids.detach().cpu(),
                "logit_targets": targets.logit_targets,
                "logit_probabilities": targets.logit_probabilities.detach().cpu(),
                "vocab_size": targets.vocab_size,
                "active_features": active_features_cpu,
                "activation_values": activation_matrix.values().detach().cpu(),
                "selected_features": selected_features_cpu,
                "feature_row_node_indices": row_to_node_index[n_logits:st].detach().cpu(),
                "logit_row_node_indices": row_to_node_index[:n_logits].detach().cpu(),
                "feature_feature_edges": feature_feature_edges,
                "logit_feature_edges": logit_feature_edges,
                "feature_error_edges": feature_error_edges,
                "feature_token_edges": feature_token_edges,
                "logit_error_edges": logit_error_edges,
                "logit_token_edges": logit_token_edges,
                "n_error_nodes": n_error_nodes,
                "n_token_nodes": int(n_pos),
                "phase4_feature_batch_size": int(phase4_feature_batch_size),
                "phase4_feature_batch_size_initial": int(
                    batch_size if feature_batch_size is None else feature_batch_size
                ),
                "phase4_feature_batch_size_max": int(max_phase4_feature_batch_size),
                "phase4_feature_batch_planner_enabled": bool(planner_enabled),
                "phase4_feature_batch_planner_status": planner_status,
                "phase4_feature_batch_planner_skip_reason": planner_skip_reason,
                "phase4_scheduler_requested_mode": phase4_scheduler_config.requested_mode,
                "phase4_scheduler_mode": phase4_scheduler_config.requested_mode,
                "phase4_scheduler_mode_requested": phase4_scheduler_config.requested_mode,
                "phase4_scheduler_version": phase4_scheduler_config.version,
                "phase4_scheduler_version_requested": phase4_scheduler_config.version,
                "phase4_scheduler_policy": phase4_scheduler_config.policy,
                "phase4_scheduler_policy_requested": phase4_scheduler_config.policy,
                "phase4_scheduler_effective_mode": phase4_scheduler_config.effective_mode,
                "phase4_scheduler_mode_effective": phase4_scheduler_config.effective_mode,
                "phase4_scheduler_effective_version": phase4_scheduler_config.effective_version,
                "phase4_scheduler_version_effective": phase4_scheduler_config.effective_version,
                "phase4_scheduler_effective_policy": phase4_scheduler_config.effective_policy,
                "phase4_scheduler_policy_effective": phase4_scheduler_config.effective_policy,
                "phase4_scheduler_effective_behavior": phase4_scheduler_config.effective_behavior,
                "phase4_scheduler_reference_execution": bool(
                    phase4_scheduler_config.requested_mode != phase4_scheduler_config.effective_mode
                ),
                "phase4_scheduler_debug": bool(phase4_scheduler_config.debug),
                "phase4_scheduler_telemetry_detail": phase4_scheduler_config.telemetry_detail,
                "phase4_refresh_optimization_requested": phase4_refresh_optimization_config.requested_mode,
                "phase4_refresh_optimization": phase4_refresh_optimization_config.requested_mode,
                "phase4_refresh_optimization_mode_requested": phase4_refresh_optimization_config.requested_mode,
                "phase4_refresh_optimization_effective": phase4_refresh_optimization_config.effective_mode,
                "phase4_refresh_optimization_mode_effective": phase4_refresh_optimization_config.effective_mode,
                "phase4_refresh_optimization_version": phase4_refresh_optimization_config.version,
                "phase4_refresh_optimization_version_requested": phase4_refresh_optimization_config.version,
                "phase4_refresh_optimization_effective_version": phase4_refresh_optimization_config.effective_version,
                "phase4_refresh_optimization_version_effective": phase4_refresh_optimization_config.effective_version,
                "phase4_refresh_optimization_effective_behavior": phase4_refresh_optimization_config.effective_behavior,
                "phase4_refresh_optimization_reference_execution": bool(
                    phase4_refresh_optimization_config.requested_mode
                    != phase4_refresh_optimization_config.effective_mode
                ),
                "phase4_refresh_prepared_chunk_cache_bytes_requested": int(
                    phase4_refresh_prepared_chunk_cache_bytes
                ),
                "phase4_refresh_prepared_chunk_cache_bytes_effective": int(
                    phase4_refresh_prepared_chunk_cache_bytes_effective
                ),
                "phase4_refresh_prepared_chunk_cache_enabled": bool(
                    phase4_refresh_prepared_chunk_cache_bytes_effective > 0
                ),
                "phase4_refresh_active_row_accumulation_requested": phase4_refresh_active_row_accumulation,
                "phase4_refresh_active_row_accumulation_effective": phase4_refresh_active_row_accumulation_effective,
                "phase4_refresh_active_row_accumulation_fallback_reason": phase4_refresh_aux_fallback_reason,
                "phase4_refresh_active_row_accumulation_applicable": bool(
                    phase4_refresh_aux_applicable
                ),
                "phase4_row_executor_requested": phase4_row_executor_config.requested_mode,
                "phase4_row_executor": phase4_row_executor_config.requested_mode,
                "phase4_row_executor_mode_requested": phase4_row_executor_config.requested_mode,
                "phase4_row_executor_effective": phase4_row_executor_config.effective_mode,
                "phase4_row_executor_mode_effective": phase4_row_executor_config.effective_mode,
                "phase4_row_executor_version": phase4_row_executor_config.version,
                "phase4_row_executor_version_requested": phase4_row_executor_config.version,
                "phase4_row_executor_effective_version": phase4_row_executor_config.effective_version,
                "phase4_row_executor_version_effective": phase4_row_executor_config.effective_version,
                "phase4_row_executor_effective_behavior": phase4_row_executor_config.effective_behavior,
                "phase4_row_executor_reference_execution": bool(
                    phase4_row_executor_config.requested_mode
                    != phase4_row_executor_config.effective_mode
                ),
                "phase4_row_reduction_requested": phase4_row_reduction_config.requested_mode,
                "phase4_row_reduction": phase4_row_reduction_config.requested_mode,
                "phase4_row_reduction_mode_requested": phase4_row_reduction_config.requested_mode,
                "phase4_row_reduction_effective": phase4_row_reduction_config.effective_mode,
                "phase4_row_reduction_mode_effective": phase4_row_reduction_config.effective_mode,
                "phase4_row_reduction_version": phase4_row_reduction_config.version,
                "phase4_row_reduction_version_requested": phase4_row_reduction_config.version,
                "phase4_row_reduction_effective_version": phase4_row_reduction_config.effective_version,
                "phase4_row_reduction_version_effective": phase4_row_reduction_config.effective_version,
                "phase4_row_reduction_effective_behavior": phase4_row_reduction_config.effective_behavior,
                "phase4_row_reduction_reference_execution": bool(
                    phase4_row_reduction_config.requested_mode
                    != phase4_row_reduction_config.effective_mode
                ),
                **{f"phase1_{key}": value for key, value in phase1_trace_batch_metadata.items()},
                "phase4_executor_configured_reference_batch_size": int(
                    phase4_executor_reference_batch_size
                ),
                "phase4_executor_reference_batch_size": int(phase4_executor_reference_batch_size),
                "phase4_executor_microbatch_size": int(phase4_executor_microbatch_size),
                "internal_precision_requested": internal_precision_requested,
                "resolved_dtype_map": resolved_dtype_map,
                "phase4_anomaly_debug_enabled": bool(phase4_anomaly_debug_enabled),
                "cross_cluster_debug_enabled": bool(cross_cluster_debug_enabled),
                "phase0_replay_mode": phase0_replay_metadata.get("mode"),
                "phase0_replay_status": phase0_replay_metadata.get("status"),
                "phase0_replay_context_policy": phase0_replay_metadata.get("context_policy"),
                "phase0_replay_donor_bundle_path": phase0_replay_metadata.get("donor_bundle_path"),
                "phase0_replay_donor_bundle_basename": phase0_replay_metadata.get(
                    "donor_bundle_basename"
                ),
                "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                    "validation_warning_count"
                ),
                "phase0_replay_validation_warnings": phase0_replay_metadata.get(
                    "validation_warnings"
                ),
                "phase0_replay_dtype_roundtrip_loss": cast(
                    dict[str, object],
                    phase0_replay_metadata.get("dtype_metadata", {}),
                ).get("dtype_roundtrip_loss"),
                "phase3_gradient_replay_mode": phase3_gradient_replay_metadata.get("mode"),
                "phase3_gradient_replay_status": phase3_gradient_replay_metadata.get("status"),
                "phase3_gradient_replay_donor_bundle_path": phase3_gradient_replay_metadata.get(
                    "donor_bundle_path"
                ),
                "phase3_gradient_replay_donor_bundle_basename": (
                    phase3_gradient_replay_metadata.get("donor_bundle_basename")
                ),
                "phase3_gradient_replay_validation_failure_count": (
                    phase3_gradient_replay_metadata.get("validation_failure_count")
                ),
                "phase3_gradient_replay_error": phase3_gradient_replay_metadata.get("error"),
                "phase3_row_replay_mode": phase3_row_replay_metadata.get("mode"),
                "phase3_row_replay_status": phase3_row_replay_metadata.get("status"),
                "phase3_row_replay_donor_bundle_path": phase3_row_replay_metadata.get(
                    "donor_bundle_path"
                ),
                "phase3_row_replay_donor_bundle_basename": phase3_row_replay_metadata.get(
                    "donor_bundle_basename"
                ),
                "phase3_row_replay_validation_failure_count": phase3_row_replay_metadata.get(
                    "validation_failure_count"
                ),
                "phase3_row_replay_error": phase3_row_replay_metadata.get("error"),
                "phase3_row_replay_source": phase3_row_replay_metadata.get("source"),
                "capture_phase0_donor_bundle_enabled": bool(capture_phase0_donor_bundle_enabled),
                "capture_phase3_seed_bundle_enabled": bool(capture_phase3_seed_bundle_enabled),
                "capture_phase3_gradient_bundle_enabled": bool(
                    capture_phase3_gradient_bundle_enabled
                ),
                "capture_phase3_row_bundle_enabled": bool(capture_phase3_row_bundle_enabled),
                "capture_feature_semantic_descriptors_enabled": bool(
                    capture_feature_semantic_descriptors_enabled
                ),
                "phase0_donor_bundle_schema_version": (
                    int(phase0_donor_bundle_payload.get("schema_version", 1))
                    if isinstance(phase0_donor_bundle_payload, dict)
                    else 1
                ),
                "phase0_donor_bundle_replay_kind": (
                    str(phase0_donor_bundle_payload.get("replay_kind", "phase0_active_features_v1"))
                    if isinstance(phase0_donor_bundle_payload, dict)
                    else "phase0_active_features_v1"
                ),
                "phase0_donor_bundle_status": (
                    str(phase0_donor_bundle_payload.get("status", "captured"))
                    if isinstance(phase0_donor_bundle_payload, dict)
                    else None
                ),
                "semantic_descriptor_top_k": int(semantic_descriptor_top_k),
                "semantic_descriptor_dim": int(semantic_descriptor_dim),
                "phase4_refresh_count": int(phase4_refresh_count),
                "phase3_frontier_buffer_metadata": phase3_frontier_buffer_metadata,
                "phase4_frontier_buffer_metadata": phase4_frontier_buffer_metadata,
                "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                "phase4_batches": int(phase4_scheduler_reference_batch_count),
                "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                "phase4_refresh_elapsed_seconds_total": round(
                    phase4_refresh_elapsed_ms_total / 1000.0,
                    6,
                ),
                "phase4_feature_batch_elapsed_seconds_total": round(
                    phase4_feature_batch_elapsed_ms_total / 1000.0,
                    6,
                ),
                "phase4_refresh_partial_influence_elapsed_seconds_total": round(
                    phase4_refresh_partial_influence_elapsed_ms_total / 1000.0,
                    6,
                ),
                "phase4_refresh_rank_topk_elapsed_seconds_total": round(
                    phase4_refresh_rank_topk_elapsed_ms_total / 1000.0,
                    6,
                ),
                "phase4_refresh_frontier_plan_elapsed_seconds_total": round(
                    phase4_refresh_frontier_plan_elapsed_ms_total / 1000.0,
                    6,
                ),
                "phase4_refresh_row_store_read_elapsed_seconds_total": round(
                    phase4_refresh_row_store_read_elapsed_ms_total / 1000.0,
                    6,
                ),
                "exact_trace_internal_dtype": exact_trace_internal_dtype_name,
                "phase0_activation_threshold_compare_mode": (
                    phase0_activation_threshold_compare_mode_resolved
                ),
                "telemetry_max_events": int(telemetry_max_events_resolved),
                "cfg": model.config,
                "scan": model.scan,
            }
            compact_output_result["phase0_replay_metadata"] = phase0_replay_metadata
            compact_output_result["phase3_gradient_replay_metadata"] = (
                phase3_gradient_replay_metadata
            )
            compact_output_result["phase3_row_replay_metadata"] = phase3_row_replay_metadata
            compact_output_result["phase3_frontier_buffer_metadata"] = (
                phase3_frontier_buffer_metadata
            )
            compact_output_result["phase4_frontier_buffer_metadata"] = (
                phase4_frontier_buffer_metadata
            )
            if capture_phase0_donor_bundle_enabled:
                compact_output_result["phase0_donor_bundle"] = phase0_donor_bundle_payload
            if capture_phase3_seed_bundle_enabled:
                compact_output_result["phase3_seed_bundle"] = phase3_seed_bundle_payload
            if capture_phase3_gradient_bundle_enabled:
                compact_output_result["phase3_gradient_bundle"] = phase3_gradient_bundle_payload
            if capture_phase3_row_bundle_enabled:
                compact_output_result["phase3_row_bundle"] = phase3_row_bundle_payload
            if capture_feature_semantic_descriptors_enabled:
                compact_output_result["feature_semantic_descriptors"] = (
                    feature_semantic_descriptors_payload
                )
            if cross_cluster_debug_summary is not None:
                cross_cluster_debug_summary["status"] = "captured"
                phase4_runtime_summary, phase4_runtime_stream = (
                    _build_cross_cluster_runtime_snapshot(
                        device=model.device,
                        ctx=ctx,
                        transcoder=model.transcoders,
                    )
                )
                phase4_entry_summary_checkpoint = {
                    "phase4_refresh_count": int(phase4_refresh_count),
                    "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                    "phase4_batches": int(phase4_scheduler_reference_batch_count),
                    "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                    **phase4_execution_metadata,
                    **phase4_runtime_summary,
                }
                _record_cross_cluster_checkpoint(
                    cross_cluster_debug_summary=cross_cluster_debug_summary,
                    cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                    checkpoint_name="phase4_entry",
                    phase="phase4",
                    summary_payload=phase4_entry_summary_checkpoint,
                    stream_payload={
                        "checkpoint_stage": "post_phase4",
                        "phase4_refresh_count": int(phase4_refresh_count),
                        "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                        "phase4_batches": int(phase4_scheduler_reference_batch_count),
                        "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                        **phase4_execution_metadata,
                        **phase4_runtime_stream,
                    },
                )
                _record_cross_cluster_checkpoint(
                    cross_cluster_debug_summary=cross_cluster_debug_summary,
                    cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                    checkpoint_name="phase4_run_summary",
                    phase="phase4",
                    summary_payload=None,
                    stream_payload={
                        "selected_feature_count": int(visited.sum().item()),
                        "phase4_feature_batch_size": int(phase4_feature_batch_size),
                        "phase4_refresh_count": int(phase4_refresh_count),
                        "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                        "phase4_batches": int(phase4_scheduler_reference_batch_count),
                        "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                        "phase4_elapsed_ms": float(phase4_elapsed_ms),
                        "phase4_refresh_elapsed_ms_total": float(phase4_refresh_elapsed_ms_total),
                        "phase4_feature_batch_elapsed_ms_total": float(
                            phase4_feature_batch_elapsed_ms_total
                        ),
                        "phase4_refresh_partial_influence_elapsed_ms_total": float(
                            phase4_refresh_partial_influence_elapsed_ms_total
                        ),
                        "phase4_refresh_rank_topk_elapsed_ms_total": float(
                            phase4_refresh_rank_topk_elapsed_ms_total
                        ),
                        "phase4_refresh_frontier_plan_elapsed_ms_total": float(
                            phase4_refresh_frontier_plan_elapsed_ms_total
                        ),
                        **phase4_execution_metadata,
                        **phase4_runtime_stream,
                    },
                )
                cross_cluster_debug_summary["checkpoint_stream_count"] = int(
                    len(cross_cluster_debug_checkpoints or [])
                )
                cross_cluster_debug_summary["batch_event_stream_count"] = int(
                    len(cross_cluster_debug_batches or [])
                )
                compact_output_result["cross_cluster_debug_summary"] = cross_cluster_debug_summary
            if cross_cluster_debug_checkpoints is not None:
                compact_output_result["cross_cluster_debug_checkpoints"] = (
                    cross_cluster_debug_checkpoints
                )
            if cross_cluster_debug_batches is not None:
                compact_output_result["cross_cluster_debug_batches"] = cross_cluster_debug_batches
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                file_backed_store_bytes = feature_row_store.nbytes
            else:
                del edge_matrix
                file_backed_store_bytes = None
            logger.info(
                "Attribution completed in "
                f"{time.time() - start_time:.2f}s | "
                f"compact_feature_edge_shape={tuple(compact_output_result['feature_feature_edges'].shape)} | "
                f"compact_logit_edge_shape={tuple(compact_output_result['logit_feature_edges'].shape)}"
                + (
                    f" | feature_row_store_bytes={file_backed_store_bytes}"
                    if file_backed_store_bytes is not None
                    else ""
                )
            )
            phase5_elapsed_ms = (time.perf_counter() - phase5_start) * 1000.0
            telemetry_observer.phase(
                name="phase5.packaging",
                phase="phase5",
                elapsed_ms=phase5_elapsed_ms,
                attrs={
                    "compact_output": True,
                    "selected_features": int(selected_features.numel()),
                    "feature_edge_rows": int(
                        compact_output_result["feature_feature_edges"].shape[0]
                    ),
                    "feature_edge_cols": int(
                        compact_output_result["feature_feature_edges"].shape[1]
                    ),
                },
                wall_clock=True,
            )
            if prefix_view_metadata is not None:
                compact_output_result["prefix_view_metadata"] = dict(prefix_view_metadata)
                validate_compact_prefix_view_output(
                    compact_output_result, n_layers=int(model.cfg.n_layers)
                )
            compact_output_result["phase0_window_state_reuse_requested"] = (
                phase0_context_override is not None
            )
            compact_output_result["phase0_window_state_reuse_effective"] = (
                phase0_context_override is not None
            )
            compact_output_result["target_logit_source"] = target_logit_source or (
                "override" if target_logits_override is not None else "context"
            )
            return compact_output_result

        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        if actual_max_feature_nodes < total_active_feats:
            col_read = torch.cat([selected_features, non_feature_nodes])
        else:
            col_read = torch.arange(total_nodes)

        final_node_count = len(col_read)
        full_edge_matrix = torch.zeros(final_node_count, final_node_count, dtype=edge_matrix.dtype)
        feature_row_order = row_to_node_index[n_logits:st].argsort()
        full_edge_matrix[:actual_max_feature_nodes] = edge_matrix[n_logits:st][feature_row_order][
            :, col_read
        ]
        full_edge_matrix[-n_logits:] = edge_matrix[:n_logits, :][:, col_read]

        graph = Graph(
            input_string=model.tokenizer.decode(input_ids[:n_pos]),
            input_tokens=input_ids[:n_pos],
            logit_targets=targets.logit_targets,
            logit_probabilities=targets.logit_probabilities,
            vocab_size=targets.vocab_size,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            selected_features=selected_features,
            adjacency_matrix=full_edge_matrix.detach(),
            cfg=model.config,
            scan=model.scan,
        )

        logger.info(
            f"Attribution completed in {time.time() - start_time:.2f}s | "
            f"{format_memory_snapshot(device=model.device, extra={'adjacency_shape': tuple(full_edge_matrix.shape)})}"
        )
        phase5_elapsed_ms = (time.perf_counter() - phase5_start) * 1000.0
        telemetry_observer.phase(
            name="phase5.packaging",
            phase="phase5",
            elapsed_ms=phase5_elapsed_ms,
            attrs={
                "compact_output": False,
                "adjacency_rows": int(full_edge_matrix.shape[0]),
                "adjacency_cols": int(full_edge_matrix.shape[1]),
            },
            wall_clock=True,
        )

        return graph
    finally:
        teardown_start = time.perf_counter()
        feature_row_store_for_cleanup = (
            feature_row_store
            if feature_row_store is not None
            else phase2_resource_owner.feature_row_store
        )
        nonfeature_row_store_for_cleanup = (
            nonfeature_row_store
            if nonfeature_row_store is not None
            else phase2_resource_owner.nonfeature_row_store
        )
        if feature_row_store_for_cleanup is not None:
            feature_row_store_for_cleanup.cleanup()
        if nonfeature_row_store_for_cleanup is not None:
            nonfeature_row_store_for_cleanup.cleanup()
        if ctx is not None:
            _log_memory_boundary(logger, "Teardown start", model.device)
            cleanup = getattr(ctx, "cleanup", None)
            if callable(cleanup):
                cleanup()
            else:
                clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
                if callable(clear_decoder_cache):
                    clear_decoder_cache()
            _log_memory_boundary(logger, "Teardown done", model.device)
        teardown_elapsed_ms = (time.perf_counter() - teardown_start) * 1000.0
        telemetry_observer.phase(
            name="teardown.cleanup",
            phase="teardown",
            elapsed_ms=teardown_elapsed_ms,
            attrs={
                "ctx_present": ctx is not None,
                "feature_row_store": feature_row_store_for_cleanup is not None,
            },
            wall_clock=True,
        )

        exc_type, exc, _ = sys.exc_info()
        if exc_type is None:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            telemetry_observer.run(
                name="attribute.done",
                elapsed_ms=run_elapsed_ms,
                attrs={"compact_output": compact_output},
                wall_clock=True,
            )
        else:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            telemetry_observer.run(
                name="attribute.failed",
                elapsed_ms=run_elapsed_ms,
                attrs={
                    "compact_output": compact_output,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc) if exc is not None else None,
                },
                wall_clock=True,
            )

        telemetry_export = telemetry_observer.close_export(include_events=True)
        if compact_output_result is not None:
            telemetry_observer.attach_compact_result(compact_output_result, telemetry_export)
            if prefix_view_metadata is not None:
                compact_output_result["prefix_view_metadata"] = dict(prefix_view_metadata)
            if anomaly_debug_result is not None:
                compact_output_result["phase4_anomaly_debug"] = anomaly_debug_result
            if (
                cross_cluster_debug_summary is not None
                and "cross_cluster_debug_summary" not in compact_output_result
            ):
                compact_output_result["cross_cluster_debug_summary"] = cross_cluster_debug_summary
            if (
                cross_cluster_debug_checkpoints is not None
                and "cross_cluster_debug_checkpoints" not in compact_output_result
            ):
                compact_output_result["cross_cluster_debug_checkpoints"] = (
                    cross_cluster_debug_checkpoints
                )
            if (
                cross_cluster_debug_batches is not None
                and "cross_cluster_debug_batches" not in compact_output_result
            ):
                compact_output_result["cross_cluster_debug_batches"] = cross_cluster_debug_batches
        else:
            if exc is not None:
                telemetry_observer.attach_exception(exc, telemetry_export)
            if profile:
                telemetry_observer.render_human_summary(logger, telemetry_export)
