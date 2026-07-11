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
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.graph import (
    Graph,
    compute_partial_feature_influences_streaming,
    compute_partial_influences,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    require_exact_chunked_provider,
)
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.utils.telemetry import (
    build_memory_before_after_attrs,
    diff_numeric_metrics,
    get_memory_snapshot,
    format_memory_snapshot,
    format_numeric_metrics,
)

from circuit_tracer.observability.exception_export import (
    _TELEMETRY_EXCEPTION_EVENTS_ATTR as _TELEMETRY_EXCEPTION_EVENTS_ATTR,
    _TELEMETRY_EXCEPTION_SUMMARY_ATTR as _TELEMETRY_EXCEPTION_SUMMARY_ATTR,
    _attach_telemetry_export_to_exception as _attach_telemetry_export_to_exception,
)
from circuit_tracer.observability.human_logs import (
    _log_batch_profile,
    _log_memory_boundary,
    _log_phase_metrics,
    _log_sparsification_profile,
    _snapshot_diagnostics,
)
from circuit_tracer.observability.lifecycle import (
    TelemetryObserver,
    _TelemetryObserver as _TelemetryObserver,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _RowTransferTelemetry as _RowTransferTelemetry,
    _build_cross_cluster_runtime_snapshot,
    _build_phase4_executor_batch_telemetry,
    _build_phase4_executor_substage_telemetry,
    _build_phase4_gpu_row_reduction_transfer_telemetry,
    _build_phase4_refresh_substage_telemetry,
    _build_row_transfer_telemetry,
    _build_tensor_transfer_estimate,
    _dtype_element_size as _dtype_element_size,
    _hash_json_payload,
    _record_cross_cluster_batch_event,
    _record_cross_cluster_checkpoint,
    _safe_float,
    _safe_int,
    _tensor_nbytes_estimate,
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

    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.perf_counter()
    input_ids = model.ensure_tokenized(prompt)
    n_input_pos = int(input_ids.shape[-1])
    if output_position is not None:
        output_position = int(output_position)
        if output_position < 0 or output_position >= n_input_pos:
            raise ValueError(
                f"output_position must be in [0, {n_input_pos}) (got {output_position})"
            )
    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(
        input_ids, prefix_view_metadata
    )
    _log_memory_boundary(logger, "Phase 0 start", model.device)

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(
            logger.info if profile else None,
            telemetry_recorder=telemetry_recorder,
        )

    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()

    configure_phase0_compare_mode = getattr(
        model.transcoders,
        "configure_phase0_activation_threshold_compare",
        None,
    )
    if callable(configure_phase0_compare_mode):
        configure_phase0_compare_mode(
            mode=phase0_activation_threshold_compare_mode_resolved,
            collect_diagnostics=cross_cluster_debug_enabled,
            sample_limit_per_layer=3,
        )

    if profile:
        logger.info(
            "Profiling enabled | "
            f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
            f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
            f"exact_chunked_provider_enabled={exact_chunked_provider_enabled} | "
            f"exact_chunked_decoder={exact_chunked_decoder} | "
            f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
            f"decoder_cache_bytes={getattr(model.transcoders, 'cross_batch_decoder_cache_bytes', 0)} | "
            f"chunked_feature_replay_window={chunked_feature_replay_window} | "
            f"error_vector_prefetch_lookahead={error_vector_prefetch_lookahead} | "
            f"stage_encoder_vecs_on_cpu={stage_encoder_vecs_on_cpu} | "
            f"stage_error_vectors_on_cpu={stage_error_vectors_on_cpu} | "
            f"row_subchunk_size={row_subchunk_size} | "
            f"planner_enabled={planner_enabled} | "
            f"feature_batch_size_max={max_phase4_feature_batch_size} | "
            f"phase1_trace_batch_policy={phase1_trace_batch_config.requested_policy} "
            f"(effective={phase1_trace_batch_config.effective_policy}, "
            f"size_max={phase1_trace_batch_config.requested_batch_size_max}, "
            f"size_max_effective={phase1_trace_batch_config.effective_batch_size_max}) | "
            f"phase4_refresh_policy={phase4_refresh_policy_config.requested_policy} "
            f"(effective={phase4_refresh_policy_config.effective_policy}, "
            f"interval_multiplier={phase4_refresh_policy_config.requested_interval_multiplier}, "
            f"interval_multiplier_effective={phase4_refresh_policy_config.effective_interval_multiplier}, "
            f"queue_multiplier_effective={phase4_refresh_policy_config.effective_queue_multiplier}) | "
            f"phase4_ranker={phase4_ranker_config.requested_mode} "
            f"(effective={phase4_ranker_config.effective_mode}) | "
            f"row_store_cache_control={row_store_cache_control_config.requested_mode} "
            f"(effective={row_store_cache_control_config.effective_mode}) | "
            f"exact_encoder_residency={exact_encoder_residency_config.requested_mode} "
            f"(effective={exact_encoder_residency_config.effective_mode}) | "
            f"exact_trace_internal_dtype={exact_trace_internal_dtype_name} | "
            f"prompt_tokens={input_ids.shape[-1]} | "
            f"source_batch_size={effective_source_batch_size} | "
            f"feature_batch_size={effective_feature_batch_size} | "
            f"logit_batch_size={effective_logit_batch_size} | "
            f"trace_batch_cap_reason={phase1_trace_batch_metadata.get('trace_batch_cap_reason')}"
        )

    if phase0_context_override is not None:
        ctx = phase0_context_override
    else:
        ctx = model.setup_attribution(
            input_ids,
            sparsification=sparsification,
            retain_full_logits=output_position is not None and output_position != n_input_pos - 1,
            chunked_feature_replay_window=chunked_feature_replay_window,
            error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
            row_subchunk_size=row_subchunk_size,
            exact_encoder_residency=exact_encoder_residency_config.effective_mode,
            internal_precision_requested=internal_precision_requested,
            resolved_dtype_map=resolved_dtype_map,
            prefix_view_length=prefix_view_length,
            decoder_chunk_cache=decoder_chunk_cache,
            decoder_cache_fingerprint=decoder_cache_fingerprint,
        )
    exact_encoder_runtime_metadata = {
        "exact_encoder_staging_destination": getattr(
            ctx, "exact_encoder_staging_destination", "none"
        ),
        "exact_encoder_materialized_during_phase0": bool(
            getattr(ctx, "exact_encoder_materialized_during_phase0", False)
        ),
        "active_encoder_shape": tuple(getattr(ctx, "encoder_vecs").shape),
        "active_encoder_bytes": int(
            getattr(ctx, "encoder_vecs").numel() * getattr(ctx, "encoder_vecs").element_size()
        ),
        "exact_encoder_pinned_effective": bool(
            getattr(ctx, "exact_encoder_pinned_effective", False)
        ),
        "exact_encoder_pinning_success": getattr(ctx, "exact_encoder_pinning_success", None),
        "exact_encoder_pinning_failure_reason": getattr(
            ctx, "exact_encoder_pinning_failure_reason", None
        ),
    }
    exact_encoder_residency_metadata.update(exact_encoder_runtime_metadata)
    phase4_execution_metadata.update(exact_encoder_runtime_metadata)
    if hasattr(ctx, "set_diagnostic_mode"):
        ctx.set_diagnostic_mode(profile)
    if capture_phase3_gradient_bundle_enabled:
        setattr(ctx, "capture_phase3_gradients", True)
    configure_ctx_trace_logging = getattr(ctx, "configure_trace_logging", None)
    if callable(configure_ctx_trace_logging):
        configure_ctx_trace_logging(
            logger.info if profile else None,
            telemetry_recorder=telemetry_recorder,
        )
    if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
        ctx.setup_diagnostic_stats.update(
            {
                "phase1_trace_batch": dict(phase1_trace_batch_metadata),
                "phase4_execution": dict(phase4_execution_metadata),
            }
        )

    prefix_view_activation_mask_metadata: dict[str, int] | None = None
    if (
        prefix_view_metadata is not None
        and prefix_view_metadata.get("mode") == "full_sequence_target_position"
    ):
        replace_phase0_activation_state = getattr(ctx, "replace_phase0_activation_state", None)
        if not callable(replace_phase0_activation_state):
            raise RuntimeError(
                "Attribution context does not support Phase-0 activation-state replacement"
            )
        prefix_view_activation_mask_metadata = _apply_prefix_view_activation_mask(
            ctx, int(prefix_view_metadata["target_position"])
        )
        if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
            ctx.setup_diagnostic_stats["prefix_view_activation_mask"] = dict(
                prefix_view_activation_mask_metadata
            )

    if diagnostic_feature_cap is not None and diagnostic_feature_cap > 0:
        before_cap, after_cap = ctx.apply_diagnostic_feature_cap(diagnostic_feature_cap)
        logger.info(
            f"Diagnostic feature cap applied before attribution rows: {before_cap} -> {after_cap} active features"
        )
    if profile and getattr(ctx, "sparsification_stats", None):
        _log_sparsification_profile(logger, ctx.sparsification_stats)

    try:
        activation_matrix = ctx.activation_matrix

        _log_phase_metrics(
            logger,
            "Precomputation",
            phase_start,
            model.device,
            active_features=ctx.activation_matrix._nnz(),
            logit_retention=getattr(ctx, "logit_retention", "full"),
        )
        phase0_elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        telemetry_observer.phase(
            name="phase0.precompute",
            phase="phase0",
            elapsed_ms=phase0_elapsed_ms,
            attrs={
                "active_features": int(ctx.activation_matrix._nnz()),
                "logit_retention": getattr(ctx, "logit_retention", "full"),
            },
            wall_clock=True,
        )
        if profile:
            if getattr(ctx, "setup_diagnostic_stats", None):
                logger.info(
                    f"Phase 0 setup diagnostics | {format_numeric_metrics(ctx.setup_diagnostic_stats, limit=20)}"
                )
            transcoder_snapshot = _snapshot_diagnostics(model.transcoders)
            if transcoder_snapshot:
                logger.info(
                    f"Precompute diagnostics | {format_numeric_metrics(transcoder_snapshot, limit=20)}"
                )
        logger.info(f"Found {ctx.activation_matrix._nnz()} active features")
        if cross_cluster_debug_summary is not None:
            phase0_runtime_summary, phase0_runtime_stream = _build_cross_cluster_runtime_snapshot(
                device=model.device,
                ctx=ctx,
                transcoder=model.transcoders,
            )
            activation_matrix = activation_matrix.coalesce()
            activation_indices = activation_matrix.indices().detach().cpu()
            activation_values = activation_matrix.values().detach().cpu()
            raw_sparse_index_hash = _hash_sparse_membership_indices(
                activation_indices,
                shape=activation_matrix.shape,
                canonicalize=False,
            )
            canonical_membership_hash = _hash_sparse_membership_indices(
                activation_indices,
                shape=activation_matrix.shape,
                canonicalize=True,
            )
            phase0_n_layers = int(activation_matrix.shape[0])
            layer_counts = (
                torch.bincount(activation_indices[0], minlength=phase0_n_layers).tolist()
                if activation_indices.numel() > 0
                else [0] * phase0_n_layers
            )
            transcoder_snapshot = phase0_runtime_summary.get("transcoder_diagnostic_snapshot")
            phase0_threshold_membership = (
                transcoder_snapshot.get("phase0_threshold_membership")
                if isinstance(transcoder_snapshot, dict)
                else None
            )
            if not isinstance(phase0_threshold_membership, dict):
                phase0_threshold_membership = None
            phase0_boundary_fingerprints = (
                transcoder_snapshot.get("phase0_boundary_fingerprints")
                if isinstance(transcoder_snapshot, dict)
                else None
            )
            if not isinstance(phase0_boundary_fingerprints, dict):
                phase0_boundary_fingerprints = None

            setup_diagnostic_stats = getattr(ctx, "setup_diagnostic_stats", None)
            phase0_pre_clt_input_fingerprints = (
                setup_diagnostic_stats.get("phase0_pre_clt_input_fingerprints")
                if isinstance(setup_diagnostic_stats, dict)
                else None
            )
            if not isinstance(phase0_pre_clt_input_fingerprints, dict):
                phase0_pre_clt_input_fingerprints = None

            phase0_boundary_global_hashes = (
                phase0_boundary_fingerprints.get("global_hashes")
                if isinstance(phase0_boundary_fingerprints, dict)
                else None
            )
            if not isinstance(phase0_boundary_global_hashes, dict):
                phase0_boundary_global_hashes = None
            activation_value_stats = _build_vector_stats(
                activation_values,
                epsilon=1e-12,
                top_k=8,
            )
            phase0_summary_checkpoint = {
                "active_feature_count": int(activation_matrix._nnz()),
                "per_layer_retained_counts": [int(v) for v in layer_counts],
                "active_feature_indices_hash": raw_sparse_index_hash,
                "active_feature_indices_hash_raw_order": raw_sparse_index_hash,
                "active_feature_membership_hash_canonical": canonical_membership_hash,
                "activation_value_stats": activation_value_stats,
                "phase0_activation_threshold_compare_mode": (
                    phase0_activation_threshold_compare_mode_resolved
                ),
                "phase0_threshold_membership": phase0_threshold_membership,
                "phase0_boundary_fingerprints": phase0_boundary_fingerprints,
                "phase0_pre_clt_input_fingerprints": phase0_pre_clt_input_fingerprints,
                "phase0_pre_clt_input_global_hash": (
                    phase0_pre_clt_input_fingerprints.get("global_hash")
                    if isinstance(phase0_pre_clt_input_fingerprints, dict)
                    else None
                ),
                "logit_retention": getattr(ctx, "logit_retention", None),
                "staging_flags": {
                    "stage_encoder_vecs_on_cpu": bool(stage_encoder_vecs_on_cpu),
                    "stage_error_vectors_on_cpu": bool(stage_error_vectors_on_cpu),
                },
                "setup_diagnostic_stats": setup_diagnostic_stats,
                **phase0_runtime_summary,
            }
            phase0_stream_checkpoint = {
                "active_feature_count": int(activation_matrix._nnz()),
                "retained_layer_count": int(phase0_n_layers),
                "retained_nonzero_layer_count": int(
                    sum(1 for value in layer_counts if int(value) > 0)
                ),
                "active_feature_indices_hash": phase0_summary_checkpoint[
                    "active_feature_indices_hash"
                ],
                "active_feature_membership_hash_canonical": canonical_membership_hash,
                "phase0_activation_threshold_compare_mode": (
                    phase0_activation_threshold_compare_mode_resolved
                ),
                "activation_value_count": int(activation_value_stats["count"]),
                "activation_value_nonfinite_count": int(activation_value_stats["nonfinite_count"]),
                "activation_value_abs_sum": _safe_float(activation_value_stats.get("abs_sum")),
                "activation_value_max": _safe_float(activation_value_stats.get("max")),
                "activation_value_effectively_all_zero": bool(
                    activation_value_stats["effectively_all_zero"]
                ),
                "phase0_threshold_membership_layer_count": (
                    int(len(phase0_threshold_membership.get("per_layer", {})))
                    if isinstance(phase0_threshold_membership, dict)
                    else None
                ),
                "phase0_threshold_membership_borderline_sample_count": (
                    int(phase0_threshold_membership.get("borderline_sample_count", 0))
                    if isinstance(phase0_threshold_membership, dict)
                    else None
                ),
                "phase0_threshold_membership_near_count_abs_lte_1e_04": (
                    int(
                        phase0_threshold_membership.get("near_counts_by_epsilon", {}).get(
                            "abs_lte_1e-04",
                            0,
                        )
                    )
                    if isinstance(phase0_threshold_membership, dict)
                    else None
                ),
                "phase0_pre_clt_input_global_hash": (
                    phase0_summary_checkpoint.get("phase0_pre_clt_input_global_hash")
                ),
                "phase0_pre_clt_input_layer_count": (
                    int(phase0_pre_clt_input_fingerprints.get("layer_count", 0))
                    if isinstance(phase0_pre_clt_input_fingerprints, dict)
                    else None
                ),
                "phase0_boundary_layer_count": (
                    int(len(phase0_boundary_fingerprints.get("per_layer", {})))
                    if isinstance(phase0_boundary_fingerprints, dict)
                    else None
                ),
                "phase0_boundary_transcoder_constants_global_hash": (
                    phase0_boundary_fingerprints.get("transcoder_constant_fingerprints", {}).get(
                        "global_hash"
                    )
                    if isinstance(phase0_boundary_fingerprints, dict)
                    else None
                ),
                "phase0_boundary_preactivation_hash_global": (
                    phase0_boundary_global_hashes.get("pre_activation_hash_global")
                    if isinstance(phase0_boundary_global_hashes, dict)
                    else None
                ),
                "phase0_boundary_margin_hash_global": (
                    phase0_boundary_global_hashes.get("compare_margin_hash_global")
                    if isinstance(phase0_boundary_global_hashes, dict)
                    else None
                ),
                "phase0_boundary_mask_membership_hash_global": (
                    phase0_boundary_global_hashes.get("mask_membership_hash_global")
                    if isinstance(phase0_boundary_global_hashes, dict)
                    else None
                ),
                "phase0_boundary_post_activation_hash_global": (
                    phase0_boundary_global_hashes.get("post_activation_hash_global")
                    if isinstance(phase0_boundary_global_hashes, dict)
                    else None
                ),
                "logit_retention": getattr(ctx, "logit_retention", None),
                "stage_encoder_vecs_on_cpu": bool(stage_encoder_vecs_on_cpu),
                "stage_error_vectors_on_cpu": bool(stage_error_vectors_on_cpu),
                "setup_diagnostic_stats_present": bool(
                    getattr(ctx, "setup_diagnostic_stats", None)
                ),
                **phase0_runtime_stream,
            }
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase0_sparse_setup",
                phase="phase0",
                summary_payload=phase0_summary_checkpoint,
                stream_payload=phase0_stream_checkpoint,
            )

        if offload and not model.skip_transcoder and not exact_chunked_decoder:
            offload_handles += offload_modules(model.transcoders, offload)

        # Phase 1: forward pass
        logger.info("Phase 1: Running forward pass")
        logger.info(
            "Phase 1 trace-batch policy | "
            f"requested_policy={phase1_trace_batch_config.requested_policy} | "
            f"effective_policy={phase1_trace_batch_config.effective_policy} | "
            f"requested_size_max={phase1_trace_batch_config.requested_batch_size_max} | "
            f"effective_size_max={phase1_trace_batch_config.effective_batch_size_max} | "
            f"effective_behavior={phase1_trace_batch_config.effective_behavior} | "
            f"source_batch_size={effective_source_batch_size} | "
            f"feature_batch_size={effective_feature_batch_size} | "
            f"logit_batch_size={effective_logit_batch_size} | "
            f"cap_reason={phase1_trace_batch_metadata.get('trace_batch_cap_reason')} | "
            f"trace_batch_size={trace_batch_size}"
        )
        phase_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 1 start", model.device)
        with model.trace() as tracer:
            with tracer.invoke(trace_input_ids.expand(trace_batch_size, -1)):
                pass

            detach_barrier = tracer.barrier(2)

            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            ctx.cache_residual(model, tracer, barrier=detach_barrier)

        _log_phase_metrics(logger, "Forward pass", phase_start, model.device)
        phase1_elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        telemetry_observer.phase(
            name="phase1.forward_pass",
            phase="phase1",
            elapsed_ms=phase1_elapsed_ms,
            attrs={
                "trace_batch_size": int(trace_batch_size),
                **phase1_trace_batch_metadata,
            },
            wall_clock=True,
        )

        if offload:
            offload_handles += offload_modules(
                [layer.mlp for layer in getattr(model.pre_logit_location, "layers")], offload
            )
            if model.skip_transcoder and not exact_chunked_decoder:
                offload_handles += offload_modules(model.transcoders, offload)

        # Phase 2: build input vector list
        logger.info("Phase 2: Building input vectors")
        phase2_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 2 start", model.device)

        # Create AttributionTargets using NNSight's unembed_weight accessor
        output_logits = (
            ctx.get_logits_at_position(output_position)[0]
            if output_position is not None and output_position != n_input_pos - 1
            else ctx.get_last_token_logits()[0]
        )
        if target_logits_override is not None:
            output_logits = target_logits_override.to(device=output_logits.device)
        targets = AttributionTargets(
            attribution_targets=attribution_targets,
            logits=output_logits,
            unembed_proj=cast(torch.Tensor, model.unembed_weight),  # NNSight uses unembed_weight
            tokenizer=model.tokenizer,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )

        log_attribution_target_info(targets, attribution_targets, logger)
        target_token_ids_tensor = torch.tensor(
            [int(target.vocab_idx) for target in targets.logit_targets],
            dtype=torch.int64,
            device=output_logits.device,
        )

        host_activation_matrix = activation_matrix.coalesce()
        host_transcoder_snapshot_for_replay = _snapshot_diagnostics(model.transcoders)
        host_clt_constants_hash = _extract_clt_constants_hash_from_snapshot(
            host_transcoder_snapshot_for_replay
        )
        host_validation_context = _build_phase0_replay_validation_context(
            input_tokens=input_ids,
            target_token_ids=target_token_ids_tensor,
            activation_matrix=host_activation_matrix,
            clt_constants_hash=host_clt_constants_hash,
        )
        host_hashes_for_replay_metadata = {
            "input_tokens_hash": host_validation_context.get("input_tokens_hash"),
            "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
            "active_feature_membership_hash_raw_order": host_validation_context.get(
                "active_feature_membership_hash_raw_order"
            ),
            "active_feature_membership_hash_canonical": host_validation_context.get(
                "active_feature_membership_hash_canonical"
            ),
            "clt_constants_hash": host_validation_context.get("clt_constants_hash"),
        }

        if phase0_replay_mode_resolved == "donor_phase0":
            assert phase0_donor_bundle_path is not None
            loaded_phase0_donor_bundle = _load_phase0_donor_bundle_npz(
                phase0_donor_bundle_path,
                context_policy=cast(
                    Literal["strict", "warn"],
                    phase0_donor_context_policy_resolved,
                ),
                validation_context=host_validation_context,
            )
            donor_activation_matrix = _build_phase0_activation_matrix_from_loaded_bundle(
                loaded_phase0_donor_bundle,
                device=host_activation_matrix.device,
            )
            replace_phase0_activation_state = getattr(ctx, "replace_phase0_activation_state", None)
            if callable(replace_phase0_activation_state):
                replace_phase0_activation_state(donor_activation_matrix)
            else:
                raise RuntimeError(
                    "Attribution context does not support Phase-0 activation-state replacement"
                )

            activation_matrix = ctx.activation_matrix.coalesce()
            donor_validation_metadata = cast(
                dict[str, object],
                loaded_phase0_donor_bundle.get("validation_metadata", {}),
            )
            donor_dtype_metadata = cast(
                dict[str, object],
                loaded_phase0_donor_bundle.get("dtype_metadata", {}),
            )
            donor_computed_hashes = cast(
                dict[str, object],
                donor_validation_metadata.get("computed_hashes", {}),
            )
            donor_stored_hashes = cast(
                dict[str, object],
                donor_validation_metadata.get("stored_hashes", {}),
            )
            donor_warning_list = [
                str(item)
                for item in cast(list[object], donor_validation_metadata.get("warnings", []))
            ]
            donor_warning_count = int(
                donor_validation_metadata.get(
                    "validation_failure_count",
                    len(donor_warning_list),
                )
            )
            phase0_replay_status = "applied_with_warnings" if donor_warning_list else "applied"
            phase0_replay_metadata = _build_phase0_replay_metadata(
                mode=phase0_replay_mode_resolved,
                status=phase0_replay_status,
                donor_bundle_path=phase0_donor_bundle_path,
                context_policy=phase0_donor_context_policy_resolved,
                validation_warnings=donor_warning_list,
                validation_failure_count=donor_warning_count,
                dtype_metadata=donor_dtype_metadata,
                host_hashes=host_hashes_for_replay_metadata,
                donor_hashes={
                    "computed": donor_computed_hashes,
                    "stored": donor_stored_hashes,
                },
                host_active_feature_count=int(host_activation_matrix._nnz()),
                donor_active_feature_count=int(activation_matrix._nnz()),
                replay_single_step_intended=True,
                note="single-step intended replay mode",
            )
            telemetry_recorder.record_event(
                scope="phase",
                name="phase2.phase0_replay",
                phase="phase2",
                attrs={
                    "phase0_replay_mode": phase0_replay_mode_resolved,
                    "phase0_replay_status": phase0_replay_status,
                    "context_policy": phase0_donor_context_policy_resolved,
                    "validation_warning_count": int(len(donor_warning_list)),
                    "dtype_roundtrip_loss": bool(
                        donor_dtype_metadata.get("dtype_roundtrip_loss", False)
                    ),
                    "host_active_feature_count": int(host_activation_matrix._nnz()),
                    "donor_active_feature_count": int(activation_matrix._nnz()),
                },
            )
        else:
            phase0_replay_metadata = _build_phase0_replay_metadata(
                mode=phase0_replay_mode_resolved,
                status="disabled",
                donor_bundle_path=None,
                context_policy=phase0_donor_context_policy_resolved,
                host_hashes=host_hashes_for_replay_metadata,
                host_active_feature_count=int(host_activation_matrix._nnz()),
                replay_single_step_intended=True,
                note="single-step intended replay mode",
            )
            telemetry_recorder.record_event(
                scope="phase",
                name="phase2.phase0_replay",
                phase="phase2",
                attrs={
                    "phase0_replay_mode": phase0_replay_mode_resolved,
                    "phase0_replay_status": "disabled",
                },
            )

        if capture_phase0_donor_bundle_enabled:
            valid_target_mask = (target_token_ids_tensor >= 0) & (
                target_token_ids_tensor < int(output_logits.shape[0])
            )
            target_logits = (
                output_logits[target_token_ids_tensor[valid_target_mask]]
                if bool(valid_target_mask.any().item())
                else None
            )
            capture_status = (
                "captured_replayed_effective_state"
                if phase0_replay_mode_resolved != "disabled"
                else "captured"
            )
            phase0_donor_bundle_payload = _build_phase0_donor_bundle_payload(
                activation_matrix=activation_matrix,
                input_tokens=input_ids,
                target_token_ids=target_token_ids_tensor,
                target_probabilities=targets.logit_probabilities,
                target_logits=target_logits,
                transcoder_diagnostic_snapshot=_snapshot_diagnostics(model.transcoders),
                status=capture_status,
            )
            phase0_donor_bundle_payload["replayed_effective_state"] = bool(
                phase0_replay_mode_resolved != "disabled"
            )
            phase0_donor_bundle_payload["phase0_replay_mode"] = phase0_replay_mode_resolved

        feat_layers, feat_pos, feat_ids = activation_matrix.indices()
        n_layers, n_pos, _ = activation_matrix.shape
        total_active_feats = activation_matrix._nnz()

        if cross_cluster_debug_summary is not None:
            phase1_runtime_summary, phase1_runtime_stream = _build_cross_cluster_runtime_snapshot(
                device=model.device,
                ctx=ctx,
                transcoder=model.transcoders,
            )
            target_token_ids = [int(target.vocab_idx) for target in targets.logit_targets]
            target_probabilities = targets.logit_probabilities.detach().cpu()
            target_probability_stats = _build_vector_stats(
                target_probabilities,
                epsilon=1e-12,
                top_k=8,
            )
            phase1_summary_checkpoint = {
                "target_count": int(len(targets)),
                "target_token_ids": target_token_ids,
                "target_token_ids_hash": _hash_index_tensor(
                    torch.tensor(target_token_ids, dtype=torch.int64)
                )
                if target_token_ids
                else None,
                "target_probability_stats": target_probability_stats,
                "target_logit_state_hash": _hash_float_tensor(
                    target_probabilities,
                    dtype=torch.float64,
                ),
                **phase1_runtime_summary,
            }
            phase1_stream_checkpoint = {
                "target_count": int(len(targets)),
                "target_token_ids_hash": phase1_summary_checkpoint["target_token_ids_hash"],
                "target_probability_count": int(target_probability_stats["count"]),
                "target_probability_nonfinite_count": int(
                    target_probability_stats["nonfinite_count"]
                ),
                "target_probability_abs_sum": _safe_float(target_probability_stats.get("abs_sum")),
                "target_probability_max": _safe_float(target_probability_stats.get("max")),
                "target_probability_effectively_all_zero": bool(
                    target_probability_stats["effectively_all_zero"]
                ),
                "target_logit_state_hash": phase1_summary_checkpoint["target_logit_state_hash"],
                **phase1_runtime_stream,
            }
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase1_target_logits",
                phase="phase1",
                summary_payload=phase1_summary_checkpoint,
                stream_payload=phase1_stream_checkpoint,
            )
            cross_cluster_debug_summary["phase0_replay_metadata"] = phase0_replay_metadata
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase2_phase0_replay",
                phase="phase2",
                summary_payload=phase0_replay_metadata,
                stream_payload={
                    "phase0_replay_mode": phase0_replay_metadata.get("mode"),
                    "phase0_replay_status": phase0_replay_metadata.get("status"),
                    "validation_warning_count": phase0_replay_metadata.get(
                        "validation_warning_count"
                    ),
                    "dtype_roundtrip_loss": cast(
                        dict[str, object],
                        phase0_replay_metadata.get("dtype_metadata", {}),
                    ).get("dtype_roundtrip_loss"),
                },
            )

        if offload:
            offload_handles += offload_modules([model.embed_location], offload)
            tied_embeds = (
                model.embed_weight.untyped_storage().data_ptr()  # type:ignore
                == model.unembed_weight.untyped_storage().data_ptr()  # type:ignore
            )
            if not tied_embeds:
                offload_handles += offload_modules([model.lm_head], offload)

        logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
        n_logits = len(targets)
        total_nodes = logit_offset + n_logits

        base_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
        actual_max_feature_nodes = base_max_feature_nodes
        phase3_frontier_buffer_metadata = _build_phase3_frontier_buffer_metadata(
            seed_feature_influences=None,
            base_max_feature_nodes=int(base_max_feature_nodes),
            total_active_features=int(total_active_feats),
            relative_epsilon=phase3_frontier_buffer_relative_epsilon,
            max_extra=int(phase3_frontier_buffer_max_extra),
        )
        row_store_capacity_feature_nodes = min(
            base_max_feature_nodes
            + (
                int(phase3_frontier_buffer_max_extra)
                if phase3_frontier_buffer_relative_epsilon is not None
                else 0
            )
            + (
                int(phase4_frontier_buffer_max_extra_total)
                if phase4_frontier_buffer_relative_epsilon is not None
                else 0
            ),
            total_active_feats,
        )
        phase4_frontier_buffer_metadata: dict[str, object] = {
            "schema_version": 1,
            "requested": bool(
                phase4_frontier_buffer_relative_epsilon is not None
                or phase4_frontier_buffer_max_extra_per_refresh > 0
                or phase4_frontier_buffer_max_extra_total > 0
            ),
            "enabled": bool(
                phase4_frontier_buffer_relative_epsilon is not None
                and phase4_frontier_buffer_max_extra_per_refresh > 0
                and phase4_frontier_buffer_max_extra_total > 0
            ),
            "effective": False,
            "relative_epsilon": None
            if phase4_frontier_buffer_relative_epsilon is None
            else float(phase4_frontier_buffer_relative_epsilon),
            "max_extra_per_refresh": int(phase4_frontier_buffer_max_extra_per_refresh),
            "max_extra_total": int(phase4_frontier_buffer_max_extra_total),
            "extra_feature_count_total": 0,
            "expanded_refresh_count": 0,
            "fallback_count": 0,
            "capacity_feature_nodes": int(row_store_capacity_feature_nodes),
            "initial_target_feature_nodes": int(base_max_feature_nodes),
            "final_actual_max_feature_nodes": int(actual_max_feature_nodes),
            "events": [],
        }
        logger.info(
            f"Will include {actual_max_feature_nodes} of {total_active_feats} feature nodes"
        )

        if use_compact_feature_row_store:
            # Benchmark-critical path only: exact chunked decoder + compact output.
            # Keep dense full-row behavior unchanged for non-compact Graph outputs.
            assert compact_output
            assert exact_chunked_decoder
            n_nonfeature_columns = int(logit_offset - total_active_feats)
            feature_row_store = _FileBackedFeatureRowStore(
                n_rows=row_store_capacity_feature_nodes + n_logits,
                n_feature_columns=total_active_feats,
                dtype=exact_trace_internal_dtype_resolved,
                row_abs_sum_dtype=exact_trace_internal_dtype_resolved,
                read_chunk_cache_bytes=256 * 1024 * 1024,
                prepared_read_cache_bytes=phase4_refresh_prepared_chunk_cache_bytes_effective,
                row_store_cache_control_mode=row_store_cache_control_config.effective_mode,
                temp_root_policy=row_store_temp_root_policy_resolved,
                temp_root=row_store_temp_root,
                preallocate=row_store_preallocate,
                telemetry_recorder=telemetry_recorder,
            )
            nonfeature_row_store = _FileBackedFeatureRowStore(
                n_rows=row_store_capacity_feature_nodes + n_logits,
                n_feature_columns=n_nonfeature_columns,
                dtype=exact_trace_internal_dtype_resolved,
                row_abs_sum_dtype=exact_trace_internal_dtype_resolved,
                read_chunk_cache_bytes=256 * 1024 * 1024,
                prepared_read_cache_bytes=0,
                row_store_cache_control_mode=row_store_cache_control_config.effective_mode,
                temp_root_policy=row_store_temp_root_policy_resolved,
                temp_root=row_store_temp_root,
                preallocate=row_store_preallocate,
                telemetry_recorder=telemetry_recorder,
            )
        else:
            edge_matrix = torch.zeros(row_store_capacity_feature_nodes + n_logits, total_nodes)

        # Maps stored row indices to original feature/node indices.
        # First populated with logit node IDs, then feature IDs in attribution order
        row_to_node_index = torch.zeros(
            row_store_capacity_feature_nodes + n_logits, dtype=torch.int32
        )

        phase2_extra: dict[str, object] = {
            "row_store_mode": (
                "compact_feature_file_backed_dense"
                if use_compact_feature_row_store
                else "dense_full"
            ),
            "phase0_replay_mode": phase0_replay_metadata.get("mode"),
            "phase0_replay_status": phase0_replay_metadata.get("status"),
            "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                "validation_warning_count"
            ),
        }
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            assert nonfeature_row_store is not None
            phase2_extra.update(
                feature_row_store="dense_memmap",
                feature_row_store_path=feature_row_store.path,
                nonfeature_row_store_path=nonfeature_row_store.path,
                row_abs_sums_shape=f"{tuple(feature_row_store.row_abs_max.shape)}",
                row_abs_max_shape=f"{tuple(feature_row_store.row_abs_max.shape)}",
                row_l1_scaled_shape=f"{tuple(feature_row_store.row_l1_scaled.shape)}",
                feature_edge_columns=total_active_feats,
                nonfeature_edge_columns=n_nonfeature_columns,
                **feature_row_store.get_diagnostic_snapshot(),
            )
        else:
            phase2_extra.update(
                edge_matrix_shape=f"{tuple(edge_matrix.shape)}",
                edge_matrix_dtype=edge_matrix.dtype,
            )

        _log_phase_metrics(
            logger,
            "Input vector build",
            phase2_start,
            model.device,
            **phase2_extra,
        )
        phase2_elapsed_ms = (time.perf_counter() - phase2_start) * 1000.0
        telemetry_observer.phase(
            name="phase2.input_vector_build",
            phase="phase2",
            elapsed_ms=phase2_elapsed_ms,
            attrs=phase2_extra,
            wall_clock=True,
        )
        if cross_cluster_debug_summary is not None:
            phase2_runtime_summary, phase2_runtime_stream = _build_cross_cluster_runtime_snapshot(
                device=model.device,
                ctx=ctx,
                transcoder=model.transcoders,
            )
            row_store_dtype_for_metrics = (
                exact_trace_internal_dtype_resolved
                if use_compact_feature_row_store
                else feature_row_storage_dtype
            )
            row_abs_sum_dtype_for_metrics = (
                exact_trace_internal_dtype_resolved
                if use_compact_feature_row_store
                else row_abs_sum_dtype
            )
            row_denominator_component_count = 2 if use_compact_feature_row_store else 1
            row_count = int(actual_max_feature_nodes + n_logits)
            row_store_expected_bytes = (
                row_count
                * int(total_active_feats)
                * torch.empty((), dtype=row_store_dtype_for_metrics).element_size()
            )
            row_abs_sums_expected_bytes = (
                row_denominator_component_count
                * row_count
                * torch.empty((), dtype=row_abs_sum_dtype_for_metrics).element_size()
            )
            phase2_summary_checkpoint = {
                "feat_layers_hash": _hash_index_tensor(feat_layers),
                "feat_pos_hash": _hash_index_tensor(feat_pos),
                "feat_ids_hash": _hash_index_tensor(feat_ids),
                "feature_count": int(total_active_feats),
                "phase0_replay_mode": phase0_replay_metadata.get("mode"),
                "phase0_replay_status": phase0_replay_metadata.get("status"),
                "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                    "validation_warning_count"
                ),
                "decoder_chunk_size": (
                    int(getattr(model.transcoders, "decoder_chunk_size", 0))
                    if getattr(model.transcoders, "decoder_chunk_size", None) is not None
                    else None
                ),
                "row_store_mode": phase2_extra.get("row_store_mode"),
                "row_denominator_component_count": int(row_denominator_component_count),
                "row_store_expected_bytes": int(row_store_expected_bytes),
                "row_abs_sums_expected_bytes": int(row_abs_sums_expected_bytes),
                "row_denominator_expected_bytes": int(row_abs_sums_expected_bytes),
                "phase4_feature_batch_size_initial": int(effective_feature_batch_size),
                **phase2_runtime_summary,
            }
            phase2_stream_checkpoint = {
                "feat_layers_hash": phase2_summary_checkpoint["feat_layers_hash"],
                "feat_pos_hash": phase2_summary_checkpoint["feat_pos_hash"],
                "feat_ids_hash": phase2_summary_checkpoint["feat_ids_hash"],
                "feature_count": int(total_active_feats),
                "phase0_replay_mode": phase0_replay_metadata.get("mode"),
                "phase0_replay_status": phase0_replay_metadata.get("status"),
                "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                    "validation_warning_count"
                ),
                "decoder_chunk_size": phase2_summary_checkpoint["decoder_chunk_size"],
                "row_store_mode": phase2_summary_checkpoint["row_store_mode"],
                "row_denominator_component_count": int(row_denominator_component_count),
                "row_store_expected_bytes": int(row_store_expected_bytes),
                "row_abs_sums_expected_bytes": int(row_abs_sums_expected_bytes),
                "row_denominator_expected_bytes": int(row_abs_sums_expected_bytes),
                "phase4_feature_batch_size_initial": int(effective_feature_batch_size),
                **phase2_runtime_stream,
            }
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase2_feature_ordering",
                phase="phase2",
                summary_payload=phase2_summary_checkpoint,
                stream_payload=phase2_stream_checkpoint,
            )

        if phase3_gradient_replay_mode_resolved == "donor":
            assert phase3_gradient_donor_bundle_path is not None
            loaded_gradient_bundle = _load_phase3_gradient_donor_bundle_npz(
                phase3_gradient_donor_bundle_path,
                target_token_ids=target_token_ids_tensor,
                active_features=activation_matrix.indices().T,
                activation_values=activation_matrix.values(),
                expected_n_layers=int(n_layers),
                expected_gradient_batch_size=int(trace_batch_size),
                expected_n_positions=int(n_pos),
                expected_d_model=int(targets.logit_vectors.shape[-1]),
                validation_policy=cast(Literal["strict"], phase3_replay_validation_policy_resolved),
            )
            gradient_tensor = cast(torch.Tensor, loaded_gradient_bundle["gradients"])
            setattr(ctx, "phase3_gradient_replay_tensor", gradient_tensor)
            setattr(ctx, "phase3_gradient_replay_status", "applied")
            gradient_validation_metadata = cast(
                dict[str, object], loaded_gradient_bundle.get("validation_metadata", {})
            )
            phase3_gradient_replay_metadata = _build_phase3_replay_metadata(
                replay_kind="phase3_gradient_replay_v1",
                mode=phase3_gradient_replay_mode_resolved,
                status="applied",
                donor_bundle_path=phase3_gradient_donor_bundle_path,
                validation_policy=phase3_replay_validation_policy_resolved,
                validation_failure_count=int(
                    gradient_validation_metadata.get("validation_failure_count", 0)
                ),
                donor_hashes=cast(
                    dict[str, object], gradient_validation_metadata.get("stored_hashes", {})
                ),
                host_hashes={
                    "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
                    "active_features_hash": _hash_index_tensor(
                        activation_matrix.indices().T.detach().cpu().reshape(-1)
                    ),
                    "activation_values_hash": _hash_tensor_raw_bytes(activation_matrix.values()),
                    "active_feature_count": int(total_active_feats),
                },
                source="donor_gradient_bundle",
                note="feature/error gradients replayed from donor; token gradient remains host-computed",
            )
        else:
            setattr(ctx, "phase3_gradient_replay_tensor", None)
            setattr(ctx, "phase3_gradient_replay_status", "disabled")

        if phase3_row_replay_mode_resolved == "donor":
            assert phase3_row_donor_bundle_path is not None
            loaded_phase3_row_donor_bundle = _load_phase3_row_donor_bundle_npz(
                phase3_row_donor_bundle_path,
                target_token_ids=target_token_ids_tensor,
                active_features=activation_matrix.indices().T,
                activation_values=activation_matrix.values(),
                expected_total_active_features=int(total_active_feats),
                validation_policy=cast(Literal["strict"], phase3_replay_validation_policy_resolved),
            )
            row_validation_metadata = cast(
                dict[str, object], loaded_phase3_row_donor_bundle.get("validation_metadata", {})
            )
            phase3_row_replay_metadata = _build_phase3_replay_metadata(
                replay_kind="phase3_row_replay_v1",
                mode=phase3_row_replay_mode_resolved,
                status="applied",
                donor_bundle_path=phase3_row_donor_bundle_path,
                validation_policy=phase3_replay_validation_policy_resolved,
                validation_failure_count=int(
                    row_validation_metadata.get("validation_failure_count", 0)
                ),
                donor_hashes=cast(
                    dict[str, object], row_validation_metadata.get("stored_hashes", {})
                ),
                host_hashes={
                    "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
                    "active_features_hash": _hash_index_tensor(
                        activation_matrix.indices().T.detach().cpu().reshape(-1)
                    ),
                    "activation_values_hash": _hash_tensor_raw_bytes(activation_matrix.values()),
                    "active_feature_count": int(total_active_feats),
                },
                source="donor_row_bundle_override",
                note=(
                    "donor row bundle overrides feature rows and row normalizers; "
                    "dense token/error columns remain host-computed"
                ),
            )

        # Phase 3: logit attribution
        logger.info("Phase 3: Computing logit attributions")
        phase3_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 3 start", model.device)
        i = -1
        total_logit_batches = max(
            (len(targets) + effective_logit_batch_size - 1) // effective_logit_batch_size,
            1,
        )
        phase3_feature_row_batches: list[torch.Tensor] = []
        phase3_row_abs_sum_batches: list[torch.Tensor] = []
        phase3_feature_abs_sum_batches: list[torch.Tensor] = []
        phase3_error_abs_sum_batches: list[torch.Tensor] = []
        phase3_token_abs_sum_batches: list[torch.Tensor] = []
        rows_cpu_staging: torch.Tensor | None = None
        phase3_compute_batch_elapsed_ms_total = 0.0
        phase3_cpu_staging_elapsed_ms_total = 0.0
        phase3_denominator_elapsed_ms_total = 0.0
        phase3_row_store_write_elapsed_ms_total = 0.0
        phase3_gpu_to_cpu_bytes_total = 0
        phase3_cpu_to_gpu_bytes_total = 0
        phase3_copy_count = 0
        for i in range(0, len(targets), effective_logit_batch_size):
            batch = targets.logit_vectors[i : i + effective_logit_batch_size]
            ctx_before = _snapshot_diagnostics(ctx) if profile else None
            transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
            batch_start = time.perf_counter()
            batch_memory_before = get_memory_snapshot(model.device)
            if phase3_gradient_replay_mode_resolved == "donor":
                setattr(ctx, "phase3_gradient_replay_column_offset", int(i))
            phase3_inject_transfer_telemetry = _build_tensor_transfer_estimate(
                prefix="inject_values",
                source=batch,
                destination_device=model.device,
            )
            if (
                phase3_inject_transfer_telemetry["inject_values_source"] == "cpu"
                and phase3_inject_transfer_telemetry["inject_values_destination"] == "cuda"
            ):
                phase3_cpu_to_gpu_bytes_total += int(
                    phase3_inject_transfer_telemetry["inject_values_transfer_bytes"]
                )
            compute_batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), n_layers),
                positions=torch.full(
                    (batch.shape[0],),
                    output_position if output_position is not None else n_pos - 1,
                ),
                inject_values=batch,
                phase_label="phase3_logits",
            )
            phase3_compute_batch_elapsed_ms = (time.perf_counter() - compute_batch_start) * 1000.0
            phase3_compute_batch_elapsed_ms_total += phase3_compute_batch_elapsed_ms
            cpu_staging_start = time.perf_counter()
            rows_cpu, rows_cpu_staging = _copy_rows_to_cpu_staging(
                rows,
                staging_buffer=rows_cpu_staging,
            )
            phase3_cpu_staging_elapsed_ms = (time.perf_counter() - cpu_staging_start) * 1000.0
            phase3_cpu_staging_elapsed_ms_total += phase3_cpu_staging_elapsed_ms
            donor_feature_rows: torch.Tensor | None = None
            donor_row_abs_sums: torch.Tensor | None = None
            donor_feature_abs_sums: torch.Tensor | None = None
            donor_error_abs_sums: torch.Tensor | None = None
            donor_token_abs_sums: torch.Tensor | None = None
            if loaded_phase3_row_donor_bundle is not None:
                end = i + batch.shape[0]
                donor_feature_rows = cast(
                    torch.Tensor,
                    loaded_phase3_row_donor_bundle["phase3_feature_rows"],
                )[i:end]
                donor_row_abs_sums = cast(
                    torch.Tensor,
                    loaded_phase3_row_donor_bundle["row_abs_sums"],
                )[i:end]
                donor_feature_abs_sums = cast(
                    torch.Tensor,
                    loaded_phase3_row_donor_bundle["feature_abs_sums"],
                )[i:end]
                donor_error_abs_sums = cast(
                    torch.Tensor,
                    loaded_phase3_row_donor_bundle["error_abs_sums"],
                )[i:end]
                donor_token_abs_sums = cast(
                    torch.Tensor,
                    loaded_phase3_row_donor_bundle["token_abs_sums"],
                )[i:end]
            denominator_start = time.perf_counter()
            (
                rows_cpu,
                row_input_slice,
                feature_row_slice,
                (row_abs_max_cpu, row_l1_scaled_cpu),
                row_abs_sums_cpu,
            ) = _resolve_phase3_effective_row_state(
                rows_cpu=rows_cpu,
                row_input_column_count=int(logit_offset),
                total_active_features=int(total_active_feats),
                dtype=exact_trace_internal_dtype_resolved,
                donor_feature_rows=donor_feature_rows,
                donor_row_abs_sums=donor_row_abs_sums,
            )
            phase3_denominator_elapsed_ms = (time.perf_counter() - denominator_start) * 1000.0
            phase3_denominator_elapsed_ms_total += phase3_denominator_elapsed_ms
            phase3_row_transfer_telemetry = _build_row_transfer_telemetry(
                rows=rows,
                rows_cpu=rows_cpu,
                row_input_slice=row_input_slice,
                feature_row_slice=feature_row_slice,
            )
            if phase3_row_transfer_telemetry["row_transfer_source"] == "cuda":
                phase3_gpu_to_cpu_bytes_total += int(
                    phase3_row_transfer_telemetry["row_transfer_bytes"]
                )
            if phase3_row_transfer_telemetry["row_transfer_destination"] == "cuda":
                phase3_cpu_to_gpu_bytes_total += int(
                    phase3_row_transfer_telemetry["row_transfer_bytes"]
                )
            if int(phase3_row_transfer_telemetry["row_transfer_bytes"]) > 0:
                phase3_copy_count += 1
            if capture_phase3_row_bundle_enabled:
                feature_rows_cpu = feature_row_slice.contiguous()
                error_start = int(total_active_feats)
                error_end = int(total_active_feats + n_layers * n_pos)
                token_end = int(logit_offset)
                phase3_feature_row_batches.append(feature_rows_cpu)
                phase3_row_abs_sum_batches.append(row_abs_sums_cpu.contiguous())
                if (
                    donor_feature_abs_sums is not None
                    and donor_error_abs_sums is not None
                    and donor_token_abs_sums is not None
                ):
                    phase3_feature_abs_sum_batches.append(donor_feature_abs_sums.contiguous())
                    phase3_error_abs_sum_batches.append(donor_error_abs_sums.contiguous())
                    phase3_token_abs_sum_batches.append(donor_token_abs_sums.contiguous())
                else:
                    phase3_feature_abs_sum_batches.append(
                        _compute_row_abs_sums(
                            feature_rows_cpu,
                            dtype=torch.float64,
                        ).contiguous()
                    )
                    phase3_error_abs_sum_batches.append(
                        _compute_row_abs_sums(
                            rows_cpu[:, error_start:error_end],
                            dtype=torch.float64,
                        ).contiguous()
                    )
                    phase3_token_abs_sum_batches.append(
                        _compute_row_abs_sums(
                            rows_cpu[:, error_end:token_end],
                            dtype=torch.float64,
                        ).contiguous()
                    )
            if anomaly_debug_result is not None:
                logit_row_batches = anomaly_debug_result.setdefault(
                    "phase3_logit_row_batches",
                    [],
                )
                assert isinstance(logit_row_batches, list)
                logit_row_batches.append(
                    {
                        "batch_index": int((i // effective_logit_batch_size) + 1),
                        "batch_row_count": int(batch.shape[0]),
                        "row_input_stats": _build_matrix_abs_stats(
                            row_input_slice,
                            epsilon=1e-12,
                            top_k=8,
                        ),
                        "row_abs_sum_stats": _build_phase4_normalization_stats(
                            (row_abs_max_cpu, row_l1_scaled_cpu),
                            clamp_epsilon=1e-8,
                        ),
                    }
                )
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                assert nonfeature_row_store is not None
                end = i + batch.shape[0]
                row_store_write_start = time.perf_counter()
                feature_row_store.append_rows(
                    row_start=i,
                    feature_rows=feature_row_slice,
                    row_denominator_scaled_l1=(row_abs_max_cpu, row_l1_scaled_cpu),
                    phase="phase3",
                )
                nonfeature_row_store.append_rows(
                    row_start=i,
                    feature_rows=rows_cpu[:, total_active_feats:logit_offset],
                    row_denominator_scaled_l1=(row_abs_max_cpu, row_l1_scaled_cpu),
                    phase="phase3",
                )
                phase3_row_store_write_elapsed_ms = (
                    time.perf_counter() - row_store_write_start
                ) * 1000.0
            else:
                row_store_write_start = time.perf_counter()
                edge_matrix[i : i + batch.shape[0], :logit_offset] = rows_cpu
                phase3_row_store_write_elapsed_ms = (
                    time.perf_counter() - row_store_write_start
                ) * 1000.0
            phase3_row_store_write_elapsed_ms_total += phase3_row_store_write_elapsed_ms
            row_to_node_index[i : i + batch.shape[0]] = (
                torch.arange(i, i + batch.shape[0]) + logit_offset
            )
            batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_memory_after = get_memory_snapshot(model.device)
            telemetry_observer.batch(
                name="phase3.logit_batch",
                phase="phase3",
                batch_index=(i // effective_logit_batch_size) + 1,
                elapsed_ms=batch_elapsed_ms,
                attrs={
                    "batch_rows": int(batch.shape[0]),
                    "batch_start_index": int(i),
                    "total_logit_batches": int(total_logit_batches),
                    "compute_batch_elapsed_ms": float(phase3_compute_batch_elapsed_ms),
                    "cpu_staging_elapsed_ms": float(phase3_cpu_staging_elapsed_ms),
                    "denominator_elapsed_ms": float(phase3_denominator_elapsed_ms),
                    "row_store_write_elapsed_ms": float(phase3_row_store_write_elapsed_ms),
                    **phase3_inject_transfer_telemetry,
                    **phase3_row_transfer_telemetry,
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
                    (row_abs_max_cpu, row_l1_scaled_cpu),
                    clamp_epsilon=1e-8,
                )
                _record_cross_cluster_batch_event(
                    cross_cluster_debug_batches=cross_cluster_debug_batches,
                    event_name="phase3.logit_batch",
                    phase="phase3",
                    event_index=(i // effective_logit_batch_size) + 1,
                    payload={
                        "batch_rows": int(batch.shape[0]),
                        "batch_start_index": int(i),
                        "total_logit_batches": int(total_logit_batches),
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
                        **get_memory_snapshot(model.device),
                    },
                )
            if profile and ((i // effective_logit_batch_size) + 1) % profile_log_interval == 0:
                _log_batch_profile(
                    logger,
                    "Phase 3",
                    (i // effective_logit_batch_size) + 1,
                    total_logit_batches,
                    batch_elapsed_ms / 1000.0,
                    ctx_before,
                    _snapshot_diagnostics(ctx),
                    transcoder_before,
                    _snapshot_diagnostics(model.transcoders),
                )

        _log_phase_metrics(
            logger,
            f"{i + 1} logit attribution(s)",
            phase3_start,
            model.device,
        )
        phase3_elapsed_ms = (time.perf_counter() - phase3_start) * 1000.0
        telemetry_observer.phase(
            name="phase3.logit_attribution",
            phase="phase3",
            elapsed_ms=phase3_elapsed_ms,
            attrs={
                "logit_count": int(len(targets)),
                "batches": int(total_logit_batches),
                "phase3_compute_batch_elapsed_ms_total": float(
                    phase3_compute_batch_elapsed_ms_total
                ),
                "phase3_cpu_staging_elapsed_ms_total": float(phase3_cpu_staging_elapsed_ms_total),
                "phase3_denominator_elapsed_ms_total": float(phase3_denominator_elapsed_ms_total),
                "phase3_row_store_write_elapsed_ms_total": float(
                    phase3_row_store_write_elapsed_ms_total
                ),
                "phase3_gpu_to_cpu_bytes_total": int(phase3_gpu_to_cpu_bytes_total),
                "phase3_cpu_to_gpu_bytes_total": int(phase3_cpu_to_gpu_bytes_total),
                "phase3_copy_count": int(phase3_copy_count),
            },
            wall_clock=True,
        )
        reset_decoder_cache = getattr(ctx, "reset_decoder_cache", None)
        if callable(reset_decoder_cache):
            reset_decoder_cache()

        phase3_target_token_ids = torch.tensor(
            [int(target.vocab_idx) for target in targets.logit_targets],
            dtype=torch.int64,
        )
        if capture_phase3_gradient_bundle_enabled:
            gradient_captures = getattr(ctx, "phase3_gradient_captures", [])
            phase3_gradient_bundle_payload = _build_phase3_gradient_bundle_payload(
                gradient_captures=(
                    gradient_captures if isinstance(gradient_captures, list) else []
                ),
                active_features=activation_matrix.indices().T,
                activation_values=activation_matrix.values(),
                target_token_ids=phase3_target_token_ids,
                target_probabilities=targets.logit_probabilities,
                status=(
                    "captured_replayed_effective_state"
                    if phase3_gradient_replay_mode_resolved != "disabled"
                    else "captured"
                ),
            )
        if capture_phase3_row_bundle_enabled:
            phase3_row_bundle_payload = _build_phase3_row_bundle_payload(
                feature_rows=phase3_feature_row_batches,
                row_abs_sums=phase3_row_abs_sum_batches,
                feature_abs_sums=phase3_feature_abs_sum_batches,
                error_abs_sums=phase3_error_abs_sum_batches,
                token_abs_sums=phase3_token_abs_sum_batches,
                active_features=activation_matrix.indices().T,
                activation_values=activation_matrix.values(),
                target_token_ids=phase3_target_token_ids,
                target_probabilities=targets.logit_probabilities,
                total_active_features=int(total_active_feats),
                error_column_count=int(n_layers * n_pos),
                token_column_count=int(n_pos),
                status=(
                    "captured_replayed_effective_state"
                    if (
                        phase3_gradient_replay_mode_resolved != "disabled"
                        or phase3_row_replay_mode_resolved != "disabled"
                    )
                    else "captured"
                ),
            )

        if (
            cross_cluster_debug_summary is not None
            or capture_phase3_seed_bundle_enabled
            or capture_feature_semantic_descriptors_enabled
            or phase3_frontier_buffer_metadata["enabled"]
        ):
            phase3_runtime_summary: dict[str, object] = {}
            phase3_runtime_stream: dict[str, object] = {}
            if cross_cluster_debug_summary is not None:
                phase3_runtime_summary, phase3_runtime_stream = (
                    _build_cross_cluster_runtime_snapshot(
                        device=model.device,
                        ctx=ctx,
                        transcoder=model.transcoders,
                    )
                )
            pre_phase4_st = int(n_logits)
            phase3_seed_summary: dict[str, object] = {
                "stored_row_count_before_phase4": pre_phase4_st,
                "actual_max_feature_nodes": int(actual_max_feature_nodes),
                "total_active_features": int(total_active_feats),
                "update_interval": int(update_interval),
                "feature_batch_size": int(effective_feature_batch_size),
                "planner_compute_dtype": _dtype_to_name(planner_compute_dtype),
                "influence_compute_dtype": _dtype_to_name(influence_compute_dtype),
                **phase3_runtime_summary,
            }
            if actual_max_feature_nodes < total_active_feats:
                normalization_input_stats: dict[str, object] | None = None
                row_store_snapshot: dict[str, float | int | None] | None = None
                if use_compact_feature_row_store:
                    assert feature_row_store is not None
                    row_denominator_prefix = (
                        feature_row_store.row_abs_max[:pre_phase4_st],
                        feature_row_store.row_l1_scaled[:pre_phase4_st],
                    )
                    seed_feature_influences = compute_partial_feature_influences_streaming(
                        lambda row_start, row_end: feature_row_store.read_feature_rows(
                            row_start,
                            row_end,
                            phase="phase3_seed_ranking",
                        ),
                        row_denominator_prefix,
                        targets.logit_probabilities,
                        row_to_node_index[:pre_phase4_st],
                        n_feature_nodes=total_active_feats,
                        n_logits=n_logits,
                        device=feature_row_store.row_abs_max.device,
                        compute_dtype=planner_compute_dtype,
                    )
                    if cross_cluster_debug_summary is not None:
                        normalization_input_stats = _build_phase4_normalization_stats(
                            (
                                row_denominator_prefix[0].detach().cpu(),
                                row_denominator_prefix[1].detach().cpu(),
                            ),
                        )
                        row_store_snapshot = feature_row_store.get_diagnostic_snapshot()
                else:
                    planner_influences = compute_partial_influences(
                        edge_matrix[:pre_phase4_st].to(dtype=planner_compute_dtype),
                        targets.logit_probabilities.to(dtype=planner_compute_dtype),
                        row_to_node_index[:pre_phase4_st],
                        device=torch.device("cpu"),
                    )
                    seed_feature_influences = planner_influences[:total_active_feats]
                    if cross_cluster_debug_summary is not None:
                        normalization_input_stats = _build_phase4_normalization_stats(
                            edge_matrix[:pre_phase4_st, :logit_offset]
                            .abs()
                            .sum(dim=1)
                            .detach()
                            .cpu(),
                        )

                unvisited_feature_rank = torch.argsort(
                    seed_feature_influences,
                    descending=True,
                ).cpu()
                candidate_scores = seed_feature_influences[unvisited_feature_rank].detach().cpu()
                phase3_frontier_buffer_metadata = _build_phase3_frontier_buffer_metadata(
                    seed_feature_influences=seed_feature_influences,
                    base_max_feature_nodes=int(base_max_feature_nodes),
                    total_active_features=int(total_active_feats),
                    relative_epsilon=phase3_frontier_buffer_relative_epsilon,
                    max_extra=int(phase3_frontier_buffer_max_extra),
                )
                actual_max_feature_nodes = int(
                    phase3_frontier_buffer_metadata["actual_max_feature_nodes"]
                )
                phase3_seed_summary["phase3_frontier_buffer_metadata"] = (
                    phase3_frontier_buffer_metadata
                )
                queue_size = min(
                    _compute_phase4_refresh_queue_window_size(
                        update_interval=update_interval,
                        phase4_feature_batch_size=effective_feature_batch_size,
                        queue_multiplier=phase4_refresh_policy_config.effective_queue_multiplier,
                    ),
                    int(actual_max_feature_nodes),
                )
                pre_locality_pending = unvisited_feature_rank[:queue_size]
                post_locality_pending = _reorder_pending_for_phase4_locality(
                    pre_locality_pending,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=getattr(model.transcoders, "decoder_chunk_size", None),
                )
                phase3_seed_summary.update(
                    {
                        "status": "captured",
                        "queue_size": int(queue_size),
                    }
                )

                if capture_phase3_seed_bundle_enabled:
                    phase3_seed_bundle_payload = _build_phase3_seed_bundle_payload(
                        active_features=activation_matrix.indices().T,
                        activation_values=activation_matrix.values(),
                        seed_feature_influences=seed_feature_influences,
                        frontier_pre_locality=pre_locality_pending,
                        frontier_post_locality=post_locality_pending,
                        queue_size=queue_size,
                        actual_max_feature_nodes=int(actual_max_feature_nodes),
                        total_active_features=int(total_active_feats),
                        status="captured",
                        planner_compute_dtype=planner_compute_dtype,
                        influence_compute_dtype=influence_compute_dtype,
                    )
                if capture_feature_semantic_descriptors_enabled:
                    feature_semantic_descriptors_payload = (
                        _build_feature_semantic_descriptors_payload(
                            active_features=activation_matrix.indices().T,
                            activation_values=activation_matrix.values(),
                            seed_feature_influences=seed_feature_influences,
                            frontier_pre_locality=pre_locality_pending,
                            frontier_post_locality=post_locality_pending,
                            total_active_features=int(total_active_feats),
                            status="captured",
                            semantic_descriptor_top_k=semantic_descriptor_top_k,
                            semantic_descriptor_dim=semantic_descriptor_dim,
                        )
                    )

                if cross_cluster_debug_summary is not None:
                    deterministic_pending = _build_phase4_deterministic_shadow_pending(
                        unvisited_feature_rank,
                        seed_feature_influences.detach().cpu(),
                        queue_size=queue_size,
                        feat_layers=feat_layers,
                        feat_positions=feat_pos,
                        feat_ids=feat_ids,
                        exact_chunked_decoder=exact_chunked_decoder,
                        decoder_chunk_size=getattr(model.transcoders, "decoder_chunk_size", None),
                    )
                    seed_cutoff_debug = _build_phase4_cutoff_debug(
                        candidate_scores,
                        queue_size=queue_size,
                    )
                    seed_influence_topk = _build_phase3_seed_influence_topk(
                        ranked_feature_indices=unvisited_feature_rank,
                        seed_feature_influences=seed_feature_influences,
                        feat_layers=feat_layers,
                        feat_positions=feat_pos,
                        feat_ids=feat_ids,
                        top_k=8,
                    )

                    phase3_seed_summary.update(
                        {
                            "feature_influence_stats": _build_vector_stats(
                                seed_feature_influences.detach().cpu(),
                                epsilon=1e-12,
                                top_k=8,
                            ),
                            "feature_influence_hash": _hash_float_tensor(
                                seed_feature_influences.detach().cpu(),
                                dtype=torch.float64,
                            ),
                            "frontier_pre_locality_hash": _hash_index_tensor(pre_locality_pending),
                            "frontier_post_locality_hash": _hash_index_tensor(
                                post_locality_pending
                            ),
                            "frontier_pre_locality_sample": [
                                int(v) for v in pre_locality_pending[:16].tolist()
                            ],
                            "frontier_post_locality_sample": [
                                int(v) for v in post_locality_pending[:16].tolist()
                            ],
                            "seed_influence_topk": seed_influence_topk,
                            "seed_influence_topk_hash": _hash_json_payload(seed_influence_topk),
                            "seed_cutoff": seed_cutoff_debug,
                            "deterministic_shadow": _compare_phase4_frontiers(
                                post_locality_pending,
                                deterministic_pending,
                            ),
                            "normalization_input_stats": normalization_input_stats,
                            "feature_row_store_summary": row_store_snapshot,
                        }
                    )

                    if shadow_debug_compute_dtype != planner_compute_dtype:
                        if use_compact_feature_row_store:
                            assert feature_row_store is not None
                            shadow_feature_influences = (
                                compute_partial_feature_influences_streaming(
                                    lambda row_start, row_end: feature_row_store.read_feature_rows(
                                        row_start,
                                        row_end,
                                        phase="phase3_seed_ranking_shadow",
                                    ),
                                    (
                                        feature_row_store.row_abs_max[:pre_phase4_st],
                                        feature_row_store.row_l1_scaled[:pre_phase4_st],
                                    ),
                                    targets.logit_probabilities,
                                    row_to_node_index[:pre_phase4_st],
                                    n_feature_nodes=total_active_feats,
                                    n_logits=n_logits,
                                    device=torch.device("cpu"),
                                    compute_dtype=shadow_debug_compute_dtype,
                                )
                            )
                        else:
                            shadow_influences = compute_partial_influences(
                                edge_matrix[:pre_phase4_st].to(dtype=shadow_debug_compute_dtype),
                                targets.logit_probabilities.to(dtype=shadow_debug_compute_dtype),
                                row_to_node_index[:pre_phase4_st],
                                device=torch.device("cpu"),
                            )
                            shadow_feature_influences = shadow_influences[:total_active_feats]
                        shadow_rank = torch.argsort(
                            shadow_feature_influences,
                            descending=True,
                        ).cpu()
                        shadow_pending = _reorder_pending_for_phase4_locality(
                            shadow_rank[:queue_size],
                            feat_layers=feat_layers,
                            feat_positions=feat_pos,
                            feat_ids=feat_ids,
                            exact_chunked_decoder=exact_chunked_decoder,
                            decoder_chunk_size=getattr(
                                model.transcoders, "decoder_chunk_size", None
                            ),
                        )
                        phase3_seed_summary["shadow_debug"] = _compare_phase4_frontiers(
                            post_locality_pending,
                            shadow_pending,
                        )
            else:
                phase3_frontier_buffer_metadata["status"] = "skipped_all_features_included"
                phase3_frontier_buffer_metadata["fallback_reason"] = "all_features_included"
                phase3_seed_summary.update(
                    {
                        "status": "skipped_all_features_included",
                        "queue_size": int(actual_max_feature_nodes),
                    }
                )
                if capture_phase3_seed_bundle_enabled:
                    phase3_seed_bundle_payload = _build_phase3_seed_bundle_payload(
                        active_features=activation_matrix.indices().T,
                        activation_values=activation_matrix.values(),
                        seed_feature_influences=torch.empty(
                            0,
                            dtype=planner_compute_dtype,
                        ),
                        frontier_pre_locality=torch.empty(0, dtype=torch.long),
                        frontier_post_locality=torch.empty(0, dtype=torch.long),
                        queue_size=int(actual_max_feature_nodes),
                        actual_max_feature_nodes=int(actual_max_feature_nodes),
                        total_active_features=int(total_active_feats),
                        status="skipped_all_features_included",
                        planner_compute_dtype=planner_compute_dtype,
                        influence_compute_dtype=influence_compute_dtype,
                    )
                if capture_feature_semantic_descriptors_enabled:
                    feature_semantic_descriptors_payload = (
                        _build_feature_semantic_descriptors_payload(
                            active_features=activation_matrix.indices().T,
                            activation_values=activation_matrix.values(),
                            seed_feature_influences=torch.empty(0, dtype=planner_compute_dtype),
                            frontier_pre_locality=torch.empty(0, dtype=torch.long),
                            frontier_post_locality=torch.empty(0, dtype=torch.long),
                            total_active_features=int(total_active_feats),
                            status="skipped_all_features_included",
                            semantic_descriptor_top_k=semantic_descriptor_top_k,
                            semantic_descriptor_dim=semantic_descriptor_dim,
                        )
                    )
            if cross_cluster_debug_summary is not None:
                deterministic_shadow = phase3_seed_summary.get("deterministic_shadow")
                shadow_debug = phase3_seed_summary.get("shadow_debug")
                normalization_input_stats = phase3_seed_summary.get("normalization_input_stats")
                feature_influence_stats = phase3_seed_summary.get("feature_influence_stats")
                phase3_stream_checkpoint = {
                    "status": phase3_seed_summary.get("status"),
                    "stored_row_count_before_phase4": int(pre_phase4_st),
                    "actual_max_feature_nodes": int(actual_max_feature_nodes),
                    "total_active_features": int(total_active_feats),
                    "update_interval": int(update_interval),
                    "feature_batch_size": int(effective_feature_batch_size),
                    "queue_size": phase3_seed_summary.get("queue_size"),
                    "feature_influence_hash": phase3_seed_summary.get("feature_influence_hash"),
                    "frontier_pre_locality_hash": phase3_seed_summary.get(
                        "frontier_pre_locality_hash"
                    ),
                    "frontier_post_locality_hash": phase3_seed_summary.get(
                        "frontier_post_locality_hash"
                    ),
                    "deterministic_shadow_overlap_fraction": (
                        _safe_float(deterministic_shadow.get("overlap_fraction"))
                        if isinstance(deterministic_shadow, dict)
                        else None
                    ),
                    "deterministic_shadow_jaccard": (
                        _safe_float(deterministic_shadow.get("jaccard_similarity"))
                        if isinstance(deterministic_shadow, dict)
                        else None
                    ),
                    "deterministic_shadow_prefix_match_count": (
                        int(deterministic_shadow.get("prefix_match_count", 0))
                        if isinstance(deterministic_shadow, dict)
                        else None
                    ),
                    "shadow_debug_overlap_fraction": (
                        _safe_float(shadow_debug.get("overlap_fraction"))
                        if isinstance(shadow_debug, dict)
                        else None
                    ),
                    "seed_influence_topk_hash": phase3_seed_summary.get("seed_influence_topk_hash"),
                    "seed_cutoff_margin": (
                        _safe_float(phase3_seed_summary.get("seed_cutoff", {}).get("cutoff_margin"))
                        if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                        else None
                    ),
                    "seed_cutoff_near_tie_count": (
                        int(phase3_seed_summary.get("seed_cutoff", {}).get("near_cutoff_count", 0))
                        if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                        else None
                    ),
                    "seed_cutoff_exact_tie_count": (
                        int(phase3_seed_summary.get("seed_cutoff", {}).get("exact_cutoff_count", 0))
                        if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                        else None
                    ),
                    "feature_influence_nonfinite_count": (
                        int(feature_influence_stats.get("nonfinite_count", 0))
                        if isinstance(feature_influence_stats, dict)
                        else None
                    ),
                    "feature_influence_abs_sum": (
                        _safe_float(feature_influence_stats.get("abs_sum"))
                        if isinstance(feature_influence_stats, dict)
                        else None
                    ),
                    "normalization_clamped_row_count": (
                        int(normalization_input_stats.get("clamped_row_count", 0))
                        if isinstance(normalization_input_stats, dict)
                        else None
                    ),
                    "normalization_clamped_row_fraction": (
                        _safe_float(normalization_input_stats.get("clamped_row_fraction"))
                        if isinstance(normalization_input_stats, dict)
                        else None
                    ),
                    **phase3_runtime_stream,
                }
                _record_cross_cluster_checkpoint(
                    cross_cluster_debug_summary=cross_cluster_debug_summary,
                    cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                    checkpoint_name="phase3_seed_ranking_pre_phase4",
                    phase="phase3",
                    summary_payload=phase3_seed_summary,
                    stream_payload=phase3_stream_checkpoint,
                )

        # Phase 4: feature attribution
        logger.info("Phase 4: Computing feature attributions")
        phase4_start = time.perf_counter()
        phase4_frontier_buffer_metadata["initial_target_feature_nodes"] = int(
            actual_max_feature_nodes
        )
        phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] = int(
            actual_max_feature_nodes
        )
        feature_rows_cpu_staging: torch.Tensor | None = None
        _log_memory_boundary(logger, "Phase 4 start", model.device)
        decoder_chunk_size = getattr(model.transcoders, "decoder_chunk_size", None)
        phase4_feature_batch_size = effective_feature_batch_size
        phase4_refresh_queue_multiplier = int(
            phase4_refresh_policy_config.effective_queue_multiplier
        )
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
        phase4_executor_microbatch_size = phase4_executor_reference_batch_size
        if phase4_row_executor_effective_mode == "streaming_v1":
            phase4_executor_microbatch_size = _resolve_phase4_streaming_v1_microbatch_size(
                phase4_executor_reference_batch_size
            )
        phase4_execution_metadata.update(
            {
                "executor_configured_reference_batch_size": int(
                    phase4_executor_reference_batch_size
                ),
                "executor_reference_batch_size": int(phase4_executor_reference_batch_size),
                "executor_microbatch_size": int(phase4_executor_microbatch_size),
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
                    phase4_refresh_row_store_read_elapsed_ms_total += (
                        refresh_row_store_read_elapsed_ms
                    )
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
                    "ranker_frontier_max_feature_nodes_cap_bound": bool(
                        max_feature_nodes_cap_bound
                    ),
                    "ranker_frontier_tie_count_at_cutoff": int(rank_selection.tie_count_at_cutoff),
                    "ranker_frontier_tie_at_cutoff": bool(rank_selection.tie_at_cutoff),
                    "ranker_frontier_tie_behavior": rank_selection.tie_behavior,
                }
                if (
                    (
                        cross_cluster_debug_enabled
                        or phase4_scheduler_config.telemetry_detail == "debug"
                    )
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
                    ranker_refresh_telemetry["ranker_frontier_near_cutoff_counts"] = (
                        near_cutoff_counts
                    )
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
                refresh_frontier_plan_elapsed_ms = (
                    time.perf_counter() - frontier_plan_start
                ) * 1000.0
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
                        "pending_hash": _hash_index_tensor(pending)
                        if pending.numel() > 0
                        else None,
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
                                float(
                                    (feature_row_store_read_stats or {}).get("read_row_count") or 0
                                )
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
                            (feature_row_store_read_stats or {}).get(
                                "read_cache_store_success_count"
                            )
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
                                    (streaming_chunk_reuse_stats or {}).get("row_reader_row_count")
                                    or 0
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
                            (streaming_chunk_reuse_stats or {}).get(
                                "chunk_cache_store_success_count"
                            )
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
                                feature_row_store.row_abs_max[:st].to(
                                    dtype=shadow_debug_compute_dtype
                                ),
                                feature_row_store.row_l1_scaled[:st].to(
                                    dtype=shadow_debug_compute_dtype
                                ),
                            )
                            float64_feature_influences = (
                                compute_partial_feature_influences_streaming(
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
                            float32_feature_influences = (
                                compute_partial_feature_influences_streaming(
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

                if phase4_row_executor_effective_mode == "streaming_v1":
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
                    transcoder_before = (
                        _snapshot_diagnostics(model.transcoders) if profile else None
                    )
                    batch_start = time.perf_counter()
                    batch_memory_before = get_memory_snapshot(model.device)
                    encoder_vectors_source_device = None
                    encoder_vectors_source_dtype = None
                    if (
                        getattr(ctx, "encoder_vecs", None) is not None
                        and ctx.encoder_vecs.numel() > 0
                    ):
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
                    if (
                        encoder_vectors_source_device == "cpu"
                        and encoder_vectors.device.type == "cuda"
                    ):
                        phase4_cpu_to_gpu_bytes_total += int(encoder_vectors_transfer_bytes)
                    compute_batch_start = time.perf_counter()
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
                    if phase4_row_reduction_config.effective_mode == "gpu_v1":
                        if not use_compact_feature_row_store:
                            raise RuntimeError(
                                "phase4_row_reduction='gpu_v1' requires compact Phase-4 row store"
                            )
                        cpu_staging_start = time.perf_counter()
                        feature_row_slice, feature_rows_cpu_staging = (
                            _copy_feature_rows_to_cpu_staging(
                                rows,
                                total_active_feats=total_active_feats,
                                staging_buffer=feature_rows_cpu_staging,
                            )
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
                    if use_compact_feature_row_store:
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
                    else:
                        assert phase4_row_reduction_config.effective_mode == "off"
                        row_store_write_start = time.perf_counter()
                        edge_matrix[st:end, :logit_offset] = rows_cpu
                        executor_row_store_write_elapsed_ms = (
                            time.perf_counter() - row_store_write_start
                        ) * 1000.0
                        row_store_append_telemetry = None
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
                    phase4_executor_compute_batch_elapsed_ms_total += (
                        executor_compute_batch_elapsed_ms
                    )
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
                        "executor_streaming_chunk_index": (
                            int(streaming_chunk_index)
                            if phase4_row_executor_effective_mode == "streaming_v1"
                            else None
                        ),
                        "executor_streaming_chunk_count": (
                            int(streaming_chunk_count)
                            if phase4_row_executor_effective_mode == "streaming_v1"
                            else None
                        ),
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
                                "idx_batch_hash": batch_locality_summary.get(
                                    "scheduler_batch_hash"
                                ),
                                "row_input_nonfinite_count": int(
                                    row_input_stats["nonfinite_count"]
                                ),
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
            if planned_boundaries is not None and planned_boundary_offset != len(
                planned_boundaries
            ):
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
                float(
                    record["feature_row_store_read_stats"]["read_cache_store_skip_disabled_count"]
                )
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("feature_row_store_read_stats"), dict)
                and record["feature_row_store_read_stats"].get(
                    "read_cache_store_skip_disabled_count"
                )
                is not None
            ]
            feature_row_store_cache_skip_too_large = [
                float(
                    record["feature_row_store_read_stats"]["read_cache_store_skip_too_large_count"]
                )
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("feature_row_store_read_stats"), dict)
                and record["feature_row_store_read_stats"].get(
                    "read_cache_store_skip_too_large_count"
                )
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
                float(
                    record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_disabled_count"]
                )
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
                and record["streaming_chunk_reuse_stats"].get(
                    "chunk_cache_store_skip_disabled_count"
                )
                is not None
            ]
            streaming_chunk_cache_skip_too_large = [
                float(
                    record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_too_large_count"]
                )
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("streaming_chunk_reuse_stats"), dict)
                and record["streaming_chunk_reuse_stats"].get(
                    "chunk_cache_store_skip_too_large_count"
                )
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
                    int(records[0]["pending_size"])
                    if records and isinstance(records[0], dict)
                    else 0
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
                "rank_signal_effectively_all_zero_refresh_count": int(
                    rank_effectively_all_zero_count
                ),
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
                    len(phase3_logit_row_batches)
                    if isinstance(phase3_logit_row_batches, list)
                    else 0
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
                    first_phase3_logit_batch.get("row_abs_sum_stats", {}).get(
                        "effectively_all_zero"
                    )
                    if isinstance(first_phase3_logit_batch, dict)
                    else None
                ),
                "phase3_logit_row_batch_0_row_l1_nonfinite_count": (
                    first_phase3_logit_batch.get("row_abs_sum_stats", {}).get("nonfinite_count")
                    if isinstance(first_phase3_logit_batch, dict)
                    else None
                ),
            }

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
        if feature_row_store is not None:
            feature_row_store.cleanup()
        if nonfeature_row_store is not None:
            nonfeature_row_store.cleanup()
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
                "feature_row_store": feature_row_store is not None,
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
