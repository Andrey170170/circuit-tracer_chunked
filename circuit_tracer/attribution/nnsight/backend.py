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
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup

from circuit_tracer.graph import (
    Graph,
)
from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    require_exact_chunked_provider,
    require_exact_row_replay_provider,
)
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.tracing.plan import ResolvedTracePlan
from circuit_tracer.tracing.problem import AttributionProblem

from circuit_tracer.observability.human_logs import _log_memory_boundary
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _build_exact_encoder_residency_metadata as _build_exact_encoder_residency_metadata,
    _build_phase4_ranker_metadata as _build_phase4_ranker_metadata,
    _build_phase4_refresh_optimization_metadata as _build_phase4_refresh_optimization_metadata,
    _build_phase4_refresh_policy_metadata as _build_phase4_refresh_policy_metadata,
    _build_phase4_row_executor_metadata as _build_phase4_row_executor_metadata,
    _build_phase4_row_reduction_metadata as _build_phase4_row_reduction_metadata,
    _build_phase4_scheduler_metadata as _build_phase4_scheduler_metadata,
    _plan_phase4_feature_batch_size_preflight as _plan_phase4_feature_batch_size_preflight,
    _resolve_exact_encoder_residency_config as _resolve_exact_encoder_residency_config,
    _resolve_phase4_feature_batch_planner_enabled as _resolve_phase4_feature_batch_planner_enabled,
    _resolve_phase4_feature_batch_planner_status as _resolve_phase4_feature_batch_planner_status,
    _resolve_phase4_ranker_config as _resolve_phase4_ranker_config,
    _resolve_phase4_refresh_optimization_config as _resolve_phase4_refresh_optimization_config,
    _resolve_phase4_refresh_policy_config as _resolve_phase4_refresh_policy_config,
    _resolve_phase4_row_executor_config as _resolve_phase4_row_executor_config,
    _resolve_phase4_row_reduction_config as _resolve_phase4_row_reduction_config,
    _resolve_phase4_scheduler_config as _resolve_phase4_scheduler_config,
    _resolve_phase4_streaming_v1_microbatch_size as _resolve_phase4_streaming_v1_microbatch_size,
)
from circuit_tracer.attribution.nnsight.phase1_policy import (
    _build_phase1_trace_batch_metadata as _build_phase1_trace_batch_metadata,
    _build_phase1_trace_batch_sizing_metadata as _build_phase1_trace_batch_sizing_metadata,
    _resolve_phase1_trace_batch_config as _resolve_phase1_trace_batch_config,
    _resolve_phase1_trace_batch_sizing as _resolve_phase1_trace_batch_sizing,
)
from circuit_tracer.attribution.nnsight.session_controls import (
    resolve_nnsight_session_controls,
    validate_nnsight_session_control_requests,
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
from circuit_tracer.attribution.nnsight.phases.phase5 import (
    Phase5Config,
    Phase5Inputs,
    run_phase5,
)
from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata as PrefixViewMetadata,
    _resolve_prefix_view_output_position as _resolve_prefix_view_output_position,
    validate_prefix_view_metadata as validate_prefix_view_metadata,
)
from circuit_tracer.attribution.nnsight.numerics import (
    _exact_trace_internal_dtype_name as _exact_trace_internal_dtype_name,
    _resolve_exact_trace_internal_dtype as _resolve_exact_trace_internal_dtype,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _resolve_internal_precision_requested as _resolve_internal_precision_requested,
    _warn_internal_precision_deprecated as _warn_internal_precision_deprecated,
    _resolve_internal_dtype_map as _resolve_internal_dtype_map,
    _dtype_from_name as _dtype_from_name,
    _build_phase4_environment_fingerprint as _build_phase4_environment_fingerprint,
)
from circuit_tracer.attribution.nnsight.replay import (
    _build_phase0_replay_metadata as _build_phase0_replay_metadata,
    _build_phase3_replay_metadata as _build_phase3_replay_metadata,
    _resolve_phase0_donor_context_policy as _resolve_phase0_donor_context_policy,
    _resolve_phase0_replay_mode as _resolve_phase0_replay_mode,
    _resolve_phase3_replay_mode as _resolve_phase3_replay_mode,
    _resolve_phase3_replay_validation_policy as _resolve_phase3_replay_validation_policy,
)
from circuit_tracer.attribution.nnsight.row_store import (
    _FileBackedFeatureRowStore as _FileBackedFeatureRowStore,
    _RowStoreCacheControlMode as _RowStoreCacheControlMode,
    _build_row_store_cache_control_metadata as _build_row_store_cache_control_metadata,
    _resolve_row_store_cache_control_config as _resolve_row_store_cache_control_config,
    _resolve_row_store_temp_root_policy as _resolve_row_store_temp_root_policy,
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


def _raise_cleanup_failures(cleanup_failures: Sequence[BaseException]) -> None:
    if not cleanup_failures:
        return
    if all(isinstance(error, Exception) for error in cleanup_failures):
        raise ExceptionGroup(
            "Attribution lifecycle cleanup failed",
            [error for error in cleanup_failures if isinstance(error, Exception)],
        )
    raise BaseExceptionGroup("Attribution lifecycle cleanup failed", list(cleanup_failures))


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


@dataclass(frozen=True)
class _ForwardOverrides:
    phase0_context: object | None = None
    target_logits: torch.Tensor | None = None
    target_logit_source: str | None = None
    decoder_chunk_cache: object | None = None
    decoder_cache_fingerprint: object | None = None


def run_nnsight_trace(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    *,
    forward_overrides: _ForwardOverrides | None = None,
) -> Graph | dict[str, object]:
    """Execute one resolved plan while owning logging and offload cleanup."""

    observability = plan.execution.observability
    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if (observability.verbose or observability.profile) and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    prefix_view_metadata = validate_prefix_view_metadata(
        prompt=problem.prompt,
        attribution_targets=problem.targets,
        prefix_view_metadata=plan.evidence_metadata.get("prefix_view_metadata"),
    )
    output_position = _resolve_prefix_view_output_position(
        prefix_view_metadata,
        problem.output_position,
    )
    offload_handles: list[Any] = []
    try:
        return _run_attribution(
            problem=problem,
            plan=plan,
            logger=logger,
            offload_handles=offload_handles,
            forward_overrides=forward_overrides or _ForwardOverrides(),
            prefix_view_metadata=prefix_view_metadata,
            output_position=output_position,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()
        if handler is not None:
            logger.removeHandler(handler)


def _run_attribution(
    *,
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    logger: logging.Logger,
    offload_handles: list[Any],
    forward_overrides: _ForwardOverrides,
    prefix_view_metadata: PrefixViewMetadata | None,
    output_position: int | None,
):
    """Resolve backend mechanisms, then execute Phases 0-5 in canonical order."""

    model = problem.model
    prompt = problem.prompt
    attribution_targets = problem.targets
    max_n_logits = problem.max_n_logits
    desired_logit_prob = problem.desired_logit_prob
    semantics = plan.semantics
    execution = plan.execution
    session = execution.session
    storage = execution.storage
    replay = execution.replay
    frontier = execution.frontier
    observability = execution.observability

    batch_size = semantics.source_batch_size
    feature_batch_size = semantics.feature_batch_size
    logit_batch_size = semantics.logit_batch_size
    max_feature_nodes = semantics.max_feature_nodes
    update_interval = semantics.update_interval
    diagnostic_feature_cap = semantics.diagnostic_feature_cap
    sparsification = semantics.sparsification
    exact_trace_internal_dtype = semantics.exact_trace_internal_dtype
    phase0_activation_threshold_compare_mode = semantics.phase0_activation_threshold_compare_mode

    nnsight_session_capacity = session.capacity
    phase3_compute_microbatch_max_rows = session.phase3_microbatch_max_rows
    phase4_compute_microbatch_max_rows = session.phase4_microbatch_max_rows
    phase1_trace_batch_policy = session.phase1_trace_batch_policy
    phase1_trace_batch_size_max = session.phase1_trace_batch_size_max

    full_retention_backend = storage.full_retention_backend
    feature_row_column_tile_size = storage.feature_column_tile_size
    influence_row_tile_size = storage.influence_row_tile_size
    influence_column_tile_size = storage.influence_column_tile_size
    row_store_cache_control = storage.cache_control
    row_store_temp_root_policy = storage.temp_root_policy
    row_store_temp_root = storage.temp_root
    row_store_preallocate = storage.preallocate
    feature_row_retention = storage.retention
    replay_tile_cache_bytes = storage.replay_tile_cache_bytes
    exact_encoder_residency = storage.exact_encoder_residency

    chunked_feature_replay_window = replay.feature_window
    error_vector_prefetch_lookahead = replay.error_vector_prefetch_lookahead
    stage_encoder_vecs_on_cpu = replay.stage_encoder_vecs_on_cpu
    stage_error_vectors_on_cpu = replay.stage_error_vectors_on_cpu
    row_subchunk_size = replay.decoder_contraction_tile
    phase0_donor_bundle = replay.phase0_donor_bundle
    phase0_replay_mode = replay.phase0_mode
    phase0_donor_context_policy = replay.phase0_donor_context_policy
    phase3_gradient_donor_bundle = replay.phase3_gradient_donor_bundle
    phase3_gradient_replay_mode = replay.phase3_gradient_mode
    phase3_row_donor_bundle = replay.phase3_row_donor_bundle
    phase3_row_replay_mode = replay.phase3_row_mode
    phase3_replay_validation_policy = replay.phase3_validation_policy

    phase4_scheduler_mode = frontier.scheduler
    phase4_scheduler_debug = frontier.scheduler_debug
    phase4_scheduler_telemetry_detail = frontier.scheduler_telemetry_detail
    phase4_refresh_optimization = frontier.refresh_optimization
    phase4_refresh_prepared_chunk_cache_bytes = frontier.refresh_prepared_chunk_cache_bytes
    phase4_refresh_active_row_accumulation = frontier.refresh_active_row_accumulation
    phase4_row_executor = frontier.row_executor
    phase4_row_reduction = frontier.row_reduction
    phase4_refresh_policy = frontier.refresh_policy
    phase4_refresh_interval_multiplier = frontier.refresh_interval_multiplier
    phase4_ranker = frontier.ranker
    plan_feature_batch_size = frontier.feature_batch_planning
    feature_batch_size_max = frontier.feature_batch_size_max
    feature_batch_target_reserved_fraction = frontier.feature_batch_target_reserved_fraction
    feature_batch_min_free_fraction = frontier.feature_batch_min_free_fraction
    feature_batch_probe_batches = frontier.feature_batch_probe_batches
    phase3_frontier_buffer_relative_epsilon = frontier.phase3_frontier_buffer_relative_epsilon
    phase3_frontier_buffer_max_extra = frontier.phase3_frontier_buffer_max_extra
    phase4_frontier_buffer_relative_epsilon = frontier.phase4_frontier_buffer_relative_epsilon
    phase4_frontier_buffer_max_extra_per_refresh = frontier.phase4_frontier_buffer_max_extra_per_refresh
    phase4_frontier_buffer_max_extra_total = frontier.phase4_frontier_buffer_max_extra_total

    offload = execution.offload
    compact_output = execution.compact_output
    verbose = observability.verbose
    profile = observability.profile
    profile_log_interval = observability.profile_log_interval
    phase4_anomaly_debug = observability.phase4_anomaly_debug
    cross_cluster_debug = observability.cross_cluster_debug
    capture_phase0_donor_bundle = observability.capture_phase0_donor_bundle
    capture_phase3_seed_bundle = observability.capture_phase3_seed_bundle
    capture_phase3_gradient_bundle = observability.capture_phase3_gradient_bundle
    capture_phase3_row_bundle = observability.capture_phase3_row_bundle
    capture_feature_semantic_descriptors = observability.capture_feature_semantic_descriptors
    semantic_descriptor_top_k = observability.semantic_descriptor_top_k
    semantic_descriptor_dim = observability.semantic_descriptor_dim
    telemetry_max_events = observability.telemetry_max_events
    telemetry_jsonl_path = observability.telemetry_jsonl_path
    telemetry_context = dict(observability.telemetry_context)
    telemetry_context["runtime_plan"] = {
        "schema_version": 3,
        "semantic_fingerprint": plan.semantic_fingerprint,
        "execution_fingerprint": plan.execution_fingerprint,
    }

    auto_scale_feature_batch_size = False
    internal_precision = None
    decoder_chunk_cache = forward_overrides.decoder_chunk_cache
    decoder_cache_fingerprint = forward_overrides.decoder_cache_fingerprint
    phase0_context_override = forward_overrides.phase0_context
    target_logits_override = forward_overrides.target_logits
    target_logit_source = forward_overrides.target_logit_source
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
    if full_retention_backend not in ("full_file", "column_tiled_v1"):
        raise ValueError("full_retention_backend must be 'full_file' or 'column_tiled_v1'")
    if min(feature_row_column_tile_size, influence_row_tile_size, influence_column_tile_size) <= 0:
        raise ValueError("feature/influence row and column tile sizes must be > 0")
    if full_retention_backend == "column_tiled_v1" and not compact_output:
        raise ValueError("column_tiled_v1 requires compact_output=True; refusing fallback")
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

    if feature_row_retention not in ("full_file", "none_recompute"):
        raise ValueError("feature_row_retention must be 'full_file' or 'none_recompute'")
    if replay_tile_cache_bytes is not None and replay_tile_cache_bytes < 0:
        raise ValueError("replay_tile_cache_bytes must be >= 0 when provided")
    if feature_row_retention == "none_recompute":
        if not compact_output:
            raise ValueError("none_recompute requires compact_output=True")
        require_exact_row_replay_provider(model.transcoders)
    tiled_or_recompute = bool(
        full_retention_backend == "column_tiled_v1"
        or feature_row_retention == "none_recompute"
    )
    if tiled_or_recompute and (
        phase3_gradient_replay_mode_resolved != "disabled"
        or phase3_row_replay_mode_resolved != "disabled"
        or capture_phase3_gradient_bundle
        or capture_phase3_row_bundle
    ):
        raise ValueError(
            "column-tiled and none_recompute row production do not yet support "
            "Phase-3 donor replay or gradient/row capture"
        )
    if full_retention_backend == "column_tiled_v1" and row_store_preallocate:
        raise ValueError(
            "column_tiled_v1 does not support row_store_preallocate=True; "
            "pass row_store_preallocate=False"
        )

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
    validate_nnsight_session_control_requests(
        nnsight_session_capacity=nnsight_session_capacity,
        phase3_compute_microbatch_max_rows=phase3_compute_microbatch_max_rows,
        phase4_compute_microbatch_max_rows=phase4_compute_microbatch_max_rows,
    )
    planner_enabled = _resolve_phase4_feature_batch_planner_enabled(
        plan_feature_batch_size=plan_feature_batch_size,
        auto_scale_feature_batch_size=auto_scale_feature_batch_size,
    )
    if (not planner_enabled) and max_phase4_feature_batch_size < effective_feature_batch_size:
        raise ValueError("feature_batch_size_max must be >= the effective feature batch size")
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
    compact_exact_provider = compact_output and exact_chunked_decoder
    compact_row_store_provider = compact_output and supports_compact_row_store
    preflight_requirements = (
        (
            full_retention_backend == "column_tiled_v1",
            compact_row_store_provider and transcoder_capabilities.architecture in ("clt", "plt"),
            "column_tiled_v1 true row production",
        ),
        (phase4_anomaly_debug_enabled, compact_exact_provider, "Phase-4 anomaly debug"),
        (cross_cluster_debug_enabled, compact_exact_provider, "cross_cluster_debug"),
        (
            capture_phase0_donor_bundle_enabled,
            compact_exact_provider,
            "capture_phase0_donor_bundle",
        ),
        (capture_phase3_seed_bundle_enabled, compact_exact_provider, "capture_phase3_seed_bundle"),
        (
            capture_phase3_gradient_bundle_enabled,
            compact_exact_provider,
            "capture_phase3_gradient_bundle",
        ),
        (capture_phase3_row_bundle_enabled, compact_exact_provider, "capture_phase3_row_bundle"),
        (
            capture_feature_semantic_descriptors_enabled,
            compact_exact_provider,
            "capture_feature_semantic_descriptors",
        ),
        (phase0_replay_mode_resolved != "disabled", compact_exact_provider, "phase0 donor replay"),
        (
            phase3_gradient_replay_mode_resolved != "disabled",
            compact_exact_provider,
            "Phase-3 gradient replay",
        ),
        (
            phase3_row_replay_mode_resolved != "disabled",
            compact_exact_provider,
            "Phase-3 row replay",
        ),
        (planner_enabled, compact_row_store_provider, "Phase-4 feature batch planner"),
    )
    for requested, supported, label in preflight_requirements:
        if requested and not supported:
            raise ValueError(f"{label} requires compact_output=True and exact provider support")
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
    attribute_start_attrs = {
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
        "full_retention_backend_requested": full_retention_backend,
        "full_retention_backend_effective": full_retention_backend,
        "feature_row_retention_requested": feature_row_retention,
        "feature_row_retention_effective": feature_row_retention,
        "replay_tile_cache_bytes_requested": int(replay_tile_cache_bytes or 0),
        "replay_tile_cache_bytes_effective": int(replay_tile_cache_bytes or 0),
        "row_store_preallocate_requested": bool(row_store_preallocate),
        "row_store_preallocate_effective": bool(
            row_store_preallocate and feature_row_retention == "full_file"
        ),
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
    }

    if auto_scale_feature_batch_size and not plan_feature_batch_size:
        logger.info(
            "Phase-4 feature batch planning | "
            "legacy auto_scale_feature_batch_size flag detected; "
            "using fixed preflight planner semantics"
        )
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

    legacy_session_capacity = max(
        effective_source_batch_size,
        effective_feature_batch_size,
        effective_logit_batch_size,
    )
    legacy_phase4_batch_rows = effective_feature_batch_size
    if phase4_row_executor_config.effective_mode == "streaming_v1":
        legacy_phase4_batch_rows = _resolve_phase4_streaming_v1_microbatch_size(
            effective_feature_batch_size
        )
    session_controls = resolve_nnsight_session_controls(
        nnsight_session_capacity=nnsight_session_capacity,
        phase3_compute_microbatch_max_rows=phase3_compute_microbatch_max_rows,
        phase4_compute_microbatch_max_rows=phase4_compute_microbatch_max_rows,
        legacy_session_capacity=legacy_session_capacity,
        legacy_phase3_batch_rows=effective_logit_batch_size,
        legacy_phase4_batch_rows=legacy_phase4_batch_rows,
    )
    trace_batch_size = session_controls.session_capacity
    phase1_trace_batch_metadata.update(
        trace_batch_size_legacy=int(phase1_trace_batch_sizing.trace_batch_size_legacy),
        trace_batch_size_effective=int(trace_batch_size),
    )
    if session_controls.metadata["legacy_derived_fields"]:
        telemetry_observer.run(
            name="compatibility.nnsight_session_controls",
            attrs=session_controls.metadata,
        )
    attribute_start_attrs.update(session_controls.metadata)
    attribute_start_attrs.update(
        {f"phase1_{key}": value for key, value in phase1_trace_batch_metadata.items()}
    )
    telemetry_observer.run(name="attribute.start", attrs=attribute_start_attrs)
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
    except BaseException as exc:
        if isinstance(exc, Phase0ExecutionError):
            ctx = exc.ctx
            phase0_failure = exc.cause
        else:
            phase0_failure = exc
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
                full_retention_backend=full_retention_backend,
                feature_row_column_tile_size=feature_row_column_tile_size,
                influence_row_tile_size=influence_row_tile_size,
                influence_column_tile_size=influence_column_tile_size,
                feature_row_retention=feature_row_retention,
                replay_tile_cache_bytes=int(replay_tile_cache_bytes or 0),
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
                compute_microbatch_max_rows=session_controls.phase3_microbatch_max_rows,
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
                full_retention_backend=full_retention_backend,
                influence_row_tile_size=influence_row_tile_size,
                influence_column_tile_size=influence_column_tile_size,
                feature_row_column_tile_size=feature_row_column_tile_size,
                feature_row_retention=feature_row_retention,
            ),
        )
        row_to_node_index = phase3_result.row_to_node_index
        rows_cpu_staging = phase3_result.rows_cpu_staging
        actual_max_feature_nodes = phase3_result.actual_max_feature_nodes
        phase3_frontier_buffer_metadata = phase3_result.phase3_frontier_buffer_metadata
        phase3_gradient_bundle_payload = phase3_result.phase3_gradient_bundle_payload
        phase3_row_bundle_payload = phase3_result.phase3_row_bundle_payload
        phase3_seed_bundle_payload = phase3_result.phase3_seed_bundle_payload
        feature_semantic_descriptors_payload = phase3_result.feature_semantic_descriptors_payload
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
                compute_microbatch_max_rows=session_controls.phase4_microbatch_max_rows,
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
                full_retention_backend=full_retention_backend,
                influence_row_tile_size=influence_row_tile_size,
                influence_column_tile_size=influence_column_tile_size,
                feature_row_column_tile_size=feature_row_column_tile_size,
                feature_row_retention=feature_row_retention,
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
        phase4_scheduler_reference_batch_count = (
            phase4_result.phase4_scheduler_reference_batch_count
        )
        phase4_executor_microbatch_count = phase4_result.phase4_executor_microbatch_count
        phase4_refresh_elapsed_ms_total = phase4_result.phase4_refresh_elapsed_ms_total
        phase4_feature_batch_elapsed_ms_total = phase4_result.phase4_feature_batch_elapsed_ms_total
        phase4_refresh_partial_influence_elapsed_ms_total = (
            phase4_result.phase4_refresh_partial_influence_elapsed_ms_total
        )
        phase4_refresh_rank_topk_elapsed_ms_total = (
            phase4_result.phase4_refresh_rank_topk_elapsed_ms_total
        )
        phase4_refresh_frontier_plan_elapsed_ms_total = (
            phase4_result.phase4_refresh_frontier_plan_elapsed_ms_total
        )
        phase4_refresh_row_store_read_elapsed_ms_total = (
            phase4_result.phase4_refresh_row_store_read_elapsed_ms_total
        )

        # Phase 5: packaging graph / compact output
        def publish_compact_output_result(result: dict[str, object]) -> None:
            nonlocal compact_output_result
            compact_output_result = result

        def release_dense_edge_matrix() -> None:
            nonlocal edge_matrix
            edge_matrix = None

        phase5_result = run_phase5(
            inputs=Phase5Inputs(
                logger=logger,
                model=model,
                ctx=ctx,
                targets=targets,
                telemetry_observer=telemetry_observer,
                activation_matrix=activation_matrix,
                visited=visited,
                edge_matrix=edge_matrix,
                row_to_node_index=row_to_node_index,
                input_ids=input_ids,
                feature_row_store=feature_row_store,
                nonfeature_row_store=nonfeature_row_store,
                phase0_replay_metadata=phase0_replay_metadata,
                phase3_gradient_replay_metadata=phase3_gradient_replay_metadata,
                phase3_row_replay_metadata=phase3_row_replay_metadata,
                phase3_frontier_buffer_metadata=phase3_frontier_buffer_metadata,
                phase4_frontier_buffer_metadata=phase4_frontier_buffer_metadata,
                phase4_execution_metadata=phase4_execution_metadata,
                phase0_donor_bundle_payload=phase0_donor_bundle_payload,
                phase3_seed_bundle_payload=phase3_seed_bundle_payload,
                phase3_gradient_bundle_payload=phase3_gradient_bundle_payload,
                phase3_row_bundle_payload=phase3_row_bundle_payload,
                feature_semantic_descriptors_payload=feature_semantic_descriptors_payload,
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                cross_cluster_debug_batches=cross_cluster_debug_batches,
                prefix_view_metadata=prefix_view_metadata,
                publish_compact_output_result=publish_compact_output_result,
                release_dense_edge_matrix=release_dense_edge_matrix,
            ),
            config=Phase5Config(
                compact_output=compact_output,
                use_compact_feature_row_store=use_compact_feature_row_store,
                capture_feature_semantic_descriptors_enabled=capture_feature_semantic_descriptors_enabled,
                capture_phase0_donor_bundle_enabled=capture_phase0_donor_bundle_enabled,
                capture_phase3_seed_bundle_enabled=capture_phase3_seed_bundle_enabled,
                capture_phase3_gradient_bundle_enabled=capture_phase3_gradient_bundle_enabled,
                capture_phase3_row_bundle_enabled=capture_phase3_row_bundle_enabled,
                cross_cluster_debug_enabled=cross_cluster_debug_enabled,
                phase4_anomaly_debug_enabled=phase4_anomaly_debug_enabled,
                n_pos=n_pos,
                n_logits=n_logits,
                st=st,
                total_active_feats=total_active_feats,
                total_nodes=total_nodes,
                actual_max_feature_nodes=actual_max_feature_nodes,
                batch_size=batch_size,
                feature_batch_size=feature_batch_size,
                max_phase4_feature_batch_size=max_phase4_feature_batch_size,
                planner_enabled=planner_enabled,
                planner_status=planner_status,
                planner_skip_reason=planner_skip_reason,
                phase4_scheduler_config=phase4_scheduler_config,
                phase4_refresh_optimization_config=phase4_refresh_optimization_config,
                phase4_row_executor_config=phase4_row_executor_config,
                phase4_row_reduction_config=phase4_row_reduction_config,
                phase1_trace_batch_metadata=phase1_trace_batch_metadata,
                internal_precision_requested=internal_precision_requested,
                resolved_dtype_map=resolved_dtype_map,
                phase0_activation_threshold_compare_mode_resolved=phase0_activation_threshold_compare_mode_resolved,
                exact_trace_internal_dtype_name=exact_trace_internal_dtype_name,
                telemetry_max_events_resolved=telemetry_max_events_resolved,
                semantic_descriptor_top_k=semantic_descriptor_top_k,
                semantic_descriptor_dim=semantic_descriptor_dim,
                phase4_feature_batch_size=phase4_feature_batch_size,
                phase4_executor_reference_batch_size=phase4_executor_reference_batch_size,
                phase4_executor_microbatch_size=phase4_executor_microbatch_size,
                phase4_refresh_count=phase4_refresh_count,
                phase4_scheduler_reference_batch_count=phase4_scheduler_reference_batch_count,
                phase4_executor_microbatch_count=phase4_executor_microbatch_count,
                phase4_elapsed_ms=phase4_elapsed_ms,
                phase4_refresh_elapsed_ms_total=phase4_refresh_elapsed_ms_total,
                phase4_feature_batch_elapsed_ms_total=phase4_feature_batch_elapsed_ms_total,
                phase4_refresh_partial_influence_elapsed_ms_total=phase4_refresh_partial_influence_elapsed_ms_total,
                phase4_refresh_rank_topk_elapsed_ms_total=phase4_refresh_rank_topk_elapsed_ms_total,
                phase4_refresh_frontier_plan_elapsed_ms_total=phase4_refresh_frontier_plan_elapsed_ms_total,
                phase4_refresh_row_store_read_elapsed_ms_total=phase4_refresh_row_store_read_elapsed_ms_total,
                phase4_refresh_prepared_chunk_cache_bytes=phase4_refresh_prepared_chunk_cache_bytes,
                phase4_refresh_prepared_chunk_cache_bytes_effective=phase4_refresh_prepared_chunk_cache_bytes_effective,
                phase4_refresh_active_row_accumulation=phase4_refresh_active_row_accumulation,
                phase4_refresh_active_row_accumulation_effective=phase4_refresh_active_row_accumulation_effective,
                phase4_refresh_aux_fallback_reason=phase4_refresh_aux_fallback_reason,
                phase4_refresh_aux_applicable=phase4_refresh_aux_applicable,
                start_time=start_time,
                phase0_context_override=phase0_context_override,
                target_logit_source=target_logit_source,
                target_logits_override=target_logits_override,
            ),
        )
        compact_output_result = phase5_result.compact_output_result
        edge_matrix = phase5_result.edge_matrix
        return phase5_result.output
    finally:
        _, primary_error, _ = sys.exc_info()
        cleanup_failures: list[BaseException] = []

        def attempt_cleanup(action: str, callback) -> object | None:
            try:
                return callback()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
                note = f"Phase D0 lifecycle action {action!r} failed: {cleanup_error!r}"
                if primary_error is not None:
                    try:
                        primary_error.add_note(note)
                    except BaseException:
                        pass
                logging.getLogger(__name__).error(note, exc_info=cleanup_error)
                return None

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
            attempt_cleanup("feature row-store cleanup", feature_row_store_for_cleanup.cleanup)
        if nonfeature_row_store_for_cleanup is not None:
            attempt_cleanup(
                "nonfeature row-store cleanup", nonfeature_row_store_for_cleanup.cleanup
            )
        if ctx is not None:

            def cleanup_context() -> None:
                _log_memory_boundary(logger, "Teardown start", model.device)
                cleanup = getattr(ctx, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                else:
                    clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
                    if callable(clear_decoder_cache):
                        clear_decoder_cache()
                _log_memory_boundary(logger, "Teardown done", model.device)

            attempt_cleanup("context cleanup", cleanup_context)
        teardown_elapsed_ms = (time.perf_counter() - teardown_start) * 1000.0
        attempt_cleanup(
            "teardown terminal-event emission",
            lambda: telemetry_observer.phase(
                name="teardown.cleanup",
                phase="teardown",
                elapsed_ms=teardown_elapsed_ms,
                attrs={
                    "ctx_present": ctx is not None,
                    "feature_row_store": feature_row_store_for_cleanup is not None,
                },
                wall_clock=True,
            ),
        )

        if primary_error is None:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            attempt_cleanup(
                "run terminal-event emission",
                lambda: telemetry_observer.run(
                    name="attribute.done",
                    elapsed_ms=run_elapsed_ms,
                    attrs={"compact_output": compact_output},
                    wall_clock=True,
                ),
            )
        else:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            attempt_cleanup(
                "run terminal-event emission",
                lambda: telemetry_observer.run(
                    name="attribute.failed",
                    elapsed_ms=run_elapsed_ms,
                    attrs={
                        "compact_output": compact_output,
                        "error_type": type(primary_error).__name__,
                        "error_message": str(primary_error),
                    },
                    wall_clock=True,
                ),
            )

        telemetry_export = attempt_cleanup(
            "sink close/export", lambda: telemetry_observer.close_export(include_events=True)
        )
        if not isinstance(telemetry_export, dict):
            telemetry_export = {"summary": {}, "events": []}
        if compact_output_result is not None:
            attempt_cleanup(
                "result attachment",
                lambda: telemetry_observer.attach_compact_result(
                    compact_output_result, telemetry_export
                ),
            )
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
            if primary_error is not None:
                attempt_cleanup(
                    "exception attachment",
                    lambda: telemetry_observer.attach_exception(primary_error, telemetry_export),
                )
            if profile:
                attempt_cleanup(
                    "human telemetry rendering",
                    lambda: telemetry_observer.render_human_summary(logger, telemetry_export),
                )

        if primary_error is None and cleanup_failures:
            _raise_cleanup_failures(cleanup_failures)
