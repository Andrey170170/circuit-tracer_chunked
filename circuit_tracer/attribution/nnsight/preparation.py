"""Validated mechanism preparation for the NNSight attribution backend."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable

import torch

from circuit_tracer.attribution.nnsight.active_decoder_rows import (
    ActiveDecoderRowAdmission,
    ActiveDecoderRowMemorySnapshot,
    ActiveDecoderRowResidencyRequirementError,
    resolve_active_decoder_row_admission,
    sample_active_decoder_row_memory,
)

from circuit_tracer.attribution.nnsight.numerics import (
    _exact_trace_internal_dtype_name,
    _resolve_exact_trace_internal_dtype,
)
from circuit_tracer.attribution.nnsight.phase1_policy import (
    _build_phase1_trace_batch_metadata,
    _build_phase1_trace_batch_sizing_metadata,
    _resolve_phase1_trace_batch_config,
    _resolve_phase1_trace_batch_sizing,
)
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _build_exact_encoder_residency_metadata,
    _build_phase4_ranker_metadata,
    _build_phase4_refresh_optimization_metadata,
    _build_phase4_refresh_policy_metadata,
    _build_phase4_row_executor_metadata,
    _build_phase4_row_reduction_metadata,
    _build_phase4_scheduler_metadata,
    _plan_phase4_feature_batch_size_preflight,
    _resolve_exact_encoder_residency_config,
    _resolve_phase4_feature_batch_planner_enabled,
    _resolve_phase4_feature_batch_planner_status,
    _resolve_phase4_ranker_config,
    _resolve_phase4_refresh_optimization_config,
    _resolve_phase4_refresh_policy_config,
    _resolve_phase4_row_executor_config,
    _resolve_phase4_row_reduction_config,
    _resolve_phase4_scheduler_config,
    _resolve_phase4_streaming_v1_microbatch_size,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_phase4_environment_fingerprint,
    _dtype_from_name,
    _resolve_internal_dtype_map,
    _resolve_internal_precision_requested,
)
from circuit_tracer.attribution.nnsight.replay import (
    _resolve_phase0_donor_context_policy,
    _resolve_phase0_replay_mode,
    _resolve_phase3_replay_mode,
    _resolve_phase3_replay_validation_policy,
)
from circuit_tracer.attribution.nnsight.row_store import (
    _build_row_store_cache_control_metadata,
    _resolve_row_store_cache_control_config,
)
from circuit_tracer.attribution.nnsight.session_controls import (
    resolve_nnsight_session_controls,
    validate_nnsight_session_control_requests,
)
from circuit_tracer.observability.events import TraceObserver
from circuit_tracer.governor.contracts import fingerprint
from circuit_tracer.execution_identity import (
    EffectiveExecutionDescriptor,
    EffectiveExecutionIdentity,
)
from circuit_tracer.tracing.plan import (
    BackwardEngineMode,
    ForwardGraphMode,
    ResolvedTracePlan,
    VjpKernelMode,
)
from circuit_tracer.tracing.problem import AttributionProblem
from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    require_exact_chunked_provider,
    require_exact_row_replay_provider,
)

_ACTIVATION_COMPARE_MODES = {
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


@dataclass(frozen=True)
class ProviderMechanisms:
    capabilities: Any
    exact_chunked: bool
    compact_row_store: bool
    decoder_chunk_cache: bool
    exact_encoder_residency: bool
    use_compact_feature_row_store: bool


@dataclass(frozen=True)
class NumericMechanisms:
    exact_dtype: torch.dtype
    exact_dtype_name: str
    internal_precision_requested: str
    dtype_map: dict[str, str]
    feature_row_storage_dtype: torch.dtype
    row_abs_sum_dtype: torch.dtype
    influence_compute_dtype: torch.dtype
    planner_compute_dtype: torch.dtype
    shadow_debug_compute_dtype: torch.dtype
    activation_compare_mode: str


@dataclass(frozen=True)
class ReplayMechanisms:
    phase0_mode: str
    phase0_context_policy: str
    phase0_bundle_path: str | None
    phase3_gradient_mode: str
    phase3_gradient_bundle_path: str | None
    phase3_row_mode: str
    phase3_row_bundle_path: str | None
    phase3_validation_policy: str


@dataclass(frozen=True)
class BatchMechanisms:
    phase1_config: Any
    phase1_metadata: dict[str, object]
    source_batch_size: int
    feature_batch_size: int
    logit_batch_size: int
    max_phase4_feature_batch_size: int
    planner_enabled: bool
    planner_status: str
    planner_skip_reason: str | None
    session_controls: Any
    trace_batch_size: int
    backward_engine_mode: BackwardEngineMode = "duplicated_lanes"
    backward_batch_capacity: int = 1
    forward_graph_mode: ForwardGraphMode = "logical_capacity"
    vjp_kernel_mode: VjpKernelMode = "nnsight_injected"
    forward_lane_count: int = 1


@dataclass(frozen=True)
class FrontierMechanisms:
    scheduler: Any
    refresh_optimization: Any
    refresh_policy: Any
    ranker: Any
    row_executor: Any
    row_reduction: Any
    row_store_cache_control: Any
    exact_encoder_residency: Any
    exact_encoder_residency_metadata: dict[str, object]
    execution_metadata: dict[str, object]
    refresh_aux_applicable: bool
    prepared_chunk_cache_bytes_effective: int
    active_row_accumulation_effective: str
    refresh_aux_fallback_reason: str | None
    feature_vjp_tape_enabled: bool = False
    feature_vjp_tape_batch_window_effective: int = 1
    feature_vjp_tape_max_bytes_effective: int = 0
    feature_vjp_tape_fallback_reason: str | None = "window_one_streaming_fallback"
    decoder_page_prefetch_depth_effective: int = 0
    decoder_page_prefetch_fallback_reason: str | None = "disabled"
    decoder_active_row_residency_requested: bool = False
    decoder_active_row_residency_requirement: str = "preferred"
    decoder_active_row_residency_effective: bool = False
    decoder_active_row_max_bytes_effective: int = 0
    decoder_active_row_safety_margin_bytes: int = 0
    decoder_active_row_admission: ActiveDecoderRowAdmission | None = None
    decoder_active_row_estimated_bytes: int | None = None
    decoder_active_row_admission_finalized: bool = False
    decoder_active_row_fallback_reason: str | None = "disabled"
    phase0_decoder_row_ranges_requested: bool = False
    phase0_decoder_row_ranges_effective: bool = False
    phase0_decoder_row_ranges_fallback_reason: str | None = "disabled"


@dataclass
class PreparedDiagnostics:
    observer: TraceObserver
    telemetry_max_events: int
    anomaly_debug_result: dict[str, object] | None
    cross_cluster_summary: dict[str, object] | None
    cross_cluster_checkpoints: list[dict[str, object]] | None
    cross_cluster_batches: list[dict[str, object]] | None


@dataclass(frozen=True)
class PreparedBackend:
    """Validated, grouped mechanisms consumed by attribution operations."""

    problem: AttributionProblem
    plan: ResolvedTracePlan
    logger: Any
    offload_handles: list[Any]
    forward_overrides: Any
    prefix_view_metadata: dict[str, object] | None
    output_position: int | None
    provider: ProviderMechanisms
    numerics: NumericMechanisms
    replay: ReplayMechanisms
    batches: BatchMechanisms
    frontier: FrontierMechanisms
    diagnostics: PreparedDiagnostics
    effective_execution: EffectiveExecutionIdentity
    start_time: float


@dataclass(frozen=True)
class PreparationDependencies:
    get_capabilities: Callable[[Any], Any] = get_transcoder_capabilities
    require_exact_provider: Callable[[Any], bool] = require_exact_chunked_provider


def resolve_phase0_activation_threshold_compare_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _ACTIVATION_COMPARE_MODES.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_ACTIVATION_COMPARE_MODES))
        raise ValueError(
            f"phase0_activation_threshold_compare_mode must be one of: {allowed} (got {value!r})"
        )
    return resolved


def resolve_telemetry_max_events(
    *,
    telemetry_max_events: int | None,
    compact_output: bool,
    exact_chunked_decoder: bool,
    profile: bool,
    phase4_anomaly_debug_enabled: bool,
) -> int:
    """Resolve the bounded telemetry event limit from explicit run policy."""
    if telemetry_max_events is not None and telemetry_max_events > 0:
        return int(telemetry_max_events)
    if compact_output and exact_chunked_decoder:
        return 20_000
    if profile or phase4_anomaly_debug_enabled:
        return 20_000
    return 20_000


def _resolve_provider_and_numerics(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    dependencies: PreparationDependencies,
) -> tuple[ProviderMechanisms, NumericMechanisms]:
    model = problem.model
    exact_dtype = _resolve_exact_trace_internal_dtype(plan.semantics.exact_trace_internal_dtype)
    exact_dtype_name = _exact_trace_internal_dtype_name(exact_dtype)
    internal_precision = _resolve_internal_precision_requested(
        None, exact_trace_internal_dtype=exact_dtype
    )
    dtype_map = _resolve_internal_dtype_map(
        internal_precision_requested=internal_precision,
        phase4_anomaly_debug_enabled=bool(plan.execution.observability.phase4_anomaly_debug),
    )
    capabilities = dependencies.get_capabilities(model.transcoders)
    exact_chunked = dependencies.require_exact_provider(model.transcoders)
    compact_row_store = bool(exact_chunked and capabilities.supports_compact_row_store)
    provider = ProviderMechanisms(
        capabilities=capabilities,
        exact_chunked=exact_chunked,
        compact_row_store=compact_row_store,
        decoder_chunk_cache=bool(exact_chunked and capabilities.supports_decoder_chunk_cache),
        exact_encoder_residency=bool(
            exact_chunked and capabilities.supports_exact_encoder_residency
        ),
        use_compact_feature_row_store=bool(plan.execution.compact_output and compact_row_store),
    )
    numerics = NumericMechanisms(
        exact_dtype=exact_dtype,
        exact_dtype_name=exact_dtype_name,
        internal_precision_requested=internal_precision,
        dtype_map=dtype_map,
        feature_row_storage_dtype=_dtype_from_name(dtype_map["feature_row_storage_dtype"]),
        row_abs_sum_dtype=_dtype_from_name(dtype_map["row_abs_sum_dtype"]),
        influence_compute_dtype=_dtype_from_name(dtype_map["influence_compute_dtype"]),
        planner_compute_dtype=_dtype_from_name(dtype_map["planner_compute_dtype"]),
        shadow_debug_compute_dtype=_dtype_from_name(dtype_map["shadow_debug_compute_dtype"]),
        activation_compare_mode=resolve_phase0_activation_threshold_compare_mode(
            plan.semantics.phase0_activation_threshold_compare_mode
        ),
    )
    return provider, numerics


def _resolve_replay(problem: AttributionProblem, plan: ResolvedTracePlan) -> ReplayMechanisms:
    replay = plan.execution.replay
    resolved = ReplayMechanisms(
        phase0_mode=_resolve_phase0_replay_mode(replay.phase0_mode),
        phase0_context_policy=_resolve_phase0_donor_context_policy(
            replay.phase0_donor_context_policy
        ),
        phase0_bundle_path=(
            os.fspath(replay.phase0_donor_bundle)
            if replay.phase0_donor_bundle is not None
            else None
        ),
        phase3_gradient_mode=_resolve_phase3_replay_mode(replay.phase3_gradient_mode),
        phase3_gradient_bundle_path=(
            os.fspath(replay.phase3_gradient_donor_bundle)
            if replay.phase3_gradient_donor_bundle is not None
            else None
        ),
        phase3_row_mode=_resolve_phase3_replay_mode(replay.phase3_row_mode),
        phase3_row_bundle_path=(
            os.fspath(replay.phase3_row_donor_bundle)
            if replay.phase3_row_donor_bundle is not None
            else None
        ),
        phase3_validation_policy=_resolve_phase3_replay_validation_policy(
            replay.phase3_validation_policy
        ),
    )
    storage = plan.execution.storage
    captures = plan.execution.observability
    if storage.retention == "none_recompute":
        require_exact_row_replay_provider(problem.model.transcoders)
    if (
        storage.full_retention_backend == "column_tiled_v1" or storage.retention == "none_recompute"
    ) and (
        resolved.phase3_gradient_mode != "disabled"
        or resolved.phase3_row_mode != "disabled"
        or captures.capture_phase3_gradient_bundle
        or captures.capture_phase3_row_bundle
    ):
        raise ValueError(
            "column-tiled and none_recompute row production do not yet support "
            "Phase-3 donor replay or gradient/row capture"
        )
    return resolved


def _resolve_frontier(
    plan: ResolvedTracePlan,
    provider: ProviderMechanisms,
) -> FrontierMechanisms:
    execution = plan.execution
    frontier = execution.frontier
    semantics = plan.semantics.frontier
    scheduler = _resolve_phase4_scheduler_config(
        phase4_scheduler_mode=semantics.scheduler,
        phase4_scheduler_debug=frontier.scheduler_debug,
        phase4_scheduler_telemetry_detail=frontier.scheduler_telemetry_detail,
    )
    refresh_optimization = _resolve_phase4_refresh_optimization_config(
        frontier.refresh_optimization,
        compact_output=execution.compact_output,
        exact_chunked_provider_enabled=provider.compact_row_store,
    )
    refresh_aux_applicable = bool(
        provider.use_compact_feature_row_store and refresh_optimization.effective_mode == "v1"
    )
    prepared_cache_bytes = (
        int(frontier.refresh_prepared_chunk_cache_bytes)
        if refresh_aux_applicable and provider.decoder_chunk_cache
        else 0
    )
    active_accumulation = (
        frontier.refresh_active_row_accumulation if refresh_aux_applicable else "zero_fill"
    )
    fallback_reason = None if refresh_aux_applicable else "not_applicable"
    refresh_metadata = _build_phase4_refresh_optimization_metadata(refresh_optimization)
    refresh_metadata.update(
        refresh_prepared_chunk_cache_bytes_requested=int(
            frontier.refresh_prepared_chunk_cache_bytes
        ),
        refresh_prepared_chunk_cache_bytes_effective=prepared_cache_bytes,
        refresh_prepared_chunk_cache_enabled=prepared_cache_bytes > 0,
        refresh_active_row_accumulation_requested=frontier.refresh_active_row_accumulation,
        refresh_active_row_accumulation_effective=active_accumulation,
        refresh_active_row_accumulation_fallback_reason=fallback_reason,
        refresh_active_row_accumulation_applicable=refresh_aux_applicable,
    )
    row_executor = _resolve_phase4_row_executor_config(
        frontier.row_executor,
        compact_output=execution.compact_output,
        exact_chunked_provider_enabled=provider.compact_row_store,
    )
    row_reduction = _resolve_phase4_row_reduction_config(
        frontier.row_reduction,
        compact_output=execution.compact_output,
        exact_chunked_provider_enabled=provider.compact_row_store,
    )
    refresh_policy = _resolve_phase4_refresh_policy_config(
        phase4_refresh_policy=semantics.refresh_policy,
        phase4_refresh_interval_multiplier=semantics.refresh_interval_multiplier,
        compact_output=execution.compact_output,
        exact_chunked_provider_enabled=provider.compact_row_store,
    )
    ranker = _resolve_phase4_ranker_config(semantics.ranker)
    cache_control = _resolve_row_store_cache_control_config(
        execution.storage.cache_control,
        compact_output=execution.compact_output,
        supports_compact_row_store=provider.compact_row_store,
    )
    residency = _resolve_exact_encoder_residency_config(
        execution.storage.exact_encoder_residency,
        supports_exact_encoder_residency=provider.exact_encoder_residency,
    )
    residency_metadata = _build_exact_encoder_residency_metadata(residency)
    tape_window_requested = int(frontier.feature_vjp_tape_batch_window)
    tape_max_bytes_requested = int(frontier.feature_vjp_tape_max_bytes)
    tape_fallback_reason = None
    if tape_window_requested <= 1:
        tape_fallback_reason = "window_one_streaming_fallback"
    elif not provider.exact_chunked:
        tape_fallback_reason = "requires_exact_chunked_provider"
    elif execution.storage.retention != "full_file":
        tape_fallback_reason = "requires_full_file_retention"
    elif execution.storage.full_retention_backend != "full_file":
        tape_fallback_reason = "requires_full_file_backend"
    tape_enabled = tape_fallback_reason is None
    tape_window_effective = tape_window_requested if tape_enabled else 1
    tape_max_bytes_effective = tape_max_bytes_requested if tape_enabled else 0
    tape_metadata = {
        "feature_vjp_tape_enabled": tape_enabled,
        "feature_vjp_tape_batch_window_requested": tape_window_requested,
        "feature_vjp_tape_batch_window_effective": tape_window_effective,
        "feature_vjp_tape_max_bytes_requested": tape_max_bytes_requested,
        "feature_vjp_tape_max_bytes_effective": tape_max_bytes_effective,
        "feature_vjp_tape_fallback_reason": tape_fallback_reason,
        "feature_vjp_tape_byte_cap_scope": ("simultaneous_host_device_and_row_ownership"),
    }
    prefetch_depth_requested = int(frontier.decoder_page_prefetch_depth)
    prefetch_fallback_reason = None
    if prefetch_depth_requested == 0:
        prefetch_fallback_reason = "disabled"
    elif not provider.exact_chunked:
        prefetch_fallback_reason = "requires_exact_chunked_provider"
    elif not bool(getattr(provider.capabilities, "supports_decoder_page_prefetch", False)):
        prefetch_fallback_reason = "provider_capability_unavailable"
    elif prefetch_depth_requested > 1:
        prefetch_fallback_reason = "only_depth_one_supported"
    prefetch_depth_effective = prefetch_depth_requested if prefetch_fallback_reason is None else 0
    prefetch_metadata = {
        "decoder_page_prefetch_depth_requested": prefetch_depth_requested,
        "decoder_page_prefetch_depth_effective": prefetch_depth_effective,
        "decoder_page_prefetch_fallback_reason": prefetch_fallback_reason,
        "decoder_page_prefetch_extra_final_page_residency_bound": (
            "depth_times_max_final_decoder_page_bytes"
        ),
        "decoder_page_prefetch_loader_dtype_conversion_transient_bytes": "unmeasured",
        "decoder_page_prefetch_pipeline_final_page_bound": (
            "current_plus_next_final_pages_at_most_two;"
            "source_and_dtype_conversion_transients_unmeasured;"
            "allocator_rounding_excluded"
        ),
        "decoder_page_prefetch_wait_telemetry_scope": (
            "host_future_only_excludes_cuda_event_stall"
        ),
    }
    active_rows_requested = bool(frontier.decoder_active_row_residency)
    active_rows_requirement = str(frontier.decoder_active_row_residency_requirement)
    active_rows_max_bytes_requested = int(frontier.decoder_active_row_max_bytes)
    active_rows_safety_margin_bytes = int(frontier.decoder_active_row_safety_margin_bytes)
    active_rows_fallback_reason = None
    if not active_rows_requested:
        active_rows_fallback_reason = "disabled"
    elif not provider.exact_chunked:
        active_rows_fallback_reason = "requires_exact_chunked_provider"
    elif not bool(getattr(provider.capabilities, "supports_active_decoder_row_residency", False)):
        active_rows_fallback_reason = "provider_capability_unavailable"
    active_rows_effective = active_rows_fallback_reason is None
    active_rows_max_bytes_effective = 0
    if active_rows_requirement == "required" and not active_rows_effective:
        raise RuntimeError(
            "required decoder active-row residency is unavailable before Phase 0: "
            f"reason={active_rows_fallback_reason}"
        )
    active_rows_metadata = {
        "decoder_active_row_residency_requested": active_rows_requested,
        "decoder_active_row_residency_requirement": active_rows_requirement,
        "decoder_active_row_residency_effective": active_rows_effective,
        "decoder_active_row_max_bytes_requested": active_rows_max_bytes_requested,
        "decoder_active_row_max_bytes_effective": active_rows_max_bytes_effective,
        "decoder_active_row_safety_margin_bytes": active_rows_safety_margin_bytes,
        "decoder_active_row_admission_reason": (
            "estimate_pending" if active_rows_effective else active_rows_fallback_reason
        ),
        "decoder_active_row_fallback_reason": active_rows_fallback_reason,
    }
    phase0_ranges_requested = bool(frontier.phase0_decoder_row_ranges)
    phase0_ranges_fallback_reason = None
    if not phase0_ranges_requested:
        phase0_ranges_fallback_reason = "disabled"
    elif not active_rows_effective:
        phase0_ranges_fallback_reason = "requires_active_row_residency"
    elif not bool(getattr(provider.capabilities, "supports_phase0_decoder_row_ranges", False)):
        phase0_ranges_fallback_reason = "provider_capability_unavailable"
    phase0_ranges_effective = phase0_ranges_fallback_reason is None
    phase0_ranges_metadata = {
        "phase0_decoder_row_ranges_requested": phase0_ranges_requested,
        "phase0_decoder_row_ranges_effective": phase0_ranges_effective,
        "phase0_decoder_row_ranges_fallback_reason": phase0_ranges_fallback_reason,
    }
    metadata = {
        **_build_phase4_scheduler_metadata(scheduler),
        **refresh_metadata,
        **_build_phase4_row_executor_metadata(row_executor),
        **_build_phase4_row_reduction_metadata(row_reduction),
        **_build_phase4_refresh_policy_metadata(refresh_policy),
        **_build_phase4_ranker_metadata(ranker),
        **_build_row_store_cache_control_metadata(cache_control),
        **residency_metadata,
        **tape_metadata,
        **prefetch_metadata,
        **active_rows_metadata,
        **phase0_ranges_metadata,
    }
    return FrontierMechanisms(
        scheduler=scheduler,
        refresh_optimization=refresh_optimization,
        refresh_policy=refresh_policy,
        ranker=ranker,
        row_executor=row_executor,
        row_reduction=row_reduction,
        row_store_cache_control=cache_control,
        exact_encoder_residency=residency,
        exact_encoder_residency_metadata=residency_metadata,
        execution_metadata=metadata,
        refresh_aux_applicable=refresh_aux_applicable,
        prepared_chunk_cache_bytes_effective=prepared_cache_bytes,
        active_row_accumulation_effective=active_accumulation,
        refresh_aux_fallback_reason=fallback_reason,
        feature_vjp_tape_enabled=tape_enabled,
        feature_vjp_tape_batch_window_effective=tape_window_effective,
        feature_vjp_tape_max_bytes_effective=tape_max_bytes_effective,
        feature_vjp_tape_fallback_reason=tape_fallback_reason,
        decoder_page_prefetch_depth_effective=prefetch_depth_effective,
        decoder_page_prefetch_fallback_reason=prefetch_fallback_reason,
        decoder_active_row_residency_requested=active_rows_requested,
        decoder_active_row_residency_requirement=active_rows_requirement,
        decoder_active_row_residency_effective=active_rows_effective,
        decoder_active_row_max_bytes_effective=active_rows_max_bytes_effective,
        decoder_active_row_safety_margin_bytes=active_rows_safety_margin_bytes,
        decoder_active_row_admission=None,
        decoder_active_row_estimated_bytes=None,
        decoder_active_row_admission_finalized=False,
        decoder_active_row_fallback_reason=active_rows_fallback_reason,
        phase0_decoder_row_ranges_requested=phase0_ranges_requested,
        phase0_decoder_row_ranges_effective=phase0_ranges_effective,
        phase0_decoder_row_ranges_fallback_reason=phase0_ranges_fallback_reason,
    )


def _resolve_batches(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    logger: Any,
    prefix_view_metadata: dict[str, object] | None,
    provider: ProviderMechanisms,
    numerics: NumericMechanisms,
    frontier: FrontierMechanisms,
    diagnostics: PreparedDiagnostics,
) -> BatchMechanisms:
    semantics = plan.semantics
    session = plan.execution.session
    frontier_plan = plan.execution.frontier
    backward_plan = plan.execution.backward
    backward_mode = backward_plan.mode
    if (
        not backward_plan.supports_phase3_gradient_replay
        and plan.execution.replay.phase3_gradient_mode != "disabled"
    ):
        raise ValueError(
            f"{backward_mode} requires native Phase-3 gradients; "
            "Phase-3 gradient replay is unsupported"
        )
    phase1_config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy=session.phase1_trace_batch_policy,
        phase1_trace_batch_size_max=session.phase1_trace_batch_size_max,
    )
    metadata = _build_phase1_trace_batch_metadata(phase1_config)
    sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=semantics.source_batch_size,
        feature_batch_size=semantics.feature_batch_size,
        logit_batch_size=semantics.logit_batch_size,
        feature_batch_size_max=frontier_plan.feature_batch_size_max,
        phase1_trace_batch_config=phase1_config,
    )
    metadata.update(_build_phase1_trace_batch_sizing_metadata(sizing))
    source_size = sizing.effective_source_batch_size
    feature_size = sizing.effective_feature_batch_size
    logit_size = sizing.effective_logit_batch_size
    max_feature_size = sizing.effective_phase4_max_feature_batch_size
    validate_nnsight_session_control_requests(
        nnsight_session_capacity=session.capacity,
        phase3_compute_microbatch_max_rows=session.phase3_microbatch_max_rows,
        phase4_execution_batch_max_rows=session.phase4_execution_batch_max_rows,
    )
    planner_enabled = _resolve_phase4_feature_batch_planner_enabled(
        plan_feature_batch_size=frontier_plan.feature_batch_planning,
        auto_scale_feature_batch_size=False,
    )
    planner_status, skip_reason = _resolve_phase4_feature_batch_planner_status(
        planner_enabled=planner_enabled,
        effective_feature_batch_size=feature_size,
        max_feature_batch_size=max_feature_size,
    )
    if planner_enabled and planner_status != "skipped_no_headroom":
        replay = plan.execution.replay
        feature_size = _plan_phase4_feature_batch_size_preflight(
            model=problem.model,
            prompt=problem.prompt,
            attribution_targets=problem.targets,
            batch_size=source_size,
            initial_feature_batch_size=min(feature_size, max_feature_size),
            effective_logit_batch_size=logit_size,
            max_feature_batch_size=max_feature_size,
            max_feature_nodes=semantics.max_feature_nodes,
            update_interval=semantics.update_interval,
            max_n_logits=problem.max_n_logits,
            desired_logit_prob=problem.desired_logit_prob,
            logger=logger,
            sparsification=semantics.sparsification,
            chunked_feature_replay_window=replay.feature_window,
            error_vector_prefetch_lookahead=replay.error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=replay.stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=replay.stage_error_vectors_on_cpu,
            row_subchunk_size=replay.decoder_contraction_tile,
            exact_encoder_residency=frontier.exact_encoder_residency.effective_mode,
            diagnostic_feature_cap=semantics.diagnostic_feature_cap,
            feature_batch_target_reserved_fraction=frontier_plan.feature_batch_target_reserved_fraction,
            feature_batch_min_free_fraction=frontier_plan.feature_batch_min_free_fraction,
            feature_batch_probe_batches=frontier_plan.feature_batch_probe_batches,
            exact_trace_internal_dtype=numerics.exact_dtype,
            internal_precision_requested=numerics.internal_precision_requested,
            resolved_dtype_map=numerics.dtype_map,
            row_abs_sum_dtype=numerics.row_abs_sum_dtype,
            planner_compute_dtype=numerics.planner_compute_dtype,
            trace_observer=diagnostics.observer,
            prefix_view_metadata=prefix_view_metadata,
            backward_engine_mode=backward_mode,
            backward_batch_capacity=backward_plan.planner_batch_capacity(
                source_rows=source_size,
                feature_rows=feature_size,
                feature_row_ceiling=max_feature_size,
                logit_rows=logit_size,
            ),
        )
        planner_status = "executed"
    legacy_phase4_rows = feature_size
    if frontier.row_executor.effective_mode == "streaming_v1":
        legacy_phase4_rows = _resolve_phase4_streaming_v1_microbatch_size(feature_size)
    controls = resolve_nnsight_session_controls(
        nnsight_session_capacity=session.capacity,
        phase3_compute_microbatch_max_rows=session.phase3_microbatch_max_rows,
        phase4_execution_batch_max_rows=session.phase4_execution_batch_max_rows,
        legacy_session_capacity=max(source_size, feature_size, logit_size),
        legacy_phase3_batch_rows=logit_size,
        legacy_phase4_batch_rows=legacy_phase4_rows,
    )
    backward_topology = backward_plan.topology(batch_capacity=int(controls.session_capacity))
    metadata.update(
        trace_batch_size_legacy=int(sizing.trace_batch_size_legacy),
        trace_batch_size_effective=int(controls.session_capacity),
        backward_engine_mode=backward_topology.mode,
        backward_batch_capacity=backward_topology.batch_capacity,
        forward_graph_mode=backward_topology.forward_graph_mode,
        vjp_kernel_mode=backward_topology.vjp_kernel_mode,
        forward_lane_count=backward_topology.forward_lane_count,
    )
    return BatchMechanisms(
        phase1_config=phase1_config,
        phase1_metadata=metadata,
        source_batch_size=source_size,
        feature_batch_size=feature_size,
        logit_batch_size=logit_size,
        max_phase4_feature_batch_size=max_feature_size,
        planner_enabled=planner_enabled,
        planner_status=planner_status,
        planner_skip_reason=skip_reason,
        session_controls=controls,
        trace_batch_size=controls.session_capacity,
        backward_engine_mode=backward_topology.mode,
        backward_batch_capacity=backward_topology.batch_capacity,
        forward_graph_mode=backward_topology.forward_graph_mode,
        vjp_kernel_mode=backward_topology.vjp_kernel_mode,
        forward_lane_count=backward_topology.forward_lane_count,
    )


def _prepare_diagnostics(
    plan: ResolvedTracePlan,
    provider: ProviderMechanisms,
    numerics: NumericMechanisms,
    replay: ReplayMechanisms,
    observer: TraceObserver,
) -> PreparedDiagnostics:
    policy = plan.execution.observability
    limit = resolve_telemetry_max_events(
        telemetry_max_events=policy.telemetry_max_events,
        compact_output=plan.execution.compact_output,
        exact_chunked_decoder=provider.exact_chunked,
        profile=policy.profile,
        phase4_anomaly_debug_enabled=policy.phase4_anomaly_debug,
    )
    anomaly = None
    if policy.phase4_anomaly_debug:
        anomaly = {
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
    cross_cluster = None
    checkpoints = None
    batches = None
    if policy.cross_cluster_debug:
        cross_cluster = {
            "schema_version": 1,
            "enabled": True,
            "status": "collecting",
            "mode": "early_phase_scalar_summary",
            "phase0_replay_mode": replay.phase0_mode,
            "phase0_donor_context_policy": replay.phase0_context_policy,
            "phase3_gradient_replay_mode": replay.phase3_gradient_mode,
            "phase3_row_replay_mode": replay.phase3_row_mode,
            "phase3_replay_validation_policy": replay.phase3_validation_policy,
            "phase0_activation_threshold_compare_mode": numerics.activation_compare_mode,
            "internal_precision_requested": numerics.internal_precision_requested,
            "resolved_dtype_map": numerics.dtype_map,
            "environment": _build_phase4_environment_fingerprint(),
            "checkpoints": {},
        }
        checkpoints = []
        batches = []
    return PreparedDiagnostics(
        observer=observer,
        telemetry_max_events=limit,
        anomaly_debug_result=anomaly,
        cross_cluster_summary=cross_cluster,
        cross_cluster_checkpoints=checkpoints,
        cross_cluster_batches=batches,
    )


def _validate_provider_requirements(
    plan: ResolvedTracePlan,
    provider: ProviderMechanisms,
    replay: ReplayMechanisms,
) -> None:
    if plan.planning_trace_plan is not None:
        requested_fetch = plan.execution.decoder.fetch_chunk_size
        loaded_fetch = getattr(provider.capabilities, "default_decoder_chunk_size", None)
        if loaded_fetch is None or int(loaded_fetch) != int(requested_fetch or 0):
            raise ValueError(
                "governed decoder fetch size was not applied while loading the provider: "
                f"loaded={loaded_fetch!r}, requested={requested_fetch!r}"
            )
    policy = plan.execution.observability
    compact_exact = plan.execution.compact_output and provider.exact_chunked
    compact_rows = plan.execution.compact_output and provider.compact_row_store
    requirements = (
        (policy.phase4_anomaly_debug, compact_exact, "Phase-4 anomaly debug"),
        (policy.cross_cluster_debug, compact_exact, "cross_cluster_debug"),
        (policy.capture_phase0_donor_bundle, compact_exact, "capture_phase0_donor_bundle"),
        (policy.capture_phase3_seed_bundle, compact_exact, "capture_phase3_seed_bundle"),
        (
            policy.capture_phase3_gradient_bundle,
            compact_exact,
            "capture_phase3_gradient_bundle",
        ),
        (policy.capture_phase3_row_bundle, compact_exact, "capture_phase3_row_bundle"),
        (
            policy.capture_feature_semantic_descriptors,
            compact_exact,
            "capture_feature_semantic_descriptors",
        ),
        (replay.phase0_mode != "disabled", compact_exact, "phase0 donor replay"),
        (replay.phase3_gradient_mode != "disabled", compact_exact, "Phase-3 gradient replay"),
        (replay.phase3_row_mode != "disabled", compact_exact, "Phase-3 row replay"),
        (
            plan.execution.frontier.feature_batch_planning,
            compact_rows,
            "Phase-4 feature batch planner",
        ),
    )
    for requested, supported, label in requirements:
        if requested and not supported:
            raise ValueError(f"{label} requires compact_output=True and exact provider support")


def _capability_descriptor(capabilities: Any) -> dict[str, Any]:
    if is_dataclass(capabilities) and not isinstance(capabilities, type):
        return asdict(capabilities)
    names = (
        "architecture",
        "checkpoint_format",
        "decoder_output_topology",
        "supports_exact_chunked_provider",
        "supports_compact_row_store",
        "supports_decoder_chunk_cache",
        "supports_exact_encoder_residency",
    )
    return {name: getattr(capabilities, name) for name in names if hasattr(capabilities, name)}


def _effective_execution_identity(
    provider: ProviderMechanisms,
    numerics: NumericMechanisms,
    replay: ReplayMechanisms,
    batches: BatchMechanisms,
    frontier: FrontierMechanisms,
    plan: ResolvedTracePlan,
) -> EffectiveExecutionIdentity:
    """Describe only mechanisms fixed by successful NNSight preparation."""
    descriptor = EffectiveExecutionDescriptor(
        schema_version=2,
        backend="nnsight",
        provider={
            "capabilities": _capability_descriptor(provider.capabilities),
            "exact_chunked": provider.exact_chunked,
            "compact_row_store": provider.compact_row_store,
            "decoder_chunk_cache": provider.decoder_chunk_cache,
            "exact_encoder_residency": provider.exact_encoder_residency,
            "use_compact_feature_row_store": provider.use_compact_feature_row_store,
        },
        numerics={
            "exact_dtype": numerics.exact_dtype_name,
            "internal_precision": numerics.internal_precision_requested,
            "dtype_map": numerics.dtype_map,
            "feature_row_storage_dtype": str(numerics.feature_row_storage_dtype),
            "row_abs_sum_dtype": str(numerics.row_abs_sum_dtype),
            "influence_compute_dtype": str(numerics.influence_compute_dtype),
            "planner_compute_dtype": str(numerics.planner_compute_dtype),
            "shadow_debug_compute_dtype": str(numerics.shadow_debug_compute_dtype),
            "activation_compare_mode": numerics.activation_compare_mode,
        },
        replay=asdict(replay),
        batches={
            "phase1_policy": batches.phase1_config.effective_policy,
            "phase1_batch_size_max": batches.phase1_config.effective_batch_size_max,
            "phase1_fallback_reason": batches.phase1_config.fallback_reason,
            "source_batch_size": batches.source_batch_size,
            "feature_batch_size": batches.feature_batch_size,
            "logit_batch_size": batches.logit_batch_size,
            "max_phase4_feature_batch_size": batches.max_phase4_feature_batch_size,
            "feature_batch_planner_enabled": batches.planner_enabled,
            "feature_batch_planner_status": batches.planner_status,
            "feature_batch_planner_skip_reason": batches.planner_skip_reason,
            "session_capacity": batches.session_controls.session_capacity,
            "phase3_microbatch_max_rows": batches.session_controls.phase3_microbatch_max_rows,
            "phase4_execution_batch_max_rows": (
                batches.session_controls.phase4_execution_batch_max_rows
            ),
            "trace_batch_size": batches.trace_batch_size,
            "backward_engine_mode": batches.backward_engine_mode,
            "backward_batch_capacity": batches.backward_batch_capacity,
            "forward_graph_mode": batches.forward_graph_mode,
            "vjp_kernel_mode": batches.vjp_kernel_mode,
            "forward_lane_count": batches.forward_lane_count,
        },
        frontier={
            "scheduler_mode": frontier.scheduler.effective_mode,
            "scheduler_version": frontier.scheduler.effective_version,
            "scheduler_policy": frontier.scheduler.effective_policy,
            "refresh_optimization_mode": frontier.refresh_optimization.effective_mode,
            "refresh_optimization_version": frontier.refresh_optimization.effective_version,
            "refresh_policy": frontier.refresh_policy.effective_policy,
            "refresh_interval_multiplier": frontier.refresh_policy.effective_interval_multiplier,
            "refresh_policy_fallback_reason": frontier.refresh_policy.fallback_reason,
            "ranker_mode": frontier.ranker.effective_mode,
            "row_executor_mode": frontier.row_executor.effective_mode,
            "row_executor_version": frontier.row_executor.effective_version,
            "row_reduction_mode": frontier.row_reduction.effective_mode,
            "row_reduction_version": frontier.row_reduction.effective_version,
            "row_store_cache_control_mode": frontier.row_store_cache_control.effective_mode,
            "row_store_cache_control_fallback_reason": frontier.row_store_cache_control.fallback_reason,
            "exact_encoder_residency_mode": frontier.exact_encoder_residency.effective_mode,
            "exact_encoder_residency_fallback_reason": frontier.exact_encoder_residency.fallback_reason,
            "refresh_aux_applicable": frontier.refresh_aux_applicable,
            "prepared_chunk_cache_bytes": frontier.prepared_chunk_cache_bytes_effective,
            "active_row_accumulation": frontier.active_row_accumulation_effective,
            "refresh_aux_fallback_reason": frontier.refresh_aux_fallback_reason,
            "feature_vjp_tape_enabled": frontier.feature_vjp_tape_enabled,
            "feature_vjp_tape_batch_window_requested": (
                plan.execution.frontier.feature_vjp_tape_batch_window
            ),
            "feature_vjp_tape_batch_window_effective": (
                frontier.feature_vjp_tape_batch_window_effective
            ),
            "feature_vjp_tape_max_bytes_requested": (
                plan.execution.frontier.feature_vjp_tape_max_bytes
            ),
            "feature_vjp_tape_max_bytes_effective": (frontier.feature_vjp_tape_max_bytes_effective),
            "feature_vjp_tape_fallback_reason": (frontier.feature_vjp_tape_fallback_reason),
            "feature_vjp_tape_byte_cap_scope": ("simultaneous_host_device_and_row_ownership"),
            "decoder_page_prefetch_depth_requested": (
                plan.execution.frontier.decoder_page_prefetch_depth
            ),
            "decoder_page_prefetch_depth_effective": (
                frontier.decoder_page_prefetch_depth_effective
            ),
            "decoder_page_prefetch_fallback_reason": (
                frontier.decoder_page_prefetch_fallback_reason
            ),
            "decoder_page_prefetch_extra_final_page_residency_bound": (
                "depth_times_max_final_decoder_page_bytes"
            ),
            "decoder_page_prefetch_loader_dtype_conversion_transient_bytes": "unmeasured",
            "decoder_page_prefetch_pipeline_final_page_bound": (
                "current_plus_next_final_pages_at_most_two;"
                "source_and_dtype_conversion_transients_unmeasured;"
                "allocator_rounding_excluded"
            ),
            "decoder_page_prefetch_wait_telemetry_scope": (
                "host_future_only_excludes_cuda_event_stall"
            ),
            "decoder_active_row_residency_requested": (
                plan.execution.frontier.decoder_active_row_residency
            ),
            "decoder_active_row_residency_requirement": (
                frontier.decoder_active_row_residency_requirement
            ),
            "decoder_active_row_residency_effective": (
                frontier.decoder_active_row_residency_effective
            ),
            "decoder_active_row_max_bytes_requested": (
                plan.execution.frontier.decoder_active_row_max_bytes
            ),
            "decoder_active_row_max_bytes_effective": (
                plan.execution.frontier.decoder_active_row_max_bytes
                if frontier.decoder_active_row_residency_effective
                else 0
            ),
            "decoder_active_row_safety_margin_bytes": (
                frontier.decoder_active_row_safety_margin_bytes
            ),
            "decoder_active_row_estimated_bytes": (frontier.decoder_active_row_estimated_bytes),
            "decoder_active_row_admission_reason": (
                None
                if frontier.decoder_active_row_admission is None
                else frontier.decoder_active_row_admission.reason
            ),
            "decoder_active_row_fallback_reason": frontier.decoder_active_row_fallback_reason,
            "phase0_decoder_row_ranges_requested": (frontier.phase0_decoder_row_ranges_requested),
            "phase0_decoder_row_ranges_effective": (frontier.phase0_decoder_row_ranges_effective),
            "phase0_decoder_row_ranges_fallback_reason": (
                frontier.phase0_decoder_row_ranges_fallback_reason
            ),
        },
        decoder={
            "fetch_chunk_size": plan.execution.decoder.fetch_chunk_size,
            "cache_enabled": plan.execution.session.decoder_cache.enabled,
            "cache_max_bytes": plan.execution.session.decoder_cache.max_bytes,
        },
        storage={
            "retention": plan.execution.storage.retention,
            "backend": plan.execution.storage.full_retention_backend,
            "feature_column_tile_size": plan.execution.storage.feature_column_tile_size,
            "influence_row_tile_size": plan.execution.storage.influence_row_tile_size,
            "influence_column_tile_size": plan.execution.storage.influence_column_tile_size,
            "cache_control": plan.execution.storage.cache_control,
            "temp_root_policy": plan.execution.storage.temp_root_policy,
            "preallocate": plan.execution.storage.preallocate,
            "replay_tile_cache_bytes": plan.execution.storage.replay_tile_cache_bytes,
            "feature_row_influence_mode": (plan.execution.storage.feature_row_influence_mode),
            "feature_row_influence_mode_requested": (
                plan.execution.storage.feature_row_influence_mode
            ),
            "feature_row_influence_requirement": (
                plan.execution.storage.feature_row_influence_requirement
            ),
            "gpu_resident_max_bytes": plan.execution.storage.gpu_resident_max_bytes,
            "gpu_window_max_bytes": plan.execution.storage.gpu_window_max_bytes,
            "gpu_resident_safety_margin_bytes": (
                plan.execution.storage.gpu_resident_safety_margin_bytes
            ),
            "exact_encoder_residency": plan.execution.storage.exact_encoder_residency,
            "placement": (
                None
                if plan.execution.storage.placement is None
                else plan.execution.storage.placement.value
            ),
        },
    )
    return EffectiveExecutionIdentity(descriptor=descriptor, fingerprint=fingerprint(descriptor))


def finalize_feature_row_influence_execution(
    prepared: PreparedBackend,
    *,
    resolved_mode: str,
    reason: str,
) -> PreparedBackend:
    """Fold the Phase-2 row-store resolution into effective execution identity."""

    identity = prepared.effective_execution
    descriptor = identity.descriptor
    if descriptor is None:
        raise RuntimeError("NNSight effective execution descriptor is missing")
    storage = {
        **descriptor.storage,
        "feature_row_influence_mode_resolved": resolved_mode,
        "feature_row_influence_resolution_reason": reason,
    }
    revised_descriptor = replace(descriptor, storage=storage)
    revised_identity = EffectiveExecutionIdentity(
        descriptor=revised_descriptor,
        fingerprint=fingerprint(revised_descriptor),
    )
    return replace(prepared, effective_execution=revised_identity)


def prepare_backend(
    *,
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    logger: Any,
    offload_handles: list[Any],
    forward_overrides: Any,
    prefix_view_metadata: dict[str, object] | None,
    output_position: int | None,
    observer: TraceObserver,
    dependencies: PreparationDependencies = PreparationDependencies(),
) -> PreparedBackend:
    """Resolve and validate every physical mechanism before Phase 0 starts."""
    provider, numerics = _resolve_provider_and_numerics(problem, plan, dependencies)
    replay = _resolve_replay(problem, plan)
    _validate_provider_requirements(plan, provider, replay)
    frontier = _resolve_frontier(plan, provider)
    diagnostics = _prepare_diagnostics(plan, provider, numerics, replay, observer)
    batches = _resolve_batches(
        problem,
        plan,
        logger,
        prefix_view_metadata,
        provider,
        numerics,
        frontier,
        diagnostics,
    )
    diagnostics.cross_cluster_summary = diagnostics.cross_cluster_summary and {
        **diagnostics.cross_cluster_summary,
        "phase1_trace_batch": batches.phase1_metadata,
        "phase4_execution": frontier.execution_metadata,
    }
    effective_execution = _effective_execution_identity(
        provider, numerics, replay, batches, frontier, plan
    )
    return PreparedBackend(
        problem=problem,
        plan=plan,
        logger=logger,
        offload_handles=offload_handles,
        forward_overrides=forward_overrides,
        prefix_view_metadata=prefix_view_metadata,
        output_position=output_position,
        provider=provider,
        numerics=numerics,
        replay=replay,
        batches=batches,
        frontier=frontier,
        diagnostics=diagnostics,
        effective_execution=effective_execution,
        start_time=time.time(),
    )


def reprepare_after_active_universe(
    prepared: PreparedBackend,
    plan: ResolvedTracePlan,
) -> PreparedBackend:
    """Rebuild only mechanisms still mutable after Phase 0.

    Provider, numerics, replay, diagnostics, source/session capacity, and Phase-0
    state are preserved.  Storage/frontier residency and Phase-3/4 microbatch
    controls are rebuilt from the final governed plan.
    """
    frontier = _resolve_frontier(plan, prepared.provider)
    prior_range_metadata = {
        key: value
        for key, value in prepared.frontier.execution_metadata.items()
        if key.startswith("phase0_decoder_row_ranges_")
    }
    range_execution_observed = (
        "phase0_decoder_row_ranges_planning_seconds" in prior_range_metadata
        or prepared.frontier.phase0_decoder_row_ranges_fallback_reason == "seed_capture_refused"
    )
    if range_execution_observed:
        execution_metadata = dict(frontier.execution_metadata)
        execution_metadata.update(prior_range_metadata)
        frontier = replace(
            frontier,
            execution_metadata=execution_metadata,
            phase0_decoder_row_ranges_requested=(
                prepared.frontier.phase0_decoder_row_ranges_requested
            ),
            phase0_decoder_row_ranges_effective=(
                prepared.frontier.phase0_decoder_row_ranges_effective
            ),
            phase0_decoder_row_ranges_fallback_reason=(
                prepared.frontier.phase0_decoder_row_ranges_fallback_reason
            ),
        )
    if prepared.frontier.decoder_active_row_admission_finalized:
        prior_frontier = prepared.frontier
        if prior_frontier.decoder_active_row_estimated_bytes is None:
            raise RuntimeError("finalized active-row admission is missing its exact byte estimate")
        prior_policy = prepared.plan.execution.frontier
        revised_policy = plan.execution.frontier
        policy_fields = (
            "decoder_active_row_residency",
            "decoder_active_row_residency_requirement",
            "decoder_active_row_max_bytes",
            "decoder_active_row_safety_margin_bytes",
        )
        changed = [
            name
            for name in policy_fields
            if getattr(prior_policy, name) != getattr(revised_policy, name)
        ]
        if changed:
            raise RuntimeError(
                "active-universe reprepare changed finalized active-row admission policy: "
                + ", ".join(changed)
            )
        execution_metadata = dict(frontier.execution_metadata)
        execution_metadata.update(
            {
                key: value
                for key, value in prior_frontier.execution_metadata.items()
                if key.startswith("decoder_active_row_")
            }
        )
        frontier = replace(
            frontier,
            execution_metadata=execution_metadata,
            decoder_active_row_residency_requirement=(
                prior_frontier.decoder_active_row_residency_requirement
            ),
            decoder_active_row_residency_effective=(
                prior_frontier.decoder_active_row_residency_effective
            ),
            decoder_active_row_max_bytes_effective=(
                prior_frontier.decoder_active_row_max_bytes_effective
            ),
            decoder_active_row_safety_margin_bytes=(
                prior_frontier.decoder_active_row_safety_margin_bytes
            ),
            decoder_active_row_estimated_bytes=(prior_frontier.decoder_active_row_estimated_bytes),
            decoder_active_row_admission_finalized=True,
            decoder_active_row_admission=prior_frontier.decoder_active_row_admission,
            decoder_active_row_fallback_reason=(prior_frontier.decoder_active_row_fallback_reason),
        )
    batches = _resolve_batches(
        prepared.problem,
        plan,
        prepared.logger,
        prepared.prefix_view_metadata,
        prepared.provider,
        prepared.numerics,
        frontier,
        prepared.diagnostics,
    )
    frozen_mismatches = []
    for name in ("source_batch_size", "feature_batch_size", "logit_batch_size", "trace_batch_size"):
        if getattr(batches, name) != getattr(prepared.batches, name):
            frozen_mismatches.append(name)
    old_controls = prepared.batches.session_controls
    new_controls = batches.session_controls
    if new_controls.session_capacity != old_controls.session_capacity:
        frozen_mismatches.append("session_capacity")
    if frozen_mismatches:
        raise RuntimeError(
            "active-universe reprepare changed frozen prepared mechanisms: "
            + ", ".join(frozen_mismatches)
        )
    effective = _effective_execution_identity(
        prepared.provider, prepared.numerics, prepared.replay, batches, frontier, plan
    )
    return replace(
        prepared,
        plan=plan,
        batches=batches,
        frontier=frontier,
        effective_execution=effective,
    )


def _finalize_active_decoder_row_frontier(
    frontier: FrontierMechanisms,
    *,
    max_bytes: int,
    estimated_bytes: int,
    safety_margin_bytes: int = 0,
    memory: ActiveDecoderRowMemorySnapshot | None = None,
) -> FrontierMechanisms:
    if not frontier.decoder_active_row_residency_effective:
        return frontier
    if memory is None:
        memory = ActiveDecoderRowMemorySnapshot(
            free_bytes=None,
            total_bytes=None,
            allocated_bytes=None,
            reserved_bytes=None,
            device="unavailable",
        )
    decision = resolve_active_decoder_row_admission(
        requested=frontier.decoder_active_row_residency_requested,
        estimated_bytes=estimated_bytes,
        hard_ceiling_bytes=max_bytes,
        safety_margin_bytes=safety_margin_bytes,
        memory=memory,
    )
    admitted = decision.admitted
    fallback_reason = (
        None
        if admitted
        else (
            "estimated_bytes_exceed_max"
            if decision.reason == "estimated_bytes_exceed_user_ceiling"
            else decision.reason
        )
    )
    metadata = dict(frontier.execution_metadata)
    metadata.update(decision.as_metadata())
    metadata.update(
        {
            "decoder_active_row_residency_requirement": (
                frontier.decoder_active_row_residency_requirement
            ),
            "decoder_active_row_residency_effective": admitted,
            "decoder_active_row_max_bytes_effective": (
                decision.effective_budget_bytes if admitted else 0
            ),
            "decoder_active_row_estimated_bytes": estimated_bytes,
            "decoder_active_row_fallback_reason": fallback_reason,
        }
    )
    return replace(
        frontier,
        execution_metadata=metadata,
        decoder_active_row_residency_effective=admitted,
        decoder_active_row_max_bytes_effective=(decision.effective_budget_bytes if admitted else 0),
        decoder_active_row_estimated_bytes=estimated_bytes,
        decoder_active_row_admission_finalized=True,
        decoder_active_row_admission=decision,
        decoder_active_row_fallback_reason=fallback_reason,
    )


def finalize_active_decoder_row_admission(
    prepared: PreparedBackend,
    *,
    estimated_bytes: int,
    memory: ActiveDecoderRowMemorySnapshot | None = None,
) -> PreparedBackend:
    """Finalize exact active-row byte admission before Phase 3 executes."""

    frontier = prepared.frontier
    if not frontier.decoder_active_row_residency_effective:
        return prepared
    max_bytes = int(prepared.plan.execution.frontier.decoder_active_row_max_bytes)
    safety_margin_bytes = int(
        prepared.plan.execution.frontier.decoder_active_row_safety_margin_bytes
    )
    if memory is None:
        memory = sample_active_decoder_row_memory(prepared.problem.model.device)
    frontier = _finalize_active_decoder_row_frontier(
        frontier,
        max_bytes=max_bytes,
        estimated_bytes=estimated_bytes,
        safety_margin_bytes=safety_margin_bytes,
        memory=memory,
    )
    decision = frontier.decoder_active_row_admission
    if (
        frontier.decoder_active_row_residency_requirement == "required"
        and decision is not None
        and not decision.admitted
    ):
        raise ActiveDecoderRowResidencyRequirementError(decision)
    effective = _effective_execution_identity(
        prepared.provider,
        prepared.numerics,
        prepared.replay,
        prepared.batches,
        frontier,
        prepared.plan,
    )
    return replace(
        prepared,
        frontier=frontier,
        effective_execution=effective,
    )


def finalize_phase0_decoder_row_range_execution(
    prepared: PreparedBackend,
    *,
    diagnostics: dict[str, object],
) -> PreparedBackend:
    """Bind requested range execution to observed Phase-0 seed capture."""

    frontier = prepared.frontier
    if not frontier.phase0_decoder_row_ranges_requested:
        return prepared
    if (
        not frontier.phase0_decoder_row_ranges_effective
        and frontier.phase0_decoder_row_ranges_fallback_reason != "seed_capture_refused"
    ):
        return prepared
    observed = "phase0_decoder_row_ranges_requested" in diagnostics
    effective = (
        bool(diagnostics.get("phase0_decoder_row_ranges_effective", False)) if observed else False
    )
    fallback_reason = (
        diagnostics.get("phase0_decoder_row_ranges_fallback_reason")
        if observed
        else "seed_capture_refused"
    )
    metadata = dict(frontier.execution_metadata)
    metadata.update(
        {
            "phase0_decoder_row_ranges_requested": True,
            "phase0_decoder_row_ranges_effective": effective,
            "phase0_decoder_row_ranges_fallback_reason": fallback_reason,
        }
    )
    frontier = replace(
        frontier,
        execution_metadata=metadata,
        phase0_decoder_row_ranges_effective=effective,
        phase0_decoder_row_ranges_fallback_reason=(
            None if fallback_reason is None else str(fallback_reason)
        ),
    )
    identity = _effective_execution_identity(
        prepared.provider,
        prepared.numerics,
        prepared.replay,
        prepared.batches,
        frontier,
        prepared.plan,
    )
    return replace(
        prepared,
        frontier=frontier,
        effective_execution=identity,
    )
