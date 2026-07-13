"""Validated mechanism preparation for the NNSight attribution backend."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable

import torch

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
from circuit_tracer.tracing.plan import ResolvedTracePlan
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
        use_compact_feature_row_store=bool(
            plan.execution.compact_output and compact_row_store
        ),
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
        storage.full_retention_backend == "column_tiled_v1"
        or storage.retention == "none_recompute"
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
        provider.use_compact_feature_row_store
        and refresh_optimization.effective_mode == "v1"
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
    metadata = {
        **_build_phase4_scheduler_metadata(scheduler),
        **refresh_metadata,
        **_build_phase4_row_executor_metadata(row_executor),
        **_build_phase4_row_reduction_metadata(row_reduction),
        **_build_phase4_refresh_policy_metadata(refresh_policy),
        **_build_phase4_ranker_metadata(ranker),
        **_build_row_store_cache_control_metadata(cache_control),
        **residency_metadata,
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
        phase4_compute_microbatch_max_rows=session.phase4_microbatch_max_rows,
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
        )
        planner_status = "executed"
    legacy_phase4_rows = feature_size
    if frontier.row_executor.effective_mode == "streaming_v1":
        legacy_phase4_rows = _resolve_phase4_streaming_v1_microbatch_size(feature_size)
    controls = resolve_nnsight_session_controls(
        nnsight_session_capacity=session.capacity,
        phase3_compute_microbatch_max_rows=session.phase3_microbatch_max_rows,
        phase4_compute_microbatch_max_rows=session.phase4_microbatch_max_rows,
        legacy_session_capacity=max(source_size, feature_size, logit_size),
        legacy_phase3_batch_rows=logit_size,
        legacy_phase4_batch_rows=legacy_phase4_rows,
    )
    metadata.update(
        trace_batch_size_legacy=int(sizing.trace_batch_size_legacy),
        trace_batch_size_effective=int(controls.session_capacity),
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
) -> EffectiveExecutionIdentity:
    """Describe only mechanisms fixed by successful NNSight preparation."""
    descriptor = EffectiveExecutionDescriptor(
        schema_version=1,
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
            "phase4_microbatch_max_rows": batches.session_controls.phase4_microbatch_max_rows,
            "trace_batch_size": batches.trace_batch_size,
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
        },
    )
    return EffectiveExecutionIdentity(descriptor=descriptor, fingerprint=fingerprint(descriptor))


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
    diagnostics = _prepare_diagnostics(
        plan, provider, numerics, replay, observer
    )
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
        provider, numerics, replay, batches, frontier
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
