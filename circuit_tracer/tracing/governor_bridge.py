"""Pre-execution bridge from canonical trace requests to governed execution plans."""

from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Any

from circuit_tracer.governor.contracts import (
    CachePolicy,
    EncoderResidency,
    FidelityMode,
    PhysicalExecutionRequirements,
    ProviderProfile,
    ResourceEnvelope,
    RowStorePolicy,
    StorageTier,
    TraceSemantics as PlanningWorkload,
    canonical_json,
)
from circuit_tracer.governor.resolver import resolve_trace_plan
from circuit_tracer.transcoder.provider import provider_fingerprint

from .plan import DecoderCachePolicy, DecoderPlan, ResolvedTracePlan
from .plan import TraceEvidence
from .request import GovernorFidelityPolicy, TraceRequest


_FIDELITY_BOOKKEEPING_FIELDS = frozenset(
    {
        "fidelity",
        "evidence_name",
        "evidence_version",
        "semantic_overrides",
        "research_overrides",
    }
)


def _prompt_token_count(problem: Any) -> int:
    prompt = problem.prompt
    if isinstance(prompt, str):
        tokenizer = getattr(problem.model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "governed pre-execution planning requires token ids or an available tokenizer"
            )
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        return len(encoded)
    if isinstance(prompt, list):
        return len(prompt)
    numel = getattr(prompt, "numel", None)
    if callable(numel):
        return int(numel())
    raise ValueError("governed planning cannot determine prompt token count")


def _target_count(problem: Any) -> int:
    targets = problem.targets
    if targets is None:
        return problem.max_n_logits
    if hasattr(targets, "shape") and len(targets.shape):
        return int(targets.shape[-1])
    return len(targets)


def _validate_provider(problem: Any, profile: ProviderProfile) -> None:
    provider = getattr(problem.model, "transcoders", None)
    if provider is None:
        raise ValueError("governed planning requires a concrete transcoder provider")
    actual = provider_fingerprint(getattr(provider, "_module", provider))
    expected = profile.identity
    mismatches: list[str] = []
    comparisons = {
        "architecture": expected.architecture,
        "n_layers": profile.dimensions.n_layers,
        "d_model": profile.dimensions.d_model,
        "decoder_output_topology": expected.decoder_topology.value,
    }
    # CLT providers report features per source layer, while the governor profile
    # stores the aggregate cross-layer feature universe. PLT widths are directly
    # comparable because their topology is same-layer.
    if expected.architecture != "clt":
        comparisons["d_transcoder"] = profile.dimensions.d_features
    for name, expected_value in comparisons.items():
        actual_value = actual.get(name)
        if actual_value is not None and actual_value != expected_value:
            mismatches.append(f"{name}={actual_value!r} (profile {expected_value!r})")
    if mismatches:
        raise ValueError("provider profile mismatch: " + ", ".join(mismatches))


def _validate_load_time_mechanisms(problem: Any, planning: Any) -> None:
    provider = getattr(problem.model, "transcoders", None)
    actual = provider_fingerprint(getattr(provider, "_module", provider))
    actual_chunk = actual.get("decoder_chunk_size")
    planned_chunk = planning.physical.decoder_fetch_chunk_size
    if actual_chunk is not None and int(actual_chunk) != int(planned_chunk):
        raise ValueError(
            "loaded decoder chunk size does not match governed plan: "
            f"loaded {actual_chunk}, planned {planned_chunk}"
        )


def _workload(
    request: TraceRequest,
    profile: ProviderProfile,
    semantic_fingerprint: str,
) -> PlanningWorkload:
    token_count = _prompt_token_count(request.problem)
    semantics = request.semantics
    session = request.execution.session
    feature_batch = semantics.feature_batch_size or semantics.source_batch_size
    logit_batch = semantics.logit_batch_size or semantics.source_batch_size
    max_features = semantics.max_feature_nodes or math.ceil(
        token_count * profile.estimated_active_features_per_token
    )
    workload = PlanningWorkload(
        prompt_token_count=token_count,
        estimated_active_features=max(
            1, math.ceil(token_count * profile.estimated_active_features_per_token)
        ),
        max_feature_nodes=max_features,
        target_count=max(1, _target_count(request.problem)),
        scenario_id=semantic_fingerprint,
        environment_label=profile.profile_name,
        source_batch_size=semantics.source_batch_size,
        feature_batch_size=feature_batch,
        logit_batch_size=logit_batch,
        phase1_source_cap=(
            session.phase1_trace_batch_size_max
            if session.phase1_trace_batch_policy == "cap_effective_batches"
            else None
        ),
        decoder_reduction_tile=request.execution.replay.decoder_contraction_tile or 4096,
        frontier_refresh_stride=semantics.update_interval,
        dtype=semantics.exact_trace_internal_dtype,
    )
    return _authorize_fidelity(workload, request.governor_fidelity)


def _authorize_fidelity(
    workload: PlanningWorkload,
    policy: GovernorFidelityPolicy,
) -> PlanningWorkload:
    if policy.mode is FidelityMode.STRICT:
        return workload

    semantic_fields = {item.name for item in fields(workload)} - _FIDELITY_BOOKKEEPING_FIELDS
    invalid = tuple(name for name in policy.override_fields if name not in semantic_fields)
    if invalid:
        names = ", ".join(invalid)
        raise ValueError(
            "unknown or non-semantic PlanningWorkload override fields: " + names
        )
    overrides = tuple(
        (name, canonical_json(getattr(workload, name))) for name in policy.override_fields
    )
    if policy.mode is FidelityMode.RESEARCH:
        return replace(
            workload,
            fidelity=policy.mode,
            research_overrides=overrides,
        )
    return replace(
        workload,
        fidelity=policy.mode,
        evidence_name=policy.evidence_name,
        evidence_version=policy.evidence_version,
        semantic_overrides=overrides,
    )


def _requirements(request: TraceRequest) -> PhysicalExecutionRequirements:
    execution = request.execution
    session = execution.session
    storage = execution.storage
    defaults = type(execution)()
    if storage.temp_root is not None:
        raise ValueError(
            "governed storage does not accept an unmanaged temp_root; "
            "declare a storage placement through the resource envelope"
        )
    row_policy = None
    if storage.retention == "none_recompute":
        row_policy = RowStorePolicy.RECOMPUTE
    elif storage.full_retention_backend == "column_tiled_v1":
        row_policy = RowStorePolicy.TILED
    return PhysicalExecutionRequirements(
        decoder_fetch_chunk_size=execution.decoder.fetch_chunk_size,
        decoder_cache_bytes=(
            session.decoder_cache.max_bytes
            if session.decoder_cache.enabled and session.decoder_cache.max_bytes is not None
            else None
        ),
        session_capacity=session.capacity,
        phase1_source_batch_size=(
            session.phase1_trace_batch_size_max
            if session.phase1_trace_batch_policy == "cap_effective_batches"
            else None
        ),
        source_microbatch_size=session.source_microbatch_max_rows,
        feature_microbatch_size=session.phase4_execution_batch_max_rows,
        logit_microbatch_size=session.phase3_microbatch_max_rows,
        replay_window=(
            execution.replay.feature_window
            if execution.replay.feature_window != defaults.replay.feature_window
            else None
        ),
        prefetch_depth=(
            execution.replay.error_vector_prefetch_lookahead
            if execution.replay.error_vector_prefetch_lookahead
            != defaults.replay.error_vector_prefetch_lookahead
            else None
        ),
        replay_tile_cache_bytes=storage.replay_tile_cache_bytes,
        encoder_residency=(
            EncoderResidency.EAGER if storage.exact_encoder_residency != "lazy" else None
        ),
        row_store_policy=row_policy,
        spill_target=storage.placement,
        cache_policy=(
            execution.decoder.provider_file_cache
            if execution.decoder.provider_file_cache is not CachePolicy.AUTO
            else None
        ),
    )


def _compile_execution(request: TraceRequest, planning: Any) -> Any:
    physical = planning.physical
    execution = request.execution
    row_policy = physical.row_store_policy
    placement = (
        StorageTier(physical.spill_target)
        if physical.spill_target is not None
        else None
    )
    storage = replace(
        execution.storage,
        retention="none_recompute" if row_policy == "recompute" else "full_file",
        full_retention_backend=("column_tiled_v1" if row_policy == "tiled" else "full_file"),
        feature_column_tile_size=(
            planning.profile.row_store_tile_column_bound
            if row_policy == "tiled"
            else execution.storage.feature_column_tile_size
        ),
        exact_encoder_residency=("active_cpu" if physical.encoder_residency == "eager" else "lazy"),
        temp_root_policy=(
            "env_node_local" if placement is StorageTier.LOCAL else "default"
        ),
        placement=placement,
        replay_tile_cache_bytes=physical.replay_tile_cache_bytes,
    )
    phase1_is_capped = (
        physical.phase1_source_batch_size < planning.semantics.source_batch_size
    )
    session = replace(
        execution.session,
        capacity=physical.session_capacity,
        source_microbatch_max_rows=physical.source_microbatch_size,
        phase3_microbatch_max_rows=physical.logit_microbatch_size,
        phase4_execution_batch_max_rows=physical.feature_microbatch_size,
        phase1_trace_batch_policy=("cap_effective_batches" if phase1_is_capped else "legacy"),
        phase1_trace_batch_size_max=(
            physical.phase1_source_batch_size if phase1_is_capped else None
        ),
        decoder_cache=DecoderCachePolicy(
            enabled=physical.decoder_cache_bytes > 0,
            max_bytes=(physical.decoder_cache_bytes or None),
        ),
    )
    return replace(
        execution,
        decoder=DecoderPlan(
            fetch_chunk_size=physical.decoder_fetch_chunk_size,
            provider_file_cache=physical.cache_policy,
        ),
        session=session,
        storage=storage,
        replay=replace(
            execution.replay,
            feature_window=physical.replay_window,
            error_vector_prefetch_lookahead=physical.prefetch_depth,
        ),
    )


def resolve_governed_trace_request(
    request: TraceRequest,
    resources: ResourceEnvelope,
    provider_profile: ProviderProfile,
    *,
    explicit_plan: ResolvedTracePlan,
    resolve_explicit: Any,
) -> ResolvedTracePlan:
    """Resolve and compile the deterministic pre-execution Phase E planning epoch."""

    _validate_provider(request.problem, provider_profile)
    workload = _workload(request, provider_profile, explicit_plan.semantic_fingerprint)
    requirements = request.physical_requirements or _requirements(request)
    planning = resolve_trace_plan(
        workload,
        provider_profile,
        resources,
        requirements,
    )
    _validate_load_time_mechanisms(request.problem, planning)
    compiled_request = replace(request, execution=_compile_execution(request, planning))
    compiled = resolve_explicit(compiled_request)
    resolved = replace(
        compiled,
        semantic_fingerprint=explicit_plan.semantic_fingerprint,
        admission_report=planning.admission,
        planning_profile=provider_profile,
        planning_envelope=resources,
        planning_workload=workload,
        planning_requirements=requirements,
        planning_trace_plan=planning,
        planning_parent_fingerprint=explicit_plan.requested_execution_fingerprint,
        planning_epoch_fingerprint=planning.execution_fingerprint,
    )
    return resolved


def compile_governed_revision(
    request: TraceRequest,
    current: ResolvedTracePlan,
    planning: Any,
    *,
    resolve_explicit: Any,
) -> ResolvedTracePlan:
    """Compile a later Phase E plan without creating another tracing path."""
    compiled = resolve_explicit(replace(request, execution=_compile_execution(request, planning)))
    return replace(
        compiled,
        semantic_fingerprint=current.semantic_fingerprint,
        admission_report=planning.admission,
        planning_profile=current.planning_profile,
        planning_envelope=current.planning_envelope,
        planning_workload=current.planning_workload,
        planning_requirements=current.planning_requirements,
        planning_trace_plan=planning,
        planning_parent_fingerprint=current.planning_epoch_fingerprint,
        planning_epoch_fingerprint=planning.execution_fingerprint,
    )


def recompile_governed_plan(
    problem: Any,
    current: ResolvedTracePlan,
    planning: Any,
) -> ResolvedTracePlan:
    """Recompile the canonical request after a live governor revision."""
    from .planning import _resolve_explicit_trace_request

    request = TraceRequest(
        problem=problem,
        semantics=current.semantics,
        execution=current.execution,
        evidence=TraceEvidence(metadata=current.evidence_metadata),
        physical_requirements=current.planning_requirements,
        governor_admission_mode=current.governor_admission_mode,
    )
    return compile_governed_revision(
        request,
        current,
        planning,
        resolve_explicit=_resolve_explicit_trace_request,
    )
