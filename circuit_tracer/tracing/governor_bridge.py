"""Pre-load bridge from canonical trace requests to governed execution plans."""

from __future__ import annotations

import math
from dataclasses import replace
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
)
from circuit_tracer.governor.resolver import resolve_trace_plan
from circuit_tracer.transcoder.provider import provider_fingerprint

from .plan import DecoderCachePolicy, DecoderPlan, ResolvedTracePlan
from .request import TraceRequest


class PlanningRefusedError(RuntimeError):
    """Raised before backend execution when governed admission refuses."""

    def __init__(self, plan: ResolvedTracePlan) -> None:
        self.plan = plan
        report = plan.admission_report
        detail = "; ".join(report.refusals) if report is not None else "unknown refusal"
        super().__init__(f"trace planning refused: {detail}")


def _prompt_token_count(problem: Any) -> int:
    prompt = problem.prompt
    if isinstance(prompt, str):
        tokenizer = getattr(problem.model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "governed pre-load planning requires token ids or an already available tokenizer"
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
        "d_transcoder": profile.dimensions.d_features,
        "decoder_output_topology": expected.decoder_topology.value,
    }
    for name, expected_value in comparisons.items():
        actual_value = actual.get(name)
        if actual_value is not None and actual_value != expected_value:
            mismatches.append(f"{name}={actual_value!r} (profile {expected_value!r})")
    if mismatches:
        raise ValueError("provider profile mismatch: " + ", ".join(mismatches))


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
    return PlanningWorkload(
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
        fidelity=FidelityMode.STRICT,
    )


def _requirements(request: TraceRequest) -> PhysicalExecutionRequirements:
    execution = request.execution
    session = execution.session
    storage = execution.storage
    defaults = type(execution)()
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
        source_microbatch_size=session.source_microbatch_max_rows,
        feature_microbatch_size=session.phase4_microbatch_max_rows,
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
        encoder_residency=(
            EncoderResidency.EAGER
            if storage.exact_encoder_residency != "lazy"
            else None
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
    storage = replace(
        execution.storage,
        retention="none_recompute" if row_policy == "recompute" else "full_file",
        full_retention_backend=(
            "column_tiled_v1" if row_policy == "tiled" else "full_file"
        ),
        feature_column_tile_size=(
            planning.profile.row_store_tile_column_bound
            if row_policy == "tiled"
            else execution.storage.feature_column_tile_size
        ),
        exact_encoder_residency=(
            "active_cpu" if physical.encoder_residency == "eager" else "lazy"
        ),
        placement=(
            StorageTier(physical.spill_target)
            if physical.spill_target is not None
            else None
        ),
    )
    capacity = max(
        physical.source_microbatch_size,
        physical.feature_microbatch_size,
        physical.logit_microbatch_size,
    )
    session = replace(
        execution.session,
        capacity=capacity,
        source_microbatch_max_rows=physical.source_microbatch_size,
        phase3_microbatch_max_rows=physical.logit_microbatch_size,
        phase4_microbatch_max_rows=physical.feature_microbatch_size,
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=physical.source_microbatch_size,
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
    """Resolve and compile the deterministic pre-load Phase E planning epoch."""

    _validate_provider(request.problem, provider_profile)
    planning = resolve_trace_plan(
        _workload(request, provider_profile, explicit_plan.semantic_fingerprint),
        provider_profile,
        resources,
        _requirements(request),
    )
    compiled_request = replace(request, execution=_compile_execution(request, planning))
    compiled = resolve_explicit(compiled_request)
    resolved = replace(
        compiled,
        semantic_fingerprint=explicit_plan.semantic_fingerprint,
        admission_report=planning.admission,
    )
    if not planning.admission.admitted:
        raise PlanningRefusedError(resolved)
    return resolved
