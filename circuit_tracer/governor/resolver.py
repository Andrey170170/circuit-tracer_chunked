from __future__ import annotations

import math
from dataclasses import replace
from itertools import product

from .contracts import (
    AdmissionReport,
    CachePolicy,
    DemandClass,
    DemandEstimate,
    DemandLifetime,
    DemandTier,
    FidelityMode,
    PhysicalExecutionConfig,
    PhysicalExecutionRequirements,
    PlanStatus,
    ProviderDimensions,
    ProviderCostMetadata,
    ProviderProfile,
    ResourceEnvelope,
    RowStorePolicy,
    TracePlan,
    TraceSemantics,
    TRUSTED_VALIDATION_EVIDENCE_REGISTRY,
    execution_fingerprint,
    semantic_fingerprint,
)


def compute_work_units(
    *,
    prompt_token_count: int,
    estimated_active_features: int,
    target_count: int,
    dtype_bytes: int,
    trace_capacity: int,
    dimensions: ProviderDimensions,
    decoder_fetch_chunk_size: int,
    effective_source_batch_size: int,
    feature_batch_size: int,
    logit_batch_size: int,
    source_microbatch_size: int,
    feature_microbatch_size: int,
    logit_microbatch_size: int,
    replay_window: int,
    prefetch_depth: int,
) -> float:
    fetch_blocks = math.ceil(dimensions.d_features / decoder_fetch_chunk_size)
    # Phase 1 is a small admission/survival stage in the calibration runs. Its
    # session width is not a sequenced source microbatch, so only the physical
    # Phase 3/4 partitions contribute to this throughput model.
    microbatch_steps = math.ceil(
        feature_batch_size / feature_microbatch_size
    ) + math.ceil(logit_batch_size / logit_microbatch_size)
    logical_work = (
        prompt_token_count
        * estimated_active_features
        * target_count
        * dtype_bytes
        * trace_capacity
        * dimensions.n_layers
        * dimensions.decoder_output_span
    )
    replay_factor = 1 + 0.02 * (replay_window - 1)
    prefetch_factor = 1 + 0.05 * prefetch_depth
    return logical_work * fetch_blocks * microbatch_steps * replay_factor / prefetch_factor


def _effective_batches(
    semantics: TraceSemantics,
) -> tuple[int, int, tuple[str, ...]]:
    effective_source = min(
        semantics.source_batch_size,
        semantics.phase1_source_cap or semantics.source_batch_size,
    )
    values = {
        "source": effective_source,
        "feature": semantics.feature_batch_size,
        "logit": semantics.logit_batch_size,
    }
    capacity = max(values.values())
    bindings = tuple(name for name, value in values.items() if value == capacity)
    return effective_source, capacity, bindings


def _validate_evidence(
    semantics: TraceSemantics,
    profile: ProviderProfile,
) -> str | None:
    if semantics.fidelity is FidelityMode.STRICT:
        return None
    if semantics.fidelity is FidelityMode.RESEARCH:
        return None
    key = (semantics.evidence_name or "", semantics.evidence_version or "")
    evidence = TRUSTED_VALIDATION_EVIDENCE_REGISTRY.get(key)
    if evidence is None:
        raise ValueError(
            "validated_relaxed has no trusted package evidence for "
            f"{key[0]!r} version {key[1]!r}"
        )
    identity = profile.identity
    expected = {
        "evidence_id": semantics.evidence_name,
        "evidence_version": semantics.evidence_version,
        "provider_type": identity.provider_type,
        "provider_version": identity.provider_version,
        "checkpoint_identity": identity.checkpoint_identity,
        "hook_identity": identity.hook_identity,
        "architecture": identity.architecture,
        "decoder_topology": identity.decoder_topology,
        "provider_approximation": identity.approximation,
        "provider_semantic_parameters": identity.semantic_parameters,
        "semantic_parameters": semantics.evidence_scope_parameters(),
        "dtype": semantics.dtype,
        "scenario_id": semantics.scenario_id,
        "window_id": semantics.window_id,
        "environment_label": semantics.environment_label,
        "allowed_semantic_overrides": semantics.semantic_overrides,
    }
    mismatches = [
        name for name, expected_value in expected.items() if getattr(evidence, name) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "validation evidence scope mismatch: " + ", ".join(sorted(mismatches))
        )
    return evidence.evidence_fingerprint


def _cache_policy(file_bytes: int, allowance: int, requested: CachePolicy) -> CachePolicy:
    if requested is not CachePolicy.AUTO:
        return requested
    if file_bytes <= allowance:
        return CachePolicy.WARM
    if allowance > 0 and file_bytes <= allowance * 4:
        return CachePolicy.BOUNDED
    return CachePolicy.STREAMING


def _disk_available(envelope: ResourceEnvelope, root: str) -> int:
    return envelope.local_disk_bytes if root == "local" else envelope.scratch_disk_bytes


def _row_store_bytes(
    policy: str,
    semantics: TraceSemantics,
    profile: ProviderProfile,
) -> int:
    if policy == "file_backed_full":
        return (semantics.max_feature_nodes + 1) * semantics.estimated_nnz * semantics.dtype_bytes
    if policy == "tiled":
        bound = profile.row_store_tile_column_bound
        if bound is None:
            raise ValueError("tiled row store requires a profile tile column bound")
        return (
            (semantics.max_feature_nodes + 1)
            * min(semantics.estimated_nnz, bound)
            * semantics.dtype_bytes
        )
    if policy == "recompute":
        return 0
    raise ValueError(f"unknown row_store_policy override: {policy}")


def _row_store_supported(policy: str, profile: ProviderProfile) -> bool:
    capabilities = profile.capabilities
    return {
        "file_backed_full": capabilities.supports_full_row_store,
        "tiled": capabilities.supports_tiled_row_store,
        "recompute": capabilities.supports_recompute_row_store,
    }[policy]


def _select_row_store(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    policy_override: str | None,
    spill_override: str | None,
) -> tuple[str, int, str | None, str | None]:
    policies = (
        (policy_override,)
        if policy_override is not None
        else ("file_backed_full", "tiled", "recompute")
    )
    roots = (spill_override,) if spill_override is not None else envelope.spill_roots
    for policy in policies:
        if not _row_store_supported(policy, profile):
            if policy_override is not None:
                raise ValueError(f"provider does not support row_store_policy={policy}")
            continue
        required = _row_store_bytes(policy, semantics, profile)
        if policy == "recompute":
            if spill_override is not None:
                raise ValueError("recompute row store cannot have a spill_target override")
            return policy, 0, None, None
        for root in roots:
            if root not in {"local", "scratch"}:
                raise ValueError(f"unknown spill_target override: {root}")
            if _disk_available(envelope, root) >= required:
                return policy, required, root, None
        if policy_override is not None:
            return (
                policy,
                required,
                None,
                f"row store {policy} requires {required} B but no configured spill tier fits",
            )
    full_bytes = _row_store_bytes("file_backed_full", semantics, profile)
    return (
        "unavailable",
        0,
        None,
        "no supported row-store rung fits; full row store requires "
        f"{full_bytes} B, local has {envelope.local_disk_bytes} B, and scratch has "
        f"{envelope.scratch_disk_bytes} B",
    )


def _resolve_physical(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    effective_source: int,
    requirements: PhysicalExecutionRequirements,
) -> tuple[PhysicalExecutionConfig, tuple[str, ...], tuple[str, ...]]:
    capabilities = profile.capabilities
    fetch = requirements.decoder_fetch_chunk_size or profile.default_fetch_chunk_size
    if not 0 < fetch <= profile.max_fetch_chunk_size:
        raise ValueError("decoder_fetch_chunk_size must be positive and within profile maximum")
    cache = (
        profile.default_decoder_cache_bytes
        if requirements.decoder_cache_bytes is None
        else requirements.decoder_cache_bytes
    )
    if cache < 0 or cache > profile.max_decoder_cache_bytes:
        raise ValueError("decoder_cache_bytes must be nonnegative and within profile maximum")
    if cache and not capabilities.supports_decoder_chunk_cache:
        raise ValueError("provider does not support decoder_cache_bytes > 0")

    logical_capacity = max(
        effective_source, semantics.feature_batch_size, semantics.logit_batch_size
    )
    session_capacity = (
        min(logical_capacity, profile.max_session_capacity)
        if requirements.session_capacity is None
        else requirements.session_capacity
    )
    if not 0 < session_capacity <= logical_capacity:
        raise ValueError("session_capacity must be positive and <= logical trace capacity")
    if session_capacity > profile.max_session_capacity:
        raise ValueError("session_capacity exceeds the provider profile maximum")

    phase1_source_batch_size = (
        min(effective_source, profile.max_phase1_source_batch_size)
        if requirements.phase1_source_batch_size is None
        else requirements.phase1_source_batch_size
    )
    if not 0 < phase1_source_batch_size <= effective_source:
        raise ValueError(
            "phase1_source_batch_size must be positive and <= effective source batch"
        )
    if phase1_source_batch_size > profile.max_phase1_source_batch_size:
        raise ValueError("phase1_source_batch_size exceeds the provider profile maximum")

    batch_specs = (
        (
            "source_microbatch_size",
            effective_source,
            profile.max_source_microbatch_size,
            requirements.source_microbatch_size,
        ),
        (
            "feature_microbatch_size",
            semantics.feature_batch_size,
            profile.max_phase4_microbatch_size,
            requirements.feature_microbatch_size,
        ),
        (
            "logit_microbatch_size",
            semantics.logit_batch_size,
            profile.max_phase3_microbatch_size,
            requirements.logit_microbatch_size,
        ),
    )
    microbatches: dict[str, int] = {}
    for name, logical_bound, profile_bound, required in batch_specs:
        value = min(logical_bound, profile_bound, session_capacity) if required is None else required
        if not 0 < value <= logical_bound:
            raise ValueError(f"{name} must be positive and <= its logical/effective capacity")
        if value > profile_bound:
            raise ValueError(f"{name} exceeds the provider profile maximum")
        if value > session_capacity:
            raise ValueError(f"{name} cannot exceed session_capacity")
        microbatches[name] = value

    replay = requirements.replay_window or profile.default_replay_window
    if not 0 < replay <= profile.max_replay_window:
        raise ValueError("replay_window must be positive and within profile maximum")
    if replay > 1 and not capabilities.supports_replay:
        raise ValueError("provider does not support replay_window > 1")
    prefetch = (
        profile.default_prefetch_depth
        if requirements.prefetch_depth is None
        else requirements.prefetch_depth
    )
    if not 0 <= prefetch <= profile.max_prefetch_depth:
        raise ValueError("prefetch_depth must be nonnegative and within profile maximum")
    if prefetch and not capabilities.supports_prefetch:
        raise ValueError("provider does not support prefetch_depth > 0")

    warnings: list[str] = []
    refusals: list[str] = []
    if requirements.encoder_residency is not None:
        residency = requirements.encoder_residency.value
    else:
        eager_bytes = semantics.estimated_nnz * profile.dimensions.d_model * semantics.dtype_bytes
        costs = profile.costs
        if costs.baseline_total_host_bytes is None:
            eager_reservation = costs.known_rigid_host_bytes + eager_bytes
        else:
            eager_increment = (
                0 if costs.reference_encoder_residency == "eager" else eager_bytes
            )
            eager_reservation = max(
                costs.baseline_total_host_bytes,
                costs.known_rigid_host_bytes + eager_increment,
            )
        eager_fits = eager_reservation <= envelope.host_budget_bytes
        preferred = profile.default_encoder_residency.value
        if preferred == "eager" and capabilities.supports_encoder_row_materialization and eager_fits:
            residency = preferred
        elif capabilities.supports_lazy_encoder_rows:
            residency = "lazy_per_request"
        elif capabilities.supports_encoder_row_materialization and eager_fits:
            residency = "eager"
        else:
            residency = "unavailable"
            refusals.append("provider exposes neither fitting eager nor lazy encoder rows")
    if residency == "eager" and not capabilities.supports_encoder_row_materialization:
        raise ValueError("provider does not support encoder_residency=eager")
    if residency == "lazy_per_request" and not capabilities.supports_lazy_encoder_rows:
        raise ValueError("provider does not support encoder_residency=lazy_per_request")
    if residency not in {"eager", "lazy_per_request", "unavailable"}:
        raise ValueError(f"unknown encoder_residency override: {residency}")

    policy_override = (
        requirements.row_store_policy.value
        if requirements.row_store_policy is not None
        else None
    )
    spill_override = (
        requirements.spill_target.value
        if requirements.spill_target is not None
        else None
    )
    row_policy, row_bytes, spill_target, row_refusal = _select_row_store(
        semantics,
        profile,
        envelope,
        policy_override,
        spill_override,
    )
    if row_refusal:
        refusals.append(row_refusal)
    if row_policy != "file_backed_full":
        warnings.append(f"selected row-store degradation rung {row_policy}")

    physical = PhysicalExecutionConfig(
        decoder_fetch_chunk_size=fetch,
        decoder_cache_bytes=cache,
        session_capacity=session_capacity,
        phase1_source_batch_size=phase1_source_batch_size,
        source_microbatch_size=microbatches["source_microbatch_size"],
        feature_microbatch_size=microbatches["feature_microbatch_size"],
        logit_microbatch_size=microbatches["logit_microbatch_size"],
        replay_window=replay,
        prefetch_depth=prefetch,
        encoder_residency=residency,
        row_store_policy=row_policy,
        row_store_bytes=row_bytes,
        spill_target=spill_target,
        cache_policy=CachePolicy.AUTO,
    )
    return physical, tuple(warnings), tuple(refusals)


def _estimate_demands(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    physical: PhysicalExecutionConfig,
    effective_source: int,
    capacity: int,
) -> tuple[DemandEstimate, ...]:
    costs = profile.costs
    dims = profile.dimensions
    dtype_bytes = semantics.dtype_bytes
    trace_elements = (
        physical.session_capacity
        * semantics.prompt_token_count
        * dims.n_layers
        * dims.decoder_output_span
        * dims.d_model
        * dtype_bytes
    )
    target_elements = (
        semantics.target_count * semantics.prompt_token_count * dims.d_model * dtype_bytes
    )
    trace_vram = math.ceil(trace_elements * costs.trace_vram_coefficient)
    target_vram = math.ceil(target_elements * costs.target_vram_coefficient)
    fetch_vram = (
        physical.decoder_fetch_chunk_size
        * dims.d_model
        * dims.decoder_output_span
        * dtype_bytes
    )
    prefetch_vram = fetch_vram * physical.prefetch_depth
    # No runtime path currently executes source_microbatch_size as a physical
    # partition. Keep the named estimate for schema stability, but do not charge
    # fictitious residency until a sequenced source executor consumes it.
    source_microbatch_vram = 0
    feature_microbatch_vram = math.ceil(
        physical.feature_microbatch_size
        * semantics.prompt_token_count
        * min(semantics.estimated_nnz, dims.d_features)
        * dtype_bytes
        * costs.feature_microbatch_vram_coefficient
    )
    logit_microbatch_vram = math.ceil(
        physical.logit_microbatch_size
        * semantics.prompt_token_count
        * semantics.target_count
        * dims.d_model
        * dtype_bytes
        * costs.logit_microbatch_vram_coefficient
    )
    replay_vram = math.ceil(
        physical.replay_window
        * semantics.prompt_token_count
        * semantics.estimated_nnz
        * dims.d_model
        * dtype_bytes
        * costs.replay_vram_coefficient
    )
    active_host = math.ceil(
        semantics.estimated_nnz * dtype_bytes * costs.active_host_coefficient
    )
    prompt_host = math.ceil(
        semantics.prompt_token_count
        * dims.n_layers
        * dims.d_model
        * dtype_bytes
        * costs.prompt_host_coefficient
    )
    if costs.baseline_total_host_bytes is not None:
        encoder_host = (
            semantics.estimated_nnz * dims.d_model * dtype_bytes
            if physical.encoder_residency == "eager"
            and costs.reference_encoder_residency != "eager"
            else 0
        )
        replay_host = (
            max(0, physical.replay_window - costs.reference_replay_window)
            * semantics.estimated_nnz
            * dtype_bytes
        )
    else:
        encoder_host = (
            semantics.estimated_nnz * dims.d_model * dtype_bytes
            if physical.encoder_residency == "eager"
            else 0
        )
        replay_host = physical.replay_window * semantics.estimated_nnz * dtype_bytes
    work_units = compute_work_units(
        prompt_token_count=semantics.prompt_token_count,
        estimated_active_features=semantics.estimated_nnz,
        target_count=semantics.target_count,
        dtype_bytes=dtype_bytes,
        trace_capacity=capacity,
        dimensions=dims,
        decoder_fetch_chunk_size=physical.decoder_fetch_chunk_size,
        effective_source_batch_size=effective_source,
        feature_batch_size=semantics.feature_batch_size,
        logit_batch_size=semantics.logit_batch_size,
        source_microbatch_size=physical.phase1_source_batch_size,
        feature_microbatch_size=physical.feature_microbatch_size,
        logit_microbatch_size=physical.logit_microbatch_size,
        replay_window=physical.replay_window,
        prefetch_depth=physical.prefetch_depth,
    )
    walltime_scale = (
        work_units
        / costs.walltime_reference_work_units
        * costs.row_store_walltime_multiplier(physical.row_store_policy)
    )
    disk_tier = (
        DemandTier.LOCAL_DISK
        if physical.spill_target == "local"
        else DemandTier.SCRATCH_DISK if physical.spill_target == "scratch" else DemandTier.LOCAL_DISK
    )
    return (
        DemandEstimate("model_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PERMANENT, costs.fixed_vram_bytes),
        DemandEstimate("trace_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, trace_vram),
        DemandEstimate("target_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, target_vram),
        DemandEstimate("decoder_fetch_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.TRANSIENT, fetch_vram),
        DemandEstimate("decoder_cache_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, physical.decoder_cache_bytes),
        DemandEstimate("prefetch_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.TRANSIENT, prefetch_vram),
        DemandEstimate("source_microbatch_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, source_microbatch_vram),
        DemandEstimate("feature_microbatch_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, feature_microbatch_vram),
        DemandEstimate("logit_microbatch_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, logit_microbatch_vram),
        DemandEstimate("replay_vram", DemandTier.VRAM, DemandClass.RIGID, DemandLifetime.PHASE, replay_vram),
        DemandEstimate("baseline_total_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PERMANENT, costs.baseline_total_host_bytes or 0),
        DemandEstimate("known_rigid_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PERMANENT, costs.known_rigid_host_bytes),
        DemandEstimate("active_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PHASE, active_host),
        DemandEstimate("prompt_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PHASE, prompt_host),
        DemandEstimate("encoder_residency_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PHASE, encoder_host),
        DemandEstimate("replay_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PHASE, replay_host),
        DemandEstimate("checkpoint_file_working_set", DemandTier.FILE_BACKED, DemandClass.ELASTIC, DemandLifetime.PERMANENT, costs.checkpoint_file_bytes),
        DemandEstimate("row_store_disk", disk_tier, DemandClass.RIGID, DemandLifetime.PHASE, physical.row_store_bytes),
        DemandEstimate("predicted_walltime_low", DemandTier.WALLTIME, DemandClass.RIGID, DemandLifetime.PHASE, costs.calibrated_walltime_low_seconds * walltime_scale, "seconds"),
        DemandEstimate("predicted_walltime_high", DemandTier.WALLTIME, DemandClass.RIGID, DemandLifetime.PHASE, costs.calibrated_walltime_high_seconds * walltime_scale, "seconds"),
    )


def _sum_estimates(estimates: tuple[DemandEstimate, ...], names: set[str]) -> int:
    return int(sum(item.amount for item in estimates if item.name in names))


def _peak_vram_total(estimates: tuple[DemandEstimate, ...]) -> int:
    permanent = _sum_estimates(estimates, {"model_vram"})
    session = _sum_estimates(
        estimates,
        {"trace_vram", "decoder_cache_vram", "source_microbatch_vram"},
    )
    phase_working_sets = (
        _sum_estimates(
            estimates,
            {"decoder_fetch_vram", "prefetch_vram", "replay_vram"},
        ),
        _sum_estimates(estimates, {"target_vram", "logit_microbatch_vram"}),
        _sum_estimates(estimates, {"feature_microbatch_vram"}),
    )
    return permanent + session + max(phase_working_sets)


def _peak_host_reservation_total(
    estimates: tuple[DemandEstimate, ...], profile: ProviderProfile
) -> int:
    phase_working_sets = (
        _sum_estimates(estimates, {"replay_host"}),
        _sum_estimates(estimates, {"prompt_host"}),
        _sum_estimates(estimates, {"active_host", "encoder_residency_host"}),
    )
    incremental_peak = max(phase_working_sets)
    baseline = profile.costs.baseline_total_host_bytes
    if baseline is None:
        return int(_amount(estimates, "known_rigid_host")) + incremental_peak
    return baseline + incremental_peak


def _amount(estimates: tuple[DemandEstimate, ...], name: str) -> float:
    return next(estimate.amount for estimate in estimates if estimate.name == name)


def _resolve_single_trace_plan(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    requirements: PhysicalExecutionRequirements | None = None,
) -> TracePlan:
    """Resolve a pure, provider-agnostic, advisory Phase B execution plan."""

    if not isinstance(profile.costs, ProviderCostMetadata) or not profile.costs.cost_model_version:
        raise ValueError("provider profile has incomplete cost metadata")
    evidence_hash = _validate_evidence(semantics, profile)
    effective_source, capacity, bindings = _effective_batches(semantics)
    requirements = requirements or PhysicalExecutionRequirements()
    physical, physical_warnings, physical_refusals = _resolve_physical(
        semantics, profile, envelope, effective_source, requirements
    )
    estimates = _estimate_demands(
        semantics, profile, physical, effective_source, capacity
    )

    warnings = list(physical_warnings)
    refusals = list(physical_refusals)
    decisions: list[str] = []
    if semantics.provider_approximation != profile.identity.approximation:
        refusals.append(
            "provider approximation mismatch: semantics request "
            f"{semantics.provider_approximation!r}, profile declares "
            f"{profile.identity.approximation!r}"
        )
    if semantics.phase1_source_cap is not None:
        uncapped_capacity = max(
            semantics.source_batch_size,
            semantics.feature_batch_size,
            semantics.logit_batch_size,
        )
        if capacity == uncapped_capacity:
            warnings.append(
                f"source cap {semantics.phase1_source_cap} does not lower trace capacity "
                f"{capacity}; binding={','.join(bindings)}"
            )

    vram_total = _peak_vram_total(estimates)
    host_total = _peak_host_reservation_total(estimates, profile)
    if profile.costs.file_cache_included_in_host_baseline:
        baseline = profile.costs.baseline_total_host_bytes or 0
        incremental_host = max(0, host_total - baseline)
        effective_file_allowance = min(
            envelope.file_cache_allowance_bytes,
            max(0, envelope.host_budget_bytes - incremental_host),
        )
        warnings.append(
            "baseline total host includes an unknown recorded file-cache component; "
            "the split is conservative and the included baseline is not reserved twice"
        )
    else:
        remaining_host = max(0, envelope.host_budget_bytes - host_total)
        effective_file_allowance = min(envelope.file_cache_allowance_bytes, remaining_host)
        if effective_file_allowance < envelope.file_cache_allowance_bytes:
            warnings.append(
                f"clamped file-cache allowance from {envelope.file_cache_allowance_bytes} B to "
                f"{effective_file_allowance} B so rigid host plus allowance stays within total host budget"
            )
    checkpoint_bytes = int(_amount(estimates, "checkpoint_file_working_set"))
    if (
        requirements.cache_policy is not None
        and envelope.cache_policy is not CachePolicy.AUTO
        and requirements.cache_policy is not envelope.cache_policy
    ):
        refusals.append(
            "explicit provider file-cache policy conflicts with the resource envelope"
        )
    requested_cache_policy = requirements.cache_policy or envelope.cache_policy
    cache_policy = _cache_policy(checkpoint_bytes, effective_file_allowance, requested_cache_policy)
    physical = replace(physical, cache_policy=cache_policy)
    if checkpoint_bytes > effective_file_allowance:
        warnings.append(
            f"checkpoint/file working set {checkpoint_bytes} B exceeds effective cache allowance "
            f"{effective_file_allowance} B; selected {cache_policy.value} policy"
        )
    if (
        physical.cache_policy is CachePolicy.STREAMING
        and not profile.capabilities.supports_streaming_decoder
    ):
        refusals.append(
            "streaming cache policy is required but provider lacks supports_streaming_decoder"
        )

    if vram_total > envelope.effective_vram_budget_bytes:
        refusals.append(
            f"VRAM allocations require {vram_total} B but budget is "
            f"{envelope.effective_vram_budget_bytes} B; reduce cache/microbatches or increase VRAM"
        )
    if host_total > envelope.host_budget_bytes:
        refusals.append(
            f"rigid host allocations require {host_total} B but total host budget is "
            f"{envelope.host_budget_bytes} B; use lazy residency or request more host memory"
        )
    walltime_high = _amount(estimates, "predicted_walltime_high")
    if walltime_high > envelope.walltime_seconds:
        refusals.append(
            f"predicted walltime upper bound {walltime_high:.2f} s exceeds allocation "
            f"{envelope.walltime_seconds:.2f} s; request more time or shrink the workload"
        )
    if physical.row_store_bytes:
        disk_available = _disk_available(envelope, physical.spill_target or "local")
        if physical.row_store_bytes > disk_available:
            refusals.append(
                f"selected row store requires {physical.row_store_bytes} B on "
                f"{physical.spill_target}, but only {disk_available} B is available"
            )

    decisions.extend(
        (
            f"effective_source_batch={effective_source}",
            f"session_capacity={physical.session_capacity}",
            f"phase1_source_batch={physical.phase1_source_batch_size}",
            f"microbatches=source:{physical.source_microbatch_size},"
            f"feature:{physical.feature_microbatch_size},logit:{physical.logit_microbatch_size}",
            f"decoder_cache_vram={physical.decoder_cache_bytes}",
            f"cache_policy={physical.cache_policy.value}",
            f"encoder_residency={physical.encoder_residency}",
            f"row_store={physical.row_store_policy}:{physical.row_store_bytes}",
            f"spill_target={physical.spill_target or 'none'}",
        )
    )
    report = AdmissionReport(
        admitted=not refusals,
        estimates=estimates,
        trace_capacity=capacity,
        binding_reasons=bindings,
        effective_file_cache_allowance_bytes=effective_file_allowance,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
        refusals=tuple(refusals),
    )
    semantic_hash = semantic_fingerprint(semantics, profile.identity)
    execution_hash = execution_fingerprint(
        profile=profile,
        envelope=envelope,
        physical=physical,
        admission=report,
        evidence_fingerprint=evidence_hash,
    )
    return TracePlan(
        semantics=semantics,
        profile=profile,
        envelope=envelope,
        physical=physical,
        admission=report,
        semantic_fingerprint=semantic_hash,
        execution_fingerprint=execution_hash,
        evidence_fingerprint=evidence_hash,
        status=(
            PlanStatus.ADVISORY_ADMITTED
            if report.admitted
            else PlanStatus.ADVISORY_REFUSED
        ),
    )


_OPTIMIZED_REQUIREMENT_FIELDS = (
    "session_capacity",
    "feature_microbatch_size",
    "logit_microbatch_size",
    "row_store_policy",
)


def _batch_breakpoint_sizes(
    *, logical_size: int, maximum: int, required: int | None
) -> tuple[int, ...]:
    if required is not None:
        return (required,)
    values = {1, maximum}
    physical_steps = 1
    numerator = logical_size - 1
    while physical_steps <= logical_size:
        quotient = numerator // physical_steps
        value = quotient + 1
        if value <= maximum:
            values.add(value)
        if quotient == 0:
            break
        physical_steps = numerator // quotient + 1
    return tuple(sorted(values, reverse=True))


def _requirement_descriptions(
    requirements: PhysicalExecutionRequirements,
) -> tuple[str, ...]:
    return tuple(
        f"{name}={getattr(requirements, name).value if hasattr(getattr(requirements, name), 'value') else getattr(requirements, name)}"
        for name in requirements.__dataclass_fields__
        if getattr(requirements, name) is not None
    )


def _candidate_summary(plan: TracePlan) -> str:
    physical = plan.physical
    refusal = plan.admission.refusals[0] if plan.admission.refusals else "not selected"
    return (
        f"session={physical.session_capacity},phase1={physical.phase1_source_batch_size},"
        f"source={physical.source_microbatch_size},phase3={physical.logit_microbatch_size},"
        f"phase4={physical.feature_microbatch_size},row_store={physical.row_store_policy}: "
        f"{refusal}"
    )


def _objective(plan: TracePlan) -> tuple[float, ...]:
    estimates = plan.admission.estimates
    walltime = _amount(estimates, "predicted_walltime_high")
    vram = _peak_vram_total(estimates)
    host = _peak_host_reservation_total(estimates, plan.profile)
    disk = _amount(estimates, "row_store_disk")
    return (
        walltime,
        float(vram),
        float(host),
        disk,
    )


def _with_optimization_report(
    plan: TracePlan,
    *,
    requirements: PhysicalExecutionRequirements,
    reported_hard_constraints: PhysicalExecutionRequirements,
    frozen_fields: tuple[str, ...],
    candidates: tuple[TracePlan, ...],
) -> TracePlan:
    admitted = tuple(candidate for candidate in candidates if candidate.admission.admitted)
    hard_constraints = _requirement_descriptions(reported_hard_constraints)
    free_fields = tuple(
        name
        for name in _OPTIMIZED_REQUIREMENT_FIELDS
        if getattr(requirements, name) is None and name not in frozen_fields
    )
    objective = _objective(plan)
    rejected = tuple(
        _candidate_summary(candidate)
        for candidate in sorted(
            (candidate for candidate in candidates if candidate is not plan),
            key=lambda candidate: (
                0 if candidate.admission.admitted else 1,
                _objective(candidate),
                candidate.execution_fingerprint,
            ),
        )[:12]
    )
    binding = tuple(
        dict.fromkeys(
            (*plan.admission.binding_reasons, *hard_constraints, *frozen_fields)
        )
    )
    report = replace(
        plan.admission,
        candidate_count=len(candidates),
        admissible_candidate_count=len(admitted),
        hard_constraints=hard_constraints,
        frozen_fields=frozen_fields,
        free_fields=free_fields,
        binding_constraints=binding,
        selected_objective=(
            ("predicted_walltime_high_seconds", objective[0]),
            ("predicted_peak_vram_bytes", objective[1]),
            ("predicted_host_bytes", objective[2]),
            ("predicted_row_store_bytes", objective[3]),
        ),
        rejected_candidates=rejected,
    )
    execution_hash = execution_fingerprint(
        profile=plan.profile,
        envelope=plan.envelope,
        physical=plan.physical,
        admission=report,
        evidence_fingerprint=plan.evidence_fingerprint,
    )
    return replace(plan, admission=report, execution_fingerprint=execution_hash)


def resolve_trace_plan(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    requirements: PhysicalExecutionRequirements | None = None,
    *,
    frozen_fields: tuple[str, ...] = (),
    reported_hard_constraints: PhysicalExecutionRequirements | None = None,
) -> TracePlan:
    """Select the fastest fitting physical plan under hard caller constraints."""

    requirements = requirements or PhysicalExecutionRequirements()
    reported_hard_constraints = reported_hard_constraints or requirements
    effective_source, _, _ = _effective_batches(semantics)
    required_batch_bounds = (
        (
            "source_microbatch_size",
            requirements.source_microbatch_size,
            min(effective_source, profile.max_source_microbatch_size),
        ),
        (
            "feature_microbatch_size",
            requirements.feature_microbatch_size,
            min(semantics.feature_batch_size, profile.max_phase4_microbatch_size),
        ),
        (
            "logit_microbatch_size",
            requirements.logit_microbatch_size,
            min(semantics.logit_batch_size, profile.max_phase3_microbatch_size),
        ),
    )
    for name, value, maximum in required_batch_bounds:
        if value is not None and value > maximum:
            raise ValueError(f"{name} exceeds its logical or provider maximum")
    required_microbatch_capacity = max(
        requirements.source_microbatch_size or 0,
        requirements.feature_microbatch_size or 0,
        requirements.logit_microbatch_size or 0,
    )
    if (
        requirements.session_capacity is not None
        and required_microbatch_capacity > requirements.session_capacity
    ):
        raise ValueError("required microbatch cannot exceed required session_capacity")
    phase1_source_batch_size = (
        min(effective_source, profile.max_phase1_source_batch_size)
        if requirements.phase1_source_batch_size is None
        else requirements.phase1_source_batch_size
    )
    policies = (
        (requirements.row_store_policy,)
        if requirements.row_store_policy is not None
        else tuple(
            policy
            for policy in RowStorePolicy
            if _row_store_supported(policy.value, profile)
            and not (
                policy is RowStorePolicy.RECOMPUTE
                and requirements.spill_target is not None
            )
        )
    )
    phase3_values = _batch_breakpoint_sizes(
        logical_size=semantics.logit_batch_size,
        maximum=min(
            semantics.logit_batch_size,
            profile.max_phase3_microbatch_size,
            profile.max_session_capacity,
            requirements.session_capacity or profile.max_session_capacity,
        ),
        required=requirements.logit_microbatch_size,
    )
    phase4_values = _batch_breakpoint_sizes(
        logical_size=semantics.feature_batch_size,
        maximum=min(
            semantics.feature_batch_size,
            profile.max_phase4_microbatch_size,
            profile.max_session_capacity,
            requirements.session_capacity or profile.max_session_capacity,
        ),
        required=requirements.feature_microbatch_size,
    )
    candidates: list[TracePlan] = []
    for phase3, phase4, policy in product(phase3_values, phase4_values, policies):
        minimum_session = max(
            phase3,
            phase4,
            requirements.source_microbatch_size or 1,
        )
        session_capacity = requirements.session_capacity or minimum_session
        if session_capacity < minimum_session:
            continue
        source = requirements.source_microbatch_size or min(
            effective_source,
            session_capacity,
            profile.max_source_microbatch_size,
        )
        candidate_requirements = replace(
            requirements,
            session_capacity=session_capacity,
            phase1_source_batch_size=phase1_source_batch_size,
            source_microbatch_size=source,
            logit_microbatch_size=phase3,
            feature_microbatch_size=phase4,
            row_store_policy=policy,
        )
        candidates.append(
            _resolve_single_trace_plan(
                semantics, profile, envelope, candidate_requirements
            )
        )
    if not candidates:
        raise ValueError("optimizer produced no candidates")
    admitted = tuple(candidate for candidate in candidates if candidate.admission.admitted)
    selected = min(
        admitted or tuple(candidates),
        key=lambda candidate: (
            0 if candidate.admission.admitted else len(candidate.admission.refusals),
            _objective(candidate),
            candidate.execution_fingerprint,
        ),
    )
    return _with_optimization_report(
        selected,
        requirements=requirements,
        reported_hard_constraints=reported_hard_constraints,
        frozen_fields=frozen_fields,
        candidates=tuple(candidates),
    )
