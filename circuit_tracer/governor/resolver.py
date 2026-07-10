from __future__ import annotations

import math
from dataclasses import replace

from .contracts import (
    AdmissionReport,
    CachePolicy,
    DemandClass,
    DemandEstimate,
    DemandLifetime,
    DemandTier,
    FidelityMode,
    PhysicalExecutionConfig,
    PlanStatus,
    ProviderDimensions,
    ProviderCostMetadata,
    ProviderProfile,
    ResourceEnvelope,
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
    microbatch_steps = (
        math.ceil(effective_source_batch_size / source_microbatch_size)
        + math.ceil(feature_batch_size / feature_microbatch_size)
        + math.ceil(logit_batch_size / logit_microbatch_size)
    )
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


def _int_override(name: str, value: int | str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"physical override {name} must be an integer")
    return value


def _string_override(name: str, value: int | str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"physical override {name} must be a string")
    return value


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
) -> tuple[PhysicalExecutionConfig, tuple[str, ...], tuple[str, ...]]:
    capabilities = profile.capabilities
    overrides = dict(envelope.physical_overrides)
    known = {
        "decoder_fetch_chunk_size",
        "decoder_cache_bytes",
        "source_microbatch_size",
        "feature_microbatch_size",
        "logit_microbatch_size",
        "replay_window",
        "prefetch_depth",
        "encoder_residency",
        "row_store_policy",
        "spill_target",
    }
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError("unknown physical override(s): " + ", ".join(unknown))

    fetch = _int_override(
        "decoder_fetch_chunk_size",
        overrides.get("decoder_fetch_chunk_size", profile.default_fetch_chunk_size),
    )
    if not 0 < fetch <= profile.max_fetch_chunk_size:
        raise ValueError("decoder_fetch_chunk_size must be positive and within profile maximum")
    cache = _int_override(
        "decoder_cache_bytes",
        overrides.get("decoder_cache_bytes", profile.default_decoder_cache_bytes),
    )
    if cache < 0 or cache > profile.max_decoder_cache_bytes:
        raise ValueError("decoder_cache_bytes must be nonnegative and within profile maximum")
    if cache and not capabilities.supports_decoder_chunk_cache:
        raise ValueError("provider does not support decoder_cache_bytes > 0")

    batch_specs = (
        ("source_microbatch_size", effective_source),
        ("feature_microbatch_size", semantics.feature_batch_size),
        ("logit_microbatch_size", semantics.logit_batch_size),
    )
    microbatches: dict[str, int] = {}
    for name, logical_bound in batch_specs:
        value = _int_override(
            name,
            overrides.get(name, min(logical_bound, profile.max_physical_microbatch)),
        )
        if not 0 < value <= logical_bound:
            raise ValueError(f"{name} must be positive and <= its logical/effective capacity")
        microbatches[name] = value

    replay = _int_override(
        "replay_window", overrides.get("replay_window", profile.default_replay_window)
    )
    if not 0 < replay <= profile.max_replay_window:
        raise ValueError("replay_window must be positive and within profile maximum")
    if replay > 1 and not capabilities.supports_replay:
        raise ValueError("provider does not support replay_window > 1")
    prefetch = _int_override(
        "prefetch_depth", overrides.get("prefetch_depth", profile.default_prefetch_depth)
    )
    if not 0 <= prefetch <= profile.max_prefetch_depth:
        raise ValueError("prefetch_depth must be nonnegative and within profile maximum")
    if prefetch and not capabilities.supports_prefetch:
        raise ValueError("provider does not support prefetch_depth > 0")

    warnings: list[str] = []
    refusals: list[str] = []
    if "encoder_residency" in overrides:
        residency = _string_override("encoder_residency", overrides["encoder_residency"])
        if residency not in {"eager", "lazy_per_request"}:
            raise ValueError(
                "encoder_residency override must be eager or lazy_per_request"
            )
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
        if capabilities.supports_encoder_row_materialization and eager_fits:
            residency = "eager"
        elif capabilities.supports_lazy_encoder_rows:
            residency = "lazy_per_request"
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
        _string_override("row_store_policy", overrides["row_store_policy"])
        if "row_store_policy" in overrides
        else None
    )
    spill_override = (
        _string_override("spill_target", overrides["spill_target"])
        if "spill_target" in overrides
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
        capacity
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
    source_microbatch_vram = math.ceil(
        physical.source_microbatch_size
        * semantics.prompt_token_count
        * dims.d_model
        * dtype_bytes
        * costs.source_microbatch_vram_coefficient
    )
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
        source_microbatch_size=physical.source_microbatch_size,
        feature_microbatch_size=physical.feature_microbatch_size,
        logit_microbatch_size=physical.logit_microbatch_size,
        replay_window=physical.replay_window,
        prefetch_depth=physical.prefetch_depth,
    )
    walltime_scale = work_units / costs.walltime_reference_work_units
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


def _tier_total(estimates: tuple[DemandEstimate, ...], tier: DemandTier) -> int:
    return int(sum(estimate.amount for estimate in estimates if estimate.tier is tier))


def _host_reservation_total(
    estimates: tuple[DemandEstimate, ...], profile: ProviderProfile
) -> int:
    incremental_components = (
        "active_host",
        "prompt_host",
        "encoder_residency_host",
        "replay_host",
    )
    incremental_total = int(
        sum(_amount(estimates, name) for name in incremental_components)
    )
    baseline = profile.costs.baseline_total_host_bytes
    if baseline is None:
        return int(_amount(estimates, "known_rigid_host")) + incremental_total
    return baseline + incremental_total


def _amount(estimates: tuple[DemandEstimate, ...], name: str) -> float:
    return next(estimate.amount for estimate in estimates if estimate.name == name)


def resolve_trace_plan(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
) -> TracePlan:
    """Resolve a pure, provider-agnostic, advisory Phase B execution plan."""

    if not isinstance(profile.costs, ProviderCostMetadata) or not profile.costs.cost_model_version:
        raise ValueError("provider profile has incomplete cost metadata")
    evidence_hash = _validate_evidence(semantics, profile)
    effective_source, capacity, bindings = _effective_batches(semantics)
    physical, physical_warnings, physical_refusals = _resolve_physical(
        semantics, profile, envelope, effective_source
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

    vram_total = _tier_total(estimates, DemandTier.VRAM)
    host_total = _host_reservation_total(estimates, profile)
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
    cache_policy = _cache_policy(checkpoint_bytes, effective_file_allowance, envelope.cache_policy)
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
