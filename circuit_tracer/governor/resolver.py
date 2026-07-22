from __future__ import annotations

import math
from dataclasses import replace

from .calibration import (
    CalibrationCatalog,
    FidelityPrediction,
    ParetoAlternative,
    PredictionSupport,
    PredictionSupportKind,
    PredictionUncertainty,
)

from .contracts import (
    AdmissionReport,
    CachePolicy,
    DemandClass,
    DemandEstimate,
    DemandLifetime,
    DemandTier,
    EncoderResidency,
    FidelityMode,
    PhysicalExecutionConfig,
    PhysicalExecutionRequirements,
    PlanningProgress,
    PlanStatus,
    ProviderDimensions,
    ProviderCostMetadata,
    ProviderProfile,
    ResourceEnvelope,
    RowStorePolicy,
    StorageTier,
    TracePlan,
    TraceSemantics,
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
    frontier_refresh_stride: int = 1,
) -> float:
    fetch_blocks = math.ceil(dimensions.d_features / decoder_fetch_chunk_size)
    # Count the independently scheduled Phase 1 source batches alongside the
    # Phase 3/4 partitions. The calibration reference uses one Phase 1 step, so
    # this keeps recorded plans anchored while pricing stricter source limits.
    phase4_execution_calls_per_refresh = math.ceil(
        frontier_refresh_stride * feature_batch_size / feature_microbatch_size
    )
    microbatch_steps = math.ceil(
        effective_source_batch_size / source_microbatch_size
    ) + phase4_execution_calls_per_refresh + math.ceil(
        logit_batch_size / logit_microbatch_size
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
    del semantics, profile
    return None


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

    semantic_capacity = max(
        effective_source, semantics.feature_batch_size, semantics.logit_batch_size
    )
    phase4_execution_bound = min(
        semantics.feature_batch_size * semantics.frontier_refresh_stride,
        profile.max_phase4_microbatch_size,
    )
    session_capacity_bound = max(semantic_capacity, phase4_execution_bound)
    required_execution_capacity = max(
        requirements.source_microbatch_size or 0,
        requirements.feature_microbatch_size or 0,
        requirements.logit_microbatch_size or 0,
    )
    session_capacity = (
        min(
            max(semantic_capacity, required_execution_capacity),
            profile.max_session_capacity,
        )
        if requirements.session_capacity is None
        else requirements.session_capacity
    )
    if not 0 < session_capacity <= session_capacity_bound:
        raise ValueError(
            "session_capacity must be positive and no larger than the largest "
            "executable semantic or Phase-4 refresh window"
        )
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
            effective_source,
            profile.max_source_microbatch_size,
            requirements.source_microbatch_size,
        ),
        (
            "feature_microbatch_size",
            phase4_execution_bound,
            semantics.feature_batch_size,
            profile.max_phase4_microbatch_size,
            requirements.feature_microbatch_size,
        ),
        (
            "logit_microbatch_size",
            semantics.logit_batch_size,
            semantics.logit_batch_size,
            profile.max_phase3_microbatch_size,
            requirements.logit_microbatch_size,
        ),
    )
    microbatches: dict[str, int] = {}
    for name, logical_bound, default_bound, profile_bound, required in batch_specs:
        value = min(default_bound, profile_bound, session_capacity) if required is None else required
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
    if row_policy == "recompute":
        replay_tile_cache_bytes = (
            profile.reference_replay_tile_cache_bytes
            if requirements.replay_tile_cache_bytes is None
            else requirements.replay_tile_cache_bytes
        )
    else:
        replay_tile_cache_bytes = 0
        if requirements.replay_tile_cache_bytes not in (None, 0):
            raise ValueError("replay_tile_cache_bytes requires recompute row storage")

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
        replay_tile_cache_bytes=replay_tile_cache_bytes,
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
    memory_models = {model.phase: model for model in profile.phase_memory_models}
    session_model = memory_models.get("session")
    if session_model is None:
        trace_vram = math.ceil(trace_elements * costs.trace_vram_coefficient)
    else:
        trace_vram = math.ceil(
            physical.session_capacity
            * session_model.session_vram_bytes_per_item
            * semantics.prompt_token_count
            / max(1, (profile.calibration_support.prompt_token_count if profile.calibration_support else semantics.prompt_token_count))
        )
    target_vram = math.ceil(target_elements * costs.target_vram_coefficient)
    phase0_model = memory_models.get("phase0")
    fetch_vram = (
        phase0_model.fixed_vram_bytes
        if phase0_model is not None and phase0_model.fixed_vram_bytes
        else (
            physical.decoder_fetch_chunk_size
            * dims.d_model
            * dims.decoder_output_span
            * dtype_bytes
        )
    )
    prefetch_vram = fetch_vram * physical.prefetch_depth
    # No runtime path currently executes source_microbatch_size as a physical
    # partition. Keep the named estimate for schema stability, but do not charge
    # fictitious residency until a sequenced source executor consumes it.
    source_microbatch_vram = 0
    if profile.phase_memory_models:
        feature_microbatch_vram = (
            physical.feature_microbatch_size
            * semantics.prompt_token_count
            * min(semantics.estimated_nnz, dims.d_features)
            * dtype_bytes
        )
        logit_microbatch_vram = (
            physical.logit_microbatch_size
            * semantics.prompt_token_count
            * semantics.target_count
            * dims.d_model
            * dtype_bytes
        )
    else:
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
    phase4_model = memory_models.get("phase4")
    encoder_row_bytes = (
        phase4_model.host_bytes_per_item
        if phase4_model is not None and phase4_model.host_bytes_per_item
        else dims.d_model * dtype_bytes
    )
    if costs.baseline_total_host_bytes is not None:
        encoder_host = (
            semantics.estimated_nnz * encoder_row_bytes
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
            semantics.estimated_nnz * encoder_row_bytes
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
        source_microbatch_size=min(
            physical.session_capacity, physical.phase1_source_batch_size
        ),
        feature_microbatch_size=physical.feature_microbatch_size,
        logit_microbatch_size=physical.logit_microbatch_size,
        replay_window=physical.replay_window,
        prefetch_depth=physical.prefetch_depth,
        frontier_refresh_stride=semantics.frontier_refresh_stride,
    )
    phase_walltimes: list[tuple[str, float, float]] = []
    if profile.phase_walltime_models and profile.calibration_support is not None:
        support = profile.calibration_support
        logical_scale = (
            semantics.prompt_token_count / support.prompt_token_count
            * semantics.estimated_nnz / support.active_features
            * semantics.target_count / support.target_count
        )
        session_steps = math.ceil(capacity / physical.session_capacity)
        phase1_steps = math.ceil(
            effective_source
            / min(physical.session_capacity, physical.phase1_source_batch_size)
        )
        phase3_steps = math.ceil(
            semantics.logit_batch_size / physical.logit_microbatch_size
        )
        phase4_steps = math.ceil(
            semantics.frontier_refresh_stride
            * semantics.feature_batch_size
            / physical.feature_microbatch_size
        )
        reference_session_steps = math.ceil(
            support.logical_batch_size / support.session_capacity
        )
        reference_phase1_steps = math.ceil(
            support.logical_batch_size
            / min(support.session_capacity, support.phase1_source_batch_size)
        )
        reference_phase3_steps = math.ceil(
            support.logical_batch_size / support.phase3_microbatch_size
        )
        reference_phase4_steps = math.ceil(
            semantics.frontier_refresh_stride
            * support.logical_batch_size
            / support.phase4_microbatch_size
        )
        for model in profile.phase_walltime_models:
            scale = logical_scale
            if model.scales_with_session_steps:
                scale *= session_steps / reference_session_steps
            if model.scales_with_phase1_steps:
                scale *= phase1_steps / reference_phase1_steps
            if model.scales_with_phase3_steps:
                scale *= phase3_steps / reference_phase3_steps
            if model.scales_with_phase4_steps:
                scale *= phase4_steps / reference_phase4_steps
            if model.affected_by_fetch:
                if support.decoder_cache_bytes:
                    cache_ratio = min(
                        1.0,
                        physical.decoder_cache_bytes / support.decoder_cache_bytes,
                    )
                    # The reference point proves the selected cache fits, but
                    # not an unbounded inverse-byte speed curve. Keep the
                    # provisional miss penalty bounded until the causal sweep.
                    scale *= 1.25 - 0.25 * cache_ratio
            if model.affected_by_replay:
                scale *= 1 + 0.02 * (physical.replay_window - support.replay_window)
            if model.affected_by_prefetch:
                scale *= (1 + 0.05 * support.prefetch_depth) / (
                    1 + 0.05 * physical.prefetch_depth
                )
            if model.affected_by_row_policy:
                scale *= costs.row_store_walltime_multiplier(
                    physical.row_store_policy
                )
                if physical.row_store_policy == "recompute":
                    reference_cache = max(1, profile.reference_replay_tile_cache_bytes)
                    cache_ratio = min(
                        1.0, physical.replay_tile_cache_bytes / reference_cache
                    )
                    scale *= 1.25 - 0.25 * cache_ratio
            phase_walltimes.append(
                (
                    model.phase,
                    model.reference_low_seconds * scale,
                    model.reference_high_seconds * scale,
                )
            )
        walltime_low = sum(item[1] for item in phase_walltimes)
        walltime_high = sum(item[2] for item in phase_walltimes)
    else:
        walltime_scale = (
            work_units
            / costs.walltime_reference_work_units
            * math.ceil(capacity / physical.session_capacity)
            * costs.row_store_walltime_multiplier(physical.row_store_policy)
        )
        walltime_low = costs.calibrated_walltime_low_seconds * walltime_scale
        walltime_high = costs.calibrated_walltime_high_seconds * walltime_scale
    disk_tier = (
        DemandTier.LOCAL_DISK
        if physical.spill_target == "local"
        else DemandTier.SCRATCH_DISK if physical.spill_target == "scratch" else DemandTier.LOCAL_DISK
    )
    estimates = (
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
        DemandEstimate("replay_tile_cache_host", DemandTier.HOST, DemandClass.RIGID, DemandLifetime.PHASE, physical.replay_tile_cache_bytes),
        DemandEstimate("checkpoint_file_working_set", DemandTier.FILE_BACKED, DemandClass.ELASTIC, DemandLifetime.PERMANENT, costs.checkpoint_file_bytes),
        DemandEstimate("row_store_disk", disk_tier, DemandClass.RIGID, DemandLifetime.PHASE, physical.row_store_bytes),
        DemandEstimate("predicted_walltime_low", DemandTier.WALLTIME, DemandClass.RIGID, DemandLifetime.PHASE, walltime_low, "seconds"),
        DemandEstimate("predicted_walltime_high", DemandTier.WALLTIME, DemandClass.RIGID, DemandLifetime.PHASE, walltime_high, "seconds"),
    )
    return estimates + tuple(
        DemandEstimate(
            f"predicted_walltime_{phase}_{bound}", DemandTier.WALLTIME,
            DemandClass.RIGID, DemandLifetime.PHASE, amount, "seconds"
        )
        for phase, low, high in phase_walltimes
        for bound, amount in (("low", low), ("high", high))
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
        _sum_estimates(estimates, {"replay_tile_cache_host", "encoder_residency_host"}),
    )
    incremental_peak = max(phase_working_sets)
    baseline = profile.costs.baseline_total_host_bytes
    if baseline is None:
        return int(_amount(estimates, "known_rigid_host")) + incremental_peak
    return baseline + incremental_peak


def _amount(estimates: tuple[DemandEstimate, ...], name: str) -> float:
    return next(estimate.amount for estimate in estimates if estimate.name == name)


def _support_metadata(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    physical: PhysicalExecutionConfig,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return confidence, extrapolated dimensions, evidence, and invalid reasons."""

    safety = profile.safety_limits
    support = profile.calibration_support
    invalid: list[str] = []
    if safety is not None:
        safety_values = {
            "prompt_token_count": (semantics.prompt_token_count, safety.max_prompt_token_count),
            "estimated_active_features": (semantics.estimated_nnz, safety.max_active_features),
            "max_feature_nodes": (semantics.max_feature_nodes, safety.max_feature_nodes),
            "target_count": (semantics.target_count, safety.max_target_count),
            "source_batch_size": (semantics.source_batch_size, safety.max_logical_batch_size),
            "feature_batch_size": (semantics.feature_batch_size, safety.max_logical_batch_size),
            "logit_batch_size": (semantics.logit_batch_size, safety.max_logical_batch_size),
            "session_capacity": (physical.session_capacity, safety.max_physical_rows),
            "phase1_source_batch_size": (physical.phase1_source_batch_size, safety.max_physical_rows),
            "source_microbatch_size": (physical.source_microbatch_size, safety.max_physical_rows),
            "feature_microbatch_size": (physical.feature_microbatch_size, safety.max_physical_rows),
            "logit_microbatch_size": (physical.logit_microbatch_size, safety.max_physical_rows),
            "decoder_cache_bytes": (
                physical.decoder_cache_bytes, safety.max_decoder_cache_bytes
            ),
            "replay_window": (physical.replay_window, safety.max_replay_window),
            "prefetch_depth": (physical.prefetch_depth, safety.max_prefetch_depth),
            "replay_tile_cache_bytes": (
                physical.replay_tile_cache_bytes,
                safety.max_replay_tile_cache_bytes,
            ),
        }
        invalid.extend(
            f"{name}={value} exceeds provider safety limit {limit}"
            for name, (value, limit) in safety_values.items()
            if value > limit
        )
    if support is None:
        return "unknown", (), (), tuple(invalid)
    support_values = {
        "prompt_token_count": (semantics.prompt_token_count, support.prompt_token_count),
        "estimated_active_features": (semantics.estimated_nnz, support.active_features),
        "max_feature_nodes": (semantics.max_feature_nodes, support.max_feature_nodes),
        "target_count": (semantics.target_count, support.target_count),
        "logical_batch_size": (
            max(semantics.source_batch_size, semantics.feature_batch_size, semantics.logit_batch_size),
            support.logical_batch_size,
        ),
        "session_capacity": (physical.session_capacity, support.session_capacity),
        "phase1_source_batch_size": (
            physical.phase1_source_batch_size, support.phase1_source_batch_size
        ),
        "phase3_microbatch_size": (
            physical.logit_microbatch_size, support.phase3_microbatch_size
        ),
        "phase4_microbatch_size": (
            physical.feature_microbatch_size, support.phase4_microbatch_size
        ),
        "decoder_cache_bytes": (
            physical.decoder_cache_bytes, support.decoder_cache_bytes
        ),
        "replay_window": (physical.replay_window, support.replay_window),
        "prefetch_depth": (physical.prefetch_depth, support.prefetch_depth),
        "replay_tile_cache_bytes": (
            physical.replay_tile_cache_bytes, support.replay_tile_cache_bytes
        ),
    }
    extrapolated = [
        name for name, (value, bound) in support_values.items() if value > bound
    ]
    if physical.row_store_policy not in support.row_store_policies:
        extrapolated.append("row_store_policy")
    if support.provisional_dimensions:
        extrapolated.append("provider_dimensions")
    extrapolated = list(dict.fromkeys(extrapolated))
    confidence = (
        "extrapolated" if extrapolated else "provisional" if support.provisional_dimensions else "calibrated"
    )
    return confidence, tuple(extrapolated), support.evidence, tuple(invalid)


def _phase_predictions(
    estimates: tuple[DemandEstimate, ...]
) -> tuple[tuple[str, float, float], ...]:
    values = {estimate.name: float(estimate.amount) for estimate in estimates}
    phases = sorted(
        {
            name.removeprefix("predicted_walltime_").removesuffix("_low")
            for name in values
            if name.startswith("predicted_walltime_")
            and name.endswith("_low")
            and name != "predicted_walltime_low"
        }
    )
    return tuple(
        (
            phase,
            values[f"predicted_walltime_{phase}_low"],
            values[f"predicted_walltime_{phase}_high"],
        )
        for phase in phases
    )


def _apply_planning_progress(
    estimates: tuple[DemandEstimate, ...],
    progress: PlanningProgress,
) -> tuple[DemandEstimate, ...]:
    if not progress.completed_phases and progress.observed_elapsed_seconds == 0:
        return estimates
    predictions = _phase_predictions(estimates)
    remaining = tuple(
        prediction
        for prediction in predictions
        if prediction[0] not in progress.completed_phases
    )
    if predictions:
        projected_low = progress.observed_elapsed_seconds + sum(
            prediction[1] for prediction in remaining
        )
        projected_high = progress.observed_elapsed_seconds + sum(
            prediction[2] for prediction in remaining
        )
    else:
        # Profiles without additive phase models cannot subtract completed
        # work safely. Retain their full estimate and add observed elapsed.
        projected_low = progress.observed_elapsed_seconds + _amount(
            estimates, "predicted_walltime_low"
        )
        projected_high = progress.observed_elapsed_seconds + _amount(
            estimates, "predicted_walltime_high"
        )
    return tuple(
        replace(
            estimate,
            amount=(
                projected_low
                if estimate.name == "predicted_walltime_low"
                else projected_high
            ),
        )
        if estimate.name in {"predicted_walltime_low", "predicted_walltime_high"}
        else estimate
        for estimate in estimates
    )


def _fidelity_coordinates(
    physical: PhysicalExecutionConfig,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, str(getattr(physical, name)))
            for name in (
                "decoder_fetch_chunk_size",
                "decoder_cache_bytes",
                "session_capacity",
                "phase1_source_batch_size",
                "source_microbatch_size",
                "feature_microbatch_size",
                "logit_microbatch_size",
                "replay_window",
                "prefetch_depth",
                "replay_tile_cache_bytes",
                "encoder_residency",
                "row_store_policy",
            )
        )
    )


def _profile_exact_prediction(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    physical: PhysicalExecutionConfig,
) -> FidelityPrediction:
    """Conservative built-in exact support for the profile's reference mechanism."""
    support = profile.calibration_support
    effective_source, _, _ = _effective_batches(semantics)
    reference_session = (
        support.session_capacity if support is not None else profile.max_session_capacity
    )
    reference_phase1 = (
        support.phase1_source_batch_size
        if support is not None
        else profile.max_phase1_source_batch_size
    )
    # Exact mechanisms remain optimizer variables. Only execution shapes known
    # to change floating-point grouping require scope-matched certification.
    expected_sensitive = {
        "decoder_fetch_chunk_size": profile.default_fetch_chunk_size,
        "session_capacity": min(
            max(
                effective_source,
                semantics.feature_batch_size,
                semantics.logit_batch_size,
                min(
                    semantics.frontier_refresh_stride * semantics.feature_batch_size,
                    profile.max_phase4_microbatch_size,
                    profile.max_session_capacity,
                ),
            ),
            reference_session,
        ),
        "phase1_source_batch_size": min(effective_source, reference_phase1),
        "source_microbatch_size": min(
            effective_source,
            reference_session,
            reference_phase1,
            profile.max_source_microbatch_size,
        ),
        "feature_microbatch_size": min(
            semantics.frontier_refresh_stride * semantics.feature_batch_size,
            support.phase4_microbatch_size if support else profile.max_phase4_microbatch_size,
            reference_session,
        ),
        "logit_microbatch_size": min(
            semantics.logit_batch_size,
            support.phase3_microbatch_size if support else profile.max_phase3_microbatch_size,
            reference_session,
        ),
    }
    extrapolated = tuple(
        name
        for name, value in expected_sensitive.items()
        if getattr(physical, name) != value
    )
    exact = not extrapolated
    return FidelityPrediction(
        metrics=(),
        support=PredictionSupport(
            PredictionSupportKind.EXACT if exact else PredictionSupportKind.NONE,
            "provider-profile-reference" if exact else None,
            0.0 if exact else None,
            exact,
            extrapolated,
        ),
        uncertainty=PredictionUncertainty((), 0.0 if exact else None),
    )


def _assess_fidelity(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    physical: PhysicalExecutionConfig,
    catalog: CalibrationCatalog | None,
) -> tuple[FidelityPrediction, float, tuple[str, ...], tuple[str, ...]]:
    prediction = (
        catalog.predict(_fidelity_coordinates(physical))
        if catalog is not None
        else _profile_exact_prediction(semantics, profile, physical)
    )
    refusals: list[str] = []
    warnings: list[str] = []
    if semantics.fidelity is FidelityMode.EXACT:
        if not prediction.support.certified_exact:
            refusals.append(
                "exact fidelity requires certified exact evidence for sensitive axes"
            )
    elif semantics.fidelity is FidelityMode.BOUNDED:
        assert semantics.fidelity_budget is not None
        disallowed = tuple(
            axis
            for axis in prediction.support.extrapolated_axes
            if axis not in semantics.fidelity_budget.allowed_sensitive_axes
        )
        if disallowed:
            refusals.append(
                "bounded candidate changes sensitive axes without allowance: "
                + ", ".join(disallowed)
            )
        for metric, floor in semantics.fidelity_budget.metric_floors:
            lower = prediction.lower_bound(metric)
            if lower is None:
                refusals.append(f"bounded fidelity has no prediction for metric {metric}")
            elif lower < floor:
                refusals.append(
                    f"bounded fidelity metric {metric} lower bound {lower:.6g} is below floor {floor:.6g}"
                )
    elif semantics.fidelity is FidelityMode.BEST_EFFORT:
        assert semantics.fidelity_budget is not None
        disallowed = tuple(
            axis
            for axis in prediction.support.extrapolated_axes
            if axis not in semantics.fidelity_budget.allowed_sensitive_axes
        )
        if disallowed:
            refusals.append(
                "best_effort candidate changes sensitive axes without allowance: "
                + ", ".join(disallowed)
            )
    elif semantics.fidelity is FidelityMode.RESEARCH:
        if prediction.support.kind in {
            PredictionSupportKind.EXTRAPOLATED,
            PredictionSupportKind.NONE,
        }:
            warnings.append(
                "research fidelity uses extrapolated or unsupported calibration evidence"
            )

    if semantics.fidelity_budget is not None:
        penalty = semantics.fidelity_budget.penalty_weight * sum(
            max(0.0, floor - (prediction.lower_bound(metric) or -1.0))
            for metric, floor in semantics.fidelity_budget.metric_floors
        )
    elif prediction.metrics:
        penalty = sum(max(0.0, 1.0 - metric.lower) for metric in prediction.metrics)
    else:
        penalty = 0.0 if prediction.support.certified_exact else 1.0
    penalty += 0.0 if prediction.support.normalized_distance is not None else 1.0
    return prediction, penalty, tuple(warnings), tuple(refusals)


def _resolve_single_trace_plan(
    semantics: TraceSemantics,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    requirements: PhysicalExecutionRequirements | None = None,
    progress: PlanningProgress = PlanningProgress(),
    catalog: CalibrationCatalog | None = None,
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
    estimates = _apply_planning_progress(estimates, progress)

    warnings = list(physical_warnings)
    refusals = list(physical_refusals)
    fidelity_prediction, fidelity_penalty, fidelity_warnings, fidelity_refusals = (
        _assess_fidelity(semantics, profile, physical, catalog)
    )
    warnings.extend(fidelity_warnings)
    refusals.extend(fidelity_refusals)
    confidence, extrapolated, calibration_evidence, invalid = _support_metadata(
        semantics, profile, physical
    )
    refusals.extend(invalid)
    if extrapolated:
        warnings.append(
            "valid provider-safety extrapolation outside calibration support: "
            + ", ".join(extrapolated)
        )
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
            f"replay_tile_cache={physical.replay_tile_cache_bytes}",
            f"cache_policy={physical.cache_policy.value}",
            f"encoder_residency={physical.encoder_residency}",
            f"row_store={physical.row_store_policy}:{physical.row_store_bytes}",
            f"spill_target={physical.spill_target or 'none'}",
        )
    )
    phase_predictions = _phase_predictions(estimates)
    report = AdmissionReport(
        admitted=not refusals,
        estimates=estimates,
        trace_capacity=capacity,
        binding_reasons=bindings,
        effective_file_cache_allowance_bytes=effective_file_allowance,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
        refusals=tuple(refusals),
        confidence=confidence,
        extrapolated_dimensions=extrapolated,
        calibration_evidence=calibration_evidence,
        phase_predictions=phase_predictions,
        remaining_projection=tuple(
            prediction
            for prediction in phase_predictions
            if prediction[0] not in progress.completed_phases
        ),
        fidelity_prediction=fidelity_prediction,
        fidelity_penalty=fidelity_penalty,
        calibration_catalog_fingerprint=(
            catalog.content_fingerprint if catalog is not None else None
        ),
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
    "phase1_source_batch_size",
    "feature_microbatch_size",
    "logit_microbatch_size",
    "decoder_cache_bytes",
    "replay_window",
    "prefetch_depth",
    "replay_tile_cache_bytes",
    "row_store_policy",
    "encoder_residency",
    "spill_target",
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
    ordered = tuple(sorted(values, reverse=True))
    if len(ordered) <= 16:
        return ordered
    # Preserve the one-through-eight-step rungs exactly; those are where a
    # roomy accelerator normally lands. Use the remaining slots to retain a
    # geometric spread toward the minimum instead of replacing 500 with 100
    # merely because the full divisor-breakpoint set is dense near zero.
    indexes = set(range(8))
    tail_start = 8
    indexes.update(
        round(tail_start + index * (len(ordered) - 1 - tail_start) / 7)
        for index in range(8)
    )
    return tuple(ordered[index] for index in sorted(indexes))


def _domain_values(required, *values):
    if required is not None:
        return (required,)
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _display_domain(values: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str(value.value if hasattr(value, "value") else value) for value in values)


def _state_numeric(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, RowStorePolicy):
        return {RowStorePolicy.FULL: 3.0, RowStorePolicy.TILED: 2.0, RowStorePolicy.RECOMPUTE: 1.0}[value]
    if isinstance(value, EncoderResidency):
        return 1.0 if value is EncoderResidency.LAZY_PER_REQUEST else 0.0
    return 0.0


def _state_speed_key(state: tuple[object, ...]) -> tuple[float, ...]:
    # Larger batches/caches/prefetch and less-degraded row policies form the
    # fast frontier. Replay window is workload semantics, not a speed knob.
    return tuple(-_state_numeric(value) for value in state)


def _state_pressure_key(state: tuple[object, ...]) -> tuple[float, ...]:
    return tuple(_state_numeric(value) for value in state)


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


def _selection_key(plan: TracePlan, fidelity: FidelityMode) -> tuple[object, ...]:
    confidence = {"calibrated": 0, "provisional": 1, "extrapolated": 2, "unknown": 3}[
        plan.admission.confidence
    ]
    support = (confidence, len(plan.admission.extrapolated_dimensions))
    objective = _objective(plan)
    if fidelity is FidelityMode.BEST_EFFORT:
        risk_adjusted_walltime = objective[0] * (1.0 + plan.admission.fidelity_penalty)
        policy = (risk_adjusted_walltime, *support, *objective)
    elif fidelity is FidelityMode.RESEARCH:
        policy = (*objective, *support)
    else:
        policy = (*support, *objective)
    return (
        0 if plan.admission.admitted else len(plan.admission.refusals),
        *policy,
    )


def _pareto_alternatives(candidates: tuple[TracePlan, ...]) -> tuple[ParetoAlternative, ...]:
    admitted = tuple(candidate for candidate in candidates if candidate.admission.admitted)
    points = tuple(
        (
            candidate,
            (
                candidate.admission.fidelity_penalty,
                *_objective(candidate),
            ),
        )
        for candidate in admitted
    )
    frontier = []
    for candidate, point in points:
        dominated = any(
            all(other_value <= value for other_value, value in zip(other, point))
            and any(other_value < value for other_value, value in zip(other, point))
            for other_candidate, other in points
            if other_candidate is not candidate
        )
        if not dominated:
            frontier.append(
                ParetoAlternative(
                    candidate.execution_fingerprint,
                    point[0], point[1], point[2], point[3], point[4],
                )
            )
    return tuple(sorted(frontier, key=lambda item: (
        item.fidelity_penalty, item.walltime_high_seconds,
        item.peak_vram_bytes, item.execution_fingerprint,
    ))[:12])


def _with_optimization_report(
    plan: TracePlan,
    *,
    requirements: PhysicalExecutionRequirements,
    reported_hard_constraints: PhysicalExecutionRequirements,
    frozen_fields: tuple[str, ...],
    candidates: tuple[TracePlan, ...],
    domain_summary: tuple[tuple[str, tuple[str, ...]], ...],
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
        domain_summary=domain_summary,
        pareto_alternatives=_pareto_alternatives(candidates),
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
    progress: PlanningProgress | None = None,
    catalog: CalibrationCatalog | None = None,
) -> TracePlan:
    """Select the fastest fitting physical plan under hard caller constraints."""

    requirements = requirements or PhysicalExecutionRequirements()
    progress = progress or PlanningProgress()
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
            profile.max_phase4_microbatch_size,
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
    phase4_refresh_window_rows = (
        semantics.frontier_refresh_stride * semantics.feature_batch_size
    )
    logical_capacity = max(
        effective_source,
        semantics.feature_batch_size,
        semantics.logit_batch_size,
        requirements.feature_microbatch_size or 0,
        min(
            phase4_refresh_window_rows,
            profile.max_phase4_microbatch_size,
            profile.max_session_capacity,
        ),
    )
    row_limit = (
        profile.safety_limits.max_physical_rows
        if profile.safety_limits is not None
        else profile.max_session_capacity
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
    session_values = _batch_breakpoint_sizes(
        logical_size=logical_capacity,
        maximum=min(logical_capacity, profile.max_session_capacity, row_limit),
        required=requirements.session_capacity,
    )
    phase1_values = _batch_breakpoint_sizes(
        logical_size=effective_source,
        maximum=min(effective_source, profile.max_phase1_source_batch_size, row_limit),
        required=requirements.phase1_source_batch_size,
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
        logical_size=phase4_refresh_window_rows,
        maximum=min(
            phase4_refresh_window_rows,
            profile.max_phase4_microbatch_size,
            profile.max_session_capacity,
            requirements.session_capacity or profile.max_session_capacity,
        ),
        required=requirements.feature_microbatch_size,
    )
    cache_values = _domain_values(
        requirements.decoder_cache_bytes,
        profile.default_decoder_cache_bytes,
        0,
        min(profile.max_decoder_cache_bytes, envelope.effective_vram_budget_bytes // 4),
    )
    replay_values = _domain_values(
        requirements.replay_window,
        profile.default_replay_window,
        1,
        profile.max_replay_window,
    )
    prefetch_values = _domain_values(
        requirements.prefetch_depth,
        profile.default_prefetch_depth,
        0,
        profile.max_prefetch_depth,
    )
    encoder_values = _domain_values(
        requirements.encoder_residency,
        profile.default_encoder_residency,
        EncoderResidency.LAZY_PER_REQUEST if profile.capabilities.supports_lazy_encoder_rows else None,
        EncoderResidency.EAGER if profile.capabilities.supports_encoder_row_materialization else None,
    )
    spill_values = (
        (requirements.spill_target,)
        if requirements.spill_target is not None
        else tuple(StorageTier(root) for root in envelope.spill_roots)
        + ((None,) if RowStorePolicy.RECOMPUTE in policies else ())
    )
    replay_cache_values = _domain_values(
        requirements.replay_tile_cache_bytes,
        profile.reference_replay_tile_cache_bytes,
        0,
        min(
            (
                profile.safety_limits.max_replay_tile_cache_bytes
                if profile.safety_limits is not None
                else profile.reference_replay_tile_cache_bytes
            ),
            envelope.host_budget_bytes // 8,
        ),
    )
    domain_summary = (
        ("session_capacity", _display_domain(session_values)),
        ("phase1_source_batch_size", _display_domain(phase1_values)),
        ("feature_microbatch_size", _display_domain(phase4_values)),
        ("logit_microbatch_size", _display_domain(phase3_values)),
        ("decoder_cache_bytes", _display_domain(cache_values)),
        ("replay_window", _display_domain(replay_values)),
        ("prefetch_depth", _display_domain(prefetch_values)),
        ("replay_tile_cache_bytes", _display_domain(replay_cache_values)),
        ("row_store_policy", _display_domain(policies)),
        ("encoder_residency", _display_domain(encoder_values)),
        ("spill_target", _display_domain(spill_values)),
    )

    # Build bounded independent domains, then deterministically prune on both
    # speed and pressure fronts before invoking the full admission arithmetic.
    states = [()]
    domains = (
        session_values, phase1_values, phase3_values, phase4_values,
        cache_values, replay_values, prefetch_values, policies,
        encoder_values, spill_values, replay_cache_values,
    )
    for domain in domains:
        expanded = list(dict.fromkeys((*state, value) for state in states for value in domain))
        if len(expanded) > 1024:
            speed = sorted(expanded, key=_state_speed_key)[:512]
            pressure = sorted(expanded, key=_state_pressure_key)[:512]
            expanded = list(dict.fromkeys((*speed, *pressure)))
        states = expanded

    candidates: list[TracePlan] = []
    seen: set[PhysicalExecutionRequirements] = set()

    def append_candidate(candidate_requirements: PhysicalExecutionRequirements) -> None:
        if candidate_requirements in seen:
            return
        seen.add(candidate_requirements)
        try:
            candidates.append(
                _resolve_single_trace_plan(
                    semantics,
                    profile,
                    envelope,
                    candidate_requirements,
                    progress,
                    catalog,
                )
            )
        except ValueError:
            pass

    # Always retain the legacy deterministic defaults as one frontier point;
    # this also fail-closes malformed explicit requirements with their precise
    # validation error instead of an opaque empty-domain error.
    candidates.append(
        _resolve_single_trace_plan(
            semantics, profile, envelope, requirements, progress, catalog
        )
    )
    seen.add(requirements)
    support = profile.calibration_support
    support_seed: PhysicalExecutionRequirements | None = None
    if support is not None:
        seed_policy = requirements.row_store_policy or RowStorePolicy.FULL
        seed_session = requirements.session_capacity or min(
            logical_capacity, support.session_capacity
        )
        seed_phase1 = requirements.phase1_source_batch_size or min(
            effective_source, support.phase1_source_batch_size
        )
        seed_source = requirements.source_microbatch_size or min(
            effective_source, seed_phase1, seed_session,
            profile.max_source_microbatch_size,
        )
        seed = replace(
            requirements,
            session_capacity=seed_session,
            phase1_source_batch_size=seed_phase1,
            source_microbatch_size=seed_source,
            feature_microbatch_size=(
                requirements.feature_microbatch_size
                or min(semantics.feature_batch_size, support.phase4_microbatch_size, seed_session)
            ),
            logit_microbatch_size=(
                requirements.logit_microbatch_size
                or min(semantics.logit_batch_size, support.phase3_microbatch_size, seed_session)
            ),
            decoder_cache_bytes=(
                requirements.decoder_cache_bytes
                if requirements.decoder_cache_bytes is not None
                else support.decoder_cache_bytes
            ),
            replay_window=requirements.replay_window or support.replay_window,
            prefetch_depth=(
                requirements.prefetch_depth
                if requirements.prefetch_depth is not None
                else support.prefetch_depth
            ),
            row_store_policy=seed_policy,
            encoder_residency=(
                requirements.encoder_residency or profile.default_encoder_residency
            ),
            spill_target=(
                None
                if seed_policy is RowStorePolicy.RECOMPUTE
                else requirements.spill_target
                or (StorageTier(envelope.spill_roots[0]) if envelope.spill_roots else None)
            ),
            replay_tile_cache_bytes=(
                requirements.replay_tile_cache_bytes
                if requirements.replay_tile_cache_bytes is not None
                else profile.reference_replay_tile_cache_bytes
                if seed_policy is RowStorePolicy.RECOMPUTE
                else 0
            ),
        )
        support_seed = seed
        append_candidate(seed)

    # The generic Cartesian beam preserves broad policy/cache coverage, while
    # these neighborhoods guarantee that physical batch rungs between the two
    # corners survive pruning. Each anchor varies one owning phase at a time,
    # plus the paired Phase-3/4 width needed when both phase peaks are binding.
    if support_seed is not None:
        for session_capacity in session_values:
            compatible_phase1 = tuple(
                value for value in phase1_values if value <= session_capacity
            )
            compatible_phase3 = tuple(
                value for value in phase3_values if value <= session_capacity
            )
            compatible_phase4 = tuple(
                value for value in phase4_values if value <= session_capacity
            )
            if not compatible_phase1 or not compatible_phase3 or not compatible_phase4:
                continue
            anchor = replace(
                support_seed,
                session_capacity=session_capacity,
                phase1_source_batch_size=compatible_phase1[0],
                source_microbatch_size=(
                    requirements.source_microbatch_size
                    or min(
                        effective_source,
                        compatible_phase1[0],
                        session_capacity,
                        profile.max_source_microbatch_size,
                    )
                ),
                logit_microbatch_size=compatible_phase3[0],
                feature_microbatch_size=compatible_phase4[0],
            )
            append_candidate(anchor)
            for value in compatible_phase1:
                append_candidate(
                    replace(
                        anchor,
                        phase1_source_batch_size=value,
                        source_microbatch_size=(
                            requirements.source_microbatch_size
                            or min(
                                effective_source,
                                value,
                                session_capacity,
                                profile.max_source_microbatch_size,
                            )
                        ),
                    )
                )
            for value in compatible_phase3:
                append_candidate(replace(anchor, logit_microbatch_size=value))
            for value in compatible_phase4:
                append_candidate(replace(anchor, feature_microbatch_size=value))
            for phase3, phase4 in zip(compatible_phase3, compatible_phase4):
                append_candidate(
                    replace(
                        anchor,
                        logit_microbatch_size=phase3,
                        feature_microbatch_size=phase4,
                    )
                )
            for value in cache_values:
                append_candidate(replace(anchor, decoder_cache_bytes=value))
    for (
        session_capacity, phase1, phase3, phase4, decoder_cache, replay,
        prefetch, policy, encoder, spill, replay_cache,
    ) in states:
        minimum_session = max(phase3, phase4, requirements.source_microbatch_size or 1)
        if session_capacity < minimum_session:
            continue
        if policy is RowStorePolicy.RECOMPUTE:
            if spill is not None:
                continue
            spill_value = None
            replay_cache_value = replay_cache
        else:
            if spill is None:
                continue
            spill_value = spill
            replay_cache_value = 0
            if requirements.replay_tile_cache_bytes not in (None, 0):
                continue
        source = requirements.source_microbatch_size or min(
            effective_source,
            phase1,
            session_capacity,
            profile.max_source_microbatch_size,
        )
        candidate_requirements = replace(
            requirements,
            session_capacity=session_capacity,
            phase1_source_batch_size=phase1,
            source_microbatch_size=source,
            logit_microbatch_size=phase3,
            feature_microbatch_size=phase4,
            decoder_cache_bytes=decoder_cache,
            replay_window=replay,
            prefetch_depth=prefetch,
            row_store_policy=policy,
            encoder_residency=encoder,
            spill_target=spill_value,
            replay_tile_cache_bytes=replay_cache_value,
        )
        append_candidate(candidate_requirements)
    if not candidates:
        raise ValueError("optimizer produced no candidates")
    candidates = list(
        {
            candidate.physical: candidate
            for candidate in candidates
        }.values()
    )
    admitted = tuple(candidate for candidate in candidates if candidate.admission.admitted)
    selected = min(
        admitted or tuple(candidates),
        key=lambda candidate: (
            _selection_key(candidate, semantics.fidelity),
            0
            if candidate.physical.encoder_residency
            == profile.default_encoder_residency.value
            else 1,
            (
                envelope.spill_roots.index(candidate.physical.spill_target)
                if candidate.physical.spill_target in envelope.spill_roots
                else len(envelope.spill_roots)
            ),
            candidate.execution_fingerprint,
        ),
    )
    return _with_optimization_report(
        selected,
        requirements=requirements,
        reported_hard_constraints=reported_hard_constraints,
        frozen_fields=frozen_fields,
        candidates=tuple(candidates),
        domain_summary=domain_summary,
    )
