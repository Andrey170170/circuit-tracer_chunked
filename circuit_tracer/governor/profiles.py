from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import (
    CalibrationSupport,
    DecoderTopology,
    PhaseMemoryModel,
    PhaseWalltimeModel,
    ProviderCapabilities,
    ProviderCostMetadata,
    ProviderDimensions,
    ProviderIdentity,
    ProviderProfile,
    ProviderSafetyLimits,
    ResourceEnvelope,
    TraceSemantics,
    immutable_mapping,
)
from .resolver import compute_work_units


GIB = 1024**3
KIB = 1024
PROFILE_VERSION = "granite-h200-v3"
PLANNER_VERSION = "governor-v0.3"


@dataclass(frozen=True)
class ResourceCalibrationObservation:
    profile_name: str
    architecture: str
    dimensions: ProviderDimensions
    requested_host_memory_bytes: int
    walltime_seconds_range: tuple[float, float]
    max_rss_bytes_range: tuple[int, int]
    batch_size: int
    fetch_chunk_size: int
    decoder_cache_bytes: int
    reference_prompt_token_count: int
    reference_active_features: int
    reference_max_feature_nodes: int
    reference_target_count: int
    validated_physical_batch_cap: int
    validated_tiled_row_store: bool = False
    validated_recompute_row_store: bool = False
    row_store_tile_column_bound: int | None = None
    evidence_class: str = "resource_calibration_only"

    def __post_init__(self) -> None:
        for name in (
            "requested_host_memory_bytes",
            "batch_size",
            "fetch_chunk_size",
            "reference_prompt_token_count",
            "reference_active_features",
            "reference_max_feature_nodes",
            "reference_target_count",
            "validated_physical_batch_cap",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.decoder_cache_bytes < 0:
            raise ValueError("decoder_cache_bytes must be nonnegative")
        if self.walltime_seconds_range[0] <= 0 or (
            self.walltime_seconds_range[1] < self.walltime_seconds_range[0]
        ):
            raise ValueError("walltime range must be positive and ordered")
        if self.max_rss_bytes_range[0] <= 0 or (
            self.max_rss_bytes_range[1] < self.max_rss_bytes_range[0]
        ):
            raise ValueError("MaxRSS range must be positive and ordered")
        if self.evidence_class != "resource_calibration_only":
            raise ValueError("calibration observations are not semantic evidence")
        if self.validated_tiled_row_store != (
            self.row_store_tile_column_bound is not None
        ):
            raise ValueError(
                "validated tiled row storage requires a tile-column bound"
            )
        if self.validated_recompute_row_store and not self.validated_tiled_row_store:
            raise ValueError("validated recompute requires validated tiled production")

    def reference_semantics(self) -> TraceSemantics:
        return TraceSemantics(
            prompt_token_count=self.reference_prompt_token_count,
            estimated_active_features=self.reference_active_features,
            max_feature_nodes=self.reference_max_feature_nodes,
            target_count=self.reference_target_count,
            scenario_id=f"{self.profile_name}-calibration",
            environment_label="granite-h200",
            source_batch_size=self.batch_size,
            feature_batch_size=self.batch_size,
            logit_batch_size=self.batch_size,
        )


@dataclass(frozen=True)
class StressRecommendation:
    size_label: str
    batch_size: int
    fetch_chunk_size: int
    decoder_cache_bytes: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.fetch_chunk_size <= 0:
            raise ValueError("stress recommendation batch/chunk must be positive")
        if self.decoder_cache_bytes < 0:
            raise ValueError("stress decoder cache must be nonnegative")


@dataclass(frozen=True)
class StressArithmeticFixture:
    recommendation: StressRecommendation
    profile: ProviderProfile
    semantics: TraceSemantics
    envelope: ResourceEnvelope


def _observation(
    name: str,
    architecture: str,
    dimensions: ProviderDimensions,
    host_gib: int,
    seconds: tuple[float, float],
    max_rss_kib: tuple[int, int],
    batch: int,
    chunk: int,
    cache_gib: int,
    active_features: int,
    physical_batch_cap: int,
    *,
    tiled_columns: int | None = None,
    recompute: bool = False,
) -> ResourceCalibrationObservation:
    return ResourceCalibrationObservation(
        profile_name=name,
        architecture=architecture,
        dimensions=dimensions,
        requested_host_memory_bytes=host_gib * GIB,
        walltime_seconds_range=seconds,
        max_rss_bytes_range=(max_rss_kib[0] * KIB, max_rss_kib[1] * KIB),
        batch_size=batch,
        fetch_chunk_size=chunk,
        decoder_cache_bytes=cache_gib * GIB,
        reference_prompt_token_count=128,
        reference_active_features=active_features,
        reference_max_feature_nodes=8192,
        reference_target_count=1,
        validated_physical_batch_cap=physical_batch_cap,
        validated_tiled_row_store=tiled_columns is not None,
        validated_recompute_row_store=recompute,
        row_store_tile_column_bound=tiled_columns,
    )


GRANITE_H200_CALIBRATIONS = (
    _observation(
        "granite_h200_1b_clt_b1000_c4096_cache8",
        "clt",
        ProviderDimensions(26, 1152, 262_144, 26),
        200,
        (83.25, 156.50),
        (4_736_848, 38_675_944),
        1000,
        4096,
        8,
        20_000,
        128,
        tiled_columns=2048,
        recompute=True,
    ),
    _observation(
        "granite_h200_1b_plt_b128_c4096_cache0",
        "plt",
        ProviderDimensions(26, 1152, 262_144, 1),
        200,
        (2717.81, 2928.02),
        (66_688_696, 70_916_012),
        128,
        4096,
        0,
        80_000,
        64,
        tiled_columns=16384,
        recompute=True,
    ),
    _observation(
        "granite_h200_4b_plt_b128_c4096_cache0",
        "plt",
        ProviderDimensions(34, 2560, 262_144, 1),
        400,
        (4972.08, 5471.70),
        (190_327_960, 194_871_248),
        128,
        4096,
        0,
        100_000,
        128,
    ),
    _observation(
        "granite_h200_12b_plt_b64_c4096_cache0",
        "plt",
        ProviderDimensions(48, 3840, 262_144, 1),
        600,
        (20878.41, 23051.72),
        (38_719_956, 244_429_132),
        64,
        4096,
        0,
        120_000,
        64,
    ),
)


_FIXED_VRAM_BY_PROFILE = {
    "granite_h200_1b_clt_b1000_c4096_cache8": 24 * GIB,
    "granite_h200_1b_plt_b128_c4096_cache0": 28 * GIB,
    "granite_h200_4b_plt_b128_c4096_cache0": 48 * GIB,
    "granite_h200_12b_plt_b64_c4096_cache0": 72 * GIB,
}
_CHECKPOINT_BYTES_BY_PROFILE = {
    "granite_h200_1b_clt_b1000_c4096_cache8": 12 * GIB,
    "granite_h200_1b_plt_b128_c4096_cache0": 16 * GIB,
    "granite_h200_4b_plt_b128_c4096_cache0": 32 * GIB,
    "granite_h200_12b_plt_b64_c4096_cache0": 64 * GIB,
}
# C2 Granite 361_base wall-clock ratios relative to each provider's full-file
# arm (A). CLT: A=111.23s, D=361.73s, E=710.21s. PLT: A=2494.32s,
# D=5918.07s, E=20581.41s. These calibrate mechanism cost only, not fidelity.
_ROW_STORE_WALLTIME_MULTIPLIERS_BY_PROFILE = {
    "granite_h200_1b_clt_b1000_c4096_cache8": (
        ("file_backed_full", 1.0),
        ("tiled", 1.5),
        ("recompute", 3.05),
    ),
    "granite_h200_1b_plt_b128_c4096_cache0": (
        ("file_backed_full", 1.0),
        ("tiled", 1.25),
        ("recompute", 5.1),
    ),
    "granite_h200_4b_plt_b128_c4096_cache0": (
        ("file_backed_full", 1.0),
        ("tiled", 1.25),
        ("recompute", 5.1),
    ),
    "granite_h200_12b_plt_b64_c4096_cache0": (
        ("file_backed_full", 1.0),
        ("tiled", 1.25),
        ("recompute", 5.1),
    ),
}

_REPLAY_TILE_CACHE_BY_ARCHITECTURE = {"clt": 1 * GIB, "plt": 4 * GIB}


def _session_memory_bytes_per_item(dimensions: ProviderDimensions) -> int:
    reference = 26 * 1152
    return round(87_000_000 * dimensions.n_layers * dimensions.d_model / reference)


def _phase_walltime_models(
    low: float, high: float
) -> tuple[PhaseWalltimeModel, ...]:
    # Additive phase shares replace the v0.2 all-phases multiplier. The total
    # remains exactly anchored to the observed interval.
    definitions = (
        ("phase0", 0.10, {"affected_by_fetch": True, "affected_by_prefetch": True}),
        ("phase1", 0.20, {"scales_with_session_steps": True, "scales_with_phase1_steps": True}),
        ("phase2", 0.10, {}),
        ("phase3", 0.20, {"scales_with_phase3_steps": True}),
        ("phase4", 0.35, {"scales_with_phase4_steps": True, "affected_by_replay": True, "affected_by_row_policy": True}),
        ("phase5", 0.05, {}),
    )
    return tuple(
        PhaseWalltimeModel(
            phase=phase,
            reference_low_seconds=low * share,
            reference_high_seconds=high * share,
            **effects,
        )
        for phase, share, effects in definitions
    )


def _profile(observation: ResourceCalibrationObservation) -> ProviderProfile:
    topology = (
        DecoderTopology.CROSS_LAYER
        if observation.architecture == "clt"
        else DecoderTopology.SAME_LAYER
    )
    semantics = observation.reference_semantics()
    physical_cap = observation.validated_physical_batch_cap
    reference_work = compute_work_units(
        prompt_token_count=semantics.prompt_token_count,
        estimated_active_features=semantics.estimated_nnz,
        target_count=semantics.target_count,
        dtype_bytes=semantics.dtype_bytes,
        trace_capacity=observation.batch_size,
        dimensions=observation.dimensions,
        decoder_fetch_chunk_size=observation.fetch_chunk_size,
        effective_source_batch_size=observation.batch_size,
        feature_batch_size=observation.batch_size,
        logit_batch_size=observation.batch_size,
        source_microbatch_size=observation.batch_size,
        feature_microbatch_size=min(observation.batch_size, physical_cap),
        logit_microbatch_size=min(observation.batch_size, physical_cap),
        replay_window=4,
        prefetch_depth=2,
    )
    return ProviderProfile(
        profile_name=observation.profile_name,
        profile_version=PROFILE_VERSION,
        planner_version=PLANNER_VERSION,
        identity=ProviderIdentity(
            provider_type="gemmascope2",
            provider_version="promoted-v2",
            checkpoint_format=observation.architecture,
            checkpoint_identity=observation.profile_name,
            hook_identity="corrected-input-hook",
            architecture=observation.architecture,
            decoder_topology=topology,
        ),
        dimensions=observation.dimensions,
        capabilities=ProviderCapabilities(
            supports_decoder_chunk_cache=True,
            supports_streaming_decoder=True,
            supports_encoder_row_materialization=True,
            supports_lazy_encoder_rows=True,
            supports_prefetch=True,
            supports_replay=True,
            supports_full_row_store=True,
            supports_tiled_row_store=True,
            supports_recompute_row_store=True,
        ),
        costs=ProviderCostMetadata(
            cost_model_version="calibrated-v3",
            fixed_vram_bytes=_FIXED_VRAM_BY_PROFILE[observation.profile_name],
            trace_vram_coefficient=0.0,
            target_vram_coefficient=1.0,
            source_microbatch_vram_coefficient=0.0,
            feature_microbatch_vram_coefficient=0.0,
            logit_microbatch_vram_coefficient=0.0,
            replay_vram_coefficient=0.0,
            known_rigid_host_bytes=0,
            baseline_total_host_bytes=observation.max_rss_bytes_range[1],
            file_cache_included_in_host_baseline=True,
            reference_replay_window=4,
            reference_encoder_residency="eager",
            active_host_coefficient=0.0,
            prompt_host_coefficient=0.0,
            checkpoint_file_bytes=_CHECKPOINT_BYTES_BY_PROFILE[observation.profile_name],
            calibrated_walltime_low_seconds=observation.walltime_seconds_range[0],
            calibrated_walltime_high_seconds=observation.walltime_seconds_range[1],
            walltime_reference_work_units=reference_work,
            row_store_walltime_multipliers=(
                _ROW_STORE_WALLTIME_MULTIPLIERS_BY_PROFILE[observation.profile_name]
            ),
        ),
        default_fetch_chunk_size=observation.fetch_chunk_size,
        max_fetch_chunk_size=(10_080 if observation.architecture == "clt" else 32_768),
        max_session_capacity=4096,
        max_phase1_source_batch_size=4096,
        max_source_microbatch_size=4096,
        max_phase3_microbatch_size=4096,
        max_phase4_microbatch_size=4096,
        default_decoder_cache_bytes=observation.decoder_cache_bytes,
        max_decoder_cache_bytes=32 * GIB,
        default_replay_window=4,
        max_replay_window=8,
        default_prefetch_depth=2,
        max_prefetch_depth=4,
        estimated_active_features_per_token=(
            observation.reference_active_features
            / observation.reference_prompt_token_count
        ),
        row_store_tile_column_bound=(
            observation.row_store_tile_column_bound
            or (16384 if observation.architecture == "plt" else 2048)
        ),
        safety_limits=ProviderSafetyLimits(
            max_prompt_token_count=4096,
            max_active_features=(
                4096
                * observation.dimensions.d_features
                * observation.dimensions.n_layers
            ),
            max_feature_nodes=(
                observation.dimensions.d_features * observation.dimensions.n_layers
            ),
            max_target_count=4096,
            max_logical_batch_size=4096,
            max_physical_rows=4096,
            max_decoder_cache_bytes=32 * GIB,
            max_replay_window=8,
            max_prefetch_depth=4,
            max_replay_tile_cache_bytes=16 * GIB,
        ),
        calibration_support=CalibrationSupport(
            prompt_token_count=observation.reference_prompt_token_count,
            active_features=observation.reference_active_features,
            max_feature_nodes=observation.reference_max_feature_nodes,
            target_count=observation.reference_target_count,
            logical_batch_size=observation.batch_size,
            session_capacity=observation.batch_size,
            phase1_source_batch_size=observation.batch_size,
            # The original exact-trace baseline is direct evidence for the
            # larger CLT batches even though later Phase-D gates used a more
            # conservative cap while validating the individual mechanisms.
            phase3_microbatch_size=observation.batch_size,
            phase4_microbatch_size=observation.batch_size,
            decoder_cache_bytes=observation.decoder_cache_bytes,
            replay_window=4,
            prefetch_depth=2,
            replay_tile_cache_bytes=(
                _REPLAY_TILE_CACHE_BY_ARCHITECTURE[observation.architecture]
            ),
            row_store_policies=(
                "file_backed_full",
                *(("tiled",) if observation.validated_tiled_row_store else ()),
                *(("recompute",) if observation.validated_recompute_row_store else ()),
            ),
            evidence=(observation.profile_name,),
            provisional_dimensions=(
                observation.dimensions.d_model != 1152
                or observation.reference_prompt_token_count != 128
            ),
        ),
        phase_memory_models=(
            PhaseMemoryModel(
                phase="session",
                session_vram_bytes_per_item=_session_memory_bytes_per_item(
                    observation.dimensions
                ),
                includes_decoder_cache=True,
            ),
            PhaseMemoryModel(phase="phase0"),
            PhaseMemoryModel(phase="phase3"),
            PhaseMemoryModel(
                phase="phase4",
                includes_replay_tile_cache=True,
            ),
        ),
        phase_walltime_models=_phase_walltime_models(
            observation.walltime_seconds_range[0],
            observation.walltime_seconds_range[1],
        ),
        reference_replay_tile_cache_bytes=(
            _REPLAY_TILE_CACHE_BY_ARCHITECTURE[observation.architecture]
        ),
    )


RECORDED_PROVIDER_PROFILES = immutable_mapping(
    {observation.profile_name: _profile(observation) for observation in GRANITE_H200_CALIBRATIONS}
)


HISTORICAL_STRESS_RECOMMENDATIONS = (
    StressRecommendation("1b", 1024, 8192, 32 * GIB),
    StressRecommendation("4b", 512, 4096, 16 * GIB),
    StressRecommendation("12b", 256, 2048, 8 * GIB),
)


def _stress_fixture(
    recommendation: StressRecommendation,
    template_name: str,
) -> StressArithmeticFixture:
    template = RECORDED_PROVIDER_PROFILES[template_name]
    semantics = TraceSemantics(
        prompt_token_count=256,
        estimated_active_features=50_000,
        max_feature_nodes=8192,
        target_count=1,
        scenario_id=f"historical-stress-{recommendation.size_label}",
        environment_label="arithmetic-fixture",
        source_batch_size=recommendation.batch_size,
        feature_batch_size=recommendation.batch_size,
        logit_batch_size=recommendation.batch_size,
    )
    reference_work = compute_work_units(
        prompt_token_count=semantics.prompt_token_count,
        estimated_active_features=semantics.estimated_nnz,
        target_count=semantics.target_count,
        dtype_bytes=semantics.dtype_bytes,
        trace_capacity=recommendation.batch_size,
        dimensions=template.dimensions,
        decoder_fetch_chunk_size=recommendation.fetch_chunk_size,
        effective_source_batch_size=recommendation.batch_size,
        feature_batch_size=recommendation.batch_size,
        logit_batch_size=recommendation.batch_size,
        source_microbatch_size=recommendation.batch_size,
        feature_microbatch_size=recommendation.batch_size,
        logit_microbatch_size=recommendation.batch_size,
        replay_window=1,
        prefetch_depth=0,
    )
    profile = replace(
        template,
        profile_name=f"historical-stress-{recommendation.size_label}",
        profile_version="historical-stress-v1",
        identity=replace(
            template.identity,
            checkpoint_identity=f"historical-stress-{recommendation.size_label}",
        ),
        costs=replace(
            template.costs,
            calibrated_walltime_low_seconds=100.0,
            calibrated_walltime_high_seconds=200.0,
            walltime_reference_work_units=reference_work,
        ),
        safety_limits=replace(
            template.safety_limits,
            max_decoder_cache_bytes=recommendation.decoder_cache_bytes,
        ),
        calibration_support=CalibrationSupport(
            prompt_token_count=semantics.prompt_token_count,
            active_features=semantics.estimated_nnz,
            max_feature_nodes=semantics.max_feature_nodes,
            target_count=semantics.target_count,
            logical_batch_size=recommendation.batch_size,
            session_capacity=recommendation.batch_size,
            phase1_source_batch_size=recommendation.batch_size,
            phase3_microbatch_size=recommendation.batch_size,
            phase4_microbatch_size=recommendation.batch_size,
            decoder_cache_bytes=recommendation.decoder_cache_bytes,
            replay_window=1,
            prefetch_depth=0,
            replay_tile_cache_bytes=0,
            row_store_policies=("file_backed_full",),
            evidence=(f"historical-stress-{recommendation.size_label}",),
        ),
        phase_walltime_models=_phase_walltime_models(100.0, 200.0),
        reference_replay_tile_cache_bytes=0,
        default_fetch_chunk_size=recommendation.fetch_chunk_size,
        max_fetch_chunk_size=recommendation.fetch_chunk_size,
        max_session_capacity=recommendation.batch_size,
        max_phase1_source_batch_size=recommendation.batch_size,
        max_source_microbatch_size=recommendation.batch_size,
        max_phase3_microbatch_size=recommendation.batch_size,
        max_phase4_microbatch_size=recommendation.batch_size,
        default_decoder_cache_bytes=recommendation.decoder_cache_bytes,
        max_decoder_cache_bytes=recommendation.decoder_cache_bytes,
        default_replay_window=1,
        default_prefetch_depth=0,
    )
    envelope = ResourceEnvelope(
        # This is an arithmetic reproduction fixture, not an H200 admission
        # claim. The historical presets require a larger synthetic envelope
        # under the corrected v0.3 residency model.
        total_vram_bytes=300 * GIB,
        host_budget_bytes=600 * GIB,
        file_cache_allowance_bytes=64 * GIB,
        local_disk_bytes=1024 * GIB,
        scratch_disk_bytes=1024 * GIB,
        walltime_seconds=1000,
    )
    return StressArithmeticFixture(recommendation, profile, semantics, envelope)


HISTORICAL_STRESS_FIXTURES = (
    _stress_fixture(
        HISTORICAL_STRESS_RECOMMENDATIONS[0],
        "granite_h200_1b_clt_b1000_c4096_cache8",
    ),
    _stress_fixture(
        HISTORICAL_STRESS_RECOMMENDATIONS[1],
        "granite_h200_4b_plt_b128_c4096_cache0",
    ),
    _stress_fixture(
        HISTORICAL_STRESS_RECOMMENDATIONS[2],
        "granite_h200_12b_plt_b64_c4096_cache0",
    ),
)
