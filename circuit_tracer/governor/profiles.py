from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import (
    DecoderTopology,
    ProviderCapabilities,
    ProviderCostMetadata,
    ProviderDimensions,
    ProviderIdentity,
    ProviderProfile,
    ResourceEnvelope,
    TraceSemantics,
    immutable_mapping,
)
from .resolver import compute_work_units


GIB = 1024**3
KIB = 1024
PROFILE_VERSION = "granite-h200-v2"
PLANNER_VERSION = "governor-v0.2"


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


def _profile(observation: ResourceCalibrationObservation) -> ProviderProfile:
    topology = (
        DecoderTopology.CROSS_LAYER
        if observation.architecture == "clt"
        else DecoderTopology.SAME_LAYER
    )
    semantics = observation.reference_semantics()
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
        feature_microbatch_size=observation.batch_size,
        logit_microbatch_size=observation.batch_size,
        replay_window=1,
        prefetch_depth=0,
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
            supports_tiled_row_store=False,
            supports_recompute_row_store=False,
        ),
        costs=ProviderCostMetadata(
            cost_model_version="calibrated-v2",
            fixed_vram_bytes=_FIXED_VRAM_BY_PROFILE[observation.profile_name],
            trace_vram_coefficient=0.0001,
            target_vram_coefficient=1.0,
            source_microbatch_vram_coefficient=0.01,
            feature_microbatch_vram_coefficient=0.01,
            logit_microbatch_vram_coefficient=0.01,
            replay_vram_coefficient=0.01,
            known_rigid_host_bytes=0,
            baseline_total_host_bytes=observation.max_rss_bytes_range[1],
            file_cache_included_in_host_baseline=True,
            reference_replay_window=1,
            reference_encoder_residency="eager",
            active_host_coefficient=0.0,
            prompt_host_coefficient=0.0,
            checkpoint_file_bytes=_CHECKPOINT_BYTES_BY_PROFILE[observation.profile_name],
            calibrated_walltime_low_seconds=observation.walltime_seconds_range[0],
            calibrated_walltime_high_seconds=observation.walltime_seconds_range[1],
            walltime_reference_work_units=reference_work,
        ),
        default_fetch_chunk_size=observation.fetch_chunk_size,
        max_fetch_chunk_size=observation.fetch_chunk_size,
        max_physical_microbatch=observation.batch_size,
        default_decoder_cache_bytes=observation.decoder_cache_bytes,
        max_decoder_cache_bytes=max(observation.decoder_cache_bytes, 8 * GIB),
        default_replay_window=1,
        max_replay_window=8,
        default_prefetch_depth=0,
        max_prefetch_depth=4,
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
        default_fetch_chunk_size=recommendation.fetch_chunk_size,
        max_fetch_chunk_size=recommendation.fetch_chunk_size,
        max_physical_microbatch=recommendation.batch_size,
        default_decoder_cache_bytes=recommendation.decoder_cache_bytes,
        max_decoder_cache_bytes=recommendation.decoder_cache_bytes,
    )
    envelope = ResourceEnvelope(
        total_vram_bytes=141 * GIB,
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
