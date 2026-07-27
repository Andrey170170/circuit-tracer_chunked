"""Domain-owned physical policies and the resolved tracing plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal, Mapping

from circuit_tracer.governor.contracts import (
    AdmissionMode,
    AdmissionReport,
    CachePolicy,
    PhysicalExecutionRequirements,
    ProviderProfile,
    ResourceEnvelope,
    StorageTier,
    TracePlan,
    TraceSemantics as PlanningWorkload,
)
from circuit_tracer.governor.calibration import CalibrationCatalog
from circuit_tracer.governor.response_models import ResponseBundle

from .problem import TraceSemantics, _nonnegative, _positive


@dataclass(frozen=True)
class DecoderCachePolicy:
    """Physical cross-trace decoder reuse owned by a tracing session."""

    enabled: bool = False
    max_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("decoder cache max_bytes must be positive")
        if not self.enabled and self.max_bytes is not None:
            raise ValueError("decoder cache max_bytes requires enabled=True")


@dataclass(frozen=True)
class DecoderPlan:
    """Provider decoder fetch and file-cache execution policy."""

    fetch_chunk_size: int | None = None
    provider_file_cache: CachePolicy = CachePolicy.AUTO

    def __post_init__(self) -> None:
        _positive("decoder fetch_chunk_size", self.fetch_chunk_size)


@dataclass(frozen=True)
class SessionPlan:
    """NNSight graph capacity and physical phase batching."""

    capacity: int | None = None
    source_microbatch_max_rows: int | None = None
    phase3_microbatch_max_rows: int | None = None
    phase4_execution_batch_max_rows: int | None = None
    phase1_trace_batch_policy: Literal["legacy", "cap_effective_batches"] = "legacy"
    phase1_trace_batch_size_max: int | None = None
    decoder_cache: DecoderCachePolicy = field(default_factory=DecoderCachePolicy)

    def __post_init__(self) -> None:
        for name in (
            "capacity",
            "source_microbatch_max_rows",
            "phase3_microbatch_max_rows",
            "phase4_execution_batch_max_rows",
        ):
            _positive(name, getattr(self, name))
        if self.capacity is not None:
            for name in (
                "source_microbatch_max_rows",
                "phase3_microbatch_max_rows",
                "phase4_execution_batch_max_rows",
            ):
                value = getattr(self, name)
                if value is not None and value > self.capacity:
                    raise ValueError(f"{name} cannot exceed session capacity")
        _positive("phase1_trace_batch_size_max", self.phase1_trace_batch_size_max)
        if (
            self.phase1_trace_batch_policy == "cap_effective_batches"
            and self.phase1_trace_batch_size_max is None
        ):
            raise ValueError("capped Phase-1 batching requires phase1_trace_batch_size_max")

@dataclass(frozen=True)
class RowStoragePlan:
    """Attribution-row retention, layout, cache, and staging policy."""

    retention: Literal["full_file", "none_recompute"] = "full_file"
    full_retention_backend: Literal["full_file", "column_tiled_v1"] = "full_file"
    feature_column_tile_size: int = 2048
    influence_row_tile_size: int = 4096
    influence_column_tile_size: int = 2048
    cache_control: Literal[
        "off",
        "fadvise_dontneed_after_append_v1",
        "fadvise_dontneed_after_append_and_read_v1",
    ] = "off"
    temp_root_policy: Literal["default", "env_node_local"] = "default"
    temp_root: str | PathLike[str] | None = None
    preallocate: bool = True
    replay_tile_cache_bytes: int | None = None
    exact_encoder_residency: Literal["lazy", "active_cpu", "active_pinned_cpu"] = "lazy"
    placement: StorageTier | None = None

    def __post_init__(self) -> None:
        for name in (
            "feature_column_tile_size",
            "influence_row_tile_size",
            "influence_column_tile_size",
        ):
            _positive(name, getattr(self, name))
        _nonnegative("replay_tile_cache_bytes", self.replay_tile_cache_bytes)
        if self.temp_root is not None and self.temp_root_policy != "default":
            raise ValueError("an explicit temp_root cannot be combined with temp_root_policy")


@dataclass(frozen=True)
class ReplayPlan:
    """Exact replay windows, donor evidence, and vector staging policy."""

    feature_window: int = 4
    error_vector_prefetch_lookahead: int = 2
    stage_encoder_vecs_on_cpu: bool | None = None
    stage_error_vectors_on_cpu: bool | None = None
    decoder_contraction_tile: int | None = None
    phase0_donor_bundle: str | PathLike[str] | None = None
    phase0_mode: Literal["disabled", "donor_phase0"] = "disabled"
    phase0_donor_context_policy: Literal["strict", "warn"] = "strict"
    phase3_gradient_donor_bundle: str | PathLike[str] | None = None
    phase3_gradient_mode: Literal["disabled", "donor"] = "disabled"
    phase3_row_donor_bundle: str | PathLike[str] | None = None
    phase3_row_mode: Literal["disabled", "donor"] = "disabled"
    phase3_validation_policy: Literal["strict"] = "strict"

    def __post_init__(self) -> None:
        _positive("feature_window", self.feature_window)
        _nonnegative("error_vector_prefetch_lookahead", self.error_vector_prefetch_lookahead)
        _positive("decoder_contraction_tile", self.decoder_contraction_tile)
        pairs = (
            (self.phase0_mode, self.phase0_donor_bundle, "phase0"),
            (self.phase3_gradient_mode, self.phase3_gradient_donor_bundle, "phase3 gradient"),
            (self.phase3_row_mode, self.phase3_row_donor_bundle, "phase3 row"),
        )
        for mode, bundle, label in pairs:
            if mode != "disabled" and bundle is None:
                raise ValueError(f"{label} replay requires its donor bundle")


@dataclass(frozen=True)
class FrontierExpansionPlan:
    """Physical Phase-4 execution, storage, and planner mechanisms."""

    scheduler_debug: bool = False
    scheduler_telemetry_detail: Literal["summary", "normal", "debug"] = "normal"
    refresh_optimization: Literal["off", "v1"] = "v1"
    refresh_prepared_chunk_cache_bytes: int = 0
    refresh_active_row_accumulation: Literal["zero_fill", "direct_v1"] = "direct_v1"
    row_executor: Literal["batched", "streaming_v1"] = "batched"
    row_reduction: Literal["off", "gpu_v1"] = "gpu_v1"
    feature_batch_planning: bool = False
    feature_batch_size_max: int | None = None
    feature_batch_target_reserved_fraction: float = 0.9
    feature_batch_min_free_fraction: float = 0.05
    feature_batch_probe_batches: int = 1
    feature_vjp_tape_batch_window: int = 1
    feature_vjp_tape_max_bytes: int = 0
    decoder_page_prefetch_depth: int = 0
    decoder_active_row_residency: bool = False
    decoder_active_row_max_bytes: int = 0
    phase0_decoder_row_ranges: bool = False

    def __post_init__(self) -> None:
        _nonnegative("refresh_prepared_chunk_cache_bytes", self.refresh_prepared_chunk_cache_bytes)
        _positive("feature_batch_size_max", self.feature_batch_size_max)
        _positive("feature_batch_probe_batches", self.feature_batch_probe_batches)
        _positive("feature_vjp_tape_batch_window", self.feature_vjp_tape_batch_window)
        _nonnegative("feature_vjp_tape_max_bytes", self.feature_vjp_tape_max_bytes)
        _nonnegative("decoder_page_prefetch_depth", self.decoder_page_prefetch_depth)
        _nonnegative("decoder_active_row_max_bytes", self.decoder_active_row_max_bytes)
        if self.feature_vjp_tape_batch_window > 1 and self.feature_vjp_tape_max_bytes == 0:
            raise ValueError(
                "feature_vjp_tape_batch_window > 1 requires "
                "feature_vjp_tape_max_bytes > 0"
            )
        for name in (
            "feature_batch_target_reserved_fraction",
            "feature_batch_min_free_fraction",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ObservabilityPolicy:
    """Telemetry sinks, human rendering, diagnostics, and artifact capture."""

    verbose: bool = False
    profile: bool = False
    profile_log_interval: int = 1
    telemetry_max_events: int | None = None
    telemetry_jsonl_path: str | PathLike[str] | None = None
    telemetry_context: Mapping[str, object] = field(default_factory=dict, repr=False)
    phase4_anomaly_debug: bool = False
    cross_cluster_debug: bool = False
    capture_phase0_donor_bundle: bool = False
    capture_phase3_seed_bundle: bool = False
    capture_phase3_gradient_bundle: bool = False
    capture_phase3_row_bundle: bool = False
    capture_feature_semantic_descriptors: bool = False
    semantic_descriptor_top_k: int = 2048
    semantic_descriptor_dim: int = 64

    def __post_init__(self) -> None:
        _positive("profile_log_interval", self.profile_log_interval)
        _positive("telemetry_max_events", self.telemetry_max_events)
        _positive("semantic_descriptor_top_k", self.semantic_descriptor_top_k)
        _positive("semantic_descriptor_dim", self.semantic_descriptor_dim)


@dataclass(frozen=True)
class DiagnosticStopPolicy:
    """Diagnostic-only termination contract; never produces a scientific graph."""

    mode: Literal["none", "phase0_probe", "transition_probe"] = "none"
    phase4_batches: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "transition_probe":
            if self.phase4_batches is None or self.phase4_batches <= 0:
                raise ValueError(
                    "transition_probe requires a positive phase4_batches count"
                )
        elif self.phase4_batches is not None:
            raise ValueError("phase4_batches is valid only for transition_probe")


@dataclass(frozen=True)
class ExecutionConstraints:
    """Explicit physical restrictions, grouped by their mechanism owners."""

    session: SessionPlan = field(default_factory=SessionPlan)
    decoder: DecoderPlan = field(default_factory=DecoderPlan)
    storage: RowStoragePlan = field(default_factory=RowStoragePlan)
    replay: ReplayPlan = field(default_factory=ReplayPlan)
    frontier: FrontierExpansionPlan = field(default_factory=FrontierExpansionPlan)
    observability: ObservabilityPolicy = field(default_factory=ObservabilityPolicy)
    diagnostic_stop: DiagnosticStopPolicy = field(default_factory=DiagnosticStopPolicy)
    offload: Literal["cpu", "disk", None] = None
    compact_output: bool = False


@dataclass(frozen=True)
class TraceEvidence:
    """Named evidence and caller provenance that do not alter execution policy."""

    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if (self.name is None) != (self.version is None):
            raise ValueError("evidence name and version must be provided together")


@dataclass(frozen=True)
class ResolvedTracePlan:
    """Validated requested policy; backend preparation may resolve effective mechanisms."""

    semantics: TraceSemantics
    execution: ExecutionConstraints
    semantic_fingerprint: str
    requested_execution_fingerprint: str
    backend: Literal["nnsight", "transformerlens"]
    governor_admission_mode: AdmissionMode = AdmissionMode.ENFORCE
    evidence_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    admission_report: AdmissionReport | None = None
    planning_profile: ProviderProfile | None = field(default=None, repr=False)
    planning_envelope: ResourceEnvelope | None = field(default=None, repr=False)
    planning_workload: PlanningWorkload | None = field(default=None, repr=False)
    planning_requirements: PhysicalExecutionRequirements | None = field(default=None, repr=False)
    planning_trace_plan: TracePlan | None = field(default=None, repr=False)
    planning_calibration_catalog: CalibrationCatalog | None = field(default=None, repr=False)
    planning_response_bundle: ResponseBundle | None = field(default=None, repr=False)
    planning_parent_fingerprint: str | None = None
    planning_epoch_fingerprint: str | None = None

    @property
    def execution_fingerprint(self) -> str:
        """Compatibility alias for the pre-execution requested-policy fingerprint."""
        return self.requested_execution_fingerprint
