from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar

from .calibration import FidelityBudget, FidelityPrediction, ParetoAlternative


FINGERPRINT_SCHEMA_VERSION = 3
_DTYPE_BYTE_WIDTHS = MappingProxyType({"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8})
_K = TypeVar("_K")
_V = TypeVar("_V")


class FidelityMode(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    BEST_EFFORT = "best_effort"
    RESEARCH = "research"


class AdmissionMode(str, Enum):
    ENFORCE = "enforce"
    ADVISORY = "advisory"


class CachePolicy(str, Enum):
    AUTO = "auto"
    WARM = "warm"
    BOUNDED = "bounded"
    STREAMING = "streaming"


class EncoderResidency(str, Enum):
    EAGER = "eager"
    LAZY_PER_REQUEST = "lazy_per_request"


class RowStorePolicy(str, Enum):
    FULL = "file_backed_full"
    TILED = "tiled"
    RECOMPUTE = "recompute"


class StorageTier(str, Enum):
    LOCAL = "local"
    SCRATCH = "scratch"


class DemandTier(str, Enum):
    VRAM = "vram"
    HOST = "host"
    FILE_BACKED = "file_backed"
    LOCAL_DISK = "local_disk"
    SCRATCH_DISK = "scratch_disk"
    WALLTIME = "walltime"


class DemandClass(str, Enum):
    RIGID = "rigid"
    ELASTIC = "elastic"


class DemandLifetime(str, Enum):
    PERMANENT = "permanent"
    PHASE = "phase"
    TRANSIENT = "transient"


class DecoderTopology(str, Enum):
    CROSS_LAYER = "cross_layer"
    SAME_LAYER = "same_layer"
    TOP_K = "top_k"


class PlanStatus(str, Enum):
    ADVISORY_ADMITTED = "advisory_admitted"
    ADVISORY_REFUSED = "advisory_refused"


def _positive(name: str, value: int | float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative(name: str, value: int | float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _nonempty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


def _sorted_unique(name: str, values: tuple[tuple[str, str], ...]) -> None:
    if tuple(sorted(values)) != values or len({key for key, _ in values}) != len(values):
        raise ValueError(f"{name} must be sorted with unique keys")


def _sorted_unique_metrics(name: str, values: tuple[tuple[str, float], ...]) -> None:
    if tuple(sorted(values)) != values or len({key for key, _ in values}) != len(values):
        raise ValueError(f"{name} must be sorted with unique keys")
    if any(not key or not isinstance(value, (int, float)) for key, value in values):
        raise ValueError(f"{name} must contain named numeric values")


def dtype_byte_width(dtype: str) -> int:
    try:
        return _DTYPE_BYTE_WIDTHS[dtype]
    except KeyError as error:
        supported = ", ".join(_DTYPE_BYTE_WIDTHS)
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {supported}") from error


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TraceSemantics:
    prompt_token_count: int
    estimated_active_features: int
    max_feature_nodes: int
    target_count: int
    scenario_id: str
    environment_label: str
    source_batch_size: int
    feature_batch_size: int
    logit_batch_size: int
    window_id: str | None = None
    phase1_source_cap: int | None = None
    decoder_reduction_tile: int = 4096
    decoder_reduction_order: str = "canonical"
    frontier_refresh_stride: int = 1
    frontier_checkpoints: tuple[int, ...] = ()
    dtype: str = "fp32"
    hooks: tuple[str, ...] = ()
    provider_approximation: str = "exact"
    feature_cap: int | None = None
    logit_cap: int | None = None
    row_store_content: str = "full"
    fidelity: FidelityMode = FidelityMode.EXACT
    fidelity_budget: FidelityBudget | None = None
    semantic_overrides: tuple[tuple[str, str], ...] = ()
    research_overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "prompt_token_count",
            "estimated_active_features",
            "max_feature_nodes",
            "target_count",
            "source_batch_size",
            "feature_batch_size",
            "logit_batch_size",
            "decoder_reduction_tile",
            "frontier_refresh_stride",
        ):
            _positive(name, getattr(self, name))
        for name in ("phase1_source_cap", "feature_cap", "logit_cap"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        _nonempty("scenario_id", self.scenario_id)
        _nonempty("environment_label", self.environment_label)
        _nonempty("decoder_reduction_order", self.decoder_reduction_order)
        _nonempty("provider_approximation", self.provider_approximation)
        _nonempty("row_store_content", self.row_store_content)
        dtype_byte_width(self.dtype)
        if any(value < 0 for value in self.frontier_checkpoints):
            raise ValueError("frontier_checkpoints must be nonnegative")
        if tuple(sorted(set(self.frontier_checkpoints))) != self.frontier_checkpoints:
            raise ValueError("frontier_checkpoints must be sorted and unique")
        _sorted_unique("semantic_overrides", self.semantic_overrides)
        _sorted_unique("research_overrides", self.research_overrides)
        for override_name, override_value in self.semantic_overrides + self.research_overrides:
            if not hasattr(self, override_name) or override_name in {
                "fidelity",
                "fidelity_budget",
                "semantic_overrides",
                "research_overrides",
            }:
                raise ValueError(f"unknown or non-semantic override field: {override_name}")
            expected = canonical_json(getattr(self, override_name))
            if override_value != expected:
                raise ValueError(
                    f"override {override_name} must equal its canonical requested value {expected}"
                )
        if self.fidelity is FidelityMode.EXACT:
            if self.fidelity_budget or self.semantic_overrides or self.research_overrides:
                raise ValueError("exact fidelity does not accept a budget or semantic overrides")
        elif self.fidelity is FidelityMode.BOUNDED:
            if (
                self.fidelity_budget is None
                or not self.fidelity_budget.metric_floors
                or not self.fidelity_budget.allowed_sensitive_axes
            ):
                raise ValueError(
                    "bounded fidelity requires metric floors and sensitive-axis allowances"
                )
            if self.research_overrides:
                raise ValueError("bounded fidelity does not accept research overrides")
        elif self.fidelity is FidelityMode.BEST_EFFORT:
            if self.fidelity_budget is None or not self.fidelity_budget.allowed_sensitive_axes:
                raise ValueError("best_effort fidelity requires explicit sensitive-axis allowances")
            if self.research_overrides:
                raise ValueError("best_effort fidelity does not accept research overrides")
        elif self.semantic_overrides:
            raise ValueError("research fidelity uses research_overrides only")

    @property
    def estimated_nnz(self) -> int:
        return self.estimated_active_features

    @property
    def dtype_bytes(self) -> int:
        return dtype_byte_width(self.dtype)

    def evidence_scope_parameters(self) -> tuple[tuple[str, str], ...]:
        values = {
            "decoder_reduction_order": self.decoder_reduction_order,
            "decoder_reduction_tile": self.decoder_reduction_tile,
            "estimated_active_features": self.estimated_active_features,
            "feature_batch_size": self.feature_batch_size,
            "feature_cap": self.feature_cap,
            "frontier_checkpoints": self.frontier_checkpoints,
            "frontier_refresh_stride": self.frontier_refresh_stride,
            "logit_batch_size": self.logit_batch_size,
            "logit_cap": self.logit_cap,
            "max_feature_nodes": self.max_feature_nodes,
            "prompt_token_count": self.prompt_token_count,
            "provider_approximation": self.provider_approximation,
            "phase1_source_cap": self.phase1_source_cap,
            "row_store_content": self.row_store_content,
            "source_batch_size": self.source_batch_size,
            "target_count": self.target_count,
            "hooks": self.hooks,
        }
        return tuple((name, canonical_json(value)) for name, value in sorted(values.items()))


@dataclass(frozen=True)
class ResourceEnvelope:
    total_vram_bytes: int
    host_budget_bytes: int
    file_cache_allowance_bytes: int
    local_disk_bytes: int
    scratch_disk_bytes: int
    walltime_seconds: float
    vram_fraction: float = 0.9
    vram_budget_bytes: int | None = None
    cache_policy: CachePolicy = CachePolicy.AUTO
    spill_roots: tuple[str, ...] = ("local", "scratch")

    def __post_init__(self) -> None:
        _positive("total_vram_bytes", self.total_vram_bytes)
        for name in (
            "host_budget_bytes",
            "file_cache_allowance_bytes",
            "local_disk_bytes",
            "scratch_disk_bytes",
        ):
            _nonnegative(name, getattr(self, name))
        _positive("walltime_seconds", self.walltime_seconds)
        if not 0 < self.vram_fraction <= 1:
            raise ValueError("vram_fraction must be in (0, 1]")
        if self.vram_budget_bytes is not None:
            _nonnegative("vram_budget_bytes", self.vram_budget_bytes)
        if tuple(dict.fromkeys(self.spill_roots)) != self.spill_roots:
            raise ValueError("spill_roots must be unique")
        if any(root not in {"local", "scratch"} for root in self.spill_roots):
            raise ValueError("spill_roots may contain only 'local' and 'scratch'")

    @property
    def effective_vram_budget_bytes(self) -> int:
        fraction_budget = int(self.total_vram_bytes * self.vram_fraction)
        return (
            fraction_budget
            if self.vram_budget_bytes is None
            else min(fraction_budget, self.vram_budget_bytes)
        )


@dataclass(frozen=True)
class ProviderIdentity:
    provider_type: str
    provider_version: str
    checkpoint_format: str
    checkpoint_identity: str
    hook_identity: str
    architecture: str
    decoder_topology: DecoderTopology
    approximation: str = "exact"
    semantic_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "provider_type",
            "provider_version",
            "checkpoint_format",
            "checkpoint_identity",
            "hook_identity",
            "architecture",
            "approximation",
        ):
            _nonempty(name, getattr(self, name))
        _sorted_unique("semantic_parameters", self.semantic_parameters)
        if self.decoder_topology is DecoderTopology.TOP_K:
            parameters = dict(self.semantic_parameters)
            raw_top_k = parameters.get("top_k")
            try:
                top_k = int(raw_top_k) if raw_top_k is not None else 0
            except ValueError as error:
                raise ValueError("TOP_K identity requires a positive integer top_k") from error
            if top_k <= 0 or raw_top_k != str(top_k):
                raise ValueError("TOP_K identity requires a positive integer top_k")


@dataclass(frozen=True)
class ProviderDimensions:
    n_layers: int
    d_model: int
    d_features: int
    decoder_output_span: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _positive(field.name, getattr(self, field.name))


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_decoder_chunk_cache: bool
    supports_streaming_decoder: bool
    supports_encoder_row_materialization: bool
    supports_lazy_encoder_rows: bool
    supports_prefetch: bool
    supports_replay: bool
    supports_full_row_store: bool = True
    supports_tiled_row_store: bool = False
    supports_recompute_row_store: bool = False


@dataclass(frozen=True)
class ProviderSafetyLimits:
    """Hard provider bounds. Requests beyond these limits are invalid."""

    max_prompt_token_count: int
    max_active_features: int
    max_feature_nodes: int
    max_target_count: int
    max_logical_batch_size: int
    max_physical_rows: int
    max_decoder_cache_bytes: int
    max_replay_window: int
    max_prefetch_depth: int
    max_replay_tile_cache_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {
                "max_decoder_cache_bytes",
                "max_prefetch_depth",
                "max_replay_tile_cache_bytes",
            }:
                _nonnegative(field.name, getattr(self, field.name))
            else:
                _positive(field.name, getattr(self, field.name))


@dataclass(frozen=True)
class CalibrationSupport:
    """Region directly supported by named calibration evidence."""

    prompt_token_count: int
    active_features: int
    max_feature_nodes: int
    target_count: int
    logical_batch_size: int
    session_capacity: int
    phase1_source_batch_size: int
    phase3_microbatch_size: int
    phase4_microbatch_size: int
    decoder_cache_bytes: int
    replay_window: int
    prefetch_depth: int
    replay_tile_cache_bytes: int
    row_store_policies: tuple[str, ...]
    evidence: tuple[str, ...]
    provisional_dimensions: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prompt_token_count", "active_features", "max_feature_nodes",
            "target_count", "logical_batch_size", "session_capacity",
            "phase1_source_batch_size", "phase3_microbatch_size",
            "phase4_microbatch_size", "replay_window",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "decoder_cache_bytes", "prefetch_depth", "replay_tile_cache_bytes"
        ):
            _nonnegative(name, getattr(self, name))
        if not self.row_store_policies or not self.evidence:
            raise ValueError("calibration support requires row policies and evidence")
        if len(set(self.row_store_policies)) != len(self.row_store_policies):
            raise ValueError("row_store_policies must be unique")
        if any(not item for item in self.evidence):
            raise ValueError("calibration evidence names must be nonempty")


@dataclass(frozen=True)
class PhaseMemoryModel:
    """Concurrent memory terms used by one execution phase."""

    phase: str
    fixed_vram_bytes: int = 0
    session_vram_bytes_per_item: int = 0
    microbatch_vram_bytes_per_item: int = 0
    host_bytes_per_item: int = 0
    includes_decoder_cache: bool = False
    includes_replay_tile_cache: bool = False

    def __post_init__(self) -> None:
        _nonempty("phase", self.phase)
        for field in fields(self):
            if field.name != "phase" and not isinstance(getattr(self, field.name), bool):
                _nonnegative(field.name, getattr(self, field.name))


@dataclass(frozen=True)
class PhaseWalltimeModel:
    """Additive calibrated phase interval and the dimensions that scale it."""

    phase: str
    reference_low_seconds: float
    reference_high_seconds: float
    scales_with_session_steps: bool = False
    scales_with_phase1_steps: bool = False
    scales_with_phase3_steps: bool = False
    scales_with_phase4_steps: bool = False
    affected_by_fetch: bool = False
    affected_by_replay: bool = False
    affected_by_prefetch: bool = False
    affected_by_row_policy: bool = False

    def __post_init__(self) -> None:
        _nonempty("phase", self.phase)
        _nonnegative("reference_low_seconds", self.reference_low_seconds)
        _nonnegative("reference_high_seconds", self.reference_high_seconds)
        if self.reference_high_seconds < self.reference_low_seconds:
            raise ValueError("phase walltime interval must be ordered")


@dataclass(frozen=True)
class ProviderCostMetadata:
    cost_model_version: str
    fixed_vram_bytes: int
    trace_vram_coefficient: float
    target_vram_coefficient: float
    source_microbatch_vram_coefficient: float
    feature_microbatch_vram_coefficient: float
    logit_microbatch_vram_coefficient: float
    replay_vram_coefficient: float
    known_rigid_host_bytes: int
    baseline_total_host_bytes: int | None
    file_cache_included_in_host_baseline: bool
    reference_replay_window: int
    reference_encoder_residency: str
    active_host_coefficient: float
    prompt_host_coefficient: float
    checkpoint_file_bytes: int
    calibrated_walltime_low_seconds: float
    calibrated_walltime_high_seconds: float
    walltime_reference_work_units: float
    row_store_walltime_multipliers: tuple[tuple[str, float], ...] = (
        ("file_backed_full", 1.0),
        ("tiled", 2.0),
        ("recompute", 6.0),
    )

    def __post_init__(self) -> None:
        _nonempty("cost_model_version", self.cost_model_version)
        for name in (
            "fixed_vram_bytes",
            "trace_vram_coefficient",
            "target_vram_coefficient",
            "source_microbatch_vram_coefficient",
            "feature_microbatch_vram_coefficient",
            "logit_microbatch_vram_coefficient",
            "replay_vram_coefficient",
            "known_rigid_host_bytes",
            "active_host_coefficient",
            "prompt_host_coefficient",
            "checkpoint_file_bytes",
        ):
            _nonnegative(name, getattr(self, name))
        if self.baseline_total_host_bytes is not None:
            _nonnegative("baseline_total_host_bytes", self.baseline_total_host_bytes)
        if self.file_cache_included_in_host_baseline and self.baseline_total_host_bytes is None:
            raise ValueError(
                "file_cache_included_in_host_baseline requires baseline_total_host_bytes"
            )
        if (
            self.baseline_total_host_bytes is not None
            and self.known_rigid_host_bytes > self.baseline_total_host_bytes
        ):
            raise ValueError("known_rigid_host_bytes cannot exceed baseline_total_host_bytes")
        _positive("reference_replay_window", self.reference_replay_window)
        if self.reference_encoder_residency not in {"eager", "lazy_per_request"}:
            raise ValueError("reference_encoder_residency must be eager or lazy_per_request")
        for name in (
            "calibrated_walltime_low_seconds",
            "calibrated_walltime_high_seconds",
            "walltime_reference_work_units",
        ):
            _positive(name, getattr(self, name))
        if self.calibrated_walltime_high_seconds < self.calibrated_walltime_low_seconds:
            raise ValueError("calibrated walltime range must be ordered")
        multipliers = dict(self.row_store_walltime_multipliers)
        if set(multipliers) != {"file_backed_full", "tiled", "recompute"}:
            raise ValueError("row_store_walltime_multipliers must define every row-store policy")
        for name, multiplier in multipliers.items():
            _positive(f"row_store_walltime_multipliers[{name}]", multiplier)

    def row_store_walltime_multiplier(self, policy: str) -> float:
        try:
            return dict(self.row_store_walltime_multipliers)[policy]
        except KeyError as error:
            raise ValueError(f"unknown row-store policy {policy!r}") from error


@dataclass(frozen=True)
class ProviderProfile:
    profile_name: str
    profile_version: str
    planner_version: str
    identity: ProviderIdentity
    dimensions: ProviderDimensions
    capabilities: ProviderCapabilities
    costs: ProviderCostMetadata
    default_fetch_chunk_size: int
    max_fetch_chunk_size: int
    max_session_capacity: int
    max_phase1_source_batch_size: int
    max_source_microbatch_size: int
    max_phase3_microbatch_size: int
    max_phase4_microbatch_size: int
    default_decoder_cache_bytes: int
    max_decoder_cache_bytes: int
    default_replay_window: int
    max_replay_window: int
    default_prefetch_depth: int
    max_prefetch_depth: int
    estimated_active_features_per_token: float = 1.0
    default_encoder_residency: EncoderResidency = EncoderResidency.LAZY_PER_REQUEST
    row_store_tile_column_bound: int | None = None
    calibration_label: str = "resource_calibration_only"
    safety_limits: ProviderSafetyLimits | None = None
    calibration_support: CalibrationSupport | None = None
    phase_memory_models: tuple[PhaseMemoryModel, ...] = ()
    phase_walltime_models: tuple[PhaseWalltimeModel, ...] = ()
    reference_replay_tile_cache_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("profile_name", "profile_version", "planner_version"):
            _nonempty(name, getattr(self, name))
        for name in (
            "default_fetch_chunk_size",
            "max_fetch_chunk_size",
            "max_session_capacity",
            "max_phase1_source_batch_size",
            "max_source_microbatch_size",
            "max_phase3_microbatch_size",
            "max_phase4_microbatch_size",
            "default_replay_window",
            "max_replay_window",
            "estimated_active_features_per_token",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "default_decoder_cache_bytes",
            "max_decoder_cache_bytes",
            "default_prefetch_depth",
            "max_prefetch_depth",
        ):
            _nonnegative(name, getattr(self, name))
        if self.default_fetch_chunk_size > self.max_fetch_chunk_size:
            raise ValueError("default fetch chunk exceeds profile maximum")
        if self.default_decoder_cache_bytes > self.max_decoder_cache_bytes:
            raise ValueError("default decoder cache exceeds profile maximum")
        if self.default_replay_window > self.max_replay_window:
            raise ValueError("default replay window exceeds profile maximum")
        if self.default_prefetch_depth > self.max_prefetch_depth:
            raise ValueError("default prefetch depth exceeds profile maximum")
        if self.row_store_tile_column_bound is not None:
            _positive("row_store_tile_column_bound", self.row_store_tile_column_bound)
        if self.capabilities.supports_tiled_row_store and self.row_store_tile_column_bound is None:
            raise ValueError("tiled row-store capability requires a tile column bound")
        if self.calibration_label != "resource_calibration_only":
            raise ValueError("provider profiles are resource calibration only")
        _nonnegative(
            "reference_replay_tile_cache_bytes",
            self.reference_replay_tile_cache_bytes,
        )
        for name, models in (
            ("phase_memory_models", self.phase_memory_models),
            ("phase_walltime_models", self.phase_walltime_models),
        ):
            phases = tuple(model.phase for model in models)
            if len(set(phases)) != len(phases):
                raise ValueError(f"{name} phases must be unique")

    @property
    def profile_fingerprint(self) -> str:
        return fingerprint({"schema_version": FINGERPRINT_SCHEMA_VERSION, "profile": self})


@dataclass(frozen=True)
class DemandEstimate:
    name: str
    tier: DemandTier
    demand_class: DemandClass
    lifetime: DemandLifetime
    amount: float
    unit: str = "bytes"

    def __post_init__(self) -> None:
        _nonempty("name", self.name)
        _nonnegative("amount", self.amount)
        if self.unit not in {"bytes", "seconds"}:
            raise ValueError("unit must be 'bytes' or 'seconds'")


@dataclass(frozen=True)
class PlanningProgress:
    """Observed prefix and completed phase set for a staged replan."""

    completed_phases: tuple[str, ...] = ()
    observed_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        phase_order = tuple(f"phase{index}" for index in range(6))
        if self.completed_phases != phase_order[: len(self.completed_phases)]:
            raise ValueError("completed_phases must be an ordered phase0..phase5 prefix")
        _nonnegative("observed_elapsed_seconds", self.observed_elapsed_seconds)


@dataclass(frozen=True)
class PhysicalExecutionConfig:
    decoder_fetch_chunk_size: int
    decoder_cache_bytes: int
    session_capacity: int
    phase1_source_batch_size: int
    source_microbatch_size: int
    feature_microbatch_size: int
    logit_microbatch_size: int
    replay_window: int
    prefetch_depth: int
    replay_tile_cache_bytes: int
    encoder_residency: str
    row_store_policy: str
    row_store_bytes: int
    spill_target: str | None
    cache_policy: CachePolicy

    def __post_init__(self) -> None:
        for name in (
            "decoder_fetch_chunk_size",
            "session_capacity",
            "phase1_source_batch_size",
            "source_microbatch_size",
            "feature_microbatch_size",
            "logit_microbatch_size",
            "replay_window",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "decoder_cache_bytes", "prefetch_depth", "replay_tile_cache_bytes",
            "row_store_bytes",
        ):
            _nonnegative(name, getattr(self, name))
        if self.row_store_policy != "recompute" and self.replay_tile_cache_bytes:
            raise ValueError("replay_tile_cache_bytes must be zero unless recomputing")
        if self.spill_target not in {None, "local", "scratch"}:
            raise ValueError("spill_target must be local, scratch, or None")


@dataclass(frozen=True)
class PhysicalExecutionRequirements:
    """Typed caller requirements that the planner must honor exactly or refuse."""

    decoder_fetch_chunk_size: int | None = None
    decoder_cache_bytes: int | None = None
    session_capacity: int | None = None
    phase1_source_batch_size: int | None = None
    source_microbatch_size: int | None = None
    feature_microbatch_size: int | None = None
    logit_microbatch_size: int | None = None
    replay_window: int | None = None
    prefetch_depth: int | None = None
    replay_tile_cache_bytes: int | None = None
    encoder_residency: EncoderResidency | None = None
    row_store_policy: RowStorePolicy | None = None
    spill_target: StorageTier | None = None
    cache_policy: CachePolicy | None = None

    def __post_init__(self) -> None:
        for name in (
            "decoder_fetch_chunk_size",
            "session_capacity",
            "phase1_source_batch_size",
            "source_microbatch_size",
            "feature_microbatch_size",
            "logit_microbatch_size",
            "replay_window",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        for name in ("decoder_cache_bytes", "prefetch_depth", "replay_tile_cache_bytes"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(name, value)
        for name, enum_type in (
            ("encoder_residency", EncoderResidency),
            ("row_store_policy", RowStorePolicy),
            ("spill_target", StorageTier),
            ("cache_policy", CachePolicy),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_type):
                raise ValueError(f"{name} must be a {enum_type.__name__}")
        if self.row_store_policy is RowStorePolicy.RECOMPUTE and self.spill_target is not None:
            raise ValueError("recompute row storage cannot require a spill target")
        if (
            self.row_store_policy is not None
            and self.row_store_policy is not RowStorePolicy.RECOMPUTE
            and self.replay_tile_cache_bytes not in (None, 0)
        ):
            raise ValueError("replay_tile_cache_bytes requires recompute row storage")


@dataclass(frozen=True)
class AdmissionReport:
    admitted: bool
    estimates: tuple[DemandEstimate, ...]
    trace_capacity: int
    binding_reasons: tuple[str, ...]
    effective_file_cache_allowance_bytes: int
    decisions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    candidate_count: int = 1
    admissible_candidate_count: int = 0
    hard_constraints: tuple[str, ...] = ()
    frozen_fields: tuple[str, ...] = ()
    free_fields: tuple[str, ...] = ()
    binding_constraints: tuple[str, ...] = ()
    selected_objective: tuple[tuple[str, float], ...] = ()
    rejected_candidates: tuple[str, ...] = ()
    confidence: str = "unknown"
    extrapolated_dimensions: tuple[str, ...] = ()
    calibration_evidence: tuple[str, ...] = ()
    phase_predictions: tuple[tuple[str, float, float], ...] = ()
    remaining_projection: tuple[tuple[str, float, float], ...] = ()
    domain_summary: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fidelity_prediction: FidelityPrediction | None = None
    fidelity_penalty: float = 0.0
    calibration_catalog_fingerprint: str | None = None
    pareto_alternatives: tuple[ParetoAlternative, ...] = ()

    def __post_init__(self) -> None:
        _positive("trace_capacity", self.trace_capacity)
        _nonnegative(
            "effective_file_cache_allowance_bytes", self.effective_file_cache_allowance_bytes
        )
        _positive("candidate_count", self.candidate_count)
        _nonnegative("admissible_candidate_count", self.admissible_candidate_count)
        if self.admissible_candidate_count > self.candidate_count:
            raise ValueError("admissible_candidate_count cannot exceed candidate_count")
        if not self.binding_reasons:
            raise ValueError("binding_reasons must not be empty")
        if self.admitted == bool(self.refusals):
            raise ValueError("admitted must be false exactly when refusals are present")
        if self.confidence not in {"calibrated", "provisional", "extrapolated", "unknown"}:
            raise ValueError("invalid admission confidence")
        _nonnegative("fidelity_penalty", self.fidelity_penalty)

    def format(self) -> str:
        lines = [
            f"admission: {'ADMIT (advisory)' if self.admitted else 'REFUSE (advisory)'}",
            "estimates:",
        ]
        for estimate in self.estimates:
            amount = (
                f"{int(estimate.amount)} B"
                if estimate.unit == "bytes"
                else f"{estimate.amount:.2f} s"
            )
            lines.append(
                f"  {estimate.name}: {amount} "
                f"[{estimate.tier.value}/{estimate.demand_class.value}/{estimate.lifetime.value}]"
            )
        lines.extend(
            (
                f"trace_capacity: {self.trace_capacity} (binding={','.join(self.binding_reasons)})",
                "effective_file_cache_allowance: "
                f"{self.effective_file_cache_allowance_bytes} B",
            )
        )
        for heading, values in (
            ("decisions", self.decisions),
            ("warnings", self.warnings),
            ("refusals", self.refusals),
        ):
            lines.append(f"{heading}:")
            lines.extend(f"  - {value}" for value in values)
            if not values:
                lines.append("  - none")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class TracePlan:
    semantics: TraceSemantics
    profile: ProviderProfile
    envelope: ResourceEnvelope
    physical: PhysicalExecutionConfig
    admission: AdmissionReport
    semantic_fingerprint: str
    execution_fingerprint: str
    evidence_fingerprint: str | None
    status: PlanStatus
    advisory: bool = True

    def __post_init__(self) -> None:
        if not self.advisory:
            raise ValueError("Phase B TracePlan outputs are advisory")
        expected = (
            PlanStatus.ADVISORY_ADMITTED
            if self.admission.admitted
            else PlanStatus.ADVISORY_REFUSED
        )
        if self.status is not expected:
            raise ValueError("status must agree with admission")

    def format(self) -> str:
        return "\n".join(
            (
                f"trace_plan: {self.status.value}",
                f"semantic_fingerprint: {self.semantic_fingerprint}",
                f"execution_fingerprint: {self.execution_fingerprint}",
                self.admission.format(),
            )
        )

    def __str__(self) -> str:
        return self.format()


def semantic_fingerprint(semantics: TraceSemantics, identity: ProviderIdentity) -> str:
    return fingerprint(
        {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "semantics": semantics,
            "provider_identity": identity,
        }
    )


def execution_fingerprint(
    *,
    profile: ProviderProfile,
    envelope: ResourceEnvelope,
    physical: PhysicalExecutionConfig,
    admission: AdmissionReport,
    evidence_fingerprint: str | None,
) -> str:
    return fingerprint(
        {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "profile_fingerprint": profile.profile_fingerprint,
            "profile_identity": profile.identity,
            "profile_version": profile.profile_version,
            "planner_version": profile.planner_version,
            "envelope": envelope,
            "physical": physical,
            "estimates": admission.estimates,
            "trace_capacity": admission.trace_capacity,
            "binding_reasons": admission.binding_reasons,
            "effective_file_cache_allowance_bytes": (
                admission.effective_file_cache_allowance_bytes
            ),
            "decisions": admission.decisions,
            "warnings": admission.warnings,
            "refusals": admission.refusals,
            "fidelity_prediction": admission.fidelity_prediction,
            "fidelity_penalty": admission.fidelity_penalty,
            "calibration_catalog_fingerprint": (
                admission.calibration_catalog_fingerprint
            ),
            "evidence_fingerprint": evidence_fingerprint,
        }
    )


def immutable_mapping(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))
