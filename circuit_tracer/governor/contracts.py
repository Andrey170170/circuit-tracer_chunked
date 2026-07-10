from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar


FINGERPRINT_SCHEMA_VERSION = 2
_DTYPE_BYTE_WIDTHS = MappingProxyType({"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8})
_K = TypeVar("_K")
_V = TypeVar("_V")


class FidelityMode(str, Enum):
    STRICT = "strict"
    VALIDATED_RELAXED = "validated_relaxed"
    RESEARCH = "research"


class CachePolicy(str, Enum):
    AUTO = "auto"
    WARM = "warm"
    BOUNDED = "bounded"
    STREAMING = "streaming"


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
    fidelity: FidelityMode = FidelityMode.STRICT
    evidence_name: str | None = None
    evidence_version: str | None = None
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
                "evidence_name",
                "evidence_version",
                "semantic_overrides",
                "research_overrides",
            }:
                raise ValueError(f"unknown or non-semantic override field: {override_name}")
            expected = canonical_json(getattr(self, override_name))
            if override_value != expected:
                raise ValueError(
                    f"override {override_name} must equal its canonical requested value {expected}"
                )
        if self.fidelity is FidelityMode.STRICT:
            if any(
                (
                    self.evidence_name,
                    self.evidence_version,
                    self.semantic_overrides,
                    self.research_overrides,
                )
            ):
                raise ValueError("strict fidelity does not accept evidence or semantic overrides")
        elif self.fidelity is FidelityMode.VALIDATED_RELAXED:
            if not (self.evidence_name and self.evidence_version and self.semantic_overrides):
                raise ValueError(
                    "validated_relaxed requires named, versioned evidence and exact overrides"
                )
            if self.research_overrides:
                raise ValueError("validated_relaxed does not accept research_overrides")
        elif not self.research_overrides:
            raise ValueError("research fidelity requires explicit research_overrides")
        elif self.evidence_name or self.evidence_version or self.semantic_overrides:
            raise ValueError("research fidelity does not accept validation evidence")

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
    physical_overrides: tuple[tuple[str, int | str], ...] = ()

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
        keys = [name for name, _ in self.physical_overrides]
        if tuple(sorted(self.physical_overrides)) != self.physical_overrides or len(set(keys)) != len(keys):
            raise ValueError("physical_overrides must be sorted with unique keys")

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
    max_physical_microbatch: int
    default_decoder_cache_bytes: int
    max_decoder_cache_bytes: int
    default_replay_window: int
    max_replay_window: int
    default_prefetch_depth: int
    max_prefetch_depth: int
    row_store_tile_column_bound: int | None = None
    calibration_label: str = "resource_calibration_only"

    def __post_init__(self) -> None:
        for name in ("profile_name", "profile_version", "planner_version"):
            _nonempty(name, getattr(self, name))
        for name in (
            "default_fetch_chunk_size",
            "max_fetch_chunk_size",
            "max_physical_microbatch",
            "default_replay_window",
            "max_replay_window",
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

    @property
    def profile_fingerprint(self) -> str:
        return fingerprint({"schema_version": FINGERPRINT_SCHEMA_VERSION, "profile": self})


@dataclass(frozen=True)
class ValidationEvidence:
    evidence_id: str
    evidence_version: str
    provider_type: str
    provider_version: str
    checkpoint_identity: str
    hook_identity: str
    architecture: str
    decoder_topology: DecoderTopology
    provider_approximation: str
    provider_semantic_parameters: tuple[tuple[str, str], ...]
    semantic_parameters: tuple[tuple[str, str], ...]
    dtype: str
    scenario_id: str
    window_id: str | None
    environment_label: str
    allowed_semantic_overrides: tuple[tuple[str, str], ...]
    source_artifact_fingerprints: tuple[str, ...]
    report_fingerprint: str
    compared_configurations: tuple[tuple[str, str], ...]
    metrics: tuple[tuple[str, float], ...]
    acceptance_thresholds: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "evidence_version",
            "provider_type",
            "provider_version",
            "checkpoint_identity",
            "hook_identity",
            "architecture",
            "provider_approximation",
            "scenario_id",
            "environment_label",
            "report_fingerprint",
        ):
            _nonempty(name, getattr(self, name))
        dtype_byte_width(self.dtype)
        _sorted_unique("provider_semantic_parameters", self.provider_semantic_parameters)
        _sorted_unique("semantic_parameters", self.semantic_parameters)
        _sorted_unique("allowed_semantic_overrides", self.allowed_semantic_overrides)
        _sorted_unique("compared_configurations", self.compared_configurations)
        _sorted_unique_metrics("metrics", self.metrics)
        _sorted_unique_metrics("acceptance_thresholds", self.acceptance_thresholds)
        if not self.allowed_semantic_overrides:
            raise ValueError("validation evidence must allow at least one exact semantic override")
        if not self.source_artifact_fingerprints or any(
            not value for value in self.source_artifact_fingerprints
        ):
            raise ValueError("validation evidence requires source artifact fingerprints")
        if tuple(sorted(set(self.source_artifact_fingerprints))) != self.source_artifact_fingerprints:
            raise ValueError("source_artifact_fingerprints must be sorted and unique")
        if not self.compared_configurations or not self.metrics or not self.acceptance_thresholds:
            raise ValueError(
                "validation evidence requires configurations, metrics, and acceptance thresholds"
            )

    @property
    def evidence_fingerprint(self) -> str:
        return fingerprint({"schema_version": 1, "validation_evidence": self})


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
class PhysicalExecutionConfig:
    decoder_fetch_chunk_size: int
    decoder_cache_bytes: int
    source_microbatch_size: int
    feature_microbatch_size: int
    logit_microbatch_size: int
    replay_window: int
    prefetch_depth: int
    encoder_residency: str
    row_store_policy: str
    row_store_bytes: int
    spill_target: str | None
    cache_policy: CachePolicy

    def __post_init__(self) -> None:
        for name in (
            "decoder_fetch_chunk_size",
            "source_microbatch_size",
            "feature_microbatch_size",
            "logit_microbatch_size",
            "replay_window",
        ):
            _positive(name, getattr(self, name))
        for name in ("decoder_cache_bytes", "prefetch_depth", "row_store_bytes"):
            _nonnegative(name, getattr(self, name))
        if self.spill_target not in {None, "local", "scratch"}:
            raise ValueError("spill_target must be local, scratch, or None")


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

    def __post_init__(self) -> None:
        _positive("trace_capacity", self.trace_capacity)
        _nonnegative(
            "effective_file_cache_allowance_bytes", self.effective_file_cache_allowance_bytes
        )
        if not self.binding_reasons:
            raise ValueError("binding_reasons must not be empty")
        if self.admitted == bool(self.refusals):
            raise ValueError("admitted must be false exactly when refusals are present")

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
            "evidence_fingerprint": evidence_fingerprint,
        }
    )


def immutable_mapping(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))


TRUSTED_VALIDATION_EVIDENCE_REGISTRY: Mapping[
    tuple[str, str], ValidationEvidence
] = immutable_mapping({})
