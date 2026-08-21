"""Provider-agnostic calibration data and response-model contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Protocol


class CalibrationSplit(str, Enum):
    FIT = "fit"
    HELDOUT = "heldout"
    REFERENCE = "reference"


class CalibrationOutcome(str, Enum):
    COMPLETED = "completed"
    OOM = "oom"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    NODE_FAIL = "node_fail"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CensoringKind(str, Enum):
    NONE = "none"
    FEASIBILITY = "feasibility"
    RUNTIME_LOWER_BOUND = "runtime_lower_bound"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    split: CalibrationSplit
    scope: tuple[tuple[str, str], ...]
    numeric_coordinates: tuple[tuple[str, float], ...] = ()
    categorical_coordinates: tuple[tuple[str, str], ...] = ()
    outcome: CalibrationOutcome = CalibrationOutcome.COMPLETED
    censoring: CensoringKind = CensoringKind.NONE
    targets: tuple[tuple[str, float], ...] = ()
    analytic_baselines: tuple[tuple[str, float], ...] = ()
    provenance_fingerprints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("calibration sample requires sample_id")
        for name, values in (
            ("scope", self.scope),
            ("numeric_coordinates", self.numeric_coordinates),
            ("categorical_coordinates", self.categorical_coordinates),
            ("targets", self.targets),
            ("analytic_baselines", self.analytic_baselines),
            ("provenance_fingerprints", self.provenance_fingerprints),
        ):
            if tuple(sorted(values)) != values or len({key for key, _ in values}) != len(values):
                raise ValueError(f"{name} must be sorted with unique keys")
        for name, value in self.numeric_coordinates + self.targets + self.analytic_baselines:
            if not name or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("numeric calibration values must be named and finite")
        if any(value <= 0 for _, value in self.analytic_baselines):
            raise ValueError("analytic baselines must be positive")
        if self.outcome is CalibrationOutcome.COMPLETED and self.censoring is not CensoringKind.NONE:
            raise ValueError("completed samples cannot be censored")
        if self.censoring is CensoringKind.NONE and self.outcome is not CalibrationOutcome.COMPLETED:
            raise ValueError("non-completed samples require censoring")

    def target(self, name: str) -> float | None:
        return dict(self.targets).get(name)


@dataclass(frozen=True)
class CalibrationDataset:
    samples: tuple[CalibrationSample, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.samples, key=lambda row: row.sample_id)) != self.samples:
            raise ValueError("calibration samples must be sorted by sample_id")
        if len({row.sample_id for row in self.samples}) != len(self.samples):
            raise ValueError("calibration sample ids must be unique")

    @property
    def fit_samples(self) -> tuple[CalibrationSample, ...]:
        return tuple(row for row in self.samples if row.split is CalibrationSplit.FIT)

    @property
    def heldout_samples(self) -> tuple[CalibrationSample, ...]:
        return tuple(row for row in self.samples if row.split is CalibrationSplit.HELDOUT)


@dataclass(frozen=True)
class FitSpec:
    model_kind: str
    target: str
    numeric_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    config: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ResponseFitConfig:
    models: tuple[FitSpec, ...]
    confidence: float = 0.95
    minimum_support: int = 3

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("response fit config requires at least one model")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if self.minimum_support < 1:
            raise ValueError("minimum_support must be positive")


@dataclass(frozen=True)
class ResponsePrediction:
    target: str
    lower: float
    estimate: float
    upper: float
    supported: bool
    method: str
    support_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelArtifact:
    model_kind: str
    model_version: int
    target: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    transforms: tuple[str, ...]
    uncertainty_method: str
    support_scope: tuple[tuple[str, str], ...]
    support_ids: tuple[str, ...]
    parameters: Mapping[str, Any] = field(repr=False)
    diagnostics: Mapping[str, Any] = field(default_factory=dict, repr=False)


class ModelFamily(Protocol):
    kind: str
    version: int

    def fit(
        self, samples: tuple[CalibrationSample, ...], spec: FitSpec, config: ResponseFitConfig
    ) -> ModelArtifact: ...

    def predict(
        self,
        artifact: ModelArtifact,
        *,
        numeric: Mapping[str, float],
        categorical: Mapping[str, str],
        scope: Mapping[str, str],
        analytic_baseline: float | None = None,
    ) -> ResponsePrediction: ...

    def validate_artifact(self, artifact: ModelArtifact) -> None: ...
