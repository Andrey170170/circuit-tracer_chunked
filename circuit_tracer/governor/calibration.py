"""Typed fidelity calibration evidence and conservative nearest prediction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math


class SensitivityClass(str, Enum):
    EXACT = "exact"
    NUMERICALLY_SENSITIVE = "numerically_sensitive"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class KnobSensitivity:
    knob: str
    classification: SensitivityClass
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.knob:
            raise ValueError("knob sensitivity requires a knob name")
        if tuple(sorted(set(self.metrics))) != self.metrics:
            raise ValueError("knob sensitivity metrics must be sorted and unique")
        if self.classification is not SensitivityClass.EXACT and not self.metrics:
            raise ValueError("sensitive knobs require affected metrics")


_GRAPH_METRICS = ("compact_graph_similarity", "frontier_identity")

DEFAULT_KNOB_SENSITIVITIES = tuple(
    sorted(
        (
            KnobSensitivity("decoder_cache_bytes", SensitivityClass.EXACT),
            KnobSensitivity(
                "decoder_fetch_chunk_size",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity("encoder_residency", SensitivityClass.EXACT),
            KnobSensitivity(
                "feature_microbatch_size",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity(
                "logit_microbatch_size",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity("prefetch_depth", SensitivityClass.EXACT),
            KnobSensitivity(
                "phase1_source_batch_size",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity("replay_tile_cache_bytes", SensitivityClass.EXACT),
            KnobSensitivity("replay_window", SensitivityClass.EXACT),
            KnobSensitivity("row_store_policy", SensitivityClass.EXACT),
            KnobSensitivity(
                "session_capacity",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity(
                "source_microbatch_size",
                SensitivityClass.NUMERICALLY_SENSITIVE,
                _GRAPH_METRICS,
            ),
            KnobSensitivity(
                "decoder_reduction_tile",
                SensitivityClass.SEMANTIC,
                _GRAPH_METRICS,
            ),
            KnobSensitivity("dtype", SensitivityClass.SEMANTIC, _GRAPH_METRICS),
            KnobSensitivity(
                "feature_batch_size", SensitivityClass.SEMANTIC, _GRAPH_METRICS
            ),
            KnobSensitivity(
                "frontier_refresh_stride",
                SensitivityClass.SEMANTIC,
                _GRAPH_METRICS,
            ),
            KnobSensitivity(
                "logit_batch_size", SensitivityClass.SEMANTIC, _GRAPH_METRICS
            ),
            KnobSensitivity(
                "source_batch_size", SensitivityClass.SEMANTIC, _GRAPH_METRICS
            ),
        ),
        key=lambda item: item.knob,
    )
)


@dataclass(frozen=True)
class FidelityBudget:
    """Conservative minimum acceptable values for higher-is-better metrics."""

    metric_floors: tuple[tuple[str, float], ...] = ()
    allowed_sensitive_axes: tuple[str, ...] = ()
    confidence: float = 0.95
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.metric_floors and not self.allowed_sensitive_axes:
            raise ValueError("fidelity budget requires metric floors or sensitive-axis allowances")
        if tuple(sorted(self.metric_floors)) != self.metric_floors:
            raise ValueError("fidelity metric floors must be sorted")
        if len({name for name, _ in self.metric_floors}) != len(self.metric_floors):
            raise ValueError("fidelity metric floors must have unique names")
        for name, value in self.metric_floors:
            if not name or not math.isfinite(value):
                raise ValueError("fidelity metric floors must be named and finite")
        if tuple(sorted(set(self.allowed_sensitive_axes))) != self.allowed_sensitive_axes:
            raise ValueError("allowed sensitive axes must be sorted, unique, and nonempty")
        if any(not axis for axis in self.allowed_sensitive_axes):
            raise ValueError("allowed sensitive axes must be sorted, unique, and nonempty")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("fidelity confidence must be in (0, 1]")
        if not math.isfinite(self.penalty_weight) or self.penalty_weight < 0.0:
            raise ValueError("fidelity penalty weight must be finite and nonnegative")


@dataclass(frozen=True)
class MetricPrediction:
    name: str
    lower: float
    estimate: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name or not all(math.isfinite(v) for v in (self.lower, self.estimate, self.upper)):
            raise ValueError("metric predictions must be named and finite")
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("metric prediction interval must be ordered")


class PredictionSupportKind(str, Enum):
    EXACT = "exact"
    NEAREST = "nearest"
    EXTRAPOLATED = "extrapolated"
    NONE = "none"


@dataclass(frozen=True)
class PredictionSupport:
    kind: PredictionSupportKind
    observation_id: str | None
    normalized_distance: float | None
    certified_exact: bool
    extrapolated_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionUncertainty:
    source_interval_widths: tuple[tuple[str, float], ...]
    distance_inflation: float | None


@dataclass(frozen=True)
class FidelityPrediction:
    metrics: tuple[MetricPrediction, ...]
    support: PredictionSupport
    uncertainty: PredictionUncertainty

    def lower_bound(self, metric: str) -> float | None:
        return next((item.lower for item in self.metrics if item.name == metric), None)


@dataclass(frozen=True)
class CalibrationObservation:
    """Normalized evidence point; coordinates and metrics are sorted canonical tuples."""

    observation_id: str
    coordinates: tuple[tuple[str, str], ...]
    metrics: tuple[MetricPrediction, ...]
    certified_exact: bool = False

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("calibration observation requires an id")
        if tuple(sorted(self.coordinates)) != self.coordinates or len(
            {name for name, _ in self.coordinates}
        ) != len(self.coordinates):
            raise ValueError("calibration coordinates must be sorted with unique names")
        if tuple(sorted(self.metrics, key=lambda item: item.name)) != self.metrics or len(
            {item.name for item in self.metrics}
        ) != len(self.metrics):
            raise ValueError("calibration metrics must be sorted with unique names")


@dataclass(frozen=True)
class CalibrationCatalog:
    observations: tuple[CalibrationObservation, ...]
    sensitivities: tuple[KnobSensitivity, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.observations, key=lambda item: item.observation_id)) != self.observations:
            raise ValueError("calibration observations must be sorted by id")
        if len({item.observation_id for item in self.observations}) != len(self.observations):
            raise ValueError("calibration observation ids must be unique")
        if tuple(sorted(self.sensitivities, key=lambda item: item.knob)) != self.sensitivities:
            raise ValueError("knob sensitivities must be sorted by knob")
        if len({item.knob for item in self.sensitivities}) != len(self.sensitivities):
            raise ValueError("knob sensitivities must have unique knobs")

    @property
    def sensitive_axes(self) -> tuple[str, ...]:
        return tuple(
            item.knob
            for item in self.sensitivities
            if item.classification is not SensitivityClass.EXACT
        )

    @property
    def content_fingerprint(self) -> str:
        payload = {
            "schema_version": 1,
            "observations": [
                {
                    "observation_id": item.observation_id,
                    "coordinates": item.coordinates,
                    "metrics": [
                        (metric.name, metric.lower, metric.estimate, metric.upper)
                        for metric in item.metrics
                    ],
                    "certified_exact": item.certified_exact,
                }
                for item in self.observations
            ],
            "sensitivities": [
                (item.knob, item.classification.value, item.metrics)
                for item in self.sensitivities
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def predict(self, coordinates: tuple[tuple[str, str], ...]) -> FidelityPrediction:
        requested = dict(coordinates)
        if not self.observations:
            return FidelityPrediction(
                metrics=(),
                support=PredictionSupport(PredictionSupportKind.NONE, None, None, False),
                uncertainty=PredictionUncertainty((), None),
            )

        def distance(observation: CalibrationObservation) -> tuple[float, str]:
            observed = dict(observation.coordinates)
            axes = set(requested) | set(observed)
            mismatches = sum(requested.get(axis) != observed.get(axis) for axis in axes)
            return (mismatches / max(1, len(axes)), observation.observation_id)

        nearest = min(self.observations, key=distance)
        normalized_distance = distance(nearest)[0]
        observed = dict(nearest.coordinates)
        extrapolated = tuple(
            axis for axis in self.sensitive_axes if requested.get(axis) != observed.get(axis)
        )
        if normalized_distance == 0:
            kind = PredictionSupportKind.EXACT
        elif extrapolated:
            kind = PredictionSupportKind.EXTRAPOLATED
        else:
            kind = PredictionSupportKind.NEAREST
        inflation = normalized_distance
        metrics = tuple(
            MetricPrediction(
                item.name,
                item.lower - inflation * max(1.0, abs(item.estimate)),
                item.estimate,
                item.upper + inflation * max(1.0, abs(item.estimate)),
            )
            for item in nearest.metrics
        )
        return FidelityPrediction(
            metrics=metrics,
            support=PredictionSupport(
                kind=kind,
                observation_id=nearest.observation_id,
                normalized_distance=normalized_distance,
                certified_exact=(nearest.certified_exact and not extrapolated),
                extrapolated_axes=extrapolated,
            ),
            uncertainty=PredictionUncertainty(
                tuple((item.name, item.upper - item.lower) for item in metrics),
                inflation,
            ),
        )


@dataclass(frozen=True)
class ParetoAlternative:
    execution_fingerprint: str
    fidelity_penalty: float
    walltime_high_seconds: float
    peak_vram_bytes: float
    host_bytes: float
    row_store_bytes: float
