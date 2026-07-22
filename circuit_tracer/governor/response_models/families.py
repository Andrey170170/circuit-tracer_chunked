"""Small deterministic NumPy response-model families."""

from __future__ import annotations

from dataclasses import replace
import math
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from .contracts import (
    CalibrationOutcome,
    CalibrationSample,
    CensoringKind,
    FitSpec,
    ModelArtifact,
    ResponseFitConfig,
    ResponsePrediction,
)


def _support_scope(rows: tuple[CalibrationSample, ...]) -> tuple[tuple[str, str], ...]:
    if not rows:
        return ()
    common = dict(rows[0].scope)
    for row in rows[1:]:
        observed = dict(row.scope)
        common = {key: value for key, value in common.items() if observed.get(key) == value}
    return tuple(sorted(common.items()))


def _scope_supported(artifact: ModelArtifact, scope: Mapping[str, str]) -> bool:
    return all(scope.get(key) == value for key, value in artifact.support_scope)


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2)


class FeasibilityEnvelopeFamily:
    kind = "feasibility_envelope"
    version = 1

    def fit(self, samples: tuple[CalibrationSample, ...], spec: FitSpec, config: ResponseFitConfig) -> ModelArtifact:
        rows = tuple(
            row for row in samples
            if row.outcome in {
                CalibrationOutcome.COMPLETED,
                CalibrationOutcome.TIMEOUT,
                CalibrationOutcome.OOM,
                CalibrationOutcome.REFUSED,
            }
        )
        grouped: dict[
            tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
            list[CalibrationSample],
        ] = {}
        for row in rows:
            observed_categories = dict(row.categorical_coordinates)
            categories = tuple(
                (name, observed_categories.get(name, "<missing>"))
                for name in spec.categorical_features
            )
            grouped.setdefault((row.scope, categories), []).append(row)
        envelopes = []
        for (profile_scope, categories), group_rows in sorted(grouped.items()):
            limits: dict[str, dict[str, float | None]] = {}
            for feature in spec.numeric_features:
                feasible = [
                    dict(row.numeric_coordinates).get(feature)
                    for row in group_rows
                    if row.outcome
                    not in {CalibrationOutcome.OOM, CalibrationOutcome.REFUSED}
                ]
                failed = [
                    dict(row.numeric_coordinates).get(feature)
                    for row in group_rows
                    if row.outcome
                    in {CalibrationOutcome.OOM, CalibrationOutcome.REFUSED}
                ]
                feasible = [float(value) for value in feasible if value is not None]
                failed = [float(value) for value in failed if value is not None]
                limits[feature] = {
                    "max_feasible": max(feasible) if feasible else None,
                    "min_infeasible": min(failed) if failed else None,
                }
            envelopes.append(
                {
                    "scope": profile_scope,
                    "categorical": categories,
                    "limits": limits,
                    "support_ids": tuple(sorted(row.sample_id for row in group_rows)),
                }
            )
        return ModelArtifact(
            self.kind, self.version, spec.target, spec.numeric_features,
            spec.categorical_features, ("conservative_axis_envelope",),
            "deterministic_boundary", _support_scope(rows),
            tuple(sorted(row.sample_id for row in rows)),
            {"envelopes": envelopes, "minimum_support": config.minimum_support},
            {"fit_count": len(rows), "fallback": len(rows) < config.minimum_support},
        )

    def predict(self, artifact: ModelArtifact, *, numeric: Mapping[str, float], categorical: Mapping[str, str], scope: Mapping[str, str], analytic_baseline: float | None = None) -> ResponsePrediction:
        del analytic_baseline
        matches = [
            envelope
            for envelope in artifact.parameters["envelopes"]
            if all(scope.get(key) == value for key, value in envelope["scope"])
            and all(
                categorical.get(key, "<missing>") == value
                for key, value in envelope["categorical"]
            )
        ]
        if not matches:
            return ResponsePrediction(
                artifact.target, 0.0, 0.0, 1.0, False, "no_matching_profile"
            )
        envelope = max(
            matches,
            key=lambda item: (
                len(item["scope"]),
                len(item["categorical"]),
                tuple(item["scope"]),
                tuple(item["categorical"]),
            ),
        )
        support_ids = tuple(envelope["support_ids"])
        supported = len(support_ids) >= int(artifact.parameters["minimum_support"])
        feasible = supported
        for feature, raw in envelope["limits"].items():
            value = numeric.get(feature)
            if value is None:
                supported = feasible = False
                break
            maximum = raw["max_feasible"]
            failure = raw["min_infeasible"]
            if maximum is not None and value > maximum:
                feasible = False
            if failure is not None and value >= failure:
                feasible = False
        estimate = 1.0 if feasible else 0.0
        lower = (
            estimate
            if not feasible
            else max(0.0, 1.0 - 1.0 / max(1, len(support_ids)))
        )
        return ResponsePrediction(
            artifact.target,
            lower,
            estimate,
            estimate,
            supported,
            "conservative_axis_envelope",
            support_ids,
        )

    def validate_artifact(self, artifact: ModelArtifact) -> None:
        if artifact.target != "feasible":
            raise ValueError("feasibility artifact target must be 'feasible'")
        if "envelopes" not in artifact.parameters:
            raise ValueError("feasibility artifact lacks scoped envelopes")


class PositiveLogRatioFamily:
    kind = "positive_log_ratio_ridge"
    version = 1

    def fit(self, samples: tuple[CalibrationSample, ...], spec: FitSpec, config: ResponseFitConfig) -> ModelArtifact:
        completed_rows = tuple(
            row for row in samples
            if row.outcome is CalibrationOutcome.COMPLETED
            and row.censoring is CensoringKind.NONE
            and row.target(spec.target) is not None
            and row.target(spec.target) > 0
            and dict(row.analytic_baselines).get(spec.target, 0) > 0
        )
        lower_bound_rows = tuple(
            row
            for row in samples
            if row.outcome is CalibrationOutcome.TIMEOUT
            and row.censoring is CensoringKind.RUNTIME_LOWER_BOUND
            and row.target(spec.target) is not None
            and row.target(spec.target) > 0
            and dict(row.analytic_baselines).get(spec.target, 0) > 0
        )
        rows = completed_rows + lower_bound_rows
        categories = {
            name: tuple(sorted({dict(row.categorical_coordinates).get(name, "<missing>") for row in rows}))
            for name in spec.categorical_features
        }
        means = {name: float(np.mean([dict(row.numeric_coordinates).get(name, 0.0) for row in rows])) if rows else 0.0 for name in spec.numeric_features}
        scales = {
            name: max(float(np.std([dict(row.numeric_coordinates).get(name, 0.0) for row in rows])), 1.0)
            if rows else 1.0 for name in spec.numeric_features
        }
        feature_schema = ("intercept",) + spec.numeric_features + tuple(
            f"{name}={value}" for name in spec.categorical_features for value in categories[name][1:]
        )
        matrix = np.asarray(
            [
                self._vector(row, spec, categories, means, scales)
                for row in completed_rows
            ],
            dtype=float,
        )
        response = np.asarray([
            math.log(float(row.target(spec.target)) / dict(row.analytic_baselines)[spec.target]) for row in completed_rows
        ], dtype=float)
        alpha = float(dict(spec.config).get("ridge_alpha", 1e-3))
        fallback = len(rows) < config.minimum_support or not len(completed_rows)
        if fallback:
            ratio = float(np.median(np.exp(response))) if len(response) else 1.0
            coefficients = np.zeros(len(feature_schema), dtype=float)
            coefficients[0] = math.log(ratio)
        else:
            penalty = np.eye(matrix.shape[1]) * alpha
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ response)
        residuals = (
            response - matrix @ coefficients
            if len(completed_rows)
            else np.asarray([], dtype=float)
        )
        residual_scale = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
        lower_bound_shortfalls = [
            math.log(
                float(row.target(spec.target))
                / dict(row.analytic_baselines)[spec.target]
            )
            - float(
                np.dot(
                    np.asarray(self._vector(row, spec, categories, means, scales)),
                    coefficients,
                )
            )
            for row in lower_bound_rows
        ]
        lower_bound_delta = max((0.0, *lower_bound_shortfalls))
        return ModelArtifact(
            self.kind, self.version, spec.target, spec.numeric_features, spec.categorical_features,
            ("standardize_numeric", "one_hot_categorical", "log_target_over_analytic_baseline", "ridge"),
            "normal_residual_log_ratio", _support_scope(rows), tuple(sorted(row.sample_id for row in rows)),
            {
                "feature_schema": feature_schema, "categories": categories, "means": means,
                "scales": scales, "coefficients": coefficients.tolist(), "ridge_alpha": alpha,
                "residual_scale": residual_scale, "confidence": config.confidence,
                "minimum_support": config.minimum_support,
                "runtime_lower_bound_delta": lower_bound_delta,
            },
            {
                "fit_count": len(completed_rows),
                "runtime_lower_bound_count": len(lower_bound_rows),
                "fallback": fallback,
                "rmse_log_ratio": residual_scale,
            },
        )

    @staticmethod
    def _vector(row: CalibrationSample, spec: FitSpec, categories: Mapping[str, tuple[str, ...]], means: Mapping[str, float], scales: Mapping[str, float]) -> list[float]:
        numeric = dict(row.numeric_coordinates)
        categorical = dict(row.categorical_coordinates)
        values = [1.0]
        values.extend((float(numeric.get(name, 0.0)) - means[name]) / scales[name] for name in spec.numeric_features)
        values.extend(1.0 if categorical.get(name, "<missing>") == value else 0.0 for name in spec.categorical_features for value in categories[name][1:])
        return values

    def predict(self, artifact: ModelArtifact, *, numeric: Mapping[str, float], categorical: Mapping[str, str], scope: Mapping[str, str], analytic_baseline: float | None = None) -> ResponsePrediction:
        if analytic_baseline is None or analytic_baseline <= 0:
            return ResponsePrediction(artifact.target, 0.0, 0.0, 0.0, False, "analytic_fallback")
        p = artifact.parameters
        known = all(name in numeric for name in artifact.numeric_features) and all(
            categorical.get(name, "<missing>") in tuple(p["categories"][name]) for name in artifact.categorical_features
        )
        supported = _scope_supported(artifact, scope) and known and len(artifact.support_ids) >= int(p["minimum_support"])
        values = [1.0]
        values.extend((numeric[name] - float(p["means"][name])) / float(p["scales"][name]) for name in artifact.numeric_features)
        values.extend(1.0 if categorical.get(name, "<missing>") == value else 0.0 for name in artifact.categorical_features for value in tuple(p["categories"][name])[1:])
        correction = math.exp(float(np.dot(np.asarray(values), np.asarray(p["coefficients"]))))
        estimate = analytic_baseline * correction
        residual_delta = _z(float(p["confidence"])) * float(p["residual_scale"])
        upper_delta = max(
            residual_delta, float(p.get("runtime_lower_bound_delta", 0.0))
        )
        return ResponsePrediction(
            artifact.target,
            analytic_baseline * math.exp(math.log(correction) - residual_delta),
            estimate,
            analytic_baseline * math.exp(math.log(correction) + upper_delta),
            supported,
            "ridge_log_ratio" if supported else "deterministic_fallback",
            artifact.support_ids,
        )

    def validate_artifact(self, artifact: ModelArtifact) -> None:
        required = {"feature_schema", "categories", "means", "scales", "coefficients", "confidence"}
        if not required <= set(artifact.parameters):
            raise ValueError("positive response artifact is incomplete")
        if len(artifact.parameters["feature_schema"]) != len(artifact.parameters["coefficients"]):
            raise ValueError("positive response feature schema and coefficients differ")


class LocalFidelityFamily:
    kind = "local_conservative_fidelity"
    version = 1

    def fit(self, samples: tuple[CalibrationSample, ...], spec: FitSpec, config: ResponseFitConfig) -> ModelArtifact:
        rows = tuple(row for row in samples if row.outcome is CalibrationOutcome.COMPLETED and row.censoring is CensoringKind.NONE and row.target(spec.target) is not None)
        points = [
            {
                "sample_id": row.sample_id,
                "numeric": dict(row.numeric_coordinates),
                "categorical": dict(row.categorical_coordinates),
                "value": row.target(spec.target),
            }
            for row in rows
        ]
        return ModelArtifact(
            self.kind, self.version, spec.target, spec.numeric_features, spec.categorical_features,
            ("normalized_manhattan_distance", "nearest_neighbor"), "nearest_neighbor_distance_inflation",
            _support_scope(rows), tuple(sorted(row.sample_id for row in rows)),
            {"points": points, "confidence": config.confidence, "minimum_support": config.minimum_support},
            {"fit_count": len(rows), "fallback": len(rows) < config.minimum_support},
        )

    def predict(self, artifact: ModelArtifact, *, numeric: Mapping[str, float], categorical: Mapping[str, str], scope: Mapping[str, str], analytic_baseline: float | None = None) -> ResponsePrediction:
        del analytic_baseline
        points = artifact.parameters["points"]
        if not points:
            return ResponsePrediction(artifact.target, 0.0, 0.0, 1.0, False, "no_support")
        ranges = {
            name: max(1.0, max(float(point["numeric"].get(name, 0.0)) for point in points) - min(float(point["numeric"].get(name, 0.0)) for point in points))
            for name in artifact.numeric_features
        }
        def distance(point: Mapping[str, Any]) -> tuple[float, str]:
            numeric_distance = sum(abs(numeric.get(name, 0.0) - float(point["numeric"].get(name, 0.0))) / ranges[name] for name in artifact.numeric_features)
            categorical_distance = sum(categorical.get(name) != point["categorical"].get(name) for name in artifact.categorical_features)
            count = max(1, len(artifact.numeric_features) + len(artifact.categorical_features))
            return ((numeric_distance + categorical_distance) / count, str(point["sample_id"]))
        nearest = min(points, key=distance)
        dist = distance(nearest)[0]
        estimate = float(nearest["value"])
        inflation = dist * max(0.05, abs(estimate))
        supported = _scope_supported(artifact, scope) and len(points) >= int(artifact.parameters["minimum_support"])
        return ResponsePrediction(artifact.target, max(0.0, estimate - inflation), estimate, min(1.0, estimate + inflation), supported, "nearest_distance_inflation" if supported else "deterministic_fallback", (str(nearest["sample_id"]),))

    def validate_artifact(self, artifact: ModelArtifact) -> None:
        if "points" not in artifact.parameters:
            raise ValueError("local fidelity artifact lacks points")


def with_diagnostics(artifact: ModelArtifact, diagnostics: Mapping[str, Any]) -> ModelArtifact:
    return replace(artifact, diagnostics={**artifact.diagnostics, **diagnostics})
