"""Fit, evaluate, serialize, and validate immutable response bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from circuit_tracer.governor.contracts import canonical_json

from .contracts import (
    CalibrationDataset, CalibrationOutcome, CalibrationSample, CensoringKind,
    ModelArtifact, ResponseFitConfig, ResponsePrediction,
)
from .families import (
    FeasibilityEnvelopeFamily, LocalFidelityFamily, PositiveLogRatioFamily,
    with_diagnostics,
)
from .registry import ModelRegistry


RESPONSE_BUNDLE_SCHEMA_VERSION = 1


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(FeasibilityEnvelopeFamily())
    registry.register(PositiveLogRatioFamily())
    registry.register(LocalFidelityFamily())
    return registry


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _artifact_payload(artifact: ModelArtifact) -> dict[str, Any]:
    return {
        "model_kind": artifact.model_kind,
        "model_version": artifact.model_version,
        "target": artifact.target,
        "numeric_features": artifact.numeric_features,
        "categorical_features": artifact.categorical_features,
        "transforms": artifact.transforms,
        "uncertainty_method": artifact.uncertainty_method,
        "support_scope": artifact.support_scope,
        "support_ids": artifact.support_ids,
        "parameters": artifact.parameters,
        "diagnostics": artifact.diagnostics,
    }


def _fit_config_payload(config: ResponseFitConfig) -> dict[str, Any]:
    return {
        "models": [asdict(spec) for spec in config.models],
        "confidence": config.confidence,
        "minimum_support": config.minimum_support,
    }


@dataclass(frozen=True)
class ResponseBundle:
    schema_version: int
    observation_set_hash: str
    fit_config: Mapping[str, Any]
    models: tuple[ModelArtifact, ...]
    support_scope: tuple[tuple[str, str], ...]
    diagnostics: Mapping[str, Any]
    content_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported response bundle schema {self.schema_version}")
        if tuple(sorted(self.models, key=lambda item: (item.target, item.model_kind))) != self.models:
            raise ValueError("response bundle models must be sorted by target and kind")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_set_hash": self.observation_set_hash,
            "fit_config": self.fit_config,
            "models": [_artifact_payload(model) for model in self.models],
            "support_scope": self.support_scope,
            "diagnostics": self.diagnostics,
        }

    def verify(self, registry: ModelRegistry | None = None) -> None:
        registry = registry or default_registry()
        if _hash(self.unsigned_payload()) != self.content_fingerprint:
            raise ValueError("response bundle content fingerprint mismatch")
        for artifact in self.models:
            family = registry.get(artifact.model_kind)
            if family.version != artifact.model_version:
                raise ValueError(
                    f"unsupported {artifact.model_kind} artifact version {artifact.model_version}"
                )
            family.validate_artifact(artifact)

    def to_json(self) -> str:
        self.verify()
        return canonical_json({**self.unsigned_payload(), "content_fingerprint": self.content_fingerprint}) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def predict(
        self,
        target: str,
        *,
        numeric: Mapping[str, float],
        categorical: Mapping[str, str],
        scope: Mapping[str, str],
        analytic_baseline: float | None = None,
        registry: ModelRegistry | None = None,
    ) -> ResponsePrediction | None:
        registry = registry or default_registry()
        artifact = next((item for item in self.models if item.target == target), None)
        if artifact is None:
            return None
        return registry.get(artifact.model_kind).predict(
            artifact, numeric=numeric, categorical=categorical, scope=scope,
            analytic_baseline=analytic_baseline,
        )


def _dataset_payload(dataset: CalibrationDataset) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in dataset.samples]


def _heldout_diagnostics(
    artifact: ModelArtifact,
    rows: tuple[CalibrationSample, ...],
    registry: ModelRegistry,
) -> dict[str, Any]:
    family = registry.get(artifact.model_kind)
    eligible = tuple(
        row for row in rows
        if row.outcome is CalibrationOutcome.COMPLETED
        and row.censoring is CensoringKind.NONE
        and row.target(artifact.target) is not None
    )
    errors: list[float] = []
    covered = 0
    for row in eligible:
        baseline = dict(row.analytic_baselines).get(artifact.target)
        prediction = family.predict(
            artifact,
            numeric=dict(row.numeric_coordinates),
            categorical=dict(row.categorical_coordinates),
            scope=dict(row.scope),
            analytic_baseline=baseline,
        )
        actual = float(row.target(artifact.target))
        errors.append(abs(prediction.estimate - actual))
        covered += prediction.lower <= actual <= prediction.upper
    return {
        "heldout_count": len(eligible),
        "heldout_mae": sum(errors) / len(errors) if errors else None,
        "heldout_interval_coverage": covered / len(eligible) if eligible else None,
    }


def fit_response_bundle(
    dataset: CalibrationDataset,
    config: ResponseFitConfig,
    *,
    registry: ModelRegistry | None = None,
) -> ResponseBundle:
    registry = registry or default_registry()
    artifacts = []
    for spec in config.models:
        family = registry.get(spec.model_kind)
        artifact = family.fit(dataset.fit_samples, spec, config)
        artifacts.append(
            with_diagnostics(
                artifact, _heldout_diagnostics(artifact, dataset.heldout_samples, registry)
            )
        )
    models = tuple(sorted(artifacts, key=lambda item: (item.target, item.model_kind)))
    scope = ()
    if models:
        common = dict(models[0].support_scope)
        for model in models[1:]:
            observed = dict(model.support_scope)
            common = {key: value for key, value in common.items() if observed.get(key) == value}
        scope = tuple(sorted(common.items()))
    diagnostics = {
        "fit_sample_count": len(dataset.fit_samples),
        "heldout_sample_count": len(dataset.heldout_samples),
        "heldout_ids": tuple(row.sample_id for row in dataset.heldout_samples),
    }
    unsigned = {
        "schema_version": RESPONSE_BUNDLE_SCHEMA_VERSION,
        "observation_set_hash": _hash(_dataset_payload(dataset)),
        "fit_config": _fit_config_payload(config),
        "models": [_artifact_payload(model) for model in models],
        "support_scope": scope,
        "diagnostics": diagnostics,
    }
    bundle = ResponseBundle(
        RESPONSE_BUNDLE_SCHEMA_VERSION, unsigned["observation_set_hash"],
        unsigned["fit_config"], models, scope, diagnostics, _hash(unsigned),
    )
    bundle.verify(registry)
    return bundle


def load_response_bundle(path: Path, *, registry: ModelRegistry | None = None) -> ResponseBundle:
    registry = registry or default_registry()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "observation_set_hash", "fit_config", "models",
        "support_scope", "diagnostics", "content_fingerprint",
    }
    if set(raw) != required:
        raise ValueError("response bundle has unexpected or missing top-level fields")
    models = tuple(
        ModelArtifact(
            model_kind=item["model_kind"], model_version=int(item["model_version"]),
            target=item["target"], numeric_features=tuple(item["numeric_features"]),
            categorical_features=tuple(item["categorical_features"]),
            transforms=tuple(item["transforms"]), uncertainty_method=item["uncertainty_method"],
            support_scope=tuple(tuple(pair) for pair in item["support_scope"]),
            support_ids=tuple(item["support_ids"]), parameters=item["parameters"],
            diagnostics=item["diagnostics"],
        )
        for item in raw["models"]
    )
    bundle = ResponseBundle(
        schema_version=int(raw["schema_version"]),
        observation_set_hash=raw["observation_set_hash"], fit_config=raw["fit_config"],
        models=models, support_scope=tuple(tuple(pair) for pair in raw["support_scope"]),
        diagnostics=raw["diagnostics"], content_fingerprint=raw["content_fingerprint"],
    )
    bundle.verify(registry)
    return bundle
