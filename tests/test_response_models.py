from __future__ import annotations

from dataclasses import replace
import json

import pytest

from circuit_tracer.governor.response_models import (
    CalibrationDataset,
    CalibrationOutcome,
    CalibrationSample,
    CalibrationSplit,
    CensoringKind,
    FitSpec,
    ModelRegistry,
    ResponseFitConfig,
    default_registry,
    fit_response_bundle,
    load_response_bundle,
)
from circuit_tracer.governor.response_models.families import PositiveLogRatioFamily
from circuit_tracer.governor import FidelityBudget, FidelityMode, resolve_trace_plan
import circuit_tracer.governor.resolver as resolver_module

from tests.test_governor_resolver import _envelope, _profile, _semantics


def _sample(
    sample_id: str,
    *,
    split: CalibrationSplit = CalibrationSplit.FIT,
    outcome: CalibrationOutcome = CalibrationOutcome.COMPLETED,
    censoring: CensoringKind = CensoringKind.NONE,
    value: float | None = 20.0,
    baseline: float = 10.0,
    session: float = 32.0,
) -> CalibrationSample:
    return CalibrationSample(
        sample_id=sample_id,
        split=split,
        scope=(("architecture", "fixture"), ("provider_type", "synthetic")),
        numeric_coordinates=(("session_capacity", session),),
        categorical_coordinates=(("row_store_policy", "file_backed_full"),),
        outcome=outcome,
        censoring=censoring,
        targets=(() if value is None else (("predicted_walltime_high", value),)),
        analytic_baselines=(("predicted_walltime_high", baseline),),
        provenance_fingerprints=(("observation", sample_id),),
    )


def _config(*, minimum_support: int = 2) -> ResponseFitConfig:
    return ResponseFitConfig(
        models=(
            FitSpec(
                model_kind="positive_log_ratio_ridge",
                target="predicted_walltime_high",
                numeric_features=("session_capacity",),
                categorical_features=("row_store_policy",),
            ),
        ),
        minimum_support=minimum_support,
    )


def test_registry_is_explicit_and_extensible() -> None:
    registry = ModelRegistry()
    registry.register(PositiveLogRatioFamily())
    assert registry.kinds == ("positive_log_ratio_ridge",)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(PositiveLogRatioFamily())
    assert set(default_registry().kinds) == {
        "feasibility_envelope",
        "local_conservative_fidelity",
        "positive_log_ratio_ridge",
    }


def test_fit_never_uses_heldout_and_consumes_timeout_as_lower_bound(tmp_path) -> None:
    dataset = CalibrationDataset(
        tuple(
            sorted(
                (
                    _sample("fit-a", value=20.0),
                    _sample("fit-b", value=22.0, session=64.0),
                    _sample(
                        "oom",
                        outcome=CalibrationOutcome.OOM,
                        censoring=CensoringKind.FEASIBILITY,
                        value=None,
                    ),
                    _sample(
                        "timeout",
                        outcome=CalibrationOutcome.TIMEOUT,
                        censoring=CensoringKind.RUNTIME_LOWER_BOUND,
                        value=9999.0,
                    ),
                    _sample(
                        "heldout",
                        split=CalibrationSplit.HELDOUT,
                        value=10_000.0,
                    ),
                ),
                key=lambda row: row.sample_id,
            )
        )
    )
    bundle = fit_response_bundle(dataset, _config())
    artifact = bundle.models[0]

    assert artifact.support_ids == ("fit-a", "fit-b", "timeout")
    assert artifact.diagnostics["fit_count"] == 2
    assert artifact.diagnostics["runtime_lower_bound_count"] == 1
    assert artifact.diagnostics["heldout_count"] == 1
    assert bundle.diagnostics["heldout_ids"] == ("heldout",)

    path = bundle.write(tmp_path / "bundle.json")
    loaded = load_response_bundle(path)
    assert loaded.content_fingerprint == bundle.content_fingerprint
    assert loaded.to_json() == bundle.to_json()
    assert path.read_text().endswith("\n")


def test_timeout_runtime_bound_only_inflates_upper_prediction() -> None:
    dataset = CalibrationDataset(
        tuple(
            sorted(
                (
                    _sample("complete-a", value=20.0),
                    _sample("complete-b", value=20.0, session=64.0),
                    _sample(
                        "timeout",
                        outcome=CalibrationOutcome.TIMEOUT,
                        censoring=CensoringKind.RUNTIME_LOWER_BOUND,
                        value=100.0,
                    ),
                ),
                key=lambda row: row.sample_id,
            )
        )
    )
    prediction = fit_response_bundle(dataset, _config()).predict(
        "predicted_walltime_high",
        numeric={"session_capacity": 32.0},
        categorical={"row_store_policy": "file_backed_full"},
        scope={"architecture": "fixture", "provider_type": "synthetic"},
        analytic_baseline=10.0,
    )

    assert prediction is not None and prediction.supported
    assert prediction.estimate == pytest.approx(20.0)
    assert prediction.upper >= 100.0


def test_feasibility_envelope_is_profile_and_category_scoped() -> None:
    def feasibility_sample(
        sample_id: str,
        *,
        profile: str,
        row_store: str,
        outcome: CalibrationOutcome,
    ) -> CalibrationSample:
        return CalibrationSample(
            sample_id=sample_id,
            split=CalibrationSplit.FIT,
            scope=(
                ("architecture", "fixture"),
                ("provider_profile", profile),
                ("provider_type", "synthetic"),
            ),
            numeric_coordinates=(("session_capacity", 16.0),),
            categorical_coordinates=(("row_store_policy", row_store),),
            outcome=outcome,
            censoring=(
                CensoringKind.NONE
                if outcome is CalibrationOutcome.COMPLETED
                else CensoringKind.FEASIBILITY
            ),
        )

    rows = tuple(
        sorted(
            (
                feasibility_sample(
                    "a-1",
                    profile="profile-a",
                    row_store="file_backed_full",
                    outcome=CalibrationOutcome.COMPLETED,
                ),
                feasibility_sample(
                    "a-2",
                    profile="profile-a",
                    row_store="file_backed_full",
                    outcome=CalibrationOutcome.COMPLETED,
                ),
                feasibility_sample(
                    "b-1",
                    profile="profile-b",
                    row_store="file_backed_full",
                    outcome=CalibrationOutcome.OOM,
                ),
                feasibility_sample(
                    "b-2",
                    profile="profile-b",
                    row_store="file_backed_full",
                    outcome=CalibrationOutcome.OOM,
                ),
                feasibility_sample(
                    "category-1",
                    profile="profile-a",
                    row_store="recompute",
                    outcome=CalibrationOutcome.OOM,
                ),
                feasibility_sample(
                    "category-2",
                    profile="profile-a",
                    row_store="recompute",
                    outcome=CalibrationOutcome.OOM,
                ),
            ),
            key=lambda row: row.sample_id,
        )
    )
    bundle = fit_response_bundle(
        CalibrationDataset(rows),
        ResponseFitConfig(
            models=(
                FitSpec(
                    model_kind="feasibility_envelope",
                    target="feasible",
                    numeric_features=("session_capacity",),
                    categorical_features=("row_store_policy",),
                ),
            ),
            minimum_support=2,
        ),
    )

    def predict(profile: str, row_store: str):
        return bundle.predict(
            "feasible",
            numeric={"session_capacity": 16.0},
            categorical={"row_store_policy": row_store},
            scope={
                "architecture": "fixture",
                "provider_profile": profile,
                "provider_type": "synthetic",
            },
        )

    assert predict("profile-a", "file_backed_full").estimate == 1.0
    assert predict("profile-b", "file_backed_full").estimate == 0.0
    assert predict("profile-a", "recompute").estimate == 0.0
    assert not predict("profile-c", "file_backed_full").supported


def test_bundle_fingerprint_and_model_kind_are_verified(tmp_path) -> None:
    dataset = CalibrationDataset(tuple(sorted((_sample("a"), _sample("b")), key=lambda row: row.sample_id)))
    path = fit_response_bundle(dataset, _config()).write(tmp_path / "bundle.json")
    raw = json.loads(path.read_text())
    raw["models"][0]["parameters"]["coefficients"][0] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fingerprint"):
        load_response_bundle(path)

    valid = fit_response_bundle(dataset, _config()).write(tmp_path / "valid.json")
    raw = json.loads(valid.read_text())
    raw["models"][0]["model_kind"] = "unregistered"
    unsigned = {key: value for key, value in raw.items() if key != "content_fingerprint"}
    from circuit_tracer.governor.contracts import fingerprint

    raw["content_fingerprint"] = fingerprint(unsigned)
    valid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unregistered"):
        load_response_bundle(valid)


def test_positive_response_has_declared_ordered_uncertainty() -> None:
    dataset = CalibrationDataset(tuple(sorted((_sample("a", value=18), _sample("b", value=22, session=64)), key=lambda row: row.sample_id)))
    bundle = fit_response_bundle(dataset, _config())
    prediction = bundle.predict(
        "predicted_walltime_high",
        numeric={"session_capacity": 32.0},
        categorical={"row_store_policy": "file_backed_full"},
        scope={"architecture": "fixture", "provider_type": "synthetic"},
        analytic_baseline=10.0,
    )
    assert prediction is not None and prediction.supported
    assert 0 < prediction.lower <= prediction.estimate <= prediction.upper
    assert bundle.models[0].uncertainty_method == "normal_residual_log_ratio"


def test_resolver_applies_supported_correction_and_falls_back_when_unsupported() -> None:
    baseline = resolve_trace_plan(_semantics(), _profile(), _envelope())
    analytic = next(
        item.amount for item in baseline.admission.estimates
        if item.name == "predicted_walltime_high"
    )
    samples = tuple(
        sorted(
            (
                _sample("a", value=analytic * 2, baseline=analytic),
                _sample("b", value=analytic * 2.2, baseline=analytic, session=64),
            ),
            key=lambda row: row.sample_id,
        )
    )
    bundle = fit_response_bundle(CalibrationDataset(samples), _config())
    corrected = resolve_trace_plan(
        _semantics(), _profile(), _envelope(), response_bundle=bundle
    )
    corrected_high = next(
        item.amount for item in corrected.admission.estimates
        if item.name == "predicted_walltime_high"
    )
    assert corrected_high > analytic
    assert corrected.admission.response_bundle_fingerprint == bundle.content_fingerprint

    unsupported_artifact = replace(
        bundle.models[0], support_scope=(("architecture", "different"),)
    )
    unsigned = bundle.unsigned_payload()
    unsigned["models"] = [
        {
            **unsigned["models"][0],
            "support_scope": unsupported_artifact.support_scope,
        }
    ]
    from circuit_tracer.governor.contracts import fingerprint
    unsupported = replace(
        bundle,
        models=(unsupported_artifact,),
        content_fingerprint=fingerprint(unsigned),
    )
    fallback = resolve_trace_plan(
        _semantics(), _profile(), _envelope(), response_bundle=unsupported
    )
    fallback_high = next(
        item.amount for item in fallback.admission.estimates
        if item.name == "predicted_walltime_high"
    )
    assert fallback_high == analytic


def test_resolver_refuses_supported_infeasible_candidate() -> None:
    profile = _profile()
    rows = tuple(
        CalibrationSample(
            sample_id=f"oom-{index}",
            split=CalibrationSplit.FIT,
            scope=tuple(
                sorted(
                    {
                        "architecture": profile.identity.architecture,
                        "provider_profile": profile.profile_name,
                        "provider_type": profile.identity.provider_type,
                    }.items()
                )
            ),
            numeric_coordinates=(("session_capacity", 1.0),),
            outcome=CalibrationOutcome.OOM,
            censoring=CensoringKind.FEASIBILITY,
        )
        for index in range(2)
    )
    bundle = fit_response_bundle(
        CalibrationDataset(rows),
        ResponseFitConfig(
            models=(
                FitSpec(
                    model_kind="feasibility_envelope",
                    target="feasible",
                    numeric_features=("session_capacity",),
                ),
            ),
            minimum_support=2,
        ),
    )

    plan = resolve_trace_plan(
        _semantics(), profile, _envelope(), response_bundle=bundle
    )

    assert not plan.admission.admitted
    assert "response model identifies candidate as infeasible" in plan.admission.refusals


def test_supported_correction_uses_nonnegative_prediction_upper_directly() -> None:
    baseline = resolver_module._resolve_single_trace_plan(
        _semantics(), _profile(), _envelope()
    )
    analytic = next(
        item.amount
        for item in baseline.admission.estimates
        if item.name == "predicted_walltime_high"
    )
    bundle = fit_response_bundle(
        CalibrationDataset(
            tuple(
                sorted(
                    (
                        _sample("a", value=analytic * 0.5, baseline=analytic),
                        _sample(
                            "b",
                            value=analytic * 0.5,
                            baseline=analytic,
                            session=64,
                        ),
                    ),
                    key=lambda row: row.sample_id,
                )
            )
        ),
        _config(),
    )

    corrected = resolver_module._resolve_single_trace_plan(
        _semantics(), _profile(), _envelope(), response_bundle=bundle
    )
    corrected_high = next(
        item.amount
        for item in corrected.admission.estimates
        if item.name == "predicted_walltime_high"
    )

    assert corrected_high == pytest.approx(analytic * 0.5)
    assert corrected_high < analytic


def test_response_bundle_fingerprint_changes_execution_fingerprint() -> None:
    bundle = fit_response_bundle(
        CalibrationDataset(
            tuple(
                sorted(
                    (_sample("a"), _sample("b", session=64.0)),
                    key=lambda row: row.sample_id,
                )
            )
        ),
        _config(),
    )
    unsigned = bundle.unsigned_payload()
    unsigned["diagnostics"] = {**unsigned["diagnostics"], "revision": "second"}
    from circuit_tracer.governor.contracts import fingerprint

    revised = replace(
        bundle,
        diagnostics=unsigned["diagnostics"],
        content_fingerprint=fingerprint(unsigned),
    )
    first = resolver_module._resolve_single_trace_plan(
        _semantics(), _profile(), _envelope(), response_bundle=bundle
    )
    second = resolver_module._resolve_single_trace_plan(
        _semantics(), _profile(), _envelope(), response_bundle=revised
    )

    assert first.admission.estimates == second.admission.estimates
    assert first.execution_fingerprint != second.execution_fingerprint


def test_fidelity_bundle_requires_supported_scope() -> None:
    baseline = resolve_trace_plan(_semantics(), _profile(), _envelope())
    session = float(baseline.physical.session_capacity)
    rows = tuple(
        CalibrationSample(
            sample_id=f"fidelity-{index}",
            split=CalibrationSplit.FIT,
            scope=(("architecture", "fixture"), ("provider_type", "synthetic")),
            numeric_coordinates=(("session_capacity", session),),
            categorical_coordinates=(
                ("row_store_policy", baseline.physical.row_store_policy),
            ),
            targets=(("edge_recall", value),),
        )
        for index, value in enumerate((0.98, 0.99))
    )
    config = ResponseFitConfig(
        models=(
            FitSpec(
                model_kind="local_conservative_fidelity",
                target="edge_recall",
                numeric_features=("session_capacity",),
                categorical_features=("row_store_policy",),
            ),
        ),
        minimum_support=2,
    )
    bundle = fit_response_bundle(CalibrationDataset(rows), config)
    semantics = replace(
        _semantics(),
        fidelity=FidelityMode.BOUNDED,
        fidelity_budget=FidelityBudget(
            (("edge_recall", 0.95),), ("session_capacity",)
        ),
    )

    prediction, _, _, refusals = resolver_module._assess_fidelity(
        semantics, _profile(), baseline.physical, None, bundle
    )
    assert prediction.lower_bound("edge_recall") == pytest.approx(0.98)
    assert refusals == ()

    artifact = replace(bundle.models[0], support_scope=(("architecture", "other"),))
    unsigned = bundle.unsigned_payload()
    unsigned["models"][0]["support_scope"] = artifact.support_scope
    from circuit_tracer.governor.contracts import fingerprint

    unsupported = replace(
        bundle,
        models=(artifact,),
        content_fingerprint=fingerprint(unsigned),
    )
    prediction, _, _, refusals = resolver_module._assess_fidelity(
        semantics, _profile(), baseline.physical, None, unsupported
    )
    assert prediction.metrics == ()
    assert any("no prediction" in reason for reason in refusals)
