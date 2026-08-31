from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from nnsight import NNsight, save
from nnsight.intervention.interleaver import Mediator
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig

from circuit_tracer.verification import (
    AcceptedGraphView,
    AliasComparatorStatus,
    AliasControlCandidate,
    AliasSelectionEvidence,
    AliasSubstitution,
    BaselineCapture,
    EvidenceCompleteness,
    FaithfulnessVerdict,
    FrozenBehavioralCalibration,
    BehavioralProbePolicy,
    BehavioralVerificationRequest,
    DeterministicInterventionRuntime,
    FeatureEvidence,
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionRuntimePort,
    InterventionSemantics,
    NNSightInterventionRuntime,
    OrderingAdmissionMode,
    RuntimeExecutionStatus,
    ScriptedVariant,
    SufficiencyStatus,
    TraceIdentity,
    VariantKind,
    plan_behavioral_variants,
    select_necessity_features,
    verify_behavior,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)
from circuit_tracer.verification.nnsight_runtime import (
    CaptureOrigin,
    NNSightInterventionPlan,
    NNSightVariantPlan,
    SelectiveProbeCapture,
    _invoke_ordered_hook_families,
    _provider_activation_delta,
    _skip_transcoder_correction,
    _translate_variant,
)


def _request(*, include_alias: bool = False, **policy: object) -> BehavioralVerificationRequest:
    high = FeatureNode(2, 4, 10)
    negative = FeatureNode(4, 5, 11)
    other = FeatureNode(2, 6, 12)
    control = FeatureNode(2, 4, 99)
    negative_control = FeatureNode(4, 5, 97)
    other_control = FeatureNode(2, 6, 96)
    alias_control = FeatureNode(2, 7, 98)
    downstream_a = FeatureNode(6, 4, 200)
    downstream_b = FeatureNode(7, 5, 201)
    downstream_c = FeatureNode(8, 6, 202)
    graph = AcceptedGraphView(
        "graph-1",
        (
            FeatureEvidence(high, 2.0, 0.8, True, 0, (FeatureValue(downstream_a, 0.4),)),
            FeatureEvidence(negative, -3.0, -0.5, True, 1, (FeatureValue(downstream_b, -0.25),)),
            FeatureEvidence(other, 1.0, 0.3, True, 2, (FeatureValue(downstream_c, 0.1),)),
            FeatureEvidence(control, 2.1, 0.01, False, necessity_control_for=high),
            FeatureEvidence(
                negative_control,
                -2.9,
                -0.02,
                False,
                necessity_control_for=negative,
            ),
            FeatureEvidence(
                other_control,
                1.1,
                0.015,
                False,
                necessity_control_for=other,
            ),
            FeatureEvidence(alias_control, 1.8, 0.005, False),
        ),
    )
    return BehavioralVerificationRequest(
        TraceIdentity(
            "trace-1",
            "graph-1",
            "provider-1",
            "semantic-1",
            "execution-1",
            tuple(range(1, 11)),
            2,
            42,
        ),
        graph,
        policy=BehavioralProbePolicy(**policy),
        aliases=(
            AliasSubstitution(
                source=high,
                substitute=control,
                substitute_absolute_preactivation=5.0,
                selection_evidence=AliasSelectionEvidence(
                    selection_policy_id="alias_decoder_match_v1",
                    calibration_fingerprint="alias-calibration-1",
                    comparison_evidence_fingerprint="alias-comparison-1",
                    baseline_graph_fingerprint="graph-1",
                    candidate_graph_fingerprint="candidate-graph-1",
                    decoder_fingerprint="decoder-1",
                    decoder_output_topology="complete_downstream_block",
                    qualified_decoder_cosine=0.95,
                    observed_decoder_cosine=0.96,
                    source_decoder_norm=1.0,
                    substitute_decoder_norm=1.1,
                    least_squares_coefficient=1.0,
                ),
                predicted_target_delta=-0.2,
                control_candidates=(AliasControlCandidate(alias_control, 0.05),),
            ),
        ) if include_alias else (),
    )


def _baseline(request: BehavioralVerificationRequest, target: float = 10.0) -> BaselineCapture:
    nodes = sorted(
        {
            value.node
            for feature in request.graph.features
            for value in feature.predicted_downstream_feature_deltas
        }
    )
    return BaselineCapture(
        target,
        (target - 1.0, target + 1.0),
        tuple(FeatureValue(node, 5.0 + index) for index, node in enumerate(nodes)),
    )


def _exact_scripts(
    request: BehavioralVerificationRequest,
    target: float = 10.0,
) -> dict[str, ScriptedVariant]:
    baseline_features = {item.node: item.preactivation for item in _baseline(request, target).feature_values}
    return {
        variant.variant_id: ScriptedVariant(
            target + (variant.predicted_target_delta or 0.0),
            raw_target_logits=(target - 1.0, target + 1.0),
            downstream_feature_values=tuple(
                FeatureValue(item.node, baseline_features[item.node] + item.preactivation)
                for item in variant.predicted_downstream_feature_deltas
            ),
        )
        for variant in plan_behavioral_variants(request)
    }
def test_planner_uses_explicit_semantics_absolute_values_and_required_no_op() -> None:
    variants = plan_behavioral_variants(_request())
    assert len(variants) == 10  # baseline capture is deliberately not counted
    assert variants[0].kind is VariantKind.NO_OP
    assert variants[0].interventions == ()
    direct = [item for item in variants if item.kind is VariantKind.DIRECT_DOUBLE]
    assert all(item.semantics is InterventionSemantics.DIRECT_FROZEN for item in direct)
    assert direct[0].interventions[0].absolute_value == 4.0
    assert direct[0].predicted_target_delta == 0.8
    assert direct[0].predicted_downstream_feature_deltas[0].preactivation == 0.4
    necessity = [
        item
        for item in variants
        if item.kind in (VariantKind.NECESSITY_HIGH, VariantKind.NECESSITY_CONTROL)
    ]
    assert all(
        item.semantics is InterventionSemantics.PROPAGATED_FROZEN_ATTENTION
        for item in necessity
    )
    assert [item.interventions[0].node.feature for item in necessity] == [10, 11, 12, 99, 97, 96]
    assert all("pair" in item.variant_id for item in necessity)
    assert all(item.interventions[0].absolute_value == 0.0 for item in necessity)


def test_churn_plan_reuses_necessity_source_and_adds_only_one_alias_variant() -> None:
    variants = plan_behavioral_variants(_request(include_alias=True))
    assert len(variants) == 8
    assert not any(item.kind is VariantKind.DIRECT_DOUBLE for item in variants)
    assert sum(item.kind is VariantKind.NECESSITY_HIGH for item in variants) == 3
    assert sum(item.kind is VariantKind.NECESSITY_CONTROL for item in variants) == 3
    alias = next(item for item in variants if item.kind is VariantKind.ALIAS_SUBSTITUTION)
    assert alias.kind is VariantKind.ALIAS_SUBSTITUTION
    assert [item.absolute_value for item in alias.interventions] == [0.0, 5.0]


def test_low_variant_limit_admits_only_complete_necessity_pairs() -> None:
    assert [item.kind for item in plan_behavioral_variants(_request(max_variants=2))] == [
        VariantKind.NO_OP
    ]
    variants = plan_behavioral_variants(_request(max_variants=3))
    assert [item.kind for item in variants] == [
        VariantKind.NO_OP,
        VariantKind.NECESSITY_HIGH,
        VariantKind.NECESSITY_CONTROL,
    ]
    assert "pair0" in variants[1].variant_id
    assert "pair0" in variants[2].variant_id


def test_required_necessity_anchor_is_included_deterministically() -> None:
    request = _request()
    required = FeatureNode(5, 7, 13)
    graph = AcceptedGraphView(
        request.graph.graph_fingerprint,
        request.graph.features
        + (
            FeatureEvidence(required, 1.0, 0.26, True, 3),
            FeatureEvidence(
                FeatureNode(5, 7, 95),
                1.1,
                0.0,
                False,
                necessity_control_for=required,
            ),
        ),
    )
    selected = select_necessity_features(
        graph,
        sample_count=3,
        required_nodes=(required,),
    )
    assert required in {item.node for item in selected}


def test_limits_and_graph_identity_are_structural_caller_errors() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        BehavioralProbePolicy(policy_id="unversioned")
    with pytest.raises(ValueError, match="max_variants"):
        BehavioralProbePolicy(max_variants=11)
    with pytest.raises(ValueError, match="max_seconds"):
        BehavioralProbePolicy(max_seconds=121)
    request = _request()
    with pytest.raises(ValueError, match="fingerprints"):
        BehavioralVerificationRequest(
            TraceIdentity("trace", "other", "provider", "semantic", "execution", (1,), 1, 1),
            request.graph,
        )


def test_complete_fake_run_uses_one_port_and_reports_closure() -> None:
    request = _request()
    variants = plan_behavioral_variants(request)
    baseline = 10.0
    scripts = _exact_scripts(request, baseline)
    runtime = DeterministicInterventionRuntime(
        _baseline(request, baseline), scripts
    )
    assert isinstance(runtime, InterventionRuntimePort)
    report = verify_behavior(request, runtime)
    assert report.evidence_completeness is EvidenceCompleteness.COMPLETE
    assert report.verdict is FaithfulnessVerdict.UNKNOWN
    assert report.runtime_status is RuntimeExecutionStatus.COMPLETE
    assert report.no_op_required is True
    assert report.no_op_passed is True
    assert report.variants_planned == len(variants)
    assert report.sufficiency is SufficiencyStatus.UNKNOWN
    assert all(
        item.closure_error == pytest.approx(0.0)
        for item in report.evidence
        if item.closure_error is not None
    )
    assert len(runtime.requests) == 1
    assert runtime.requests[0].observed_downstream_nodes == tuple(
        sorted(
            {
                value.node
                for variant in variants
                for value in variant.predicted_downstream_feature_deltas
            }
        )
    )


def test_fake_refuses_predicted_over_budget_variant_before_start() -> None:
    calibration = FrozenBehavioralCalibration(
        "partial_runtime_v1",
        "behavioral_closure_v1",
        direct_max_mean_abs_closure=1.0,
    )
    request = _request(
        calibration=calibration,
        max_seconds=5.0,
        cleanup_reserve_seconds=1.0,
        predicted_baseline_seconds=1.0,
        predicted_variant_seconds=1.0,
    )
    variants = plan_behavioral_variants(request)
    scripts = {
        item.variant_id: ScriptedVariant(
            10.0,
            actual_seconds=1.0,
            predicted_seconds=4.0 if index == 1 else 1.0,
        )
        for index, item in enumerate(variants)
    }
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(
            BaselineCapture(10.0, (9.0, 11.0)), scripts, baseline_actual_seconds=1.0
        ),
    )
    assert report.evidence_completeness is EvidenceCompleteness.PARTIAL
    assert report.verdict is FaithfulnessVerdict.UNKNOWN
    assert report.variants_completed == 1
    assert report.refusal is not None and report.refusal.code == "deadline_admission"
    assert report.refusal.before_variant_id == variants[1].variant_id


def test_in_flight_overrun_and_runtime_exception_are_behavioral_unknown_or_partial() -> None:
    request = _request(max_seconds=5.0, cleanup_reserve_seconds=1.0)
    variants = plan_behavioral_variants(request)
    scripts = {
        item.variant_id: ScriptedVariant(10.0, actual_seconds=4.5 if index == 0 else 0.0)
        for index, item in enumerate(variants)
    }
    overrun = DeterministicInterventionRuntime(
        BaselineCapture(10.0, (9.0, 11.0)), scripts, baseline_actual_seconds=1.0
    )
    report = verify_behavior(request, overrun)
    assert report.evidence_completeness is EvidenceCompleteness.PARTIAL
    assert report.refusal is not None and report.refusal.code == "deadline_in_flight_overrun"

    class BrokenRuntime:
        def evaluate(self, request: object) -> object:
            raise RuntimeError("backend lost")

    unknown = verify_behavior(request, BrokenRuntime())  # type: ignore[arg-type]
    assert unknown.evidence_completeness is EvidenceCompleteness.UNKNOWN
    assert unknown.verdict is FaithfulnessVerdict.UNKNOWN
    assert unknown.refusal is not None and unknown.refusal.code == "runtime_exception"


def test_no_op_drift_or_unconfirmed_teardown_makes_evidence_unknown() -> None:
    request = _request()
    variants = plan_behavioral_variants(request)
    scripts = {item.variant_id: ScriptedVariant(10.0) for item in variants}
    scripts["no_op"] = ScriptedVariant(10.1)
    report = verify_behavior(
        request, DeterministicInterventionRuntime(BaselineCapture(10.0, (9.0, 11.0)), scripts)
    )
    assert report.evidence_completeness is EvidenceCompleteness.UNKNOWN
    assert report.no_op_passed is False

    scripts["no_op"] = ScriptedVariant(10.0)
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(
            BaselineCapture(10.0, (9.0, 11.0)), scripts, cleanup_completed=False
        ),
    )
    assert report.evidence_completeness is EvidenceCompleteness.UNKNOWN
    assert "runtime teardown was not confirmed" in report.reasons


def test_calibrated_no_op_drift_is_a_hard_contradiction() -> None:
    calibration = FrozenBehavioralCalibration(
        "no-op-v1",
        "behavioral_closure_v1",
        direct_max_relative_closure=0.10,
    )
    request = _request(calibration=calibration)
    scripts = _exact_scripts(request)
    scripts["no_op"] = ScriptedVariant(10.0 + 2e-6)
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(_baseline(request), scripts),
    )
    assert report.no_op_passed is False
    assert report.evidence_completeness is EvidenceCompleteness.UNKNOWN
    assert report.verdict is FaithfulnessVerdict.CONTRADICTED


def test_frozen_calibration_separates_complete_evidence_from_supported_verdict() -> None:
    calibration = FrozenBehavioralCalibration(
        calibration_id="granite-behavior-v1",
        policy_id="behavioral_closure_v1",
        direct_max_mean_relative_closure=0.05,
        direct_max_relative_closure=0.10,
        direct_min_sign_agreement=0.95,
        downstream_max_mean_relative_closure=0.05,
        downstream_max_p95_relative_closure=0.10,
        necessity_min_predicted_realized_spearman=0.8,
        necessity_min_median_high_control_effect_ratio=2.0,
    )
    request = _request(calibration=calibration)
    scripts = _exact_scripts(request)
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(
            _baseline(request), scripts
        ),
    )
    assert report.evidence_completeness is EvidenceCompleteness.COMPLETE
    assert report.verdict is FaithfulnessVerdict.SUPPORTED
    assert report.calibration_id == "granite-behavior-v1"
    assert report.metrics.direct_mean_abs_closure == pytest.approx(0.0)
    assert report.metrics.direct_mean_relative_closure == pytest.approx(0.0)
    assert report.metrics.direct_max_relative_closure == pytest.approx(0.0)
    assert report.metrics.direct_sign_agreement == pytest.approx(1.0)
    assert report.metrics.necessity_high_vs_control_separation == pytest.approx(0.485)
    assert report.metrics.necessity_predicted_realized_spearman == pytest.approx(1.0)
    assert report.metrics.necessity_median_high_control_effect_ratio == pytest.approx(0.5 / 0.015)
    assert report.alias_comparator_status is AliasComparatorStatus.NOT_APPLICABLE
    assert report.metrics.downstream_mean_abs_closure == pytest.approx(0.0)
    assert report.metrics.downstream_mean_relative_closure == pytest.approx(0.0)
    assert report.metrics.downstream_p95_relative_closure == pytest.approx(0.0)
    first_direct = next(item for item in report.evidence if item.kind is VariantKind.DIRECT_DOUBLE)
    assert first_direct.downstream_closure[0].node == FeatureNode(6, 4, 200)
    assert first_direct.downstream_closure[0].predicted_delta == pytest.approx(0.4)
    assert first_direct.downstream_closure[0].observed_delta == pytest.approx(0.4)
    assert first_direct.downstream_closure[0].closure_error == pytest.approx(0.0)
    assert report.raw_execution.baseline is not None
    assert report.raw_execution.baseline.raw_target_logits == (9.0, 11.0)
    assert any(item.downstream_closure for item in report.evidence)
    assert report.sufficiency is SufficiencyStatus.UNKNOWN


def test_calibrated_failed_threshold_is_contradicted_and_missing_metric_is_unknown() -> None:
    direct_calibration = FrozenBehavioralCalibration(
        "direct-v1",
        "behavioral_closure_v1",
        direct_max_relative_closure=0.10,
    )
    request = _request(calibration=direct_calibration)
    variants = plan_behavioral_variants(request)
    scripts = _exact_scripts(request)
    scripts.update(
        {
            item.variant_id: ScriptedVariant(
                10.5,
                downstream_feature_values=scripts[item.variant_id].downstream_feature_values,
            )
            for item in variants
            if item.kind is VariantKind.DIRECT_DOUBLE
        }
    )
    contradicted = verify_behavior(
        request,
        DeterministicInterventionRuntime(_baseline(request), scripts),
    )
    assert contradicted.evidence_completeness is EvidenceCompleteness.COMPLETE
    assert contradicted.verdict is FaithfulnessVerdict.CONTRADICTED

    alias_calibration = FrozenBehavioralCalibration(
        "alias-v1",
        "behavioral_closure_v1",
        alias_max_relative_effect_error=0.2,
    )
    request_without_alias_probe = _request(
        include_alias=True,
        calibration=alias_calibration,
        max_variants=7,
    )
    variants = plan_behavioral_variants(request_without_alias_probe)
    scripts = _exact_scripts(request_without_alias_probe)
    unknown = verify_behavior(
        request_without_alias_probe,
        DeterministicInterventionRuntime(_baseline(request_without_alias_probe), scripts),
    )
    assert unknown.evidence_completeness is EvidenceCompleteness.COMPLETE
    assert unknown.verdict is FaithfulnessVerdict.UNKNOWN


def test_calibrated_mean_review_band_is_inconclusive_but_hard_limit_passes() -> None:
    calibration = FrozenBehavioralCalibration(
        "review-v1",
        "behavioral_closure_v1",
        direct_max_mean_relative_closure=0.05,
        direct_max_relative_closure=0.10,
    )
    request = _request(calibration=calibration)
    scripts = _exact_scripts(request)
    baseline = _baseline(request)
    scripts.update(
        {
            variant.variant_id: ScriptedVariant(
                baseline.target_value + 0.94 * variant.predicted_target_delta,
                downstream_feature_values=scripts[
                    variant.variant_id
                ].downstream_feature_values,
            )
            for variant in plan_behavioral_variants(request)
            if variant.kind is VariantKind.DIRECT_DOUBLE
            and variant.predicted_target_delta is not None
        }
    )
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(baseline, scripts),
    )
    assert report.metrics.direct_mean_relative_closure == pytest.approx(0.06)
    assert report.metrics.direct_max_relative_closure == pytest.approx(0.06)
    assert report.verdict is FaithfulnessVerdict.INCONCLUSIVE


def test_all_public_numeric_evidence_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureEvidence(FeatureNode(0, 0, 0), float("nan"), 1.0, True, 0)
    with pytest.raises(ValueError, match="finite"):
        BaselineCapture(1.0, (float("inf"),))
    with pytest.raises(ValueError):
        BehavioralProbePolicy(max_seconds=float("nan"))
    with pytest.raises(ValueError):
        ScriptedVariant(1.0, actual_seconds=float("inf"))


def test_necessity_claim_is_omitted_without_a_same_layer_control() -> None:
    request = _request()
    graph = AcceptedGraphView(
        request.graph.graph_fingerprint,
        tuple(item for item in request.graph.features if item.selected)
        + (FeatureEvidence(FeatureNode(9, 4, 99), 2.1, 0.01, False),),
    )
    no_same_layer_control = BehavioralVerificationRequest(request.identity, graph)
    kinds = {item.kind for item in plan_behavioral_variants(no_same_layer_control)}
    assert VariantKind.NECESSITY_HIGH not in kinds
    assert VariantKind.NECESSITY_CONTROL not in kinds


def test_alias_substitution_reuses_necessity_source_without_alias_control() -> None:
    request = _request(include_alias=True)
    alias = request.aliases[0]
    without_controls = BehavioralVerificationRequest(
        request.identity,
        request.graph,
        aliases=(
            AliasSubstitution(
                source=alias.source,
                substitute=alias.substitute,
                substitute_absolute_preactivation=alias.substitute_absolute_preactivation,
                selection_evidence=alias.selection_evidence,
                predicted_target_delta=alias.predicted_target_delta,
            ),
        ),
    )
    variants = plan_behavioral_variants(without_controls)
    assert sum(item.kind is VariantKind.ALIAS_SUBSTITUTION for item in variants) == 1
    assert not any(
        item.kind in (VariantKind.ALIAS_SOURCE_ABLATION, VariantKind.ALIAS_CONTROL)
        for item in variants
    )
    report = verify_behavior(
        without_controls,
        DeterministicInterventionRuntime(
            _baseline(without_controls), _exact_scripts(without_controls)
        ),
    )
    assert report.alias_comparator_status is AliasComparatorStatus.COMPLETE


def test_hook_families_get_separate_ordered_mediators() -> None:
    events: list[tuple[str, int]] = []

    class _Invoke:
        def __init__(self, tracer: object, index: int) -> None:
            self.tracer = tracer
            self.index = index

        def __enter__(self) -> None:
            self.tracer.current = self.index

        def __exit__(self, *args: object) -> None:
            self.tracer.current = None

    class _Tracer:
        current: int | None = None

        def __init__(self) -> None:
            self.invocations = 0

        def invoke(self) -> _Invoke:
            self.invocations += 1
            return _Invoke(self, self.invocations)

    tracer = _Tracer()

    def action(name: str) -> None:
        assert tracer.current is not None
        events.append((name, tracer.current))

    _invoke_ordered_hook_families(
        tracer,
        attention=lambda: action("attention"),
        layernorm_groups=(lambda: action("ln0"), lambda: action("ln1")),
        feature_input=lambda: action("feature_input"),
        feature_output=lambda: action("feature_output"),
    )
    assert [name for name, _ in events] == [
        "attention",
        "ln0",
        "ln1",
        "feature_input",
        "feature_output",
    ]
    assert len({invocation for _, invocation in events}) == len(events)


def test_provider_activation_and_skip_semantics_are_applied_to_preactivations() -> None:
    class _Provider:
        @staticmethod
        def apply_activation_function_to_feature(
            layer: int, feature: int, values: torch.Tensor
        ) -> torch.Tensor:
            del layer, feature
            return torch.relu(values)

        @staticmethod
        def encode_layer(
            values: torch.Tensor,
            layer: int,
            *,
            apply_activation_function: bool,
        ) -> torch.Tensor:
            del layer, apply_activation_function
            return values

        @staticmethod
        def compute_skip(layer: int, values: torch.Tensor) -> torch.Tensor:
            return values * 2

    provider = _Provider()
    assert _provider_activation_delta(provider, 0, 0, torch.tensor(-1.0), 3.0).item() == 3.0
    correction = _skip_transcoder_correction(
        provider,
        0,
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    assert torch.equal(correction, torch.tensor([2.0, 4.0]))


def test_propagated_injection_refuses_a_missing_activation_barrier() -> None:
    source = FeatureNode(0, 0, 0)
    plan = NNSightVariantPlan(
        "malformed_propagated",
        InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
        (NNSightInterventionPlan(source, 0.0, 1.0, -1.0, (0,)),),
        (),
        (source,),
        True,
        False,
        False,
    )
    host = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1),
        transcoders=object(),
    )

    with pytest.raises(RuntimeError, match="requires an activation barrier"):
        NNSightReplacementModel._verification_inject(
            host,
            plan,
            {},
            [],
            [],
            target_position=1,
            target_token_id=0,
            activation_barrier=None,
        )


def _execution_request(request: BehavioralVerificationRequest) -> InterventionExecutionRequest:
    variants = plan_behavioral_variants(request)
    observed_nodes = tuple(
        sorted(
            {
                value.node
                for variant in variants
                for value in variant.predicted_downstream_feature_deltas
            }
        )
    )
    return InterventionExecutionRequest(
        request.identity,
        request.target,
        variants,
        observed_nodes,
        request.policy.max_seconds,
        request.policy.cleanup_reserve_seconds,
        request.policy.predicted_baseline_seconds,
        request.policy.predicted_variant_seconds,
    )


class _MockProvider:
    capabilities = TranscoderCapabilities(
        architecture="clt",
        checkpoint_format="mock",
        decoder_output_topology="cross_layer",
    )
    d_transcoder = 512


class _MockConfig:
    n_layers = 10


class _MockHFConfig:
    vocab_size = 256


class _MockGemma3TextConfig:
    vocab_size = 256


class _MockGemma3Config:
    text_config = _MockGemma3TextConfig()


class _MockSelectiveNNSightModel:
    backend = "nnsight"
    transcoders = _MockProvider()
    device = "cpu"
    cfg = _MockConfig()
    config = _MockHFConfig()
    zero_positions: tuple[int, ...] | slice = ()
    verification_intervened_capture_ordering_qualified = True

    def __init__(
        self,
        *,
        fail_baseline: bool = False,
        fail_variant: bool = False,
        fail_release: bool = False,
    ) -> None:
        self.baseline_calls: list[tuple[FeatureNode, ...]] = []
        self.variant_plans: list[object] = []
        self.released = False
        self.health_checks = 0
        self.fail_baseline = fail_baseline
        self.fail_variant = fail_variant
        self.fail_release = fail_release

    def feature_intervention(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("full-cache feature_intervention must not be used")

    def setup_intervention_with_freeze(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("full-cache freeze setup must not be used")

    def _verification_capture_baseline(
        self,
        prompt_token_ids: tuple[int, ...],
        retained_nodes: tuple[FeatureNode, ...],
        **kwargs: object,
    ) -> SelectiveProbeCapture:
        self.baseline_calls.append(retained_nodes)
        if self.fail_baseline:
            raise RuntimeError("mock baseline failed")
        return SelectiveProbeCapture(
            11.0,
            1.0,
            tuple((node, float(index + 1)) for index, node in enumerate(retained_nodes)),
            CaptureOrigin.BASELINE_FORWARD,
            {"selective": True},
        )

    def _verification_run_variant(
        self,
        prompt_token_ids: tuple[int, ...],
        plan: NNSightVariantPlan,
        baseline_state: object,
        **kwargs: object,
    ) -> SelectiveProbeCapture:
        self.variant_plans.append(plan)
        if self.fail_variant:
            raise RuntimeError("mock trace failed")
        retained = tuple(sorted(set(plan.observed_nodes) | set(plan.retain_intervention_nodes)))
        return SelectiveProbeCapture(
            12.0,
            1.0,
            tuple((node, float(index + 101)) for index, node in enumerate(retained)),
            CaptureOrigin.INTERVENED_FORWARD,
        )

    def _verification_release(self, baseline_state: object) -> None:
        self.released = True
        if self.fail_release:
            raise RuntimeError("mock cleanup failed")

    def _verification_health_check(self, baseline_state: object) -> bool:
        self.health_checks += 1
        return not self.fail_release


class _TinyRealNNSightVerificationHost:
    def __init__(self) -> None:
        self.model = NNsight(torch.nn.Linear(3, 5, bias=False))
        self.zero_positions = slice(0, 0)
        self.attention_locs = ()
        self.layernorm_scale_locs = ()
        self.feature_input_locs = ()
        self.feature_output_locs = ()
        self.skip_transcoder = False

    def _verification_tokens(self, prompt_token_ids: tuple[int, ...]) -> torch.Tensor:
        del prompt_token_ids
        return torch.ones(1, 2, 3)

    @contextmanager
    def _verification_probe_scope(self):
        yield

    @contextmanager
    def zero_softcap(self):
        yield

    def trace(self):
        return self.model.trace()

    @property
    def output(self):
        return SimpleNamespace(logits=self.model.output)

    def _verification_encode_layers(self, retained_nodes, **kwargs):
        del retained_nodes, kwargs
        return {0: self.model.output}

    _verification_save_feature_values = staticmethod(
        NNSightReplacementModel._verification_save_feature_values
    )

    def _verification_inject(
        self,
        plan,
        activations,
        objective_handles,
        feature_value_handles,
        *,
        target_position,
        target_token_id,
        activation_barrier,
        direct_effects_barriers,
    ):
        del activation_barrier, direct_effects_barriers
        retained = tuple(
            node
            for node in sorted(set(plan.observed_nodes) | set(plan.retain_intervention_nodes))
            if node.layer in activations
        )
        feature_value_handles.extend(
            self._verification_save_feature_values(activations, retained)
        )
        intervention = plan.interventions[0]
        node = intervention.node
        activation = activations[node.layer][0, node.position, node.feature]
        logits = self.output.logits[0, target_position - 1] + activation * 0
        objective_handles.extend((save(logits[target_token_id]), save(logits.mean())))


class _TinyRealNNSightDirectFreezeHost(_TinyRealNNSightVerificationHost):
    def __init__(self) -> None:
        super().__init__()
        self.model = NNsight(
            torch.nn.Sequential(
                torch.nn.Linear(3, 3, bias=False),
                torch.nn.Linear(3, 5, bias=False),
            )
        )
        self.feature_output_locs = (self.model[0], self.model[1])

    _verification_freeze_feature_output = (
        NNSightReplacementModel._verification_freeze_feature_output
    )

    def _verification_inject(
        self,
        plan,
        activations,
        objective_handles,
        feature_value_handles,
        *,
        target_position,
        target_token_id,
        activation_barrier,
        direct_effects_barriers,
    ):
        del activation_barrier
        retained = tuple(
            node
            for node in sorted(set(plan.observed_nodes) | set(plan.retain_intervention_nodes))
            if node.layer in activations
        )
        feature_value_handles.extend(
            self._verification_save_feature_values(activations, retained)
        )
        intervention = plan.interventions[0]
        node = intervention.node
        activation = activations[node.layer][0, node.position, node.feature]
        for direct_effects_barrier in direct_effects_barriers:
            direct_effects_barrier()
        logits = self.output.logits[0, target_position - 1] + activation * 0
        objective_handles.extend((save(logits[target_token_id]), save(logits.mean())))


class _TinyProductionInjectHost(_TinyRealNNSightVerificationHost):
    """Small PLT-shaped host that exercises the production injection method."""

    class _Provider:
        skip_connection = False

        @staticmethod
        def apply_activation_function_to_feature(
            layer: int, feature: int, values: torch.Tensor
        ) -> torch.Tensor:
            del layer, feature
            return torch.relu(values)

        @staticmethod
        def encode_layer(
            values: torch.Tensor,
            layer: int,
            *,
            apply_activation_function: bool,
        ) -> torch.Tensor:
            del layer, apply_activation_function
            return values * 1

        @staticmethod
        def _get_decoder_vectors(layer: int, feature_ids: torch.Tensor) -> torch.Tensor:
            del layer
            return torch.ones(
                len(feature_ids),
                8,
                dtype=torch.float32,
                device=feature_ids.device,
            )

    def __init__(self) -> None:
        super().__init__()
        config = Gemma3TextConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            layer_types=["sliding_attention"] * 4,
            use_cache=False,
        )
        multimodal_config = Gemma3Config(
            text_config=config,
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 14,
                "patch_size": 14,
            },
            image_token_index=16,
        )
        self.model = NNsight(Gemma3ForConditionalGeneration(multimodal_config).eval())
        self.feature_output_locs = tuple(
            self.model.model.language_model.layers[index].post_feedforward_layernorm
            for index in range(4)
        )
        self.transcoders = self._Provider()
        self.cfg = SimpleNamespace(n_layers=4)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def get_feature_output_loc(self, layer: int):
        return self.model.model.language_model.layers[layer].post_feedforward_layernorm

    def get_feature_input_loc(self, layer: int):
        return self.model.model.language_model.layers[layer].pre_feedforward_layernorm

    def _verification_tokens(self, prompt_token_ids: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor((prompt_token_ids,), dtype=torch.long)

    @property
    def output(self):
        return SimpleNamespace(logits=self.model.output.logits)

    _verification_encode_layers = NNSightReplacementModel._verification_encode_layers
    _verification_freeze_feature_output = (
        NNSightReplacementModel._verification_freeze_feature_output
    )
    _verification_inject = NNSightReplacementModel._verification_inject


def test_real_nnsight_baseline_returns_helper_saved_feature_values_across_invoke() -> None:
    host = _TinyRealNNSightVerificationHost()
    retained_node = FeatureNode(0, 0, 1)

    capture = NNSightReplacementModel._verification_capture_baseline(
        host,
        (1, 2),
        (retained_node,),
        target_position=2,
        target_token_id=1,
        retain_attention_state=False,
        retain_direct_freeze_state=False,
    )

    assert capture.origin is CaptureOrigin.BASELINE_FORWARD
    assert tuple(node for node, _ in capture.feature_values) == (retained_node,)


def test_real_nnsight_variant_returns_helper_saved_feature_values_across_invoke() -> None:
    host = _TinyRealNNSightVerificationHost()
    retained_node = FeatureNode(0, 0, 1)
    plan = NNSightVariantPlan(
        "no_op",
        InterventionSemantics.DIRECT_FROZEN,
        (),
        (retained_node,),
        (),
        False,
        False,
        False,
    )

    capture = NNSightReplacementModel._verification_run_variant(
        host,
        (1, 2),
        plan,
        None,
        target_position=2,
        target_token_id=1,
    )

    assert capture.origin is CaptureOrigin.INTERVENED_FORWARD
    assert tuple(node for node, _ in capture.feature_values) == (retained_node,)


def test_real_nnsight_intervention_reencodes_inside_its_invoke() -> None:
    host = _TinyRealNNSightVerificationHost()
    retained_node = FeatureNode(0, 0, 1)
    baseline = NNSightReplacementModel._verification_capture_baseline(
        host,
        (1, 2),
        (retained_node,),
        target_position=2,
        target_token_id=1,
        retain_attention_state=False,
        retain_direct_freeze_state=False,
    )
    plan = NNSightVariantPlan(
        "direct",
        InterventionSemantics.DIRECT_FROZEN,
        (NNSightInterventionPlan(retained_node, 2.0, 1.0, 1.0, (0,)),),
        (retained_node,),
        (retained_node,),
        False,
        False,
        False,
    )

    capture = NNSightReplacementModel._verification_run_variant(
        host,
        (1, 2),
        plan,
        baseline.retained_state,
        target_position=2,
        target_token_id=1,
    )

    assert capture.origin is CaptureOrigin.INTERVENED_FORWARD
    assert tuple(node for node, _ in capture.feature_values) == (retained_node,)


def test_real_nnsight_direct_freeze_preserves_ordered_feature_outputs() -> None:
    host = _TinyRealNNSightDirectFreezeHost()
    retained_node = FeatureNode(0, 0, 1)
    baseline = NNSightReplacementModel._verification_capture_baseline(
        host,
        (1, 2),
        (retained_node,),
        target_position=2,
        target_token_id=1,
        retain_attention_state=False,
        retain_direct_freeze_state=True,
    )
    plan = NNSightVariantPlan(
        "direct",
        InterventionSemantics.DIRECT_FROZEN,
        (NNSightInterventionPlan(retained_node, 2.0, 1.0, 1.0, (0,)),),
        (retained_node,),
        (retained_node,),
        False,
        True,
        False,
    )

    try:
        capture = NNSightReplacementModel._verification_run_variant(
            host,
            (1, 2),
            plan,
            baseline.retained_state,
            target_position=2,
            target_token_id=1,
        )
    except Exception as error:
        message = str(error)
        assert (
            "ValueError: Execution complete but `model.1.output.i0` was not provided."
            in message
        )
        assert "Did you call an Envoy out of order?" in message
        raise

    assert capture.origin is CaptureOrigin.INTERVENED_FORWARD
    assert tuple(node for node, _ in capture.feature_values) == (retained_node,)


def test_real_nnsight_production_inject_does_not_preencode_downstream_observations() -> None:
    host = _TinyProductionInjectHost()
    retained_node = FeatureNode(1, 0, 1)
    downstream_node = FeatureNode(3, 0, 1)
    baseline = NNSightReplacementModel._verification_capture_baseline(
        host,
        (1, 2),
        (retained_node, downstream_node),
        target_position=2,
        target_token_id=1,
        retain_attention_state=False,
        retain_direct_freeze_state=True,
    )
    plan = NNSightVariantPlan(
        "direct_early",
        InterventionSemantics.DIRECT_FROZEN,
        (NNSightInterventionPlan(retained_node, 2.0, 1.0, 1.0, (1,)),),
        (downstream_node,),
        (retained_node,),
        False,
        True,
        False,
    )
    try:
        capture = NNSightReplacementModel._verification_run_variant(
            host,
            (1, 2),
            plan,
            baseline.retained_state,
            target_position=2,
            target_token_id=1,
        )
    except Exception as error:
        original = getattr(error, "original", None)
        assert type(original) is Mediator.OutOfOrderError
        assert (
            "layers.1.post_feedforward_layernorm.output.i0" in str(error)
        )
        raise

    assert capture.origin is CaptureOrigin.INTERVENED_FORWARD
    assert tuple(node for node, _ in capture.feature_values) == (
        retained_node,
        downstream_node,
    )


def test_nnsight_translation_preserves_topology_and_exact_direct_delta() -> None:
    variants = plan_behavioral_variants(_request())
    direct = next(item for item in variants if item.kind is VariantKind.DIRECT_DOUBLE)
    clt = _translate_variant(direct, architecture="clt", n_layers=10, observed_nodes=())
    assert clt.freeze_attention is True
    assert clt.freeze_feature_outputs is True
    assert clt.freeze_layernorm_denominators is True
    assert clt.interventions[0].exact_graph_delta == 2.0
    assert clt.interventions[0].output_layers == tuple(range(2, 10))
    plt = _translate_variant(direct, architecture="plt", n_layers=10, observed_nodes=())
    assert plt.interventions[0].output_layers == (2,)

    propagated = next(item for item in variants if item.kind is VariantKind.NECESSITY_HIGH)
    translated = _translate_variant(
        propagated, architecture="clt", n_layers=10, observed_nodes=()
    )
    assert translated.freeze_attention is True
    assert translated.freeze_feature_outputs is False
    assert translated.freeze_layernorm_denominators is False
    assert translated.interventions[0].exact_graph_delta is None


def test_nnsight_adapter_retains_only_selective_union_isolates_variants_and_cleans_up() -> None:
    request = _request(max_variants=2)
    execution = _execution_request(request)
    model = _MockSelectiveNNSightModel()
    sync_after_release: list[bool] = []
    result = NNSightInterventionRuntime(
        model,
        synchronize=lambda model: sync_after_release.append(model.released),
        ordering_admission_mode=OrderingAdmissionMode.QUALIFIED,
    ).evaluate(execution)
    assert result.status is RuntimeExecutionStatus.COMPLETE
    assert result.cleanup_completed is True
    assert model.released is True
    assert sync_after_release[-1] is True
    assert model.health_checks == 1
    expected_retained = tuple(
        sorted(
            set(execution.observed_downstream_nodes)
            | {
                intervention.node
                for variant in execution.variants
                for intervention in variant.interventions
            }
        )
    )
    assert model.baseline_calls == [expected_retained]
    assert len(model.variant_plans) == len(execution.variants)
    assert [item.variant_id for item in result.observations] == [
        item.variant_id for item in execution.variants
    ]
    assert result.baseline is not None
    assert {item.preactivation for item in result.baseline.feature_values}.isdisjoint(
        {
            item.preactivation
            for observation in result.observations
            for item in (
                observation.downstream_feature_values
                + observation.intervention_feature_values
            )
        }
    )
    assert all(
        {item.node for item in observation.downstream_feature_values}.issubset(
            set(execution.observed_downstream_nodes)
        )
        for observation in result.observations
    )


def test_nnsight_adapter_executes_with_gemma3_nested_text_config() -> None:
    request = _execution_request(_request(max_variants=1))
    model = _MockSelectiveNNSightModel()
    model.config = _MockGemma3Config()

    result = NNSightInterventionRuntime(
        model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)

    assert result.status is RuntimeExecutionStatus.COMPLETE
    assert result.refusal is None
    assert model.baseline_calls


def test_nnsight_adapter_refuses_unsupported_backend_without_starting_trace() -> None:
    request = _execution_request(_request(max_variants=1))
    model = _MockSelectiveNNSightModel()
    model.backend = "transformer_lens"
    result = NNSightInterventionRuntime(model, synchronize=lambda model: None).evaluate(request)
    assert result.status is RuntimeExecutionStatus.REFUSED
    assert result.refusal is not None and result.refusal.code == "unsupported_backend"
    assert model.baseline_calls == []


def test_nnsight_adapter_refuses_unqualified_ordering_and_canonical_zero_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _execution_request(_request(max_variants=3))
    model = _MockSelectiveNNSightModel()
    unqualified = NNSightInterventionRuntime(
        model, synchronize=lambda model: None
    ).evaluate(request)
    assert unqualified.status is RuntimeExecutionStatus.REFUSED
    assert unqualified.refusal is not None and unqualified.refusal.code == "ordering_unqualified"
    assert unqualified.ordering_admission_mode is None
    assert model.baseline_calls == []

    model = _MockSelectiveNNSightModel()
    model.verification_intervened_capture_ordering_qualified = False
    false_qualification = NNSightInterventionRuntime(
        model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.QUALIFIED,
    ).evaluate(request)
    assert false_qualification.status is RuntimeExecutionStatus.REFUSED
    assert false_qualification.refusal is not None
    assert false_qualification.refusal.code == "ordering_unqualified"
    assert false_qualification.ordering_admission_mode is OrderingAdmissionMode.QUALIFIED

    model = _MockSelectiveNNSightModel()
    model.zero_positions = slice(4, 5)
    zeroed = NNSightInterventionRuntime(
        model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    assert zeroed.status is RuntimeExecutionStatus.REFUSED
    assert zeroed.refusal is not None and zeroed.refusal.code == "canonical_zero_position"
    assert zeroed.ordering_admission_mode is OrderingAdmissionMode.CANDIDATE_SMOKE

    model = _MockSelectiveNNSightModel()
    monkeypatch.setattr(model.transcoders, "d_transcoder", 5)
    out_of_bounds = NNSightInterventionRuntime(
        model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    assert out_of_bounds.status is RuntimeExecutionStatus.REFUSED
    assert out_of_bounds.refusal is not None and out_of_bounds.refusal.code == "invalid_bounds"


def test_nnsight_adapter_converts_trace_and_cleanup_failures_to_typed_results() -> None:
    request = _execution_request(_request(max_variants=1))
    trace_failure_model = _MockSelectiveNNSightModel(fail_variant=True)
    failed = NNSightInterventionRuntime(
        trace_failure_model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    assert failed.status is RuntimeExecutionStatus.REFUSED
    assert failed.refusal is not None and failed.refusal.code == "runtime_failure"
    assert failed.cleanup_completed is True
    assert trace_failure_model.released is True

    baseline_failure_model = _MockSelectiveNNSightModel(fail_baseline=True)
    baseline_failed = NNSightInterventionRuntime(
        baseline_failure_model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    assert baseline_failed.status is RuntimeExecutionStatus.REFUSED
    assert baseline_failed.refusal is not None and baseline_failed.refusal.code == "runtime_failure"
    assert baseline_failed.cleanup_completed is True
    assert baseline_failure_model.released is True
    assert baseline_failure_model.health_checks == 1

    cleanup_failure_model = _MockSelectiveNNSightModel(fail_release=True)
    cleanup_failed = NNSightInterventionRuntime(
        cleanup_failure_model,
        synchronize=lambda model: None,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    assert cleanup_failed.status is RuntimeExecutionStatus.PARTIAL
    assert cleanup_failed.refusal is not None and cleanup_failed.refusal.code == "cleanup_failure"
    assert cleanup_failed.cleanup_completed is False


def test_nnsight_ordering_admission_is_canonical_fingerprinted_evidence() -> None:
    request = _request(max_variants=2)
    report = verify_behavior(
        request,
        NNSightInterventionRuntime(
            _MockSelectiveNNSightModel(),
            synchronize=lambda model: None,
            ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
        ),
    )
    document = json.loads(report.to_json())
    assert report.ordering_admission_mode is OrderingAdmissionMode.CANDIDATE_SMOKE
    assert document["report"]["ordering_admission_mode"] == "candidate_smoke"
    assert document["report"]["raw_execution"]["ordering_admission_mode"] == (
        "candidate_smoke"
    )

    qualified_execution = replace(
        report.raw_execution,
        ordering_admission_mode=OrderingAdmissionMode.QUALIFIED,
    )
    qualified_report = replace(
        report,
        ordering_admission_mode=OrderingAdmissionMode.QUALIFIED,
        raw_execution=qualified_execution,
    )
    assert qualified_report.evidence_fingerprint != report.evidence_fingerprint


def test_report_json_is_canonical_versioned_and_fingerprinted() -> None:
    request = _request(include_alias=True)
    report = verify_behavior(
        request,
        DeterministicInterventionRuntime(
            _baseline(request), _exact_scripts(request)
        ),
    )
    first = report.to_json()
    second = report.to_json()
    document = json.loads(first)
    assert first == second
    assert document["schema"] == "behavioral_faithfulness_report"
    assert document["schema_version"] == 2
    assert document["evidence_fingerprint"] == report.evidence_fingerprint
    assert document["report"]["trace_identity"]["provider_fingerprint"] == "provider-1"
    assert document["report"]["variant_recipes"][0]["kind"] == "no_op"
    selection = document["report"]["alias_selections"][0]
    assert selection["selection_evidence"]["selection_policy_id"] == (
        "alias_decoder_match_v1"
    )
    assert selection["selection_evidence"]["comparison_evidence_fingerprint"] == (
        "alias-comparison-1"
    )
    changed_alias = replace(
        report.alias_selections[0],
        selection_evidence=replace(
            report.alias_selections[0].selection_evidence,
            least_squares_coefficient=1.25,
        ),
    )
    assert replace(report, alias_selections=(changed_alias,)).evidence_fingerprint != (
        report.evidence_fingerprint
    )
    alias_recipe = next(
        item
        for item in document["report"]["variant_recipes"]
        if item["kind"] == "alias_substitution"
    )
    assert alias_recipe["semantics"] == "propagated_frozen_attention"
    assert alias_recipe["interventions"][0]["graph_baseline_value"] == 2.0
    assert alias_recipe["interventions"][0]["graph_delta"] == -2.0
