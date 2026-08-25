from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

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
    verify_behavior,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities
from circuit_tracer.verification.nnsight_runtime import (
    CaptureOrigin,
    NNSightVariantPlan,
    SelectiveProbeCapture,
    _invoke_ordered_hook_families,
    _provider_activation_delta,
    _skip_transcoder_correction,
    _translate_variant,
)


def _request(**policy: object) -> BehavioralVerificationRequest:
    high = FeatureNode(2, 4, 10)
    negative = FeatureNode(4, 5, 11)
    other = FeatureNode(2, 6, 12)
    control = FeatureNode(2, 4, 99)
    alias_control = FeatureNode(2, 7, 98)
    downstream_a = FeatureNode(6, 4, 200)
    downstream_b = FeatureNode(7, 5, 201)
    downstream_c = FeatureNode(8, 6, 202)
    graph = AcceptedGraphView(
        "graph-1",
        (
            FeatureEvidence(high, 2.0, 0.8, True, 0, (FeatureValue(downstream_a, 0.4),)),
            FeatureEvidence(negative, -3.0, -0.5, True, 1, (FeatureValue(downstream_b, -0.25),)),
            FeatureEvidence(other, 1.0, 0.2, True, 2, (FeatureValue(downstream_c, 0.1),)),
            FeatureEvidence(control, 2.1, 0.01, False),
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
        ),
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
    assert len(variants) == 8  # baseline capture is deliberately not counted
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
    assert [item.interventions[0].node.feature for item in necessity] == [10, 99]
    assert all(item.interventions[0].absolute_value == 0.0 for item in necessity)
    alias = next(item for item in variants if item.kind is VariantKind.ALIAS_SUBSTITUTION)
    assert alias.kind is VariantKind.ALIAS_SUBSTITUTION
    assert [item.absolute_value for item in alias.interventions] == [0.0, 5.0]


def test_limits_and_graph_identity_are_structural_caller_errors() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        BehavioralProbePolicy(policy_id="unversioned")
    with pytest.raises(ValueError, match="max_variants"):
        BehavioralProbePolicy(max_variants=9)
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


def test_frozen_calibration_separates_complete_evidence_from_supported_verdict() -> None:
    calibration = FrozenBehavioralCalibration(
        calibration_id="granite-behavior-v1",
        policy_id="behavioral_closure_v1",
        direct_max_mean_abs_closure=0.01,
        direct_min_sign_agreement=1.0,
        necessity_min_high_control_separation=0.5,
        alias_max_mean_abs_closure=0.01,
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
    assert report.metrics.direct_sign_agreement == pytest.approx(1.0)
    assert report.metrics.necessity_high_vs_control_separation == pytest.approx(0.79)
    assert report.metrics.alias_mean_abs_target_delta == pytest.approx(0.2)
    assert report.metrics.alias_mean_abs_closure == pytest.approx(0.0)
    assert report.alias_comparator_status is AliasComparatorStatus.COMPLETE
    assert report.metrics.alias_substitution_vs_source_ablation == pytest.approx(0.6)
    assert report.metrics.alias_control_vs_source_ablation == pytest.approx(0.8)
    assert report.metrics.alias_substitution_advantage == pytest.approx(-0.2)
    assert report.metrics.downstream_mean_abs_closure == pytest.approx(0.0)
    first_direct = next(item for item in report.evidence if item.kind is VariantKind.DIRECT_DOUBLE)
    assert first_direct.downstream_closure[0].node == FeatureNode(6, 4, 200)
    assert first_direct.downstream_closure[0].predicted_delta == pytest.approx(0.4)
    assert first_direct.downstream_closure[0].observed_delta == pytest.approx(0.4)
    assert first_direct.downstream_closure[0].closure_error == pytest.approx(0.0)
    assert report.raw_execution.baseline is not None
    assert report.raw_execution.baseline.raw_target_logits == (9.0, 11.0)
    assert any(item.downstream_closure for item in report.evidence)
    assert report.sufficiency is SufficiencyStatus.UNKNOWN


def test_calibrated_failed_threshold_is_contradicted_and_missing_metric_is_inconclusive() -> None:
    direct_calibration = FrozenBehavioralCalibration(
        "direct-v1",
        "behavioral_closure_v1",
        direct_max_mean_abs_closure=0.01,
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
        alias_max_mean_abs_closure=0.1,
    )
    request_without_alias_probe = _request(
        calibration=alias_calibration,
        max_variants=6,
    )
    variants = plan_behavioral_variants(request_without_alias_probe)
    scripts = _exact_scripts(request_without_alias_probe)
    inconclusive = verify_behavior(
        request_without_alias_probe,
        DeterministicInterventionRuntime(_baseline(request_without_alias_probe), scripts),
    )
    assert inconclusive.evidence_completeness is EvidenceCompleteness.COMPLETE
    assert inconclusive.verdict is FaithfulnessVerdict.INCONCLUSIVE


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


def test_alias_comparator_is_not_applicable_without_control_similarity_data() -> None:
    request = _request()
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
    assert not any(item.kind.value.startswith("alias_") for item in variants)
    report = verify_behavior(
        without_controls,
        DeterministicInterventionRuntime(
            _baseline(without_controls), _exact_scripts(without_controls)
        ),
    )
    assert report.alias_comparator_status is AliasComparatorStatus.NOT_APPLICABLE


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
        def apply_activation_function(layer: int, values: torch.Tensor) -> torch.Tensor:
            return torch.relu(values)

        @staticmethod
        def compute_skip(layer: int, values: torch.Tensor) -> torch.Tensor:
            return values * 2

    provider = _Provider()
    assert _provider_activation_delta(provider, 0, torch.tensor(-1.0), 3.0).item() == 3.0
    correction = _skip_transcoder_correction(
        provider,
        0,
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    assert torch.equal(correction, torch.tensor([2.0, 4.0]))


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
    request = _execution_request(_request(max_variants=2))
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
    request = _request(max_variants=4)
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
    assert document["schema_version"] == 1
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
    direct_recipe = next(
        item
        for item in document["report"]["variant_recipes"]
        if item["kind"] == "direct_double"
    )
    assert direct_recipe["semantics"] == "direct_frozen"
    assert direct_recipe["interventions"][0]["graph_baseline_value"] == 2.0
    assert direct_recipe["interventions"][0]["graph_delta"] == 2.0
