from __future__ import annotations

from .contracts import (
    AcceptedGraphView,
    AliasComparatorStatus,
    AliasSubstitution,
    BehavioralAggregateMetrics,
    BehavioralFaithfulnessReport,
    BehavioralVerificationRequest,
    DownstreamClosure,
    EvidenceCompleteness,
    FaithfulnessVerdict,
    FeatureEvidence,
    InterventionExecutionRequest,
    InterventionExecutionResult,
    InterventionSemantics,
    InterventionVariant,
    PreactivationIntervention,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    VariantEvidence,
    VariantKind,
)
from .runtime import InterventionRuntimePort


def _order(feature: FeatureEvidence) -> tuple[float, int, int, int]:
    return (-abs(feature.target_influence), feature.node.layer, feature.node.position, feature.node.feature)


def select_direct_features(
    graph: AcceptedGraphView,
    *,
    sample_count: int,
) -> tuple[FeatureEvidence, ...]:
    """Select the exact deterministic direct-probe prefix used by the planner."""

    remaining = sorted(
        (
            item
            for item in graph.features
            if item.selected and item.baseline_preactivation != 0
        ),
        key=_order,
    )
    chosen: list[FeatureEvidence] = []
    while remaining and len(chosen) < sample_count:
        def diversity(item: FeatureEvidence) -> tuple[int, int, float, int, int, int]:
            return (
                sum((prior.target_influence >= 0) == (item.target_influence >= 0) for prior in chosen),
                sum(
                    (prior.node.layer, prior.node.position) == (item.node.layer, item.node.position)
                    for prior in chosen
                ),
                *_order(item),
            )

        feature = min(remaining, key=diversity)
        remaining.remove(feature)
        chosen.append(feature)
    return tuple(chosen)


def _necessity_pair(request: BehavioralVerificationRequest) -> tuple[FeatureEvidence, FeatureEvidence] | None:
    selected = sorted((item for item in request.graph.features if item.selected), key=_order)
    if not selected:
        return None
    high = selected[0]
    controls = [
        item
        for item in request.graph.features
        if not item.selected and item.node.layer == high.node.layer
    ]
    if not controls:
        return None
    control = min(
        controls,
        key=lambda item: (
            item.node.position != high.node.position,
            abs(abs(item.baseline_preactivation) - abs(high.baseline_preactivation)),
            abs(item.target_influence),
            item.node,
        ),
    )
    return high, control


def _alias_comparator(
    request: BehavioralVerificationRequest,
) -> tuple[AliasSubstitution, FeatureEvidence] | None:
    feature_by_node = {item.node: item for item in request.graph.features}
    for alias in request.aliases[: request.policy.alias_sample_count]:
        source = feature_by_node[alias.source]
        candidates = [
            candidate
            for candidate in alias.control_candidates
            if candidate.node.layer == source.node.layer
            and not feature_by_node[candidate.node].selected
        ]
        if candidates:
            chosen = min(candidates, key=lambda item: (item.similarity_to_source, item.node))
            return alias, feature_by_node[chosen.node]
    return None


def plan_behavioral_variants(request: BehavioralVerificationRequest) -> tuple[InterventionVariant, ...]:
    """Plan deterministic probes. Baseline capture is separate; no-op is variant zero."""

    variants = [
        InterventionVariant(
            "no_op", VariantKind.NO_OP, InterventionSemantics.DIRECT_FROZEN, (), 0.0
        )
    ]
    necessity_pair = _necessity_pair(request)
    alias_comparator = _alias_comparator(request)
    necessity_slots = 2 if necessity_pair is not None else 0
    minimum_direct_slots = min(1, request.policy.direct_sample_count)
    alias_slots = 0
    if alias_comparator is not None and (
        request.policy.max_variants >= 1 + minimum_direct_slots + necessity_slots + 3
    ):
        alias_slots = 3
    direct_slots = max(
        0,
        request.policy.max_variants - 1 - necessity_slots - alias_slots,
    )
    for feature in select_direct_features(
        request.graph,
        sample_count=request.policy.direct_sample_count,
    )[:direct_slots]:
        node = feature.node
        variants.append(
            InterventionVariant(
                f"direct_double_l{node.layer}_p{node.position}_f{node.feature}",
                VariantKind.DIRECT_DOUBLE,
                InterventionSemantics.DIRECT_FROZEN,
                (
                    PreactivationIntervention(
                        node,
                        2.0 * feature.baseline_preactivation,
                        feature.baseline_preactivation,
                        feature.baseline_preactivation,
                    ),
                ),
                # Doubling adds one baseline copy: a one-copy graph-column delta.
                feature.target_influence,
                feature.predicted_downstream_feature_deltas,
            )
        )
    if necessity_pair is not None:
        for kind, feature in zip(
            (VariantKind.NECESSITY_HIGH, VariantKind.NECESSITY_CONTROL),
            necessity_pair,
            strict=True,
        ):
            node = feature.node
            variants.append(
                InterventionVariant(
                    f"{kind.value}_l{node.layer}_p{node.position}_f{node.feature}",
                    kind,
                    InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                    (
                        PreactivationIntervention(
                            node,
                            0.0,
                            feature.baseline_preactivation,
                            -feature.baseline_preactivation,
                        ),
                    ),
                    -feature.target_influence,
                )
            )
    feature_by_node = {item.node: item for item in request.graph.features}
    if alias_slots:
        assert alias_comparator is not None
        alias, control = alias_comparator
        source = feature_by_node[alias.source]
        substitute = feature_by_node[alias.substitute]
        variants.extend(
            (
                InterventionVariant(
                    "alias_source_ablation_0",
                    VariantKind.ALIAS_SOURCE_ABLATION,
                    InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                    (
                        PreactivationIntervention(
                            alias.source,
                            0.0,
                            source.baseline_preactivation,
                            -source.baseline_preactivation,
                        ),
                    ),
                    -source.target_influence,
                ),
            InterventionVariant(
                "alias_substitution_0",
                VariantKind.ALIAS_SUBSTITUTION,
                InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                (
                    PreactivationIntervention(
                        alias.source,
                        0.0,
                        source.baseline_preactivation,
                        -source.baseline_preactivation,
                    ),
                    PreactivationIntervention(
                        alias.substitute,
                        alias.substitute_absolute_preactivation,
                        substitute.baseline_preactivation,
                        alias.substitute_absolute_preactivation
                        - substitute.baseline_preactivation,
                    ),
                ),
                alias.predicted_target_delta,
            ),
                InterventionVariant(
                    "alias_control_0",
                    VariantKind.ALIAS_CONTROL,
                    InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                    (
                        PreactivationIntervention(
                            alias.source,
                            0.0,
                            source.baseline_preactivation,
                            -source.baseline_preactivation,
                        ),
                        PreactivationIntervention(
                            control.node,
                            alias.substitute_absolute_preactivation,
                            control.baseline_preactivation,
                            alias.substitute_absolute_preactivation
                            - control.baseline_preactivation,
                        ),
                    ),
                    None,
                ),
            )
        )
    return tuple(variants[: request.policy.max_variants])


def _failure(error: Exception) -> InterventionExecutionResult:
    return InterventionExecutionResult(
        RuntimeExecutionStatus.REFUSED,
        None,
        (),
        RuntimeRefusal("runtime_exception", f"{type(error).__name__}: {error}"),
        0.0,
        False,
    )


def _make_report(
    request: BehavioralVerificationRequest,
    variants: tuple[InterventionVariant, ...],
    result: InterventionExecutionResult,
) -> BehavioralFaithfulnessReport:
    baseline = None if result.baseline is None else result.baseline.target_value
    observations = {item.variant_id: item for item in result.observations}
    reasons: list[str] = []
    evidence: list[VariantEvidence] = []
    no_op_passed: bool | None = None
    downstream_missing = False
    if baseline is None:
        reasons.append("baseline capture unavailable")
    for variant in variants:
        observation = observations.get(variant.variant_id)
        realized = None if baseline is None or observation is None else observation.target_value - baseline
        closure = (
            None
            if realized is None or variant.predicted_target_delta is None
            else realized - variant.predicted_target_delta
        )
        sign_agreement = None
        if realized is not None and variant.predicted_target_delta not in (None, 0.0):
            sign_agreement = (realized > 0) == (variant.predicted_target_delta > 0)
        baseline_features = (
            {} if result.baseline is None else {item.node: item.preactivation for item in result.baseline.feature_values}
        )
        predicted_downstream = {
            item.node: item.preactivation for item in variant.predicted_downstream_feature_deltas
        }
        observed_downstream = (
            {} if observation is None else {
                item.node: item.preactivation for item in observation.downstream_feature_values
            }
        )
        for node in predicted_downstream:
            if node not in baseline_features or node not in observed_downstream:
                downstream_missing = True
                reasons.append(f"downstream closure unavailable: {variant.variant_id}:{node}")
        downstream = tuple(
            DownstreamClosure(
                node=node,
                baseline_preactivation=baseline_features[node],
                observed_preactivation=observed_downstream[node],
                predicted_delta=predicted_delta,
                observed_delta=observed_downstream[node] - baseline_features[node],
                closure_error=(
                    observed_downstream[node] - baseline_features[node] - predicted_delta
                ),
            )
            for node, predicted_delta in predicted_downstream.items()
            if node in baseline_features and node in observed_downstream
        )
        evidence.append(
            VariantEvidence(
                variant.variant_id,
                variant.kind,
                variant.predicted_target_delta,
                realized,
                closure,
                sign_agreement,
                downstream,
            )
        )
        if observation is None:
            reasons.append(f"variant not evaluated: {variant.variant_id}")
        if variant.kind is VariantKind.NO_OP and realized is not None and baseline is not None:
            tolerance = request.policy.no_op_absolute_tolerance + (
                request.policy.no_op_relative_tolerance * abs(baseline)
            )
            no_op_passed = abs(realized) <= tolerance
            if not no_op_passed:
                reasons.append("required no-op changed the target functional")
    if not result.cleanup_completed:
        reasons.append("runtime teardown was not confirmed")
    if baseline is None or no_op_passed is not True or not result.cleanup_completed:
        completeness = EvidenceCompleteness.UNKNOWN
    elif (
        result.status is RuntimeExecutionStatus.COMPLETE
        and len(observations) == len(variants)
        and not downstream_missing
    ):
        completeness = EvidenceCompleteness.COMPLETE
    else:
        completeness = EvidenceCompleteness.PARTIAL

    direct = [item for item in evidence if item.kind is VariantKind.DIRECT_DOUBLE]
    direct_closure = [abs(item.closure_error) for item in direct if item.closure_error is not None]
    direct_sign = [item.sign_agreement for item in direct if item.sign_agreement is not None]
    necessity_high = next(
        (item.realized_target_delta for item in evidence if item.kind is VariantKind.NECESSITY_HIGH),
        None,
    )
    necessity_control = next(
        (item.realized_target_delta for item in evidence if item.kind is VariantKind.NECESSITY_CONTROL),
        None,
    )
    aliases = [item for item in evidence if item.kind is VariantKind.ALIAS_SUBSTITUTION]
    alias_deltas = [abs(item.realized_target_delta) for item in aliases if item.realized_target_delta is not None]
    alias_closure = [abs(item.closure_error) for item in aliases if item.closure_error is not None]
    alias_source = next(
        (
            item.realized_target_delta
            for item in evidence
            if item.kind is VariantKind.ALIAS_SOURCE_ABLATION
        ),
        None,
    )
    alias_substitution = next(
        (
            item.realized_target_delta
            for item in evidence
            if item.kind is VariantKind.ALIAS_SUBSTITUTION
        ),
        None,
    )
    alias_control = next(
        (item.realized_target_delta for item in evidence if item.kind is VariantKind.ALIAS_CONTROL),
        None,
    )
    alias_planned = {
        item.kind
        for item in variants
        if item.kind
        in (
            VariantKind.ALIAS_SOURCE_ABLATION,
            VariantKind.ALIAS_SUBSTITUTION,
            VariantKind.ALIAS_CONTROL,
        )
    }
    if not alias_planned:
        alias_status = AliasComparatorStatus.NOT_APPLICABLE
    elif None not in (alias_source, alias_substitution, alias_control):
        alias_status = AliasComparatorStatus.COMPLETE
    else:
        alias_status = AliasComparatorStatus.PARTIAL
    substitution_vs_ablation = (
        alias_substitution - alias_source
        if alias_substitution is not None and alias_source is not None
        else None
    )
    control_vs_ablation = (
        alias_control - alias_source
        if alias_control is not None and alias_source is not None
        else None
    )
    downstream_errors = [
        abs(item.closure_error)
        for variant_evidence in evidence
        for item in variant_evidence.downstream_closure
        if item.closure_error is not None
    ]
    metrics = BehavioralAggregateMetrics(
        direct_mean_abs_closure=(sum(direct_closure) / len(direct_closure) if direct_closure else None),
        direct_sign_agreement=(
            sum(agreement is True for agreement in direct_sign) / len(direct_sign)
            if direct_sign
            else None
        ),
        necessity_high_vs_control_separation=(
            abs(necessity_high) - abs(necessity_control)
            if necessity_high is not None and necessity_control is not None
            else None
        ),
        alias_mean_abs_target_delta=(sum(alias_deltas) / len(alias_deltas) if alias_deltas else None),
        alias_mean_abs_closure=(sum(alias_closure) / len(alias_closure) if alias_closure else None),
        alias_substitution_vs_source_ablation=substitution_vs_ablation,
        alias_control_vs_source_ablation=control_vs_ablation,
        alias_substitution_advantage=(
            abs(substitution_vs_ablation) - abs(control_vs_ablation)
            if substitution_vs_ablation is not None and control_vs_ablation is not None
            else None
        ),
        downstream_mean_abs_closure=(
            sum(downstream_errors) / len(downstream_errors) if downstream_errors else None
        ),
    )

    calibration = request.policy.calibration
    if result.status is not RuntimeExecutionStatus.COMPLETE:
        verdict = FaithfulnessVerdict.UNKNOWN
        reasons.append("non-complete runtime execution cannot support a faithfulness verdict")
    elif calibration is None:
        verdict = FaithfulnessVerdict.UNKNOWN
        reasons.append("no frozen behavioral calibration is attached")
    elif completeness is EvidenceCompleteness.UNKNOWN:
        verdict = FaithfulnessVerdict.UNKNOWN
    else:
        checks: list[bool] = []
        missing = False
        for threshold, metric, less_equal in (
            (calibration.direct_max_mean_abs_closure, metrics.direct_mean_abs_closure, True),
            (calibration.direct_min_sign_agreement, metrics.direct_sign_agreement, False),
            (
                calibration.necessity_min_high_control_separation,
                metrics.necessity_high_vs_control_separation,
                False,
            ),
            (calibration.alias_max_mean_abs_closure, metrics.alias_mean_abs_closure, True),
        ):
            if threshold is None:
                continue
            if metric is None:
                missing = True
            else:
                checks.append(metric <= threshold if less_equal else metric >= threshold)
        if any(check is False for check in checks):
            verdict = FaithfulnessVerdict.CONTRADICTED
        elif missing or not checks:
            verdict = FaithfulnessVerdict.INCONCLUSIVE
        else:
            verdict = FaithfulnessVerdict.SUPPORTED
    return BehavioralFaithfulnessReport(
        policy_id=request.policy.policy_id,
        calibration_id=None if calibration is None else calibration.calibration_id,
        trace_identity=request.identity,
        evidence_completeness=completeness,
        verdict=verdict,
        runtime_status=result.status,
        target=request.target.functional,
        baseline_target_value=baseline,
        variants_planned=len(variants),
        variants_completed=len(observations),
        no_op_required=True,
        no_op_passed=no_op_passed,
        alias_comparator_status=alias_status,
        alias_selections=request.aliases,
        variant_recipes=variants,
        evidence=tuple(evidence),
        metrics=metrics,
        raw_execution=result,
        reasons=tuple(reasons),
        refusal=result.refusal,
        ordering_admission_mode=result.ordering_admission_mode,
    )


def verify_behavior(
    request: BehavioralVerificationRequest, runtime: InterventionRuntimePort
) -> BehavioralFaithfulnessReport:
    variants = plan_behavioral_variants(request)
    execution = InterventionExecutionRequest(
        identity=request.identity,
        target=request.target,
        variants=variants,
        observed_downstream_nodes=tuple(
            sorted(
                {
                    value.node
                    for variant in variants
                    for value in variant.predicted_downstream_feature_deltas
                }
            )
        ),
        deadline_seconds=request.policy.max_seconds,
        cleanup_reserve_seconds=request.policy.cleanup_reserve_seconds,
        predicted_baseline_seconds=request.policy.predicted_baseline_seconds,
        predicted_variant_seconds=request.policy.predicted_variant_seconds,
    )
    try:
        result = runtime.evaluate(execution)
    except Exception as error:  # Runtime failures are behavioral unknown, not caller misuse.
        result = _failure(error)
    return _make_report(request, variants, result)
