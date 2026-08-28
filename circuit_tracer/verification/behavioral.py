from __future__ import annotations

import math

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
    FeatureNode,
    InterventionExecutionRequest,
    InterventionExecutionResult,
    InterventionSemantics,
    InterventionVariant,
    PreactivationIntervention,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    VariantEvidence,
    VariantKind,
    DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA,
    DEFAULT_RELATIVE_ERROR_EPSILON,
)
from .runtime import InterventionRuntimePort


def _order(feature: FeatureEvidence) -> tuple[float, int, int, int]:
    return (-abs(feature.target_influence), feature.node.layer, feature.node.position, feature.node.feature)


def select_direct_features(
    graph: AcceptedGraphView,
    *,
    sample_count: int,
    min_abs_predicted_target_delta: float = (
        DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA
    ),
) -> tuple[FeatureEvidence, ...]:
    """Select the exact deterministic direct-probe prefix used by the planner."""

    return _select_features(
        graph,
        sample_count=sample_count,
        min_abs_predicted_target_delta=min_abs_predicted_target_delta,
    )


def _select_features(
    graph: AcceptedGraphView,
    *,
    sample_count: int,
    min_abs_predicted_target_delta: float,
    required_nodes: tuple[FeatureNode, ...] = (),
) -> tuple[FeatureEvidence, ...]:
    if sample_count < 0 or min_abs_predicted_target_delta < 0:
        raise ValueError("sample count and direct eligibility threshold must be nonnegative")

    remaining = sorted(
        (
            item
            for item in graph.features
            if item.selected and item.baseline_preactivation != 0
            and abs(item.target_influence) >= min_abs_predicted_target_delta
        ),
        key=_order,
    )
    required = set(required_nodes)
    if len(required) != len(required_nodes) or len(required) > sample_count:
        raise ValueError("required necessity nodes must be unique and fit the sample")
    chosen = sorted(
        (item for item in remaining if item.node in required),
        key=_order,
    )
    if len(chosen) != len(required):
        raise ValueError("required necessity node is not an eligible selected feature")
    remaining = [item for item in remaining if item.node not in required]
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


def select_necessity_features(
    graph: AcceptedGraphView,
    *,
    sample_count: int,
    min_abs_predicted_target_delta: float = (
        DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA
    ),
    required_nodes: tuple[FeatureNode, ...] = (),
) -> tuple[FeatureEvidence, ...]:
    """Select deterministic high-effect anchors before controls are materialized."""

    return _select_features(
        graph,
        sample_count=sample_count,
        min_abs_predicted_target_delta=min_abs_predicted_target_delta,
        required_nodes=required_nodes,
    )


def select_necessity_pairs(
    graph: AcceptedGraphView,
    *,
    sample_count: int,
    min_abs_predicted_target_delta: float = (
        DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA
    ),
    required_nodes: tuple[FeatureNode, ...] = (),
) -> tuple[tuple[FeatureEvidence, FeatureEvidence], ...]:
    """Resolve the explicit, identity-bound control owned by each selected anchor."""

    control_by_owner = {
        item.necessity_control_for: item
        for item in graph.features
        if item.necessity_control_for is not None
    }
    return tuple(
        (high, control_by_owner[high.node])
        for high in select_necessity_features(
            graph,
            sample_count=sample_count,
            min_abs_predicted_target_delta=min_abs_predicted_target_delta,
            required_nodes=required_nodes,
        )
        if high.node in control_by_owner
    )


def _alias_substitution(
    request: BehavioralVerificationRequest,
    pairs: tuple[tuple[FeatureEvidence, FeatureEvidence], ...],
) -> AliasSubstitution | None:
    high_nodes = {high.node for high, _control in pairs}
    for alias in request.aliases[: request.policy.alias_sample_count]:
        if alias.source in high_nodes:
            return alias
    return None


def _necessity_variants(
    pairs: tuple[tuple[FeatureEvidence, FeatureEvidence], ...],
) -> tuple[InterventionVariant, ...]:
    highs: list[InterventionVariant] = []
    controls: list[InterventionVariant] = []
    for pair_index, (high, control) in enumerate(pairs):
        for kind, feature, destination in (
            (VariantKind.NECESSITY_HIGH, high, highs),
            (VariantKind.NECESSITY_CONTROL, control, controls),
        ):
            node = feature.node
            destination.append(
                InterventionVariant(
                    f"{kind.value}_pair{pair_index}_l{node.layer}_p{node.position}_f{node.feature}",
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
    return (*highs, *controls)


def plan_behavioral_variants(request: BehavioralVerificationRequest) -> tuple[InterventionVariant, ...]:
    """Plan bounded deterministic probes with no-op, necessity, direct, then alias priority."""

    variants = [
        InterventionVariant(
            "no_op", VariantKind.NO_OP, InterventionSemantics.DIRECT_FROZEN, (), 0.0
        )
    ]
    calibration = request.policy.calibration
    direct_min_effect = (
        DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA
        if calibration is None
        else calibration.direct_min_abs_predicted_target_delta
    )
    required_alias_sources = tuple(
        dict.fromkeys(
            alias.source
            for alias in request.aliases[: request.policy.alias_sample_count]
        )
    )[: request.policy.necessity_sample_count]
    pairs = select_necessity_pairs(
        request.graph,
        sample_count=request.policy.necessity_sample_count,
        min_abs_predicted_target_delta=direct_min_effect,
        required_nodes=required_alias_sources,
    )
    alias = _alias_substitution(request, pairs)
    necessity_variants = _necessity_variants(pairs)

    # A churn probe reuses the matching necessity-high ablation as its source
    # baseline, so the whole comparator stays at eight variants rather than
    # reserving an independent source/control triplet.
    churn_slots = 1 + len(necessity_variants) + 1
    if alias is not None and churn_slots <= request.policy.max_variants:
        variants.extend(necessity_variants)
        feature_by_node = {item.node: item for item in request.graph.features}
        source = feature_by_node[alias.source]
        substitute = feature_by_node[alias.substitute]
        variants.append(
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
            )
        )
        return tuple(variants)

    direct_slots = max(
        0,
        request.policy.max_variants - 1 - len(necessity_variants),
    )
    direct_features = select_direct_features(
        request.graph,
        sample_count=request.policy.direct_sample_count,
        min_abs_predicted_target_delta=direct_min_effect,
    )
    for feature in direct_features[:direct_slots]:
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
    remaining = request.policy.max_variants - len(variants)
    admitted_pair_count = min(len(pairs), remaining // 2)
    variants.extend(necessity_variants[:admitted_pair_count])
    variants.extend(
        necessity_variants[
            len(pairs) : len(pairs) + admitted_pair_count
        ]
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


def _relative_error(predicted: float, realized: float, *, epsilon: float) -> float:
    return abs(realized - predicted) / max(abs(predicted), abs(realized), epsilon)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _median(values: list[float]) -> float | None:
    return _percentile(values, 0.5)


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for original_index, _value in ordered[start:end]:
            ranks[original_index] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    left_centered = [value - left_mean for value in left_ranks]
    right_centered = [value - right_mean for value in right_ranks]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


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
            sign_agreement = realized * variant.predicted_target_delta > 0
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

    calibration = request.policy.calibration
    relative_epsilon = (
        DEFAULT_RELATIVE_ERROR_EPSILON
        if calibration is None
        else calibration.relative_error_epsilon
    )
    direct = [item for item in evidence if item.kind is VariantKind.DIRECT_DOUBLE]
    direct_closure = [abs(item.closure_error) for item in direct if item.closure_error is not None]
    direct_sign = [item.sign_agreement for item in direct if item.sign_agreement is not None]
    direct_complete = (
        len(direct) == request.policy.direct_sample_count
        and len(direct_closure) == len(direct)
        and len(direct_sign) == len(direct)
    )
    direct_relative = (
        [
            _relative_error(
                item.predicted_target_delta,
                item.realized_target_delta,
                epsilon=relative_epsilon,
            )
            for item in direct
            if item.predicted_target_delta is not None
            and item.realized_target_delta is not None
        ]
        if direct_complete
        else []
    )
    if len(direct_relative) != len(direct):
        direct_relative = []

    necessity_high_evidence = [
        item for item in evidence if item.kind is VariantKind.NECESSITY_HIGH
    ]
    necessity_control_evidence = [
        item for item in evidence if item.kind is VariantKind.NECESSITY_CONTROL
    ]
    necessity_complete = (
        len(necessity_high_evidence) == request.policy.necessity_sample_count
        and len(necessity_control_evidence) == request.policy.necessity_sample_count
        and all(
            item.predicted_target_delta is not None and item.realized_target_delta is not None
            for item in (*necessity_high_evidence, *necessity_control_evidence)
        )
    )
    necessity_predicted = (
        [
            abs(item.predicted_target_delta)
            for item in (*necessity_high_evidence, *necessity_control_evidence)
            if item.predicted_target_delta is not None
        ]
        if necessity_complete
        else []
    )
    necessity_realized = (
        [
            abs(item.realized_target_delta)
            for item in (*necessity_high_evidence, *necessity_control_evidence)
            if item.realized_target_delta is not None
        ]
        if necessity_complete
        else []
    )
    high_realized = (
        [
            abs(item.realized_target_delta)
            for item in necessity_high_evidence
            if item.realized_target_delta is not None
        ]
        if necessity_complete
        else []
    )
    control_realized = (
        [
            abs(item.realized_target_delta)
            for item in necessity_control_evidence
            if item.realized_target_delta is not None
        ]
        if necessity_complete
        else []
    )
    aliases = [item for item in evidence if item.kind is VariantKind.ALIAS_SUBSTITUTION]
    alias_deltas = [abs(item.realized_target_delta) for item in aliases if item.realized_target_delta is not None]
    alias_closure = [abs(item.closure_error) for item in aliases if item.closure_error is not None]
    recipe_by_id = {item.variant_id: item for item in variants}
    alias_recipe = next(
        (item for item in variants if item.kind is VariantKind.ALIAS_SUBSTITUTION),
        None,
    )
    selected_alias = (
        next(
            (
                alias
                for alias in request.aliases
                if alias_recipe is not None
                and tuple(item.node for item in alias_recipe.interventions)
                == (alias.source, alias.substitute)
            ),
            None,
        )
        if alias_recipe is not None
        else None
    )
    alias_source = next(
        (
            item.realized_target_delta
            for item in necessity_high_evidence
            if selected_alias is not None
            and recipe_by_id[item.variant_id].interventions[0].node == selected_alias.source
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
    if not request.aliases:
        alias_status = AliasComparatorStatus.NOT_APPLICABLE
    elif alias_planned and None not in (alias_source, alias_substitution):
        alias_status = AliasComparatorStatus.COMPLETE
    else:
        alias_status = AliasComparatorStatus.PARTIAL
    substitution_vs_ablation = (
        alias_substitution - alias_source
        if alias_substitution is not None and alias_source is not None
        else None
    )
    control_vs_ablation = None
    downstream_errors = [
        abs(item.closure_error)
        for variant_evidence in evidence
        for item in variant_evidence.downstream_closure
        if item.closure_error is not None
    ]
    downstream_relative = [
        _relative_error(
            item.predicted_delta,
            item.observed_delta,
            epsilon=relative_epsilon,
        )
        for variant_evidence in direct
        for item in variant_evidence.downstream_closure
        if item.observed_delta is not None
    ]
    direct_mean_relative = (
        sum(direct_relative) / len(direct_relative) if direct_relative else None
    )
    necessity_high_median = _median(high_realized)
    necessity_control_median = _median(control_realized)
    metrics = BehavioralAggregateMetrics(
        direct_mean_abs_closure=(sum(direct_closure) / len(direct_closure) if direct_closure else None),
        direct_mean_relative_closure=direct_mean_relative,
        direct_max_relative_closure=(max(direct_relative) if direct_relative else None),
        direct_sign_agreement=(
            sum(agreement is True for agreement in direct_sign) / len(direct_sign)
            if direct_complete
            else None
        ),
        necessity_high_vs_control_separation=(
            necessity_high_median - necessity_control_median
            if necessity_high_median is not None and necessity_control_median is not None
            else None
        ),
        necessity_predicted_realized_spearman=_spearman(
            necessity_predicted,
            necessity_realized,
        ),
        necessity_median_high_control_effect_ratio=(
            necessity_high_median / max(necessity_control_median, relative_epsilon)
            if necessity_high_median is not None and necessity_control_median is not None
            else None
        ),
        alias_mean_abs_target_delta=(sum(alias_deltas) / len(alias_deltas) if alias_deltas else None),
        alias_mean_abs_closure=(sum(alias_closure) / len(alias_closure) if alias_closure else None),
        alias_relative_effect_error=(
            _relative_error(
                aliases[0].predicted_target_delta,
                aliases[0].realized_target_delta,
                epsilon=relative_epsilon,
            )
            if len(aliases) == 1
            and aliases[0].predicted_target_delta is not None
            and aliases[0].realized_target_delta is not None
            else None
        ),
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
        downstream_mean_relative_closure=(
            sum(downstream_relative) / len(downstream_relative)
            if downstream_relative
            else None
        ),
        downstream_p95_relative_closure=_percentile(downstream_relative, 0.95),
    )

    if result.status is not RuntimeExecutionStatus.COMPLETE:
        verdict = FaithfulnessVerdict.UNKNOWN
        reasons.append("non-complete runtime execution cannot support a faithfulness verdict")
    elif calibration is None:
        verdict = FaithfulnessVerdict.UNKNOWN
        reasons.append("no frozen behavioral calibration is attached")
    elif no_op_passed is False:
        verdict = FaithfulnessVerdict.CONTRADICTED
        reasons.append("required no-op exceeded the calibrated hard tolerance")
    elif completeness is EvidenceCompleteness.UNKNOWN:
        verdict = FaithfulnessVerdict.UNKNOWN
    else:
        hard_checks: list[bool] = []
        review_checks: list[bool] = []
        missing = False
        alias_applicable = bool(request.aliases)
        hard_threshold_checks = [
            (
                calibration.necessity_min_predicted_realized_spearman,
                metrics.necessity_predicted_realized_spearman,
                False,
            ),
            (
                calibration.necessity_min_median_high_control_effect_ratio,
                metrics.necessity_median_high_control_effect_ratio,
                False,
            ),
            (
                calibration.necessity_min_high_control_separation,
                metrics.necessity_high_vs_control_separation,
                False,
            ),
        ]
        review_threshold_checks: list[tuple[float | None, float | None, bool]] = []
        if alias_applicable:
            hard_threshold_checks.extend(
                (
                    (
                        calibration.alias_max_relative_effect_error,
                        metrics.alias_relative_effect_error,
                        True,
                    ),
                    (
                        calibration.alias_max_mean_abs_closure,
                        metrics.alias_mean_abs_closure,
                        True,
                    ),
                )
            )
        else:
            hard_threshold_checks.extend(
                (
                    (
                        calibration.direct_max_relative_closure,
                        metrics.direct_max_relative_closure,
                        True,
                    ),
                    (
                        calibration.direct_max_mean_abs_closure,
                        metrics.direct_mean_abs_closure,
                        True,
                    ),
                    (
                        calibration.direct_min_sign_agreement,
                        metrics.direct_sign_agreement,
                        False,
                    ),
                    (
                        calibration.downstream_max_p95_relative_closure,
                        metrics.downstream_p95_relative_closure,
                        True,
                    ),
                )
            )
            review_threshold_checks.extend(
                (
                    (
                        calibration.direct_max_mean_relative_closure,
                        metrics.direct_mean_relative_closure,
                        True,
                    ),
                    (
                        calibration.downstream_max_mean_relative_closure,
                        metrics.downstream_mean_relative_closure,
                        True,
                    ),
                )
            )
        for threshold, metric, less_equal in hard_threshold_checks:
            if threshold is None:
                continue
            if metric is None:
                missing = True
            else:
                hard_checks.append(metric <= threshold if less_equal else metric >= threshold)
        for threshold, metric, less_equal in review_threshold_checks:
            if threshold is None:
                continue
            if metric is None:
                missing = True
            else:
                review_checks.append(metric <= threshold if less_equal else metric >= threshold)
        if any(check is False for check in hard_checks):
            verdict = FaithfulnessVerdict.CONTRADICTED
            reasons.append("calibrated behavioral hard limit failed")
        elif missing:
            verdict = FaithfulnessVerdict.UNKNOWN
            reasons.append("calibrated behavioral metric is unavailable")
        elif any(check is False for check in review_checks):
            verdict = FaithfulnessVerdict.INCONCLUSIVE
            reasons.append("calibrated behavioral mean is in the review band")
        elif not hard_checks and not review_checks:
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
