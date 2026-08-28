from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


MAX_PROBE_VARIANTS = 10
MAX_PROBE_SECONDS = 120.0
DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA = 0.25
DEFAULT_RELATIVE_ERROR_EPSILON = 1e-6


class InterventionSemantics(str, Enum):
    DIRECT_FROZEN = "direct_frozen"
    PROPAGATED_FROZEN_ATTENTION = "propagated_frozen_attention"


class TargetFunctional(str, Enum):
    CENTERED_UNSOFTCAPPED_LOGIT = "centered_unsoftcapped_logit"


class VariantKind(str, Enum):
    NO_OP = "no_op"
    DIRECT_DOUBLE = "direct_double"
    NECESSITY_HIGH = "necessity_high"
    NECESSITY_CONTROL = "necessity_control"
    ALIAS_SOURCE_ABLATION = "alias_source_ablation"
    ALIAS_SUBSTITUTION = "alias_substitution"
    ALIAS_CONTROL = "alias_control"


class RuntimeExecutionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    REFUSED = "refused"


class OrderingAdmissionMode(str, Enum):
    CANDIDATE_SMOKE = "candidate_smoke"
    QUALIFIED = "qualified"


class EvidenceCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class FaithfulnessVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    UNKNOWN = "unknown"


class SufficiencyStatus(str, Enum):
    UNKNOWN = "unknown"


class AliasComparatorStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


def _finite(name: str, *values: float) -> None:
    if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite numbers")


def _versioned_policy_id(value: str) -> bool:
    prefix, separator, version = value.rpartition("_v")
    return bool(prefix and separator and version.isdigit())


@dataclass(frozen=True, order=True)
class FeatureNode:
    layer: int
    position: int
    feature: int

    def __post_init__(self) -> None:
        coordinates = (self.layer, self.position, self.feature)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates):
            raise ValueError("feature coordinates must be integers")
        if min(coordinates) < 0:
            raise ValueError("feature coordinates must be nonnegative")


@dataclass(frozen=True)
class FeatureEvidence:
    node: FeatureNode
    baseline_preactivation: float
    target_influence: float
    selected: bool
    selection_rank: int | None = None
    predicted_downstream_feature_deltas: tuple["FeatureValue", ...] = ()
    necessity_control_for: FeatureNode | None = None

    def __post_init__(self) -> None:
        _finite("feature evidence", self.baseline_preactivation, self.target_influence)
        if self.selected != (self.selection_rank is not None):
            raise ValueError("selected features require a rank and controls must not have one")
        if self.selection_rank is not None and self.selection_rank < 0:
            raise ValueError("selection_rank must be nonnegative")
        if self.selected and self.necessity_control_for is not None:
            raise ValueError("selected features cannot be necessity controls")
        nodes = tuple(item.node for item in self.predicted_downstream_feature_deltas)
        if len(set(nodes)) != len(nodes):
            raise ValueError("predicted downstream feature deltas must be unique")


@dataclass(frozen=True)
class AcceptedGraphView:
    graph_fingerprint: str
    features: tuple[FeatureEvidence, ...]

    def __post_init__(self) -> None:
        nodes = tuple(item.node for item in self.features)
        if not self.graph_fingerprint or not self.features:
            raise ValueError("accepted graph requires a fingerprint and features")
        if len(set(nodes)) != len(nodes):
            raise ValueError("accepted graph feature nodes must be unique")
        if not any(item.selected for item in self.features):
            raise ValueError("accepted graph must contain a selected feature")
        feature_by_node = {item.node: item for item in self.features}
        control_owners = [
            item.necessity_control_for
            for item in self.features
            if item.necessity_control_for is not None
        ]
        if len(set(control_owners)) != len(control_owners):
            raise ValueError("necessity controls must have unique selected owners")
        for item in self.features:
            owner_node = item.necessity_control_for
            if owner_node is None:
                continue
            owner = feature_by_node.get(owner_node)
            if owner is None or not owner.selected:
                raise ValueError("necessity control owner must be a selected graph feature")
            if owner.node.layer != item.node.layer:
                raise ValueError("necessity controls must be matched within a layer")


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    graph_fingerprint: str
    provider_fingerprint: str
    semantic_fingerprint: str
    execution_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    target_position: int
    target_token_id: int

    def __post_init__(self) -> None:
        fingerprints = (
            self.graph_fingerprint,
            self.provider_fingerprint,
            self.semantic_fingerprint,
            self.execution_fingerprint,
        )
        if not self.trace_id or not all(fingerprints) or not self.prompt_token_ids:
            raise ValueError("trace identity fields must not be empty")
        integer_fields = (*self.prompt_token_ids, self.target_position, self.target_token_id)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise ValueError("trace token and target coordinates must be integers")
        if not 1 <= self.target_position <= len(self.prompt_token_ids):
            raise ValueError("target_position must be a 1-based prefix length")
        if self.target_token_id < 0:
            raise ValueError("target_token_id must be nonnegative")


@dataclass(frozen=True)
class TargetState:
    functional: TargetFunctional = TargetFunctional.CENTERED_UNSOFTCAPPED_LOGIT

    def __post_init__(self) -> None:
        if self.functional is not TargetFunctional.CENTERED_UNSOFTCAPPED_LOGIT:
            raise ValueError("v1 requires centered unsoftcapped logits")


@dataclass(frozen=True)
class AliasControlCandidate:
    node: FeatureNode
    similarity_to_source: float

    def __post_init__(self) -> None:
        _finite("alias control similarity", self.similarity_to_source)
        if not -1 <= self.similarity_to_source <= 1:
            raise ValueError("alias control similarity must be in [-1, 1]")


@dataclass(frozen=True)
class AliasSelectionEvidence:
    """Exact, identity-bound evidence used to admit one alias recipe."""

    selection_policy_id: str
    calibration_fingerprint: str
    comparison_evidence_fingerprint: str
    baseline_graph_fingerprint: str
    candidate_graph_fingerprint: str
    decoder_fingerprint: str
    decoder_output_topology: str
    qualified_decoder_cosine: float
    observed_decoder_cosine: float
    source_decoder_norm: float
    substitute_decoder_norm: float
    least_squares_coefficient: float

    def __post_init__(self) -> None:
        if not _versioned_policy_id(self.selection_policy_id):
            raise ValueError("alias selection policy_id must end in a numeric _vN version")
        identities = (
            self.calibration_fingerprint,
            self.comparison_evidence_fingerprint,
            self.baseline_graph_fingerprint,
            self.candidate_graph_fingerprint,
            self.decoder_fingerprint,
        )
        if not all(identities):
            raise ValueError("alias selection identities must not be empty")
        if self.decoder_output_topology not in {"complete_downstream_block", "same_layer"}:
            raise ValueError("unsupported alias decoder output topology")
        _finite(
            "alias selection evidence",
            self.qualified_decoder_cosine,
            self.observed_decoder_cosine,
            self.source_decoder_norm,
            self.substitute_decoder_norm,
            self.least_squares_coefficient,
        )
        if not -1 <= self.qualified_decoder_cosine <= 1:
            raise ValueError("qualified decoder cosine must be in [-1, 1]")
        if not -1 <= self.observed_decoder_cosine <= 1:
            raise ValueError("observed decoder cosine must be in [-1, 1]")
        if min(self.source_decoder_norm, self.substitute_decoder_norm) <= 0:
            raise ValueError("alias decoder norms must be positive")


@dataclass(frozen=True)
class AliasSubstitution:
    source: FeatureNode
    substitute: FeatureNode
    substitute_absolute_preactivation: float
    selection_evidence: AliasSelectionEvidence
    predicted_target_delta: float | None = None
    control_candidates: tuple[AliasControlCandidate, ...] = ()

    def __post_init__(self) -> None:
        _finite("alias substitution", self.substitute_absolute_preactivation)
        if self.predicted_target_delta is not None:
            _finite("alias predicted target delta", self.predicted_target_delta)
        if self.source == self.substitute:
            raise ValueError("alias source and substitute must differ")
        nodes = tuple(item.node for item in self.control_candidates)
        if len(set(nodes)) != len(nodes):
            raise ValueError("alias control candidates must be unique")
        if self.source in nodes or self.substitute in nodes:
            raise ValueError("alias control must differ from source and substitute")


@dataclass(frozen=True)
class FrozenBehavioralCalibration:
    """Immutable identity and thresholds fitted outside the verifier."""

    calibration_id: str
    policy_id: str
    direct_min_abs_predicted_target_delta: float = (
        DEFAULT_DIRECT_MIN_ABS_PREDICTED_TARGET_DELTA
    )
    relative_error_epsilon: float = DEFAULT_RELATIVE_ERROR_EPSILON
    direct_max_mean_relative_closure: float | None = None
    direct_max_relative_closure: float | None = None
    direct_max_mean_abs_closure: float | None = None
    direct_min_sign_agreement: float | None = None
    downstream_max_mean_relative_closure: float | None = None
    downstream_max_p95_relative_closure: float | None = None
    necessity_min_predicted_realized_spearman: float | None = None
    necessity_min_median_high_control_effect_ratio: float | None = None
    necessity_min_high_control_separation: float | None = None
    alias_max_relative_effect_error: float | None = None
    alias_max_mean_abs_closure: float | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id or not _versioned_policy_id(self.policy_id):
            raise ValueError("calibration identity fields must not be empty")
        thresholds = (
            self.direct_max_mean_relative_closure,
            self.direct_max_relative_closure,
            self.direct_max_mean_abs_closure,
            self.direct_min_sign_agreement,
            self.downstream_max_mean_relative_closure,
            self.downstream_max_p95_relative_closure,
            self.necessity_min_predicted_realized_spearman,
            self.necessity_min_median_high_control_effect_ratio,
            self.necessity_min_high_control_separation,
            self.alias_max_relative_effect_error,
            self.alias_max_mean_abs_closure,
        )
        if all(value is None for value in thresholds):
            raise ValueError("calibration must contain at least one threshold")
        for value in thresholds:
            if value is not None:
                _finite("calibration thresholds", value)
                if value < 0:
                    raise ValueError("calibration thresholds must be nonnegative")
        _finite(
            "calibration numeric policy",
            self.direct_min_abs_predicted_target_delta,
            self.relative_error_epsilon,
        )
        if self.direct_min_abs_predicted_target_delta <= 0:
            raise ValueError("direct eligibility threshold must be positive")
        if self.relative_error_epsilon != DEFAULT_RELATIVE_ERROR_EPSILON:
            raise ValueError("relative error epsilon must be exactly 1e-6 for v1")
        if self.direct_min_sign_agreement is not None and self.direct_min_sign_agreement > 1:
            raise ValueError("direct_min_sign_agreement must be at most one")
        if (
            self.necessity_min_predicted_realized_spearman is not None
            and self.necessity_min_predicted_realized_spearman > 1
        ):
            raise ValueError("necessity Spearman threshold must be at most one")


@dataclass(frozen=True)
class BehavioralProbePolicy:
    policy_id: str = "behavioral_closure_v1"
    calibration: FrozenBehavioralCalibration | None = None
    max_variants: int = MAX_PROBE_VARIANTS
    max_seconds: float = MAX_PROBE_SECONDS
    direct_sample_count: int = 3
    necessity_sample_count: int = 3
    alias_sample_count: int = 1
    cleanup_reserve_seconds: float = 1.0
    predicted_baseline_seconds: float = 1.0
    predicted_variant_seconds: float = 1.0
    no_op_absolute_tolerance: float = 1e-6
    no_op_relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not _versioned_policy_id(self.policy_id):
            raise ValueError("policy_id must end in a numeric _vN version")
        if self.calibration is not None and self.calibration.policy_id != self.policy_id:
            raise ValueError("calibration policy_id must match the probe policy")
        if not 1 <= self.max_variants <= MAX_PROBE_VARIANTS:
            raise ValueError(f"max_variants must be in [1, {MAX_PROBE_VARIANTS}]")
        if not 0 < self.max_seconds <= MAX_PROBE_SECONDS:
            raise ValueError(f"max_seconds must be in (0, {MAX_PROBE_SECONDS}]")
        values = (
            self.direct_sample_count,
            self.necessity_sample_count,
            self.alias_sample_count,
            self.cleanup_reserve_seconds,
            self.predicted_baseline_seconds,
            self.predicted_variant_seconds,
            self.no_op_absolute_tolerance,
            self.no_op_relative_tolerance,
        )
        if any(value < 0 for value in values):
            raise ValueError("counts, budgets, predictions, and tolerances must be nonnegative")
        _finite(
            "probe policy",
            self.max_seconds,
            self.cleanup_reserve_seconds,
            self.predicted_baseline_seconds,
            self.predicted_variant_seconds,
            self.no_op_absolute_tolerance,
            self.no_op_relative_tolerance,
        )
        if self.cleanup_reserve_seconds >= self.max_seconds:
            raise ValueError("cleanup reserve must be smaller than the deadline")
        if self.calibration is not None:
            if self.direct_sample_count != 3 or self.necessity_sample_count != 3:
                raise ValueError(
                    "calibrated v1 verification requires three direct and necessity samples"
                )
            if (
                self.no_op_absolute_tolerance != 1e-6
                or self.no_op_relative_tolerance != 0.0
            ):
                raise ValueError("calibrated v1 requires an absolute 1e-6 no-op tolerance")


@dataclass(frozen=True)
class BehavioralVerificationRequest:
    identity: TraceIdentity
    graph: AcceptedGraphView
    target: TargetState = TargetState()
    policy: BehavioralProbePolicy = BehavioralProbePolicy()
    aliases: tuple[AliasSubstitution, ...] = ()

    def __post_init__(self) -> None:
        if self.identity.graph_fingerprint != self.graph.graph_fingerprint:
            raise ValueError("trace and accepted graph fingerprints differ")
        graph_nodes = {feature.node for feature in self.graph.features}
        if any(alias.source not in graph_nodes or alias.substitute not in graph_nodes for alias in self.aliases):
            raise ValueError("alias nodes must belong to the accepted graph")
        if any(
            candidate.node not in graph_nodes
            for alias in self.aliases
            for candidate in alias.control_candidates
        ):
            raise ValueError("alias control candidates must belong to the accepted graph")


@dataclass(frozen=True)
class PreactivationIntervention:
    node: FeatureNode
    absolute_value: float
    graph_baseline_value: float | None = None
    graph_delta: float | None = None

    def __post_init__(self) -> None:
        _finite("preactivation intervention", self.absolute_value)
        supplied = (self.graph_baseline_value is not None, self.graph_delta is not None)
        if supplied[0] != supplied[1]:
            raise ValueError("graph baseline and delta must be supplied together")
        if all(supplied):
            assert self.graph_baseline_value is not None and self.graph_delta is not None
            _finite("graph intervention evidence", self.graph_baseline_value, self.graph_delta)
            expected = self.graph_baseline_value + self.graph_delta
            if not math.isclose(self.absolute_value, expected, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("absolute intervention must equal graph baseline plus delta")


@dataclass(frozen=True)
class InterventionVariant:
    variant_id: str
    kind: VariantKind
    semantics: InterventionSemantics
    interventions: tuple[PreactivationIntervention, ...]
    predicted_target_delta: float | None
    predicted_downstream_feature_deltas: tuple["FeatureValue", ...] = ()

    def __post_init__(self) -> None:
        if self.predicted_target_delta is not None:
            _finite("predicted target delta", self.predicted_target_delta)
        nodes = tuple(item.node for item in self.interventions)
        if not self.variant_id or len(set(nodes)) != len(nodes):
            raise ValueError("variant id must be set and intervention nodes unique")
        if (self.kind is VariantKind.NO_OP) != (not self.interventions):
            raise ValueError("only the no-op variant may have no interventions")
        downstream_nodes = tuple(item.node for item in self.predicted_downstream_feature_deltas)
        if len(set(downstream_nodes)) != len(downstream_nodes):
            raise ValueError("predicted downstream feature deltas must be unique")


@dataclass(frozen=True)
class InterventionExecutionRequest:
    identity: TraceIdentity
    target: TargetState
    variants: tuple[InterventionVariant, ...]
    observed_downstream_nodes: tuple[FeatureNode, ...]
    deadline_seconds: float
    cleanup_reserve_seconds: float
    predicted_baseline_seconds: float
    predicted_variant_seconds: float

    def __post_init__(self) -> None:
        _finite(
            "execution request",
            self.deadline_seconds,
            self.cleanup_reserve_seconds,
            self.predicted_baseline_seconds,
            self.predicted_variant_seconds,
        )
        if not self.variants or self.variants[0].kind is not VariantKind.NO_OP:
            raise ValueError("first variant must be the required no-op")
        if sum(item.kind is VariantKind.NO_OP for item in self.variants) != 1:
            raise ValueError("exactly one no-op is required")
        if len(self.variants) > MAX_PROBE_VARIANTS:
            raise ValueError(f"at most {MAX_PROBE_VARIANTS} variants are permitted")
        expected_nodes = tuple(
            sorted(
                {
                    value.node
                    for variant in self.variants
                    for value in variant.predicted_downstream_feature_deltas
                }
            )
        )
        if self.observed_downstream_nodes != expected_nodes:
            raise ValueError("observed_downstream_nodes must equal the planned prediction union")
        if not 0 < self.deadline_seconds <= MAX_PROBE_SECONDS:
            raise ValueError("deadline must be in (0, 120]")
        if min(self.cleanup_reserve_seconds, self.predicted_baseline_seconds, self.predicted_variant_seconds) < 0:
            raise ValueError("runtime estimates must be nonnegative")


@dataclass(frozen=True)
class FeatureValue:
    node: FeatureNode
    preactivation: float

    def __post_init__(self) -> None:
        _finite("feature value", self.preactivation)


@dataclass(frozen=True)
class BaselineCapture:
    target_value: float
    raw_target_logits: tuple[float, ...]
    feature_values: tuple[FeatureValue, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_target_logits:
            raise ValueError("baseline raw_target_logits must not be empty")
        _finite("baseline capture", self.target_value, *self.raw_target_logits)
        nodes = tuple(item.node for item in self.feature_values)
        if len(set(nodes)) != len(nodes):
            raise ValueError("baseline feature values must be unique")


@dataclass(frozen=True)
class VariantObservation:
    variant_id: str
    target_value: float
    raw_target_logits: tuple[float, ...]
    downstream_feature_values: tuple[FeatureValue, ...]
    elapsed_seconds: float
    intervention_feature_values: tuple[FeatureValue, ...] = ()

    def __post_init__(self) -> None:
        if not self.variant_id or not self.raw_target_logits or self.elapsed_seconds < 0:
            raise ValueError("invalid variant observation")
        _finite("variant observation", self.target_value, *self.raw_target_logits, self.elapsed_seconds)
        nodes = tuple(item.node for item in self.downstream_feature_values)
        if len(set(nodes)) != len(nodes):
            raise ValueError("downstream feature values must be unique")
        intervention_nodes = tuple(item.node for item in self.intervention_feature_values)
        if len(set(intervention_nodes)) != len(intervention_nodes):
            raise ValueError("intervention feature values must be unique")


@dataclass(frozen=True)
class RuntimeRefusal:
    code: str
    detail: str
    before_variant_id: str | None = None


@dataclass(frozen=True)
class InterventionExecutionResult:
    status: RuntimeExecutionStatus
    baseline: BaselineCapture | None
    observations: tuple[VariantObservation, ...]
    refusal: RuntimeRefusal | None
    elapsed_seconds: float
    cleanup_completed: bool
    deadline_overrun_seconds: float = 0.0
    ordering_admission_mode: OrderingAdmissionMode | None = None

    def __post_init__(self) -> None:
        _finite("execution result", self.elapsed_seconds, self.deadline_overrun_seconds)
        if self.ordering_admission_mode is not None and not isinstance(
            self.ordering_admission_mode, OrderingAdmissionMode
        ):
            raise ValueError("invalid ordering admission mode")
        ids = tuple(item.variant_id for item in self.observations)
        if len(set(ids)) != len(ids) or min(self.elapsed_seconds, self.deadline_overrun_seconds) < 0:
            raise ValueError("invalid execution result")
        if (self.status is RuntimeExecutionStatus.COMPLETE) == (self.refusal is not None):
            raise ValueError("only partial/refused results require a refusal")


@dataclass(frozen=True)
class VariantEvidence:
    variant_id: str
    kind: VariantKind
    predicted_target_delta: float | None
    realized_target_delta: float | None
    closure_error: float | None
    sign_agreement: bool | None
    downstream_closure: tuple["DownstreamClosure", ...]

    def __post_init__(self) -> None:
        values = (
            self.predicted_target_delta,
            self.realized_target_delta,
            self.closure_error,
        )
        _finite("variant evidence", *(value for value in values if value is not None))


@dataclass(frozen=True)
class DownstreamClosure:
    node: FeatureNode
    baseline_preactivation: float | None
    observed_preactivation: float
    predicted_delta: float
    observed_delta: float | None
    closure_error: float | None

    def __post_init__(self) -> None:
        values = (
            self.baseline_preactivation,
            self.observed_preactivation,
            self.predicted_delta,
            self.observed_delta,
            self.closure_error,
        )
        _finite("downstream closure", *(value for value in values if value is not None))


@dataclass(frozen=True)
class BehavioralAggregateMetrics:
    direct_mean_abs_closure: float | None
    direct_mean_relative_closure: float | None
    direct_max_relative_closure: float | None
    direct_sign_agreement: float | None
    necessity_high_vs_control_separation: float | None
    necessity_predicted_realized_spearman: float | None
    necessity_median_high_control_effect_ratio: float | None
    alias_mean_abs_target_delta: float | None
    alias_mean_abs_closure: float | None
    alias_relative_effect_error: float | None
    alias_substitution_vs_source_ablation: float | None
    alias_control_vs_source_ablation: float | None
    alias_substitution_advantage: float | None
    downstream_mean_abs_closure: float | None
    downstream_mean_relative_closure: float | None
    downstream_p95_relative_closure: float | None

    def __post_init__(self) -> None:
        values = (
            self.direct_mean_abs_closure,
            self.direct_mean_relative_closure,
            self.direct_max_relative_closure,
            self.direct_sign_agreement,
            self.necessity_high_vs_control_separation,
            self.necessity_predicted_realized_spearman,
            self.necessity_median_high_control_effect_ratio,
            self.alias_mean_abs_target_delta,
            self.alias_mean_abs_closure,
            self.alias_relative_effect_error,
            self.alias_substitution_vs_source_ablation,
            self.alias_control_vs_source_ablation,
            self.alias_substitution_advantage,
            self.downstream_mean_abs_closure,
            self.downstream_mean_relative_closure,
            self.downstream_p95_relative_closure,
        )
        _finite("behavioral aggregate metrics", *(value for value in values if value is not None))
        if self.direct_sign_agreement is not None and not 0 <= self.direct_sign_agreement <= 1:
            raise ValueError("direct_sign_agreement must be in [0, 1]")
        if (
            self.necessity_predicted_realized_spearman is not None
            and not -1 <= self.necessity_predicted_realized_spearman <= 1
        ):
            raise ValueError("necessity Spearman must be in [-1, 1]")


@dataclass(frozen=True)
class BehavioralFaithfulnessReport:
    policy_id: str
    calibration_id: str | None
    trace_identity: TraceIdentity
    evidence_completeness: EvidenceCompleteness
    verdict: FaithfulnessVerdict
    runtime_status: RuntimeExecutionStatus
    target: TargetFunctional
    baseline_target_value: float | None
    variants_planned: int
    variants_completed: int
    no_op_required: bool
    no_op_passed: bool | None
    alias_comparator_status: AliasComparatorStatus
    alias_selections: tuple[AliasSubstitution, ...]
    variant_recipes: tuple[InterventionVariant, ...]
    evidence: tuple[VariantEvidence, ...]
    metrics: BehavioralAggregateMetrics
    raw_execution: InterventionExecutionResult
    reasons: tuple[str, ...]
    refusal: RuntimeRefusal | None
    ordering_admission_mode: OrderingAdmissionMode | None = None
    sufficiency: SufficiencyStatus = SufficiencyStatus.UNKNOWN
    deadline_contract: str = "cooperative; an admitted in-flight operation may overrun"

    def __post_init__(self) -> None:
        if self.baseline_target_value is not None:
            _finite("report baseline target", self.baseline_target_value)
        if self.ordering_admission_mode is not self.raw_execution.ordering_admission_mode:
            raise ValueError("report ordering admission must match raw execution")

    @property
    def evidence_fingerprint(self) -> str:
        from .serialization import evidence_fingerprint

        return evidence_fingerprint(self)

    def to_json(self) -> str:
        from .serialization import report_to_json

        return report_to_json(self)
