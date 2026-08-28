from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import BehavioralFaithfulnessReport, FeatureNode, FeatureValue


SCHEMA = "behavioral_faithfulness_report"
SCHEMA_VERSION = 2


def _node(node: FeatureNode) -> dict[str, int]:
    return {"layer": node.layer, "position": node.position, "feature": node.feature}


def _feature(value: FeatureValue) -> dict[str, object]:
    return {"node": _node(value.node), "preactivation": value.preactivation}


def _payload(report: BehavioralFaithfulnessReport) -> dict[str, Any]:
    identity = report.trace_identity
    baseline = report.raw_execution.baseline
    return {
        "policy_id": report.policy_id,
        "calibration_id": report.calibration_id,
        "trace_identity": {
            "trace_id": identity.trace_id,
            "graph_fingerprint": identity.graph_fingerprint,
            "provider_fingerprint": identity.provider_fingerprint,
            "semantic_fingerprint": identity.semantic_fingerprint,
            "execution_fingerprint": identity.execution_fingerprint,
            "prompt_token_ids": list(identity.prompt_token_ids),
            "target_position": identity.target_position,
            "target_token_id": identity.target_token_id,
        },
        "evidence_completeness": report.evidence_completeness.value,
        "verdict": report.verdict.value,
        "runtime_status": report.runtime_status.value,
        "ordering_admission_mode": (
            None
            if report.ordering_admission_mode is None
            else report.ordering_admission_mode.value
        ),
        "target": report.target.value,
        "baseline_target_value": report.baseline_target_value,
        "variants_planned": report.variants_planned,
        "variants_completed": report.variants_completed,
        "no_op_required": report.no_op_required,
        "no_op_passed": report.no_op_passed,
        "alias_comparator_status": report.alias_comparator_status.value,
        "alias_selections": [
            {
                "source": _node(alias.source),
                "substitute": _node(alias.substitute),
                "substitute_absolute_preactivation": (
                    alias.substitute_absolute_preactivation
                ),
                "predicted_target_delta": alias.predicted_target_delta,
                "selection_evidence": {
                    "selection_policy_id": alias.selection_evidence.selection_policy_id,
                    "calibration_fingerprint": (
                        alias.selection_evidence.calibration_fingerprint
                    ),
                    "comparison_evidence_fingerprint": (
                        alias.selection_evidence.comparison_evidence_fingerprint
                    ),
                    "baseline_graph_fingerprint": (
                        alias.selection_evidence.baseline_graph_fingerprint
                    ),
                    "candidate_graph_fingerprint": (
                        alias.selection_evidence.candidate_graph_fingerprint
                    ),
                    "decoder_fingerprint": alias.selection_evidence.decoder_fingerprint,
                    "decoder_output_topology": (
                        alias.selection_evidence.decoder_output_topology
                    ),
                    "qualified_decoder_cosine": (
                        alias.selection_evidence.qualified_decoder_cosine
                    ),
                    "observed_decoder_cosine": (
                        alias.selection_evidence.observed_decoder_cosine
                    ),
                    "source_decoder_norm": alias.selection_evidence.source_decoder_norm,
                    "substitute_decoder_norm": (
                        alias.selection_evidence.substitute_decoder_norm
                    ),
                    "least_squares_coefficient": (
                        alias.selection_evidence.least_squares_coefficient
                    ),
                },
                "control_candidates": [
                    {
                        "node": _node(control.node),
                        "similarity_to_source": control.similarity_to_source,
                    }
                    for control in alias.control_candidates
                ],
            }
            for alias in report.alias_selections
        ],
        "sufficiency": report.sufficiency.value,
        "deadline_contract": report.deadline_contract,
        "variant_recipes": [
            {
                "variant_id": variant.variant_id,
                "kind": variant.kind.value,
                "semantics": variant.semantics.value,
                "interventions": [
                    {
                        "node": _node(intervention.node),
                        "graph_baseline_value": intervention.graph_baseline_value,
                        "absolute_value": intervention.absolute_value,
                        "graph_delta": intervention.graph_delta,
                    }
                    for intervention in variant.interventions
                ],
                "predicted_target_delta": variant.predicted_target_delta,
                "predicted_downstream_feature_deltas": [
                    _feature(value) for value in variant.predicted_downstream_feature_deltas
                ],
            }
            for variant in report.variant_recipes
        ],
        "metrics": {
            "direct_mean_abs_closure": report.metrics.direct_mean_abs_closure,
            "direct_mean_relative_closure": report.metrics.direct_mean_relative_closure,
            "direct_max_relative_closure": report.metrics.direct_max_relative_closure,
            "direct_sign_agreement": report.metrics.direct_sign_agreement,
            "necessity_high_vs_control_separation": (
                report.metrics.necessity_high_vs_control_separation
            ),
            "necessity_predicted_realized_spearman": (
                report.metrics.necessity_predicted_realized_spearman
            ),
            "necessity_median_high_control_effect_ratio": (
                report.metrics.necessity_median_high_control_effect_ratio
            ),
            "alias_mean_abs_target_delta": report.metrics.alias_mean_abs_target_delta,
            "alias_mean_abs_closure": report.metrics.alias_mean_abs_closure,
            "alias_relative_effect_error": report.metrics.alias_relative_effect_error,
            "alias_substitution_vs_source_ablation": (
                report.metrics.alias_substitution_vs_source_ablation
            ),
            "alias_control_vs_source_ablation": (
                report.metrics.alias_control_vs_source_ablation
            ),
            "alias_substitution_advantage": report.metrics.alias_substitution_advantage,
            "downstream_mean_abs_closure": report.metrics.downstream_mean_abs_closure,
            "downstream_mean_relative_closure": (
                report.metrics.downstream_mean_relative_closure
            ),
            "downstream_p95_relative_closure": (
                report.metrics.downstream_p95_relative_closure
            ),
        },
        "evidence": [
            {
                "variant_id": item.variant_id,
                "kind": item.kind.value,
                "predicted_target_delta": item.predicted_target_delta,
                "realized_target_delta": item.realized_target_delta,
                "closure_error": item.closure_error,
                "sign_agreement": item.sign_agreement,
                "downstream_closure": [
                    {
                        "node": _node(closure.node),
                        "baseline_preactivation": closure.baseline_preactivation,
                        "observed_preactivation": closure.observed_preactivation,
                        "predicted_delta": closure.predicted_delta,
                        "observed_delta": closure.observed_delta,
                        "closure_error": closure.closure_error,
                    }
                    for closure in item.downstream_closure
                ],
            }
            for item in report.evidence
        ],
        "raw_execution": {
            "status": report.raw_execution.status.value,
            "ordering_admission_mode": (
                None
                if report.raw_execution.ordering_admission_mode is None
                else report.raw_execution.ordering_admission_mode.value
            ),
            "baseline": (
                None
                if baseline is None
                else {
                    "target_value": baseline.target_value,
                    "raw_target_logits": list(baseline.raw_target_logits),
                    "feature_values": [_feature(item) for item in baseline.feature_values],
                }
            ),
            "observations": [
                {
                    "variant_id": item.variant_id,
                    "target_value": item.target_value,
                    "raw_target_logits": list(item.raw_target_logits),
                    "downstream_feature_values": [
                        _feature(value) for value in item.downstream_feature_values
                    ],
                    "intervention_feature_values": [
                        _feature(value) for value in item.intervention_feature_values
                    ],
                    "elapsed_seconds": item.elapsed_seconds,
                }
                for item in report.raw_execution.observations
            ],
            "refusal": (
                None
                if report.raw_execution.refusal is None
                else {
                    "code": report.raw_execution.refusal.code,
                    "detail": report.raw_execution.refusal.detail,
                    "before_variant_id": report.raw_execution.refusal.before_variant_id,
                }
            ),
            "elapsed_seconds": report.raw_execution.elapsed_seconds,
            "cleanup_completed": report.raw_execution.cleanup_completed,
            "deadline_overrun_seconds": report.raw_execution.deadline_overrun_seconds,
        },
        "reasons": list(report.reasons),
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evidence_fingerprint(report: BehavioralFaithfulnessReport) -> str:
    return hashlib.sha256(_canonical(_payload(report)).encode("utf-8")).hexdigest()


def report_to_json(report: BehavioralFaithfulnessReport) -> str:
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_fingerprint": evidence_fingerprint(report),
        "report": _payload(report),
    }
    return _canonical(document)
