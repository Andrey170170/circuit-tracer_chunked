"""Small coordinator for the canonical backend contracts."""

from __future__ import annotations

from typing import Any

from .plan import ResolvedTracePlan
from .problem import AttributionProblem
from .result import TraceResult, TraceStatus


def run_trace(problem: AttributionProblem, plan: ResolvedTracePlan) -> TraceResult:
    """Run one already-resolved trace through its backend-owned phase pipeline."""

    if plan.backend == "nnsight":
        from circuit_tracer.attribution.nnsight.backend import run_nnsight_trace

        output = run_nnsight_trace(problem, plan)
    elif plan.backend == "transformerlens":
        output = _run_transformerlens_trace(problem, plan)
    else:  # resolved plans are closed over the supported backend set
        raise AssertionError(f"unreachable backend: {plan.backend}")
    summary = (
        output.get("telemetry_summary", {})
        if isinstance(output, dict)
        else getattr(output, "telemetry_summary", {})
    )
    return TraceResult(
        output=output,
        semantic_fingerprint=plan.semantic_fingerprint,
        execution_fingerprint=plan.execution_fingerprint,
        status=TraceStatus.SUCCEEDED,
        telemetry_summary=summary,
        admission_report=plan.admission_report,
    )


def _run_transformerlens_trace(problem: AttributionProblem, plan: ResolvedTracePlan) -> Any:
    from circuit_tracer.attribution.attribute_transformerlens import attribute

    semantics = plan.semantics
    return attribute(
        prompt=problem.prompt,
        model=problem.model,
        attribution_targets=problem.targets,
        max_n_logits=problem.max_n_logits,
        desired_logit_prob=problem.desired_logit_prob,
        batch_size=semantics.source_batch_size,
        max_feature_nodes=semantics.max_feature_nodes,
        offload=plan.execution.offload,
        verbose=plan.execution.observability.verbose,
        update_interval=semantics.update_interval,
        profile=plan.execution.observability.profile,
        profile_log_interval=plan.execution.observability.profile_log_interval,
        diagnostic_feature_cap=semantics.diagnostic_feature_cap,
        sparsification=semantics.sparsification,
    )

