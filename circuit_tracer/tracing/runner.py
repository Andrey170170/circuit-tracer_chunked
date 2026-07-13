"""Small coordinator for the canonical backend contracts."""

from __future__ import annotations

import logging

from circuit_tracer.observability.events import TraceEvent
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.observability.run_scope import TraceRunScope

from circuit_tracer.execution_identity import ExecutionIdentityState
from .plan import ResolvedTracePlan
from .problem import AttributionProblem
from .result import TraceResult, TraceStatus


def run_trace(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    *,
    forward_overrides: object | None = None,
) -> TraceResult:
    """Run one already-resolved trace through its backend-owned phase pipeline."""
    observer, logger = _open_observability(plan)
    execution_identity = ExecutionIdentityState(plan.requested_execution_fingerprint)
    scope = TraceRunScope(
        observer=observer,
        logger=logger,
        compact_output=plan.execution.compact_output,
        profile=plan.execution.observability.profile,
        execution_identity=execution_identity,
    )
    observer.observe(
        TraceEvent(
            scope="run",
            name="attribute.start",
            attrs={
                "backend": plan.backend,
                "profile": plan.execution.observability.profile,
                "compact_output": plan.execution.compact_output,
                "semantic_fingerprint": plan.semantic_fingerprint,
                "requested_execution_fingerprint": plan.requested_execution_fingerprint,
                "effective_execution_fingerprint": None,
                "execution_fingerprint": plan.requested_execution_fingerprint,
            },
        )
    )
    try:
        if plan.backend == "nnsight":
            from circuit_tracer.attribution.nnsight.backend import run_nnsight_trace

            output = run_nnsight_trace(
                problem,
                plan,
                observer=observer,
                forward_overrides=forward_overrides,
                execution_identity=execution_identity,
            )
        elif plan.backend == "transformerlens":
            if forward_overrides is not None:
                raise ValueError("forward overrides require the NNSight backend")
            from circuit_tracer.attribution.transformerlens.backend import (
                run_transformerlens_trace,
            )

            execution_identity.mark_requested_as_effective()
            output = run_transformerlens_trace(problem, plan, observer=observer)
        else:  # resolved plans are closed over the supported backend set
            raise AssertionError(f"unreachable backend: {plan.backend}")
    except BaseException as primary_error:
        scope.close(primary_error)
        raise
    evidence = scope.close(None)
    return TraceResult(
        output=output,
        semantic_fingerprint=plan.semantic_fingerprint,
        requested_execution_fingerprint=plan.requested_execution_fingerprint,
        effective_execution_fingerprint=execution_identity.effective_fingerprint,
        effective_execution=(
            None
            if execution_identity.effective is None
            else execution_identity.effective.descriptor
        ),
        status=TraceStatus.SUCCEEDED,
        telemetry_summary=evidence.summary,
        telemetry_events=evidence.events,
        admission_report=plan.admission_report,
    )


def _open_observability(plan: ResolvedTracePlan) -> tuple[TelemetryObserver, logging.Logger]:
    policy = plan.execution.observability
    logger = logging.getLogger("attribution")
    context = dict(policy.telemetry_context)
    context["runtime_plan"] = {
        "schema_version": 3,
        "semantic_fingerprint": plan.semantic_fingerprint,
        "requested_execution_fingerprint": plan.requested_execution_fingerprint,
        "execution_fingerprint": plan.requested_execution_fingerprint,
    }
    max_events = (
        int(policy.telemetry_max_events)
        if policy.telemetry_max_events is not None and policy.telemetry_max_events > 0
        else 20_000
    )
    return (
        TelemetryObserver.create(
            enabled=bool(
                policy.profile
                or plan.execution.compact_output
                or policy.phase4_anomaly_debug
            ),
            max_events=max_events,
            jsonl_path=policy.telemetry_jsonl_path,
            static_context=context,
            logger=logger,
        ),
        logger,
    )
