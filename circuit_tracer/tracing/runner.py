"""Small coordinator for the canonical backend contracts."""

from __future__ import annotations

import logging
from dataclasses import replace

from circuit_tracer.governor.contracts import AdmissionMode
from circuit_tracer.governor.runtime import (
    RuntimePlanningRefusedError,
    TorchLoadedStateSampler,
    TraceGovernorRuntime,
)
from circuit_tracer.observability.events import TraceEvent
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.observability.run_scope import TraceRunScope

from circuit_tracer.execution_identity import ExecutionIdentityState
from .plan import ResolvedTracePlan
from .problem import AttributionProblem
from .result import TraceResult, TraceStatus
from circuit_tracer.diagnostic import ProbeCompletion


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
        governor_admission_mode=plan.governor_admission_mode.value,
    )
    governor_runtime = None
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
                "governor_admission_mode": plan.governor_admission_mode.value,
            },
        )
    )
    try:
        if plan.planning_trace_plan is not None:
            if any(
                value is None
                for value in (
                    plan.planning_workload,
                    plan.planning_profile,
                    plan.planning_envelope,
                    plan.planning_requirements,
                )
            ):
                raise RuntimeError("governed plan is missing immutable planning inputs")
            governor_runtime = TraceGovernorRuntime(
                plan=plan.planning_trace_plan,
                workload=plan.planning_workload,
                profile=plan.planning_profile,
                envelope=plan.planning_envelope,
                requirements=plan.planning_requirements,
                observer=observer,
                admission_mode=plan.governor_admission_mode,
                calibration_catalog=plan.planning_calibration_catalog,
                response_bundle=plan.planning_response_bundle,
            )
            governor_runtime.pre_execution_admission()
            if (
                not governor_runtime.current_plan.admission.admitted
                and plan.governor_admission_mode is AdmissionMode.ENFORCE
            ):
                return _refused_result(plan, scope, governor_runtime, "pre_execution_admission")
            provider = getattr(problem.model, "transcoders", None)
            observation = TorchLoadedStateSampler().sample(getattr(provider, "_module", provider))
            revision = governor_runtime.loaded_state_calibration(observation)
            from .governor_bridge import recompile_governed_plan

            plan = recompile_governed_plan(problem, plan, revision.plan)
            plan = replace(
                plan,
                planning_profile=governor_runtime.profile,
                planning_envelope=governor_runtime.envelope,
                planning_workload=governor_runtime.workload,
            )
            if (
                not revision.plan.admission.admitted
                and plan.governor_admission_mode is AdmissionMode.ENFORCE
            ):
                return _refused_result(plan, scope, governor_runtime, "loaded_state_calibration")
        if plan.backend == "nnsight":
            from circuit_tracer.attribution.nnsight.backend import run_nnsight_trace

            output = run_nnsight_trace(
                problem,
                plan,
                observer=observer,
                forward_overrides=forward_overrides,
                execution_identity=execution_identity,
                governor_runtime=governor_runtime,
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
    except RuntimePlanningRefusedError as refusal:
        if governor_runtime is None:
            raise AssertionError("runtime planning refusal requires a governor runtime")
        from .governor_bridge import recompile_governed_plan

        plan = recompile_governed_plan(problem, plan, refusal.revision.plan)
        plan = replace(
            plan,
            planning_profile=governor_runtime.profile,
            planning_envelope=governor_runtime.envelope,
            planning_workload=governor_runtime.workload,
            planning_requirements=governor_runtime.requirements,
            planning_parent_fingerprint=refusal.revision.parent_execution_fingerprint,
            planning_epoch_fingerprint=refusal.revision.execution_fingerprint,
        )
        return _refused_result(plan, scope, governor_runtime, "active_universe_replan")
    except BaseException as primary_error:
        if governor_runtime is not None:
            governor_runtime.observer.observe(
                TraceEvent(
                    scope="run",
                    name="planning.failure",
                    attrs={
                        "error_type": type(primary_error).__name__,
                        "error_message": str(primary_error),
                    },
                )
            )
            governor_runtime.close()
        scope.close(primary_error)
        raise
    if governor_runtime is not None:
        governor_runtime.close()
    if isinstance(output, ProbeCompletion):
        evidence = scope.close(None, terminal_status="probe_completed")
        summary = dict(evidence.summary)
        summary.update(
            diagnostic_stop_mode=output.mode,
            phase4_batches_completed=output.phase4_batches_completed,
            diagnostic_metadata=dict(output.diagnostic_metadata),
        )
        return TraceResult(
            output=None,
            semantic_fingerprint=plan.semantic_fingerprint,
            requested_execution_fingerprint=plan.requested_execution_fingerprint,
            effective_execution_fingerprint=execution_identity.effective_fingerprint,
            effective_execution=(
                None
                if execution_identity.effective is None
                else execution_identity.effective.descriptor
            ),
            status=TraceStatus.PROBE_COMPLETED,
            telemetry_summary=summary,
            telemetry_events=evidence.events,
            admission_report=(
                plan.admission_report
                if governor_runtime is None
                else governor_runtime.current_plan.admission
            ),
        )
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
        admission_report=(
            plan.admission_report
            if governor_runtime is None
            else governor_runtime.current_plan.admission
        ),
    )


def _refused_result(
    plan: ResolvedTracePlan,
    scope: TraceRunScope,
    runtime: TraceGovernorRuntime,
    epoch: str,
) -> TraceResult:
    report = runtime.current_plan.admission
    runtime.observer.observe(
        TraceEvent(
            scope="run",
            name="planning.refusal",
            attrs={
                "epoch": epoch,
                "refusals": report.refusals,
            },
        )
    )
    runtime.close()
    evidence = scope.close(None, terminal_status="refused")
    return TraceResult(
        output=None,
        semantic_fingerprint=plan.semantic_fingerprint,
        requested_execution_fingerprint=plan.requested_execution_fingerprint,
        status=TraceStatus.REFUSED,
        telemetry_summary=evidence.summary,
        telemetry_events=evidence.events,
        admission_report=report,
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
        "governor_admission_mode": plan.governor_admission_mode.value,
    }
    max_events = (
        int(policy.telemetry_max_events)
        if policy.telemetry_max_events is not None and policy.telemetry_max_events > 0
        else 20_000
    )
    default_enabled = bool(
        policy.profile
        or plan.execution.compact_output
        or policy.phase4_anomaly_debug
        or plan.planning_trace_plan is not None
    )
    enabled = (
        default_enabled
        if policy.telemetry_enabled is None
        else bool(policy.telemetry_enabled)
    )
    return (
        TelemetryObserver.create(
            enabled=enabled,
            max_events=max_events,
            jsonl_path=policy.telemetry_jsonl_path,
            static_context=context,
            logger=logger,
        ),
        logger,
    )
