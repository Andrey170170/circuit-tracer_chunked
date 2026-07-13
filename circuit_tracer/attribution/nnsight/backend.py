"""Thin coordinator for the NNSight attribution backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from circuit_tracer.attribution.nnsight.execution import (
    AttributionExecution,
    BackendOperations,
)
from circuit_tracer.attribution.nnsight.phases.phase0 import run_phase0
from circuit_tracer.attribution.nnsight.phases.phase1 import (
    _run_phase1_forward_pass,
)
from circuit_tracer.attribution.nnsight.phases.phase2 import run_phase2
from circuit_tracer.attribution.nnsight.phases.phase3 import run_phase3
from circuit_tracer.attribution.nnsight.phases.phase4 import run_phase4
from circuit_tracer.attribution.nnsight.phases.phase5 import run_phase5
from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata,
    _resolve_prefix_view_output_position,
    validate_prefix_view_metadata,
)
from circuit_tracer.attribution.nnsight.preparation import (
    PreparationDependencies,
    prepare_backend,
)
from circuit_tracer.attribution.nnsight.run_scope import AttributionRunScope
from circuit_tracer.graph import Graph
from circuit_tracer.observability.events import TraceObserver
from circuit_tracer.tracing.plan import ResolvedTracePlan
from circuit_tracer.execution_identity import ExecutionIdentityState
from circuit_tracer.tracing.problem import AttributionProblem
from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    require_exact_chunked_provider,
)

if TYPE_CHECKING:
    from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides


def run_nnsight_trace(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    *,
    observer: TraceObserver | None = None,
    forward_overrides: ForwardOverrides | None = None,
    execution_identity: ExecutionIdentityState,
) -> Graph | dict[str, object]:
    """Execute one resolved plan while owning logging and module offload cleanup."""
    from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides

    observability = plan.execution.observability
    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if (observability.verbose or observability.profile) and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    prefix_view_metadata = validate_prefix_view_metadata(
        prompt=problem.prompt,
        attribution_targets=problem.targets,
        prefix_view=problem.prefix_view,
        prefix_view_metadata=plan.evidence_metadata.get("prefix_view_metadata"),
    )
    output_position = _resolve_prefix_view_output_position(
        prefix_view_metadata,
        problem.output_position,
    )
    if observer is None:
        raise RuntimeError("run_nnsight_trace must be invoked through the canonical trace runner")
    offload_handles: list[Any] = []
    try:
        return _execute_prepared_trace(
            problem=problem,
            plan=plan,
            logger=logger,
            offload_handles=offload_handles,
            forward_overrides=forward_overrides or ForwardOverrides(),
            prefix_view_metadata=prefix_view_metadata,
            output_position=output_position,
            observer=observer,
            execution_identity=execution_identity,
        )
    finally:
        if handler is not None:
            logger.removeHandler(handler)


def _execute_prepared_trace(
    *,
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    logger: logging.Logger,
    offload_handles: list[Any],
    forward_overrides: ForwardOverrides,
    prefix_view_metadata: PrefixViewMetadata | None,
    output_position: int | None,
    observer: TraceObserver,
    execution_identity: ExecutionIdentityState,
) -> Graph | dict[str, object]:
    """Prepare mechanisms, execute Phase 0-5 operations, then close the lifecycle."""
    scope = AttributionRunScope(offload_handles=offload_handles)
    try:
        prepared = prepare_backend(
            problem=problem,
            plan=plan,
            logger=logger,
            offload_handles=offload_handles,
            forward_overrides=forward_overrides,
            prefix_view_metadata=prefix_view_metadata,
            output_position=output_position,
            observer=observer,
            dependencies=PreparationDependencies(
                get_capabilities=get_transcoder_capabilities,
                require_exact_provider=require_exact_chunked_provider,
            ),
        )
        execution_identity.mark_effective(prepared.effective_execution)
        execution = AttributionExecution(
            prepared=prepared,
            scope=scope,
            operations=BackendOperations(
                run_phase0=run_phase0,
                run_phase1=_run_phase1_forward_pass,
                run_phase2=run_phase2,
                run_phase3=run_phase3,
                run_phase4=run_phase4,
                run_phase5=run_phase5,
            ),
        )
        return execution.run()
    finally:
        import sys

        _, primary_error, _ = sys.exc_info()
        scope.close(primary_error)
