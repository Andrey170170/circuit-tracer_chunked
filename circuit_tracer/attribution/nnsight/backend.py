"""Thin coordinator for the NNSight attribution backend."""

from __future__ import annotations

import logging
import sys
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
from circuit_tracer.observability.human_logs import _log_memory_boundary
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.tracing.plan import ResolvedTracePlan
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
    forward_overrides: ForwardOverrides | None = None,
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
        prefix_view_metadata=plan.evidence_metadata.get("prefix_view_metadata"),
    )
    output_position = _resolve_prefix_view_output_position(
        prefix_view_metadata,
        problem.output_position,
    )
    offload_handles: list[Any] = []
    try:
        return _run_attribution(
            problem=problem,
            plan=plan,
            logger=logger,
            offload_handles=offload_handles,
            forward_overrides=forward_overrides or ForwardOverrides(),
            prefix_view_metadata=prefix_view_metadata,
            output_position=output_position,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()
        if handler is not None:
            logger.removeHandler(handler)


def _run_attribution(
    *,
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    logger: logging.Logger,
    offload_handles: list[Any],
    forward_overrides: ForwardOverrides,
    prefix_view_metadata: PrefixViewMetadata | None,
    output_position: int | None,
) -> Graph | dict[str, object]:
    """Prepare mechanisms, execute Phase 0-5 operations, then close the lifecycle."""
    prepared = prepare_backend(
        problem=problem,
        plan=plan,
        logger=logger,
        offload_handles=offload_handles,
        forward_overrides=forward_overrides,
        prefix_view_metadata=prefix_view_metadata,
        output_position=output_position,
        dependencies=PreparationDependencies(
            get_capabilities=get_transcoder_capabilities,
            require_exact_provider=require_exact_chunked_provider,
            telemetry_observer_type=TelemetryObserver,
        ),
    )
    scope = AttributionRunScope(
        logger=logger,
        model=problem.model,
        telemetry_observer=prepared.diagnostics.observer,
        compact_output=plan.execution.compact_output,
        profile=plan.execution.observability.profile,
        prefix_view_metadata=prefix_view_metadata,
        log_memory_boundary=_log_memory_boundary,
        anomaly_debug_result=prepared.diagnostics.anomaly_debug_result,
        cross_cluster_debug_summary=prepared.diagnostics.cross_cluster_summary,
        cross_cluster_debug_checkpoints=prepared.diagnostics.cross_cluster_checkpoints,
        cross_cluster_debug_batches=prepared.diagnostics.cross_cluster_batches,
    )
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
    try:
        return execution.run()
    finally:
        _, primary_error, _ = sys.exc_info()
        scope.close(primary_error)
