"""Circuit Tracer public API.

Tracing has one typed entrypoint under :mod:`circuit_tracer.tracing`. Governor
contracts remain available from :mod:`circuit_tracer.governor`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.attribution.phase0_stats import attribute_phase0_stats
    from circuit_tracer.attribution.sparsification import SparsificationConfig
    from circuit_tracer.graph import Graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.tracing import (
        AttributionProblem,
        DecoderCachePolicy,
        EffectiveExecutionDescriptor,
        EffectiveExecutionIdentity,
        ExecutionConstraints,
        FrontierExpansionPlan,
        FrontierSemantics,
        ObservabilityPolicy,
        PrefixViewTarget,
        ReplayPlan,
        ResolvedTracePlan,
        RowStoragePlan,
        SessionPlan,
        SessionWindow,
        TraceEvidence,
        TraceRequest,
        TraceResult,
        TraceSemantics,
        TraceSession,
        TraceStatus,
        open_session,
        resolve_trace_request,
        trace_batch,
        trace_one,
    )

__all__ = [
    "AttributionProblem",
    "DecoderCachePolicy",
    "EffectiveExecutionDescriptor",
    "EffectiveExecutionIdentity",
    "attribute_phase0_stats",
    "ExecutionConstraints",
    "FrontierExpansionPlan",
    "FrontierSemantics",
    "Graph",
    "ObservabilityPolicy",
    "PrefixViewTarget",
    "ReplayPlan",
    "ReplacementModel",
    "ResolvedTracePlan",
    "RowStoragePlan",
    "SessionPlan",
    "SessionWindow",
    "SparsificationConfig",
    "TraceEvidence",
    "TraceRequest",
    "TraceResult",
    "TraceSemantics",
    "TraceSession",
    "TraceStatus",
    "open_session",
    "resolve_trace_request",
    "trace_batch",
    "trace_one",
]


def __getattr__(name: str):
    lazy_imports = {
        "attribute_phase0_stats": (
            "circuit_tracer.attribution.phase0_stats",
            "attribute_phase0_stats",
        ),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
        "SparsificationConfig": (
            "circuit_tracer.attribution.sparsification",
            "SparsificationConfig",
        ),
    }
    if name in set(__all__) - set(lazy_imports):
        lazy_imports[name] = ("circuit_tracer.tracing", name)
    if name not in lazy_imports:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attribute_name = lazy_imports[name]
    module = __import__(module_name, fromlist=[attribute_name])
    return getattr(module, attribute_name)
