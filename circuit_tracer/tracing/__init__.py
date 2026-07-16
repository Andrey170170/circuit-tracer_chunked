"""Canonical typed tracing API."""

from circuit_tracer.governor.contracts import AdmissionMode, FidelityMode

from .api import open_session, trace_batch, trace_one
from circuit_tracer.execution_identity import (
    EffectiveExecutionDescriptor,
    EffectiveExecutionIdentity,
)
from .plan import (
    DecoderCachePolicy,
    DecoderPlan,
    ExecutionConstraints,
    FrontierExpansionPlan,
    ObservabilityPolicy,
    ReplayPlan,
    ResolvedTracePlan,
    RowStoragePlan,
    SessionPlan,
    TraceEvidence,
)
from .planning import resolve_trace_request
from .problem import AttributionProblem, FrontierSemantics, PrefixViewTarget, TraceSemantics
from .request import GovernorFidelityPolicy, TraceRequest
from .result import TraceResult, TraceStatus
from .session import SessionWindow, TraceSession

__all__ = [
    "AttributionProblem",
    "AdmissionMode",
    "DecoderCachePolicy",
    "DecoderPlan",
    "ExecutionConstraints",
    "FidelityMode",
    "EffectiveExecutionDescriptor",
    "EffectiveExecutionIdentity",
    "FrontierExpansionPlan",
    "FrontierSemantics",
    "GovernorFidelityPolicy",
    "ObservabilityPolicy",
    "PrefixViewTarget",
    "ReplayPlan",
    "ResolvedTracePlan",
    "RowStoragePlan",
    "SessionPlan",
    "SessionWindow",
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
