"""Canonical typed tracing API."""

from circuit_tracer.governor.contracts import AdmissionMode, FidelityMode
from circuit_tracer.governor.calibration import (
    CalibrationCatalog,
    CalibrationObservation,
    DEFAULT_KNOB_SENSITIVITIES,
    FidelityBudget,
    FidelityPrediction,
    KnobSensitivity,
    MetricPrediction,
    ParetoAlternative,
    PredictionSupport,
    PredictionSupportKind,
    PredictionUncertainty,
    SensitivityClass,
)

from .api import open_session, trace_batch, trace_one
from circuit_tracer.execution_identity import (
    EffectiveExecutionDescriptor,
    EffectiveExecutionIdentity,
)
from .plan import (
    DecoderCachePolicy,
    DecoderPlan,
    DiagnosticStopPolicy,
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
    "CalibrationCatalog",
    "CalibrationObservation",
    "DEFAULT_KNOB_SENSITIVITIES",
    "AdmissionMode",
    "DecoderCachePolicy",
    "DecoderPlan",
    "DiagnosticStopPolicy",
    "ExecutionConstraints",
    "FidelityMode",
    "FidelityBudget",
    "FidelityPrediction",
    "EffectiveExecutionDescriptor",
    "EffectiveExecutionIdentity",
    "FrontierExpansionPlan",
    "FrontierSemantics",
    "GovernorFidelityPolicy",
    "KnobSensitivity",
    "MetricPrediction",
    "ObservabilityPolicy",
    "PrefixViewTarget",
    "ParetoAlternative",
    "PredictionSupport",
    "PredictionSupportKind",
    "PredictionUncertainty",
    "ReplayPlan",
    "ResolvedTracePlan",
    "RowStoragePlan",
    "SessionPlan",
    "SessionWindow",
    "SensitivityClass",
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
