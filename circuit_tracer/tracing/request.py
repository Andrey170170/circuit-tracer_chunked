"""Canonical public tracing request."""

from dataclasses import dataclass, field

from .plan import ExecutionConstraints, TraceEvidence
from .problem import AttributionProblem, TraceSemantics


@dataclass(frozen=True)
class TraceRequest:
    """One validated attribution problem and its explicitly owned policies."""

    problem: AttributionProblem
    semantics: TraceSemantics = field(default_factory=TraceSemantics)
    execution: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    evidence: TraceEvidence = field(default_factory=TraceEvidence)

