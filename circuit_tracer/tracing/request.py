"""Canonical public tracing request."""

from dataclasses import dataclass, field

from circuit_tracer.governor.contracts import FidelityMode, PhysicalExecutionRequirements

from .plan import ExecutionConstraints, TraceEvidence
from .problem import AttributionProblem, TraceSemantics


@dataclass(frozen=True)
class GovernorFidelityPolicy:
    """Authorize named semantic deviations in governed planning."""

    mode: FidelityMode = FidelityMode.STRICT
    override_fields: tuple[str, ...] = ()
    evidence_name: str | None = None
    evidence_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FidelityMode):
            raise ValueError("governor fidelity mode must be a FidelityMode")
        if any(not isinstance(name, str) or not name for name in self.override_fields):
            raise ValueError("governor fidelity override fields must be nonempty strings")
        if (
            tuple(sorted(self.override_fields)) != self.override_fields
            or len(set(self.override_fields)) != len(self.override_fields)
        ):
            raise ValueError("governor fidelity override fields must be sorted and unique")
        if (self.evidence_name is None) != (self.evidence_version is None):
            raise ValueError("governor fidelity evidence name and version must be provided together")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (self.evidence_name, self.evidence_version)
        ):
            raise ValueError(
                "governor fidelity evidence name and version must be nonempty strings"
            )

        has_evidence = self.evidence_name is not None
        if self.mode is FidelityMode.STRICT:
            if self.override_fields or has_evidence:
                raise ValueError("strict governor fidelity accepts no overrides or evidence")
        elif self.mode is FidelityMode.RESEARCH:
            if not self.override_fields:
                raise ValueError("research governor fidelity requires override fields")
            if has_evidence:
                raise ValueError("research governor fidelity accepts no evidence")
        elif self.mode is FidelityMode.VALIDATED_RELAXED:
            if not self.override_fields or not has_evidence:
                raise ValueError(
                    "validated_relaxed governor fidelity requires override fields and evidence"
                )


@dataclass(frozen=True)
class TraceRequest:
    """One validated attribution problem and its explicitly owned policies."""

    problem: AttributionProblem
    semantics: TraceSemantics = field(default_factory=TraceSemantics)
    execution: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    evidence: TraceEvidence = field(default_factory=TraceEvidence)
    physical_requirements: PhysicalExecutionRequirements | None = None
    governor_fidelity: GovernorFidelityPolicy = field(default_factory=GovernorFidelityPolicy)
