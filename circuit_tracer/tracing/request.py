"""Canonical public tracing request."""

from dataclasses import dataclass, field

from circuit_tracer.governor.contracts import (
    AdmissionMode,
    FidelityMode,
    PhysicalExecutionRequirements,
)
from circuit_tracer.governor.calibration import CalibrationCatalog, FidelityBudget

from .plan import ExecutionConstraints, TraceEvidence
from .problem import AttributionProblem, TraceSemantics


@dataclass(frozen=True)
class GovernorFidelityPolicy:
    """Authorize named semantic deviations in governed planning."""

    mode: FidelityMode = FidelityMode.EXACT
    budget: FidelityBudget | None = None
    override_fields: tuple[str, ...] = ()

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
        if self.mode is FidelityMode.EXACT and (self.override_fields or self.budget):
            raise ValueError("exact governor fidelity accepts no overrides or budget")
        if self.mode is FidelityMode.BOUNDED and (
            self.budget is None
            or not self.budget.metric_floors
            or not self.budget.allowed_sensitive_axes
        ):
            raise ValueError(
                "bounded governor fidelity requires metric floors and sensitive-axis allowances"
            )
        if self.mode is FidelityMode.BEST_EFFORT and (
            self.budget is None or not self.budget.allowed_sensitive_axes
        ):
            raise ValueError(
                "best_effort governor fidelity requires explicit sensitive-axis allowances"
            )
        if self.mode is FidelityMode.RESEARCH and self.budget is not None:
            raise ValueError("research governor fidelity accepts no bounded budget")


@dataclass(frozen=True)
class TraceRequest:
    """One validated attribution problem and its explicitly owned policies."""

    problem: AttributionProblem
    semantics: TraceSemantics = field(default_factory=TraceSemantics)
    execution: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    evidence: TraceEvidence = field(default_factory=TraceEvidence)
    physical_requirements: PhysicalExecutionRequirements | None = None
    governor_fidelity: GovernorFidelityPolicy = field(default_factory=GovernorFidelityPolicy)
    calibration_catalog: CalibrationCatalog | None = None
    governor_admission_mode: AdmissionMode = AdmissionMode.ENFORCE

    def __post_init__(self) -> None:
        if not isinstance(self.governor_admission_mode, AdmissionMode):
            raise ValueError("governor admission mode must be an AdmissionMode")
