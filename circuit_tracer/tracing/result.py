"""Canonical tracing results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from circuit_tracer.execution_identity import EffectiveExecutionDescriptor


class TraceStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PROBE_COMPLETED = "probe_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUSED = "refused"


@dataclass(frozen=True)
class TraceResult:
    output: Any
    semantic_fingerprint: str
    requested_execution_fingerprint: str
    status: TraceStatus
    effective_execution_fingerprint: str | None = None
    effective_execution: EffectiveExecutionDescriptor | None = None
    telemetry_summary: Mapping[str, Any] = field(default_factory=dict)
    telemetry_events: tuple[Mapping[str, Any], ...] = ()
    admission_report: Any = None

    @property
    def graph(self) -> Any:
        return self.output

    @property
    def execution_fingerprint(self) -> str:
        """Compatibility identity: effective after preparation, requested otherwise."""
        return self.effective_execution_fingerprint or self.requested_execution_fingerprint
