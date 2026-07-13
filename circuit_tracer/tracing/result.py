"""Canonical tracing results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TraceStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TraceResult:
    output: Any
    semantic_fingerprint: str
    execution_fingerprint: str
    status: TraceStatus
    telemetry_summary: Mapping[str, Any] = field(default_factory=dict)
    admission_report: Any = None

    @property
    def graph(self) -> Any:
        return self.output

