"""Typed events and probes emitted by tracing domain code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class TraceEvent:
    """A scalar lifecycle or domain event, independent of any sink schema."""

    scope: Literal["run", "phase", "batch", "op"]
    name: str
    phase: str | None = None
    step_index: int | None = None
    batch_index: int | None = None
    elapsed_ms: float | int | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)
    wall_clock: bool = False


@dataclass(frozen=True)
class MemoryBoundary:
    label: str
    device: Any
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseMetrics:
    label: str
    started_at: float
    device: Any
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchProfile:
    label: str
    batch_index: int
    total_batches: int | None
    elapsed_seconds: float
    context_before: Mapping[str, object] | None
    context_after: Mapping[str, object] | None
    transcoder_before: Mapping[str, object] | None
    transcoder_after: Mapping[str, object] | None


@dataclass(frozen=True)
class SparsificationProfile:
    stats: Mapping[str, object]


@dataclass(frozen=True)
class DiagnosticsMessage:
    label: str
    diagnostics: Mapping[str, object]
    limit: int = 20


@dataclass(frozen=True)
class DiagnosticSnapshot:
    source: Any


@dataclass(frozen=True)
class MemorySnapshot:
    device: Any


@dataclass(frozen=True)
class CudaMemoryProbe:
    """Typed request for one CUDA allocator probe."""

    operation: Literal["snapshot", "reset_peak", "synchronize"]
    device: Any = None


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """CUDA allocator state returned by a resource capability."""

    available: bool
    current_reserved_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    total_bytes: int | None = None


@dataclass(frozen=True)
class MemorySnapshotAttrs:
    snapshot: Mapping[str, object] | None
    keys: tuple[str, ...] | None = None
    prefix: str = "memory"


@dataclass(frozen=True)
class MemoryDelta:
    before: Mapping[str, object]
    after: Mapping[str, object]
    keys: tuple[str, ...] | None = None


@dataclass(frozen=True)
class NumericDelta:
    before: Mapping[str, object] | None
    after: Mapping[str, object] | None


@dataclass(frozen=True)
class RuntimeSnapshot:
    device: Any
    context: Any = None
    transcoder: Any = None


Observation = (
    TraceEvent
    | MemoryBoundary
    | PhaseMetrics
    | BatchProfile
    | SparsificationProfile
    | DiagnosticsMessage
    | DiagnosticSnapshot
    | MemorySnapshot
    | MemorySnapshotAttrs
    | MemoryDelta
    | NumericDelta
    | RuntimeSnapshot
)


class TraceObserver(Protocol):
    """The only observability capability visible to tracing domain code."""

    def observe(self, observation: Observation) -> object | None: ...
