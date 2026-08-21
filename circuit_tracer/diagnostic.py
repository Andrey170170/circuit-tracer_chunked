"""Backend-neutral diagnostic termination values."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ProbeCompletion:
    """Non-scientific terminal result returned through normal cleanup."""

    mode: str
    phase4_batches_completed: int = 0
    diagnostic_metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    diagnostic_artifacts: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
