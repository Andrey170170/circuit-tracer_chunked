"""Backend-neutral diagnostic termination values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeCompletion:
    """Non-scientific terminal result returned through normal cleanup."""

    mode: str
    phase4_batches_completed: int = 0
