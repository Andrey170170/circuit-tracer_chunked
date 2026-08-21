"""Bounded resource-sampling policy for attribution hot paths."""

from __future__ import annotations


PHASE4_DENSE_RESOURCE_SAMPLES = 3
PHASE4_RESOURCE_SAMPLE_INTERVAL = 32


def should_sample_phase4_resources(
    *,
    sample_index: int,
    final: bool = False,
) -> bool:
    """Sample transitions densely, steady state periodically, and final state."""
    return (
        sample_index <= PHASE4_DENSE_RESOURCE_SAMPLES
        or sample_index % PHASE4_RESOURCE_SAMPLE_INTERVAL == 0
        or final
    )


def should_sample_batch_resources(
    *,
    phase_label: str,
    phase_batch_index: int,
    retain_graph: bool,
) -> bool:
    """Preserve non-Phase-4 evidence while bounding Phase-4 sampling work."""
    if phase_label != "phase4_features":
        return True
    return should_sample_phase4_resources(
        sample_index=phase_batch_index,
        final=not retain_graph,
    )
