"""Phase D direct controls for bounded NNSight execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NNSightSessionControls:
    session_capacity: int
    phase3_microbatch_max_rows: int
    phase4_microbatch_max_rows: int
    metadata: dict[str, object]


def ordered_physical_ranges(
    *, total_rows: int, logical_batch_rows: int, physical_batch_rows: int
) -> list[tuple[int, int, int]]:
    """Return ``(logical index, start, end)`` ranges in canonical row order."""
    ranges: list[tuple[int, int, int]] = []
    for logical_index, logical_start in enumerate(range(0, total_rows, logical_batch_rows)):
        logical_end = min(logical_start + logical_batch_rows, total_rows)
        for start in range(logical_start, logical_end, physical_batch_rows):
            ranges.append((logical_index, start, min(start + physical_batch_rows, logical_end)))
    return ranges


def resolve_nnsight_session_controls(
    *,
    nnsight_session_capacity: int | None,
    phase3_compute_microbatch_max_rows: int | None,
    phase4_compute_microbatch_max_rows: int | None,
    legacy_session_capacity: int,
    legacy_phase3_batch_rows: int,
    legacy_phase4_batch_rows: int,
) -> NNSightSessionControls:
    """Validate direct controls and translate omitted values compatibly."""
    requested = {
        "nnsight_session_capacity": nnsight_session_capacity,
        "phase3_compute_microbatch_max_rows": phase3_compute_microbatch_max_rows,
        "phase4_compute_microbatch_max_rows": phase4_compute_microbatch_max_rows,
    }
    for name, value in requested.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 when provided")

    capacity = (
        legacy_session_capacity
        if nnsight_session_capacity is None
        else nnsight_session_capacity
    )
    phase3_rows = (
        legacy_phase3_batch_rows
        if phase3_compute_microbatch_max_rows is None
        else phase3_compute_microbatch_max_rows
    )
    phase4_rows = (
        legacy_phase4_batch_rows
        if phase4_compute_microbatch_max_rows is None
        else phase4_compute_microbatch_max_rows
    )
    if phase3_rows > capacity:
        raise ValueError(
            "phase3_compute_microbatch_max_rows must be <= nnsight_session_capacity "
            f"(effective values: {phase3_rows} > {capacity})"
        )
    if phase4_rows > capacity:
        raise ValueError(
            "phase4_compute_microbatch_max_rows must be <= nnsight_session_capacity "
            f"(effective values: {phase4_rows} > {capacity})"
        )

    derived = [name for name, value in requested.items() if value is None]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "compatibility_translation": "nnsight_session_controls_v1",
        "legacy_derived_fields": derived,
        "requested_session_capacity": nnsight_session_capacity,
        "effective_session_capacity": int(capacity),
        "requested_phase3_microbatch_max_rows": phase3_compute_microbatch_max_rows,
        "effective_phase3_microbatch_max_rows": int(phase3_rows),
        "requested_phase4_microbatch_max_rows": phase4_compute_microbatch_max_rows,
        "effective_phase4_microbatch_max_rows": int(phase4_rows),
    }
    return NNSightSessionControls(
        session_capacity=int(capacity),
        phase3_microbatch_max_rows=int(phase3_rows),
        phase4_microbatch_max_rows=int(phase4_rows),
        metadata=metadata,
    )
