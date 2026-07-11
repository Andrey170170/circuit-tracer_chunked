import pytest

from circuit_tracer.attribution.nnsight.session_controls import (
    ordered_physical_ranges,
    resolve_nnsight_session_controls,
)


def test_ordered_physical_ranges_preserve_logical_groups_and_canonical_order() -> None:
    ranges = ordered_physical_ranges(
        total_rows=9, logical_batch_rows=4, physical_batch_rows=3
    )

    assert ranges == [(0, 0, 3), (0, 3, 4), (1, 4, 7), (1, 7, 8), (2, 8, 9)]
    assert [row for _, start, end in ranges for row in range(start, end)] == list(range(9))


def _resolve(**overrides: int | None):
    values = {
        "nnsight_session_capacity": None,
        "phase3_compute_microbatch_max_rows": None,
        "phase4_compute_microbatch_max_rows": None,
        "legacy_session_capacity": 8,
        "legacy_phase3_batch_rows": 8,
        "legacy_phase4_batch_rows": 4,
    }
    values.update(overrides)
    return resolve_nnsight_session_controls(**values)  # type: ignore[arg-type]


def test_omitted_controls_preserve_legacy_values_with_versioned_metadata() -> None:
    controls = _resolve()

    assert (controls.session_capacity, controls.phase3_microbatch_max_rows) == (8, 8)
    assert controls.phase4_microbatch_max_rows == 4
    assert controls.metadata["schema_version"] == 1
    assert controls.metadata["compatibility_translation"] == "nnsight_session_controls_v1"
    assert controls.metadata["legacy_derived_fields"] == [
        "nnsight_session_capacity",
        "phase3_compute_microbatch_max_rows",
        "phase4_compute_microbatch_max_rows",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "nnsight_session_capacity",
        "phase3_compute_microbatch_max_rows",
        "phase4_compute_microbatch_max_rows",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_explicit_controls_must_be_positive(name: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{name} must be > 0"):
        _resolve(**{name: value})


@pytest.mark.parametrize(
    "name", ["phase3_compute_microbatch_max_rows", "phase4_compute_microbatch_max_rows"]
)
def test_microbatch_cannot_exceed_session_capacity(name: str) -> None:
    with pytest.raises(ValueError, match="must be <= nnsight_session_capacity"):
        _resolve(nnsight_session_capacity=2, **{name: 3})
