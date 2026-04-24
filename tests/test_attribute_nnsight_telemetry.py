import inspect
from typing import get_args

import pytest
import torch

from circuit_tracer.attribution.attribute import attribute as attribute_entrypoint
from circuit_tracer.attribution.attribute_nnsight import (
    _copy_rows_to_cpu_staging,
    _compute_row_abs_sums,
    _compute_row_denominator_scaled_l1,
    _FileBackedFeatureRowStore,
    _resolve_exact_trace_internal_dtype,
    attribute as nnsight_attribute,
)
from circuit_tracer.utils.telemetry import TelemetryRecorder


def test_file_backed_feature_row_store_emits_structured_events() -> None:
    recorder = TelemetryRecorder(enabled=True)
    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        read_chunk_cache_bytes=4096,
        telemetry_recorder=recorder,
    )

    try:
        rows = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 0.75]], dtype=torch.float32)
        row_denominator_scaled_l1 = _compute_row_denominator_scaled_l1(rows, dtype=torch.float32)
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=row_denominator_scaled_l1,
            phase="phase3",
        )

        read_rows = store.read_feature_rows(0, 2, phase="phase4")
        assert read_rows.shape == (2, 3)
        cached_rows = store.read_feature_rows(0, 2, phase="phase4")
        assert torch.allclose(cached_rows, read_rows)

        dense = store.materialize_dense_feature_slice(
            row_start=0,
            row_end=2,
            selected_feature_columns=torch.tensor([0, 2]),
            phase="phase5",
        )
        assert dense.shape == (2, 2)
    finally:
        store.cleanup()

    stats = store.get_diagnostic_snapshot()
    assert stats["append_call_count"] == 1
    assert stats["read_call_count"] == 2
    assert stats["read_cache_hit_count"] == 1
    assert stats["read_cache_miss_count"] == 1
    assert stats["materialize_call_count"] == 1

    summary = recorder.build_summary()
    assert summary["counts_by_scope"]["op"] >= 3
    events = recorder.export(include_events=True)["events"]
    names = {event["name"] for event in events}
    assert "feature_row_store.append_rows" in names
    assert "feature_row_store.read_rows" in names
    assert "feature_row_store.materialize_dense_slice" in names


def test_file_backed_feature_row_store_read_cache_invalidates_on_overlap_append() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        read_chunk_cache_bytes=4096,
    )

    try:
        store.append_rows(
            row_start=0,
            feature_rows=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([6.0, 15.0], dtype=torch.float32),
        )
        first_read = store.read_feature_rows(0, 2)
        second_read = store.read_feature_rows(0, 2)
        assert torch.allclose(first_read, second_read)

        # Overlapping append should invalidate the cached [0, 2) read.
        store.append_rows(
            row_start=1,
            feature_rows=torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([27.0], dtype=torch.float32),
        )
        refreshed = store.read_feature_rows(0, 2)
    finally:
        store.cleanup()

    assert refreshed[1, 0].item() == 9.0
    stats = store.get_diagnostic_snapshot()
    assert stats["read_cache_hit_count"] == 1
    assert stats["read_cache_miss_count"] == 2


def test_file_backed_feature_row_store_read_cache_too_large_is_reported() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        read_chunk_cache_bytes=8,
    )

    try:
        store.append_rows(
            row_start=0,
            feature_rows=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([6.0, 15.0], dtype=torch.float32),
        )
        _ = store.read_feature_rows(0, 2)
        _ = store.read_feature_rows(0, 2)
    finally:
        store.cleanup()

    stats = store.get_diagnostic_snapshot()
    assert stats["read_cache_enabled"] == 1
    assert stats["read_cache_store_success_count"] == 0
    assert stats["read_cache_store_skip_too_large_count"] == 2
    assert stats["read_cache_hit_count"] == 0
    assert stats["read_cache_miss_count"] == 2


def test_exact_trace_internal_dtype_resolution_supports_fp32_and_fp64() -> None:
    assert _resolve_exact_trace_internal_dtype("fp32") == torch.float32
    assert _resolve_exact_trace_internal_dtype("FP64") == torch.float64


def test_exact_trace_internal_dtype_default_is_fp32_on_public_entrypoints() -> None:
    assert (
        inspect.signature(attribute_entrypoint).parameters["exact_trace_internal_dtype"].default
        == "fp32"
    )
    assert (
        inspect.signature(nnsight_attribute).parameters["exact_trace_internal_dtype"].default
        == "fp32"
    )


def test_phase4_scheduler_defaults_match_between_public_entrypoints() -> None:
    entrypoint_sig = inspect.signature(attribute_entrypoint)
    nnsight_sig = inspect.signature(nnsight_attribute)

    assert (
        entrypoint_sig.parameters["phase4_scheduler_mode"].default
        == nnsight_sig.parameters["phase4_scheduler_mode"].default
        == "locality"
    )
    assert (
        entrypoint_sig.parameters["phase4_scheduler_debug"].default
        == nnsight_sig.parameters["phase4_scheduler_debug"].default
        is False
    )
    assert (
        entrypoint_sig.parameters["phase4_scheduler_telemetry_detail"].default
        == nnsight_sig.parameters["phase4_scheduler_telemetry_detail"].default
        == "normal"
    )
    assert (
        entrypoint_sig.parameters["phase4_refresh_optimization"].default
        == nnsight_sig.parameters["phase4_refresh_optimization"].default
        == "off"
    )
    assert (
        entrypoint_sig.parameters["phase4_row_executor"].default
        == nnsight_sig.parameters["phase4_row_executor"].default
        == "batched"
    )


def test_phase4_scheduler_mode_type_hints_include_planner_v2() -> None:
    entrypoint_mode_annotation = (
        inspect.signature(attribute_entrypoint).parameters["phase4_scheduler_mode"].annotation
    )
    nnsight_mode_annotation = (
        inspect.signature(nnsight_attribute).parameters["phase4_scheduler_mode"].annotation
    )

    entrypoint_modes = set(get_args(entrypoint_mode_annotation))
    nnsight_modes = set(get_args(nnsight_mode_annotation))
    assert "planner_v2" in entrypoint_modes
    assert "planner_v2" in nnsight_modes


def test_phase4_execution_flag_type_hints_include_new_modes() -> None:
    entrypoint_sig = inspect.signature(attribute_entrypoint)
    nnsight_sig = inspect.signature(nnsight_attribute)

    entry_refresh_modes = set(
        get_args(entrypoint_sig.parameters["phase4_refresh_optimization"].annotation)
    )
    nnsight_refresh_modes = set(
        get_args(nnsight_sig.parameters["phase4_refresh_optimization"].annotation)
    )
    assert "v1" in entry_refresh_modes
    assert "v1" in nnsight_refresh_modes

    entry_row_modes = set(get_args(entrypoint_sig.parameters["phase4_row_executor"].annotation))
    nnsight_row_modes = set(get_args(nnsight_sig.parameters["phase4_row_executor"].annotation))
    assert "streaming_v1" in entry_row_modes
    assert "streaming_v1" in nnsight_row_modes


def test_exact_trace_internal_dtype_resolution_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="exact_trace_internal_dtype"):
        _resolve_exact_trace_internal_dtype("bf16")


def test_compute_row_abs_sums_uses_requested_dtype() -> None:
    rows = torch.tensor([[1.0, -2.0], [0.125, -0.5]], dtype=torch.float32)

    row_abs_fp32 = _compute_row_abs_sums(rows, dtype=torch.float32)
    row_abs_fp64 = _compute_row_abs_sums(rows, dtype=torch.float64)

    assert row_abs_fp32.dtype == torch.float32
    assert row_abs_fp64.dtype == torch.float64
    assert torch.allclose(row_abs_fp32.to(dtype=torch.float64), row_abs_fp64)


def test_copy_rows_to_cpu_staging_reuses_existing_buffer() -> None:
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rows_cpu, staging = _copy_rows_to_cpu_staging(first, staging_buffer=None)

    assert staging is None
    assert rows_cpu.data_ptr() == first.data_ptr()

    second = torch.tensor([[100.0, 101.0, 102.0, 103.0]], dtype=torch.float32)
    rows_cpu, staging = _copy_rows_to_cpu_staging(
        second,
        staging_buffer=staging,
        dtype=torch.float64,
    )
    assert staging is not None
    assert rows_cpu.dtype == torch.float64
    assert torch.allclose(rows_cpu, second.to(dtype=torch.float64))

    smaller = torch.tensor([[7.0, 8.0, 9.0, 10.0]], dtype=torch.float64)
    prior_ptr = staging.data_ptr()
    rows_cpu, staging = _copy_rows_to_cpu_staging(smaller, staging_buffer=staging)
    assert staging is not None
    assert staging.data_ptr() == prior_ptr
    assert torch.allclose(rows_cpu, smaller)


def test_copy_rows_to_cpu_staging_resizes_when_batch_grows() -> None:
    small = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    rows_cpu, staging = _copy_rows_to_cpu_staging(small, staging_buffer=None)

    assert staging is None
    assert rows_cpu.data_ptr() == small.data_ptr()

    grow = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    rows_cpu, staging = _copy_rows_to_cpu_staging(grow, staging_buffer=staging)
    assert staging is None
    assert rows_cpu.data_ptr() == grow.data_ptr()

    needs_copy = grow.to(dtype=torch.float32)
    rows_cpu, staging = _copy_rows_to_cpu_staging(
        needs_copy,
        staging_buffer=staging,
        dtype=torch.float64,
    )
    assert staging is not None
    assert staging.shape == (3, 4)
    assert torch.allclose(rows_cpu, grow)

    larger = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    prior_ptr = staging.data_ptr()
    rows_cpu, staging = _copy_rows_to_cpu_staging(
        larger,
        staging_buffer=staging,
        dtype=torch.float64,
    )
    assert staging is not None
    assert staging.shape == (5, 4)
    assert staging.data_ptr() != prior_ptr
    assert torch.allclose(rows_cpu, larger.to(dtype=torch.float64))


def test_compute_row_denominator_scaled_l1_builds_stable_components() -> None:
    rows = torch.tensor(
        [
            [1e38, -1e38, 1e38, -1e38],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    row_abs_max, row_l1_scaled = _compute_row_denominator_scaled_l1(rows, dtype=torch.float32)

    assert row_abs_max.dtype == torch.float32
    assert row_l1_scaled.dtype == torch.float32
    assert row_abs_max[0].item() == pytest.approx(1e38)
    assert row_l1_scaled[0].item() == pytest.approx(4.0)
    assert row_abs_max[1].item() == pytest.approx(0.0)
    assert row_l1_scaled[1].item() == pytest.approx(0.0)


def test_compute_row_denominator_scaled_l1_handles_infinite_rows_without_nan() -> None:
    rows = torch.tensor([[float("inf"), 1.0, 0.0]], dtype=torch.float32)

    row_abs_max, row_l1_scaled = _compute_row_denominator_scaled_l1(rows, dtype=torch.float32)

    assert torch.isinf(row_abs_max).all()
    assert torch.equal(row_l1_scaled, torch.ones_like(row_l1_scaled))


def test_compute_row_denominator_scaled_l1_chunked_matches_reference_on_strided_input() -> None:
    padded = torch.zeros((3, 5003), dtype=torch.float32)
    padded[0, 1:] = 2.0
    padded[2, 1] = float("inf")
    rows = padded[:, 1:]
    assert not rows.is_contiguous()

    row_abs_max, row_l1_scaled = _compute_row_denominator_scaled_l1(rows, dtype=torch.float32)

    reference_abs = rows.to(device="cpu", dtype=torch.float32).abs()
    reference_row_abs_max = reference_abs.amax(dim=1)
    reference_row_l1_scaled = torch.zeros_like(reference_row_abs_max)
    nonzero_rows = (reference_row_abs_max > 0) & torch.isfinite(reference_row_abs_max)
    if bool(nonzero_rows.any()):
        reference_row_l1_scaled[nonzero_rows] = (
            reference_abs[nonzero_rows] / reference_row_abs_max[nonzero_rows].unsqueeze(1)
        ).sum(dim=1)
    infinite_rows = torch.isinf(reference_row_abs_max)
    if bool(infinite_rows.any()):
        reference_row_l1_scaled[infinite_rows] = 1

    assert torch.allclose(row_abs_max, reference_row_abs_max)
    assert torch.allclose(row_l1_scaled, reference_row_l1_scaled)
    assert row_l1_scaled[1].item() == pytest.approx(0.0)
    assert row_l1_scaled[2].item() == pytest.approx(1.0)


def test_file_backed_feature_row_store_append_rows_supports_strided_cpu_slices() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
        row_abs_sum_dtype=torch.float64,
    )

    try:
        padded_rows = torch.tensor(
            [[1.0, 2.0, 3.0, 99.0], [0.0, 0.0, 0.0, 77.0]],
            dtype=torch.float32,
        )
        rows = padded_rows[:, :3]
        assert not rows.is_contiguous()

        store.append_rows(
            row_start=0,
            feature_rows=rows,
            full_row_abs_sums=torch.tensor([6.0, 0.0], dtype=torch.float64),
        )
        restored = store.read_feature_rows(0, 2)
    finally:
        store.cleanup()

    assert torch.allclose(restored, rows)
    assert torch.allclose(store.row_abs_max[:2], torch.tensor([6.0, 0.0], dtype=torch.float64))
    assert torch.allclose(store.row_l1_scaled[:2], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_file_backed_feature_row_store_append_rows_accepts_strided_scaled_l1_tuple() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
        row_abs_sum_dtype=torch.float64,
    )

    try:
        padded_rows = torch.tensor(
            [[1.0, -2.0, 3.0, 5.0], [0.0, 0.0, 0.0, 7.0]],
            dtype=torch.float32,
        )
        rows = padded_rows[:, :3]
        assert not rows.is_contiguous()
        row_denominator = _compute_row_denominator_scaled_l1(rows, dtype=torch.float64)

        store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=row_denominator,
        )
        restored = store.read_feature_rows(0, 2)
    finally:
        store.cleanup()

    assert torch.allclose(restored, rows)
    assert torch.allclose(store.row_abs_max[:2], row_denominator[0])
    assert torch.allclose(store.row_l1_scaled[:2], row_denominator[1])


def test_file_backed_feature_row_store_append_rows_works_with_read_only_memmap_view() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
    )

    try:
        assert store._rows is not None
        store._rows.flags.writeable = False

        rows = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]], dtype=torch.float32)
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(rows, dtype=torch.float32),
        )
        store._rows.flags.writeable = True
        restored = store.read_feature_rows(0, 2)
    finally:
        store.cleanup()

    assert torch.allclose(restored, rows)


def test_file_backed_row_store_materialize_dtype_tracks_denominator_dtype() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=2,
        dtype=torch.float32,
        row_abs_sum_dtype=torch.float64,
    )

    try:
        rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(rows, dtype=torch.float64),
        )
        dense = store.materialize_dense_feature_slice(
            row_start=0,
            row_end=2,
            selected_feature_columns=torch.tensor([0, 1]),
        )
    finally:
        store.cleanup()

    assert dense.dtype == torch.float64


def test_file_backed_row_store_materialize_same_dtype_preserves_order_after_cleanup() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=3,
        n_feature_columns=5,
        dtype=torch.float32,
        row_abs_sum_dtype=torch.float32,
    )

    rows = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=torch.float32,
    )
    selected = torch.tensor([4, 1, 3], dtype=torch.long)

    try:
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            full_row_abs_sums=torch.tensor([15.0, 40.0, 65.0], dtype=torch.float32),
        )
        dense = store.materialize_dense_feature_slice(
            row_start=0,
            row_end=3,
            selected_feature_columns=selected,
            col_chunk_size=2,
        )
    finally:
        store.cleanup()

    expected = rows[:, selected]
    assert torch.allclose(dense, expected)
    dense[0, 0] = -123.0
    assert dense[0, 0].item() == pytest.approx(-123.0)
