import pytest
import torch

from circuit_tracer.attribution.attribute_nnsight import (
    _compute_row_abs_sums,
    _compute_row_denominator_scaled_l1,
    _FileBackedFeatureRowStore,
    _resolve_exact_trace_internal_dtype,
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
