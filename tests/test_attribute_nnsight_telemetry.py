import os
from typing import get_args, get_type_hints

import pytest
import torch

from circuit_tracer.attribution.nnsight.phase_support import (
    _copy_rows_to_cpu_staging,
    _resolve_phase3_effective_row_state,
)
from circuit_tracer.attribution.nnsight.numerics import (
    _resolve_exact_trace_internal_dtype,
    _row_abs_sums_to_scaled_l1,
)
from circuit_tracer.attribution.nnsight.replay import (
    _compute_row_abs_sums,
    _compute_row_denominator_scaled_l1,
)
from circuit_tracer.attribution.nnsight.row_store import (
    _FileBackedFeatureRowStore,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_phase4_refresh_substage_telemetry,
    _build_row_transfer_telemetry,
)
from circuit_tracer.observability.exception_export import _attach_telemetry_export_to_exception
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.tracing import (
    ExecutionConstraints,
    FrontierExpansionPlan,
    FrontierSemantics,
    RowStoragePlan,
    SessionPlan,
    TraceSemantics,
)
from circuit_tracer.utils.telemetry import TelemetryRecorder


def test_file_backed_feature_row_store_emits_structured_events() -> None:
    recorder = TelemetryRecorder(enabled=True)
    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        read_chunk_cache_bytes=4096,
        trace_observer=TelemetryObserver(recorder),
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


def test_telemetry_export_attaches_to_exception() -> None:
    recorder = TelemetryRecorder(enabled=True)
    recorder.record_event(
        scope="phase",
        name="phase1.forward",
        phase="phase1",
        elapsed_ms=12.5,
        attrs={"active_features": 3},
    )
    exc = RuntimeError("synthetic failure")

    _attach_telemetry_export_to_exception(exc, recorder.export(include_events=True))

    summary = getattr(exc, "circuit_tracer_telemetry_summary")
    events = getattr(exc, "circuit_tracer_telemetry_events")
    assert summary["event_count"] == 1
    assert events == [
        {
            "t_rel_ms": events[0]["t_rel_ms"],
            "scope": "phase",
            "name": "phase1.forward",
            "phase": "phase1",
            "elapsed_ms": 12.5,
            "attrs": {"active_features": 3},
        }
    ]


def test_file_backed_feature_row_store_prepared_cache_hits_invalidates_and_skips() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=3,
        n_feature_columns=2,
        dtype=torch.float32,
        prepared_read_cache_bytes=32,
    )
    try:
        rows = torch.tensor([[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0]], dtype=torch.float32)
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(rows, dtype=torch.float32),
        )
        first = store.read_prepared_feature_rows(0, 2, device="cpu", dtype=torch.float32)
        second = store.read_prepared_feature_rows(0, 2, device="cpu", dtype=torch.float32)
        assert torch.equal(first, rows[:2].abs())
        assert second.data_ptr() == first.data_ptr()

        store.append_rows(
            row_start=1,
            feature_rows=torch.tensor([[7.0, -8.0]], dtype=torch.float32),
            row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(
                torch.tensor([[7.0, -8.0]], dtype=torch.float32), dtype=torch.float32
            ),
        )
        refreshed = store.read_prepared_feature_rows(0, 2, device="cpu", dtype=torch.float32)
        assert torch.equal(refreshed, torch.tensor([[1.0, 2.0], [7.0, 8.0]]))
        stats = store.get_diagnostic_snapshot()
        assert stats["prepared_read_cache_hit_count"] == 1
        assert stats["prepared_read_cache_miss_count"] == 2
        assert stats["prepared_read_cache_invalidation_entry_count"] == 1
        assert stats["prepared_read_cache_store_success_count"] >= 2
    finally:
        store.cleanup()

    tiny_store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=2,
        dtype=torch.float32,
        prepared_read_cache_bytes=4,
    )
    try:
        rows = torch.ones((2, 2), dtype=torch.float32)
        tiny_store.append_rows(
            row_start=0,
            feature_rows=rows,
            row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(rows, dtype=torch.float32),
        )
        tiny_store.read_prepared_feature_rows(0, 2, device="cpu", dtype=torch.float32)
        tiny_stats = tiny_store.get_diagnostic_snapshot()
        assert tiny_stats["prepared_read_cache_store_skip_too_large_count"] == 1
        assert tiny_stats["prepared_read_cache_entry_count"] == 0
    finally:
        tiny_store.cleanup()


def test_phase4_refresh_telemetry_exports_gpu_row_tier_counters() -> None:
    payload = _build_phase4_refresh_substage_telemetry(
        telemetry_detail="normal",
        partial_influence_elapsed_ms=1.0,
        rank_topk_elapsed_ms=2.0,
        frontier_plan_elapsed_ms=3.0,
        row_store_read_elapsed_ms=4.0,
        influence_normalization_elapsed_ms=5.0,
        influence_matmul_elapsed_ms=6.0,
        chunk_request_count=7,
        active_row_chunk_count=8,
        row_reader_row_count=9,
        solver_iteration_count=10,
        feature_row_store_read_stats={
            "gpu_row_tier_read_hits": 11,
            "gpu_row_tier_read_hit_rows": 12,
            "gpu_row_tier_read_hit_bytes": 13,
            "gpu_row_tier_read_fallbacks": 0,
            "gpu_row_tier_read_fallback_rows": 0,
            "gpu_row_tier_avoided_file_read_bytes": 14,
            "gpu_row_tier_avoided_h2d_bytes": 15,
            "gpu_row_tier_d2h_bytes": 21,
            "gpu_row_tier_read_transfer_elapsed_ms": 22.5,
            "gpu_row_tier_copy_failures": 0,
            "gpu_row_tier_append_calls": 16,
            "gpu_row_tier_append_rows": 17,
            "gpu_row_tier_append_bytes": 18,
            "gpu_row_tier_high_water_bytes": 19,
            "gpu_row_tier_owned_bytes": 20,
        },
    )

    assert payload["feature_row_store_gpu_tier_read_hits"] == 11
    assert payload["feature_row_store_gpu_tier_read_hit_rows"] == 12
    assert payload["feature_row_store_gpu_tier_read_hit_bytes"] == 13
    assert payload["feature_row_store_gpu_tier_read_fallbacks"] == 0
    assert payload["feature_row_store_gpu_tier_avoided_file_read_bytes"] == 14
    assert payload["feature_row_store_gpu_tier_avoided_h2d_bytes"] == 15
    assert payload["feature_row_store_gpu_tier_d2h_bytes"] == 21
    assert payload["feature_row_store_gpu_tier_read_transfer_elapsed_ms"] == 22.5
    assert payload["feature_row_store_gpu_tier_copy_failures"] == 0
    assert payload["feature_row_store_gpu_tier_append_bytes"] == 18
    assert payload["feature_row_store_gpu_tier_owned_bytes"] == 20


def test_file_backed_feature_row_store_temp_root_default_and_explicit(tmp_path) -> None:
    default_store = _FileBackedFeatureRowStore(
        n_rows=1,
        n_feature_columns=1,
        dtype=torch.float32,
    )
    try:
        default_stats = default_store.get_diagnostic_snapshot()
        assert default_stats["temp_root_policy"] == "default"
        assert default_stats["temp_root_selected"] is None
        assert default_stats["temp_root_fallback_reason"] is None
    finally:
        default_store.cleanup()

    explicit_root = tmp_path / "rows"
    explicit_root.mkdir()
    store = _FileBackedFeatureRowStore(
        n_rows=1,
        n_feature_columns=1,
        dtype=torch.float32,
        temp_root_policy="env_node_local",
        temp_root=explicit_root,
    )
    try:
        stats = store.get_diagnostic_snapshot()
        assert stats["temp_root_policy"] == "env_node_local"
        assert stats["temp_root_requested"] == os.fspath(explicit_root)
        assert stats["temp_root_selected"] == os.fspath(explicit_root)
        assert os.fspath(store.path).startswith(os.fspath(explicit_root))
    finally:
        store.cleanup()


def test_file_backed_feature_row_store_env_node_local_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLURM_TMPDIR", os.fspath(tmp_path / "missing_slurm"))
    monkeypatch.setenv("TMPDIR", os.fspath(tmp_path / "missing_tmp"))
    store = _FileBackedFeatureRowStore(
        n_rows=1,
        n_feature_columns=1,
        dtype=torch.float32,
        temp_root_policy="env_node_local",
    )
    try:
        stats = store.get_diagnostic_snapshot()
        assert stats["temp_root_policy"] == "env_node_local"
        assert stats["temp_root_selected"] == "/tmp"
        assert stats["temp_root_fallback_reason"] is None
    finally:
        store.cleanup()


def test_file_backed_feature_row_store_preallocation_unavailable(monkeypatch) -> None:
    monkeypatch.delattr(os, "posix_fallocate", raising=False)
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
        preallocate=True,
    )
    try:
        stats = store.get_diagnostic_snapshot()
        assert stats["preallocate_requested"] == 1
        assert stats["preallocate_available"] == 0
        assert stats["preallocate_status"] == "unavailable"
        assert "unavailable" in str(stats["preallocate_error"])
    finally:
        store.cleanup()


def test_file_backed_feature_row_store_preallocation_failure(monkeypatch) -> None:
    def fail_fallocate(fd: int, offset: int, length: int) -> None:
        raise OSError("synthetic fallocate failure")

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate, raising=False)
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
        preallocate=True,
    )
    try:
        stats = store.get_diagnostic_snapshot()
        assert stats["preallocate_requested"] == 1
        assert stats["preallocate_available"] == 1
        assert stats["preallocate_status"] == "failed"
        assert "synthetic fallocate failure" in str(stats["preallocate_error"])
    finally:
        store.cleanup()


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


def test_row_transfer_telemetry_reports_shapes_without_cpu_copy_transfer() -> None:
    rows = torch.ones((2, 4), dtype=torch.float32)
    telemetry = _build_row_transfer_telemetry(
        rows=rows,
        rows_cpu=rows,
        row_input_slice=rows[:, :3],
        feature_row_slice=rows[:, :2],
    )

    assert telemetry["row_transfer_source"] == "cpu"
    assert telemetry["row_transfer_destination"] == "cpu"
    assert telemetry["row_transfer_count"] == 2
    assert telemetry["row_transfer_bytes"] == 0
    assert telemetry["row_input_bytes"] == 2 * 3 * rows.element_size()
    assert telemetry["feature_row_bytes"] == 2 * 2 * rows.element_size()


def test_row_transfer_telemetry_counts_dtype_materialization_bytes() -> None:
    rows = torch.ones((2, 4), dtype=torch.float32)
    rows_cpu = rows.to(dtype=torch.float64)
    telemetry = _build_row_transfer_telemetry(
        rows=rows,
        rows_cpu=rows_cpu,
        row_input_slice=rows_cpu[:, :3],
        feature_row_slice=rows_cpu[:, :2],
    )

    assert telemetry["row_transfer_source"] == "cpu"
    assert telemetry["row_transfer_destination"] == "cpu"
    assert telemetry["row_transfer_count"] == 2
    assert telemetry["row_transfer_bytes"] == rows_cpu.numel() * rows_cpu.element_size()
    assert telemetry["row_input_bytes"] == 2 * 3 * rows_cpu.element_size()
    assert telemetry["feature_row_bytes"] == 2 * 2 * rows_cpu.element_size()


def test_file_backed_feature_row_store_full_row_abs_sums_uses_scaled_representation() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=1,
        n_feature_columns=2,
        dtype=torch.float32,
        row_abs_sum_dtype=torch.float32,
    )

    try:
        store.append_rows(
            row_start=0,
            feature_rows=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([4e38], dtype=torch.float64),
        )
    finally:
        store.cleanup()

    assert torch.isfinite(store.row_abs_max).all()
    assert torch.isfinite(store.row_l1_scaled).all()
    assert store.row_abs_max[0].item() == pytest.approx(torch.finfo(torch.float32).max)
    assert store.row_l1_scaled[0].item() == pytest.approx(4e38 / torch.finfo(torch.float32).max)


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


def test_file_backed_feature_row_store_cache_control_fadvise_tracks_append_byte_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int, int]] = []

    def _fake_posix_fadvise(fd: int, offset: int, length: int, advice: int) -> int:
        calls.append((fd, offset, length, advice))
        return 0

    monkeypatch.setattr(os, "posix_fadvise", _fake_posix_fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 7, raising=False)

    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        row_store_cache_control_mode="fadvise_dontneed_after_append_v1",
    )

    try:
        store.append_rows(
            row_start=1,
            feature_rows=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([6.0, 15.0], dtype=torch.float32),
        )
    finally:
        store.cleanup()

    assert len(calls) == 1
    _, offset, length, advice = calls[0]
    assert offset == 12
    assert length == 24
    assert advice == 7

    stats = store.get_diagnostic_snapshot()
    assert stats["row_store_cache_control_effective_mode"] == "fadvise_dontneed_after_append_v1"
    assert stats["row_store_cache_control_advisory_call_count"] == 1
    assert stats["row_store_cache_control_advisory_bytes"] == 24
    assert stats["row_store_cache_control_append_advisory_call_count"] == 1
    assert stats["row_store_cache_control_append_advisory_bytes"] == 24
    assert stats["row_store_cache_control_read_advisory_call_count"] == 0
    assert stats["row_store_cache_control_advisory_failure_count"] == 0
    assert stats["row_store_cache_control_advisory_unavailable_count"] == 0


def test_file_backed_feature_row_store_cache_control_fadvise_tracks_safe_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int, int]] = []

    def _fake_posix_fadvise(fd: int, offset: int, length: int, advice: int) -> int:
        calls.append((fd, offset, length, advice))
        return 0

    monkeypatch.setattr(os, "posix_fadvise", _fake_posix_fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 7, raising=False)

    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        row_store_cache_control_mode="fadvise_dontneed_after_append_and_read_v1",
    )

    try:
        store.append_rows(
            row_start=1,
            feature_rows=torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([6.0, 15.0], dtype=torch.float32),
        )
        prepared = store.read_prepared_feature_rows(1, 3, device="cpu", dtype=torch.float32)
        assert torch.equal(
            prepared,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        )
    finally:
        store.cleanup()

    assert [(offset, length, advice) for _, offset, length, advice in calls] == [
        (12, 24, 7),
        (12, 24, 7),
    ]
    stats = store.get_diagnostic_snapshot()
    assert stats["row_store_cache_control_append_advisory_call_count"] == 1
    assert stats["row_store_cache_control_append_advisory_bytes"] == 24
    assert stats["row_store_cache_control_read_advisory_call_count"] == 1
    assert stats["row_store_cache_control_read_advisory_bytes"] == 24
    assert stats["row_store_cache_control_advisory_call_count"] == 2
    assert stats["row_store_cache_control_advisory_bytes"] == 48


def test_file_backed_feature_row_store_cache_control_unavailable_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    monkeypatch.delattr(os, "POSIX_FADV_DONTNEED", raising=False)

    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=2,
        dtype=torch.float32,
        row_store_cache_control_mode="fadvise_dontneed_after_append_v1",
    )

    try:
        store.append_rows(
            row_start=0,
            feature_rows=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            full_row_abs_sums=torch.tensor([3.0], dtype=torch.float32),
        )
    finally:
        store.cleanup()

    stats = store.get_diagnostic_snapshot()
    assert stats["row_store_cache_control_effective_mode"] == "fadvise_dontneed_after_append_v1"
    assert stats["row_store_cache_control_advisory_call_count"] == 0
    assert stats["row_store_cache_control_advisory_bytes"] == 0
    assert stats["row_store_cache_control_advisory_unavailable_count"] == 1
    assert stats["row_store_cache_control_advisory_failure_count"] == 0


def test_exact_trace_internal_dtype_resolution_supports_fp32_and_fp64() -> None:
    assert _resolve_exact_trace_internal_dtype("fp32") == torch.float32
    assert _resolve_exact_trace_internal_dtype("FP64") == torch.float64


def test_exact_trace_internal_dtype_default_is_fp32_on_canonical_semantics() -> None:
    assert TraceSemantics().exact_trace_internal_dtype == "fp32"


def test_execution_defaults_live_with_their_canonical_owners() -> None:
    execution = ExecutionConstraints()
    assert execution.frontier.scheduler_debug is False
    assert execution.frontier.scheduler_telemetry_detail == "normal"
    assert execution.frontier.refresh_optimization == "v1"
    assert execution.frontier.row_executor == "batched"
    assert execution.session.phase1_trace_batch_policy == "legacy"
    assert execution.session.phase1_trace_batch_size_max is None
    assert TraceSemantics().frontier.refresh_policy == "standard"
    assert TraceSemantics().frontier.refresh_interval_multiplier == 1
    assert TraceSemantics().frontier.ranker == "argsort"
    assert execution.storage.cache_control == "off"
    assert execution.storage.exact_encoder_residency == "lazy"


def test_canonical_execution_policy_type_hints_include_supported_modes() -> None:
    frontier_hints = get_type_hints(FrontierExpansionPlan)
    session_hints = get_type_hints(SessionPlan)
    semantics_hints = get_type_hints(FrontierSemantics)
    storage_hints = get_type_hints(RowStoragePlan)

    assert "planner_v2" in get_args(semantics_hints["scheduler"])
    assert "v1" in get_args(frontier_hints["refresh_optimization"])
    assert "streaming_v1" in get_args(frontier_hints["row_executor"])
    assert "cap_effective_batches" in get_args(session_hints["phase1_trace_batch_policy"])
    assert "deferred_v1" in get_args(semantics_hints["refresh_policy"])
    assert "topk_v1" in get_args(semantics_hints["ranker"])
    assert "fadvise_dontneed_after_append_v1" in get_args(storage_hints["cache_control"])
    assert "fadvise_dontneed_after_append_and_read_v1" in get_args(
        storage_hints["cache_control"]
    )
    assert "active_pinned_cpu" in get_args(storage_hints["exact_encoder_residency"])


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


def test_row_abs_sums_to_scaled_l1_handles_zero_and_finite_rows() -> None:
    row_abs_max, row_l1_scaled = _row_abs_sums_to_scaled_l1(
        torch.tensor([0.0, 7.5], dtype=torch.float64),
        dtype=torch.float32,
    )

    assert row_abs_max.dtype == torch.float32
    assert row_l1_scaled.dtype == torch.float32
    assert torch.allclose(row_abs_max, torch.tensor([0.0, 7.5], dtype=torch.float32))
    assert torch.equal(row_l1_scaled, torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_row_abs_sums_to_scaled_l1_avoids_fp32_overflow_for_large_raw_sums() -> None:
    row_abs_max, row_l1_scaled = _row_abs_sums_to_scaled_l1(
        torch.tensor([4e38], dtype=torch.float64),
        dtype=torch.float32,
    )

    assert row_abs_max.dtype == torch.float32
    assert row_l1_scaled.dtype == torch.float32
    assert torch.isfinite(row_abs_max).all()
    assert torch.isfinite(row_l1_scaled).all()
    assert row_abs_max.item() == pytest.approx(torch.finfo(torch.float32).max)
    assert row_l1_scaled.item() == pytest.approx(4e38 / torch.finfo(torch.float32).max)


def test_resolve_phase3_effective_row_state_without_donor_provides_capture_row_sums() -> None:
    rows = torch.tensor(
        [
            [1.0, -2.0, 3.0, 99.0],
            [0.0, 0.0, 0.0, 88.0],
        ],
        dtype=torch.float32,
    )

    (
        effective_rows,
        row_input_slice,
        feature_row_slice,
        row_denominator_scaled_l1,
        row_abs_sums_cpu,
    ) = _resolve_phase3_effective_row_state(
        rows_cpu=rows,
        row_input_column_count=3,
        total_active_features=2,
        dtype=torch.float32,
    )

    assert effective_rows is rows
    assert torch.allclose(row_input_slice, rows[:, :3])
    assert torch.allclose(feature_row_slice, rows[:, :2])
    assert row_abs_sums_cpu.dtype == torch.float64
    assert torch.allclose(row_abs_sums_cpu, torch.tensor([6.0, 0.0], dtype=torch.float64))
    assert torch.allclose(row_denominator_scaled_l1[0], torch.tensor([3.0, 0.0]))
    assert torch.allclose(row_denominator_scaled_l1[1], torch.tensor([2.0, 0.0]))


def test_resolve_phase3_effective_row_state_no_donor_capture_sums_do_not_overflow_fp32() -> None:
    rows = torch.tensor([[1e38, -1e38, 1e38, -1e38]], dtype=torch.float32)

    _, _, _, row_denominator_scaled_l1, row_abs_sums_cpu = _resolve_phase3_effective_row_state(
        rows_cpu=rows,
        row_input_column_count=4,
        total_active_features=2,
        dtype=torch.float32,
    )

    assert torch.isfinite(row_denominator_scaled_l1[0]).all()
    assert torch.isfinite(row_denominator_scaled_l1[1]).all()
    assert row_denominator_scaled_l1[0].item() == pytest.approx(1e38)
    assert row_denominator_scaled_l1[1].item() == pytest.approx(4.0)
    assert row_abs_sums_cpu.dtype == torch.float64
    assert torch.isfinite(row_abs_sums_cpu).all()
    assert row_abs_sums_cpu.item() == pytest.approx(4e38)


def test_resolve_phase3_effective_row_state_uses_donor_rows_and_denominators() -> None:
    rows = torch.tensor(
        [
            [1.0, 2.0, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    donor_feature_rows = torch.tensor(
        [[9.0, 8.0], [7.0, 6.0]],
        dtype=torch.float64,
    )
    donor_row_abs_sums = torch.tensor([100.0, 0.0], dtype=torch.float64)

    (
        effective_rows,
        row_input_slice,
        feature_row_slice,
        row_denominator_scaled_l1,
        row_abs_sums_cpu,
    ) = _resolve_phase3_effective_row_state(
        rows_cpu=rows,
        row_input_column_count=4,
        total_active_features=2,
        dtype=torch.float32,
        donor_feature_rows=donor_feature_rows,
        donor_row_abs_sums=donor_row_abs_sums,
    )

    assert effective_rows is not rows
    assert torch.allclose(rows[:, :2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(feature_row_slice, donor_feature_rows.to(dtype=torch.float32))
    assert torch.allclose(row_input_slice[:, :2], donor_feature_rows.to(dtype=torch.float32))
    assert torch.allclose(row_input_slice[:, 2:], rows[:, 2:])
    assert row_abs_sums_cpu.dtype == torch.float64
    assert torch.allclose(row_abs_sums_cpu, donor_row_abs_sums)
    assert torch.allclose(row_denominator_scaled_l1[0], torch.tensor([100.0, 0.0]))
    assert torch.equal(row_denominator_scaled_l1[1], torch.tensor([1.0, 0.0]))


def test_resolve_phase3_effective_row_state_donor_denominator_avoids_fp32_overflow() -> None:
    rows = torch.tensor([[1.0, 2.0, 10.0, 20.0]], dtype=torch.float32)
    donor_feature_rows = torch.tensor([[9.0, 8.0]], dtype=torch.float64)
    donor_row_abs_sums = torch.tensor([4e38], dtype=torch.float64)

    (
        _,
        row_input_slice,
        feature_row_slice,
        row_denominator_scaled_l1,
        row_abs_sums_cpu,
    ) = _resolve_phase3_effective_row_state(
        rows_cpu=rows,
        row_input_column_count=4,
        total_active_features=2,
        dtype=torch.float32,
        donor_feature_rows=donor_feature_rows,
        donor_row_abs_sums=donor_row_abs_sums,
    )

    assert torch.allclose(feature_row_slice, donor_feature_rows.to(dtype=torch.float32))
    assert torch.allclose(row_input_slice[:, :2], donor_feature_rows.to(dtype=torch.float32))
    assert torch.isfinite(row_denominator_scaled_l1[0]).all()
    assert torch.isfinite(row_denominator_scaled_l1[1]).all()
    assert row_denominator_scaled_l1[0].item() == pytest.approx(torch.finfo(torch.float32).max)
    assert row_denominator_scaled_l1[1].item() == pytest.approx(
        4e38 / torch.finfo(torch.float32).max
    )
    assert row_abs_sums_cpu.dtype == torch.float64
    assert row_abs_sums_cpu.item() == pytest.approx(4e38)


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
