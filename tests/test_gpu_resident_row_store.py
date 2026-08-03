from __future__ import annotations

import gc

import pytest
import torch

from circuit_tracer.attribution.nnsight.replay import (
    _compute_row_denominator_scaled_l1,
)
from circuit_tracer.attribution.nnsight.row_store import (
    _FileBackedFeatureRowStore,
    _GpuResidentFeatureRowStore,
    estimate_gpu_row_tier_capacity,
)
from circuit_tracer.graph import compute_partial_feature_influences_streaming
from circuit_tracer.tracing.plan import RowStoragePlan


def _append(
    store: _GpuResidentFeatureRowStore,
    rows: torch.Tensor,
    *,
    row_start: int = 0,
) -> None:
    store.append_rows(
        row_start=row_start,
        feature_rows=rows.cpu(),
        resident_feature_rows=rows,
        row_denominator_scaled_l1=_compute_row_denominator_scaled_l1(
            rows,
            dtype=torch.float32,
        ),
        phase="phase4",
    )


def test_gpu_row_tier_capacity_is_exact() -> None:
    capacity = estimate_gpu_row_tier_capacity(
        n_rows=8193,
        n_feature_columns=127_488,
        dtype=torch.float32,
    )

    assert capacity.element_size == 4
    assert capacity.logical_bytes == 8193 * 127_488 * 4
    assert capacity.required_bytes == capacity.logical_bytes


def test_feature_row_influence_modes_require_their_byte_budgets() -> None:
    with pytest.raises(ValueError, match="cuda_full"):
        RowStoragePlan(feature_row_influence_mode="cuda_full")
    with pytest.raises(ValueError, match="cuda_windowed"):
        RowStoragePlan(feature_row_influence_mode="cuda_windowed")
    with pytest.raises(ValueError, match="cuda_file_windowed"):
        RowStoragePlan(feature_row_influence_mode="cuda_file_windowed")
    with pytest.raises(ValueError, match="full-resident and window"):
        RowStoragePlan(
            feature_row_influence_mode="auto",
            gpu_resident_max_bytes=1024,
        )

    assert (
        RowStoragePlan(
            feature_row_influence_mode="cuda_full",
            gpu_resident_max_bytes=1024,
        ).feature_row_influence_mode
        == "cuda_full"
    )


def test_gpu_row_tier_refuses_before_allocation_and_falls_back() -> None:
    backing = _FileBackedFeatureRowStore(
        n_rows=3,
        n_feature_columns=2,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        max_bytes=1024,
        safety_margin_bytes=0,
        device="cpu",
    )
    rows = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32)
    try:
        _append(store, rows)
        restored = store.read_feature_rows(0, 2, phase="phase4")

        assert torch.equal(restored, rows)
        assert not store.admission.admitted
        assert store.admission.reason == "cuda_full_device_not_cuda"
        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_read_fallbacks"] == 1
        assert stats["gpu_row_tier_read_hits"] == 0
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_row_tier_allocation_failure_preserves_file_path() -> None:
    backing = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=2,
        dtype=torch.float32,
    )

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("synthetic allocation failure")

    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        max_bytes=1024,
        safety_margin_bytes=0,
        device="cuda",
        allocator=fail_allocation,
    )
    rows = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], dtype=torch.float32)
    try:
        _append(store, rows)

        assert not store.admission.admitted
        assert store.admission.reason == "cuda_full_allocation_failed:RuntimeError"
        assert torch.equal(store.read_feature_rows(0, 2, phase="phase4"), rows)
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_full_row_tier_ranges_prepared_reads_and_cleanup() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated(device)
    backing = _FileBackedFeatureRowStore(
        n_rows=5,
        n_feature_columns=4,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        max_bytes=1024,
        safety_margin_bytes=0,
        device=device,
    )
    first = torch.tensor(
        [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
        dtype=torch.float32,
        device=device,
    )
    second = torch.tensor(
        [[9.0, -10.0, 11.0, -12.0], [-13.0, 14.0, -15.0, 16.0]],
        dtype=torch.float32,
        device=device,
    )
    try:
        assert store.admission.admitted
        _append(store, first, row_start=0)

        resident = store.read_feature_rows(0, 2, phase="phase4")
        assert resident.device.type == "cuda"
        assert torch.equal(resident, first)

        fallback = store.read_feature_rows(0, 3, phase="phase4")
        assert fallback.device.type == "cpu"
        assert torch.equal(fallback[:2], first.cpu())

        _append(store, second, row_start=2)
        all_rows = store.read_feature_rows(0, 4, phase="phase4")
        expected = torch.cat((first, second), dim=0)
        assert torch.equal(all_rows, expected)

        prepared = store.read_prepared_feature_rows(
            1,
            4,
            device=device,
            dtype=torch.float32,
            phase="phase4",
        )
        assert torch.equal(prepared.cpu(), expected[1:4].abs().cpu())
        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_read_hits"] == 3
        assert stats["gpu_row_tier_read_fallbacks"] == 1
        assert stats["gpu_row_tier_avoided_file_read_bytes"] == (2 + 4 + 3) * 4 * 4
        assert stats["gpu_row_tier_d2h_bytes"] == 0
        assert stats["gpu_row_tier_avoided_h2d_bytes"] == (2 + 4 + 3) * 4 * 4
        assert stats["gpu_row_tier_owned_bytes"] == 5 * 4 * 4
        assert stats["gpu_row_tier_host_mirror_owned_bytes"] == 0
        assert stats["feature_row_influence_mode_resolved"] == "cuda_full"
        del resident, fallback, all_rows, expected, prepared
    finally:
        store.cleanup()
        del first, second
        gc.collect()
        torch.cuda.empty_cache()

    assert torch.cuda.memory_allocated(device) <= allocated_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_row_tier_streaming_solver_matches_file_reference() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = torch.tensor(
        [
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    denominators = torch.ones(rows.shape[0], dtype=torch.float32)
    row_to_node_index = torch.tensor([4, 1, 0], dtype=torch.int32)
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    backing = _FileBackedFeatureRowStore(
        n_rows=3,
        n_feature_columns=4,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        max_bytes=1024,
        safety_margin_bytes=0,
        device=device,
    )
    try:
        _append(store, rows.to(device))
        reference = compute_partial_feature_influences_streaming(
            lambda start, end: backing.read_feature_rows(start, end),
            denominators,
            logit_p,
            row_to_node_index,
            n_feature_nodes=4,
            n_logits=1,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            active_row_only_chunks=True,
            active_row_accumulation="direct_v1",
        )
        candidate = compute_partial_feature_influences_streaming(
            lambda start, end: store.read_feature_rows(start, end, phase="phase4"),
            denominators,
            logit_p,
            row_to_node_index,
            n_feature_nodes=4,
            n_logits=1,
            device=store.influence_device,
            compute_dtype=torch.float32,
            active_row_only_chunks=True,
            active_row_accumulation="direct_v1",
        )

        assert candidate.device.type == "cuda"
        assert torch.equal(candidate.cpu(), reference)
    finally:
        store.cleanup()


def test_cpu_prepared_mode_serves_precomputed_absolute_rows() -> None:
    backing = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        mode="cpu_prepared",
        max_bytes=0,
        safety_margin_bytes=0,
        device="cpu",
    )
    rows = torch.tensor(
        [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]],
        dtype=torch.float32,
    )
    try:
        _append(store, rows)

        assert store.resolved_influence_mode == "cpu_prepared"
        assert store.phase4_prepared_read_available
        prepared = store.read_prepared_feature_rows(
            1,
            3,
            device="cpu",
            dtype=torch.float32,
            phase="phase4",
        )
        assert torch.equal(prepared, rows[1:3].abs())
        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_prepared_host_mirror_owned_bytes"] == 4 * 3 * 4
        assert stats["gpu_row_tier_prepared_host_mirror_read_bytes"] == 2 * 3 * 4
        assert stats["gpu_row_tier_avoided_file_read_bytes"] == 2 * 3 * 4
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_windowed_mode_bounds_hbm_and_streams_host_rows() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    backing = _FileBackedFeatureRowStore(
        n_rows=5,
        n_feature_columns=4,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        mode="cuda_windowed",
        max_bytes=0,
        window_max_bytes=4 * 4 * 4,
        safety_margin_bytes=0,
        device=device,
    )
    rows = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) - 8
    try:
        assert store.resolved_influence_mode == "cuda_windowed"
        assert store.influence_row_chunk_size == 2
        _append(store, rows)

        streamed = store.read_feature_rows(1, 3, phase="phase4")
        assert streamed.device.type == "cuda"
        assert torch.equal(streamed, rows[1:3])
        with pytest.raises(RuntimeError, match="exceeds admitted CUDA window"):
            store.read_feature_rows(0, 3, phase="phase4")

        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_owned_bytes"] == 4 * 4 * 4
        assert stats["gpu_row_tier_pinned_host_bytes"] == 4 * 4 * 4
        assert stats["gpu_row_tier_window_buffer_count"] == 2
        assert stats["gpu_row_tier_host_mirror_owned_bytes"] == 5 * 4 * 4
        assert stats["gpu_row_tier_h2d_bytes"] == 2 * 4 * 4
        assert stats["gpu_row_tier_window_read_calls"] == 1
        assert stats["gpu_row_tier_window_read_rows"] == 2
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_file_windowed_mode_streams_backing_without_full_host_mirror() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    backing = _FileBackedFeatureRowStore(
        n_rows=5,
        n_feature_columns=4,
        dtype=torch.float32,
        row_store_cache_control_mode="fadvise_dontneed_after_append_and_read_v1",
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        mode="cuda_file_windowed",
        max_bytes=0,
        window_max_bytes=4 * 4 * 4,
        safety_margin_bytes=0,
        device=device,
    )
    rows = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) - 8
    try:
        assert store.resolved_influence_mode == "cuda_file_windowed"
        assert store.influence_row_chunk_size == 2
        _append(store, rows)

        streamed = store.read_feature_rows(1, 3, phase="phase4")
        assert streamed.device.type == "cuda"
        assert torch.equal(streamed, rows[1:3])

        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_owned_bytes"] == 4 * 4 * 4
        assert stats["gpu_row_tier_pinned_host_bytes"] == 4 * 4 * 4
        assert stats["gpu_row_tier_host_mirror_owned_bytes"] == 0
        assert stats["gpu_row_tier_host_mirror_read_bytes"] == 0
        assert stats["gpu_row_tier_file_window_read_bytes"] == 2 * 4 * 4
        assert stats["gpu_row_tier_avoided_file_read_bytes"] == 0
        assert stats["gpu_row_tier_append_bytes"] == 0
        assert stats["read_call_count"] == 1
        assert stats["direct_read_into_call_count"] == 1
        assert stats["direct_read_into_bytes"] == 2 * 4 * 4
        assert stats["read_cache_entry_count"] == 0
        assert stats["row_store_cache_control_read_advisory_call_count"] == 1
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_auto_mode_falls_from_full_residency_to_cuda_window() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    backing = _FileBackedFeatureRowStore(
        n_rows=5,
        n_feature_columns=4,
        dtype=torch.float32,
    )
    store = _GpuResidentFeatureRowStore(
        backing_store=backing,
        mode="auto",
        max_bytes=5 * 4 * 4 - 1,
        window_max_bytes=4 * 4 * 4,
        safety_margin_bytes=0,
        device=device,
    )
    try:
        assert store.resolved_influence_mode == "cuda_windowed"
        assert store.admission.requested_mode == "auto"
        assert store.admission.resolved_mode == "cuda_windowed"
        assert store.admission.window_rows == 2
        assert store.admission.window_buffer_count == 2
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_windowed_solver_matches_cuda_full_with_fixed_chunk_geometry() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = torch.tensor(
        [
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    denominators = torch.ones(rows.shape[0], dtype=torch.float32)
    row_to_node_index = torch.tensor([4, 1, 0], dtype=torch.int32)
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    stores: list[_GpuResidentFeatureRowStore] = []
    try:
        for mode, full_bytes, window_bytes in (
            ("cuda_full", 1024, 0),
            ("cuda_windowed", 0, 4 * 4 * 4),
            ("cuda_file_windowed", 0, 4 * 4 * 4),
        ):
            backing = _FileBackedFeatureRowStore(
                n_rows=3,
                n_feature_columns=4,
                dtype=torch.float32,
            )
            store = _GpuResidentFeatureRowStore(
                backing_store=backing,
                mode=mode,
                max_bytes=full_bytes,
                window_max_bytes=window_bytes,
                safety_margin_bytes=0,
                device=device,
            )
            _append(store, rows)
            stores.append(store)

        results = [
            compute_partial_feature_influences_streaming(
                lambda start, end, active_store=store: active_store.read_feature_rows(
                    start,
                    end,
                    phase="phase4",
                ),
                denominators,
                logit_p,
                row_to_node_index,
                n_feature_nodes=4,
                n_logits=1,
                device=device,
                compute_dtype=torch.float32,
                row_chunk_size=2,
                active_row_only_chunks=True,
                active_row_accumulation="direct_v1",
                row_batch_reader=(
                    store.iter_phase4_feature_rows
                    if store.resolved_influence_mode
                    in {"cuda_windowed", "cuda_file_windowed"}
                    else None
                ),
            )
            for store in stores
        ]

        assert torch.equal(results[0], results[1])
        assert torch.equal(results[0], results[2])
        for window_store in stores[1:]:
            window_stats = window_store.get_diagnostic_snapshot()
            assert window_stats["gpu_row_tier_h2d_bytes"] > 0
            assert window_stats["gpu_row_tier_window_read_calls"] > 1
            assert window_stats["gpu_row_tier_window_prefetch_calls"] > 1
            assert (
                window_stats["gpu_row_tier_window_stream_wait_count"]
                == window_stats["gpu_row_tier_window_prefetch_calls"]
            )
    finally:
        for store in stores:
            store.cleanup()
