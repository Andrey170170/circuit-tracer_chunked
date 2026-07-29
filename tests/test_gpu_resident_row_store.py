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
        assert store.admission.reason == "device_not_cuda"
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
        assert store.admission.reason == "allocation_failed:RuntimeError"
        assert torch.equal(store.read_feature_rows(0, 2, phase="phase4"), rows)
    finally:
        store.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_row_tier_exact_ranges_prepared_reads_and_cleanup() -> None:
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
        assert resident.device.type == "cpu"
        assert torch.equal(resident, first.cpu())

        fallback = store.read_feature_rows(0, 3, phase="phase4")
        assert fallback.device.type == "cpu"
        assert torch.equal(fallback[:2], first.cpu())

        _append(store, second, row_start=2)
        all_rows = store.read_feature_rows(0, 4, phase="phase4")
        expected = torch.cat((first, second), dim=0).cpu()
        assert torch.equal(all_rows, expected)

        prepared = store.read_prepared_feature_rows(
            1,
            4,
            device=device,
            dtype=torch.float32,
            phase="phase4",
        )
        assert torch.equal(prepared.cpu(), expected[1:4].abs())
        prepared_cpu = store.read_prepared_feature_rows(
            1,
            4,
            device="cpu",
            dtype=torch.float32,
            phase="phase4",
        )
        assert prepared_cpu.storage_offset() == 0
        assert torch.equal(prepared_cpu, expected[1:4].abs())
        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_read_hits"] == 4
        assert stats["gpu_row_tier_read_fallbacks"] == 1
        assert stats["gpu_row_tier_avoided_file_read_bytes"] == (2 + 4 + 3 + 3) * 4 * 4
        assert stats["gpu_row_tier_d2h_bytes"] == (2 + 4) * 4 * 4
        assert stats["gpu_row_tier_avoided_h2d_bytes"] == 3 * 4 * 4
        assert stats["gpu_row_tier_owned_bytes"] == 5 * 4 * 4
        assert stats["gpu_row_tier_prepared_host_mirror_owned_bytes"] == 5 * 4 * 4
        assert stats["gpu_row_tier_prepared_host_mirror_read_bytes"] == 3 * 4 * 4
        del resident, fallback, all_rows, expected, prepared, prepared_cpu
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
            lambda start, end: store.read_prepared_feature_rows(
                start,
                end,
                device=store.influence_device,
                dtype=torch.float32,
                phase="phase4",
            ),
            denominators,
            logit_p,
            row_to_node_index,
            n_feature_nodes=4,
            n_logits=1,
            device=store.influence_device,
            compute_dtype=torch.float32,
            active_row_only_chunks=True,
            row_reader_returns_prepared=True,
            active_row_accumulation="direct_v1",
        )

        assert candidate.device.type == "cpu"
        assert torch.equal(candidate, reference)
        stats = store.get_diagnostic_snapshot()
        assert stats["gpu_row_tier_prepared_host_mirror_read_bytes"] > 0
    finally:
        store.cleanup()
