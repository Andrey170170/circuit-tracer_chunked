from __future__ import annotations

import gc
import weakref

import pytest
import torch
from safetensors.torch import save_file

import circuit_tracer.transcoder.phase0_decoder_ranges as decoder_ranges_module
import circuit_tracer.transcoder.single_layer_transcoder as single_layer_module
from circuit_tracer.transcoder.phase0_decoder_ranges import (
    load_decoder_row_ranges,
    plan_decoder_row_ranges,
)
from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set


def _write_transcoder(path, decoder: torch.Tensor) -> None:
    d_transcoder, d_model = decoder.shape
    save_file(
        {
            "W_enc": torch.eye(d_transcoder, d_model),
            "W_dec": decoder,
            "b_enc": torch.zeros(d_transcoder),
            "b_dec": torch.arange(d_model, dtype=decoder.dtype).div(10),
        },
        str(path),
    )


def _provider(path, *, decoder_chunk_size: int = 4):
    return load_transcoder_set(
        {0: str(path)},
        "synthetic",
        "in",
        "out",
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=True,
        lazy_decoder=True,
        exact_chunked_provider=True,
        decoder_chunk_size=decoder_chunk_size,
    )


def _sparse(feature_ids: list[int], values: list[float]) -> torch.Tensor:
    positions = torch.arange(len(feature_ids), dtype=torch.long).remainder(3)
    return torch.sparse_coo_tensor(
        torch.stack((positions, torch.tensor(feature_ids))),
        torch.tensor(values),
        size=(3, 16),
        check_invariants=True,
    ).coalesce()


def test_row_range_loader_sorts_deduplicates_and_reads_coalesced_ranges(
    tmp_path, monkeypatch
) -> None:
    decoder = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    path = tmp_path / "layer_0.safetensors"
    _write_transcoder(path, decoder)
    plan = plan_decoder_row_ranges(
        torch.tensor([11, 2, 1, 3, 2, 10]),
        d_model=3,
        d_transcoder=16,
        itemsize=4,
        decoder_chunk_size=4,
        max_gap_rows=1,
        max_overfetch_fraction=0.5,
        max_range_count=8,
        max_singleton_range_fraction=0.5,
        max_ranges_per_baseline_page=4,
    )
    assert plan.admitted
    assert plan.unique_feature_ids.tolist() == [1, 2, 3, 10, 11]
    assert [(row_range.start, row_range.stop) for row_range in plan.ranges] == [
        (1, 4),
        (10, 12),
    ]

    real_safe_open = decoder_ranges_module.safe_open
    requested_slices: list[tuple[int | None, int | None]] = []

    class _SliceProxy:
        def __init__(self, safe_slice) -> None:
            self._safe_slice = safe_slice

        def get_shape(self):
            return self._safe_slice.get_shape()

        def __getitem__(self, row_slice):
            requested_slices.append((row_slice.start, row_slice.stop))
            return self._safe_slice[row_slice]

    class _CheckpointProxy:
        def __init__(self, checkpoint) -> None:
            self._checkpoint = checkpoint

        def __enter__(self):
            self._checkpoint.__enter__()
            return self

        def __exit__(self, *args):
            return self._checkpoint.__exit__(*args)

        def get_slice(self, key):
            return _SliceProxy(self._checkpoint.get_slice(key))

    def tracked_safe_open(*args, **kwargs):
        return _CheckpointProxy(real_safe_open(*args, **kwargs))

    monkeypatch.setattr(decoder_ranges_module, "safe_open", tracked_safe_open)
    rows, read_seconds, gather_seconds = load_decoder_row_ranges(
        path=str(path),
        key="W_dec",
        plan=plan,
        dtype=torch.float32,
    )

    assert torch.equal(rows, decoder[torch.tensor([1, 2, 3, 10, 11])])
    assert requested_slices == [(1, 4), (10, 12)]
    assert read_seconds >= 0
    assert gather_seconds >= 0


def test_range_planner_refuses_singleton_fragmentation() -> None:
    plan = plan_decoder_row_ranges(
        torch.tensor([0, 10, 20, 30, 40]),
        d_model=3,
        d_transcoder=64,
        itemsize=4,
        decoder_chunk_size=64,
        max_gap_rows=8,
        max_overfetch_fraction=1.0,
        max_range_count=4096,
        max_singleton_range_fraction=0.5,
        max_ranges_per_baseline_page=4,
    )

    assert not plan.admitted
    assert plan.fallback_reason == "singleton_range_fraction_exceeds_max"
    assert plan.baseline_full_page_count == 1


def test_range_planner_bounds_fragmentation_relative_to_full_pages() -> None:
    plan = plan_decoder_row_ranges(
        torch.tensor([0, 1, 11, 12, 22, 23, 33, 34, 44, 45]),
        d_model=3,
        d_transcoder=128,
        itemsize=4,
        decoder_chunk_size=128,
        max_gap_rows=8,
        max_overfetch_fraction=1.0,
        max_range_count=4096,
        max_singleton_range_fraction=0.5,
        max_ranges_per_baseline_page=4,
    )

    assert not plan.admitted
    assert plan.fallback_reason == "range_fragmentation_exceeds_baseline_ratio"
    assert plan.baseline_full_page_count == 1


def test_phase0_ranges_preserve_chunk_reconstruction_and_reuse_rows_for_seed(
    tmp_path, monkeypatch
) -> None:
    decoder = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3).div(13)
    path = tmp_path / "layer_0.safetensors"
    _write_transcoder(path, decoder)
    provider = _provider(path)
    sparse = _sparse([1, 2, 3, 13, 14, 15], [0.5, -1.0, 2.0, 0.25, -0.75, 1.5])

    baseline, baseline_seed, baseline_bytes = provider._decode_sparse_with_decoder_chunks(
        0,
        sparse,
        capture_decoder_row_seed=True,
    )
    monkeypatch.setattr(
        provider,
        "get_decoder_chunk",
        lambda *args, **kwargs: pytest.fail("selective path reloaded a full decoder page"),
    )
    got, seed, materialized_bytes, telemetry = (
        provider._decode_sparse_with_decoder_row_ranges(0, sparse)
    )

    assert torch.equal(got, baseline)
    assert baseline_seed is not None
    assert seed is not None
    assert torch.equal(seed.feature_ids, baseline_seed.feature_ids)
    assert torch.equal(seed.rows, baseline_seed.rows)
    assert telemetry.effective is True
    assert telemetry.fallback_reason is None
    assert telemetry.unique_row_count == int(seed.feature_ids.numel())
    assert telemetry.logical_requested_bytes == seed.rows.numel() * seed.rows.element_size()
    assert telemetry.logical_materialized_bytes == materialized_bytes
    assert telemetry.logical_materialized_bytes < baseline_bytes
    assert telemetry.range_request_count == len(telemetry.range_rows)


def test_phase0_ranges_fall_back_to_exact_full_pages_when_overfetch_is_excessive(
    tmp_path, monkeypatch
) -> None:
    decoder = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3).div(7)
    path = tmp_path / "layer_0.safetensors"
    _write_transcoder(path, decoder)
    provider = _provider(path)
    sparse = _sparse([0, 2], [0.75, -1.25])
    baseline, baseline_seed, baseline_bytes = provider._decode_sparse_with_decoder_chunks(
        0,
        sparse,
        capture_decoder_row_seed=True,
    )
    full_page_reads = 0
    original_get_decoder_chunk = provider.get_decoder_chunk

    def tracked_get_decoder_chunk(*args, **kwargs):
        nonlocal full_page_reads
        full_page_reads += 1
        return original_get_decoder_chunk(*args, **kwargs)

    monkeypatch.setattr(provider, "get_decoder_chunk", tracked_get_decoder_chunk)
    got, seed, materialized_bytes, telemetry = (
        provider._decode_sparse_with_decoder_row_ranges(0, sparse)
    )

    assert torch.equal(got, baseline)
    assert baseline_seed is not None
    assert seed is not None
    assert torch.equal(seed.rows, baseline_seed.rows)
    assert full_page_reads == 1
    assert materialized_bytes == baseline_bytes
    assert telemetry.effective is False
    assert telemetry.fallback_reason == "overfetch_fraction_exceeds_max"
    assert telemetry.range_request_count == 0
    assert telemetry.logical_materialized_bytes == baseline_bytes


def test_phase0_range_reconstruction_failure_releases_staged_rows(
    tmp_path, monkeypatch
) -> None:
    decoder = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    path = tmp_path / "layer_0.safetensors"
    _write_transcoder(path, decoder)
    provider = _provider(path)
    sparse = _sparse([1, 2, 3], [1.0, 2.0, 3.0])
    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def staged_rows(**kwargs):
        rows, read_seconds, gather_seconds = load_decoder_row_ranges(**kwargs)
        staged_refs.append(weakref.ref(rows))
        return rows, read_seconds, gather_seconds

    monkeypatch.setattr(single_layer_module, "load_decoder_row_ranges", staged_rows)
    transcoder = provider.transcoders[0]
    transcoder.W_skip = torch.nn.Parameter(torch.eye(3))

    def fail_skip(_input_acts):
        raise RuntimeError("synthetic reconstruction failure")

    monkeypatch.setattr(transcoder, "compute_skip", fail_skip)
    with pytest.raises(RuntimeError, match="synthetic reconstruction failure"):
        provider._decode_sparse_with_decoder_row_ranges(
            0,
            sparse,
            torch.zeros((3, 3)),
        )
    gc.collect()
    assert staged_refs and all(row_ref() is None for row_ref in staged_refs)
