from __future__ import annotations

import gc
import weakref

import pytest
import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTape,
    FeatureVjpTapeEntry,
)
from circuit_tracer.tracing.plan import FrontierExpansionPlan


def _entry(index: int, total_nbytes: int) -> FeatureVjpTapeEntry:
    host_nbytes = total_nbytes // 3
    device_nbytes = total_nbytes // 3
    row_nbytes = total_nbytes - host_nbytes - device_nbytes
    return FeatureVjpTapeEntry(
        batch_call_index=index,
        gradients=(torch.ones(1),),
        row_buffer=torch.ones(1),
        batch_size=1,
        host_nbytes=host_nbytes,
        device_nbytes=device_nbytes,
        row_nbytes=row_nbytes,
        total_nbytes=total_nbytes,
        pinned_host_nbytes=0,
        pageable_host_nbytes=host_nbytes,
    )


def test_tape_enforces_batch_and_byte_caps_and_clears_deterministically() -> None:
    tape = FeatureVjpTape(max_batches=2, max_bytes=10)
    first = _entry(1, 6)
    tape.append(first)

    assert not tape.can_accept(5)
    with pytest.raises(BufferError):
        tape.append(_entry(2, 5))
    assert tape.high_watermark_bytes == 6

    tape.clear()
    assert tape.entries == ()
    assert tape.current_bytes == 0
    assert first.gradients == ()
    assert first.row_buffer.numel() == 0


def test_tape_window_one_is_unbounded_streaming_fallback() -> None:
    tape = FeatureVjpTape(max_batches=1, max_bytes=0)
    tape.append(_entry(1, 1_000_000))
    assert tape.batch_count == 1
    assert not tape.can_accept(1)


def test_frontier_tape_controls_are_validated_as_physical_policy() -> None:
    assert FrontierExpansionPlan().feature_vjp_tape_batch_window == 1
    with pytest.raises(ValueError, match="requires feature_vjp_tape_max_bytes"):
        FrontierExpansionPlan(feature_vjp_tape_batch_window=2)
    configured = FrontierExpansionPlan(
        feature_vjp_tape_batch_window=2,
        feature_vjp_tape_max_bytes=1024,
    )
    assert configured.feature_vjp_tape_max_bytes == 1024


def test_tape_estimate_accounts_bf16_host_to_fp32_device_expansion() -> None:
    ctx = object.__new__(AttributionContext)
    ctx._feature_output_activations = [
        torch.zeros((4, 3, 5), dtype=torch.bfloat16),
        torch.zeros((4, 3, 5), dtype=torch.bfloat16),
        torch.zeros((4, 3, 5), dtype=torch.bfloat16),
    ]
    ctx._row_size = 7

    estimate = ctx.estimate_feature_vjp_tape_entry_nbytes(
        layers=torch.tensor([2]),
        batch_size=2,
    )

    gradient_numel = 2 * 2 * 3 * 5
    assert estimate.host_nbytes == gradient_numel * torch.bfloat16.itemsize
    assert estimate.device_nbytes == gradient_numel * torch.float32.itemsize
    assert estimate.row_nbytes == 2 * 7 * torch.float32.itemsize
    assert estimate.total_nbytes == (
        estimate.host_nbytes + estimate.device_nbytes + estimate.row_nbytes
    )


def test_tape_clear_releases_gradient_and_row_tensor_references() -> None:
    gradient = torch.ones(4)
    row_buffer = torch.ones(5)
    gradient_ref = weakref.ref(gradient)
    row_ref = weakref.ref(row_buffer)
    entry = FeatureVjpTapeEntry(
        batch_call_index=1,
        gradients=(gradient,),
        row_buffer=row_buffer,
        batch_size=1,
        host_nbytes=16,
        device_nbytes=16,
        row_nbytes=20,
        total_nbytes=52,
        pinned_host_nbytes=0,
        pageable_host_nbytes=16,
    )
    tape = FeatureVjpTape(max_batches=2, max_bytes=100)
    tape.append(entry)
    del gradient, row_buffer

    tape.clear()
    gc.collect()

    assert gradient_ref() is None
    assert row_ref() is None
