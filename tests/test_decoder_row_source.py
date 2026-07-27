from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.decoder_row_source import (
    DecoderRowKey,
    DecoderRowOrder,
    DecoderRowRefusal,
    DecoderRowRefusalCode,
    DecoderTensorSpec,
    MappedSafetensorsDecoderRowSource,
)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_same_layer_rows_preserve_caller_order_and_raw_values(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    path = tmp_path / "plt.safetensors"
    decoder = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(dtype)
    save_file({"w_dec": decoder}, path)
    events: list[object] = []
    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(2, path, "w_dec")],
        max_staging_bytes=decoder[0].numel() * decoder.element_size() * 2,
        telemetry=events.append,
    )
    keys = [
        DecoderRowKey(2, 4),
        DecoderRowKey(2, 1),
        DecoderRowKey(2, 4),
        DecoderRowKey(2, 0),
    ]

    seed = source.materialize(keys)

    assert not isinstance(seed, DecoderRowRefusal)
    expected = decoder[torch.tensor([4, 1, 4, 0])]
    assert torch.equal(_bytes(seed.rows), _bytes(expected))
    assert seed.keys == tuple(keys)
    assert seed.telemetry.occurrence_row_count == 4
    assert seed.telemetry.unique_row_count == 3
    assert seed.telemetry.backend_materialized_bytes == 3 * decoder[0].nbytes
    assert seed.telemetry.requested_row_bytes == 3 * decoder[0].nbytes
    assert seed.telemetry.occurrence_row_bytes == 4 * decoder[0].nbytes
    assert seed.telemetry.backend_request_count == 2
    assert seed.telemetry.mapping_count == 1
    assert seed.telemetry.range_count == 0
    assert seed.telemetry.read_count == 0
    assert seed.telemetry.output_bytes == 4 * decoder[0].nbytes
    assert seed.telemetry.temporary_staging_high_water_bytes == 2 * decoder[0].nbytes
    assert events[-1] == seed.telemetry


def test_cross_layer_output_slots_are_deduplicated_sorted_and_reordered(tmp_path: Path) -> None:
    first_path = tmp_path / "clt_0.safetensors"
    second_path = tmp_path / "clt_1.safetensors"
    first = torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4)
    second = (torch.arange(4 * 2 * 4, dtype=torch.float32) + 100).reshape(4, 2, 4)
    save_file({"W_dec_0": first}, first_path)
    save_file({"W_dec_1": second}, second_path)
    source = MappedSafetensorsDecoderRowSource(
        [
            DecoderTensorSpec(0, first_path, "W_dec_0"),
            DecoderTensorSpec(1, second_path, "W_dec_1"),
        ],
        max_staging_bytes=32,
    )
    keys = [
        DecoderRowKey(1, 3, 0),
        DecoderRowKey(0, 2, 2),
        DecoderRowKey(0, 0, 1),
        DecoderRowKey(1, 3, 0),
        DecoderRowKey(0, 2, 0),
    ]

    seed = source.materialize(keys)
    unique_seed = source.materialize(keys, order=DecoderRowOrder.SORTED_UNIQUE)

    assert not isinstance(seed, DecoderRowRefusal)
    assert not isinstance(unique_seed, DecoderRowRefusal)
    expected = torch.stack([second[3, 0], first[2, 2], first[0, 1], second[3, 0], first[2, 0]])
    assert torch.equal(_bytes(seed.rows), _bytes(expected))
    assert unique_seed.keys == tuple(sorted(set(keys)))
    unique_expected = torch.stack(
        [
            first[0, 1],
            first[2, 0],
            first[2, 2],
            second[3, 0],
        ]
    )
    assert torch.equal(_bytes(unique_seed.rows), _bytes(unique_expected))


def test_planner_refuses_per_row_gathers_and_invalid_keys(tmp_path: Path) -> None:
    path = tmp_path / "decoder.safetensors"
    decoder = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"w_dec": decoder}, path)
    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec")],
        max_staging_bytes=decoder[0].nbytes,
    )

    per_row = source.estimate([DecoderRowKey(0, 0), DecoderRowKey(0, 1)])
    invalid = source.estimate([DecoderRowKey(0, 3)])

    assert per_row.refusal is not None
    assert per_row.refusal.code is DecoderRowRefusalCode.PER_ROW_REQUESTS
    assert invalid.refusal is not None
    assert invalid.refusal.code is DecoderRowRefusalCode.INVALID_KEY


def test_chunk_staging_budget_does_not_scale_with_retained_output(tmp_path: Path) -> None:
    path = tmp_path / "decoder.safetensors"
    decoder = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 4)
    save_file({"w_dec": decoder}, path)
    row_bytes = decoder[0].nbytes
    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec")],
        max_staging_bytes=2 * row_bytes,
    )
    keys = [DecoderRowKey(0, feature_id) for feature_id in range(39, -1, -1)]

    seed = source.materialize(keys)

    assert not isinstance(seed, DecoderRowRefusal)
    assert seed.telemetry.backend_request_count == 20
    assert seed.telemetry.output_bytes == 40 * row_bytes
    assert seed.telemetry.temporary_staging_high_water_bytes == 2 * row_bytes
    assert torch.equal(_bytes(seed.rows), _bytes(decoder.flip(0)))
    source.release("test_complete")


def test_source_identity_change_refuses_materialization(tmp_path: Path) -> None:
    path = tmp_path / "decoder.safetensors"
    decoder = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"w_dec": decoder}, path)
    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec")],
        max_staging_bytes=32,
    )
    path.unlink()
    save_file({"w_dec": decoder + 1}, path)

    result = source.materialize([DecoderRowKey(0, 0)])

    assert isinstance(result, DecoderRowRefusal)
    assert result.code is DecoderRowRefusalCode.SOURCE_MISMATCH


def test_expected_source_fingerprint_and_release_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "decoder.safetensors"
    decoder = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"w_dec": decoder}, path)
    initial = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec")],
        max_staging_bytes=32,
    )
    tensor_fingerprint = initial.fingerprint.tensors[0].fingerprint
    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec", tensor_fingerprint)],
        max_staging_bytes=32,
    )
    seed = source.materialize([DecoderRowKey(0, 1)])
    reused = source.materialize([DecoderRowKey(0, 2)])
    assert not isinstance(seed, DecoderRowRefusal)
    assert not isinstance(reused, DecoderRowRefusal)
    assert seed.telemetry.mapping_open_count == 1
    assert reused.telemetry.mapping_open_count == 0
    expected_after_close = seed.rows.clone()

    released = source.release("phase0_complete")
    repeated = source.release("duplicate_cleanup")
    after_release = source.materialize([DecoderRowKey(0, 1)])

    assert released.outcome == "released"
    assert released.mapping_count == 1
    assert released.handle_count == 1
    assert released.mapped_address_span_bytes == path.stat().st_size
    assert torch.equal(_bytes(seed.rows), _bytes(expected_after_close))
    assert repeated.outcome == "already_released"
    assert isinstance(after_release, DecoderRowRefusal)
    assert after_release.code is DecoderRowRefusalCode.RELEASED

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        MappedSafetensorsDecoderRowSource(
            [DecoderTensorSpec(0, path, "w_dec", "wrong")],
            max_staging_bytes=32,
        )


def test_telemetry_callback_failure_does_not_change_materialization_or_release(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decoder.safetensors"
    decoder = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"w_dec": decoder}, path)

    def fail_telemetry(_: object) -> None:
        raise RuntimeError("sink unavailable")

    source = MappedSafetensorsDecoderRowSource(
        [DecoderTensorSpec(0, path, "w_dec")],
        max_staging_bytes=32,
        telemetry=fail_telemetry,
    )

    seed = source.materialize([DecoderRowKey(0, 1)])
    released = source.release("complete")

    assert not isinstance(seed, DecoderRowRefusal)
    assert torch.equal(_bytes(seed.rows), _bytes(decoder[1].unsqueeze(0)))
    assert released.outcome == "released"
