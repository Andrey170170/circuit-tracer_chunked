from __future__ import annotations

import pytest
import torch

from circuit_tracer.attribution.attribute_nnsight import (
    _hash_token_ids,
    validate_prefix_view_metadata,
)


def _metadata(**overrides):
    payload = {
        "schema_version": 1,
        "mode": "independent_prefix",
        "trajectory_id": "traj_a",
        "trace_id": "trace_a_tok000000",
        "target_position": 3,
        "prefix_token_count": 3,
        "target_token_ids": [42],
        "prefix_token_ids_sha256": _hash_token_ids([10, 11, 12]),
    }
    payload.update(overrides)
    return payload


def test_validate_prefix_view_metadata_success_normalizes_payload() -> None:
    normalized = validate_prefix_view_metadata(
        prompt=torch.tensor([10, 11, 12], dtype=torch.long),
        attribution_targets=torch.tensor([42], dtype=torch.long),
        prefix_view_metadata=_metadata(),
    )

    assert normalized == _metadata()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prefix_token_count": 2}, "prefix_token_count"),
        ({"target_position": 4}, "target_position"),
        ({"target_token_ids": [99]}, "target_token_ids"),
        ({"prefix_token_ids_sha256": "bad"}, "prefix_token_ids_sha256"),
    ],
)
def test_validate_prefix_view_metadata_rejects_mismatches(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prefix_view_metadata(
            prompt=[10, 11, 12],
            attribution_targets=torch.tensor([42], dtype=torch.long),
            prefix_view_metadata=_metadata(**overrides),
        )


def test_validate_prefix_view_metadata_is_backward_compatible_without_metadata() -> None:
    assert (
        validate_prefix_view_metadata(
            prompt=[10, 11, 12],
            attribution_targets=torch.tensor([42], dtype=torch.long),
            prefix_view_metadata=None,
        )
        is None
    )
