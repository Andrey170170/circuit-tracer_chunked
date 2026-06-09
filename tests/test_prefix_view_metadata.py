from __future__ import annotations

import pytest
import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.attribute_nnsight import (
    _compact_nonfeature_column_counts,
    _compact_selected_feature_columns,
    _hash_token_ids,
    _resolve_prefix_view_output_position,
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


def test_validate_prefix_view_metadata_full_sequence_success() -> None:
    full_sequence = [10, 11, 12, 42, 43]
    normalized = validate_prefix_view_metadata(
        prompt=full_sequence,
        attribution_targets=torch.tensor([42], dtype=torch.long),
        prefix_view_metadata=_metadata(
            mode="full_sequence_target_position",
            output_position=2,
            input_token_count=len(full_sequence),
            full_sequence_token_count=len(full_sequence),
            input_token_ids_sha256=_hash_token_ids(full_sequence),
        ),
    )

    assert normalized is not None
    assert normalized["mode"] == "full_sequence_target_position"
    assert normalized["output_position"] == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_position": 3}, "output_position"),
        ({"input_token_count": 4}, "input_token_count"),
        ({"input_token_ids_sha256": "bad"}, "input_token_ids_sha256"),
    ],
)
def test_validate_prefix_view_metadata_full_sequence_rejects_mismatch(overrides, message) -> None:
    full_sequence = [10, 11, 12, 42, 43]
    metadata = _metadata(
        mode="full_sequence_target_position",
        output_position=2,
        input_token_count=len(full_sequence),
        input_token_ids_sha256=_hash_token_ids(full_sequence),
    )
    metadata.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_prefix_view_metadata(
            prompt=full_sequence,
            attribution_targets=torch.tensor([42], dtype=torch.long),
            prefix_view_metadata=metadata,
        )


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


def test_resolve_prefix_view_output_position_infers_full_sequence_target() -> None:
    metadata = _metadata(
        mode="full_sequence_target_position",
        input_token_count=5,
        full_sequence_token_count=5,
        input_token_ids_sha256=_hash_token_ids([10, 11, 12, 42, 43]),
    )

    normalized = validate_prefix_view_metadata(
        prompt=[10, 11, 12, 42, 43],
        attribution_targets=[42],
        prefix_view_metadata=metadata,
    )

    assert _resolve_prefix_view_output_position(normalized, None) == 2


def test_resolve_prefix_view_output_position_rejects_argument_mismatch() -> None:
    normalized = validate_prefix_view_metadata(
        prompt=[10, 11, 12, 42, 43],
        attribution_targets=[42],
        prefix_view_metadata=_metadata(
            mode="full_sequence_target_position",
            output_position=2,
            input_token_count=5,
            full_sequence_token_count=5,
            input_token_ids_sha256=_hash_token_ids([10, 11, 12, 42, 43]),
        ),
    )

    with pytest.raises(ValueError, match="output_position"):
        _resolve_prefix_view_output_position(normalized, 3)


def test_prefix_view_state_truncates_position_indexed_context() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 3, 2, 4],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        values=torch.ones(4),
        size=(2, 5, 8),
    ).coalesce()
    full_logits = torch.randn(1, 5, 8)
    ctx = AttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.randn(2, 5, 3),
        token_vectors=torch.randn(5, 3),
        decoder_vecs=torch.empty(0, 3),
        encoder_vecs=torch.empty(0, 3),
        encoder_to_decoder_map=torch.empty(0, dtype=torch.long),
        decoder_locations=torch.empty(2, 0, dtype=torch.long),
        logits=full_logits[:, -1:],
        full_logits=full_logits,
        stage_error_vectors_on_cpu=False,
    )

    stats = ctx.apply_prefix_view_state(3)

    assert stats["old_position_count"] == 5
    assert stats["new_position_count"] == 3
    assert stats["masked_active_feature_count"] == 2
    assert tuple(ctx.activation_matrix.shape) == (2, 3, 8)
    assert int(ctx.activation_matrix.coalesce().indices()[1].max()) == 2
    assert tuple(ctx.error_vectors.shape) == (2, 3, 3)
    assert tuple(ctx.token_vectors.shape) == (3, 3)
    assert ctx._row_size == int(ctx.activation_matrix._nnz()) + (2 + 1) * 3
    assert torch.equal(ctx.get_logits_at_position(2), full_logits[:, 2])


def test_prefix_view_state_compacts_feature_row_state() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 3, 2, 4],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        values=torch.ones(4),
        size=(2, 5, 8),
    ).coalesce()
    encoder_vecs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    decoder_vecs = torch.arange(100, 112, dtype=torch.float32).reshape(4, 3)
    ctx = AttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.randn(2, 5, 3),
        token_vectors=torch.randn(5, 3),
        decoder_vecs=decoder_vecs.clone(),
        encoder_vecs=encoder_vecs.clone(),
        encoder_to_decoder_map=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        decoder_locations=torch.tensor([[0, 0, 1, 1], [0, 3, 2, 4]], dtype=torch.long),
        logits=torch.randn(1, 1, 8),
        stage_error_vectors_on_cpu=False,
    )

    ctx.apply_prefix_view_state(3)

    assert torch.equal(ctx.encoder_vecs, encoder_vecs[[0, 2]])
    assert torch.equal(ctx.decoder_vecs, decoder_vecs[[0, 2]])
    assert torch.equal(ctx.encoder_to_decoder_map, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(
        ctx.decoder_locations,
        torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
    )


def test_compact_selected_feature_columns_drops_future_position_ordinals() -> None:
    selected = torch.tensor([0, 2, 4, 5, 8], dtype=torch.long)

    compact = _compact_selected_feature_columns(selected, n_feature_columns=5)

    assert torch.equal(compact, torch.tensor([0, 2, 4], dtype=torch.long))


def test_compact_nonfeature_columns_use_prefix_visible_positions() -> None:
    n_error, n_token, total = _compact_nonfeature_column_counts(n_layers=26, compact_token_count=73)

    assert n_error == 26 * 73
    assert n_token == 73
    assert total == 27 * 73
