from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.prefix_view import (
    _compact_nonfeature_column_counts,
    _compact_selected_feature_columns,
    _hash_token_ids,
    _resolve_prefix_view_output_position,
    _resolve_prefix_view_trace_input_ids,
    validate_compact_prefix_view_output,
    validate_prefix_view_metadata,
)
from circuit_tracer.replacement_model import (
    replacement_model_nnsight as replacement_model_nnsight_module,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel


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


def test_resolve_prefix_view_trace_input_ids_truncates_full_sequence() -> None:
    input_ids = torch.arange(8)
    metadata = _metadata(
        mode="full_sequence_target_position",
        target_position=3,
        prefix_token_count=3,
        input_token_count=8,
        full_sequence_token_count=8,
        input_token_ids_sha256=_hash_token_ids(input_ids.tolist()),
    )

    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(input_ids, metadata)

    assert prefix_view_length == 3
    assert trace_input_ids.tolist() == [0, 1, 2]
    assert trace_input_ids.is_contiguous()


def test_resolve_prefix_view_trace_input_ids_keeps_independent_prefix() -> None:
    input_ids = torch.arange(3)

    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(
        input_ids, _metadata()
    )

    assert prefix_view_length is None
    assert trace_input_ids is input_ids


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


def test_setup_attribution_prefix_view_traces_only_prefix(monkeypatch) -> None:
    class FakeTrace:
        def __init__(self, model, tokens: torch.Tensor) -> None:
            self.model = model
            self.tokens = tokens

        def __enter__(self):
            token_count = int(self.tokens.numel())
            self.model.traced_tokens = self.tokens.detach().clone()
            self.model.feature_input_locs = [
                SimpleNamespace(output=torch.ones(1, token_count, 3) * layer)
                for layer in range(self.model.cfg.n_layers)
            ]
            self.model.feature_output_locs = [
                SimpleNamespace(output=torch.ones(1, token_count, 3) * (layer + 1))
                for layer in range(self.model.cfg.n_layers)
            ]
            self.model.output = SimpleNamespace(logits=torch.randn(1, token_count, 20))
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTranscoders:
        def compute_attribution_components(self, mlp_in_cache, zero_positions, **kwargs):
            n_layers, n_pos, d_model = mlp_in_cache.shape
            return {
                "activation_matrix": torch.sparse_coo_tensor(
                    torch.empty((3, 0), dtype=torch.long),
                    torch.empty(0),
                    size=(n_layers, n_pos, 4),
                ).coalesce(),
                "reconstruction": torch.zeros_like(mlp_in_cache),
                "decoder_vecs": torch.empty(0, d_model),
                "encoder_vecs": torch.empty(0, d_model),
                "encoder_to_decoder_map": torch.empty(0, dtype=torch.long),
                "decoder_locations": torch.empty(2, 0, dtype=torch.long),
            }

    monkeypatch.setattr(replacement_model_nnsight_module, "save", lambda tensor: tensor)

    model = SimpleNamespace()
    model.cfg = SimpleNamespace(n_layers=2)
    model.device = torch.device("cpu")
    model.zero_positions = []
    model.embed_weight = torch.randn(20, 3)
    model.transcoders = FakeTranscoders()
    model.trace = lambda tokens: FakeTrace(model, tokens)

    ctx = NNSightReplacementModel.setup_attribution(
        model,  # type: ignore[arg-type]
        torch.arange(8),
        prefix_view_length=3,
        retain_full_logits=True,
    )

    assert model.traced_tokens.tolist() == [0, 1, 2]
    assert ctx.setup_diagnostic_stats["token_count"] == 8
    assert ctx.setup_diagnostic_stats["phase0_token_count"] == 3
    assert tuple(ctx.full_logits.shape) == (1, 3, 20)
    assert ctx.logit_source_shape == (1, 3, 20)
    assert tuple(ctx.token_vectors.shape) == (3, 3)
    assert torch.equal(ctx.get_logits_at_position(2), ctx.get_last_token_logits())


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


def _compact_prefix_payload(**overrides):
    payload = {
        "prefix_view_metadata": _metadata(mode="full_sequence_target_position"),
        "n_token_nodes": 3,
        "n_error_nodes": 2 * 3,
        "active_features": torch.tensor([[0, 0, 7], [1, 2, 9]], dtype=torch.long),
        "selected_features": torch.tensor([0, 1], dtype=torch.long),
        "feature_row_node_indices": torch.tensor([0, 5], dtype=torch.long),
        "logit_row_node_indices": torch.tensor([2], dtype=torch.long),
        "feature_error_edges": torch.zeros(2, 6),
        "logit_error_edges": torch.zeros(1, 6),
        "feature_token_edges": torch.zeros(2, 3),
        "logit_token_edges": torch.zeros(1, 3),
    }
    payload.update(overrides)
    return payload


def test_validate_compact_prefix_view_output_accepts_prefix_shapes() -> None:
    validate_compact_prefix_view_output(_compact_prefix_payload(), n_layers=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_token_nodes": 5}, "n_token_nodes"),
        ({"n_error_nodes": 10}, "n_error_nodes"),
        ({"feature_token_edges": torch.zeros(2, 5)}, "feature_token_edges"),
        ({"active_features": torch.tensor([[0, 3, 7]])}, "future positions"),
    ],
)
def test_validate_compact_prefix_view_output_rejects_mismatch(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_compact_prefix_view_output(_compact_prefix_payload(**overrides), n_layers=2)


class _FakeDecoderProvider:
    decoder_chunk_size = 16

    def __init__(self) -> None:
        self.created = 0
        self.cleared = 0

    def create_decoder_block_cache(self):
        self.created += 1
        return {"cache": self.created}

    def clear_decoder_block_cache(self, cache) -> None:
        if cache is not None:
            self.cleared += 1


def _context_with_cache(provider, cache=None, fingerprint=None):
    return AttributionContext(
        activation_matrix=torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0], [0]], dtype=torch.long),
            values=torch.ones(1),
            size=(1, 1, 4),
        ).coalesce(),
        error_vectors=torch.randn(1, 1, 2),
        token_vectors=torch.randn(1, 2),
        decoder_vecs=torch.randn(1, 2),
        encoder_vecs=torch.randn(1, 2),
        encoder_to_decoder_map=torch.tensor([0], dtype=torch.long),
        decoder_locations=torch.tensor([[0], [0]], dtype=torch.long),
        logits=torch.randn(1, 1, 8),
        decoder_provider=provider,
        chunked_decoder_state={
            "source_layers": torch.tensor([0]),
            "feature_ids": torch.tensor([0]),
        },
        stage_error_vectors_on_cpu=False,
        decoder_chunk_cache=cache,
        decoder_cache_fingerprint=fingerprint,
    )


def test_shared_decoder_cache_is_not_cleared_on_context_cleanup() -> None:
    provider = _FakeDecoderProvider()

    class Cache:
        fingerprint = ("model", 16)

    shared = Cache()

    ctx = _context_with_cache(provider, cache=shared, fingerprint=("model", 16))

    assert ctx.decoder_chunk_cache is shared
    assert provider.created == 0
    ctx.cleanup()
    assert provider.cleared == 0


def test_owned_decoder_cache_preserves_default_cleanup_behavior() -> None:
    provider = _FakeDecoderProvider()

    ctx = _context_with_cache(provider)

    assert provider.created == 1
    ctx.cleanup()
    assert provider.cleared == 1


def test_shared_decoder_cache_fingerprint_mismatch_rejected() -> None:
    class Cache:
        fingerprint = "actual"

    with pytest.raises(ValueError, match="fingerprint"):
        _context_with_cache(_FakeDecoderProvider(), cache=Cache(), fingerprint="expected")


def test_shared_decoder_cache_without_fingerprint_rejected_when_expected() -> None:
    class Cache:
        pass

    with pytest.raises(ValueError, match="missing fingerprint"):
        _context_with_cache(_FakeDecoderProvider(), cache=Cache(), fingerprint="expected")


def test_shared_decoder_cache_requires_expected_fingerprint() -> None:
    class Cache:
        fingerprint = "actual"

    with pytest.raises(ValueError, match="requires fingerprint"):
        _context_with_cache(_FakeDecoderProvider(), cache=Cache())
