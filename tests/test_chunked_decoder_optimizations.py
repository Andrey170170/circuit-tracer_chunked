from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.attribution.attribute import attribute as attribute_top_level
from circuit_tracer.attribution.attribute_nnsight import (
    _build_phase4_probe_pending_frontier,
    _compute_phase4_planned_feature_batch_size,
    _reorder_pending_for_phase4_locality,
)
from circuit_tracer.attribution.context_nnsight import (
    AttributionContext as NNSightAttributionContext,
)
from circuit_tracer.transcoder.cross_layer_transcoder import (
    DecoderChunkCache,
    load_clt,
    load_gemma_scope_2_clt,
)


class FakeDecoderProvider:
    def __init__(
        self,
        blocks: dict[int, torch.Tensor],
        chunk_size: int = 1,
        *,
        enable_cache: bool = True,
    ) -> None:
        self.blocks = blocks
        self.decoder_chunk_size = chunk_size
        self.enable_cache = enable_cache
        self.load_calls: list[tuple[int, int]] = []
        self.clear_calls = 0

    def create_decoder_block_cache(self):
        return {} if self.enable_cache else None

    def clear_decoder_block_cache(self, cache) -> None:
        self.clear_calls += 1
        if cache is not None:
            cache.clear()

    def get_decoder_chunk(self, layer_id: int, chunk_id: int, decoder_cache=None) -> torch.Tensor:
        cache_key = (layer_id, chunk_id)
        if decoder_cache is not None and cache_key in decoder_cache:
            return decoder_cache[cache_key]

        start = chunk_id * self.decoder_chunk_size
        stop = min(start + self.decoder_chunk_size, self.blocks[layer_id].shape[0])
        self.load_calls.append((layer_id, chunk_id))
        result = self.blocks[layer_id][start:stop]
        if decoder_cache is not None:
            decoder_cache[cache_key] = result
        return result


class GuardrailDecoderProvider:
    def __init__(
        self,
        blocks: dict[int, torch.Tensor],
        *,
        chunk_size: int = 1,
        cache_max_bytes: int = 16,
    ) -> None:
        self.blocks = blocks
        self.decoder_chunk_size = chunk_size
        self.cache_max_bytes = cache_max_bytes
        self.auto_disable_reasons: list[str] = []
        self.stats = {
            "decoder_cache_hit_count": 0,
            "decoder_cache_miss_count": 0,
            "decoder_cache_eviction_count": 0,
            "decoder_cache_skip_count": 0,
            "decoder_cache_auto_disable_count": 0,
            "decoder_cache_bytes_resident": 0,
            "decoder_cache_max_bytes": cache_max_bytes,
        }

    def create_decoder_block_cache(self):
        return DecoderChunkCache(self.cache_max_bytes)

    def clear_decoder_block_cache(self, cache) -> None:
        if cache is not None:
            cache.clear()
        self.stats["decoder_cache_bytes_resident"] = 0

    def get_diagnostic_snapshot(self):
        return dict(self.stats)

    def note_decoder_cache_auto_disabled(self, reason: str) -> None:
        self.auto_disable_reasons.append(reason)
        self.stats["decoder_cache_auto_disable_count"] += 1
        self.stats["decoder_cache_bytes_resident"] = 0

    def get_decoder_chunk(self, layer_id: int, chunk_id: int, decoder_cache=None) -> torch.Tensor:
        cache_key = (layer_id, chunk_id)
        if decoder_cache is not None:
            cached = decoder_cache.get(cache_key)
            if cached is not None:
                self.stats["decoder_cache_hit_count"] += 1
                self.stats["decoder_cache_bytes_resident"] = decoder_cache.bytes_resident
                return cached

        self.stats["decoder_cache_miss_count"] += 1
        start = chunk_id * self.decoder_chunk_size
        stop = min(start + self.decoder_chunk_size, self.blocks[layer_id].shape[0])
        result = self.blocks[layer_id][start:stop]
        if decoder_cache is not None:
            evicted = decoder_cache.put(cache_key, result)
            self.stats["decoder_cache_eviction_count"] += len(evicted)
            self.stats["decoder_cache_bytes_resident"] = decoder_cache.bytes_resident
        return result


def _make_chunked_context(context_cls, *, enable_cache: bool = True):
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        values=torch.tensor([2.0, 3.0, 5.0]),
        size=(3, 2, 2),
        check_invariants=True,
    ).coalesce()
    provider = FakeDecoderProvider(
        {
            0: torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
                ]
            ),
            1: torch.tensor([[[1.0, -1.0], [1.0, 2.0]]]),
        },
        chunk_size=1,
        enable_cache=enable_cache,
    )
    ctx = context_cls(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(3, 2, 2),
        token_vectors=torch.zeros(2, 2),
        decoder_vecs=torch.empty((0, 2)),
        encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1),
        decoder_provider=provider,
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
    )
    ctx._batch_buffer = torch.zeros(ctx._row_size, 2)
    return ctx, provider


def _assert_chunked_attr_helper(context_cls) -> None:
    grads_by_output_layer = [
        torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ]
        ),
        torch.tensor(
            [
                [[5.0, 50.0], [6.0, 60.0]],
                [[7.0, 70.0], [8.0, 80.0]],
            ]
        ),
        None,
    ]
    expected_feature_rows = torch.tensor(
        [
            [102.0, 146.0],
            [372.0, 504.0],
            [-270.0, -360.0],
        ]
    )
    expected_loads = [(0, 0), (0, 1), (1, 0)]

    ctx, provider = _make_chunked_context(context_cls, enable_cache=True)
    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    assert ctx._chunked_layer_spans == [(0, 2), (2, 3), None]
    assert torch.allclose(ctx._batch_buffer[:3], expected_feature_rows)
    assert torch.count_nonzero(ctx._batch_buffer[3:]) == 0
    assert provider.load_calls == expected_loads

    cached_result = ctx._batch_buffer.clone()
    ctx._batch_buffer.zero_()
    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    assert torch.allclose(ctx._batch_buffer[:3], expected_feature_rows)
    assert provider.load_calls == expected_loads

    ctx.clear_decoder_cache()
    ctx._batch_buffer.zero_()
    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    assert torch.allclose(ctx._batch_buffer, cached_result)
    assert provider.load_calls == expected_loads + expected_loads

    ctx.reset_decoder_cache()
    ctx._batch_buffer.zero_()
    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    assert torch.allclose(ctx._batch_buffer, cached_result)
    assert provider.load_calls == expected_loads + expected_loads + expected_loads
    ctx._batch_buffer.zero_()
    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    assert torch.allclose(ctx._batch_buffer, cached_result)
    assert provider.load_calls == expected_loads + expected_loads + expected_loads

    uncached_ctx, uncached_provider = _make_chunked_context(context_cls, enable_cache=False)
    uncached_ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    assert torch.allclose(uncached_ctx._batch_buffer, cached_result)
    assert uncached_provider.load_calls == expected_loads


def test_nnsight_chunked_attr_reuses_decoder_block_loads() -> None:
    _assert_chunked_attr_helper(NNSightAttributionContext)


def test_nnsight_chunked_attr_requires_sorted_source_layers() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1], [0, 0]]),
        values=torch.tensor([1.0, 2.0]),
        size=(2, 2, 1),
    ).coalesce()

    with pytest.raises(ValueError, match="sorted by layer"):
        NNSightAttributionContext(
            activation_matrix=activation_matrix,
            error_vectors=torch.zeros(2, 2, 2),
            token_vectors=torch.zeros(2, 2),
            decoder_vecs=torch.empty((0, 2)),
            encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
            encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
            decoder_locations=torch.empty((2, 0), dtype=torch.long),
            logits=torch.zeros(1),
            decoder_provider=FakeDecoderProvider(
                {0: torch.zeros(1, 2, 2), 1: torch.zeros(1, 1, 2)}
            ),
            chunked_decoder_state={
                "source_layers": torch.tensor([1, 0]),
                "positions": torch.tensor([1, 0]),
                "feature_ids": torch.tensor([0, 0]),
                "activation_values": torch.tensor([2.0, 1.0]),
            },
        )


def test_transformerlens_chunked_attr_reuses_decoder_block_loads() -> None:
    pytest.importorskip("transformer_lens")
    from circuit_tracer.attribution.context_transformerlens import (
        AttributionContext as TransformerLensAttributionContext,
    )

    _assert_chunked_attr_helper(TransformerLensAttributionContext)


def _assert_chunked_attr_subchunks_large_decoder_bucket(context_cls) -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 1, 0, 1, 0]]),
        values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=(2, 5, 2),
        check_invariants=True,
    ).coalesce()
    provider = FakeDecoderProvider(
        {
            0: torch.tensor(
                [
                    [[1.0, 0.0], [10.0, 1.0]],
                    [[0.0, 1.0], [1.0, 10.0]],
                ]
            ),
        },
        chunk_size=2,
        enable_cache=True,
    )
    ctx = context_cls(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 5, 2),
        token_vectors=torch.zeros(5, 2),
        decoder_vecs=torch.empty((0, 2)),
        encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1),
        decoder_provider=provider,
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
    )
    ctx._batch_buffer = torch.zeros(ctx._row_size, 2)
    grads_by_output_layer = [
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0]],
            ]
        ),
        torch.tensor(
            [
                [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
                [[12.0, 11.0], [14.0, 13.0], [16.0, 15.0], [18.0, 17.0], [20.0, 19.0]],
            ]
        ),
    ]

    expected = torch.zeros(activation_matrix._nnz(), 2)
    positions = activation_matrix.indices()[1]
    feature_ids = activation_matrix.indices()[2]
    activations = activation_matrix.values()
    decoder_block = provider.blocks[0]
    for row_idx in range(activation_matrix._nnz()):
        position = int(positions[row_idx].item())
        feature_id = int(feature_ids[row_idx].item())
        activation = activations[row_idx]
        total = torch.zeros(2)
        for output_layer, grads in enumerate(grads_by_output_layer):
            decoder_vec = decoder_block[feature_id, output_layer]
            total += torch.einsum("bd,d->b", grads[:, position], decoder_vec) * activation
        expected[row_idx] = total

    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    assert torch.allclose(ctx._batch_buffer[: activation_matrix._nnz()], expected)
    assert provider.load_calls == [(0, 0)]


def test_nnsight_chunked_attr_subchunks_large_decoder_bucket() -> None:
    _assert_chunked_attr_subchunks_large_decoder_bucket(NNSightAttributionContext)


def test_transformerlens_chunked_attr_subchunks_large_decoder_bucket() -> None:
    pytest.importorskip("transformer_lens")
    from circuit_tracer.attribution.context_transformerlens import (
        AttributionContext as TransformerLensAttributionContext,
    )

    _assert_chunked_attr_subchunks_large_decoder_bucket(TransformerLensAttributionContext)


def test_nnsight_row_subchunk_override_matches_default_replay() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 1, 0, 1, 0]]),
        values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        size=(2, 5, 2),
        check_invariants=True,
    ).coalesce()
    blocks = {
        0: torch.tensor(
            [
                [[1.0, 0.0], [10.0, 1.0]],
                [[0.0, 1.0], [1.0, 10.0]],
            ]
        ),
    }
    grads_by_output_layer = [
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0]],
            ]
        ),
        torch.tensor(
            [
                [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
                [[12.0, 11.0], [14.0, 13.0], [16.0, 15.0], [18.0, 17.0], [20.0, 19.0]],
            ]
        ),
    ]

    def _make_ctx(*, row_subchunk_size: int | None) -> NNSightAttributionContext:
        provider = FakeDecoderProvider(blocks=blocks, chunk_size=2, enable_cache=True)
        ctx = NNSightAttributionContext(
            activation_matrix=activation_matrix,
            error_vectors=torch.zeros(2, 5, 2),
            token_vectors=torch.zeros(5, 2),
            decoder_vecs=torch.empty((0, 2)),
            encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
            encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
            decoder_locations=torch.empty((2, 0), dtype=torch.long),
            logits=torch.zeros(1),
            decoder_provider=provider,
            chunked_decoder_state={
                "source_layers": activation_matrix.indices()[0],
                "positions": activation_matrix.indices()[1],
                "feature_ids": activation_matrix.indices()[2],
                "activation_values": activation_matrix.values(),
            },
            row_subchunk_size=row_subchunk_size,
        )
        ctx._batch_buffer = torch.zeros(ctx._row_size, 2)
        return ctx

    baseline_ctx = _make_ctx(row_subchunk_size=None)
    baseline_ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    custom_ctx = _make_ctx(row_subchunk_size=1)
    custom_ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    assert torch.allclose(custom_ctx._batch_buffer, baseline_ctx._batch_buffer)
    assert custom_ctx.get_diagnostic_snapshot()["row_subchunk_size"] == 1.0


def _create_gemmascope2_clt_files(
    tmp_path: Path,
    n_layers: int = 3,
    d_model: int = 8,
    d_transcoder: int = 16,
) -> dict[int, str]:
    paths: dict[int, str] = {}
    for layer_idx in range(n_layers):
        layer_path = tmp_path / f"layer_{layer_idx}.safetensors"
        save_file(
            {
                "w_enc": torch.randn(d_model, d_transcoder),
                "b_enc": torch.randn(d_transcoder),
                "b_dec": torch.randn(d_model),
                "threshold": torch.randn(d_transcoder).abs(),
                "w_dec": torch.randn(d_transcoder, n_layers, d_model),
            },
            str(layer_path),
        )
        paths[layer_idx] = str(layer_path)
    return paths


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_reconstruction_matches_saved_components_with_small_chunks(
    tmp_path: Path, chunk_size: int
) -> None:
    torch.manual_seed(0)
    layer_paths = _create_gemmascope2_clt_files(tmp_path)
    clt = load_gemma_scope_2_clt(
        layer_paths,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )
    eager_chunked_clt = load_gemma_scope_2_clt(
        layer_paths,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )
    standard_dir = tmp_path / f"standard_clt_{chunk_size}"
    standard_dir.mkdir()
    eager_chunked_clt.to_safetensors(str(standard_dir))
    standard_clt = load_clt(
        str(standard_dir),
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
        exact_chunked_decoder=False,
    )
    inputs = torch.randn(clt.n_layers, 4, clt.d_model, dtype=clt.dtype)

    components = clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))
    baseline = standard_clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))
    clt.reset_diagnostic_stats()
    reconstructed = clt.compute_reconstruction_chunked(
        components["activation_matrix"],
        inputs,
        chunk_size=chunk_size,
    )

    diagnostics = clt.get_diagnostic_snapshot()
    assert torch.allclose(reconstructed, baseline["reconstruction"])
    assert diagnostics["decoder_load_count"] == diagnostics["reconstruction_chunk_count"]


def test_decoder_chunk_cache_is_bounded_and_observable(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layer_paths = _create_gemmascope2_clt_files(tmp_path, n_layers=3, d_model=4, d_transcoder=8)
    clt = load_gemma_scope_2_clt(
        layer_paths,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
        decoder_chunk_size=2,
        cross_batch_decoder_cache_bytes=64,
    )

    cache = clt.create_decoder_block_cache()
    assert cache is not None

    first = clt.get_decoder_chunk(0, 0, decoder_cache=cache)
    clt.get_decoder_chunk(0, 0, decoder_cache=cache)
    clt.get_decoder_chunk(0, 1, decoder_cache=cache)
    clt.get_decoder_chunk(0, 2, decoder_cache=cache)

    stats = clt.get_diagnostic_snapshot()
    assert first.shape == (2, 3, 4)
    assert stats["decoder_cache_hit_count"] == 1
    assert stats["decoder_cache_miss_count"] == 3
    assert cast(int, stats["decoder_cache_eviction_count"]) >= 1
    assert cast(int, stats["decoder_cache_bytes_resident"]) <= 64
    assert stats["decoder_cache_max_bytes"] == 64

    clt.clear_decoder_block_cache(cache)
    cleared_stats = clt.get_diagnostic_snapshot()
    assert cleared_stats["decoder_cache_bytes_resident"] == 0


def test_exact_chunked_encoder_vectors_are_cpu_staged_and_materialized_equivalently() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        values=torch.tensor([1.0, 2.0, 3.0]),
        size=(2, 2, 2),
        check_invariants=True,
    ).coalesce()
    encoder_vecs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 2, 4),
        token_vectors=torch.zeros(2, 4),
        decoder_vecs=torch.empty((0, 4)),
        encoder_vecs=encoder_vecs.clone(),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1, 1, 5),
        decoder_provider=FakeDecoderProvider({0: torch.zeros(2, 2, 4), 1: torch.zeros(1, 1, 4)}),
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
    )

    assert ctx.encoder_vecs.device.type == "cpu"
    batch = ctx.materialize_encoder_vectors(torch.tensor([2, 0]), device=torch.device("cpu"))
    assert torch.equal(batch, encoder_vecs[torch.tensor([2, 0])])


def test_exact_chunked_lazy_encoder_materialization_matches_eager_rows(tmp_path: Path) -> None:
    torch.manual_seed(0)
    clt = load_gemma_scope_2_clt(
        _create_gemmascope2_clt_files(tmp_path),
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )
    inputs = torch.randn(clt.n_layers, 4, clt.d_model, dtype=clt.dtype)

    eager_components = clt.compute_attribution_components(
        inputs,
        zero_positions=slice(0, 1),
        materialize_encoder_vecs=True,
    )
    lazy_components = clt.compute_attribution_components(
        inputs,
        zero_positions=slice(0, 1),
        materialize_encoder_vecs=False,
    )

    eager_activation = cast(torch.Tensor, eager_components["activation_matrix"])
    lazy_activation = cast(torch.Tensor, lazy_components["activation_matrix"])
    eager_encoder_vecs = cast(torch.Tensor, eager_components["encoder_vecs"])
    lazy_encoder_vecs = cast(torch.Tensor, lazy_components["encoder_vecs"])

    assert torch.equal(lazy_activation.indices(), eager_activation.indices())
    assert torch.allclose(lazy_activation.values(), eager_activation.values())
    assert lazy_encoder_vecs.shape == (0, clt.d_model)
    assert eager_activation._nnz() > 0

    ctx = NNSightAttributionContext(
        activation_matrix=lazy_activation,
        error_vectors=torch.zeros(clt.n_layers, inputs.shape[1], clt.d_model, dtype=clt.dtype),
        token_vectors=torch.zeros(inputs.shape[1], clt.d_model, dtype=clt.dtype),
        decoder_vecs=cast(torch.Tensor, lazy_components["decoder_vecs"]),
        encoder_vecs=lazy_encoder_vecs,
        encoder_to_decoder_map=cast(torch.Tensor, lazy_components["encoder_to_decoder_map"]),
        decoder_locations=cast(torch.Tensor, lazy_components["decoder_locations"]),
        logits=torch.zeros(1, 1, 1, dtype=clt.dtype),
        decoder_provider=clt,
        chunked_decoder_state=cast(
            dict[str, torch.Tensor], lazy_components["chunked_decoder_state"]
        ),
    )

    nnz = eager_activation._nnz()
    row_probe = torch.randperm(nnz)[: min(5, nnz)]
    lazy_rows = ctx.materialize_encoder_vectors(row_probe, device=torch.device("cpu"))
    assert torch.allclose(lazy_rows, eager_encoder_vecs[row_probe])

    cap = max(1, nnz // 2)
    selected = (
        torch.topk(eager_activation.values().abs(), k=cap, sorted=False).indices.sort().values
    )
    before_cap, after_cap = ctx.apply_diagnostic_feature_cap(cap)
    assert before_cap == nnz
    assert after_cap == cap
    capped_rows = ctx.materialize_encoder_vectors(torch.arange(cap), device=torch.device("cpu"))
    assert torch.allclose(capped_rows, eager_encoder_vecs[selected])


def test_exact_chunked_error_vector_prefetch_window_stays_bounded() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]),
        values=torch.ones(4),
        size=(4, 1, 1),
        check_invariants=True,
    ).coalesce()
    error_vectors = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8)
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=error_vectors,
        token_vectors=torch.zeros(1, 8),
        decoder_vecs=torch.empty((0, 8)),
        encoder_vecs=torch.ones((activation_matrix._nnz(), 8)),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1, 1, 5),
        decoder_provider=FakeDecoderProvider({0: torch.zeros(1, 4, 8)}),
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
        error_vector_prefetch_lookahead=2,
    )

    assert torch.equal(
        ctx.get_error_vectors_for_layer(3, device=torch.device("cpu")), error_vectors[3]
    )
    assert set(ctx._materialized_error_vector_layers) == {2, 3}

    assert torch.equal(
        ctx.get_error_vectors_for_layer(1, device=torch.device("cpu")), error_vectors[1]
    )
    assert set(ctx._materialized_error_vector_layers) == {0, 1}


def test_reorder_pending_for_phase4_locality_groups_layer_then_chunk_then_position() -> None:
    pending = torch.tensor([5, 1, 4, 0, 3, 2], dtype=torch.long)
    feat_layers = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.long)
    feat_positions = torch.tensor([2, 1, 0, 2, 1, 0], dtype=torch.long)
    feat_ids = torch.tensor([9, 7, 1, 4, 6, 3], dtype=torch.long)

    reordered = _reorder_pending_for_phase4_locality(
        pending,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=4,
    )

    assert torch.equal(reordered, torch.tensor([5, 1, 3, 2, 4, 0], dtype=torch.long))


def test_phase4_planner_batch_size_grows_and_respects_max_cap() -> None:
    assert (
        _compute_phase4_planned_feature_batch_size(
            128,
            max_feature_batch_size=256,
            observed_reserved_bytes=8 * 1024**3,
            total_cuda_bytes=40 * 1024**3,
            target_reserved_fraction=0.9,
            min_free_fraction=0.05,
        )
        == 256
    )


def test_phase4_planner_batch_size_shrinks_when_probe_is_over_budget() -> None:
    assert (
        _compute_phase4_planned_feature_batch_size(
            128,
            max_feature_batch_size=256,
            observed_reserved_bytes=32 * 1024**3,
            total_cuda_bytes=40 * 1024**3,
            target_reserved_fraction=0.7,
            min_free_fraction=0.05,
        )
        == 111
    )


def test_phase4_planner_batch_size_uses_min_free_fraction_guardrail() -> None:
    assert (
        _compute_phase4_planned_feature_batch_size(
            128,
            max_feature_batch_size=512,
            observed_reserved_bytes=16 * 1024**3,
            total_cuda_bytes=40 * 1024**3,
            target_reserved_fraction=0.95,
            min_free_fraction=0.2,
        )
        == 256
    )


def test_phase4_probe_frontier_uses_ranked_first_frontier_then_locality() -> None:
    feature_influences = torch.tensor([0.1, 0.7, 0.2, 0.9, 0.3, 0.8], dtype=torch.float32)
    feat_layers = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.long)
    feat_positions = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    feat_ids = torch.tensor([5, 8, 2, 1, 6, 4], dtype=torch.long)

    pending = _build_phase4_probe_pending_frontier(
        feature_influences=feature_influences,
        total_active_feats=6,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=4,
        initial_feature_batch_size=2,
        feature_batch_probe_batches=1,
        update_interval=2,
        max_feature_nodes=3,
    )

    assert torch.equal(pending, torch.tensor([3, 5], dtype=torch.long))


def test_phase4_probe_frontier_preserves_full_frontier_order_when_all_features_included() -> None:
    feat_layers = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    feat_positions = torch.tensor([2, 0, 1, 3], dtype=torch.long)
    feat_ids = torch.tensor([8, 1, 6, 3], dtype=torch.long)

    pending = _build_phase4_probe_pending_frontier(
        feature_influences=torch.tensor([0.4, 0.9, 0.2, 0.7], dtype=torch.float32),
        total_active_feats=4,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=4,
        initial_feature_batch_size=2,
        feature_batch_probe_batches=2,
        update_interval=4,
        max_feature_nodes=4,
    )

    assert torch.equal(pending, torch.tensor([0, 1, 2, 3], dtype=torch.long))


def test_top_level_attribute_rejects_phase4_planner_flags() -> None:
    class _DummyModel:
        backend = "nnsight"

    with pytest.raises(
        ValueError,
        match=r"unsupported via circuit_tracer\.attribution\.attribute",
    ):
        attribute_top_level(
            prompt="hello",
            model=cast(object, _DummyModel()),
            plan_feature_batch_size=True,
        )


def test_chunked_feature_replay_windows_match_full_replay() -> None:
    grads_by_output_layer = [
        torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ]
        ),
        torch.tensor(
            [
                [[5.0, 50.0], [6.0, 60.0]],
                [[7.0, 70.0], [8.0, 80.0]],
            ]
        ),
        None,
    ]

    full_ctx, _ = _make_chunked_context(NNSightAttributionContext, enable_cache=True)
    full_ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)
    expected = full_ctx._batch_buffer.clone()

    windowed_ctx, _ = _make_chunked_context(NNSightAttributionContext, enable_cache=True)
    windowed_ctx._compute_chunked_feature_attributions_from_grads(
        [grads_by_output_layer[0], None, None]
    )
    windowed_ctx._compute_chunked_feature_attributions_from_grads(
        [None, grads_by_output_layer[1], None]
    )

    assert torch.allclose(windowed_ctx._batch_buffer, expected)


def test_decoder_cache_stays_enabled_on_churn() -> None:
    n_chunks = 16
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.stack(
            [
                torch.zeros(n_chunks, dtype=torch.long),
                torch.arange(n_chunks, dtype=torch.long),
                torch.arange(n_chunks, dtype=torch.long),
            ]
        ),
        values=torch.ones(n_chunks),
        size=(1, n_chunks, n_chunks),
        check_invariants=True,
    ).coalesce()
    provider = GuardrailDecoderProvider(
        {0: torch.ones(n_chunks, 1, 2, dtype=torch.float32)},
        cache_max_bytes=8,
    )
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(1, n_chunks, 2),
        token_vectors=torch.zeros(n_chunks, 2),
        decoder_vecs=torch.empty((0, 2)),
        encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1, 1, 5),
        decoder_provider=provider,
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
    )
    ctx._batch_buffer = torch.zeros(ctx._row_size, 1)
    grads = [torch.ones(1, n_chunks, 2)]

    ctx._compute_chunked_feature_attributions_from_grads(grads)

    assert ctx.decoder_chunk_cache is not None
    assert not provider.auto_disable_reasons
    assert provider.stats["decoder_cache_eviction_count"] > 0


def test_decoder_cache_guardrail_keeps_useful_cache_enabled() -> None:
    n_chunks = 8
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.stack(
            [
                torch.zeros(n_chunks, dtype=torch.long),
                torch.arange(n_chunks, dtype=torch.long),
                torch.arange(n_chunks, dtype=torch.long),
            ]
        ),
        values=torch.ones(n_chunks),
        size=(1, n_chunks, n_chunks),
        check_invariants=True,
    ).coalesce()
    provider = GuardrailDecoderProvider(
        {0: torch.ones(n_chunks, 1, 2, dtype=torch.float32)},
        cache_max_bytes=64,
    )
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(1, n_chunks, 2),
        token_vectors=torch.zeros(n_chunks, 2),
        decoder_vecs=torch.empty((0, 2)),
        encoder_vecs=torch.zeros((activation_matrix._nnz(), 2)),
        encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
        decoder_locations=torch.empty((2, 0), dtype=torch.long),
        logits=torch.zeros(1, 1, 5),
        decoder_provider=provider,
        chunked_decoder_state={
            "source_layers": activation_matrix.indices()[0],
            "positions": activation_matrix.indices()[1],
            "feature_ids": activation_matrix.indices()[2],
            "activation_values": activation_matrix.values(),
        },
    )
    ctx._batch_buffer = torch.zeros(ctx._row_size, 1)
    grads = [torch.ones(1, n_chunks, 2)]

    ctx._compute_chunked_feature_attributions_from_grads(grads)
    ctx._batch_buffer.zero_()
    ctx._compute_chunked_feature_attributions_from_grads(grads)

    assert ctx.decoder_chunk_cache is not None
    assert not provider.auto_disable_reasons
    assert provider.stats["decoder_cache_hit_count"] == n_chunks


def test_context_cleanup_is_idempotent_and_clears_buffers() -> None:
    ctx, provider = _make_chunked_context(NNSightAttributionContext, enable_cache=True)
    ctx.get_error_vectors_for_layer(1, device=torch.device("cpu"))
    ctx.cleanup()
    ctx.cleanup()

    assert provider.clear_calls >= 1
    assert ctx.decoder_chunk_cache is None
    assert ctx.encoder_vecs.numel() == 0
    assert ctx.error_vectors.numel() == 0
    assert ctx.token_vectors.numel() == 0
    assert ctx.logits.numel() == 0
