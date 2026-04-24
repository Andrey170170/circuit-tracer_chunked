from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.attribution.attribute import attribute as attribute_top_level
from circuit_tracer.attribution.attribute_nnsight import (
    _build_cross_cluster_runtime_snapshot,
    _build_matrix_abs_stats,
    _build_phase4_batch_locality_summary,
    _build_phase4_normalization_stats,
    _build_phase4_planner_v2_candidate_window,
    _select_phase4_planner_v2_membership,
    _apply_phase4_planner_v2_refresh_plan,
    _build_phase4_deterministic_shadow_pending,
    _build_phase4_cutoff_debug,
    _build_phase4_probe_pending_frontier,
    _build_phase4_scheduler_metadata,
    _build_phase4_scheduler_plan_telemetry,
    _record_cross_cluster_batch_event,
    _record_cross_cluster_checkpoint,
    _compute_row_abs_sums,
    _build_vector_stats,
    _compare_phase4_frontiers,
    _compute_phase4_planned_feature_batch_size,
    _compute_phase4_locality_shaped_batch_end,
    _compute_phase4_locality_shaped_frontier_size,
    _plan_phase4_frontier_membership_preserving_v1,
    _reorder_pending_for_phase4_locality,
    _resolve_internal_dtype_map,
    _resolve_internal_precision_requested,
    _resolve_phase4_anomaly_debug_enabled,
    _resolve_phase4_feature_batch_planner_status,
    _resolve_phase4_refresh_optimization_mode,
    _resolve_phase4_refresh_optimization_config,
    _build_phase4_refresh_optimization_metadata,
    _resolve_phase4_row_executor_mode,
    _resolve_phase4_row_executor_config,
    _build_phase4_row_executor_metadata,
    _resolve_phase4_scheduler_mode,
    _resolve_phase4_scheduler_config,
    _resolve_phase4_scheduler_telemetry_detail,
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
    staged_encoder_source = encoder_vecs.clone()
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 2, 4),
        token_vectors=torch.zeros(2, 4),
        decoder_vecs=torch.empty((0, 4)),
        encoder_vecs=staged_encoder_source,
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
    assert ctx.encoder_vecs.data_ptr() != staged_encoder_source.data_ptr()
    batch = ctx.materialize_encoder_vectors(torch.tensor([2, 0]), device=torch.device("cpu"))
    assert torch.equal(batch, encoder_vecs[torch.tensor([2, 0])])


def test_stage_tensor_on_cpu_preserves_existing_cpu_layout() -> None:
    source = torch.arange(24, dtype=torch.float32, requires_grad=True).reshape(4, 6).transpose(0, 1)
    assert source.device.type == "cpu"
    assert not source.is_contiguous()

    staged = NNSightAttributionContext._stage_tensor_on_cpu(source)

    assert staged.device.type == "cpu"
    assert staged.data_ptr() != source.data_ptr()
    assert staged.stride() == source.stride()
    assert not staged.requires_grad


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

    assert ctx.error_vectors.data_ptr() != error_vectors.data_ptr()
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


def test_phase4_scheduler_mode_resolves_and_accepts_legacy_alias() -> None:
    assert _resolve_phase4_scheduler_mode("locality") == "locality"
    assert _resolve_phase4_scheduler_mode("planner_v1") == "planner_v1"
    assert _resolve_phase4_scheduler_mode("planner_v2") == "planner_v2"
    assert _resolve_phase4_scheduler_mode("legacy") == "locality"


def test_phase4_scheduler_config_planner_v2_tracks_requested_and_effective_policy() -> None:
    config = _resolve_phase4_scheduler_config(
        phase4_scheduler_mode="planner_v2",
        phase4_scheduler_debug=False,
        phase4_scheduler_telemetry_detail="normal",
    )
    metadata = _build_phase4_scheduler_metadata(config)

    assert metadata["scheduler_requested_mode"] == "planner_v2"
    assert metadata["scheduler_mode_requested"] == "planner_v2"
    assert metadata["scheduler_mode"] == "planner_v2"
    assert metadata["scheduler_version"] == "planner_v2"
    assert metadata["scheduler_version_requested"] == "planner_v2"
    assert metadata["scheduler_policy"] == "bounded_membership_selection"
    assert metadata["scheduler_policy_requested"] == "bounded_membership_selection"
    assert metadata["scheduler_effective_mode"] == "planner_v2"
    assert metadata["scheduler_mode_effective"] == "planner_v2"
    assert metadata["scheduler_effective_version"] == "planner_v2"
    assert metadata["scheduler_version_effective"] == "planner_v2"
    assert metadata["scheduler_effective_policy"] == "bounded_membership_selection"
    assert metadata["scheduler_policy_effective"] == "bounded_membership_selection"
    assert metadata["scheduler_effective_behavior"] == "requested"
    assert metadata["scheduler_reference_execution"] is False


def test_phase4_scheduler_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="phase4_scheduler_mode must be one of"):
        _resolve_phase4_scheduler_mode("unsupported")


def test_phase4_scheduler_telemetry_detail_resolves_aliases() -> None:
    assert _resolve_phase4_scheduler_telemetry_detail("summary") == "summary"
    assert _resolve_phase4_scheduler_telemetry_detail("normal") == "normal"
    assert _resolve_phase4_scheduler_telemetry_detail("debug") == "debug"
    assert _resolve_phase4_scheduler_telemetry_detail("compact") == "summary"
    assert _resolve_phase4_scheduler_telemetry_detail("full") == "debug"


def test_phase4_scheduler_telemetry_detail_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match="phase4_scheduler_telemetry_detail must be one of",
    ):
        _resolve_phase4_scheduler_telemetry_detail("verbose")


def test_phase4_refresh_optimization_mode_resolves_and_rejects_unknown() -> None:
    assert _resolve_phase4_refresh_optimization_mode("off") == "off"
    assert _resolve_phase4_refresh_optimization_mode("v1") == "v1"
    with pytest.raises(ValueError, match="phase4_refresh_optimization must be one of"):
        _resolve_phase4_refresh_optimization_mode("v2")


def test_phase4_refresh_optimization_metadata_tracks_requested_and_effective_modes() -> None:
    config = _resolve_phase4_refresh_optimization_config("v1")
    metadata = _build_phase4_refresh_optimization_metadata(config)

    assert metadata["refresh_optimization_requested"] == "v1"
    assert metadata["refresh_optimization_mode_requested"] == "v1"
    assert metadata["refresh_optimization"] == "v1"
    assert metadata["refresh_optimization_effective"] == "off"
    assert metadata["refresh_optimization_mode_effective"] == "off"
    assert metadata["refresh_optimization_reference_execution"] is True


def test_phase4_row_executor_mode_resolves_and_rejects_unknown() -> None:
    assert _resolve_phase4_row_executor_mode("batched") == "batched"
    assert _resolve_phase4_row_executor_mode("streaming_v1") == "streaming_v1"
    with pytest.raises(ValueError, match="phase4_row_executor must be one of"):
        _resolve_phase4_row_executor_mode("streaming_v2")


def test_phase4_row_executor_metadata_tracks_requested_and_effective_modes() -> None:
    config = _resolve_phase4_row_executor_config("streaming_v1")
    metadata = _build_phase4_row_executor_metadata(config)

    assert metadata["row_executor_requested"] == "streaming_v1"
    assert metadata["row_executor_mode_requested"] == "streaming_v1"
    assert metadata["row_executor"] == "streaming_v1"
    assert metadata["row_executor_effective"] == "batched"
    assert metadata["row_executor_mode_effective"] == "batched"
    assert metadata["row_executor_reference_execution"] is True


def test_phase4_batch_locality_summary_reports_layer_and_chunk_ranges() -> None:
    summary = _build_phase4_batch_locality_summary(
        torch.tensor([3, 1, 2], dtype=torch.long),
        feat_layers=torch.tensor([2, 1, 1, 3], dtype=torch.long),
        feat_ids=torch.tensor([0, 4, 6, 7], dtype=torch.long),
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert summary["scheduler_batch_hash"] is not None
    assert summary["scheduler_batch_distinct_source_layer_count"] == 2
    assert summary["scheduler_batch_source_layer_min"] == 1
    assert summary["scheduler_batch_source_layer_max"] == 3
    assert summary["scheduler_batch_distinct_decoder_chunk_count"] == 2
    assert summary["scheduler_batch_decoder_chunk_min"] == 2
    assert summary["scheduler_batch_decoder_chunk_max"] == 3
    assert summary["scheduler_batch_monotonic_chunk_order"] is False


def test_phase4_batch_locality_summary_treats_cross_layer_chunk_resets_as_monotonic() -> None:
    summary = _build_phase4_batch_locality_summary(
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        feat_layers=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        feat_ids=torch.tensor([4, 6, 0, 2], dtype=torch.long),
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert summary["scheduler_batch_distinct_source_layer_count"] == 2
    assert summary["scheduler_batch_decoder_chunk_min"] == 0
    assert summary["scheduler_batch_decoder_chunk_max"] == 3
    assert summary["scheduler_batch_monotonic_chunk_order"] is True


def test_phase4_planner_v1_preserves_membership_and_boundaries() -> None:
    pending_candidates = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    feat_layers = torch.zeros(6, dtype=torch.long)
    feat_positions = torch.arange(6, dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    plan = _plan_phase4_frontier_membership_preserving_v1(
        pending_candidates,
        max_batch_size=3,
        max_batches=2,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert torch.equal(plan.selected_frontier, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert plan.batch_boundaries == [(0, 2), (2, 4)]
    assert plan.boundary_reason_counts == {"split_at_last_boundary": 1, "tail_complete": 1}
    assert plan.selected_membership_hash is not None
    assert plan.selected_order_hash is not None
    assert plan.invariant_summary["membership_preserved"] is True
    assert plan.locality_fragmentation_summary["selected_count"] == 4
    assert plan.locality_fragmentation_summary["batch_count"] == 2


def test_phase4_planner_v1_rejects_invalid_parameters_and_duplicates() -> None:
    feat_layers = torch.zeros(4, dtype=torch.long)
    feat_positions = torch.arange(4, dtype=torch.long)
    feat_ids = torch.arange(4, dtype=torch.long)

    with pytest.raises(ValueError, match="max_batch_size must be > 0"):
        _plan_phase4_frontier_membership_preserving_v1(
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            max_batch_size=0,
            max_batches=1,
            feat_layers=feat_layers,
            feat_positions=feat_positions,
            feat_ids=feat_ids,
            exact_chunked_decoder=False,
            decoder_chunk_size=None,
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        _plan_phase4_frontier_membership_preserving_v1(
            torch.tensor([0, 1, 1, 2], dtype=torch.long),
            max_batch_size=2,
            max_batches=2,
            feat_layers=feat_layers,
            feat_positions=feat_positions,
            feat_ids=feat_ids,
            exact_chunked_decoder=False,
            decoder_chunk_size=None,
        )


def test_phase4_planner_v2_candidate_window_includes_reference_frontier() -> None:
    unvisited_feature_rank = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    candidate_scores = torch.tensor([1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92])
    reference_frontier = torch.tensor([0, 4, 7], dtype=torch.long)

    candidate_window, telemetry = _build_phase4_planner_v2_candidate_window(
        unvisited_feature_rank,
        reference_frontier=reference_frontier,
        reference_frontier_size=3,
        candidate_scores=candidate_scores,
    )

    assert torch.equal(candidate_window, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long))
    assert telemetry["scheduler_planner_v2_enabled"] is True
    assert telemetry["scheduler_planner_v2_reference_frontier_size"] == 3
    assert telemetry["scheduler_planner_v2_candidate_window_size"] == 8
    assert telemetry["scheduler_planner_v2_candidate_window_includes_reference"] is True
    assert telemetry["scheduler_planner_v2_candidate_window_order_hash"] is not None


def test_phase4_planner_v2_candidate_window_respects_min_score_ratio_near_cutoff() -> None:
    unvisited_feature_rank = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    candidate_scores = torch.tensor([1.0, 0.99, 0.80, 0.79])
    reference_frontier = torch.tensor([0, 1], dtype=torch.long)

    candidate_window, telemetry = _build_phase4_planner_v2_candidate_window(
        unvisited_feature_rank,
        reference_frontier=reference_frontier,
        reference_frontier_size=2,
        candidate_scores=candidate_scores,
    )

    assert torch.equal(candidate_window, torch.tensor([0, 1], dtype=torch.long))
    assert telemetry["scheduler_planner_v2_candidate_window_size"] == 2
    assert telemetry["scheduler_planner_v2_score_threshold_applied"] is True
    assert telemetry["scheduler_planner_v2_score_threshold"] == pytest.approx(0.99 * 0.995)


def test_phase4_planner_v2_candidate_window_handles_empty_reference_and_short_unvisited() -> None:
    candidate_window, telemetry = _build_phase4_planner_v2_candidate_window(
        torch.tensor([9], dtype=torch.long),
        reference_frontier=torch.tensor([], dtype=torch.long),
        reference_frontier_size=0,
        candidate_scores=torch.tensor([0.5]),
    )

    assert candidate_window.numel() == 0
    assert telemetry["scheduler_planner_v2_reference_frontier_size"] == 0
    assert telemetry["scheduler_planner_v2_candidate_window_size"] == 0
    assert telemetry["scheduler_planner_v2_candidate_window_order_hash"] is None
    assert telemetry["scheduler_planner_v2_candidate_window_includes_reference"] is True


def test_phase4_planner_v2_selection_preserves_locked_prefix_and_bounds() -> None:
    selected_membership, telemetry = _select_phase4_planner_v2_membership(
        unvisited_feature_rank=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        reference_frontier=torch.tensor([0, 4, 5, 6], dtype=torch.long),
        reference_frontier_size=4,
        candidate_window=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
        candidate_scores=torch.tensor(
            [1.0, 0.9995, 0.999, 0.9985, 0.998, 0.9975, 0.997, 0.9965],
            dtype=torch.float64,
        ),
        visited=torch.zeros(8, dtype=torch.bool),
        feat_layers=torch.tensor([0, 0, 1, 0, 1, 2, 3, 1], dtype=torch.long),
        feat_ids=torch.zeros(8, dtype=torch.long),
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert telemetry["scheduler_planner_v2_selection_applied"] is True
    assert telemetry["scheduler_planner_v2_fallback_to_reference"] is False
    assert selected_membership.numel() == 4
    assert torch.unique(selected_membership).numel() == 4
    assert {0, 4}.issubset(set(selected_membership.tolist()))
    assert int(telemetry["scheduler_planner_v2_replacement_count"]) <= 1
    assert float(telemetry["scheduler_planner_v2_selected_score_ratio"]) >= 0.995


def test_phase4_planner_v2_selection_falls_back_when_score_ratio_fails() -> None:
    selected_membership, telemetry = _select_phase4_planner_v2_membership(
        unvisited_feature_rank=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        reference_frontier=torch.tensor([0, 3, 4, 5], dtype=torch.long),
        reference_frontier_size=4,
        candidate_window=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        candidate_scores=torch.tensor([1.0, 0.5, 0.49, 0.99, 0.98, 0.97], dtype=torch.float64),
        visited=torch.zeros(6, dtype=torch.bool),
        feat_layers=torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.long),
        feat_ids=torch.zeros(6, dtype=torch.long),
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert telemetry["scheduler_planner_v2_selection_applied"] is False
    assert telemetry["scheduler_planner_v2_fallback_to_reference"] is True
    assert telemetry["scheduler_planner_v2_fallback_reason"] == "score_ratio_below_threshold"
    assert torch.equal(selected_membership, torch.tensor([0, 3, 4, 5], dtype=torch.long))


def test_phase4_planner_v2_refresh_fallback_reuses_reference_plan_when_invalid() -> None:
    feat_layers = torch.tensor([0, 0, 1, 0, 1, 2, 3, 1], dtype=torch.long)
    feat_positions = torch.arange(8, dtype=torch.long)
    feat_ids = torch.zeros(8, dtype=torch.long)
    reference_plan = _plan_phase4_frontier_membership_preserving_v1(
        torch.tensor([0, 4, 5, 6], dtype=torch.long),
        max_batch_size=2,
        max_batches=2,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
        apply_locality_reorder=False,
    )
    visited = torch.zeros(8, dtype=torch.bool)
    visited[4] = True

    selected_plan, _candidate_window, telemetry = _apply_phase4_planner_v2_refresh_plan(
        reference_plan=reference_plan,
        unvisited_feature_rank=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        candidate_scores=torch.tensor(
            [1.0, 0.9995, 0.999, 0.9985, 0.998, 0.9975, 0.997, 0.9965],
            dtype=torch.float64,
        ),
        visited=visited,
        max_batch_size=2,
        max_batches=2,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert telemetry["scheduler_planner_v2_fallback_to_reference"] is True
    assert telemetry["scheduler_planner_v2_fallback_reason"] == "reference_contains_visited_feature"
    assert torch.equal(selected_plan.selected_frontier, reference_plan.selected_frontier)
    assert selected_plan.batch_boundaries == reference_plan.batch_boundaries
    assert selected_plan.invariant_summary["planner_v2_fallback_to_reference"] is True


def test_phase4_planner_v2_refresh_can_change_membership_for_better_grouping() -> None:
    feat_layers = torch.tensor([0, 0, 1, 0, 1, 2, 3, 1], dtype=torch.long)
    feat_positions = torch.arange(8, dtype=torch.long)
    feat_ids = torch.zeros(8, dtype=torch.long)
    reference_plan = _plan_phase4_frontier_membership_preserving_v1(
        torch.tensor([0, 4, 5, 6], dtype=torch.long),
        max_batch_size=2,
        max_batches=2,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
        apply_locality_reorder=False,
    )

    selected_plan, _candidate_window, telemetry = _apply_phase4_planner_v2_refresh_plan(
        reference_plan=reference_plan,
        unvisited_feature_rank=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        candidate_scores=torch.tensor(
            [1.0, 0.9995, 0.999, 0.9985, 0.998, 0.9975, 0.997, 0.9965],
            dtype=torch.float64,
        ),
        visited=torch.zeros(8, dtype=torch.bool),
        max_batch_size=2,
        max_batches=2,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert telemetry["scheduler_planner_v2_selection_applied"] is True
    assert telemetry["scheduler_planner_v2_selection_changed_membership"] is True
    assert telemetry["scheduler_planner_v2_fallback_to_reference"] is False
    assert int(telemetry["scheduler_planner_v2_replacement_count"]) == 1
    assert int(telemetry["scheduler_planner_v2_group_count_delta"]) >= 1
    assert selected_plan.selected_membership_hash != reference_plan.selected_membership_hash
    assert selected_plan.invariant_summary["planner_v2_changed_membership"] is True


def test_phase4_planner_v2_refresh_fails_closed_when_candidate_window_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feat_layers = torch.tensor([0, 0, 1, 0], dtype=torch.long)
    feat_positions = torch.arange(4, dtype=torch.long)
    feat_ids = torch.zeros(4, dtype=torch.long)
    reference_plan = _plan_phase4_frontier_membership_preserving_v1(
        torch.tensor([0, 1], dtype=torch.long),
        max_batch_size=2,
        max_batches=1,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
        apply_locality_reorder=False,
    )

    def _raise_candidate_window(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight._build_phase4_planner_v2_candidate_window",
        _raise_candidate_window,
    )

    selected_plan, candidate_window, telemetry = _apply_phase4_planner_v2_refresh_plan(
        reference_plan=reference_plan,
        unvisited_feature_rank=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        candidate_scores=torch.tensor([1.0, 0.9, 0.8, 0.7], dtype=torch.float64),
        visited=torch.zeros(4, dtype=torch.bool),
        max_batch_size=2,
        max_batches=1,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert candidate_window.numel() == 0
    assert torch.equal(selected_plan.selected_frontier, reference_plan.selected_frontier)
    assert telemetry["scheduler_planner_v2_fallback_to_reference"] is True
    assert telemetry["scheduler_planner_v2_selection_applied"] is False
    assert (
        telemetry["scheduler_planner_v2_fallback_reason"]
        == "planner_v2_selection_error:RuntimeError"
    )


def test_phase4_scheduler_plan_telemetry_reports_full_frontier_planner_metadata() -> None:
    pending_candidates = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    feat_layers = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    feat_positions = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    plan = _plan_phase4_frontier_membership_preserving_v1(
        pending_candidates,
        max_batch_size=2,
        max_batches=None,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
        apply_locality_reorder=False,
    )
    telemetry = _build_phase4_scheduler_plan_telemetry(
        phase4_frontier_plan=plan,
        telemetry_detail="normal",
    )

    assert telemetry["scheduler_plan_frontier_size"] == 4
    assert telemetry["scheduler_plan_membership_hash"] == plan.selected_membership_hash
    assert telemetry["scheduler_plan_order_hash"] == plan.selected_order_hash
    assert telemetry["scheduler_plan_batch_count"] == 2
    assert telemetry["scheduler_plan_boundary_reason_counts"] == plan.boundary_reason_counts
    assert telemetry["scheduler_plan_invariants"] == plan.invariant_summary


def test_phase4_locality_shaped_batch_end_prefers_layer_chunk_boundaries() -> None:
    pending = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    feat_layers = torch.zeros(6, dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    end0 = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=0,
        max_batch_size=3,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )
    end1 = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=end0,
        max_batch_size=3,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )
    end2 = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=end1,
        max_batch_size=3,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert end0 == 2
    assert end1 == 4
    assert end2 == 6


def test_phase4_locality_shaped_batch_end_keeps_baseline_when_split_unavoidable() -> None:
    pending = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    feat_layers = torch.zeros(4, dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    end = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=0,
        max_batch_size=2,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=8,
    )

    assert end == 2


def test_phase4_locality_shaped_batch_end_avoids_tiny_split_batches() -> None:
    pending = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    feat_layers = torch.zeros(4, dtype=torch.long)
    feat_ids = torch.tensor([0, 2, 3, 3], dtype=torch.long)

    end = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=0,
        max_batch_size=3,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert end == 3


def test_phase4_locality_shaped_batch_end_avoids_preserving_long_suffix_run() -> None:
    pending = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
    feat_layers = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    feat_ids = torch.zeros(7, dtype=torch.long)

    end = _compute_phase4_locality_shaped_batch_end(
        pending,
        pending_offset=0,
        max_batch_size=6,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert end == 6


def test_phase4_locality_shaped_frontier_size_preserves_update_interval_batch_cadence() -> None:
    pending = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    feat_layers = torch.zeros(6, dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

    frontier_size = _compute_phase4_locality_shaped_frontier_size(
        pending,
        max_batch_size=3,
        max_batches=2,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
    )

    assert frontier_size == 4


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


def test_phase4_planner_status_skips_when_no_headroom() -> None:
    assert _resolve_phase4_feature_batch_planner_status(
        planner_enabled=True,
        effective_feature_batch_size=128,
        max_feature_batch_size=128,
    ) == (
        "skipped_no_headroom",
        "feature_batch_size_max does not exceed initial feature_batch_size",
    )


def test_phase4_planner_status_is_pending_when_growth_is_possible() -> None:
    assert _resolve_phase4_feature_batch_planner_status(
        planner_enabled=True,
        effective_feature_batch_size=128,
        max_feature_batch_size=256,
    ) == ("pending", None)


def test_phase4_anomaly_debug_enabled_from_flag() -> None:
    assert _resolve_phase4_anomaly_debug_enabled(True) is True


def test_phase4_anomaly_debug_enabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE4_ANOMALY_DEBUG", "1")
    assert _resolve_phase4_anomaly_debug_enabled(False) is True


def test_internal_precision_contract_resolves_float64_defaults() -> None:
    precision = _resolve_internal_precision_requested("float64")
    dtype_map = _resolve_internal_dtype_map(
        internal_precision_requested=precision,
        phase4_anomaly_debug_enabled=False,
    )

    assert dtype_map["internal_precision_requested"] == "float64"
    assert dtype_map["feature_row_storage_dtype"] == "float32"
    assert dtype_map["row_abs_sum_dtype"] == "float64"
    assert dtype_map["influence_compute_dtype"] == "float64"
    assert dtype_map["planner_compute_dtype"] == "float64"
    assert dtype_map["shadow_debug_compute_dtype"] == "float64"


def test_internal_precision_contract_resolves_float32_math() -> None:
    precision = _resolve_internal_precision_requested("float32")
    dtype_map = _resolve_internal_dtype_map(
        internal_precision_requested=precision,
        phase4_anomaly_debug_enabled=False,
    )

    assert dtype_map["internal_precision_requested"] == "float32"
    assert dtype_map["feature_row_storage_dtype"] == "float32"
    assert dtype_map["row_abs_sum_dtype"] == "float32"
    assert dtype_map["influence_compute_dtype"] == "float32"
    assert dtype_map["planner_compute_dtype"] == "float32"


def test_build_phase4_cutoff_debug_reports_margin_and_ties() -> None:
    scores = torch.tensor([1.0, 0.9, 0.9, 0.5], dtype=torch.float32)
    result = _build_phase4_cutoff_debug(scores, queue_size=2)

    assert result["cutoff_rank"] == 1
    assert result["cutoff_score"] == pytest.approx(0.9)
    assert result["next_score"] == pytest.approx(0.9)
    assert result["cutoff_margin"] == pytest.approx(0.0)
    assert result["exact_cutoff_count"] == 2
    assert result["near_cutoff_count"] >= 2


def test_build_vector_stats_reports_effective_zero_signal() -> None:
    stats = _build_vector_stats(torch.tensor([0.0, 0.0, 1e-13], dtype=torch.float32), epsilon=1e-12)

    assert stats["count"] == 3
    assert stats["nonzero_count"] == 1
    assert stats["effective_nonzero_count"] == 0
    assert stats["all_zero"] is False
    assert stats["effectively_all_zero"] is True


def test_build_vector_stats_reports_nonfinite_counts() -> None:
    stats = _build_vector_stats(
        torch.tensor([0.0, float("inf"), float("nan")], dtype=torch.float32)
    )

    assert stats["count"] == 3
    assert stats["finite_count"] == 1
    assert stats["posinf_count"] == 1
    assert stats["nan_count"] == 1
    assert stats["nonfinite_count"] == 2


def test_compute_row_abs_sums_uses_float64_accumulation() -> None:
    rows = torch.tensor([[1e38, 1e38, 1e38]], dtype=torch.float32)
    result = _compute_row_abs_sums(rows)

    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()
    assert result[0].item() == pytest.approx(3e38)


def test_build_matrix_abs_stats_reports_row_l1_nonfinite_counts() -> None:
    stats = _build_matrix_abs_stats(
        torch.tensor([[1.0, float("inf")], [0.0, 0.0]], dtype=torch.float32)
    )

    assert stats["nonfinite_count"] == 1
    assert stats["row_l1_stats"]["posinf_count"] == 1


def test_phase4_normalization_stats_reports_clamped_rows() -> None:
    stats = _build_phase4_normalization_stats(
        torch.tensor([0.0, 1e-9, 2.0], dtype=torch.float32),
        clamp_epsilon=1e-8,
    )

    assert stats["clamped_row_count"] == 2
    assert stats["clamped_row_fraction"] == pytest.approx(2 / 3)


def test_record_cross_cluster_checkpoint_updates_summary_and_stream() -> None:
    summary: dict[str, object] = {"checkpoints": {}}
    stream: list[dict[str, object]] = []

    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=summary,
        cross_cluster_debug_checkpoints=stream,
        checkpoint_name="phase1_target_logits",
        phase="phase1",
        summary_payload={"target_count": 2, "target_token_ids_hash": "abc123"},
        stream_payload={"target_count": 2, "target_probability_abs_sum": 0.95},
    )

    checkpoints = summary.get("checkpoints")
    assert isinstance(checkpoints, dict)
    assert checkpoints["phase1_target_logits"]["target_count"] == 2
    assert len(stream) == 1
    assert stream[0]["checkpoint_name"] == "phase1_target_logits"
    assert stream[0]["phase"] == "phase1"
    assert stream[0]["target_probability_abs_sum"] == pytest.approx(0.95)


def test_record_cross_cluster_batch_event_emits_scalar_event_record() -> None:
    events: list[dict[str, object]] = []

    _record_cross_cluster_batch_event(
        cross_cluster_debug_batches=events,
        event_name="phase4.refresh",
        phase="phase4",
        event_index=3,
        payload={"queue_size": 64, "rank_abs_sum": 12.5, "rank_effectively_all_zero": False},
    )

    assert len(events) == 1
    assert events[0]["event_name"] == "phase4.refresh"
    assert events[0]["phase"] == "phase4"
    assert events[0]["event_index"] == 3
    assert events[0]["queue_size"] == 64
    assert events[0]["rank_abs_sum"] == pytest.approx(12.5)


def test_build_cross_cluster_runtime_snapshot_emits_memory_and_hashes() -> None:
    summary_payload, stream_payload = _build_cross_cluster_runtime_snapshot(
        device=torch.device("cpu")
    )

    assert "memory_snapshot" in summary_payload
    assert "rss_current_gib" in stream_payload
    assert "rss_gib" in stream_payload
    assert "ctx_diagnostic_snapshot_hash" in stream_payload
    assert "transcoder_diagnostic_snapshot_hash" in stream_payload


def test_compare_phase4_frontiers_reports_overlap_and_first_difference() -> None:
    result = _compare_phase4_frontiers(
        torch.tensor([1, 3, 5], dtype=torch.long),
        torch.tensor([1, 4, 5], dtype=torch.long),
    )

    assert result["overlap_count"] == 2
    assert result["changed_selected_nodes"] == 2
    assert result["first_differing_rank"] == 1


def test_build_phase4_deterministic_shadow_pending_breaks_ties_stably() -> None:
    candidate_indices = torch.tensor([4, 2, 1, 3], dtype=torch.long)
    feature_influences = torch.tensor([0.0, 0.7, 0.7, 0.6, 0.7], dtype=torch.float32)
    feat_layers = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    feat_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    feat_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    pending = _build_phase4_deterministic_shadow_pending(
        candidate_indices,
        feature_influences,
        queue_size=3,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=False,
        decoder_chunk_size=None,
    )

    assert torch.equal(pending, torch.tensor([1, 2, 4], dtype=torch.long))


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


def test_top_level_attribute_forwards_phase4_scheduler_args_to_nnsight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_attribute(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("circuit_tracer.attribution.attribute_nnsight.attribute", _fake_attribute)

    class _DummyModel:
        backend = "nnsight"

    result = attribute_top_level(
        prompt="hello",
        model=cast(object, _DummyModel()),
        phase4_scheduler_mode="planner_v1",
        phase4_scheduler_debug=True,
        phase4_scheduler_telemetry_detail="debug",
        phase4_refresh_optimization="v1",
        phase4_row_executor="streaming_v1",
    )

    assert result is sentinel
    assert captured["phase4_scheduler_mode"] == "planner_v1"
    assert captured["phase4_scheduler_debug"] is True
    assert captured["phase4_scheduler_telemetry_detail"] == "debug"
    assert captured["phase4_refresh_optimization"] == "v1"
    assert captured["phase4_row_executor"] == "streaming_v1"


def test_top_level_attribute_accepts_default_phase4_scheduler_args_on_transformerlens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_attribute(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_transformerlens.attribute",
        _fake_attribute,
    )

    class _DummyModel:
        backend = "transformerlens"

    result = attribute_top_level(
        prompt="hello",
        model=cast(object, _DummyModel()),
        phase4_scheduler_mode="locality",
        phase4_scheduler_debug=False,
        phase4_scheduler_telemetry_detail="normal",
        phase4_refresh_optimization="off",
        phase4_row_executor="batched",
    )

    assert result is sentinel
    assert "phase4_scheduler_mode" not in captured
    assert "phase4_scheduler_debug" not in captured
    assert "phase4_scheduler_telemetry_detail" not in captured
    assert "phase4_refresh_optimization" not in captured
    assert "phase4_row_executor" not in captured


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        {"phase4_scheduler_mode": "planner_v1"},
        {"phase4_scheduler_mode": "planner_v2"},
        {"phase4_scheduler_debug": True},
        {"phase4_scheduler_telemetry_detail": "summary"},
        {"phase4_refresh_optimization": "v1"},
        {"phase4_row_executor": "streaming_v1"},
    ],
)
def test_top_level_attribute_rejects_non_default_phase4_scheduler_args_on_transformerlens(
    scheduler_kwargs: dict[str, object],
) -> None:
    class _DummyModel:
        backend = "transformerlens"

    with pytest.raises(
        ValueError,
        match=r"Phase-4 execution settings are only supported for the NNSight backend",
    ):
        attribute_top_level(
            prompt="hello",
            model=cast(object, _DummyModel()),
            **scheduler_kwargs,
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


def test_chunked_attr_fallback_handles_nonmonotonic_chunk_ids_within_layer() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 0], [0, 1, 2], [0, 2, 1]]),
        values=torch.tensor([1.0, 1.0, 1.0]),
        size=(2, 3, 3),
        check_invariants=True,
    ).coalesce()
    provider = FakeDecoderProvider(
        {
            0: torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 1.0], [2.0, 0.0]],
                ]
            )
        },
        chunk_size=2,
        enable_cache=True,
    )
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 3, 2),
        token_vectors=torch.zeros(3, 2),
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
    ctx._batch_buffer = torch.zeros(ctx._row_size, 1)

    grads_by_output_layer = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        torch.tensor([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]),
    ]

    expected = torch.zeros(activation_matrix._nnz(), 1)
    positions = activation_matrix.indices()[1]
    feature_ids = activation_matrix.indices()[2]
    activations = activation_matrix.values()
    decoder_block = provider.blocks[0]
    for row_idx in range(activation_matrix._nnz()):
        position = int(positions[row_idx].item())
        feature_id = int(feature_ids[row_idx].item())
        activation = activations[row_idx]
        total = torch.zeros(1)
        for output_layer, grads in enumerate(grads_by_output_layer):
            decoder_vec = decoder_block[feature_id, output_layer]
            total += torch.einsum("bd,d->b", grads[:, position], decoder_vec) * activation
        expected[row_idx] = total

    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    assert torch.allclose(ctx._batch_buffer[: activation_matrix._nnz()], expected)
    assert provider.load_calls == [(0, 0), (0, 1)]


def test_chunked_attr_monotonic_chunk_fast_path_matches_reference() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 1, 2, 3]]),
        values=torch.tensor([1.0, 1.5, 2.0, 0.5]),
        size=(2, 4, 4),
        check_invariants=True,
    ).coalesce()
    provider = FakeDecoderProvider(
        {
            0: torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 1.0], [2.0, 0.0]],
                    [[2.0, 1.0], [0.0, 2.0]],
                ]
            )
        },
        chunk_size=2,
        enable_cache=True,
    )
    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 4, 2),
        token_vectors=torch.zeros(4, 2),
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
    ctx._batch_buffer = torch.zeros(ctx._row_size, 1)

    grads_by_output_layer = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]),
        torch.tensor([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]]),
    ]

    expected = torch.zeros(activation_matrix._nnz(), 1)
    positions = activation_matrix.indices()[1]
    feature_ids = activation_matrix.indices()[2]
    activations = activation_matrix.values()
    decoder_block = provider.blocks[0]
    for row_idx in range(activation_matrix._nnz()):
        position = int(positions[row_idx].item())
        feature_id = int(feature_ids[row_idx].item())
        activation = activations[row_idx]
        total = torch.zeros(1)
        for output_layer, grads in enumerate(grads_by_output_layer):
            decoder_vec = decoder_block[feature_id, output_layer]
            total += torch.einsum("bd,d->b", grads[:, position], decoder_vec) * activation
        expected[row_idx] = total

    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    assert torch.allclose(ctx._batch_buffer[: activation_matrix._nnz()], expected)
    assert provider.load_calls == [(0, 0), (0, 1)]


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
