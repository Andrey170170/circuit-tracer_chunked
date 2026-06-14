from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.attribution.attribute import attribute as attribute_top_level
from circuit_tracer.attribution.attribute_nnsight import (
    _annotate_phase4_selection_on_feature_semantic_descriptors,
    _build_cross_cluster_runtime_snapshot,
    _build_matrix_abs_stats,
    _build_feature_semantic_descriptors_payload,
    _build_phase0_activation_matrix_from_loaded_bundle,
    _build_phase0_donor_bundle_payload,
    _build_phase0_replay_metadata,
    _build_phase0_replay_validation_context,
    _build_phase4_batch_locality_summary,
    _build_phase4_executor_batch_telemetry,
    _build_phase4_executor_substage_telemetry,
    _build_phase4_normalization_stats,
    _compute_row_denominator_scaled_l1,
    _copy_feature_rows_to_cpu_staging,
    _FileBackedFeatureRowStore,
    _build_phase4_planner_v2_candidate_window,
    _build_phase4_refresh_substage_telemetry,
    _select_phase4_planner_v2_membership,
    _apply_phase4_planner_v2_refresh_plan,
    _build_phase4_deterministic_shadow_pending,
    _build_phase4_cutoff_debug,
    _build_phase4_probe_pending_frontier,
    _build_phase3_seed_bundle_payload,
    _load_phase0_donor_bundle_npz,
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
    _compute_phase4_refresh_queue_window_size,
    _plan_phase4_frontier_membership_preserving_v1,
    _reorder_pending_for_phase4_locality,
    _resolve_internal_dtype_map,
    _resolve_internal_precision_requested,
    _resolve_telemetry_max_events,
    _resolve_phase0_activation_threshold_compare_mode,
    _resolve_phase0_donor_context_policy,
    _resolve_phase0_replay_mode,
    _resolve_phase4_anomaly_debug_enabled,
    _resolve_phase4_feature_batch_planner_status,
    _hash_sparse_membership_indices,
    _resolve_phase4_refresh_optimization_mode,
    _resolve_phase4_refresh_optimization_config,
    _build_phase4_refresh_optimization_metadata,
    _resolve_phase1_trace_batch_policy,
    _resolve_phase1_trace_batch_size_max,
    _resolve_phase1_trace_batch_config,
    _build_phase1_trace_batch_metadata,
    _resolve_phase1_trace_batch_sizing,
    _build_phase1_trace_batch_sizing_metadata,
    _resolve_phase4_refresh_policy,
    _resolve_phase4_refresh_interval_multiplier,
    _resolve_phase4_refresh_policy_config,
    _build_phase4_refresh_policy_metadata,
    _resolve_phase4_ranker,
    _resolve_phase4_ranker_config,
    _build_phase4_ranker_metadata,
    _select_phase4_frontier_rank_selection,
    _resolve_row_store_cache_control,
    _resolve_row_store_cache_control_config,
    _build_row_store_cache_control_metadata,
    _resolve_exact_encoder_residency,
    _resolve_exact_encoder_residency_config,
    _build_exact_encoder_residency_metadata,
    _resolve_phase4_row_executor_mode,
    _resolve_phase4_row_executor_config,
    _resolve_phase4_streaming_v1_microbatch_size,
    _build_phase4_row_executor_metadata,
    _resolve_phase4_row_reduction_config,
    _resolve_phase4_row_reduction_mode,
    _build_phase4_row_reduction_metadata,
    _build_phase4_gpu_row_reduction_transfer_telemetry,
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


def test_nnsight_context_replace_phase0_activation_state_refreshes_chunked_state() -> None:
    ctx, provider = _make_chunked_context(NNSightAttributionContext, enable_cache=True)
    assert ctx.decoder_chunk_cache is not None

    donor_activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 2], [1, 0], [1, 0]], dtype=torch.int64),
        values=torch.tensor([4.0, -3.0], dtype=torch.float32),
        size=(3, 2, 2),
        check_invariants=True,
    ).coalesce()

    stats = ctx.replace_phase0_activation_state(donor_activation_matrix)

    assert stats["old_active_feature_count"] == 3
    assert stats["new_active_feature_count"] == 2
    assert ctx._row_size == donor_activation_matrix._nnz() + (3 + 1) * 2
    assert provider.clear_calls >= 1
    assert ctx.decoder_chunk_cache is not None
    assert torch.equal(ctx.activation_matrix.indices(), donor_activation_matrix.indices())
    assert torch.equal(ctx.activation_matrix.values(), donor_activation_matrix.values())
    assert ctx._chunked_layer_spans == [(0, 1), None, (1, 2)]
    assert ctx.chunked_decoder_state is not None
    assert torch.equal(
        ctx.chunked_decoder_state["source_layers"],
        donor_activation_matrix.indices()[0],
    )
    assert torch.equal(
        ctx.chunked_decoder_state["positions"],
        donor_activation_matrix.indices()[1],
    )
    assert torch.equal(
        ctx.chunked_decoder_state["feature_ids"],
        donor_activation_matrix.indices()[2],
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


def test_exact_chunked_active_cpu_encoder_residency_stages_materialized_table() -> None:
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
        encoder_vecs=encoder_vecs,
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
        stage_encoder_vecs_on_cpu=False,
        exact_encoder_residency="active_cpu",
        materialized_encoder_vecs_during_phase0=True,
    )

    assert ctx.encoder_vecs.device.type == "cpu"
    assert ctx.exact_encoder_residency_requested == "active_cpu"
    assert ctx.exact_encoder_residency_effective == "active_cpu"
    assert ctx.exact_encoder_staging_destination == "cpu"
    assert ctx.exact_encoder_materialized_during_phase0 is True
    assert ctx.exact_encoder_pinned_requested is False
    assert ctx.exact_encoder_pinned_effective is False


def test_exact_chunked_active_pinned_cpu_encoder_residency_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        values=torch.tensor([1.0, 2.0, 3.0]),
        size=(2, 2, 2),
        check_invariants=True,
    ).coalesce()
    encoder_vecs = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def _stage_with_forced_pin_failure(tensor: torch.Tensor, *, pin_memory: bool):
        staged = NNSightAttributionContext._stage_tensor_on_cpu(tensor)
        assert pin_memory is True
        return staged, False, "RuntimeError: pinning unavailable"

    monkeypatch.setattr(
        NNSightAttributionContext,
        "_stage_encoder_tensor",
        staticmethod(_stage_with_forced_pin_failure),
    )

    ctx = NNSightAttributionContext(
        activation_matrix=activation_matrix,
        error_vectors=torch.zeros(2, 2, 4),
        token_vectors=torch.zeros(2, 4),
        decoder_vecs=torch.empty((0, 4)),
        encoder_vecs=encoder_vecs,
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
        exact_encoder_residency="active_pinned_cpu",
        materialized_encoder_vecs_during_phase0=True,
    )

    assert ctx.encoder_vecs.device.type == "cpu"
    assert ctx.exact_encoder_residency_requested == "active_pinned_cpu"
    assert ctx.exact_encoder_residency_effective == "active_pinned_cpu"
    assert ctx.exact_encoder_pinned_requested is True
    assert ctx.exact_encoder_pinned_effective is False
    assert ctx.exact_encoder_pinning_success is False
    assert ctx.exact_encoder_pinning_failure_reason is not None
    assert ctx.exact_encoder_staging_destination == "cpu"


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
    config = _resolve_phase4_refresh_optimization_config(
        "v1",
        compact_output=True,
        exact_chunked_decoder=True,
    )
    metadata = _build_phase4_refresh_optimization_metadata(config)

    assert metadata["refresh_optimization_requested"] == "v1"
    assert metadata["refresh_optimization_mode_requested"] == "v1"
    assert metadata["refresh_optimization"] == "v1"
    assert metadata["refresh_optimization_effective"] == "v1"
    assert metadata["refresh_optimization_mode_effective"] == "v1"
    assert metadata["refresh_optimization_reference_execution"] is False


def test_phase4_refresh_optimization_falls_back_off_when_compact_refresh_unavailable() -> None:
    config = _resolve_phase4_refresh_optimization_config(
        "v1",
        compact_output=False,
        exact_chunked_decoder=False,
    )
    metadata = _build_phase4_refresh_optimization_metadata(config)

    assert metadata["refresh_optimization_requested"] == "v1"
    assert metadata["refresh_optimization"] == "v1"
    assert metadata["refresh_optimization_effective"] == "off"
    assert metadata["refresh_optimization_mode_effective"] == "off"
    assert metadata["refresh_optimization_effective_version"] == "off_v1"
    assert metadata["refresh_optimization_reference_execution"] is True
    assert metadata["refresh_optimization_effective_behavior"] == "off_reference_execution"


def test_phase1_trace_batch_policy_config_validates_and_tracks_effective_behavior() -> None:
    assert _resolve_phase1_trace_batch_policy("legacy") == "legacy"
    assert _resolve_phase1_trace_batch_policy("cap_effective_batches") == "cap_effective_batches"
    assert _resolve_phase1_trace_batch_size_max(None) is None
    assert _resolve_phase1_trace_batch_size_max(64) == 64
    with pytest.raises(ValueError, match="phase1_trace_batch_policy must be one of"):
        _resolve_phase1_trace_batch_policy("cap")
    with pytest.raises(ValueError, match="phase1_trace_batch_size_max must be > 0"):
        _resolve_phase1_trace_batch_size_max(0)

    config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=64,
    )
    metadata = _build_phase1_trace_batch_metadata(config)
    assert metadata["trace_batch_policy_requested"] == "cap_effective_batches"
    assert metadata["trace_batch_policy_effective"] == "cap_effective_batches"
    assert metadata["trace_batch_policy_default"] == "legacy"
    assert metadata["trace_batch_policy_reference_execution"] is False
    assert metadata["trace_batch_size_max_requested"] == 64
    assert metadata["trace_batch_size_max_effective"] == 64
    assert metadata["trace_batch_size_max_default"] is None
    assert metadata["trace_batch_size_max_reference_execution"] is False
    assert metadata["trace_batch_policy_fallback_reason"] is None

    cap_missing_max = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=None,
    )
    cap_missing_max_metadata = _build_phase1_trace_batch_metadata(cap_missing_max)
    assert cap_missing_max_metadata["trace_batch_policy_effective"] == "legacy"
    assert cap_missing_max_metadata["trace_batch_policy_reference_execution"] is True
    assert (
        cap_missing_max_metadata["trace_batch_policy_effective_behavior"]
        == "legacy_fallback_missing_batch_size_max"
    )
    assert (
        cap_missing_max_metadata["trace_batch_policy_fallback_reason"]
        == "cap_effective_batches requested without phase1_trace_batch_size_max; "
        "falling back to legacy execution"
    )
    cap_missing_max_sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=128,
        feature_batch_size=None,
        logit_batch_size=None,
        feature_batch_size_max=None,
        phase1_trace_batch_config=cap_missing_max,
    )
    cap_missing_max_sizing_metadata = _build_phase1_trace_batch_sizing_metadata(
        cap_missing_max_sizing
    )
    assert cap_missing_max_sizing_metadata["trace_batch_cap_applied"] is False
    assert (
        cap_missing_max_sizing_metadata["trace_batch_cap_reason"]
        == "cap_effective_batches_fallback_missing_batch_size_max"
    )

    legacy_with_cap = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="legacy",
        phase1_trace_batch_size_max=64,
    )
    legacy_with_cap_metadata = _build_phase1_trace_batch_metadata(legacy_with_cap)
    assert legacy_with_cap_metadata["trace_batch_policy_effective"] == "legacy"
    assert legacy_with_cap_metadata["trace_batch_size_max_effective"] is None
    assert legacy_with_cap_metadata["trace_batch_size_max_reference_execution"] is True


def test_phase1_trace_batch_sizing_caps_source_batch_only() -> None:
    config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=64,
    )
    sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=128,
        feature_batch_size=None,
        logit_batch_size=96,
        feature_batch_size_max=200,
        phase1_trace_batch_config=config,
    )
    metadata = _build_phase1_trace_batch_sizing_metadata(sizing)

    assert metadata["source_batch_size_requested"] == 128
    assert metadata["source_batch_size_effective"] == 64
    assert metadata["source_batch_size_cap_applied"] is True
    assert metadata["feature_batch_size_requested"] == 128
    assert metadata["feature_batch_size_defaulted"] is True
    assert metadata["feature_batch_size_effective"] == 128
    assert metadata["feature_batch_size_cap_applied"] is False
    assert metadata["logit_batch_size_requested"] == 96
    assert metadata["logit_batch_size_effective"] == 96
    assert metadata["logit_batch_size_cap_applied"] is False
    assert metadata["phase4_feature_batch_size_max_requested"] == 200
    assert metadata["phase4_feature_batch_size_max_effective"] == 200
    assert metadata["phase4_feature_batch_size_max_cap_applied"] is False
    assert metadata["trace_batch_size_legacy"] == 128
    assert metadata["trace_batch_size_effective_pre_planner"] == 64
    assert metadata["trace_batch_size_cap_applied"] is True
    assert metadata["trace_batch_cap_applied"] is True
    assert metadata["trace_batch_cap_reason"] == "cap_effective_batches_applied"


def test_phase1_trace_batch_sizing_cap_does_not_reduce_later_phase_batches() -> None:
    config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=64,
    )
    sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=128,
        feature_batch_size=192,
        logit_batch_size=160,
        feature_batch_size_max=256,
        phase1_trace_batch_config=config,
    )

    assert sizing.effective_source_batch_size == 64
    assert sizing.effective_feature_batch_size == 192
    assert sizing.effective_logit_batch_size == 160
    assert sizing.effective_phase4_max_feature_batch_size == 256


def test_phase1_trace_batch_sizing_legacy_policy_ignores_requested_cap() -> None:
    config = _resolve_phase1_trace_batch_config(
        phase1_trace_batch_policy="legacy",
        phase1_trace_batch_size_max=64,
    )
    sizing = _resolve_phase1_trace_batch_sizing(
        batch_size=128,
        feature_batch_size=80,
        logit_batch_size=None,
        feature_batch_size_max=256,
        phase1_trace_batch_config=config,
    )
    metadata = _build_phase1_trace_batch_sizing_metadata(sizing)

    assert metadata["source_batch_size_effective"] == 128
    assert metadata["feature_batch_size_effective"] == 80
    assert metadata["logit_batch_size_effective"] == 128
    assert metadata["phase4_feature_batch_size_max_effective"] == 256
    assert metadata["trace_batch_cap_applied"] is False
    assert metadata["trace_batch_cap_reason"] == "legacy_policy_ignores_phase1_trace_batch_size_max"


def test_phase4_refresh_policy_config_validates_and_activates_deferred_when_applicable() -> None:
    assert _resolve_phase4_refresh_policy("standard") == "standard"
    assert _resolve_phase4_refresh_policy("deferred_v1") == "deferred_v1"
    assert _resolve_phase4_refresh_interval_multiplier(3) == 3
    with pytest.raises(ValueError, match="phase4_refresh_policy must be one of"):
        _resolve_phase4_refresh_policy("deferred")
    with pytest.raises(ValueError, match="phase4_refresh_interval_multiplier must be > 0"):
        _resolve_phase4_refresh_interval_multiplier(0)

    config = _resolve_phase4_refresh_policy_config(
        phase4_refresh_policy="deferred_v1",
        phase4_refresh_interval_multiplier=3,
        compact_output=True,
        exact_chunked_decoder=True,
    )
    metadata = _build_phase4_refresh_policy_metadata(config)
    assert metadata["refresh_policy_requested"] == "deferred_v1"
    assert metadata["refresh_policy_effective"] == "deferred_v1"
    assert metadata["refresh_policy_default"] == "standard"
    assert metadata["refresh_policy_reference_execution"] is False
    assert metadata["refresh_policy_applicable"] is True
    assert metadata["refresh_policy_fallback_reason"] is None
    assert metadata["refresh_interval_multiplier_requested"] == 3
    assert metadata["refresh_interval_multiplier_effective"] == 3
    assert metadata["refresh_interval_multiplier_default"] == 1
    assert metadata["refresh_interval_multiplier_reference_execution"] is False
    assert metadata["refresh_queue_multiplier_effective"] == 3


def test_phase4_refresh_policy_config_falls_back_when_deferred_path_unavailable() -> None:
    config = _resolve_phase4_refresh_policy_config(
        phase4_refresh_policy="deferred_v1",
        phase4_refresh_interval_multiplier=3,
        compact_output=False,
        exact_chunked_decoder=False,
    )
    metadata = _build_phase4_refresh_policy_metadata(config)

    assert metadata["refresh_policy_requested"] == "deferred_v1"
    assert metadata["refresh_policy_effective"] == "standard"
    assert metadata["refresh_policy_reference_execution"] is True
    assert metadata["refresh_policy_applicable"] is False
    assert metadata["refresh_policy_fallback_reason"] is not None
    assert metadata["refresh_interval_multiplier_requested"] == 3
    assert metadata["refresh_interval_multiplier_effective"] == 1
    assert metadata["refresh_queue_multiplier_effective"] == 1
    assert metadata["refresh_interval_multiplier_reference_execution"] is True

    standard_nondefault = _resolve_phase4_refresh_policy_config(
        phase4_refresh_policy="standard",
        phase4_refresh_interval_multiplier=3,
        compact_output=True,
        exact_chunked_decoder=True,
    )
    standard_nondefault_metadata = _build_phase4_refresh_policy_metadata(standard_nondefault)
    assert standard_nondefault_metadata["refresh_policy_effective"] == "standard"
    assert standard_nondefault_metadata["refresh_interval_multiplier_effective"] == 1
    assert standard_nondefault_metadata["refresh_queue_multiplier_effective"] == 1
    assert standard_nondefault_metadata["refresh_interval_multiplier_reference_execution"] is True


def test_phase4_refresh_queue_window_size_scales_with_multiplier_and_reduces_refreshes() -> None:
    feature_count = 100
    standard_queue = _compute_phase4_refresh_queue_window_size(
        update_interval=2,
        phase4_feature_batch_size=8,
        queue_multiplier=1,
    )
    deferred_queue = _compute_phase4_refresh_queue_window_size(
        update_interval=2,
        phase4_feature_batch_size=8,
        queue_multiplier=3,
    )

    assert standard_queue == 16
    assert deferred_queue == 48

    standard_refreshes = (feature_count + standard_queue - 1) // standard_queue
    deferred_refreshes = (feature_count + deferred_queue - 1) // deferred_queue
    assert deferred_refreshes < standard_refreshes


def test_phase4_ranker_config_validates_and_tracks_effective_mode() -> None:
    assert _resolve_phase4_ranker("argsort") == "argsort"
    assert _resolve_phase4_ranker("topk_v1") == "topk_v1"
    with pytest.raises(ValueError, match="phase4_ranker must be one of"):
        _resolve_phase4_ranker("topk")

    config = _resolve_phase4_ranker_config("topk_v1")
    metadata = _build_phase4_ranker_metadata(config)
    assert metadata["ranker_requested"] == "topk_v1"
    assert metadata["ranker_effective"] == "topk_v1"
    assert metadata["ranker_default"] == "argsort"
    assert metadata["ranker_reference_execution"] is False


def test_phase4_ranker_topk_matches_argsort_frontier_without_ties() -> None:
    feature_influences = torch.tensor([0.9, 0.2, 0.8, 0.7, 0.1, 0.6], dtype=torch.float64)
    visited = torch.tensor([False, True, False, False, False, False], dtype=torch.bool)

    argsort_selection = _select_phase4_frontier_rank_selection(
        feature_influences=feature_influences,
        visited=visited,
        frontier_size=3,
        ranker_mode="argsort",
    )
    topk_selection = _select_phase4_frontier_rank_selection(
        feature_influences=feature_influences,
        visited=visited,
        frontier_size=3,
        ranker_mode="topk_v1",
    )

    assert torch.equal(topk_selection.selected_frontier, argsort_selection.selected_frontier)
    assert topk_selection.selected_order_hash == argsort_selection.selected_order_hash
    assert topk_selection.tie_at_cutoff is False


def test_phase4_ranker_topk_tie_metadata_documents_cutoff_membership_behavior() -> None:
    feature_influences = torch.tensor([1.0, 0.5, 0.5, 0.1], dtype=torch.float64)
    visited = torch.zeros(4, dtype=torch.bool)

    argsort_selection = _select_phase4_frontier_rank_selection(
        feature_influences=feature_influences,
        visited=visited,
        frontier_size=2,
        ranker_mode="argsort",
    )
    topk_selection = _select_phase4_frontier_rank_selection(
        feature_influences=feature_influences,
        visited=visited,
        frontier_size=2,
        ranker_mode="topk_v1",
    )

    topk_set = set(topk_selection.selected_frontier.tolist())
    assert topk_selection.selected_count == 2
    assert 0 in topk_set
    assert topk_set.issubset({0, 1, 2})
    assert topk_selection.tie_count_at_cutoff >= 2
    assert topk_selection.tie_at_cutoff is True
    assert "ties at the cutoff" in topk_selection.tie_behavior
    assert "argsort" in argsort_selection.tie_behavior


def test_row_store_cache_control_config_validates_and_tracks_effective_mode() -> None:
    assert _resolve_row_store_cache_control("off") == "off"
    assert (
        _resolve_row_store_cache_control("fadvise_dontneed_after_append_v1")
        == "fadvise_dontneed_after_append_v1"
    )
    assert (
        _resolve_row_store_cache_control("fadvise_dontneed_after_append_and_read_v1")
        == "fadvise_dontneed_after_append_and_read_v1"
    )
    with pytest.raises(ValueError, match="row_store_cache_control must be one of"):
        _resolve_row_store_cache_control("fadvise")

    config = _resolve_row_store_cache_control_config(
        "fadvise_dontneed_after_append_v1",
        compact_output=True,
        exact_chunked_decoder=True,
    )
    metadata = _build_row_store_cache_control_metadata(config)
    assert metadata["row_store_cache_control_requested"] == "fadvise_dontneed_after_append_v1"
    assert metadata["row_store_cache_control_effective"] == "fadvise_dontneed_after_append_v1"
    assert metadata["row_store_cache_control_default"] == "off"
    assert metadata["row_store_cache_control_reference_execution"] is False
    assert metadata["row_store_cache_control_applicable"] is True
    assert metadata["row_store_cache_control_fallback_reason"] is None

    read_config = _resolve_row_store_cache_control_config(
        "fadvise_dontneed_after_append_and_read_v1",
        compact_output=True,
        exact_chunked_decoder=True,
    )
    read_metadata = _build_row_store_cache_control_metadata(read_config)
    assert (
        read_metadata["row_store_cache_control_effective"]
        == "fadvise_dontneed_after_append_and_read_v1"
    )

    fallback_config = _resolve_row_store_cache_control_config(
        "fadvise_dontneed_after_append_v1",
        compact_output=False,
        exact_chunked_decoder=False,
    )
    fallback_metadata = _build_row_store_cache_control_metadata(fallback_config)
    assert fallback_metadata["row_store_cache_control_effective"] == "off"
    assert fallback_metadata["row_store_cache_control_reference_execution"] is True
    assert fallback_metadata["row_store_cache_control_applicable"] is False
    assert fallback_metadata["row_store_cache_control_fallback_reason"] is not None


def test_exact_encoder_residency_config_validates_tracks_effective_mode_and_fallback() -> None:
    assert _resolve_exact_encoder_residency("lazy") == "lazy"
    assert _resolve_exact_encoder_residency("active_cpu") == "active_cpu"
    assert _resolve_exact_encoder_residency("active_pinned_cpu") == "active_pinned_cpu"
    with pytest.raises(ValueError, match="exact_encoder_residency must be one of"):
        _resolve_exact_encoder_residency("active")

    config = _resolve_exact_encoder_residency_config(
        "active_pinned_cpu",
        exact_chunked_decoder=True,
    )
    metadata = _build_exact_encoder_residency_metadata(config)
    assert metadata["exact_encoder_residency_requested"] == "active_pinned_cpu"
    assert metadata["exact_encoder_residency_effective"] == "active_pinned_cpu"
    assert metadata["exact_encoder_residency_applicable"] is True
    assert metadata["exact_encoder_residency_fallback_reason"] is None
    assert metadata["exact_encoder_materialize_phase0"] is True
    assert metadata["exact_encoder_staging_destination_planned"] == "pinned_cpu"
    assert metadata["exact_encoder_pinned_requested"] is True
    assert metadata["exact_encoder_pinned_planned"] is True
    assert metadata["exact_encoder_pinned_effective"] is None
    assert metadata["exact_encoder_pinning_success"] is None
    assert metadata["exact_encoder_residency_default"] == "lazy"
    assert metadata["exact_encoder_residency_reference_execution"] is False

    fallback_config = _resolve_exact_encoder_residency_config(
        "active_pinned_cpu",
        exact_chunked_decoder=False,
    )
    fallback_metadata = _build_exact_encoder_residency_metadata(fallback_config)
    assert fallback_metadata["exact_encoder_residency_effective"] == "lazy"
    assert fallback_metadata["exact_encoder_residency_applicable"] is False
    assert fallback_metadata["exact_encoder_residency_fallback_reason"] is not None
    assert fallback_metadata["exact_encoder_materialize_phase0"] is False
    assert fallback_metadata["exact_encoder_pinned_requested"] is True
    assert fallback_metadata["exact_encoder_pinned_planned"] is False
    assert fallback_metadata["exact_encoder_pinned_effective"] is None
    assert fallback_metadata["exact_encoder_residency_reference_execution"] is True


def test_phase4_row_executor_mode_resolves_and_rejects_unknown() -> None:
    assert _resolve_phase4_row_executor_mode("batched") == "batched"
    assert _resolve_phase4_row_executor_mode("streaming_v1") == "streaming_v1"
    with pytest.raises(ValueError, match="phase4_row_executor must be one of"):
        _resolve_phase4_row_executor_mode("streaming_v2")


def test_phase4_row_executor_metadata_tracks_requested_and_effective_modes() -> None:
    config = _resolve_phase4_row_executor_config(
        "streaming_v1",
        compact_output=True,
        exact_chunked_decoder=True,
    )
    metadata = _build_phase4_row_executor_metadata(config)

    assert metadata["row_executor_requested"] == "streaming_v1"
    assert metadata["row_executor_mode_requested"] == "streaming_v1"
    assert metadata["row_executor"] == "streaming_v1"
    assert metadata["row_executor_effective"] == "streaming_v1"
    assert metadata["row_executor_mode_effective"] == "streaming_v1"
    assert metadata["row_executor_reference_execution"] is False


def test_phase4_row_executor_falls_back_to_batched_when_streaming_path_unavailable() -> None:
    config = _resolve_phase4_row_executor_config(
        "streaming_v1",
        compact_output=False,
        exact_chunked_decoder=False,
    )
    metadata = _build_phase4_row_executor_metadata(config)

    assert metadata["row_executor_requested"] == "streaming_v1"
    assert metadata["row_executor_effective"] == "batched"
    assert metadata["row_executor_mode_effective"] == "batched"
    assert metadata["row_executor_effective_version"] == "batched_v1"
    assert metadata["row_executor_reference_execution"] is True
    assert metadata["row_executor_effective_behavior"] == "batched_reference_execution"


def test_phase4_row_reduction_off_metadata_and_gpu_v1_rejection() -> None:
    assert _resolve_phase4_row_reduction_mode("off") == "off"
    assert _resolve_phase4_row_reduction_mode("gpu_v1") == "gpu_v1"
    with pytest.raises(ValueError, match="phase4_row_reduction must be one of"):
        _resolve_phase4_row_reduction_mode("gpu_v2")

    metadata = _build_phase4_row_reduction_metadata(
        _resolve_phase4_row_reduction_config(
            "off",
            compact_output=True,
            exact_chunked_decoder=True,
        )
    )
    assert metadata["row_reduction_requested"] == "off"
    assert metadata["row_reduction_effective"] == "off"
    assert metadata["row_reduction_mode_effective"] == "off"
    assert metadata["row_reduction_reference_execution"] is False

    gpu_metadata = _build_phase4_row_reduction_metadata(
        _resolve_phase4_row_reduction_config(
            "gpu_v1",
            compact_output=True,
            exact_chunked_decoder=True,
        )
    )
    assert gpu_metadata["row_reduction_requested"] == "gpu_v1"
    assert gpu_metadata["row_reduction_effective"] == "gpu_v1"
    assert gpu_metadata["row_reduction_effective_version"] == "gpu_v1_staged"
    assert gpu_metadata["row_reduction_reference_execution"] is False

    fallback_metadata = _build_phase4_row_reduction_metadata(
        _resolve_phase4_row_reduction_config(
            "gpu_v1",
            compact_output=False,
            exact_chunked_decoder=True,
        )
    )
    assert fallback_metadata["row_reduction_requested"] == "gpu_v1"
    assert fallback_metadata["row_reduction_effective"] == "off"
    assert fallback_metadata["row_reduction_reference_execution"] is True


def test_phase4_gpu_row_reduction_transfer_telemetry_counts_compact_bytes() -> None:
    rows = torch.zeros((2, 10), dtype=torch.float32)
    feature_rows = rows[:, :3].contiguous()
    row_abs_max = torch.zeros(2, dtype=torch.float64)
    row_l1_scaled = torch.zeros(2, dtype=torch.float64)

    telemetry = _build_phase4_gpu_row_reduction_transfer_telemetry(
        rows=rows,
        feature_row_slice=feature_rows,
        row_abs_max=row_abs_max,
        row_l1_scaled=row_l1_scaled,
    )

    assert telemetry["row_reduction_backend"] == "gpu_v1"
    assert telemetry["row_reduction_baseline_full_row_transfer_bytes"] == 80
    assert telemetry["row_reduction_compact_transfer_bytes"] == 56
    assert telemetry["row_reduction_gpu_to_cpu_bytes_saved"] == 24
    assert telemetry["row_transfer_bytes"] == 56


def test_copy_feature_rows_to_cpu_staging_reuses_sized_buffer() -> None:
    rows = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    feature_rows, staging = _copy_feature_rows_to_cpu_staging(
        rows,
        total_active_feats=3,
        staging_buffer=None,
        dtype=torch.float64,
    )

    assert staging is not None
    assert feature_rows.device.type == "cpu"
    assert feature_rows.is_contiguous()
    assert torch.equal(feature_rows, rows[:, :3].to(torch.float64))

    larger = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    feature_rows_again, staging_again = _copy_feature_rows_to_cpu_staging(
        larger,
        total_active_feats=3,
        staging_buffer=staging,
        dtype=torch.float64,
    )

    assert staging_again is not staging
    assert torch.equal(feature_rows_again, larger[:, :3].to(torch.float64))

    smaller = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    feature_rows_smaller, staging_smaller = _copy_feature_rows_to_cpu_staging(
        smaller,
        total_active_feats=3,
        staging_buffer=staging_again,
        dtype=torch.float64,
    )

    assert staging_smaller is staging_again
    assert torch.equal(feature_rows_smaller, smaller[:, :3].to(torch.float64))


def test_file_backed_feature_row_store_append_rows_returns_substage_telemetry() -> None:
    store = _FileBackedFeatureRowStore(
        n_rows=2,
        n_feature_columns=3,
        dtype=torch.float32,
    )
    try:
        telemetry = store.append_rows(
            row_start=0,
            feature_rows=torch.ones((2, 3), dtype=torch.float32),
            row_denominator_scaled_l1=(
                torch.ones(2, dtype=torch.float32),
                torch.ones(2, dtype=torch.float32),
            ),
        )
    finally:
        store.cleanup()

    expected_keys = {
        "row_store_append_cpu_prepare_elapsed_ms",
        "row_store_append_contiguous_elapsed_ms",
        "row_store_append_numpy_elapsed_ms",
        "row_store_append_pwrite_elapsed_ms",
        "row_store_append_denominator_copy_elapsed_ms",
        "row_store_append_total_elapsed_ms",
    }
    assert expected_keys <= telemetry.keys()
    assert all(isinstance(telemetry[key], float) for key in expected_keys)


def test_row_denominator_scaled_l1_preserve_device_matches_cpu_path() -> None:
    rows = torch.tensor([[1.0, -3.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    cpu_max, cpu_scaled = _compute_row_denominator_scaled_l1(rows, dtype=torch.float64)
    same_device_max, same_device_scaled = _compute_row_denominator_scaled_l1(
        rows,
        dtype=torch.float64,
        preserve_device=True,
    )

    assert same_device_max.device == rows.device
    assert same_device_scaled.device == rows.device
    assert torch.equal(cpu_max, same_device_max)
    assert torch.equal(cpu_scaled, same_device_scaled)


def test_phase4_streaming_v1_microbatch_size_is_capped() -> None:
    assert _resolve_phase4_streaming_v1_microbatch_size(1) == 1
    assert _resolve_phase4_streaming_v1_microbatch_size(64) == 64
    assert _resolve_phase4_streaming_v1_microbatch_size(256) == 64


def test_phase4_executor_batch_telemetry_separates_scheduler_and_microbatch_counts() -> None:
    telemetry = _build_phase4_executor_batch_telemetry(
        scheduler_reference_batch_index=2,
        scheduler_reference_batch_count=4,
        scheduler_reference_batch_rows=128,
        executor_microbatch_index=9,
        executor_microbatch_count=12,
        executor_configured_reference_batch_size=128,
        executor_microbatch_rows=32,
        executor_microbatch_size=32,
    )

    assert telemetry["phase4_batch_count"] == 4
    assert telemetry["phase4_batches"] == 4
    assert telemetry["phase4_executor_microbatch_count"] == 12
    assert telemetry["scheduler_reference_batch_index"] == 2
    assert telemetry["scheduler_reference_batch_rows"] == 128
    assert telemetry["executor_microbatch_index"] == 9
    assert telemetry["executor_microbatch_rows"] == 32
    assert telemetry["executor_configured_reference_batch_size"] == 128
    assert telemetry["executor_reference_batch_size"] == 128
    assert telemetry["executor_microbatch_size"] == 32


def test_phase4_executor_batch_telemetry_keeps_reference_and_microbatch_indices_separate() -> None:
    first = _build_phase4_executor_batch_telemetry(
        scheduler_reference_batch_index=0,
        scheduler_reference_batch_count=1,
        scheduler_reference_batch_rows=64,
        executor_microbatch_index=1,
        executor_microbatch_count=1,
        executor_configured_reference_batch_size=64,
        executor_microbatch_rows=32,
        executor_microbatch_size=32,
    )
    second = _build_phase4_executor_batch_telemetry(
        scheduler_reference_batch_index=1,
        scheduler_reference_batch_count=2,
        scheduler_reference_batch_rows=64,
        executor_microbatch_index=2,
        executor_microbatch_count=2,
        executor_configured_reference_batch_size=64,
        executor_microbatch_rows=32,
        executor_microbatch_size=32,
    )

    assert first["phase4_batch_count"] == 1
    assert second["phase4_batch_count"] == 2
    assert first["scheduler_reference_batch_index"] == 0
    assert second["scheduler_reference_batch_index"] == 1
    assert first["executor_microbatch_index"] == 1
    assert second["executor_microbatch_index"] == 2


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


def test_phase4_anomaly_debug_ignores_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHASE4_ANOMALY_DEBUG", "1")
    assert _resolve_phase4_anomaly_debug_enabled(False) is False


def test_telemetry_max_events_ignores_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CIRCUIT_TRACER_TELEMETRY_MAX_EVENTS", "17")
    assert (
        _resolve_telemetry_max_events(
            telemetry_max_events=None,
            compact_output=False,
            exact_chunked_decoder=False,
            profile=False,
            phase4_anomaly_debug_enabled=False,
        )
        == 20_000
    )


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


def test_internal_precision_defaults_to_exact_trace_internal_dtype() -> None:
    assert (
        _resolve_internal_precision_requested(
            None,
            exact_trace_internal_dtype=torch.float32,
        )
        == "float32"
    )
    assert (
        _resolve_internal_precision_requested(
            None,
            exact_trace_internal_dtype=torch.float64,
        )
        == "float64"
    )


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


def test_build_phase3_seed_bundle_payload_canonicalizes_cpu_tensors() -> None:
    payload = _build_phase3_seed_bundle_payload(
        active_features=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        activation_values=torch.tensor([0.25, -0.75], dtype=torch.float32),
        seed_feature_influences=torch.tensor([0.4, 0.2, 0.1], dtype=torch.float64),
        frontier_pre_locality=torch.tensor([2, 0], dtype=torch.int64),
        frontier_post_locality=torch.tensor([0, 2], dtype=torch.int64),
        queue_size=2,
        actual_max_feature_nodes=3,
        total_active_features=9,
        status="captured",
        planner_compute_dtype=torch.float64,
        influence_compute_dtype=torch.float64,
    )

    assert payload["status"] == "captured"
    assert payload["queue_size"] == 2
    assert payload["actual_max_feature_nodes"] == 3
    assert payload["total_active_features"] == 9
    assert payload["planner_compute_dtype"] == "float64"
    assert payload["influence_compute_dtype"] == "float64"
    assert torch.equal(
        payload["active_features"],
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
    )
    assert torch.equal(
        payload["activation_values"],
        torch.tensor([0.25, -0.75], dtype=torch.float32),
    )
    assert torch.equal(
        payload["seed_feature_influences"],
        torch.tensor([0.4, 0.2, 0.1], dtype=torch.float64),
    )
    assert torch.equal(
        payload["frontier_pre_locality"],
        torch.tensor([2, 0], dtype=torch.int64),
    )
    assert torch.equal(
        payload["frontier_post_locality"],
        torch.tensor([0, 2], dtype=torch.int64),
    )


def test_build_phase0_donor_bundle_payload_captures_hashes_and_metadata() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[1, 0], [0, 1], [3, 2]], dtype=torch.int64),
        values=torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
        size=(3, 2, 8),
        check_invariants=True,
    ).coalesce()

    payload = _build_phase0_donor_bundle_payload(
        activation_matrix=activation_matrix,
        input_tokens=torch.tensor([11, 22, 33], dtype=torch.int64),
        target_token_ids=torch.tensor([2, 4], dtype=torch.int64),
        target_probabilities=torch.tensor([0.7, 0.2], dtype=torch.float32),
        target_logits=torch.tensor([3.5, -1.0], dtype=torch.float32),
        transcoder_diagnostic_snapshot={
            "phase0_boundary_fingerprints": {
                "transcoder_constant_fingerprints": {"global_hash": "clt1234"}
            }
        },
        status="captured",
    )

    assert payload["schema_version"] == 1
    assert payload["replay_kind"] == "phase0_active_features_v1"
    assert payload["status"] == "captured"
    assert payload["activation_values_dtype"] == "bfloat16"
    assert payload["activation_matrix_shape"] == [3, 2, 8]
    assert payload["active_feature_count"] == 2
    assert payload["input_token_count"] == 3
    assert payload["target_count"] == 2
    assert payload["clt_constants_hash"] == "clt1234"
    assert isinstance(payload["active_feature_membership_hash_raw_order"], str)
    assert isinstance(payload["active_feature_membership_hash_canonical"], str)
    assert isinstance(payload["active_feature_values_hash"], str)
    assert isinstance(payload["input_tokens_hash"], str)
    assert isinstance(payload["target_token_ids_hash"], str)
    assert isinstance(payload["target_probability_hash"], str)
    assert isinstance(payload["target_logit_hash"], str)

    assert torch.equal(
        cast(torch.Tensor, payload["active_features"]),
        activation_matrix.indices().T.to(dtype=torch.int64),
    )
    assert torch.equal(
        cast(torch.Tensor, payload["activation_values"]),
        activation_matrix.values().to(dtype=torch.bfloat16),
    )
    assert torch.equal(
        cast(torch.Tensor, payload["active_feature_layer_counts"]),
        torch.tensor([1, 1, 0], dtype=torch.int64),
    )


def _write_phase0_donor_bundle_npz_for_test(
    path: Path,
    payload: dict[str, object],
    *,
    include_raw_uint16: bool,
) -> None:
    activation_values = cast(torch.Tensor, payload["activation_values"]).detach().cpu().contiguous()
    activation_values_dtype = str(payload.get("activation_values_dtype", "")).replace("torch.", "")
    if not activation_values_dtype:
        activation_values_dtype = str(activation_values.dtype).replace("torch.", "")

    activation_values_np = activation_values.to(dtype=torch.float32).numpy()
    activation_values_raw_uint16 = (
        activation_values.view(torch.uint16).numpy()
        if include_raw_uint16 and activation_values.dtype == torch.bfloat16
        else np.empty((0,), dtype=np.uint16)
    )

    np.savez_compressed(
        path,
        active_features=cast(torch.Tensor, payload["active_features"]).detach().cpu().numpy(),
        activation_values=activation_values_np,
        activation_values_dtype=np.array(activation_values_dtype),
        activation_values_raw_uint16=activation_values_raw_uint16,
        activation_matrix_shape=np.asarray(
            payload.get("activation_matrix_shape", []), dtype=np.int64
        ),
        active_feature_count=np.array(int(payload.get("active_feature_count", 0)), dtype=np.int64),
        active_feature_membership_hash_raw_order=np.array(
            str(payload.get("active_feature_membership_hash_raw_order", ""))
        ),
        active_feature_membership_hash_canonical=np.array(
            str(payload.get("active_feature_membership_hash_canonical", ""))
        ),
        active_feature_values_hash=np.array(str(payload.get("active_feature_values_hash", ""))),
        input_tokens=cast(torch.Tensor, payload["input_tokens"]).detach().cpu().numpy(),
        input_token_count=np.array(int(payload.get("input_token_count", 0)), dtype=np.int64),
        input_tokens_hash=np.array(str(payload.get("input_tokens_hash", ""))),
        target_token_ids=cast(torch.Tensor, payload["target_token_ids"]).detach().cpu().numpy(),
        target_count=np.array(int(payload.get("target_count", 0)), dtype=np.int64),
        target_token_ids_hash=np.array(str(payload.get("target_token_ids_hash", ""))),
        clt_constants_hash=np.array(str(payload.get("clt_constants_hash", ""))),
        schema_version=np.array(int(payload.get("schema_version", 0)), dtype=np.int64),
        replay_kind=np.array(str(payload.get("replay_kind", ""))),
        status=np.array(str(payload.get("status", ""))),
    )


def _build_phase0_payload_fixture() -> dict[str, object]:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[1, 0], [0, 1], [3, 2]], dtype=torch.int64),
        values=torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
        size=(3, 2, 8),
        check_invariants=True,
    ).coalesce()
    return _build_phase0_donor_bundle_payload(
        activation_matrix=activation_matrix,
        input_tokens=torch.tensor([11, 22, 33], dtype=torch.int64),
        target_token_ids=torch.tensor([2, 4], dtype=torch.int64),
        target_probabilities=torch.tensor([0.7, 0.2], dtype=torch.float32),
        target_logits=torch.tensor([3.5, -1.0], dtype=torch.float32),
        transcoder_diagnostic_snapshot={
            "phase0_boundary_fingerprints": {
                "transcoder_constant_fingerprints": {"global_hash": "clt1234"}
            }
        },
        status="captured",
    )


def test_load_phase0_donor_bundle_npz_reconstructs_bf16_from_raw_sidecar(
    tmp_path: Path,
) -> None:
    payload = _build_phase0_payload_fixture()
    bundle_path = tmp_path / "step_000_phase0_donor_bundle.npz"
    _write_phase0_donor_bundle_npz_for_test(bundle_path, payload, include_raw_uint16=True)

    loaded = _load_phase0_donor_bundle_npz(bundle_path, context_policy="strict")

    assert cast(torch.Tensor, loaded["activation_values"]).dtype == torch.bfloat16
    assert torch.equal(
        cast(torch.Tensor, loaded["activation_values"]),
        cast(torch.Tensor, payload["activation_values"]),
    )

    dtype_metadata = cast(dict[str, object], loaded["dtype_metadata"])
    validation_metadata = cast(dict[str, object], loaded["validation_metadata"])
    assert dtype_metadata["exact_bfloat16_reconstruction"] is True
    assert dtype_metadata["dtype_roundtrip_loss"] is False
    assert validation_metadata["validated"] is True
    assert validation_metadata["warnings"] == []


def test_load_phase0_donor_bundle_npz_without_raw_sidecar_tracks_roundtrip_loss(
    tmp_path: Path,
) -> None:
    payload = _build_phase0_payload_fixture()
    bundle_path = tmp_path / "step_000_phase0_donor_bundle_no_raw.npz"
    _write_phase0_donor_bundle_npz_for_test(bundle_path, payload, include_raw_uint16=False)

    loaded = _load_phase0_donor_bundle_npz(bundle_path, context_policy="strict")
    dtype_metadata = cast(dict[str, object], loaded["dtype_metadata"])

    assert cast(torch.Tensor, loaded["activation_values"]).dtype == torch.float32
    assert dtype_metadata["exact_bfloat16_reconstruction"] is False
    assert dtype_metadata["dtype_roundtrip_loss"] is True


def test_load_phase0_donor_bundle_npz_strict_fails_on_context_mismatch(tmp_path: Path) -> None:
    payload = _build_phase0_payload_fixture()
    bundle_path = tmp_path / "step_000_phase0_donor_bundle_strict_mismatch.npz"
    _write_phase0_donor_bundle_npz_for_test(bundle_path, payload, include_raw_uint16=True)

    with pytest.raises(ValueError, match="validation_context"):
        _load_phase0_donor_bundle_npz(
            bundle_path,
            context_policy="strict",
            validation_context={
                "input_tokens_hash": "deadbeef",
                "target_token_ids": torch.tensor([999, 1000], dtype=torch.int64),
                "active_feature_membership_hash_canonical": "badc0de",
            },
        )


def test_load_phase0_donor_bundle_npz_warn_policy_returns_warnings(tmp_path: Path) -> None:
    payload = _build_phase0_payload_fixture()
    bundle_path = tmp_path / "step_000_phase0_donor_bundle_warn_mismatch.npz"
    _write_phase0_donor_bundle_npz_for_test(bundle_path, payload, include_raw_uint16=True)

    loaded = _load_phase0_donor_bundle_npz(
        bundle_path,
        context_policy="warn",
        validation_context={
            "input_tokens_hash": "deadbeef",
            "target_token_ids_hash": "beaded",
        },
    )

    validation_metadata = cast(dict[str, object], loaded["validation_metadata"])
    warnings = cast(list[str], validation_metadata["warnings"])

    assert len(warnings) >= 1
    assert any("validation_context" in warning for warning in warnings)
    assert cast(torch.Tensor, loaded["active_features"]).shape == (2, 3)


def test_build_phase0_replay_validation_context_hashes_host_state() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0], [2, 3]], dtype=torch.int64),
        values=torch.tensor([0.5, -0.2], dtype=torch.float32),
        size=(2, 2, 8),
        check_invariants=True,
    ).coalesce()
    target_token_ids = torch.tensor([7, 8], dtype=torch.int64)
    validation_context = _build_phase0_replay_validation_context(
        input_tokens=torch.tensor([11, 22, 33], dtype=torch.int64),
        target_token_ids=target_token_ids,
        activation_matrix=activation_matrix,
        clt_constants_hash="clt1234",
    )

    assert validation_context["clt_constants_hash"] == "clt1234"
    assert validation_context["input_tokens_hash"]
    assert validation_context["target_token_ids_hash"]
    assert validation_context["active_feature_membership_hash_raw_order"]
    assert validation_context["active_feature_membership_hash_canonical"]


def test_build_phase0_activation_matrix_from_loaded_bundle_reconstructs_sparse_matrix() -> None:
    loaded_bundle = {
        "active_features": torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int64),
        "activation_values": torch.tensor([0.25, -0.75], dtype=torch.float32),
        "activation_matrix_shape": (2, 2, 8),
    }

    activation_matrix = _build_phase0_activation_matrix_from_loaded_bundle(
        loaded_bundle,
        device=torch.device("cpu"),
    )

    assert activation_matrix.shape == (2, 2, 8)
    assert activation_matrix._nnz() == 2
    assert torch.equal(
        activation_matrix.indices().T,
        cast(torch.Tensor, loaded_bundle["active_features"]),
    )
    assert torch.equal(
        activation_matrix.values(),
        cast(torch.Tensor, loaded_bundle["activation_values"]),
    )


def test_build_phase0_replay_metadata_tracks_warnings_and_dtype_loss() -> None:
    metadata = _build_phase0_replay_metadata(
        mode="donor_phase0",
        status="applied_with_warnings",
        donor_bundle_path="/tmp/step_000_phase0_donor_bundle.npz",
        context_policy="warn",
        validation_warnings=["target_token_ids_hash mismatch"],
        validation_failure_count=1,
        dtype_metadata={"dtype_roundtrip_loss": True},
        host_hashes={"input_tokens_hash": "abcd"},
        donor_hashes={"computed": {"input_tokens_hash": "efgh"}},
        host_active_feature_count=10,
        donor_active_feature_count=9,
    )

    assert metadata["status"] == "applied_with_warnings"
    assert metadata["mode"] == "donor_phase0"
    assert metadata["context_policy"] == "warn"
    assert metadata["donor_bundle_basename"] == "step_000_phase0_donor_bundle.npz"
    assert metadata["validation_warning_count"] == 1
    assert cast(dict[str, object], metadata["dtype_metadata"])["dtype_roundtrip_loss"] is True


def test_phase0_replay_mode_and_context_policy_resolution() -> None:
    assert _resolve_phase0_replay_mode("disabled") == "disabled"
    assert _resolve_phase0_replay_mode("donor_phase0") == "donor_phase0"
    assert _resolve_phase0_donor_context_policy("strict") == "strict"
    assert _resolve_phase0_donor_context_policy("warn") == "warn"

    with pytest.raises(ValueError, match="phase0_replay_mode"):
        _resolve_phase0_replay_mode("legacy")
    with pytest.raises(ValueError, match="phase0_donor_context_policy"):
        _resolve_phase0_donor_context_policy("ignore")


def test_build_feature_semantic_descriptors_payload_is_bounded_and_deterministic() -> None:
    payload = _build_feature_semantic_descriptors_payload(
        active_features=torch.tensor(
            [[0, 0, 7], [0, 1, 3], [1, 0, 4], [1, 1, 9], [2, 0, 5]],
            dtype=torch.int64,
        ),
        activation_values=torch.tensor([0.2, -0.5, 0.1, 0.9, -0.1], dtype=torch.float32),
        seed_feature_influences=torch.tensor([0.4, 0.2, 0.7, 0.1, 0.3], dtype=torch.float64),
        frontier_pre_locality=torch.tensor([2, 1], dtype=torch.int64),
        frontier_post_locality=torch.tensor([1, 4], dtype=torch.int64),
        total_active_features=5,
        status="captured",
        semantic_descriptor_top_k=3,
        semantic_descriptor_dim=8,
    )
    payload_again = _build_feature_semantic_descriptors_payload(
        active_features=torch.tensor(
            [[0, 0, 7], [0, 1, 3], [1, 0, 4], [1, 1, 9], [2, 0, 5]],
            dtype=torch.int64,
        ),
        activation_values=torch.tensor([0.2, -0.5, 0.1, 0.9, -0.1], dtype=torch.float32),
        seed_feature_influences=torch.tensor([0.4, 0.2, 0.7, 0.1, 0.3], dtype=torch.float64),
        frontier_pre_locality=torch.tensor([2, 1], dtype=torch.int64),
        frontier_post_locality=torch.tensor([1, 4], dtype=torch.int64),
        total_active_features=5,
        status="captured",
        semantic_descriptor_top_k=3,
        semantic_descriptor_dim=8,
    )

    assert payload["status"] == "captured"
    assert payload["descriptor_version"] == "v1"
    assert payload["descriptor_kind"] == "fallback_identity_metadata_v1"
    assert payload["descriptor_dim"] == 8
    assert payload["semantic_descriptor_top_k"] == 3
    assert payload["candidate_count"] == 3
    assert payload["total_active_features"] == 5
    assert cast(torch.Tensor, payload["candidate_features"]).shape == (3, 3)
    assert cast(torch.Tensor, payload["candidate_row_indices"]).shape == (3,)
    assert cast(torch.Tensor, payload["activation_value"]).shape == (3,)
    assert cast(torch.Tensor, payload["seed_influence"]).shape == (3,)
    assert cast(torch.Tensor, payload["seed_rank"]).shape == (3,)
    assert cast(torch.Tensor, payload["is_top_seed"]).dtype == torch.bool
    assert cast(torch.Tensor, payload["is_frontier_pre"]).dtype == torch.bool
    assert cast(torch.Tensor, payload["is_frontier_post"]).dtype == torch.bool
    assert cast(torch.Tensor, payload["frontier_pre_rank"]).dtype == torch.int64
    assert cast(torch.Tensor, payload["frontier_post_rank"]).dtype == torch.int64
    assert cast(torch.Tensor, payload["semantic_sketch"]).shape == (3, 8)
    assert cast(torch.Tensor, payload["semantic_sketch"]).dtype == torch.float32

    assert torch.equal(
        cast(torch.Tensor, payload["candidate_row_indices"]),
        cast(torch.Tensor, payload_again["candidate_row_indices"]),
    )
    assert torch.allclose(
        cast(torch.Tensor, payload["semantic_sketch"]),
        cast(torch.Tensor, payload_again["semantic_sketch"]),
    )


def test_build_feature_semantic_descriptors_payload_handles_missing_seed_scores() -> None:
    payload = _build_feature_semantic_descriptors_payload(
        active_features=torch.tensor([[0, 0, 1], [0, 1, 2], [1, 0, 3]], dtype=torch.int64),
        activation_values=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        seed_feature_influences=torch.empty(0, dtype=torch.float64),
        frontier_pre_locality=torch.tensor([1], dtype=torch.int64),
        frontier_post_locality=torch.tensor([2], dtype=torch.int64),
        total_active_features=3,
        status="skipped_all_features_included",
        semantic_descriptor_top_k=4,
        semantic_descriptor_dim=6,
    )

    assert payload["seed_influence_available"] is False
    assert torch.equal(
        cast(torch.Tensor, payload["seed_rank"]),
        torch.full((2,), -1, dtype=torch.int64),
    )
    assert torch.equal(
        cast(torch.Tensor, payload["is_top_seed"]),
        torch.zeros(2, dtype=torch.bool),
    )


def test_annotate_phase4_selection_on_feature_semantic_descriptors_marks_membership() -> None:
    payload = _build_feature_semantic_descriptors_payload(
        active_features=torch.tensor([[0, 0, 1], [0, 1, 2], [1, 0, 3]], dtype=torch.int64),
        activation_values=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        seed_feature_influences=torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64),
        frontier_pre_locality=torch.tensor([0, 1], dtype=torch.int64),
        frontier_post_locality=torch.tensor([1, 2], dtype=torch.int64),
        total_active_features=3,
        status="captured",
        semantic_descriptor_top_k=3,
        semantic_descriptor_dim=4,
    )

    _annotate_phase4_selection_on_feature_semantic_descriptors(
        payload,
        selected_features=torch.tensor([2, 0], dtype=torch.int64),
    )

    assert payload["phase4_selection_available"] is True
    assert torch.equal(
        cast(torch.Tensor, payload["is_selected_phase4"]),
        torch.tensor([True, False, True], dtype=torch.bool),
    )
    assert torch.equal(
        cast(torch.Tensor, payload["phase4_selected_rank"]),
        torch.tensor([1, -1, 0], dtype=torch.int64),
    )


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


def test_phase4_refresh_substage_telemetry_is_compact_in_summary_mode() -> None:
    telemetry = _build_phase4_refresh_substage_telemetry(
        telemetry_detail="summary",
        partial_influence_elapsed_ms=12.5,
        rank_topk_elapsed_ms=3.25,
        frontier_plan_elapsed_ms=1.75,
        row_store_read_elapsed_ms=6.0,
        influence_normalization_elapsed_ms=2.0,
        influence_matmul_elapsed_ms=1.5,
        chunk_request_count=7,
        active_row_chunk_count=5,
        row_reader_row_count=1024,
        solver_iteration_count=3,
    )

    assert telemetry == {
        "refresh_partial_influence_elapsed_ms": pytest.approx(12.5),
        "refresh_rank_topk_elapsed_ms": pytest.approx(3.25),
        "refresh_frontier_plan_elapsed_ms": pytest.approx(1.75),
    }


def test_phase4_refresh_substage_telemetry_includes_detailed_fields() -> None:
    telemetry = _build_phase4_refresh_substage_telemetry(
        telemetry_detail="normal",
        partial_influence_elapsed_ms=10.0,
        rank_topk_elapsed_ms=2.0,
        frontier_plan_elapsed_ms=1.0,
        row_store_read_elapsed_ms=4.0,
        influence_normalization_elapsed_ms=0.75,
        influence_matmul_elapsed_ms=0.5,
        chunk_request_count=9,
        active_row_chunk_count=6,
        row_reader_row_count=2048,
        solver_iteration_count=4,
        row_chunk_strategy="active_row_contiguous_chunks",
        row_weight_nonzero_row_count=768,
        row_weight_zero_row_count=1280,
        row_reader_overread_zero_row_count=0,
        active_row_range_count=6,
        streaming_chunk_reuse_stats={
            "active_row_scan_elapsed_ms_total": 0.1,
            "chunk_allocation_zero_fill_elapsed_ms_total": 0.2,
            "transfer_cast_abs_elapsed_ms_total": 0.3,
            "cache_lookup_elapsed_ms_total": 0.4,
            "cache_store_elapsed_ms_total": 0.5,
            "cache_eviction_elapsed_ms_total": 0.6,
            "row_weight_update_elapsed_ms_total": 0.7,
            "accounted_elapsed_ms_total": 8.8,
            "unaccounted_elapsed_ms_total": 1.2,
        },
        feature_row_store_read_stats={
            "prepared_read_cache_hit_count": 1,
            "prepared_read_cache_miss_count": 2,
            "prepared_read_cache_hit_row_count": 3,
            "prepared_read_cache_miss_row_count": 4,
            "prepared_read_cache_eviction_count": 5,
            "prepared_read_cache_invalidation_count": 6,
            "prepared_read_cache_invalidation_entry_count": 7,
            "prepared_read_cache_store_attempt_count": 8,
            "prepared_read_cache_store_success_count": 9,
            "prepared_read_cache_store_skip_disabled_count": 10,
            "prepared_read_cache_store_skip_too_large_count": 11,
            "prepared_read_cache_prepare_elapsed_ms_total": 12.5,
            "prepared_read_cache_entry_count": 13,
            "prepared_read_cache_nbytes": 14,
        },
    )

    assert telemetry["refresh_row_store_read_elapsed_ms"] == pytest.approx(4.0)
    assert telemetry["refresh_influence_normalization_elapsed_ms"] == pytest.approx(0.75)
    assert telemetry["refresh_influence_matmul_elapsed_ms"] == pytest.approx(0.5)
    assert telemetry["refresh_chunk_request_count"] == 9
    assert telemetry["refresh_active_row_chunk_count"] == 6
    assert telemetry["refresh_rows_touched"] == 2048
    assert telemetry["refresh_solver_iteration_count"] == 4
    assert telemetry["refresh_row_chunk_strategy"] == "active_row_contiguous_chunks"
    assert telemetry["refresh_row_weight_nonzero_rows"] == 768
    assert telemetry["refresh_row_weight_zero_rows"] == 1280
    assert telemetry["refresh_row_reader_overread_zero_rows"] == 0
    assert telemetry["refresh_active_row_range_count"] == 6
    assert telemetry["refresh_active_row_scan_elapsed_ms"] == pytest.approx(0.1)
    assert telemetry["refresh_chunk_allocation_zero_fill_elapsed_ms"] == pytest.approx(0.2)
    assert telemetry["refresh_transfer_cast_abs_elapsed_ms"] == pytest.approx(0.3)
    assert telemetry["refresh_cache_lookup_elapsed_ms"] == pytest.approx(0.4)
    assert telemetry["refresh_cache_store_elapsed_ms"] == pytest.approx(0.5)
    assert telemetry["refresh_cache_eviction_elapsed_ms"] == pytest.approx(0.6)
    assert telemetry["refresh_row_weight_update_elapsed_ms"] == pytest.approx(0.7)
    assert telemetry["refresh_accounted_elapsed_ms"] == pytest.approx(8.8)
    assert telemetry["refresh_unaccounted_elapsed_ms"] == pytest.approx(1.2)
    assert telemetry["feature_row_store_prepared_read_cache_hits"] == 1
    assert telemetry["feature_row_store_prepared_read_cache_misses"] == 2
    assert telemetry["feature_row_store_prepared_read_cache_hit_rows"] == 3
    assert telemetry["feature_row_store_prepared_read_cache_miss_rows"] == 4
    assert telemetry["feature_row_store_prepared_read_cache_evictions"] == 5
    assert telemetry["feature_row_store_prepared_read_cache_invalidations"] == 6
    assert telemetry["feature_row_store_prepared_read_cache_invalidation_entries"] == 7
    assert telemetry["feature_row_store_prepared_read_cache_store_attempts"] == 8
    assert telemetry["feature_row_store_prepared_read_cache_store_success"] == 9
    assert telemetry["feature_row_store_prepared_read_cache_store_skip_disabled"] == 10
    assert telemetry["feature_row_store_prepared_read_cache_store_skip_too_large"] == 11
    assert telemetry["feature_row_store_prepared_read_cache_prepare_elapsed_ms"] == pytest.approx(
        12.5
    )
    assert telemetry["feature_row_store_prepared_read_cache_entry_count"] == 13
    assert telemetry["feature_row_store_prepared_read_cache_nbytes"] == 14


def test_phase4_executor_substage_telemetry_summary_vs_normal() -> None:
    summary = _build_phase4_executor_substage_telemetry(
        telemetry_detail="summary",
        encoder_materialize_elapsed_ms=0.25,
        compute_batch_elapsed_ms=5.0,
        cpu_staging_elapsed_ms=1.0,
        denominator_elapsed_ms=0.5,
        row_store_write_elapsed_ms=0.75,
        batch_elapsed_ms=8.0,
    )
    normal = _build_phase4_executor_substage_telemetry(
        telemetry_detail="normal",
        encoder_materialize_elapsed_ms=0.25,
        compute_batch_elapsed_ms=5.0,
        cpu_staging_elapsed_ms=1.0,
        denominator_elapsed_ms=0.5,
        row_store_write_elapsed_ms=0.75,
        batch_elapsed_ms=8.0,
    )

    assert summary["executor_encoder_materialize_elapsed_ms"] == pytest.approx(0.25)
    assert summary["executor_compute_batch_elapsed_ms"] == pytest.approx(5.0)
    assert summary["executor_accounted_elapsed_ms"] == pytest.approx(7.5)
    assert summary["executor_overhead_elapsed_ms"] == pytest.approx(0.5)
    assert "executor_cpu_staging_elapsed_ms" not in summary

    assert normal["executor_cpu_staging_elapsed_ms"] == pytest.approx(1.0)
    assert normal["executor_denominator_elapsed_ms"] == pytest.approx(0.5)
    assert normal["executor_row_store_write_elapsed_ms"] == pytest.approx(0.75)


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
    assert result["prefix_match_count"] == 1
    assert result["jaccard_similarity"] == pytest.approx(0.5)


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


def test_phase0_activation_threshold_compare_mode_resolution() -> None:
    assert _resolve_phase0_activation_threshold_compare_mode("DEFAULT") == "baseline"
    assert _resolve_phase0_activation_threshold_compare_mode("bfloat16") == "bf16"
    assert _resolve_phase0_activation_threshold_compare_mode("fp64") == "fp64"

    with pytest.raises(ValueError, match="phase0_activation_threshold_compare_mode"):
        _resolve_phase0_activation_threshold_compare_mode("float16")


def test_hash_sparse_membership_indices_canonicalizes_ordering() -> None:
    indices_a = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 2],
            [3, 4, 1],
        ],
        dtype=torch.long,
    )
    indices_b = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 0],
            [1, 3, 4],
        ],
        dtype=torch.long,
    )
    shape = (2, 3, 8)

    raw_hash_a = _hash_sparse_membership_indices(indices_a, shape=shape, canonicalize=False)
    raw_hash_b = _hash_sparse_membership_indices(indices_b, shape=shape, canonicalize=False)
    canonical_hash_a = _hash_sparse_membership_indices(
        indices_a,
        shape=shape,
        canonicalize=True,
    )
    canonical_hash_b = _hash_sparse_membership_indices(
        indices_b,
        shape=shape,
        canonicalize=True,
    )

    assert raw_hash_a != raw_hash_b
    assert canonical_hash_a == canonical_hash_b


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
        phase1_trace_batch_policy="cap_effective_batches",
        phase1_trace_batch_size_max=32,
        phase4_refresh_policy="deferred_v1",
        phase4_refresh_interval_multiplier=2,
        phase4_ranker="topk_v1",
        row_store_cache_control="fadvise_dontneed_after_append_v1",
        exact_encoder_residency="active_cpu",
    )

    assert result is sentinel
    assert captured["phase4_scheduler_mode"] == "planner_v1"
    assert captured["phase4_scheduler_debug"] is True
    assert captured["phase4_scheduler_telemetry_detail"] == "debug"
    assert captured["phase4_refresh_optimization"] == "v1"
    assert captured["phase4_row_executor"] == "streaming_v1"
    assert captured["phase1_trace_batch_policy"] == "cap_effective_batches"
    assert captured["phase1_trace_batch_size_max"] == 32
    assert captured["phase4_refresh_policy"] == "deferred_v1"
    assert captured["phase4_refresh_interval_multiplier"] == 2
    assert captured["phase4_ranker"] == "topk_v1"
    assert captured["row_store_cache_control"] == "fadvise_dontneed_after_append_v1"
    assert captured["exact_encoder_residency"] == "active_cpu"


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
        phase4_refresh_optimization="v1",
        phase4_row_executor="batched",
        phase1_trace_batch_policy="legacy",
        phase1_trace_batch_size_max=None,
        phase4_refresh_policy="standard",
        phase4_refresh_interval_multiplier=1,
        phase4_ranker="argsort",
        row_store_cache_control="off",
        exact_encoder_residency="lazy",
    )

    assert result is sentinel
    assert "phase4_scheduler_mode" not in captured
    assert "phase4_scheduler_debug" not in captured
    assert "phase4_scheduler_telemetry_detail" not in captured
    assert "phase4_refresh_optimization" not in captured
    assert "phase4_row_executor" not in captured
    assert "phase1_trace_batch_policy" not in captured
    assert "phase1_trace_batch_size_max" not in captured
    assert "phase4_refresh_policy" not in captured
    assert "phase4_refresh_interval_multiplier" not in captured
    assert "phase4_ranker" not in captured
    assert "row_store_cache_control" not in captured
    assert "exact_encoder_residency" not in captured


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        {"phase4_scheduler_mode": "planner_v1"},
        {"phase4_scheduler_mode": "planner_v2"},
        {"phase4_scheduler_debug": True},
        {"phase4_scheduler_telemetry_detail": "summary"},
        {"phase4_refresh_optimization": "off"},
        {"phase4_row_executor": "streaming_v1"},
        {"phase1_trace_batch_policy": "cap_effective_batches"},
        {"phase1_trace_batch_size_max": 8},
        {"phase4_refresh_policy": "deferred_v1"},
        {"phase4_refresh_interval_multiplier": 2},
        {"phase4_ranker": "topk_v1"},
        {"row_store_cache_control": "fadvise_dontneed_after_append_v1"},
        {"exact_encoder_residency": "active_cpu"},
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
