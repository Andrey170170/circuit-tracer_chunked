from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.attribution.context_nnsight import (
    AttributionContext as NNSightAttributionContext,
)
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt, load_gemma_scope_2_clt


class FakeDecoderProvider:
    def __init__(self, blocks: dict[int, torch.Tensor], chunk_size: int = 1) -> None:
        self.blocks = blocks
        self.decoder_chunk_size = chunk_size
        self.load_calls: list[tuple[int, tuple[int, ...]]] = []

    def get_decoder_block(
        self, layer_id: int, feat_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        feat_ids = torch.arange(self.blocks[layer_id].shape[0]) if feat_ids is None else feat_ids
        feat_ids_cpu = feat_ids.cpu()
        self.load_calls.append((layer_id, tuple(int(x) for x in feat_ids_cpu.tolist())))
        return self.blocks[layer_id][feat_ids_cpu]


def _make_chunked_context(context_cls):
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
    ctx, provider = _make_chunked_context(context_cls)

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

    ctx._compute_chunked_feature_attributions_from_grads(grads_by_output_layer)

    expected_feature_rows = torch.tensor(
        [
            [102.0, 146.0],
            [372.0, 504.0],
            [-270.0, -360.0],
        ]
    )
    assert ctx._chunked_layer_spans == [(0, 2), (2, 3), None]
    assert torch.allclose(ctx._batch_buffer[:3], expected_feature_rows)
    assert torch.count_nonzero(ctx._batch_buffer[3:]) == 0
    assert provider.load_calls == [(0, (0,)), (0, (1,)), (1, (0,))]


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
