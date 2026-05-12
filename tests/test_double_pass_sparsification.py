from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file

from circuit_tracer.attribution.sparsification import (
    SparsificationConfig,
    filter_sparse_activations,
    select_candidate_feature_indices,
)
from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt
from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set


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


def _create_single_layer_transcoder_files(
    tmp_path: Path,
    n_layers: int = 3,
    d_model: int = 8,
    d_sae: int = 16,
) -> dict[int, str]:
    paths: dict[int, str] = {}
    for layer_idx in range(n_layers):
        layer_path = tmp_path / f"plt_layer_{layer_idx}.safetensors"
        save_file(
            {
                "W_enc": torch.randn(d_sae, d_model),
                "W_dec": torch.randn(d_sae, d_model),
                "b_enc": torch.randn(d_sae),
                "b_dec": torch.randn(d_model),
            },
            str(layer_path),
        )
        paths[layer_idx] = str(layer_path)
    return paths


def test_select_candidate_feature_indices_uses_layer_position_topk_then_global_cap():
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
            ]
        ),
        values=torch.tensor([1.0, 3.0, 2.0, 4.0, 0.5, 5.0]),
        size=(2, 2, 2),
        check_invariants=True,
    ).coalesce()

    selected, stats = select_candidate_feature_indices(
        activation_matrix,
        SparsificationConfig(per_layer_position_topk=1, global_cap=3),
    )

    assert selected.tolist() == [1, 3, 5]
    assert stats["candidate_count_before"] == 6
    assert stats["candidate_count_after"] == 3
    assert stats["per_layer_candidate_counts"] == {0: 3, 1: 3}
    assert stats["per_layer_retained_counts"] == {0: 1, 1: 2}
    retained_activation_mass = cast(float, stats["retained_activation_mass"])
    assert abs(retained_activation_mass - (12.0 / 15.5)) < 1e-6


def test_select_candidate_feature_indices_retained_mass_avoids_float32_overflow() -> None:
    activation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 2, 3],
            ]
        ),
        values=torch.tensor([1e38, 1e38, 1e38, 1e38], dtype=torch.float32),
        size=(1, 1, 4),
        check_invariants=True,
    ).coalesce()

    selected, stats = select_candidate_feature_indices(
        activation_matrix,
        SparsificationConfig(global_cap=2),
    )

    assert selected.numel() == 2
    retained_activation_mass = cast(float, stats["retained_activation_mass"])
    assert abs(retained_activation_mass - 0.5) < 1e-6


def test_chunked_components_apply_sparsification_before_reconstruction(tmp_path: Path):
    torch.manual_seed(0)
    clt = load_gemma_scope_2_clt(
        _create_gemmascope2_clt_files(tmp_path),
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )
    inputs = torch.randn(clt.n_layers, 5, clt.d_model, dtype=clt.dtype)

    full_components = clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))
    config = SparsificationConfig(per_layer_position_topk=1, global_cap=4)
    sparse_components = clt.compute_attribution_components(
        inputs,
        zero_positions=slice(0, 1),
        sparsification=config,
    )

    selected, expected_stats = select_candidate_feature_indices(
        full_components["activation_matrix"],
        config,
    )
    expected_activation_matrix = filter_sparse_activations(
        full_components["activation_matrix"], selected
    )
    expected_reconstruction = clt.compute_reconstruction_chunked(expected_activation_matrix, inputs)

    assert sparse_components["activation_matrix"]._nnz() == expected_activation_matrix._nnz()
    assert torch.equal(
        sparse_components["activation_matrix"].indices(),
        expected_activation_matrix.indices(),
    )
    assert torch.allclose(
        sparse_components["activation_matrix"].values(),
        expected_activation_matrix.values(),
    )
    assert torch.allclose(sparse_components["reconstruction"], expected_reconstruction)
    assert sparse_components["chunked_decoder_state"]["source_layers"].numel() == (
        expected_activation_matrix._nnz()
    )
    assert (
        sparse_components["sparsification_stats"]["candidate_count_after"]
        == expected_stats["candidate_count_after"]
    )
    assert (
        sparse_components["sparsification_stats"]["per_layer_retained_counts"]
        == expected_stats["per_layer_retained_counts"]
    )


def test_chunked_components_match_baseline_when_budget_is_unconstrained(tmp_path: Path):
    torch.manual_seed(0)
    clt = load_gemma_scope_2_clt(
        _create_gemmascope2_clt_files(tmp_path),
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )
    inputs = torch.randn(clt.n_layers, 4, clt.d_model, dtype=clt.dtype)

    baseline = clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))
    all_kept = clt.compute_attribution_components(
        inputs,
        zero_positions=slice(0, 1),
        sparsification=SparsificationConfig(per_layer_position_topk=10_000),
    )

    assert torch.equal(
        baseline["activation_matrix"].indices(),
        all_kept["activation_matrix"].indices(),
    )
    assert torch.allclose(
        baseline["activation_matrix"].values(),
        all_kept["activation_matrix"].values(),
    )
    assert torch.allclose(baseline["reconstruction"], all_kept["reconstruction"])
    assert torch.allclose(baseline["encoder_vecs"], all_kept["encoder_vecs"])
    assert (
        all_kept["sparsification_stats"]["candidate_count_before"]
        == baseline["activation_matrix"]._nnz()
    )
    assert (
        all_kept["sparsification_stats"]["candidate_count_after"]
        == baseline["activation_matrix"]._nnz()
    )


def test_single_layer_components_match_baseline_when_budget_is_unconstrained(tmp_path: Path):
    torch.manual_seed(0)
    transcoders = load_transcoder_set(
        transcoder_paths=_create_single_layer_transcoder_files(tmp_path),
        scan="test_scan",
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=True,
    )
    mlp_inputs = torch.randn(len(transcoders.transcoders), 5, 8)

    baseline = transcoders.compute_attribution_components(mlp_inputs, zero_positions=slice(0, 1))
    all_kept = transcoders.compute_attribution_components(
        mlp_inputs,
        zero_positions=slice(0, 1),
        sparsification=SparsificationConfig(per_layer_position_topk=10_000),
    )

    baseline_activation_matrix = cast(torch.Tensor, baseline["activation_matrix"])
    all_kept_activation_matrix = cast(torch.Tensor, all_kept["activation_matrix"])
    baseline_reconstruction = cast(torch.Tensor, baseline["reconstruction"])
    all_kept_reconstruction = cast(torch.Tensor, all_kept["reconstruction"])
    baseline_encoder_vecs = cast(torch.Tensor, baseline["encoder_vecs"])
    all_kept_encoder_vecs = cast(torch.Tensor, all_kept["encoder_vecs"])
    baseline_decoder_vecs = cast(torch.Tensor, baseline["decoder_vecs"])
    all_kept_decoder_vecs = cast(torch.Tensor, all_kept["decoder_vecs"])
    baseline_encoder_to_decoder_map = cast(torch.Tensor, baseline["encoder_to_decoder_map"])
    all_kept_encoder_to_decoder_map = cast(torch.Tensor, all_kept["encoder_to_decoder_map"])

    assert torch.equal(
        baseline_activation_matrix.indices(),
        all_kept_activation_matrix.indices(),
    )
    assert torch.allclose(
        baseline_activation_matrix.values(),
        all_kept_activation_matrix.values(),
    )
    assert torch.allclose(baseline_reconstruction, all_kept_reconstruction)
    assert torch.allclose(baseline_encoder_vecs, all_kept_encoder_vecs)
    assert torch.allclose(baseline_decoder_vecs, all_kept_decoder_vecs)
    assert torch.equal(baseline_encoder_to_decoder_map, all_kept_encoder_to_decoder_map)
