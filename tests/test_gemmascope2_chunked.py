from pathlib import Path

import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.cross_layer_transcoder import load_clt, load_gemma_scope_2_clt


def _create_gemmascope2_clt_files(
    tmp_path: Path,
    n_layers=3,
    d_model=8,
    d_transcoder=16,
    duplicate_last_path: bool = False,
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

    if duplicate_last_path:
        paths[n_layers] = paths[n_layers - 1]

    return paths


def test_gemmascope2_exact_chunked_matches_standard_clt_conversion(tmp_path: Path):
    paths = _create_gemmascope2_clt_files(tmp_path, duplicate_last_path=True)

    chunked_clt = load_gemma_scope_2_clt(
        paths,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )
    eager_clt = load_gemma_scope_2_clt(
        paths,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert chunked_clt.exact_chunked_decoder is True
    assert chunked_clt.n_layers == eager_clt.n_layers == 3

    standard_dir = tmp_path / "standard_clt"
    standard_dir.mkdir()
    eager_clt.to_safetensors(str(standard_dir))
    standard_clt = load_clt(
        str(standard_dir),
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
        exact_chunked_decoder=False,
    )

    inputs = torch.randn(chunked_clt.n_layers, 5, chunked_clt.d_model, dtype=chunked_clt.dtype)
    chunked_outputs = chunked_clt(inputs)
    standard_outputs = standard_clt(inputs)

    assert torch.allclose(chunked_outputs, standard_outputs, atol=5e-2, rtol=5e-2)

    attribution_data = chunked_clt.compute_attribution_components(
        inputs, zero_positions=slice(0, 1)
    )
    assert "chunked_decoder_state" in attribution_data
    assert attribution_data["decoder_vecs"].numel() == 0
    assert attribution_data["encoder_to_decoder_map"].numel() == 0
    assert torch.allclose(
        attribution_data["reconstruction"][:, 1:], standard_outputs[:, 1:], atol=5e-2, rtol=5e-2
    )
