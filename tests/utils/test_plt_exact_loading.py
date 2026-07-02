from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from circuit_tracer.utils.caching import get_cached_path, load_transcoders_from_cache
from circuit_tracer.utils.hf_utils import load_transcoders


def _write_gemmascope2_plt(path: Path, d_model: int = 3, d_transcoder: int = 5) -> None:
    save_file(
        {
            "w_enc": torch.randn(d_model, d_transcoder),
            "w_dec": torch.randn(d_transcoder, d_model),
            "b_enc": torch.randn(d_transcoder),
            "b_dec": torch.randn(d_model),
            "threshold": torch.zeros(d_transcoder),
        },
        str(path),
    )


def _config(path: Path, *, exact: bool | None = None) -> dict:
    config = {
        "model_kind": "transcoder_set",
        "repo_id": "local/snapshot-style",
        "scan": "local/snapshot-style",
        "feature_input_hook": "blocks.{layer}.hook_resid_pre",
        "feature_output_hook": "blocks.{layer}.hook_mlp_out",
        "transcoders": [str(path)],
    }
    if exact is not None:
        config["supports_exact_chunked_provider"] = exact
    return config


def test_lowercase_gemmascope2_exact_forces_lazy_flags_and_defaults(tmp_path: Path):
    layer_path = tmp_path / "layer_0.safetensors"
    _write_gemmascope2_plt(layer_path)

    transcoders = load_transcoders(
        _config(layer_path, exact=True),
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert transcoders.exact_chunked_provider is True
    assert transcoders[0].weight_format == "gemmascope2"
    assert transcoders[0].lazy_encoder is True
    assert transcoders[0].lazy_decoder is True
    assert transcoders.capabilities.decoder_output_topology == "same_layer"
    assert transcoders.capabilities.default_cross_batch_decoder_cache_bytes == 0
    assert transcoders.create_decoder_block_cache() is None


def test_lowercase_gemmascope2_without_exact_stays_non_exact(tmp_path: Path):
    layer_path = tmp_path / "layer_0.safetensors"
    _write_gemmascope2_plt(layer_path)

    transcoders = load_transcoders(
        _config(layer_path),
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert transcoders.exact_chunked_provider is False
    assert transcoders[0].weight_format == "gemmascope2"
    assert transcoders[0].lazy_encoder is False
    assert transcoders[0].lazy_decoder is False


def test_provider_metadata_only_exact_config_enables_lazy_exact(tmp_path: Path):
    layer_path = tmp_path / "layer_0.safetensors"
    _write_gemmascope2_plt(layer_path)
    config = _config(layer_path)
    config["transcoder_capabilities"] = {"supports_exact_chunked_provider": True}

    transcoders = load_transcoders(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert transcoders.exact_chunked_provider is True
    assert transcoders[0].lazy_encoder is True
    assert transcoders[0].lazy_decoder is True
    assert config["transcoder_capability_source"] == "provider_metadata"


def test_provider_fingerprint_only_exact_config_enables_lazy_exact(tmp_path: Path):
    layer_path = tmp_path / "layer_0.safetensors"
    _write_gemmascope2_plt(layer_path)
    source_config = _config(layer_path, exact=True)
    load_transcoders(source_config, device=torch.device("cpu"), dtype=torch.float32)
    fingerprint = source_config["transcoder_provider_fingerprint"]

    config = _config(layer_path)
    config["transcoder_provider_fingerprint"] = fingerprint
    transcoders = load_transcoders(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert transcoders.exact_chunked_provider is True
    assert transcoders[0].lazy_encoder is True
    assert transcoders[0].lazy_decoder is True
    assert config["transcoder_capability_source"] == "provider_fingerprint"


def _write_cached_config(cache_path: Path, *, exact: bool, capability_only: bool = False) -> None:
    cache_path.mkdir(parents=True)
    _write_gemmascope2_plt(cache_path / "layer_0.safetensors")
    config = _config(cache_path / "layer_0.safetensors", exact=None if capability_only else exact)
    if capability_only:
        config["transcoder_capabilities"] = {"supports_exact_chunked_provider": exact}
    config.pop("repo_id")
    config.pop("transcoders")
    with open(cache_path / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)


def test_cached_exact_transcoder_set_missing_fingerprint_rejects(tmp_path: Path):
    hf_ref = "local/exact-missing-fingerprint"
    cache_path = get_cached_path(hf_ref, tmp_path)
    _write_cached_config(cache_path, exact=True)

    with pytest.raises(ValueError, match="transcoder_provider_fingerprint"):
        load_transcoders_from_cache(hf_ref, cache_dir=tmp_path, device=torch.device("cpu"))


def test_cached_provider_capable_transcoder_set_missing_fingerprint_rejects(tmp_path: Path):
    hf_ref = "local/provider-capable-missing-fingerprint"
    cache_path = get_cached_path(hf_ref, tmp_path)
    _write_cached_config(cache_path, exact=True, capability_only=True)

    with pytest.raises(ValueError, match="transcoder_provider_fingerprint"):
        load_transcoders_from_cache(hf_ref, cache_dir=tmp_path, device=torch.device("cpu"))


def test_cached_legacy_non_exact_missing_fingerprint_loads(tmp_path: Path):
    hf_ref = "local/legacy-missing-fingerprint"
    cache_path = get_cached_path(hf_ref, tmp_path)
    _write_cached_config(cache_path, exact=False)

    transcoders, _ = load_transcoders_from_cache(
        hf_ref, cache_dir=tmp_path, device=torch.device("cpu")
    )

    assert transcoders.exact_chunked_provider is False
