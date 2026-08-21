from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import save_file

from circuit_tracer.transcoder.provider import exact_chunked_provider_usable, provider_fingerprint
from circuit_tracer import AttributionProblem, TraceRequest, resolve_trace_request
from circuit_tracer.utils.caching import load_transcoders_from_cache, save_transcoders_to_cache
from circuit_tracer.utils.hf_utils import load_transcoders


def _write_standard_plt(path: Path, *, d_model: int = 2, d_transcoder: int = 130) -> None:
    save_file(
        {
            "W_enc": torch.zeros(d_transcoder, d_model),
            "W_dec": torch.zeros(d_transcoder, d_model),
            "b_enc": torch.arange(1, d_transcoder + 1, dtype=torch.float32),
            "b_dec": torch.zeros(d_model),
        },
        str(path),
    )


def _target_topk_config(path: Path) -> dict:
    return {
        "model_kind": "transcoder_set",
        "repo_id": "local/target-topk",
        "scan": "local/target-topk",
        "feature_input_hook": "blocks.{layer}.hook_resid_pre",
        "feature_output_hook": "blocks.{layer}.hook_mlp_out",
        "transcoders": [str(path)],
        "activation": "topk",
        "k": 128,
    }


def test_target_topk_config_keeps_exactly_128_features(tmp_path: Path) -> None:
    layer_path = tmp_path / "layer_0.safetensors"
    _write_standard_plt(layer_path)

    transcoders = load_transcoders(
        _target_topk_config(layer_path),
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    activations = transcoders.encode_layer(torch.zeros(1, 2), layer_id=0)

    assert torch.equal(
        activations,
        torch.tensor([[0.0, 0.0, *range(3, 131)]], dtype=torch.float32),
    )


def test_target_topk_config_selects_exact_chunked_provider(tmp_path: Path) -> None:
    layer_path = tmp_path / "layer_0.safetensors"
    _write_standard_plt(layer_path)
    config = _target_topk_config(layer_path)

    transcoders = load_transcoders(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert exact_chunked_provider_usable(transcoders)
    assert transcoders[0].lazy_encoder is True
    assert transcoders[0].lazy_decoder is True
    assert config["transcoder_capability_source"] == "topk_exact_policy"


def test_target_topk_activation_and_exact_provider_survive_cache_round_trip(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source_layer_0.safetensors"
    _write_standard_plt(source_path)
    source_config_path = tmp_path / "config.yaml"
    source_config_path.write_text(
        yaml.safe_dump(
            {
                "model_kind": "transcoder_set",
                "feature_input_hook": "blocks.{layer}.hook_resid_pre",
                "feature_output_hook": "blocks.{layer}.hook_mlp_out",
                "transcoders": [str(source_path)],
                "activation": "topk",
                "k": 128,
            }
        )
    )
    monkeypatch.setattr(
        "circuit_tracer.utils.caching.hf_hub_download",
        lambda **_kwargs: str(source_config_path),
    )
    cache_root = tmp_path / "cache"

    save_transcoders_to_cache(
        "local/target-topk",
        cache_dir=cache_root,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    transcoders, cached_config = load_transcoders_from_cache(
        "local/target-topk",
        cache_dir=cache_root,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    activations = transcoders.encode_layer(torch.zeros(1, 2), layer_id=0)
    assert torch.equal(
        activations,
        torch.tensor([[0.0, 0.0, *range(3, 131)]], dtype=torch.float32),
    )
    assert exact_chunked_provider_usable(transcoders)
    assert cached_config["activation"] == "topk"
    assert cached_config["k"] == 128


def test_provider_fingerprint_distinguishes_topk_128_from_relu(tmp_path: Path) -> None:
    layer_path = tmp_path / "layer_0.safetensors"
    _write_standard_plt(layer_path)
    topk_config = _target_topk_config(layer_path)
    topk64_config = dict(topk_config, k=64)
    relu_config = dict(topk_config, activation="relu")
    relu_config.pop("k")

    topk = load_transcoders(topk_config, device=torch.device("cpu"), dtype=torch.float32)
    topk64 = load_transcoders(topk64_config, device=torch.device("cpu"), dtype=torch.float32)
    relu = load_transcoders(relu_config, device=torch.device("cpu"), dtype=torch.float32)

    topk_fingerprint = provider_fingerprint(topk)
    relu_fingerprint = provider_fingerprint(relu)
    assert topk_fingerprint["activation_kind"] == "topk"
    assert topk_fingerprint["activation_k"] == 128
    assert relu_fingerprint["activation_kind"] == "relu"
    assert relu_fingerprint["activation_k"] is None
    assert topk_fingerprint != relu_fingerprint

    def semantic_fingerprint(provider) -> str:
        model = SimpleNamespace(
            backend="nnsight",
            config=SimpleNamespace(_name_or_path="local/model", architectures=("Test",)),
            transcoders=provider,
        )
        return resolve_trace_request(
            TraceRequest(problem=AttributionProblem(model=model, prompt=[1, 2]))
        ).semantic_fingerprint

    assert semantic_fingerprint(topk) != semantic_fingerprint(relu)
    assert semantic_fingerprint(topk) != semantic_fingerprint(topk64)
