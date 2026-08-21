from __future__ import annotations

import pytest
import torch
import yaml
from safetensors.torch import save_file

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.transcoder.provider import (
    TranscoderCapabilities,
    normalize_provider_fingerprints_for_comparison,
    provider_fingerprint,
)
from circuit_tracer.transcoder.single_layer_transcoder import (
    load_relu_transcoder,
    load_transcoder_set,
)
from circuit_tracer.utils.caching import get_cached_path, load_transcoders_from_cache
from circuit_tracer.utils.hf_utils import _validate_configured_provider_fingerprint


def _legacy_fingerprint(provider: object, *, checkpoint_identity: str) -> dict[str, object]:
    fingerprint = provider_fingerprint(
        provider,
        checkpoint_format="standard",
        checkpoint_identity=checkpoint_identity,
    )
    fingerprint.pop("supports_active_decoder_row_residency")
    fingerprint.pop("supports_phase0_decoder_row_ranges")
    return fingerprint


def test_legacy_normalization_ignores_only_missing_active_row_capability() -> None:
    legacy = {
        "schema_version": 1,
        "architecture": "plt",
        "decoder_chunk_size": 2,
    }
    current = {
        **legacy,
        "supports_active_decoder_row_residency": True,
    }
    normalized_legacy, normalized_current = normalize_provider_fingerprints_for_comparison(
        legacy, current
    )
    assert normalized_legacy == normalized_current
    assert normalized_legacy["supports_active_decoder_row_residency"] is False
    assert normalized_legacy["supports_phase0_decoder_row_ranges"] is False

    explicit_false = {
        **legacy,
        "supports_active_decoder_row_residency": False,
    }
    normalized_false, normalized_true = normalize_provider_fingerprints_for_comparison(
        explicit_false, current
    )
    assert normalized_false != normalized_true

    other_mismatch = dict(current, decoder_chunk_size=4)
    normalized_legacy, normalized_mismatch = normalize_provider_fingerprints_for_comparison(
        legacy, other_mismatch
    )
    assert normalized_legacy != normalized_mismatch


class _ConfiguredProvider:
    def __init__(self, architecture: str) -> None:
        self.n_layers = 2
        self.d_model = 4
        self.d_transcoder = 8
        self.dtype = torch.float32
        self.scan = f"legacy-{architecture}"
        self.capabilities = TranscoderCapabilities(
            architecture=architecture,  # type: ignore[arg-type]
            checkpoint_format="standard",
            supports_exact_chunked_provider=True,
            supports_active_decoder_row_residency=architecture == "plt",
            decoder_output_topology="same_layer" if architecture == "plt" else "cross_layer",
            default_decoder_chunk_size=2,
            default_cross_batch_decoder_cache_bytes=0,
        )


@pytest.mark.parametrize("architecture", ["plt", "clt"])
def test_legacy_config_fingerprint_comparison_accepts_missing_capability(
    architecture: str,
) -> None:
    provider = _ConfiguredProvider(architecture)
    expected = _legacy_fingerprint(provider, checkpoint_identity=provider.scan)
    _validate_configured_provider_fingerprint(
        {"transcoder_provider_fingerprint": expected},
        provider,
    )

    mismatched = dict(expected, decoder_chunk_size=4)
    with pytest.raises(ValueError, match="provider_fingerprint mismatch"):
        _validate_configured_provider_fingerprint(
            {"transcoder_provider_fingerprint": mismatched},
            provider,
        )


def test_legacy_plt_cache_fingerprint_loads(tmp_path) -> None:
    cache_ref = "legacy/plt"
    cache_path = get_cached_path(cache_ref, tmp_path)
    cache_path.mkdir(parents=True)
    source = tmp_path / "source_plt.safetensors"
    save_file(
        {
            "W_enc": torch.randn(6, 4),
            "W_dec": torch.randn(6, 4),
            "b_enc": torch.randn(6),
            "b_dec": torch.randn(4),
        },
        str(source),
    )
    transcoder = load_relu_transcoder(
        source,
        0,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )
    layer_path = cache_path / "layer_0.safetensors"
    transcoder.to_safetensors(str(layer_path))
    provider = load_transcoder_set(
        {0: str(layer_path)},
        scan=cache_ref,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device("cpu"),
        exact_chunked_provider=True,
        decoder_chunk_size=2,
    )
    config = {
        "model_kind": "transcoder_set",
        "scan": cache_ref,
        "feature_input_hook": "hook_resid_mid",
        "feature_output_hook": "hook_mlp_out",
        "decoder_chunk_size": 2,
        "cross_batch_decoder_cache_bytes": 0,
        "transcoder_provider_fingerprint": _legacy_fingerprint(
            provider, checkpoint_identity=cache_ref
        ),
    }
    (cache_path / "config.yaml").write_text(yaml.safe_dump(config))

    loaded, _ = load_transcoders_from_cache(
        cache_ref,
        cache_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert loaded.capabilities.supports_active_decoder_row_residency is True


def test_legacy_clt_cache_fingerprint_loads(tmp_path) -> None:
    cache_ref = "legacy-clt"
    cache_path = get_cached_path(cache_ref, tmp_path)
    clt = CrossLayerTranscoder(
        n_layers=2,
        d_transcoder=4,
        d_model=3,
        lazy_decoder=False,
        exact_chunked_decoder=True,
        decoder_chunk_size=2,
        cross_batch_decoder_cache_bytes=0,
        scan=cache_ref,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    clt.to_safetensors(str(cache_path))
    config = {
        "model_kind": "cross_layer_transcoder",
        "scan": cache_ref,
        "feature_input_hook": "hook_resid_mid",
        "feature_output_hook": "hook_mlp_out",
        "decoder_chunk_size": 2,
        "cross_batch_decoder_cache_bytes": 0,
        "transcoder_provider_fingerprint": _legacy_fingerprint(clt, checkpoint_identity=cache_ref),
    }
    (cache_path / "config.yaml").write_text(yaml.safe_dump(config))

    loaded, _ = load_transcoders_from_cache(
        cache_ref,
        cache_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert loaded.capabilities.supports_active_decoder_row_residency is False
