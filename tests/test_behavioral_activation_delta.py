from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from circuit_tracer.transcoder.activation_functions import JumpReLU, TopK
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    TranscoderSet,
)
from circuit_tracer.verification.nnsight_runtime import _provider_activation_delta


def test_provider_activation_delta_supports_vector_threshold_jumprelu() -> None:
    feature_width = 3
    transcoder = SingleLayerTranscoder(
        d_model=1,
        d_transcoder=feature_width,
        activation_function=JumpReLU(
            torch.tensor([0.25, 1.0, 2.0]), bandwidth=0.1
        ),
        layer_idx=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    provider = TranscoderSet(
        {0: transcoder},
        feature_input_hook="feature_input",
        feature_output_hook="feature_output",
    )

    delta = _provider_activation_delta(
        provider,
        layer=0,
        feature=1,
        baseline_preactivation=torch.tensor(0.5),
        absolute_preactivation=1.5,
    )

    assert torch.equal(delta, torch.tensor(1.5))


def test_cross_layer_provider_activation_delta_selects_feature_threshold() -> None:
    provider = CrossLayerTranscoder(
        n_layers=1,
        d_transcoder=3,
        d_model=1,
        activation_function="jump_relu",
        lazy_encoder=False,
        lazy_decoder=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        provider.activation_function.threshold[0, 0] = torch.tensor([0.25, 1.0, 2.0])

    delta = _provider_activation_delta(
        provider,
        layer=0,
        feature=1,
        baseline_preactivation=torch.tensor(0.5),
        absolute_preactivation=1.5,
    )

    assert torch.equal(delta, torch.tensor(1.5))


def test_transcoder_set_provider_activation_delta_supports_relu() -> None:
    transcoder = SingleLayerTranscoder(
        d_model=1,
        d_transcoder=3,
        activation_function=F.relu,
        layer_idx=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    provider = TranscoderSet(
        {0: transcoder},
        feature_input_hook="feature_input",
        feature_output_hook="feature_output",
    )

    delta = _provider_activation_delta(
        provider,
        layer=0,
        feature=1,
        baseline_preactivation=torch.tensor(-0.5),
        absolute_preactivation=1.5,
    )

    assert torch.equal(delta, torch.tensor(1.5))


def test_cross_layer_provider_activation_delta_supports_relu() -> None:
    provider = CrossLayerTranscoder(
        n_layers=1,
        d_transcoder=3,
        d_model=1,
        activation_function="relu",
        lazy_encoder=False,
        lazy_decoder=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    delta = _provider_activation_delta(
        provider,
        layer=0,
        feature=1,
        baseline_preactivation=torch.tensor(-0.5),
        absolute_preactivation=1.5,
    )

    assert torch.equal(delta, torch.tensor(1.5))


def test_transcoder_set_refuses_selected_feature_topk_activation() -> None:
    transcoder = SingleLayerTranscoder(
        d_model=1,
        d_transcoder=3,
        activation_function=TopK(k=1),
        layer_idx=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    provider = TranscoderSet(
        {0: transcoder},
        feature_input_hook="feature_input",
        feature_output_hook="feature_output",
    )

    with pytest.raises(ValueError, match="featurewise activation"):
        _provider_activation_delta(
            provider,
            layer=0,
            feature=1,
            baseline_preactivation=torch.tensor(0.5),
            absolute_preactivation=1.5,
        )


def test_cross_layer_provider_refuses_selected_feature_topk_activation() -> None:
    provider = CrossLayerTranscoder(
        n_layers=1,
        d_transcoder=3,
        d_model=1,
        activation_function="relu",
        lazy_encoder=False,
        lazy_decoder=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    provider.activation_function = TopK(k=1)

    with pytest.raises(ValueError, match="featurewise activation"):
        _provider_activation_delta(
            provider,
            layer=0,
            feature=1,
            baseline_preactivation=torch.tensor(0.5),
            absolute_preactivation=1.5,
        )
