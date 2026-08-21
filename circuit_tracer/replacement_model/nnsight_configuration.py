"""Replacement-model configuration ownership for the NNSight backend."""

from __future__ import annotations

from typing import Protocol, cast

import torch

from circuit_tracer.replacement_model.model_adapter import NNSightModelAdapter
from circuit_tracer.utils.tl_nnsight_mapping import (
    convert_nnsight_config_to_transformerlens,
    get_mapping,
)


class ConfigurableReplacementModel(Protocol):
    config: object
    device: object
    dtype: torch.dtype

    def eval(self): ...

    def parameters(self): ...

    @staticmethod
    def _resolve_attr(root: object, attr_path: str): ...


def configure_nnsight_replacement_model(
    model: ConfigurableReplacementModel,
    transcoder_set,
    adapter: NNSightModelAdapter,
) -> None:
    """Install hook locations, weights, and frozen transcoder runtime state."""
    model.backend = "nnsight"  # type: ignore[attr-defined]
    model.eval()
    model.cfg = convert_nnsight_config_to_transformerlens(model.config)  # type: ignore[attr-defined, arg-type]
    model.model_adapter = adapter  # type: ignore[attr-defined]
    model.zero_positions = adapter.ignored_token_positions  # type: ignore[attr-defined]

    transcoder_set.to(model.device, model.dtype)
    model.transcoders = transcoder_set  # type: ignore[attr-defined]
    model.skip_transcoder = transcoder_set.skip_connection  # type: ignore[attr-defined]

    nnsight_config = get_mapping(adapter.architecture)
    model._feature_input_pattern, model._feature_input_io = nnsight_config.feature_hook_mapping[  # type: ignore[attr-defined]
        transcoder_set.feature_input_hook
    ]
    model._feature_output_pattern, _ = nnsight_config.feature_hook_mapping[  # type: ignore[attr-defined]
        transcoder_set.feature_output_hook
    ]
    model._attention_pattern = nnsight_config.attention_location_pattern  # type: ignore[attr-defined]
    model._layernorm_scale_patterns = nnsight_config.layernorm_scale_location_patterns  # type: ignore[attr-defined]
    model._pre_logit_location = nnsight_config.pre_logit_location  # type: ignore[attr-defined]
    model._embed_location = nnsight_config.embed_location  # type: ignore[attr-defined]
    model.embed_weight = cast(  # type: ignore[attr-defined]
        torch.Tensor, model._resolve_attr(model, nnsight_config.embed_weight)
    )
    model.unembed_weight = cast(  # type: ignore[attr-defined]
        torch.Tensor, model._resolve_attr(model, nnsight_config.unembed_weight)
    )
    # ``LanguageModel.scan`` is NNSight's callable Envoy API. Keep transcoder
    # provenance under a distinct name so replacement-model setup cannot
    # shadow that inherited method.
    model.scan_name = transcoder_set.scan  # type: ignore[attr-defined]

    for parameter in model.parameters():
        parameter.requires_grad = False
