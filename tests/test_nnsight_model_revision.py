from types import SimpleNamespace

import torch

from circuit_tracer.replacement_model.replacement_model_nnsight import (
    LanguageModel,
    NNSightReplacementModel,
)


def test_nnsight_model_revision_reaches_config_and_weight_load(monkeypatch) -> None:
    calls = {}
    config = SimpleNamespace()

    def load_config(model_name, **kwargs):
        calls["config"] = (model_name, kwargs)
        return config

    def init_language_model(self, model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        self.config = config
        self.revision = kwargs.get("revision")

    monkeypatch.setattr(
        "circuit_tracer.replacement_model.replacement_model_nnsight.AutoConfig.from_pretrained",
        load_config,
    )
    monkeypatch.setattr(LanguageModel, "__init__", init_language_model)
    monkeypatch.setattr(
        NNSightReplacementModel,
        "_configure_replacement_model",
        lambda self, _transcoders: None,
    )

    model = NNSightReplacementModel.from_pretrained_and_transcoders(
        model_name="local/model",
        transcoders=object(),
        device=torch.device("cpu"),
        revision="deadbeef",
    )

    assert calls["config"] == ("local/model", {"revision": "deadbeef"})
    assert calls["model"][0] == "local/model"
    assert calls["model"][1]["revision"] == "deadbeef"
    assert model.revision == "deadbeef"
