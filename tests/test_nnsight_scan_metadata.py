from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from circuit_tracer.replacement_model.model_adapter import NNSightModelAdapter
from circuit_tracer.replacement_model.nnsight_configuration import (
    configure_nnsight_replacement_model,
)
from circuit_tracer.graph import Graph


class _FakeNNSightModel:
    def __init__(self) -> None:
        self.config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            num_hidden_layers=1,
            hidden_size=4,
            head_dim=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=8,
            vocab_size=16,
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(16, 4))
        )
        self.lm_head = SimpleNamespace(weight=torch.zeros(16, 4))

    def scan(self):
        return "nnsight-envoy-scan"

    def eval(self):
        return self

    def parameters(self):
        return ()

    @staticmethod
    def _resolve_attr(root: object, attr_path: str):
        current = root
        for token in attr_path.split("."):
            current = getattr(current, token)
        return current


class _FakeTranscoderSet:
    scan = "local/topk@revision"
    skip_connection = False
    feature_input_hook = "hook_resid_mid"
    feature_output_hook = "hook_mlp_out"

    def to(self, *_args):
        return self


def test_nnsight_configuration_preserves_envoy_scan_and_records_transcoder_scan_name() -> None:
    model = _FakeNNSightModel()

    configure_nnsight_replacement_model(
        model,
        _FakeTranscoderSet(),
        NNSightModelAdapter(architecture="LlamaForCausalLM"),
    )

    assert model.scan() == "nnsight-envoy-scan"
    assert model.scan_name == "local/topk@revision"


def test_graph_from_pt_reads_legacy_scan_name(tmp_path) -> None:
    path = tmp_path / "legacy-scan-name.pt"
    torch.save(
        {
            "input_string": "test",
            "input_tokens": torch.tensor([1]),
            "active_features": torch.empty((0, 3), dtype=torch.long),
            "adjacency_matrix": torch.zeros((2, 2)),
            "cfg": _FakeNNSightModel().config,
            "selected_features": torch.empty(0, dtype=torch.long),
            "activation_values": torch.empty(0),
            "logit_targets": torch.tensor([2]),
            "logit_probabilities": torch.tensor([1.0]),
            "vocab_size": 16,
            "scan_name": "legacy/topk@revision",
        },
        path,
    )

    graph = Graph.from_pt(str(path))

    assert graph.scan == "legacy/topk@revision"


def test_graph_from_pt_rejects_conflicting_scan_aliases(tmp_path) -> None:
    path = tmp_path / "conflicting-scan-names.pt"
    torch.save({"scan": "current", "scan_name": "legacy"}, path)

    with pytest.raises(ValueError, match="conflicting scan and scan_name"):
        Graph.from_pt(str(path))
