import gc

import pytest
import torch

from circuit_tracer import ReplacementModel, SparsificationConfig
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from tests.conftest import has_32gb


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemmascope2_exact_chunked_nnsight_smoke():
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-pt",
        "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine",
        backend="nnsight",
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    assert isinstance(model, NNSightReplacementModel)
    assert getattr(model.transcoders, "exact_chunked_decoder", False)

    graph = attribute_nnsight(
        "If Alice has 3 apples and buys 2 more, she has",
        model,
        max_n_logits=4,
        batch_size=16,
        max_feature_nodes=128,
        verbose=True,
        offload="cpu",
    )

    expected_nodes = (
        len(graph.selected_features)
        + model.cfg.n_layers * len(graph.input_tokens)
        + len(graph.input_tokens)
        + len(graph.logit_targets)
    )
    assert graph.adjacency_matrix.shape == (expected_nodes, expected_nodes)


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemmascope2_exact_chunked_nnsight_sparsified_smoke():
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-pt",
        "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine",
        backend="nnsight",
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    assert isinstance(model, NNSightReplacementModel)
    assert getattr(model.transcoders, "exact_chunked_decoder", False)

    graph = attribute_nnsight(
        "If Alice has 3 apples and buys 2 more, she has",
        model,
        max_n_logits=4,
        batch_size=16,
        max_feature_nodes=128,
        sparsification=SparsificationConfig(
            per_layer_position_topk=8,
            global_cap=256,
        ),
        verbose=True,
        offload="cpu",
    )

    expected_nodes = (
        len(graph.selected_features)
        + model.cfg.n_layers * len(graph.input_tokens)
        + len(graph.input_tokens)
        + len(graph.logit_targets)
    )
    assert len(graph.selected_features) <= 128
    assert graph.adjacency_matrix.shape == (expected_nodes, expected_nodes)
