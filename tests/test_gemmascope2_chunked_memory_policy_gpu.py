import gc

import pytest
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from tests.conftest import has_32gb


def _load_gemmascope2_model() -> NNSightReplacementModel:
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-pt",
        "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine",
        backend="nnsight",
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    assert isinstance(model, NNSightReplacementModel)
    return model


def _assert_graphs_close(graph_a, graph_b) -> None:
    assert torch.equal(graph_a.selected_features, graph_b.selected_features)
    assert graph_a.adjacency_matrix.shape == graph_b.adjacency_matrix.shape
    diff = (graph_a.adjacency_matrix - graph_b.adjacency_matrix).abs()
    max_flat_idx = int(diff.argmax().item())
    max_row, max_col = divmod(max_flat_idx, diff.shape[1])
    a_value = float(graph_a.adjacency_matrix[max_row, max_col].item())
    b_value = float(graph_b.adjacency_matrix[max_row, max_col].item())
    assert torch.allclose(
        graph_a.adjacency_matrix,
        graph_b.adjacency_matrix,
        atol=5e-3,
        rtol=2e-2,
    ), (
        f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, "
        f"max_idx=({max_row},{max_col}), a_value={a_value:.6f}, b_value={b_value:.6f}"
    )


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemmascope2_exact_chunked_setup_retains_last_token_logits_only_by_default():
    model = _load_gemmascope2_model()

    ctx = model.setup_attribution("Paris is the capital of")
    ctx_full = model.setup_attribution("Paris is the capital of", retain_full_logits=True)

    assert ctx.logits.shape[1] == 1
    assert ctx_full.full_logits is not None
    assert torch.allclose(ctx.get_last_token_logits(), ctx_full.full_logits[:, -1])

    ctx.cleanup()
    ctx_full.cleanup()


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemmascope2_exact_chunked_split_batches_match_default_graph():
    prompt = "If Alice has 3 apples and buys 2 more, she has"
    default_model = _load_gemmascope2_model()
    graph_default = attribute_nnsight(
        prompt,
        default_model,
        max_n_logits=4,
        batch_size=8,
        max_feature_nodes=64,
        offload="cpu",
    )

    split_model = _load_gemmascope2_model()
    graph_split = attribute_nnsight(
        prompt,
        split_model,
        max_n_logits=4,
        batch_size=8,
        max_feature_nodes=64,
        offload="cpu",
        feature_batch_size=4,
        logit_batch_size=2,
        profile=True,
        profile_log_interval=1,
    )

    _assert_graphs_close(graph_default, graph_split)


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemmascope2_exact_chunked_sequential_reuse_matches_fresh_model():
    prompt = "If Alice has 3 apples and buys 2 more, she has"

    fresh_model = _load_gemmascope2_model()
    fresh_graph = attribute_nnsight(
        prompt,
        fresh_model,
        max_n_logits=4,
        batch_size=8,
        max_feature_nodes=64,
        offload="cpu",
    )

    reused_model = _load_gemmascope2_model()
    first_graph = attribute_nnsight(
        prompt,
        reused_model,
        max_n_logits=4,
        batch_size=8,
        max_feature_nodes=64,
        offload="cpu",
    )
    second_graph = attribute_nnsight(
        prompt,
        reused_model,
        max_n_logits=4,
        batch_size=8,
        max_feature_nodes=64,
        offload="cpu",
        profile=True,
        profile_log_interval=1,
    )

    _assert_graphs_close(fresh_graph, first_graph)
    _assert_graphs_close(fresh_graph, second_graph)
