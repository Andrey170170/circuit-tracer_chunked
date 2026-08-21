import gc

import pytest
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)
from circuit_tracer.tracing import AttributionProblem, ExecutionConstraints, TraceRequest, trace_one
from tests.conftest import has_32gb


def _trace_graph(prompt, model, *, offload: str | None = None):
    return trace_one(
        TraceRequest(
            problem=AttributionProblem(prompt=prompt, model=model),
            execution=ExecutionConstraints(offload=offload),
        )
    ).graph


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_offload_tl():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
    assert isinstance(model, TransformerLensReplacementModel)

    original_device = model.cfg.device  # type:ignore
    assert isinstance(original_device, torch.device)

    _trace_graph(s, model)
    with pytest.raises(ValueError, match="only default execution constraints"):
        _trace_graph(s, model, offload="cpu")

    for param in model.parameters():
        assert param.device.type == original_device.type

    with pytest.raises(ValueError, match="only default execution constraints"):
        _trace_graph(s, model, offload="disk")

    for param in model.parameters():
        assert param.device.type == original_device.type


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_offload_nnsight():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)

    original_device = model.device
    assert isinstance(original_device, torch.device)

    graph_none = _trace_graph(s, model)
    graph_cpu = _trace_graph(s, model, offload="cpu")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_cpu.adjacency_matrix, atol=1e-5, rtol=1e-3
    )

    for param in model.parameters():
        assert param.device.type == original_device.type

    graph_disk = _trace_graph(s, model, offload="disk")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_disk.adjacency_matrix, atol=1e-5, rtol=1e-3
    )

    for param in model.parameters():
        assert param.device.type == original_device.type


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_offload_nnsight_gemma_3():
    s = "The National Digital Analytics Group (ND"
    model_name = "google/gemma-3-4b-pt"
    transcoder_set = "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine"
    model = ReplacementModel.from_pretrained(model_name, transcoder_set, backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)

    original_device = model.device
    assert isinstance(original_device, torch.device)

    _trace_graph(s, model, offload="cpu")
    for param in model.parameters():
        assert param.device.type == original_device.type


if __name__ == "__main__":
    torch.manual_seed(42)
    test_offload_tl()
    test_offload_nnsight()
    test_offload_nnsight_gemma_3()
