from __future__ import annotations

import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight import forward_session
from circuit_tracer.attribution.nnsight.forward_session import ForwardTraceSession
from circuit_tracer.tracing import AttributionProblem, TraceRequest, resolve_trace_request


class FakeBackendModel:
    backend = "nnsight"


def _request(model, *, output_position: int | None = None) -> TraceRequest:
    return TraceRequest(
        problem=AttributionProblem(
            model=model,
            prompt=[1, 2, 3, 4],
            output_position=output_position,
        )
    )


def _sparse_activation() -> torch.Tensor:
    indices = torch.tensor([[0, 0, 1, 1], [0, 2, 1, 3], [5, 6, 7, 8]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return torch.sparse_coo_tensor(indices, values, size=(2, 4, 10)).coalesce()


def test_derive_prefix_view_context_filters_without_mutating_parent() -> None:
    activation = _sparse_activation()
    ctx = AttributionContext(
        activation_matrix=activation,
        error_vectors=torch.randn(2, 4, 3),
        token_vectors=torch.randn(4, 3),
        decoder_vecs=torch.empty(0, 3),
        encoder_vecs=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        encoder_to_decoder_map=torch.empty(0, dtype=torch.long),
        decoder_locations=torch.empty(2, 0, dtype=torch.long),
        logits=torch.randn(1, 4, 11),
        full_logits=torch.randn(1, 4, 11),
        chunked_decoder_state={
            "source_layers": activation.indices()[0].contiguous(),
            "positions": activation.indices()[1].contiguous(),
            "feature_ids": activation.indices()[2].contiguous(),
            "activation_values": activation.values().contiguous(),
        },
    )

    view = ctx.derive_prefix_view_context(2)

    assert tuple(ctx.activation_matrix.shape) == (2, 4, 10)
    assert int(ctx.activation_matrix._nnz()) == 4
    assert tuple(view.activation_matrix.shape) == (2, 2, 10)
    assert int(view.activation_matrix._nnz()) == 2
    assert view.token_vectors.shape[0] == 2
    assert view.error_vectors.shape[1] == 2
    assert view.full_logits is ctx.full_logits
    assert view.chunked_decoder_state is not None
    assert view.chunked_decoder_state["positions"].max().item() < 2
    assert torch.equal(view.encoder_vecs, ctx.encoder_vecs[torch.tensor([0, 2])])


def test_forward_session_delegates_prefix_when_reuse_disabled(monkeypatch) -> None:
    calls = []

    def fake_run(problem, plan, **kwargs):
        calls.append((problem, plan, kwargs))
        return "graph"

    monkeypatch.setattr(forward_session, "run_nnsight_trace", fake_run)
    model = FakeBackendModel()
    selected = _request(model, output_position=1)
    plan = resolve_trace_request(selected)
    session = ForwardTraceSession(
        model=model,
        full_token_ids=[1, 2, 3, 4],
        window_max_prefix_len=4,
        reuse_phase0_window_state=False,
        reuse_target_logits=False,
    )

    assert session.trace_target_position(2, selected, plan) == "graph"
    problem, _, kwargs = calls[0]
    assert problem.prompt.tolist() == [1, 2]
    assert problem.output_position == 1
    assert kwargs == {}


def test_forward_session_reuses_context_and_window_logits(monkeypatch) -> None:
    calls = []

    class FakeContext:
        def __init__(self):
            self.derived = object()
            self.cleaned = False

        def derive_prefix_view_context(self, target_position):
            assert target_position == 3
            return self.derived

        def get_logits_at_position(self, position):
            assert position == 2
            return torch.arange(10, dtype=torch.float32).reshape(1, 10)

        def cleanup(self):
            self.cleaned = True

    class FakeModel(FakeBackendModel):
        def __init__(self):
            self.ctx = FakeContext()
            self.setup_calls = 0

        def setup_attribution(self, token_ids, **kwargs):
            self.setup_calls += 1
            assert token_ids.tolist() == [1, 2, 3, 4]
            assert kwargs["retain_full_logits"] is True
            return self.ctx

    def fake_run(problem, plan, **kwargs):
        calls.append((problem, plan, kwargs))
        return "graph"

    monkeypatch.setattr(forward_session, "run_nnsight_trace", fake_run)
    model = FakeModel()
    selected = _request(model, output_position=2)
    plan = resolve_trace_request(selected)
    session = ForwardTraceSession(
        model=model,
        full_token_ids=[1, 2, 3, 4, 5],
        window_max_prefix_len=4,
        reuse_phase0_window_state=True,
        reuse_target_logits=True,
    )

    assert session.trace_target_position(3, selected, plan) == "graph"
    assert model.setup_calls == 1
    overrides = calls[0][2]["forward_overrides"]
    assert overrides.phase0_context is model.ctx.derived
    assert overrides.target_logit_source == "full_sequence_window_logits"
    assert torch.equal(overrides.target_logits, torch.arange(10, dtype=torch.float32))
    session.close()
    assert model.ctx.cleaned
