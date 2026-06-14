import torch

from circuit_tracer.attribution import attribute_nnsight
from circuit_tracer.attribution.context_nnsight import AttributionContext


def _sparse_activation():
    indices = torch.tensor([[0, 0, 1, 1], [0, 2, 1, 3], [5, 6, 7, 8]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return torch.sparse_coo_tensor(indices, values, size=(2, 4, 10)).coalesce()


def test_derive_prefix_view_context_filters_without_mutating_parent():
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


def test_full_sequence_session_delegates_when_reuse_disabled(monkeypatch):
    calls = []

    def fake_attribute(prompt, model, **kwargs):
        calls.append((prompt, model, kwargs))
        return "graph"

    monkeypatch.setattr(attribute_nnsight, "attribute", fake_attribute)
    session = attribute_nnsight.FullSequenceWindowAttributionSession(
        model=object(),
        full_token_ids=[1, 2, 3, 4],
        window_max_prefix_len=4,
    )

    assert session.attribute_target_position(2, max_n_logits=1) == "graph"
    prompt, _, kwargs = calls[0]
    assert prompt.tolist() == [1, 2]
    assert kwargs["output_position"] == 1
    assert kwargs["max_n_logits"] == 1


def test_full_sequence_session_reuses_context_and_window_logits(monkeypatch):
    calls = []

    class FakeContext:
        def __init__(self):
            self.derived = object()

        def derive_prefix_view_context(self, target_position):
            assert target_position == 3
            return self.derived

        def get_logits_at_position(self, position):
            assert position == 2
            logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
            return logits

    class FakeModel:
        def __init__(self):
            self.ctx = FakeContext()
            self.setup_calls = 0

        def setup_attribution(self, token_ids, **kwargs):
            self.setup_calls += 1
            assert token_ids.tolist() == [1, 2, 3, 4]
            assert kwargs["retain_full_logits"] is True
            return self.ctx

    def fake_attribute(prompt, model, **kwargs):
        calls.append((prompt, kwargs))
        return "graph"

    monkeypatch.setattr(attribute_nnsight, "attribute", fake_attribute)
    model = FakeModel()
    session = attribute_nnsight.FullSequenceWindowAttributionSession(
        model=model,
        full_token_ids=[1, 2, 3, 4, 5],
        window_max_prefix_len=4,
        reuse_phase0_window_state=True,
        reuse_target_logits=True,
    )

    assert session.attribute_target_position(3) == "graph"
    assert model.setup_calls == 1
    _, kwargs = calls[0]
    assert kwargs["_phase0_context_override"] is model.ctx.derived
    assert kwargs["_target_logit_source"] == "full_sequence_window_logits"
    assert torch.equal(kwargs["_target_logits_override"], torch.arange(10, dtype=torch.float32))
