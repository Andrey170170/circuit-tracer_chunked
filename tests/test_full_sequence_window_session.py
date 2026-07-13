from __future__ import annotations

import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.context_state import (
    AttributionTensorState,
    ContextExecutionPolicy,
    ContextNumericPolicy,
    DecoderRuntime,
)
from circuit_tracer.attribution.nnsight.forward_session import ForwardTraceSession
from circuit_tracer.tracing import AttributionProblem, PrefixViewTarget, TraceRequest


class FakeBackendModel:
    backend = "nnsight"


def _request(
    model,
    *,
    output_position: int | None = None,
    prefix_view: PrefixViewTarget | None = None,
) -> TraceRequest:
    return TraceRequest(
        problem=AttributionProblem(
            model=model,
            prompt=[1, 2, 3, 4],
            output_position=output_position,
            prefix_view=prefix_view,
        )
    )


def _sparse_activation() -> torch.Tensor:
    indices = torch.tensor([[0, 0, 1, 1], [0, 2, 1, 3], [5, 6, 7, 8]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return torch.sparse_coo_tensor(indices, values, size=(2, 4, 10)).coalesce()


def _context(
    *,
    activation_matrix: torch.Tensor,
    error_vectors: torch.Tensor,
    token_vectors: torch.Tensor,
    decoder_vecs: torch.Tensor,
    encoder_vecs: torch.Tensor,
    encoder_to_decoder_map: torch.Tensor,
    decoder_locations: torch.Tensor,
    logits: torch.Tensor,
    full_logits: torch.Tensor | None = None,
    chunked_decoder_state: dict[str, torch.Tensor] | None = None,
) -> AttributionContext:
    return AttributionContext(
        tensor_state=AttributionTensorState(
            activation_matrix=activation_matrix,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vectors=decoder_vecs,
            encoder_vectors=encoder_vecs,
            encoder_to_decoder_map=encoder_to_decoder_map,
            decoder_locations=decoder_locations,
            logits=logits,
            full_logits=full_logits,
        ),
        execution_policy=ContextExecutionPolicy.resolve(
            chunked_decoder_state=chunked_decoder_state,
            encoder_vectors=encoder_vecs,
            error_vectors=error_vectors,
            exact_encoder_residency="lazy",
            stage_encoder_vectors_on_cpu=False,
            stage_error_vectors_on_cpu=False,
            error_vector_prefetch_lookahead=1,
            chunked_feature_replay_window=4,
            row_subchunk_size=None,
        ),
        decoder_runtime=DecoderRuntime.resolve(
            provider=None,
            chunked_state=chunked_decoder_state,
        ),
        numeric_policy=ContextNumericPolicy(),
    )


def test_derive_prefix_view_context_filters_without_mutating_parent() -> None:
    activation = _sparse_activation()
    ctx = _context(
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


def test_forward_session_prepares_prefix_when_reuse_disabled() -> None:
    model = FakeBackendModel()
    selected = _request(
        model,
        output_position=1,
        prefix_view=PrefixViewTarget(mode="independent_prefix", target_position=2),
    )
    session = ForwardTraceSession(
        model=model,
        full_token_ids=[1, 2, 3, 4],
        window_max_prefix_len=4,
        reuse_phase0_window_state=False,
        reuse_target_logits=False,
    )

    problem, overrides = session.prepare_target_position(2, selected)
    assert problem.prompt.tolist() == [1, 2]
    assert problem.output_position == 1
    assert overrides.decoder_chunk_cache is None


def test_forward_session_reuses_context_and_window_logits() -> None:
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

    model = FakeModel()
    selected = _request(
        model,
        output_position=2,
        prefix_view=PrefixViewTarget(
            mode="full_sequence_target_position",
            target_position=3,
        ),
    )
    session = ForwardTraceSession(
        model=model,
        full_token_ids=[1, 2, 3, 4, 5],
        window_max_prefix_len=4,
        reuse_phase0_window_state=True,
        reuse_target_logits=True,
    )

    problem, overrides = session.prepare_target_position(3, selected)
    assert problem is selected.problem
    assert model.setup_calls == 1
    assert overrides.phase0_context is model.ctx.derived
    assert overrides.target_logit_source == "full_sequence_window_logits"
    assert torch.equal(overrides.target_logits, torch.arange(10, dtype=torch.float32))
    session.close()
    assert model.ctx.cleaned
