from types import SimpleNamespace

import pytest

from circuit_tracer import (
    AllActiveSources,
    AttributionProblem,
    TokenPositionSources,
    TraceRequest,
    compile_source_selection,
    resolve_trace_request,
)


class _Model:
    backend = "nnsight"
    config = SimpleNamespace(_name_or_path="test-model", architectures=("TestModel",))


def test_nnsight_provider_identity_uses_scan_name_and_model_revision() -> None:
    def fingerprint(*, scan_name: str, revision: str, scan) -> str:
        model = _Model()
        model.scan_name = scan_name
        model.scan = scan
        model.revision = revision
        return resolve_trace_request(
            TraceRequest(problem=AttributionProblem(model=model, prompt=[1, 2, 3]))
        ).semantic_fingerprint

    baseline = fingerprint(
        scan_name="local/topk@abc", revision="model-revision", scan=lambda: None
    )
    assert baseline == fingerprint(
        scan_name="local/topk@abc", revision="model-revision", scan=lambda: "different"
    )
    assert baseline != fingerprint(
        scan_name="local/topk@def", revision="model-revision", scan=lambda: None
    )
    assert baseline != fingerprint(
        scan_name="local/topk@abc", revision="other-revision", scan=lambda: None
    )


def _request(source_selection=AllActiveSources(), *, backend="nnsight") -> TraceRequest:
    model = _Model()
    model.backend = backend
    return TraceRequest(
        problem=AttributionProblem(
            model=model,
            prompt=[1, 2, 3],
            source_selection=source_selection,
        )
    )


def test_source_selection_values_are_strict_and_canonical() -> None:
    assert AllActiveSources().kind == "all_active"
    assert AllActiveSources().version == 1
    selected = TokenPositionSources(positions=(0, 2), max_features_per_position=7)
    assert selected.kind == "token_positions"
    assert selected.version == 1

    with pytest.raises(ValueError, match="tuple"):
        TokenPositionSources(positions=[0, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        TokenPositionSources(positions=())
    with pytest.raises(ValueError, match="sorted and unique"):
        TokenPositionSources(positions=(2, 0))
    with pytest.raises(ValueError, match="sorted and unique"):
        TokenPositionSources(positions=(0, 0))
    with pytest.raises(ValueError, match="nonnegative integers"):
        TokenPositionSources(positions=(-1,))
    with pytest.raises(ValueError, match="positive integer"):
        TokenPositionSources(positions=(0,), max_features_per_position=0)


def test_source_selection_changes_semantic_fingerprint_but_default_is_compatible() -> None:
    implicit = resolve_trace_request(
        TraceRequest(problem=AttributionProblem(model=_Model(), prompt=[1, 2, 3]))
    )
    explicit = resolve_trace_request(_request(AllActiveSources()))
    restricted = resolve_trace_request(_request(TokenPositionSources(positions=(0, 2))))

    assert implicit.semantic_fingerprint == explicit.semantic_fingerprint
    assert restricted.semantic_fingerprint != implicit.semantic_fingerprint


def test_transformerlens_rejects_restricted_source_selection() -> None:
    with pytest.raises(ValueError, match="restricted source selection requires the NNSight"):
        resolve_trace_request(
            _request(TokenPositionSources(positions=(0,)), backend="transformerlens")
        )


def test_token_position_selection_compiles_exact_membership_with_stable_ties() -> None:
    import torch

    # Canonical sparse order gives global active-feature indices 0..6.  At
    # position 1, indices 1 and 4 tie at |3|; the lower global index wins.
    indices = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 1, 2], [4, 1, 2, 0, 3, 5, 2]]
    )
    values = torch.tensor([9.0, 3.0, 8.0, 2.0, -3.0, 1.0, 7.0])
    activations = torch.sparse_coo_tensor(indices, values, size=(2, 3, 8)).coalesce()

    eligible = compile_source_selection(
        TokenPositionSources(positions=(1, 2), max_features_per_position=2),
        activations,
        target_position=2,
    )

    assert eligible.tolist() == [1, 2, 4, 6]


def test_token_position_selection_rejects_out_of_range_and_future_positions() -> None:
    import torch

    activations = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [0, 2], [1, 1]]),
        torch.tensor([1.0, 2.0]),
        size=(1, 3, 4),
    ).coalesce()

    with pytest.raises(ValueError, match="future source position"):
        compile_source_selection(
            TokenPositionSources(positions=(2,)), activations, target_position=1
        )
    with pytest.raises(ValueError, match="out of range"):
        compile_source_selection(
            TokenPositionSources(positions=(3,)), activations, target_position=3
        )


def test_nnsight_feature_plan_applies_global_cap_after_source_eligibility() -> None:
    import torch

    from circuit_tracer.attribution.nnsight.phases.phase2_storage import (
        FrontierBufferPolicy,
        Phase2ExecutionPolicy,
        plan_active_feature_storage,
    )

    indices = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 1, 2], [4, 1, 2, 0, 3, 5, 2]]
    )
    values = torch.tensor([9.0, 3.0, 8.0, 2.0, -3.0, 1.0, 7.0])
    activations = torch.sparse_coo_tensor(indices, values, size=(2, 3, 8)).coalesce()

    plan = plan_active_feature_storage(
        logger=SimpleNamespace(info=lambda _message: None),
        model=object(),
        activation_matrix=activations,
        n_logits=2,
        offload_handles=[],
        frontier=FrontierBufferPolicy(None, 0, None, 0, 0),
        execution=Phase2ExecutionPolicy(
            offload=None,
            max_feature_nodes=3,
            compact_output=False,
            exact_chunked_decoder=True,
            use_compact_feature_row_store=True,
            exact_dtype=torch.float32,
            effective_feature_batch_size=2,
            trace_batch_size=1,
            source_selection=TokenPositionSources(
                positions=(1, 2), max_features_per_position=2
            ),
            target_position=2,
        ),
    )

    assert plan.eligible_feature_indices is not None
    assert plan.eligible_feature_indices.tolist() == [1, 2, 4, 6]
    assert plan.base_max_feature_nodes == 3
    assert plan.actual_max_feature_nodes == 3
    assert plan.row_store_capacity_feature_nodes == 3
    assert plan.total_active_feats == 7
