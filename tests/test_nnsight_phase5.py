from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution.nnsight.phases.phase5 import (
    BatchExecutionSummary,
    DiagnosticArtifacts,
    GraphAssemblyLimits,
    GraphAssemblyRuntime,
    GraphAssemblyState,
    GraphOutputOwnership,
    NumericExecutionSummary,
    OutputArtifactPolicy,
    Phase4PolicySummary,
    Phase4TimingSummary,
    Phase4WorkSummary,
    Phase5Config,
    Phase5Inputs,
    ReplayArtifacts,
    RunProvenance,
    run_phase5,
)
import circuit_tracer.attribution.nnsight.phases.phase5_full as phase5_full
import circuit_tracer.attribution.nnsight.phases.phase5_publication as phase5_publication
import circuit_tracer.attribution.nnsight.phase_support as phase_support
from circuit_tracer.observability.events import MemoryBoundary, RuntimeSnapshot, TraceEvent


class FakeObserver:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.phases: list[dict[str, object]] = []

    def observe(self, observation: object) -> object | None:
        if isinstance(observation, TraceEvent):
            self.phases.append({"name": observation.name, "attrs": dict(observation.attrs)})
            if self.failure is not None:
                raise self.failure
        elif isinstance(observation, RuntimeSnapshot):
            return {}, {}
        elif isinstance(observation, MemoryBoundary):
            return None
        return None


class FakeRowStore:
    def __init__(
        self, *, n_feature_columns: int, values: dict[tuple[int, int], torch.Tensor]
    ) -> None:
        self.n_feature_columns = n_feature_columns
        self.nbytes = 321
        self.values = values
        self.calls: list[tuple[int, int, torch.Tensor, str]] = []

    def materialize_dense_feature_slice(
        self,
        *,
        row_start: int,
        row_end: int,
        selected_feature_columns: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        self.calls.append((row_start, row_end, selected_feature_columns, phase))
        return self.values[(row_start, row_end)]


def _policy() -> SimpleNamespace:
    return SimpleNamespace(
        requested_mode="off",
        version="test-v1",
        policy="fixed",
        effective_mode="off",
        effective_version="test-v1",
        effective_policy="fixed",
        effective_behavior="reference",
        debug=False,
        telemetry_detail="minimal",
    )


def _config(*, compact_output: bool = True) -> Phase5Config:
    policy = _policy()
    return Phase5Config(
        output_policy=OutputArtifactPolicy(
            compact_output=compact_output,
            use_compact_feature_row_store=False,
            capture_feature_semantic_descriptors=False,
            capture_phase0_donor_bundle=False,
            capture_phase3_seed_bundle=False,
            capture_phase3_gradient_bundle=False,
            capture_phase3_row_bundle=False,
            cross_cluster_debug_enabled=False,
            phase4_anomaly_debug_enabled=False,
        ),
        graph_limits=GraphAssemblyLimits(1, 1, 2, 1, 4, 1),
        batches=BatchExecutionSummary(2, None, 4, False, "disabled", None, {"trace_batch_size": 1}),
        phase4_policy=Phase4PolicySummary(
            policy, policy, policy, policy, 0, 0, "off", "off", None, False
        ),
        numerics=NumericExecutionSummary(
            "fp32", {"edge_matrix": "float32"}, "fp32", "fp32", 123
        ),
        phase4_work=Phase4WorkSummary(8, 4, 2, 2, 1, 3, 4, 5),
        phase4_timings=Phase4TimingSummary(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
        provenance=RunProvenance(0.0, None, "context", None),
    )


def _inputs(
    observer: FakeObserver,
    published: list[dict[str, object]],
    released: list[bool],
) -> Phase5Inputs:
    activation_matrix = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0]]), torch.tensor([1.5]), (1, 1, 1)
    ).coalesce()
    return Phase5Inputs(
        runtime=GraphAssemblyRuntime(
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            model=SimpleNamespace(
                tokenizer=SimpleNamespace(decode=lambda tokens: "decoded"),
                cfg=SimpleNamespace(n_layers=1),
                config=SimpleNamespace(n_layers=1),
                scan="test-scan",
                device=torch.device("cpu"),
                transcoders=SimpleNamespace(),
            ),
            ctx=SimpleNamespace(),
            targets=SimpleNamespace(
                logit_targets=[], logit_probabilities=torch.tensor([0.25]), vocab_size=17
            ),
            observer=observer,
            input_ids=torch.tensor([11]),
        ),
        graph=GraphAssemblyState(
            activation_matrix=activation_matrix,
            visited=torch.tensor([True]),
            edge_matrix=torch.arange(8, dtype=torch.float32).reshape(2, 4),
            row_to_node_index=torch.tensor([3, 0]),
            feature_row_store=None,
            nonfeature_row_store=None,
        ),
        replay=ReplayArtifacts(
            {"status": "native", "mode": "off"},
            {"status": "native", "mode": "off"},
            {"status": "native", "mode": "off"},
            None, None, None, None,
        ),
        diagnostics=DiagnosticArtifacts(
            {"phase": 3}, {"phase": 4}, {"executor": "reference"},
            None, None, None, None,
        ),
        output=GraphOutputOwnership(
            None, published.append, lambda: released.append(True)
        ),
    )


def test_phase5_compact_result_preserves_identity_metadata_and_dense_release() -> None:
    published: list[dict[str, object]] = []
    released: list[bool] = []
    observer = FakeObserver()

    result = run_phase5(
        inputs=_inputs(observer, published, released), config=_config(compact_output=True)
    )

    assert result.output is result.compact_output_result is published[0]
    assert result.edge_matrix is None
    assert released == [True]
    assert result.output["selected_features"].tolist() == [0]
    assert result.output["phase0_replay_metadata"] == {"status": "native", "mode": "off"}
    assert result.output["phase4_scheduler_effective_behavior"] == "reference"
    assert result.output["phase4_refresh_count"] == 3
    assert result.output["telemetry_max_events"] == 123
    assert result.output["target_logit_source"] == "context"
    assert observer.phases[0]["name"] == "phase5.packaging"


def test_phase5_noncompact_returns_graph_output(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[dict[str, object]] = []
    released: list[bool] = []
    captured: dict[str, object] = {}

    class FakeGraph:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(phase5_full, "Graph", FakeGraph)
    inputs = _inputs(FakeObserver(), published, released)
    result = run_phase5(inputs=inputs, config=_config(compact_output=False))

    assert isinstance(result.output, FakeGraph)
    assert result.compact_output_result is None
    assert result.edge_matrix is inputs.graph.edge_matrix
    assert published == []
    assert released == []
    assert torch.equal(captured["selected_features"], torch.tensor([0]))
    assert torch.equal(
        captured["adjacency_matrix"],
        torch.tensor(
            [
                [4.0, 5.0, 6.0, 7.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        ),
    )


def test_phase5_failure_after_compact_publication_preserves_attachment_identity() -> None:
    published: list[dict[str, object]] = []
    released: list[bool] = []
    failure = RuntimeError("injected packaging failure")

    with pytest.raises(RuntimeError, match="injected packaging failure"):
        run_phase5(
            inputs=_inputs(FakeObserver(failure), published, released),
            config=_config(compact_output=True),
        )

    assert len(published) == 1
    assert published[0]["selected_features"].tolist() == [0]
    assert published[0]["phase4_scheduler_effective_behavior"] == "reference"
    assert released == [True]


def test_phase5_compact_row_stores_materialize_cpu_slices_and_validate_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[dict[str, object]] = []
    released: list[bool] = []
    feature_store = FakeRowStore(
        n_feature_columns=1,
        values={(1, 2): torch.tensor([[10.0]]), (0, 1): torch.tensor([[20.0]])},
    )
    nonfeature_store = FakeRowStore(
        n_feature_columns=2,
        values={
            (1, 2): torch.tensor([[30.0, 31.0]]),
            (0, 1): torch.tensor([[40.0, 41.0]]),
        },
    )
    validated: list[tuple[dict[str, object], int]] = []
    monkeypatch.setattr(
        phase5_publication,
        "validate_compact_prefix_view_output",
        lambda compact, *, n_layers: validated.append((compact, n_layers)),
    )
    inputs = _inputs(FakeObserver(), published, released)
    inputs = replace(
        inputs,
        graph=replace(
            inputs.graph,
            feature_row_store=feature_store,
            nonfeature_row_store=nonfeature_store,
        ),
        output=replace(inputs.output, prefix_view_metadata={"target_position": 1}),
    )

    result = run_phase5(
        inputs=inputs,
        config=replace(
            _config(),
            output_policy=replace(
                _config().output_policy, use_compact_feature_row_store=True
            ),
        ),
    )

    assert result.edge_matrix is inputs.graph.edge_matrix
    assert released == []
    assert result.output["feature_feature_edges"].tolist() == [[10.0]]
    assert result.output["logit_feature_edges"].tolist() == [[20.0]]
    assert result.output["feature_error_edges"].tolist() == [[30.0]]
    assert result.output["feature_token_edges"].tolist() == [[31.0]]
    assert result.output["logit_error_edges"].tolist() == [[40.0]]
    assert result.output["logit_token_edges"].tolist() == [[41.0]]
    assert [(call[0], call[1], call[2].tolist(), call[3]) for call in feature_store.calls] == [
        (1, 2, [0], "phase5"),
        (0, 1, [0], "phase5"),
    ]
    assert [(call[0], call[1], call[2].tolist(), call[3]) for call in nonfeature_store.calls] == [
        (1, 2, [0, 1], "phase5"),
        (0, 1, [0, 1], "phase5"),
    ]
    assert validated == [(result.output, 1)]


def test_phase5_annotates_descriptors_and_records_cross_cluster_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[dict[str, object]] = []
    released: list[bool] = []
    descriptors: dict[str, object] = {}
    summary: dict[str, object] = {}
    checkpoints: list[dict[str, object]] = []
    annotation_calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        phase_support,
        "_annotate_phase4_selection_on_feature_semantic_descriptors",
        lambda payload, *, selected_features: annotation_calls.append(selected_features.clone()),
    )

    def record_checkpoint(**kwargs: object) -> None:
        checkpoints.append({"name": kwargs["checkpoint_name"], "phase": kwargs["phase"]})

    monkeypatch.setattr(
        phase5_publication, "_record_cross_cluster_checkpoint", record_checkpoint
    )
    inputs = _inputs(FakeObserver(), published, released)
    inputs = replace(
        inputs,
        diagnostics=replace(
            inputs.diagnostics,
            feature_semantic_descriptors_payload=descriptors,
            cross_cluster_debug_summary=summary,
            cross_cluster_debug_checkpoints=checkpoints,
            cross_cluster_debug_batches=[],
        ),
    )

    result = run_phase5(
        inputs=inputs,
        config=replace(
            _config(),
            output_policy=replace(
                _config().output_policy,
                capture_feature_semantic_descriptors=True,
                cross_cluster_debug_enabled=True,
            ),
        ),
    )

    assert [selected.tolist() for selected in annotation_calls] == [[0]]
    assert result.output["feature_semantic_descriptors"] is descriptors
    assert summary == {
        "status": "captured",
        "checkpoint_stream_count": 2,
        "batch_event_stream_count": 0,
    }
    assert checkpoints == [
        {"name": "phase4_entry", "phase": "phase4"},
        {"name": "phase4_run_summary", "phase": "phase4"},
    ]
    assert result.output["cross_cluster_debug_summary"] is summary
    assert result.output["cross_cluster_debug_checkpoints"] is checkpoints
