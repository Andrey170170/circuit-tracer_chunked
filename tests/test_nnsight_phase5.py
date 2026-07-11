from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import circuit_tracer.attribution.nnsight.phases.phase5 as phase5
from circuit_tracer.attribution.nnsight.phases.phase5 import (
    Phase5Config,
    Phase5Inputs,
    run_phase5,
)


class FakeObserver:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.phases: list[dict[str, object]] = []

    def phase(self, **payload: object) -> None:
        self.phases.append(payload)
        if self.failure is not None:
            raise self.failure


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
        compact_output=compact_output,
        use_compact_feature_row_store=False,
        capture_feature_semantic_descriptors_enabled=False,
        capture_phase0_donor_bundle_enabled=False,
        capture_phase3_seed_bundle_enabled=False,
        capture_phase3_gradient_bundle_enabled=False,
        capture_phase3_row_bundle_enabled=False,
        cross_cluster_debug_enabled=False,
        phase4_anomaly_debug_enabled=False,
        n_pos=1,
        n_logits=1,
        st=2,
        total_active_feats=1,
        total_nodes=4,
        actual_max_feature_nodes=1,
        batch_size=2,
        feature_batch_size=None,
        max_phase4_feature_batch_size=4,
        planner_enabled=False,
        planner_status="disabled",
        planner_skip_reason=None,
        phase4_scheduler_config=policy,
        phase4_refresh_optimization_config=policy,
        phase4_row_executor_config=policy,
        phase4_row_reduction_config=policy,
        phase1_trace_batch_metadata={"trace_batch_size": 1},
        internal_precision_requested="fp32",
        resolved_dtype_map={"edge_matrix": "float32"},
        phase0_activation_threshold_compare_mode_resolved="fp32",
        exact_trace_internal_dtype_name="fp32",
        telemetry_max_events_resolved=123,
        semantic_descriptor_top_k=8,
        semantic_descriptor_dim=4,
        phase4_feature_batch_size=2,
        phase4_executor_reference_batch_size=2,
        phase4_executor_microbatch_size=1,
        phase4_refresh_count=3,
        phase4_scheduler_reference_batch_count=4,
        phase4_executor_microbatch_count=5,
        phase4_elapsed_ms=6.0,
        phase4_refresh_elapsed_ms_total=7.0,
        phase4_feature_batch_elapsed_ms_total=8.0,
        phase4_refresh_partial_influence_elapsed_ms_total=9.0,
        phase4_refresh_rank_topk_elapsed_ms_total=10.0,
        phase4_refresh_frontier_plan_elapsed_ms_total=11.0,
        phase4_refresh_row_store_read_elapsed_ms_total=12.0,
        phase4_refresh_prepared_chunk_cache_bytes=0,
        phase4_refresh_prepared_chunk_cache_bytes_effective=0,
        phase4_refresh_active_row_accumulation="off",
        phase4_refresh_active_row_accumulation_effective="off",
        phase4_refresh_aux_fallback_reason=None,
        phase4_refresh_aux_applicable=False,
        start_time=0.0,
        phase0_context_override=None,
        target_logit_source="context",
        target_logits_override=None,
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
        telemetry_observer=observer,
        activation_matrix=activation_matrix,
        visited=torch.tensor([True]),
        edge_matrix=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        row_to_node_index=torch.tensor([3, 0]),
        input_ids=torch.tensor([11]),
        feature_row_store=None,
        nonfeature_row_store=None,
        phase0_replay_metadata={"status": "native", "mode": "off"},
        phase3_gradient_replay_metadata={"status": "native", "mode": "off"},
        phase3_row_replay_metadata={"status": "native", "mode": "off"},
        phase3_frontier_buffer_metadata={"phase": 3},
        phase4_frontier_buffer_metadata={"phase": 4},
        phase4_execution_metadata={"executor": "reference"},
        phase0_donor_bundle_payload=None,
        phase3_seed_bundle_payload=None,
        phase3_gradient_bundle_payload=None,
        phase3_row_bundle_payload=None,
        feature_semantic_descriptors_payload=None,
        cross_cluster_debug_summary=None,
        cross_cluster_debug_checkpoints=None,
        cross_cluster_debug_batches=None,
        prefix_view_metadata=None,
        publish_compact_output_result=published.append,
        release_dense_edge_matrix=lambda: released.append(True),
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

    monkeypatch.setattr(phase5, "Graph", FakeGraph)
    inputs = _inputs(FakeObserver(), published, released)
    result = run_phase5(inputs=inputs, config=_config(compact_output=False))

    assert isinstance(result.output, FakeGraph)
    assert result.compact_output_result is None
    assert result.edge_matrix is inputs.edge_matrix
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
        phase5,
        "validate_compact_prefix_view_output",
        lambda compact, *, n_layers: validated.append((compact, n_layers)),
    )
    inputs = replace(
        _inputs(FakeObserver(), published, released),
        feature_row_store=feature_store,
        nonfeature_row_store=nonfeature_store,
        prefix_view_metadata={"target_position": 1},
    )

    result = run_phase5(
        inputs=inputs,
        config=replace(_config(), use_compact_feature_row_store=True),
    )

    assert result.edge_matrix is inputs.edge_matrix
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
        phase5,
        "_annotate_phase4_selection_on_feature_semantic_descriptors",
        lambda payload, *, selected_features: annotation_calls.append(selected_features.clone()),
    )
    monkeypatch.setattr(
        phase5,
        "_build_cross_cluster_runtime_snapshot",
        lambda **_: ({"runtime": "summary"}, {"runtime": "stream"}),
    )

    def record_checkpoint(**kwargs: object) -> None:
        checkpoints.append({"name": kwargs["checkpoint_name"], "phase": kwargs["phase"]})

    monkeypatch.setattr(phase5, "_record_cross_cluster_checkpoint", record_checkpoint)
    inputs = replace(
        _inputs(FakeObserver(), published, released),
        feature_semantic_descriptors_payload=descriptors,
        cross_cluster_debug_summary=summary,
        cross_cluster_debug_checkpoints=checkpoints,
        cross_cluster_debug_batches=[],
    )

    result = run_phase5(
        inputs=inputs,
        config=replace(
            _config(),
            capture_feature_semantic_descriptors_enabled=True,
            cross_cluster_debug_enabled=True,
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
