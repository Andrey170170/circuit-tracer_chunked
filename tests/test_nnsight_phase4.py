from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import circuit_tracer.attribution.nnsight.phases.phase4 as phase4
from circuit_tracer.attribution.nnsight.phases.phase4 import (
    Phase4Config,
    Phase4Inputs,
    run_phase4,
)


class FakeObserver:
    def __init__(self) -> None:
        self.phases: list[dict[str, object]] = []
        self.batches: list[dict[str, object]] = []

    def phase(self, **payload: object) -> None:
        self.phases.append(payload)

    def batch(self, **payload: object) -> None:
        self.batches.append(payload)


class FakeRowStore:
    def __init__(self) -> None:
        self.append_calls: list[dict[str, object]] = []

    def append_rows(
        self,
        *,
        row_start: int,
        feature_rows: torch.Tensor,
        row_denominator_scaled_l1: tuple[torch.Tensor, torch.Tensor],
        phase: str,
    ) -> dict[str, float]:
        self.append_calls.append(
            {
                "row_start": row_start,
                "feature_rows": feature_rows.clone(),
                "row_denominator_scaled_l1": tuple(
                    value.clone() for value in row_denominator_scaled_l1
                ),
                "phase": phase,
            }
        )
        return {}


def _policy(**values: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "requested_mode": "off",
        "requested_policy": "fixed",
        "requested_interval_multiplier": 1,
        "effective_interval_multiplier": 1,
        "effective_queue_multiplier": 1,
        "version": "test",
        "policy": "test",
        "effective_mode": "off",
        "effective_version": "test",
        "effective_policy": "fixed",
        "effective_behavior": "test",
        "debug": False,
        "telemetry_detail": "minimal",
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _config() -> Phase4Config:
    return Phase4Config(
        actual_max_feature_nodes=0,
        total_active_feats=0,
        n_logits=1,
        logit_offset=1,
        effective_feature_batch_size=2,
        compute_microbatch_max_rows=2,
        max_phase4_feature_batch_size=4,
        update_interval=1,
        row_store_capacity_feature_nodes=0,
        exact_trace_internal_dtype_resolved=torch.float32,
        influence_compute_dtype=torch.float32,
        shadow_debug_compute_dtype=torch.float64,
        exact_chunked_decoder=False,
        use_compact_feature_row_store=False,
        planner_enabled=False,
        planner_status="disabled",
        planner_skip_reason=None,
        phase4_debug_summary_enabled=False,
        cross_cluster_debug_enabled=False,
        phase4_frontier_buffer_relative_epsilon=None,
        phase4_frontier_buffer_max_extra_per_refresh=0,
        phase4_frontier_buffer_max_extra_total=0,
        phase4_refresh_prepared_chunk_cache_bytes_effective=0,
        phase4_refresh_active_row_accumulation_effective="off",
        phase4_scheduler_config=_policy(),
        phase4_refresh_optimization_config=_policy(),
        phase4_refresh_policy_config=_policy(),
        phase4_ranker_config=_policy(),
        phase4_row_executor_config=_policy(),
        phase4_row_reduction_config=_policy(),
        row_store_cache_control_config=_policy(),
        exact_encoder_residency_config=_policy(),
        profile=False,
        profile_log_interval=1,
        verbose=False,
    )


def _inputs(observer: FakeObserver) -> Phase4Inputs:
    row_to_node_index = torch.tensor([17])
    rows_cpu_staging = torch.empty((2, 1))
    anomaly_debug_result: dict[str, object] = {"records": []}
    return Phase4Inputs(
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        model=SimpleNamespace(
            device=torch.device("cpu"),
            transcoders=SimpleNamespace(decoder_chunk_size=None),
        ),
        ctx=SimpleNamespace(),
        targets=SimpleNamespace(logit_probabilities=torch.tensor([1.0])),
        edge_matrix=torch.zeros((1, 1)),
        feat_ids=torch.empty(0, dtype=torch.long),
        feat_layers=torch.empty(0, dtype=torch.long),
        feat_pos=torch.empty(0, dtype=torch.long),
        feature_row_store=None,
        nonfeature_row_store=None,
        row_to_node_index=row_to_node_index,
        telemetry_observer=observer,
        cross_cluster_debug_summary=None,
        cross_cluster_debug_checkpoints=None,
        cross_cluster_debug_batches=None,
        anomaly_debug_result=anomaly_debug_result,
        phase4_frontier_buffer_metadata={"enabled": False},
        phase4_execution_metadata={"contract_marker": "preserved"},
        rows_cpu_staging=rows_cpu_staging,
    )


def test_phase4_returns_boundary_state_without_replacing_owned_buffers(monkeypatch) -> None:
    observer = FakeObserver()
    inputs = _inputs(observer)
    monkeypatch.setattr(phase4, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase4, "_log_phase_metrics", lambda *args, **kwargs: None)

    result = run_phase4(inputs=inputs, config=_config())

    assert result.visited.dtype == torch.bool
    assert result.visited.numel() == 0
    assert result.actual_max_feature_nodes == 0
    assert result.edge_matrix is inputs.edge_matrix
    assert result.feature_row_store is inputs.feature_row_store
    assert result.nonfeature_row_store is inputs.nonfeature_row_store
    assert result.row_to_node_index is inputs.row_to_node_index
    assert result.rows_cpu_staging is inputs.rows_cpu_staging
    assert result.st == 1
    assert result.phase4_frontier_buffer_metadata is inputs.phase4_frontier_buffer_metadata
    assert result.phase4_execution_metadata is inputs.phase4_execution_metadata
    assert result.cross_cluster_debug_summary is inputs.cross_cluster_debug_summary
    assert result.cross_cluster_debug_checkpoints is inputs.cross_cluster_debug_checkpoints
    assert result.cross_cluster_debug_batches is inputs.cross_cluster_debug_batches
    assert result.anomaly_debug_result is inputs.anomaly_debug_result
    assert result.anomaly_debug_result["status"] == "captured_refresh_debug"
    assert result.phase4_feature_batch_size == 2
    assert result.phase4_executor_reference_batch_size == 2
    assert result.phase4_executor_microbatch_size == 2
    assert result.phase4_refresh_count == 0
    assert result.phase4_scheduler_reference_batch_count == 0
    assert result.phase4_executor_microbatch_count == 0
    assert len(observer.phases) == 1


def _nonzero_inputs(observer: FakeObserver) -> Phase4Inputs:
    inputs = _inputs(observer)
    return replace(
        inputs,
        ctx=SimpleNamespace(
            encoder_vecs=torch.tensor([[2.0]]),
            materialize_encoder_vectors=lambda idx_batch: torch.tensor([[2.0]]),
            compute_batch=lambda **kwargs: torch.tensor([[3.0, -4.0]]),
        ),
        edge_matrix=torch.zeros((2, 2)),
        feat_ids=torch.tensor([23]),
        feat_layers=torch.tensor([0]),
        feat_pos=torch.tensor([0]),
        row_to_node_index=torch.tensor([7, -1]),
        cross_cluster_debug_summary={},
        cross_cluster_debug_checkpoints=[],
        cross_cluster_debug_batches=[],
        phase4_frontier_buffer_metadata={"enabled": False},
    )


def test_phase4_nonzero_dense_execution_writes_rows_and_returns_owned_state(
    monkeypatch,
) -> None:
    observer = FakeObserver()
    inputs = _nonzero_inputs(observer)
    config = replace(
        _config(), actual_max_feature_nodes=1, total_active_feats=1, logit_offset=2
    )
    monkeypatch.setattr(phase4, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase4, "_log_phase_metrics", lambda *args, **kwargs: None)

    result = run_phase4(inputs=inputs, config=config)

    assert result.visited.tolist() == [True]
    assert result.st == 2
    assert result.edge_matrix is inputs.edge_matrix
    assert torch.equal(result.edge_matrix[1], torch.tensor([3.0, -4.0]))
    assert result.row_to_node_index is inputs.row_to_node_index
    assert result.row_to_node_index.tolist() == [7, 0]
    assert result.rows_cpu_staging is inputs.rows_cpu_staging
    assert result.phase4_execution_metadata is inputs.phase4_execution_metadata
    assert result.phase4_execution_metadata["executor_reference_batch_size"] == 2
    assert result.phase4_frontier_buffer_metadata is inputs.phase4_frontier_buffer_metadata
    assert result.phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] == 1
    assert result.cross_cluster_debug_summary is inputs.cross_cluster_debug_summary
    assert result.cross_cluster_debug_checkpoints is inputs.cross_cluster_debug_checkpoints
    assert result.cross_cluster_debug_batches is inputs.cross_cluster_debug_batches
    assert len(result.cross_cluster_debug_checkpoints) == 1
    assert len(result.cross_cluster_debug_batches) == 1
    assert result.phase4_scheduler_reference_batch_count == 1
    assert result.phase4_executor_microbatch_count == 1
    assert result.phase4_refresh_count == 0
    assert len(observer.batches) == 1
    assert observer.batches[0]["attrs"]["visited_features"] == 1
    assert len(observer.phases) == 1
    assert observer.phases[0]["attrs"]["phase4_executor_microbatch_count"] == 1


def test_phase4_nonzero_compact_execution_appends_partitioned_rows_to_owned_stores(
    monkeypatch,
) -> None:
    observer = FakeObserver()
    feature_row_store = FakeRowStore()
    nonfeature_row_store = FakeRowStore()
    inputs = replace(
        _nonzero_inputs(observer),
        feature_row_store=feature_row_store,
        nonfeature_row_store=nonfeature_row_store,
    )
    config = replace(
        _config(),
        actual_max_feature_nodes=1,
        total_active_feats=1,
        logit_offset=2,
        row_store_capacity_feature_nodes=1,
        use_compact_feature_row_store=True,
    )
    monkeypatch.setattr(phase4, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase4, "_log_phase_metrics", lambda *args, **kwargs: None)

    result = run_phase4(inputs=inputs, config=config)

    assert result.feature_row_store is feature_row_store
    assert result.nonfeature_row_store is nonfeature_row_store
    assert result.rows_cpu_staging is inputs.rows_cpu_staging
    assert result.row_to_node_index.tolist() == [7, 0]
    assert result.visited.tolist() == [True]
    assert len(feature_row_store.append_calls) == 1
    assert len(nonfeature_row_store.append_calls) == 1
    for append_call in (*feature_row_store.append_calls, *nonfeature_row_store.append_calls):
        assert append_call["row_start"] == 1
        assert append_call["phase"] == "phase4"
        row_abs_max, row_l1_scaled = append_call["row_denominator_scaled_l1"]
        assert torch.equal(row_abs_max, torch.tensor([4.0]))
        assert torch.equal(row_l1_scaled, torch.tensor([1.75]))
    assert torch.equal(
        feature_row_store.append_calls[0]["feature_rows"], torch.tensor([[3.0]])
    )
    assert torch.equal(
        nonfeature_row_store.append_calls[0]["feature_rows"], torch.tensor([[-4.0]])
    )


def test_phase4_propagates_failure_after_dense_write_without_finalizing(monkeypatch) -> None:
    observer = FakeObserver()
    inputs = _nonzero_inputs(observer)
    config = replace(
        _config(), actual_max_feature_nodes=1, total_active_feats=1, logit_offset=2
    )
    failure = RuntimeError("injected post-write phase4 failure")

    def fail_after_write(**payload: object) -> None:
        raise failure

    monkeypatch.setattr(phase4, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase4, "_log_phase_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(observer, "batch", fail_after_write)

    with pytest.raises(RuntimeError, match="injected post-write phase4 failure") as exc_info:
        run_phase4(inputs=inputs, config=config)

    assert exc_info.value is failure
    assert torch.equal(inputs.edge_matrix[1], torch.tensor([3.0, -4.0]))
    assert inputs.row_to_node_index.tolist() == [7, 0]
    assert observer.phases == []


def test_phase4_physical_microbatches_preserve_order_and_refresh_result(monkeypatch) -> None:
    def inputs_for(observer: FakeObserver) -> Phase4Inputs:
        def materialize(indices: torch.Tensor) -> torch.Tensor:
            return indices.to(dtype=torch.float32).reshape(-1, 1)

        def compute_batch(**kwargs: object) -> torch.Tensor:
            values = kwargs["inject_values"]
            assert isinstance(values, torch.Tensor)
            return torch.cat((values, values + 10, values + 20, values + 30), dim=1)

        return replace(
            _nonzero_inputs(observer),
            ctx=SimpleNamespace(materialize_encoder_vectors=materialize, compute_batch=compute_batch),
            edge_matrix=torch.zeros((4, 4)),
            feat_ids=torch.tensor([20, 21, 22]),
            feat_layers=torch.zeros(3, dtype=torch.long),
            feat_pos=torch.zeros(3, dtype=torch.long),
            row_to_node_index=torch.tensor([7, -1, -1, -1]),
        )

    monkeypatch.setattr(phase4, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase4, "_log_phase_metrics", lambda *args, **kwargs: None)
    whole = run_phase4(
        inputs=inputs_for(FakeObserver()),
        config=replace(_config(), actual_max_feature_nodes=3, total_active_feats=3, logit_offset=4, effective_feature_batch_size=3, compute_microbatch_max_rows=3),
    )
    split_observer = FakeObserver()
    split = run_phase4(
        inputs=inputs_for(split_observer),
        config=replace(_config(), actual_max_feature_nodes=3, total_active_feats=3, logit_offset=4, effective_feature_batch_size=3, compute_microbatch_max_rows=2),
    )

    assert torch.equal(split.edge_matrix, whole.edge_matrix)
    assert torch.equal(split.row_to_node_index, whole.row_to_node_index)
    assert split.phase4_refresh_count == whole.phase4_refresh_count
    assert [batch["batch_index"] for batch in split_observer.batches] == [1, 2]
    assert all(batch["attrs"]["executor_physically_split"] for batch in split_observer.batches)
