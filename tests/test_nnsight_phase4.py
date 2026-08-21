import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTapeByteEstimate,
    FeatureVjpTapeEntry,
)
from circuit_tracer.attribution.nnsight.phases.phase4 import (
    Phase4Config,
    Phase4Inputs,
    run_phase4,
)
from circuit_tracer.attribution.nnsight.phases.phase4_batches import (
    _DECODER_DELTA_COUNTER_KEYS,
    _DECODER_PREFETCH_CURRENT_KEYS,
    _DECODER_PREFETCH_HIGH_WATERMARK_KEYS,
    _start_cuda_kernel_timer,
)
from circuit_tracer.observability.events import (
    BatchProfile,
    DiagnosticSnapshot,
    MemoryBoundary,
    MemoryDelta,
    MemorySnapshot,
    NumericDelta,
    PhaseMetrics,
    TraceEvent,
)


class FakeObserver:
    def __init__(self) -> None:
        self.phases: list[dict[str, object]] = []
        self.batches: list[dict[str, object]] = []

    def observe(self, observation: object) -> object | None:
        if isinstance(observation, TraceEvent):
            payload = {"name": observation.name, "attrs": dict(observation.attrs)}
            if observation.scope == "batch":
                payload["batch_index"] = observation.batch_index
                self.batches.append(payload)
            elif observation.scope == "phase":
                self.phases.append(payload)
        elif isinstance(observation, (MemoryBoundary, PhaseMetrics, BatchProfile)):
            return None
        elif isinstance(observation, (MemorySnapshot, MemoryDelta, NumericDelta)):
            return {}
        elif isinstance(observation, DiagnosticSnapshot):
            return None
        return None


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
        execution_batch_max_rows=2,
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


def test_cuda_kernel_timer_uses_runtime_cuda_when_encoder_vectors_are_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[str] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            recorded.append("record")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    state = SimpleNamespace(
        config=SimpleNamespace(diagnostic_stop_after_batches=1),
        model=SimpleNamespace(device=torch.device("cuda")),
        encoder_vectors=torch.zeros(1),
    )

    timer = _start_cuda_kernel_timer(state)

    assert timer is not None
    assert recorded == ["record"]

    state.config.diagnostic_stop_after_batches = None

    assert _start_cuda_kernel_timer(state) is None
    assert recorded == ["record"]


def test_phase4_returns_boundary_state_without_replacing_owned_buffers() -> None:
    observer = FakeObserver()
    inputs = _inputs(observer)

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
    assert (
        result.phase4_execution_metadata["phase4_feature_vjp_actual_decoder_page_load_count_total"]
        == 0
    )
    assert (
        result.phase4_execution_metadata["phase4_feature_vjp_actual_decoder_load_bytes_total"] == 0
    )
    assert result.cross_cluster_debug_summary is inputs.cross_cluster_debug_summary
    assert result.cross_cluster_debug_checkpoints is inputs.cross_cluster_debug_checkpoints
    assert result.cross_cluster_debug_batches is inputs.cross_cluster_debug_batches
    assert result.anomaly_debug_result is inputs.anomaly_debug_result
    assert result.anomaly_debug_result["status"] == "captured_refresh_debug"
    assert result.phase4_feature_batch_size == 2
    assert result.phase4_semantic_batch_max_rows == 2
    assert result.phase4_execution_batch_max_rows == 2
    assert result.phase4_execution_batch_count == 0
    assert result.phase4_refresh_count == 0
    assert result.phase4_scheduler_reference_batch_count == 0
    assert len(observer.phases) == 1


def test_phase4_rejects_preexisting_prefetch_owner_even_at_depth_zero() -> None:
    inputs = _inputs(FakeObserver())
    inputs.ctx._decoder_page_prefetch = object()

    with pytest.raises(RuntimeError, match="open decoder prefetch lifecycle"):
        run_phase4(inputs=inputs, config=_config())


def test_phase4_closes_prefetch_before_terminal_lifecycle_telemetry() -> None:
    observer = FakeObserver()
    inputs = _inputs(observer)
    values = {
        key: 0.0 if key.endswith("_seconds") else 0
        for key in (
            *_DECODER_DELTA_COUNTER_KEYS,
            *_DECODER_PREFETCH_HIGH_WATERMARK_KEYS,
            *_DECODER_PREFETCH_CURRENT_KEYS,
        )
    }
    lifecycle: list[str] = []

    class _Provider:
        def get_diagnostic_snapshot(self):
            return dict(values)

    provider = _Provider()

    class _Context:
        _decoder_page_prefetch = None
        decoder_page_prefetch_depth = 0

        def open_decoder_page_prefetch(self, *, depth: int):
            assert depth == 1
            lifecycle.append("open")
            owner = object()
            self._decoder_page_prefetch = owner
            values["decoder_prefetch_owner_open_count"] = 1
            values["decoder_prefetch_owner_count"] = 1
            values["decoder_prefetch_owner_high_watermark"] = 1
            return owner

        def close_decoder_page_prefetch(self, owner) -> None:
            assert owner is self._decoder_page_prefetch
            lifecycle.append("close")
            values["decoder_prefetch_owner_close_count"] = 1
            values["decoder_prefetch_owner_count"] = 0
            self._decoder_page_prefetch = None

    inputs = replace(
        inputs,
        ctx=_Context(),
        model=SimpleNamespace(device=torch.device("cpu"), transcoders=provider),
    )
    run_phase4(
        inputs=inputs,
        config=replace(_config(), decoder_page_prefetch_depth=1),
    )

    phase = next(item for item in observer.phases if item["name"] == "phase4.feature_attribution")
    attrs = phase["attrs"]
    assert lifecycle == ["open", "close"]
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_open_count_total"] == 1
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_close_count_total"] == 1
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_count_final"] == 0
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_high_watermark"] == 1

    depth_zero_observer = FakeObserver()
    run_phase4(
        inputs=replace(inputs, telemetry_observer=depth_zero_observer),
        config=_config(),
    )
    depth_zero_phase = next(
        item for item in depth_zero_observer.phases if item["name"] == "phase4.feature_attribution"
    )
    depth_zero_attrs = depth_zero_phase["attrs"]
    assert (
        depth_zero_attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_open_count_total"] == 0
    )
    assert (
        depth_zero_attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_close_count_total"] == 0
    )
    assert depth_zero_attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_high_watermark"] == 0


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
    config = replace(_config(), actual_max_feature_nodes=1, total_active_feats=1, logit_offset=2)

    result = run_phase4(inputs=inputs, config=config)

    assert result.visited.tolist() == [True]
    assert result.st == 2
    assert result.edge_matrix is inputs.edge_matrix
    assert torch.equal(result.edge_matrix[1], torch.tensor([3.0, -4.0]))
    assert result.row_to_node_index is inputs.row_to_node_index
    assert result.row_to_node_index.tolist() == [7, 0]
    assert result.rows_cpu_staging is inputs.rows_cpu_staging
    assert result.phase4_execution_metadata is inputs.phase4_execution_metadata
    assert result.phase4_execution_metadata["phase4_semantic_batch_max_rows"] == 2
    assert result.phase4_frontier_buffer_metadata is inputs.phase4_frontier_buffer_metadata
    assert result.phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] == 1
    assert result.cross_cluster_debug_summary is inputs.cross_cluster_debug_summary
    assert result.cross_cluster_debug_checkpoints is inputs.cross_cluster_debug_checkpoints
    assert result.cross_cluster_debug_batches is inputs.cross_cluster_debug_batches
    assert len(result.cross_cluster_debug_checkpoints) == 1
    assert len(result.cross_cluster_debug_batches) == 1
    assert result.phase4_scheduler_reference_batch_count == 1
    assert result.phase4_execution_batch_count == 1
    assert result.phase4_refresh_count == 0
    assert len(observer.batches) == 1
    assert observer.batches[0]["attrs"]["visited_features"] == 1
    assert len(observer.phases) == 1
    assert observer.phases[0]["attrs"]["phase4_execution_batch_count"] == 1


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
    assert torch.equal(feature_row_store.append_calls[0]["feature_rows"], torch.tensor([[3.0]]))
    assert torch.equal(nonfeature_row_store.append_calls[0]["feature_rows"], torch.tensor([[-4.0]]))


def test_phase4_propagates_failure_after_dense_write_without_finalizing(monkeypatch) -> None:
    observer = FakeObserver()
    inputs = _nonzero_inputs(observer)
    config = replace(_config(), actual_max_feature_nodes=1, total_active_feats=1, logit_offset=2)
    failure = RuntimeError("injected post-write phase4 failure")

    original_observe = observer.observe

    def fail_after_write(observation: object) -> object | None:
        if isinstance(observation, TraceEvent) and observation.name == "phase4.feature_batch":
            raise failure
        return original_observe(observation)

    monkeypatch.setattr(observer, "observe", fail_after_write)

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
            ctx=SimpleNamespace(
                materialize_encoder_vectors=materialize, compute_batch=compute_batch
            ),
            edge_matrix=torch.zeros((4, 4)),
            feat_ids=torch.tensor([20, 21, 22]),
            feat_layers=torch.zeros(3, dtype=torch.long),
            feat_pos=torch.zeros(3, dtype=torch.long),
            row_to_node_index=torch.tensor([7, -1, -1, -1]),
        )

    whole = run_phase4(
        inputs=inputs_for(FakeObserver()),
        config=replace(
            _config(),
            actual_max_feature_nodes=3,
            total_active_feats=3,
            logit_offset=4,
            effective_feature_batch_size=3,
            execution_batch_max_rows=3,
        ),
    )
    split_observer = FakeObserver()
    split = run_phase4(
        inputs=inputs_for(split_observer),
        config=replace(
            _config(),
            actual_max_feature_nodes=3,
            total_active_feats=3,
            logit_offset=4,
            effective_feature_batch_size=3,
            execution_batch_max_rows=2,
        ),
    )

    assert torch.equal(split.edge_matrix, whole.edge_matrix)
    assert torch.equal(split.row_to_node_index, whole.row_to_node_index)
    assert split.phase4_refresh_count == whole.phase4_refresh_count
    assert [batch["batch_index"] for batch in split_observer.batches] == [1, 2]
    assert all(batch["attrs"]["phase4_execution_batch_split"] for batch in split_observer.batches)


def test_phase4_coalesces_semantic_batches_without_changing_schedule_or_results() -> None:
    def run(execution_batch_max_rows: int):
        observer = FakeObserver()
        materialized: list[list[int]] = []
        compute_calls = 0

        def materialize(indices: torch.Tensor) -> torch.Tensor:
            materialized.append(indices.tolist())
            return indices.to(dtype=torch.float32).reshape(-1, 1)

        def compute_batch(**kwargs: object) -> torch.Tensor:
            nonlocal compute_calls
            compute_calls += 1
            values = kwargs["inject_values"]
            assert isinstance(values, torch.Tensor)
            return torch.cat(tuple(values + offset for offset in range(5)), dim=1)

        inputs = replace(
            _nonzero_inputs(observer),
            ctx=SimpleNamespace(
                materialize_encoder_vectors=materialize,
                compute_batch=compute_batch,
            ),
            edge_matrix=torch.zeros((5, 5)),
            feat_ids=torch.tensor([20, 21, 22, 23]),
            feat_layers=torch.zeros(4, dtype=torch.long),
            feat_pos=torch.zeros(4, dtype=torch.long),
            row_to_node_index=torch.tensor([7, -1, -1, -1, -1]),
        )
        result = run_phase4(
            inputs=inputs,
            config=replace(
                _config(),
                actual_max_feature_nodes=4,
                total_active_feats=4,
                logit_offset=5,
                effective_feature_batch_size=2,
                execution_batch_max_rows=execution_batch_max_rows,
                update_interval=2,
            ),
        )
        return result, observer, materialized, compute_calls

    semantic, semantic_observer, semantic_order, semantic_calls = run(2)
    coalesced, coalesced_observer, coalesced_order, coalesced_calls = run(4)

    assert semantic_calls == 2
    assert coalesced_calls == 1
    assert [row for batch in semantic_order for row in batch] == [
        row for batch in coalesced_order for row in batch
    ]
    assert torch.equal(coalesced.edge_matrix, semantic.edge_matrix)
    assert torch.equal(coalesced.row_to_node_index, semantic.row_to_node_index)
    assert torch.equal(coalesced.visited, semantic.visited)
    assert coalesced.phase4_refresh_count == semantic.phase4_refresh_count
    assert coalesced.phase4_scheduler_reference_batch_count == 2
    assert coalesced.phase4_execution_batch_count == 1
    assert semantic.phase4_execution_batch_count == 2
    assert [
        (item["checkpoint_name"], item["phase"])
        for item in coalesced.cross_cluster_debug_checkpoints
    ] == [
        (item["checkpoint_name"], item["phase"])
        for item in semantic.cross_cluster_debug_checkpoints
    ]

    semantic_attrs = [batch["attrs"] for batch in semantic_observer.batches]
    coalesced_attrs = coalesced_observer.batches[0]["attrs"]
    assert [attrs["phase4_semantic_batch_rows"] for attrs in semantic_attrs] == [(2,), (2,)]
    assert coalesced_attrs["phase4_semantic_batch_rows"] == (2, 2)
    assert coalesced_attrs["phase4_semantic_batch_index_start"] == 0
    assert coalesced_attrs["phase4_semantic_batch_index_end"] == 1
    assert coalesced_attrs["phase4_execution_batch_rows"] == 4
    assert coalesced_attrs["phase4_execution_batch_coalesced"] is True
    assert "executor_microbatch_rows" not in coalesced_attrs


def test_phase4_coalescing_never_crosses_refresh_frontiers() -> None:
    observer = FakeObserver()
    materialized: list[list[int]] = []

    def materialize(indices: torch.Tensor) -> torch.Tensor:
        materialized.append(indices.tolist())
        return indices.to(dtype=torch.float32).reshape(-1, 1)

    def compute_batch(**kwargs: object) -> torch.Tensor:
        values = kwargs["inject_values"]
        assert isinstance(values, torch.Tensor)
        return torch.cat(tuple(values + offset for offset in range(7)), dim=1)

    inputs = replace(
        _nonzero_inputs(observer),
        ctx=SimpleNamespace(
            materialize_encoder_vectors=materialize,
            compute_batch=compute_batch,
        ),
        edge_matrix=torch.zeros((6, 7)),
        feat_ids=torch.arange(6),
        feat_layers=torch.zeros(6, dtype=torch.long),
        feat_pos=torch.zeros(6, dtype=torch.long),
        row_to_node_index=torch.tensor([6, -1, -1, -1, -1, -1]),
        cross_cluster_debug_summary=None,
        cross_cluster_debug_checkpoints=None,
        cross_cluster_debug_batches=None,
        anomaly_debug_result=None,
    )

    result = run_phase4(
        inputs=inputs,
        config=replace(
            _config(),
            actual_max_feature_nodes=5,
            total_active_feats=6,
            logit_offset=7,
            effective_feature_batch_size=1,
            execution_batch_max_rows=4,
            update_interval=2,
            phase4_ranker_config=_policy(requested_mode="argsort", effective_mode="argsort"),
        ),
    )

    assert result.phase4_refresh_count == 3
    assert result.phase4_scheduler_reference_batch_count == 5
    assert result.phase4_execution_batch_count == 3
    assert [len(batch) for batch in materialized] == [2, 2, 1]
    attrs = [
        batch["attrs"] for batch in observer.batches if batch["name"] == "phase4.feature_batch"
    ]
    assert [batch["scheduler_refresh_index"] for batch in attrs] == [0, 1, 2]
    assert [batch["phase4_semantic_batch_rows"] for batch in attrs] == [
        (1, 1),
        (1, 1),
        (1,),
    ]
    assert [batch["phase4_execution_batch_rows"] for batch in attrs] == [2, 2, 1]


class _FakePrefetchTranscoders:
    def __init__(self) -> None:
        self.snapshot = {
            "decoder_chunk_request_count": 0,
            "decoder_chunk_request_bytes": 0,
            "decoder_load_count": 0,
            "decoder_load_bytes": 0,
            "decoder_cache_hit_count": 0,
            "decoder_prefetch_request_count": 0,
            "decoder_prefetch_load_count": 0,
            "decoder_prefetch_load_bytes": 0,
            "decoder_prefetch_cache_hit_count": 0,
            "decoder_prefetch_consume_hit_count": 0,
            "decoder_prefetch_host_wait_count": 0,
            "decoder_prefetch_host_wait_seconds": 0.0,
            "decoder_prefetch_in_flight_count": 0,
            "decoder_prefetch_in_flight_high_watermark": 0,
            "decoder_prefetch_in_flight_bytes": 0,
            "decoder_prefetch_in_flight_bytes_high_watermark": 0,
            "decoder_prefetch_consumer_active_count": 0,
            "decoder_prefetch_consumer_active_bytes": 0,
            "decoder_prefetch_consumer_retained_count": 0,
            "decoder_prefetch_consumer_retained_bytes": 0,
            "decoder_prefetch_consumer_retained_bytes_high_watermark": 0,
            "decoder_prefetch_consumer_retirement_count": 0,
            "decoder_prefetch_consumer_backpressure_count": 0,
            "decoder_prefetch_consumer_backpressure_seconds": 0.0,
            "decoder_prefetch_pipeline_owned_final_page_count": 0,
            "decoder_prefetch_pipeline_owned_final_page_high_watermark": 0,
            "decoder_prefetch_pipeline_owned_final_page_bytes": 0,
            "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": 0,
            "decoder_prefetch_owner_count": 0,
            "decoder_prefetch_owner_high_watermark": 0,
            "decoder_prefetch_owner_open_count": 0,
            "decoder_prefetch_owner_close_count": 0,
        }

    def get_diagnostic_snapshot(self) -> dict[str, int | float]:
        return dict(self.snapshot)

    def record_replay(self) -> None:
        self.snapshot["decoder_prefetch_request_count"] += 2
        self.snapshot["decoder_prefetch_load_count"] += 1
        self.snapshot["decoder_prefetch_load_bytes"] += 16
        self.snapshot["decoder_prefetch_cache_hit_count"] += 1
        self.snapshot["decoder_prefetch_consume_hit_count"] += 2
        self.snapshot["decoder_prefetch_host_wait_count"] += 1
        self.snapshot["decoder_prefetch_host_wait_seconds"] += 0.25
        self.snapshot["decoder_prefetch_in_flight_high_watermark"] += 1
        self.snapshot["decoder_prefetch_in_flight_bytes_high_watermark"] += 16
        self.snapshot["decoder_prefetch_consumer_retirement_count"] += 1
        self.snapshot["decoder_prefetch_consumer_backpressure_count"] += 1
        self.snapshot["decoder_prefetch_consumer_backpressure_seconds"] += 0.5
        self.snapshot["decoder_prefetch_owner_open_count"] += 1
        self.snapshot["decoder_prefetch_owner_close_count"] += 1
        self.snapshot["decoder_prefetch_consumer_retained_bytes_high_watermark"] += 8
        self.snapshot["decoder_prefetch_pipeline_owned_final_page_high_watermark"] += 1
        self.snapshot["decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark"] += 16
        self.snapshot["decoder_prefetch_owner_high_watermark"] += 1


class _FakeTapeContext:
    def __init__(
        self,
        *,
        row_width: int,
        entry_bytes: int = 10,
        transcoders: _FakePrefetchTranscoders | None = None,
    ) -> None:
        self.row_width = row_width
        self.entry_bytes = entry_bytes
        self.transcoders = transcoders
        self.replay_windows: list[list[int]] = []
        self.next_call_index = 0
        self.previous_replay_storage_refs: list[weakref.ReferenceType[object]] = []
        self.previous_replay_storage_released: list[bool] = []

    def materialize_encoder_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.to(dtype=torch.float32).reshape(-1, 1)

    def _rows(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(tuple(values + offset for offset in range(self.row_width)), dim=1)

    def compute_batch(self, **kwargs: object) -> torch.Tensor:
        values = kwargs["inject_values"]
        assert isinstance(values, torch.Tensor)
        return self._rows(values)

    def estimate_feature_vjp_tape_entry_nbytes(
        self, *, layers: torch.Tensor, batch_size: int
    ) -> FeatureVjpTapeByteEstimate:
        del layers, batch_size
        return FeatureVjpTapeByteEstimate(
            host_nbytes=self.entry_bytes // 3,
            device_nbytes=self.entry_bytes // 3,
            row_nbytes=self.entry_bytes - (2 * (self.entry_bytes // 3)),
            total_nbytes=self.entry_bytes,
        )

    def capture_feature_vjp_batch(self, **kwargs: object) -> FeatureVjpTapeEntry:
        if self.previous_replay_storage_refs:
            gc.collect()
            self.previous_replay_storage_released.append(
                all(ref() is None for ref in self.previous_replay_storage_refs)
            )
            self.previous_replay_storage_refs.clear()
        values = kwargs["inject_values"]
        assert isinstance(values, torch.Tensor)
        self.next_call_index += 1
        rows = self._rows(values)
        return FeatureVjpTapeEntry(
            batch_call_index=self.next_call_index,
            gradients=(),
            row_buffer=rows.T.clone(),
            batch_size=len(values),
            host_nbytes=self.entry_bytes // 3,
            device_nbytes=self.entry_bytes // 3,
            row_nbytes=self.entry_bytes - (2 * (self.entry_bytes // 3)),
            total_nbytes=self.entry_bytes,
            pinned_host_nbytes=0,
            pageable_host_nbytes=self.entry_bytes // 3,
        )

    def replay_feature_vjp_tape(
        self,
        entries: tuple[FeatureVjpTapeEntry, ...],
        *,
        phase_label: str,
    ) -> list[torch.Tensor]:
        del phase_label
        if self.transcoders is not None:
            self.transcoders.record_replay()
        self.replay_windows.append([entry.batch_call_index for entry in entries])
        storages = [entry.row_buffer.untyped_storage() for entry in entries]
        self.previous_replay_storage_refs = [weakref.ref(storage) for storage in storages]
        return [entry.row_buffer.T[: entry.batch_size] for entry in entries]


def _run_tape_phase4(
    *,
    window: int,
    max_bytes: int,
    update_interval: int = 2,
    partial_frontier: bool = False,
    gpu_reduction: bool = False,
    prefetch_diagnostics: bool = False,
) -> tuple[object, FakeObserver, _FakeTapeContext]:
    observer = FakeObserver()
    total_features = 6 if partial_frontier else 4
    actual_features = 5 if partial_frontier else 4
    row_width = total_features + 1
    transcoders = _FakePrefetchTranscoders() if prefetch_diagnostics else None
    ctx = _FakeTapeContext(row_width=row_width, transcoders=transcoders)
    inputs = replace(
        _nonzero_inputs(observer),
        ctx=ctx,
        model=SimpleNamespace(
            device=torch.device("cpu"),
            transcoders=(
                transcoders if transcoders is not None else SimpleNamespace(decoder_chunk_size=2)
            ),
        ),
        edge_matrix=torch.zeros((actual_features + 1, row_width)),
        feat_ids=torch.arange(20, 20 + total_features),
        feat_layers=torch.zeros(total_features, dtype=torch.long),
        feat_pos=torch.zeros(total_features, dtype=torch.long),
        row_to_node_index=torch.full((actual_features + 1,), -1, dtype=torch.long),
        cross_cluster_debug_summary=None if partial_frontier else {},
        cross_cluster_debug_checkpoints=None if partial_frontier else [],
        cross_cluster_debug_batches=None if partial_frontier else [],
        anomaly_debug_result=None if partial_frontier else {"records": []},
        feature_row_store=FakeRowStore() if gpu_reduction else None,
        nonfeature_row_store=FakeRowStore() if gpu_reduction else None,
    )
    result = run_phase4(
        inputs=inputs,
        config=replace(
            _config(),
            actual_max_feature_nodes=actual_features,
            total_active_feats=total_features,
            logit_offset=row_width,
            effective_feature_batch_size=1,
            execution_batch_max_rows=1,
            update_interval=update_interval,
            exact_chunked_decoder=True,
            feature_vjp_tape_batch_window=window,
            feature_vjp_tape_max_bytes=max_bytes,
            feature_vjp_tape_enabled=window > 1,
            feature_vjp_tape_fallback_reason=(
                None if window > 1 else "window_one_streaming_fallback"
            ),
            use_compact_feature_row_store=gpu_reduction,
            phase4_row_reduction_config=_policy(
                effective_mode="gpu_v1" if gpu_reduction else "off"
            ),
            phase4_ranker_config=_policy(
                requested_mode="argsort",
                effective_mode="argsort" if partial_frontier else "off",
            ),
        ),
    )
    return result, observer, ctx


def test_phase4_feature_vjp_tape_window_two_matches_window_one_and_reuses_windows() -> None:
    baseline, _, _ = _run_tape_phase4(window=1, max_bytes=0)
    taped, observer, ctx = _run_tape_phase4(window=2, max_bytes=20)

    assert torch.equal(taped.edge_matrix, baseline.edge_matrix)
    assert torch.equal(taped.row_to_node_index, baseline.row_to_node_index)
    assert ctx.replay_windows == [[1, 2], [3, 4]]
    attrs = next(
        event["attrs"] for event in observer.phases if event["name"] == "phase4.feature_attribution"
    )
    assert attrs["phase4_feature_vjp_tape_window_count"] == 2
    assert attrs["phase4_feature_vjp_tape_batch_count"] == 4
    assert attrs["phase4_feature_vjp_tape_high_watermark_bytes"] == 20
    assert attrs["phase4_feature_vjp_planned_decoder_traversal_numerator"] == 2
    assert attrs["phase4_feature_vjp_planned_decoder_traversal_denominator"] == 4


def test_phase4_tape_preserves_prefetch_deltas_and_high_watermarks() -> None:
    _, observer, _ = _run_tape_phase4(window=2, max_bytes=20, prefetch_diagnostics=True)

    attrs = next(
        event["attrs"] for event in observer.phases if event["name"] == "phase4.feature_attribution"
    )
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_request_count_total"] == 4
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_load_count_total"] == 2
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_load_bytes_total"] == 32
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_cache_hit_count_total"] == 2
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_consume_hit_count_total"] == 4
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_host_wait_count_total"] == 2
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_host_wait_seconds_total"] == 0.5
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_in_flight_high_watermark"] == 2
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_in_flight_bytes_high_watermark"] == 32
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_in_flight_count_final"] == 0
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_in_flight_bytes_final"] == 0
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_consumer_retirement_count_total"] == 2
    assert (
        attrs["phase4_feature_vjp_actual_decoder_prefetch_consumer_backpressure_seconds_total"]
        == 1.0
    )
    assert (
        attrs["phase4_feature_vjp_actual_decoder_prefetch_consumer_retained_bytes_high_watermark"]
        == 16
    )
    assert (
        attrs["phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_high_watermark"]
        == 2
    )
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_high_watermark"] == 2
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_consumer_active_count_final"] == 0
    assert (
        attrs["phase4_feature_vjp_actual_decoder_prefetch_pipeline_owned_final_page_bytes_final"]
        == 0
    )
    assert attrs["phase4_feature_vjp_actual_decoder_prefetch_owner_count_final"] == 0


def test_phase4_feature_vjp_tape_cap_flushes_and_never_crosses_frontier() -> None:
    _, cap_observer, cap_ctx = _run_tape_phase4(window=2, max_bytes=9)
    _, _, frontier_ctx = _run_tape_phase4(
        window=2,
        max_bytes=20,
        partial_frontier=True,
    )

    assert cap_ctx.replay_windows == []
    cap_attrs = next(
        event["attrs"]
        for event in cap_observer.phases
        if event["name"] == "phase4.feature_attribution"
    )
    assert cap_attrs["phase4_feature_vjp_tape_oversize_fallback_batches"] == 4
    assert cap_attrs["phase4_feature_vjp_tape_batch_count"] == 0
    assert frontier_ctx.replay_windows == [[1, 2], [3, 4], [5]]


def test_phase4_tape_releases_gpu_reduction_row_storage_before_next_window() -> None:
    _, _, ctx = _run_tape_phase4(
        window=2,
        max_bytes=20,
        gpu_reduction=True,
    )

    assert ctx.replay_windows == [[1, 2], [3, 4]]
    assert ctx.previous_replay_storage_released == [True]
