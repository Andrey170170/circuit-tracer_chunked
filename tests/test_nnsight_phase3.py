from types import SimpleNamespace

import pytest
import torch

import circuit_tracer.attribution.nnsight.phases.phase3 as phase3
from circuit_tracer.attribution.nnsight.phases.phase3 import (
    Phase3Config,
    Phase3Inputs,
    run_phase3,
)


class FakeTargets:
    def __init__(self, count: int = 1) -> None:
        self.logit_vectors = torch.arange(1, count + 1, dtype=torch.float32).reshape(-1, 1)
        self.logit_probabilities = torch.ones(count)
        self.logit_targets = [SimpleNamespace(vocab_idx=7 + index) for index in range(count)]

    def __len__(self) -> int:
        return int(self.logit_vectors.shape[0])


class FakeContext:
    def __init__(self, rows: torch.Tensor | Exception) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def compute_batch(self, **kwargs: object) -> torch.Tensor:
        self.calls.append("compute_batch")
        if isinstance(self.rows, Exception):
            raise self.rows
        layers = kwargs.get("layers")
        count = int(layers.shape[0]) if isinstance(layers, torch.Tensor) else 1
        return self.rows[:count]

    def reset_decoder_cache(self) -> None:
        self.calls.append("reset_decoder_cache")


class FakeObserver:
    def __init__(self) -> None:
        self.batches: list[dict[str, object]] = []
        self.phases: list[dict[str, object]] = []

    def batch(self, **payload: object) -> None:
        self.batches.append(payload)

    def phase(self, **payload: object) -> None:
        self.phases.append(payload)


def _config(**overrides: object) -> Phase3Config:
    values: dict[str, object] = {
        "effective_logit_batch_size": 1,
        "compute_microbatch_max_rows": 1,
        "effective_feature_batch_size": 1,
        "output_position": 0,
        "n_layers": 1,
        "n_pos": 1,
        "n_logits": 1,
        "logit_offset": 3,
        "total_active_feats": 1,
        "base_max_feature_nodes": 1,
        "actual_max_feature_nodes": 1,
        "exact_trace_internal_dtype_resolved": torch.float32,
        "phase3_gradient_replay_mode_resolved": "disabled",
        "phase3_row_replay_mode_resolved": "disabled",
        "capture_phase3_gradient_bundle_enabled": False,
        "capture_phase3_row_bundle_enabled": False,
        "capture_phase3_seed_bundle_enabled": False,
        "capture_feature_semantic_descriptors_enabled": False,
        "phase3_frontier_buffer_relative_epsilon": None,
        "phase3_frontier_buffer_max_extra": 0,
        "update_interval": 1,
        "planner_compute_dtype": torch.float32,
        "influence_compute_dtype": torch.float32,
        "shadow_debug_compute_dtype": torch.float32,
        "phase4_refresh_policy_config": SimpleNamespace(effective_queue_multiplier=1),
        "exact_chunked_decoder": False,
        "use_compact_feature_row_store": False,
        "semantic_descriptor_top_k": 1,
        "semantic_descriptor_dim": 1,
        "profile": False,
        "profile_log_interval": 1,
    }
    values.update(overrides)
    return Phase3Config(**values)  # type: ignore[arg-type]


def _inputs(ctx: FakeContext, observer: FakeObserver) -> Phase3Inputs:
    activation_matrix = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0]]),
        torch.tensor([1.0]),
        (1, 1, 1),
    ).coalesce()
    return Phase3Inputs(
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        model=SimpleNamespace(device=torch.device("cpu"), transcoders=SimpleNamespace()),
        ctx=ctx,
        targets=FakeTargets(),  # type: ignore[arg-type]
        activation_matrix=activation_matrix,
        feat_layers=torch.tensor([0]),
        feat_pos=torch.tensor([0]),
        feat_ids=torch.tensor([0]),
        feature_row_store=None,
        nonfeature_row_store=None,
        edge_matrix=torch.zeros((1, 3)),
        row_to_node_index=torch.full((1,), -1, dtype=torch.long),
        telemetry_observer=observer,
        cross_cluster_debug_summary=None,
        cross_cluster_debug_checkpoints=None,
        cross_cluster_debug_batches=None,
        anomaly_debug_result={"existing": True},
        loaded_phase3_row_donor_bundle=None,
        phase3_frontier_buffer_metadata={"enabled": False},
        phase3_gradient_bundle_payload={"previous": "gradient"},
        phase3_row_bundle_payload={"previous": "row"},
        phase3_seed_bundle_payload={"previous": "seed"},
        feature_semantic_descriptors_payload={"previous": "descriptors"},
    )


def test_phase3_returns_updated_state_and_captured_payloads(monkeypatch) -> None:
    ctx = FakeContext(torch.tensor([[2.0, 3.0, 4.0]]))
    observer = FakeObserver()
    inputs = _inputs(ctx, observer)
    gradient_payload = {"capture": "gradient"}
    row_payload = {"capture": "row"}
    expected_staging_buffer = torch.empty((1, 3))
    monkeypatch.setattr(phase3, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase3, "_log_phase_metrics", lambda *args: None)
    monkeypatch.setattr(
        phase3,
        "_copy_rows_to_cpu_staging",
        lambda rows, staging_buffer: (
            rows,
            staging_buffer if staging_buffer is not None else expected_staging_buffer,
        ),
    )
    monkeypatch.setattr(phase3, "_build_phase3_gradient_bundle_payload", lambda **kwargs: gradient_payload)
    monkeypatch.setattr(phase3, "_build_phase3_row_bundle_payload", lambda **kwargs: row_payload)

    result = run_phase3(
        inputs=inputs,
        config=_config(
            capture_phase3_gradient_bundle_enabled=True,
            capture_phase3_row_bundle_enabled=True,
        ),
    )

    assert result.stored_row_count == 1
    assert torch.equal(result.row_to_node_index, torch.tensor([3]))
    assert result.rows_cpu_staging is expected_staging_buffer
    assert result.actual_max_feature_nodes == 1
    assert result.phase3_gradient_bundle_payload is gradient_payload
    assert result.phase3_row_bundle_payload is row_payload
    assert result.phase3_seed_bundle_payload == {"previous": "seed"}
    assert result.feature_semantic_descriptors_payload == {"previous": "descriptors"}
    assert result.anomaly_debug_result is inputs.anomaly_debug_result
    assert result.anomaly_debug_result["phase3_logit_row_batches"][0]["batch_row_count"] == 1
    assert len(observer.batches) == 1
    assert len(observer.phases) == 1
    assert ctx.calls == ["compute_batch", "reset_decoder_cache"]


def test_phase3_propagates_compute_batch_exception_without_finalizing(monkeypatch) -> None:
    failure = RuntimeError("injected phase3 failure")
    ctx = FakeContext(failure)
    observer = FakeObserver()
    monkeypatch.setattr(phase3, "_log_memory_boundary", lambda *args: None)

    with pytest.raises(RuntimeError, match="injected phase3 failure"):
        run_phase3(inputs=_inputs(ctx, observer), config=_config())

    assert ctx.calls == ["compute_batch"]
    assert observer.batches == []
    assert observer.phases == []


def test_phase3_physical_batches_use_global_event_indices(monkeypatch) -> None:
    class BatchAwareContext(FakeContext):
        def compute_batch(self, **kwargs: object) -> torch.Tensor:
            self.calls.append("compute_batch")
            inject_values = kwargs["inject_values"]
            assert isinstance(inject_values, torch.Tensor)
            return torch.cat((inject_values, inject_values + 1, inject_values + 2), dim=1)

    ctx = BatchAwareContext(torch.empty(0))
    observer = FakeObserver()
    cross_cluster_debug_batches: list[dict[str, object]] = []
    inputs = _inputs(ctx, observer)
    inputs = Phase3Inputs(
        **{
            **inputs.__dict__,
            "targets": FakeTargets(3),
            "edge_matrix": torch.zeros((3, 3)),
            "row_to_node_index": torch.full((3,), -1, dtype=torch.long),
            "cross_cluster_debug_batches": cross_cluster_debug_batches,
        }
    )
    monkeypatch.setattr(phase3, "_log_memory_boundary", lambda *args: None)
    monkeypatch.setattr(phase3, "_log_phase_metrics", lambda *args: None)
    monkeypatch.setattr(
        phase3,
        "_copy_rows_to_cpu_staging",
        lambda rows, staging_buffer: (rows, staging_buffer),
    )

    run_phase3(
        inputs=inputs,
        config=_config(
            effective_logit_batch_size=3,
            compute_microbatch_max_rows=2,
            n_logits=3,
        ),
    )

    assert [batch["batch_index"] for batch in observer.batches] == [1, 2]
    assert [batch["attrs"]["logical_batch_index"] for batch in observer.batches] == [1, 1]
    assert [batch["attrs"]["physical_batch_index"] for batch in observer.batches] == [1, 2]
    assert all(batch["attrs"]["total_physical_batches"] == 2 for batch in observer.batches)
    assert [event["event_index"] for event in cross_cluster_debug_batches] == [1, 2]
    assert [event["logical_batch_index"] for event in cross_cluster_debug_batches] == [1, 1]
