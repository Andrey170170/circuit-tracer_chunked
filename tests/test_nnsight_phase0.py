from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution.nnsight.phases.phase0 import (
    Phase0CleanupOwner,
    Phase0Config,
    Phase0ExecutionError,
    Phase0Inputs,
    run_phase0,
)


class FakeLogger:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def info(self, message: object, *args: object, **kwargs: object) -> None:
        self.calls.append(f"log:{message}")


class FakeTranscoders:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def configure_trace_logging(self, callback: object, *, telemetry_recorder: object) -> None:
        self.calls.append("transcoder.configure")

    def reset_diagnostic_stats(self) -> None:
        self.calls.append("transcoder.reset")

    def configure_phase0_activation_threshold_compare(self, **kwargs: object) -> None:
        self.calls.append("transcoder.compare")


class FakeContext:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.encoder_vecs = torch.zeros((2, 3))
        self.activation_matrix = torch.sparse_coo_tensor(
            torch.tensor(
                [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ]
            ),
            torch.tensor([1.0, 2.0]),
            (2, 2, 3),
            check_invariants=True,
        ).coalesce()
        self.setup_diagnostic_stats: dict[str, object] = {}
        self.logit_retention = "last"

    def set_diagnostic_mode(self, enabled: bool) -> None:
        self.calls.append(f"ctx.diagnostic:{enabled}")

    def configure_trace_logging(self, callback: object, *, telemetry_recorder: object) -> None:
        self.calls.append("ctx.configure")

    def replace_phase0_activation_state(self, matrix: torch.Tensor) -> None:
        self.calls.append("ctx.replace")

    def apply_prefix_view_state(self, target_position: int) -> dict[str, int]:
        self.calls.append(f"ctx.prefix:{target_position}")
        return {"target_position": target_position, "masked_feature_count": 1}

    def cleanup(self) -> None:
        self.calls.append("ctx.cleanup")


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, calls: list[str], ctx: FakeContext) -> None:
        self.calls = calls
        self.ctx = ctx
        self.transcoders = FakeTranscoders(calls)

    def ensure_tokenized(self, prompt: object) -> torch.Tensor:
        self.calls.append("model.tokenize")
        return torch.tensor([10, 20, 30, 40])

    def setup_attribution(self, input_ids: torch.Tensor, **kwargs: object) -> FakeContext:
        self.calls.append("model.setup")
        self.setup_kwargs = kwargs
        return self.ctx


class FakeObserver:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def phase(self, **payload: object) -> None:
        self.calls.append(f"event:{payload['name']}")


def _config(**overrides: object) -> Phase0Config:
    values: dict[str, object] = {
        "output_position": None,
        "profile": False,
        "phase0_activation_threshold_compare_mode": "baseline",
        "cross_cluster_debug_enabled": False,
        "exact_chunked_provider_enabled": True,
        "exact_chunked_decoder": True,
        "chunked_feature_replay_window": 4,
        "error_vector_prefetch_lookahead": 2,
        "stage_encoder_vecs_on_cpu": False,
        "stage_error_vectors_on_cpu": False,
        "row_subchunk_size": 8,
        "planner_enabled": False,
        "max_phase4_feature_batch_size": 16,
        "phase1_trace_batch_config": SimpleNamespace(
            requested_policy="legacy",
            effective_policy="legacy",
            requested_batch_size_max=None,
            effective_batch_size_max=None,
        ),
        "phase1_trace_batch_metadata": {"trace_batch_cap_reason": None},
        "phase4_refresh_policy_config": SimpleNamespace(
            requested_policy="legacy",
            effective_policy="legacy",
            requested_interval_multiplier=1,
            effective_interval_multiplier=1,
            effective_queue_multiplier=1,
        ),
        "phase4_ranker_config": SimpleNamespace(requested_mode="legacy", effective_mode="legacy"),
        "row_store_cache_control_config": SimpleNamespace(
            requested_mode="none", effective_mode="none"
        ),
        "exact_encoder_residency_config": SimpleNamespace(
            requested_mode="device", effective_mode="device"
        ),
        "exact_trace_internal_dtype_name": "fp32",
        "effective_source_batch_size": 2,
        "effective_feature_batch_size": 2,
        "effective_logit_batch_size": 2,
        "internal_precision_requested": "fp32",
        "resolved_dtype_map": {"activation_dtype": "fp32"},
        "decoder_chunk_cache": None,
        "decoder_cache_fingerprint": None,
        "capture_phase3_gradient_bundle_enabled": False,
        "diagnostic_feature_cap": None,
    }
    values.update(overrides)
    return Phase0Config(**values)  # type: ignore[arg-type]


def _inputs(
    calls: list[str],
    model: FakeModel,
    *,
    override: FakeContext | None = None,
    prefix_view_metadata: dict[str, object] | None = None,
    cross_cluster_debug_summary: dict[str, object] | None = None,
    checkpoints: list[dict[str, object]] | None = None,
) -> Phase0Inputs:
    return Phase0Inputs(
        logger=FakeLogger(calls),
        model=model,
        prompt=[10, 20, 30, 40],
        sparsification=None,
        telemetry_observer=FakeObserver(calls),
        telemetry_recorder=object(),
        phase0_context_override=override,
        prefix_view_metadata=prefix_view_metadata,  # type: ignore[arg-type]
        exact_encoder_residency_metadata={},
        phase4_execution_metadata={},
        cross_cluster_debug_summary=cross_cluster_debug_summary,
        cross_cluster_debug_checkpoints=checkpoints if checkpoints is not None else [],
        cleanup_owner=Phase0CleanupOwner(),
    )


def test_phase0_ordinary_setup_returns_runtime_values_and_preserves_order(monkeypatch) -> None:
    calls: list[str] = []
    ctx = FakeContext(calls)
    model = FakeModel(calls, ctx)
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_memory_boundary",
        lambda *args, **kwargs: calls.append("memory"),
    )
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_phase_metrics",
        lambda *args, **kwargs: calls.append("metrics"),
    )

    result = run_phase0(inputs=_inputs(calls, model), config=_config(output_position=1))

    assert result.ctx is ctx
    assert result.n_input_pos == 4
    assert result.output_position == 1
    assert torch.equal(result.trace_input_ids, result.input_ids)
    assert result.activation_matrix is ctx.activation_matrix
    assert model.setup_kwargs["retain_full_logits"] is True
    assert calls.index("model.tokenize") < calls.index("model.setup")
    assert calls.index("transcoder.compare") < calls.index("model.setup")
    assert calls.index("ctx.configure") < calls.index("metrics")
    assert calls.index("metrics") < calls.index("event:phase0.precompute")
    assert result.phase4_execution_metadata["active_encoder_shape"] == (2, 3)


def test_phase0_override_prefix_mask_and_event_checkpoint_order(monkeypatch) -> None:
    calls: list[str] = []
    ctx = FakeContext(calls)
    model = FakeModel(calls, ctx)
    checkpoints: list[dict[str, object]] = []
    summary: dict[str, object] = {"checkpoints": {}}
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_memory_boundary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_phase_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._build_cross_cluster_runtime_snapshot",
        lambda **kwargs: ({}, {}),
    )

    def record_checkpoint(**kwargs: object) -> None:
        calls.append("checkpoint:phase0_sparse_setup")

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._record_cross_cluster_checkpoint",
        record_checkpoint,
    )
    prefix_metadata = {
        "mode": "full_sequence_target_position",
        "target_position": 3,
    }

    result = run_phase0(
        inputs=_inputs(
            calls,
            model,
            override=ctx,
            prefix_view_metadata=prefix_metadata,
            cross_cluster_debug_summary=summary,
            checkpoints=checkpoints,
        ),
        config=_config(cross_cluster_debug_enabled=True),
    )

    assert "model.setup" not in calls
    assert result.prefix_view_length == 3
    assert torch.equal(result.trace_input_ids, torch.tensor([10, 20, 30]))
    assert result.prefix_view_activation_mask_metadata == {
        "target_position": 3,
        "masked_feature_count": 1,
    }
    assert calls.index("ctx.prefix:3") < calls.index("event:phase0.precompute")
    assert calls.index("event:phase0.precompute") < calls.index("checkpoint:phase0_sparse_setup")


def test_phase0_metrics_failure_exposes_context_and_original_cause(monkeypatch) -> None:
    calls: list[str] = []
    ctx = FakeContext(calls)
    model = FakeModel(calls, ctx)
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_memory_boundary",
        lambda *args, **kwargs: None,
    )

    def fail_metrics(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected metrics failure")

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_phase_metrics",
        fail_metrics,
    )

    try:
        run_phase0(inputs=_inputs(calls, model), config=_config())
    except Phase0ExecutionError as exc:
        assert exc.ctx is ctx
        assert isinstance(exc.cause, RuntimeError)
        assert str(exc.cause) == "injected metrics failure"
    else:
        raise AssertionError("expected Phase0ExecutionError")

    assert calls.count("ctx.cleanup") == 0


def test_phase0_base_exception_exposes_context_and_original_cause(monkeypatch) -> None:
    calls: list[str] = []
    ctx = FakeContext(calls)
    model = FakeModel(calls, ctx)
    original_error = KeyboardInterrupt("injected interrupt")
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_memory_boundary",
        lambda *args, **kwargs: None,
    )

    def fail_metrics(*args: object, **kwargs: object) -> None:
        raise original_error

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase0._log_phase_metrics",
        fail_metrics,
    )

    with pytest.raises(Phase0ExecutionError) as raised:
        run_phase0(inputs=_inputs(calls, model), config=_config())

    assert raised.value.ctx is ctx
    assert raised.value.cause is original_error
    assert calls.count("ctx.cleanup") == 0

