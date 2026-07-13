from types import SimpleNamespace

import torch

from circuit_tracer.attribution.nnsight.phases.phase1 import _run_phase1_forward_pass
from circuit_tracer.observability.events import MemoryBoundary, PhaseMetrics, TraceEvent


def test_phase1_forward_pass_call_and_event_order_and_payload(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeLogger:
        def info(self, message: object, *args: object, **kwargs: object) -> None:
            calls.append(("log", message))

    class FakeContext:
        def run_forward_pass(
            self,
            model: object,
            trace_input_ids: torch.Tensor,
            *,
            trace_batch_size: int,
        ) -> None:
            calls.append(
                (
                    "forward",
                    (model, trace_input_ids.clone(), trace_batch_size),
                )
            )

    class FakeObserver:
        def observe(self, observation: object) -> None:
            calls.append(("observation", observation))

    timestamps = iter((10.0, 10.5))
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.phases.phase1.time.perf_counter",
        lambda: next(timestamps),
    )

    model = SimpleNamespace(device=torch.device("cpu"))
    trace_input_ids = torch.tensor([[1, 2, 3]])
    metadata = {
        "trace_batch_cap_reason": "test-cap",
        "trace_batch_size_effective": 4,
    }
    config = SimpleNamespace(
        requested_policy="cap_effective_batches",
        effective_policy="cap_effective_batches",
        requested_batch_size_max=4,
        effective_batch_size_max=4,
        effective_behavior="capped",
    )

    _run_phase1_forward_pass(
        logger=FakeLogger(),
        model=model,
        ctx=FakeContext(),
        trace_input_ids=trace_input_ids,
        trace_batch_size=4,
        trace_batch_config=config,
        trace_batch_metadata=metadata,
        effective_source_batch_size=2,
        effective_feature_batch_size=4,
        effective_logit_batch_size=3,
        telemetry_observer=FakeObserver(),  # type: ignore[arg-type]
    )

    assert [name for name, _ in calls] == [
        "log",
        "log",
        "observation",
        "forward",
        "observation",
        "observation",
    ]
    assert calls[0][1] == "Phase 1: Running forward pass"
    assert calls[1][1] == (
        "Phase 1 trace-batch policy | requested_policy=cap_effective_batches | "
        "effective_policy=cap_effective_batches | requested_size_max=4 | "
        "effective_size_max=4 | effective_behavior=capped | source_batch_size=2 | "
        "feature_batch_size=4 | logit_batch_size=3 | cap_reason=test-cap | "
        "trace_batch_size=4"
    )
    assert calls[2][1] == MemoryBoundary("Phase 1 start", torch.device("cpu"))
    forward_model, forward_ids, forward_batch_size = calls[3][1]
    assert forward_model is model
    assert torch.equal(forward_ids, trace_input_ids)
    assert forward_batch_size == 4
    assert calls[4][1] == PhaseMetrics("Forward pass", 10.0, torch.device("cpu"))
    assert calls[5][1] == TraceEvent(
        scope="phase",
        name="phase1.forward_pass",
        phase="phase1",
        elapsed_ms=500.0,
        attrs={"trace_batch_size": 4, **metadata},
        wall_clock=True,
    )
