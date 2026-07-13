from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest

from circuit_tracer.tracing import (
    AttributionProblem,
    ExecutionConstraints,
    FrontierExpansionPlan,
    ObservabilityPolicy,
    ReplayPlan,
    RowStoragePlan,
    SessionPlan,
    SessionWindow,
    TraceEvidence,
    TraceRequest,
    TraceSemantics,
    TraceStatus,
    open_session,
    resolve_trace_request,
    trace_batch,
    trace_one,
)


class FakeModel:
    backend = "nnsight"
    provider_id = "fake-v1"


def request(
    value: int | list[int] = 1,
    *,
    semantics: TraceSemantics | None = None,
    execution: ExecutionConstraints | None = None,
    evidence: TraceEvidence | None = None,
    **problem_overrides: object,
) -> TraceRequest:
    prompt = value if isinstance(value, list) else [value]
    return TraceRequest(
        problem=AttributionProblem(
            prompt=prompt,
            model=FakeModel(),
            **problem_overrides,
        ),
        semantics=semantics or TraceSemantics(),
        execution=execution or ExecutionConstraints(),
        evidence=evidence or TraceEvidence(),
    )


def test_every_top_level_export_is_reachable() -> None:
    import circuit_tracer

    for name in circuit_tracer.__all__:
        assert getattr(circuit_tracer, name) is not None, name


def test_trace_one_passes_owned_problem_and_resolved_plan(monkeypatch) -> None:
    captured = {}

    def execute(problem, plan):
        captured.update(problem=problem, plan=plan)
        return {"value": problem.prompt[0], "telemetry_summary": {"events": 2}}

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute
    )
    selected = request(
        3,
        semantics=TraceSemantics(source_batch_size=7, feature_batch_size=5),
        execution=ExecutionConstraints(
            session=SessionPlan(capacity=8),
            storage=RowStoragePlan(full_retention_backend="column_tiled_v1"),
            replay=ReplayPlan(feature_window=2),
            frontier=FrontierExpansionPlan(scheduler="planner_v1"),
            observability=ObservabilityPolicy(profile=True),
        ),
        evidence=TraceEvidence(name="gate", version="v1"),
    )

    result = trace_one(selected)

    assert result.output["value"] == 3
    assert result.telemetry_summary == {"events": 2}
    assert captured["problem"] is selected.problem
    plan = captured["plan"]
    assert plan.semantics is selected.semantics
    assert plan.execution is selected.execution
    assert plan.backend == "nnsight"


def test_fingerprints_are_stable_and_split_semantics_from_execution() -> None:
    first = resolve_trace_request(request(4))
    same = resolve_trace_request(request(4))
    physical = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(session=SessionPlan(capacity=8)),
        )
    )
    semantic = resolve_trace_request(
        request(4, semantics=TraceSemantics(source_batch_size=8))
    )

    assert first.semantic_fingerprint == same.semantic_fingerprint
    assert first.execution_fingerprint == same.execution_fingerprint
    assert first.semantic_fingerprint == physical.semantic_fingerprint
    assert first.execution_fingerprint != physical.execution_fingerprint
    assert first.semantic_fingerprint != semantic.semantic_fingerprint
    assert first.execution_fingerprint == semantic.execution_fingerprint


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("max_n_logits", 2, 3),
        ("desired_logit_prob", 0.8, 0.9),
        ("targets", ["a"], ["b"]),
        ("output_position", 1, 2),
    ],
)
def test_problem_controls_change_semantic_fingerprint(field, first, second) -> None:
    left = resolve_trace_request(request(**{field: first}))
    right = resolve_trace_request(request(**{field: second}))
    assert left.semantic_fingerprint != right.semantic_fingerprint


def test_invalid_requests_fail_before_backend_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace",
        lambda *_: pytest.fail("backend reached before validation"),
    )
    with pytest.raises(ValueError, match="microbatch"):
        SessionPlan(capacity=2, phase3_microbatch_max_rows=3)
    with pytest.raises(ValueError, match="evidence name and version"):
        TraceEvidence(name="gate")


def test_transformerlens_rejects_nondefault_execution_constraints() -> None:
    model = FakeModel()
    model.backend = "transformerlens"
    selected = TraceRequest(
        problem=AttributionProblem(model=model, prompt=[1]),
        execution=ExecutionConstraints(session=SessionPlan(capacity=8)),
    )
    with pytest.raises(ValueError, match="only default execution constraints"):
        resolve_trace_request(selected)


def test_mixed_batch_order_isolation_failure_and_cancellation(monkeypatch) -> None:
    def execute(problem, _plan):
        if problem.prompt == [2]:
            raise LookupError("bad shape")
        return list(problem.prompt)

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute
    )
    results = trace_batch([request(1), request(2), request(3)], failure="return")
    assert [result.status for result in results] == [
        TraceStatus.SUCCEEDED,
        TraceStatus.FAILED,
        TraceStatus.SUCCEEDED,
    ]
    assert [result.output for result in results] == [[1], None, [3]]
    with pytest.raises(LookupError, match="bad shape"):
        trace_batch([request(1), request(2), request(3)])

    cancelled = threading.Event()
    cancelled.set()
    results = trace_batch(
        [request(1), request(3)], failure="return", cancellation=cancelled
    )
    assert [result.status for result in results] == [TraceStatus.CANCELLED] * 2


def test_batch_does_not_convert_base_exceptions(monkeypatch) -> None:
    def cancel(*_args):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", cancel
    )
    with pytest.raises(asyncio.CancelledError):
        trace_batch([request(1)], failure="return")


def test_graph_style_telemetry_summary_is_extracted(monkeypatch) -> None:
    class Output:
        telemetry_summary = {"event_count": 9}

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace",
        lambda *_: Output(),
    )
    assert trace_one(request()).telemetry_summary == {"event_count": 9}


def test_session_reuse_reset_close_and_failure_recovery(monkeypatch) -> None:
    delegates = []

    class Delegate:
        def __init__(self, **_kwargs):
            self.closed = 0
            self.fail = True
            delegates.append(self)

        def trace_target_position(self, position, _request, _plan):
            if self.fail:
                self.fail = False
                raise RuntimeError("injected")
            return f"graph-{position}"

        def close(self):
            self.closed += 1

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.forward_session.ForwardTraceSession",
        Delegate,
    )
    session = open_session(request([1, 2, 3]), window=SessionWindow(max_prefix_len=3))
    with pytest.raises(RuntimeError, match="injected"):
        session.trace_window(2, reuse=True)
    assert session.trace_window(2, reuse=True).output == "graph-2"
    assert len(delegates) == 1
    session.reset()
    assert delegates[0].closed == 1
    with pytest.raises(RuntimeError, match="injected"):
        session.trace_window(2, reuse=True)
    session.close()
    assert delegates[1].closed == 1
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.trace()


def test_session_window_resolves_effective_request(monkeypatch) -> None:
    calls = []

    class Delegate:
        def __init__(self, **_kwargs):
            pass

        def trace_target_position(self, position, selected, plan):
            calls.append((position, selected, plan))
            return "graph"

        def close(self):
            pass

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.forward_session.ForwardTraceSession",
        Delegate,
    )
    base = request(
        [1, 2, 3],
        targets=["base"],
        semantics=TraceSemantics(source_batch_size=6),
        execution=ExecutionConstraints(
            storage=RowStoragePlan(influence_row_tile_size=2)
        ),
    )
    session = open_session(base)
    first = session.trace_window(2, reuse=True)
    override = replace(base, problem=replace(base.problem, max_n_logits=3))
    second = session.trace_window(3, reuse=True, request=override)

    assert calls[0][1].problem.targets == ["base"]
    assert calls[0][2].semantics.source_batch_size == 6
    assert calls[0][2].execution.storage.influence_row_tile_size == 2
    assert calls[1][1].problem.max_n_logits == 3
    assert calls[0][1].problem.output_position == 1
    assert calls[1][1].problem.output_position == 2
    assert first.semantic_fingerprint != second.semantic_fingerprint
