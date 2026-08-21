from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest

from circuit_tracer.observability.events import DiagnosticSnapshot, TraceEvent
from circuit_tracer.diagnostic import ProbeCompletion
from circuit_tracer.tracing import (
    AttributionProblem,
    DecoderCachePolicy,
    ExecutionConstraints,
    FrontierExpansionPlan,
    FrontierSemantics,
    ObservabilityPolicy,
    PrefixViewTarget,
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
from circuit_tracer.transcoder.provider import TranscoderCapabilities
from circuit_tracer.tracing.runner import _open_observability


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

    def execute(
        problem,
        plan,
        *,
        observer,
        forward_overrides,
        execution_identity,
        governor_runtime,
    ):
        assert governor_runtime is None
        execution_identity.mark_requested_as_effective()
        captured.update(
            problem=problem,
            plan=plan,
            observer=observer,
            forward_overrides=forward_overrides,
        )
        return {"value": problem.prompt[0]}

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)
    selected = request(
        3,
        semantics=TraceSemantics(
            source_batch_size=7,
            feature_batch_size=5,
            frontier=FrontierSemantics(scheduler="planner_v1"),
        ),
        execution=ExecutionConstraints(
            session=SessionPlan(capacity=8),
            storage=RowStoragePlan(full_retention_backend="column_tiled_v1"),
            replay=ReplayPlan(feature_window=2),
            frontier=FrontierExpansionPlan(row_executor="streaming_v1"),
            observability=ObservabilityPolicy(profile=True),
        ),
        evidence=TraceEvidence(name="gate", version="v1"),
    )

    result = trace_one(selected)

    assert result.output["value"] == 3
    assert result.effective_execution_fingerprint == result.requested_execution_fingerprint
    assert result.execution_fingerprint == result.effective_execution_fingerprint
    assert result.telemetry_summary["event_count"] == 2
    assert result.telemetry_summary["requested_execution_fingerprint"] == (
        result.requested_execution_fingerprint
    )
    assert result.telemetry_summary["effective_execution_fingerprint"] == (
        result.effective_execution_fingerprint
    )
    assert result.telemetry_events[-1]["attrs"]["effective_execution_fingerprint"] == (
        result.effective_execution_fingerprint
    )
    assert captured["problem"] is selected.problem
    plan = captured["plan"]
    assert plan.semantics is selected.semantics
    assert plan.execution is selected.execution
    assert plan.backend == "nnsight"
    assert captured["forward_overrides"] is None


def test_fingerprints_are_stable_and_split_semantics_from_execution() -> None:
    first = resolve_trace_request(request(4))
    same = resolve_trace_request(request(4))
    physical = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(session=SessionPlan(capacity=8)),
        )
    )
    semantic = resolve_trace_request(request(4, semantics=TraceSemantics(source_batch_size=8)))

    assert first.semantic_fingerprint == same.semantic_fingerprint
    assert first.execution_fingerprint == same.execution_fingerprint
    assert first.semantic_fingerprint == physical.semantic_fingerprint
    assert first.execution_fingerprint != physical.execution_fingerprint
    assert first.semantic_fingerprint != semantic.semantic_fingerprint
    assert first.execution_fingerprint == semantic.execution_fingerprint


def test_frontier_membership_controls_are_semantic_not_physical() -> None:
    base = resolve_trace_request(request(4))
    membership = resolve_trace_request(
        request(
            4,
            semantics=TraceSemantics(
                frontier=FrontierSemantics(
                    scheduler="planner_v1",
                    refresh_policy="deferred_v1",
                    refresh_interval_multiplier=2,
                    ranker="topk_v1",
                    phase3_buffer_relative_epsilon=0.01,
                    phase3_buffer_max_extra=2,
                    phase4_buffer_relative_epsilon=0.02,
                    phase4_buffer_max_extra_per_refresh=3,
                    phase4_buffer_max_extra_total=4,
                )
            ),
        )
    )
    mechanism = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(
                frontier=FrontierExpansionPlan(
                    row_executor="streaming_v1",
                    refresh_optimization="off",
                )
            ),
        )
    )

    assert membership.semantic_fingerprint != base.semantic_fingerprint
    assert membership.execution_fingerprint == base.execution_fingerprint
    assert mechanism.semantic_fingerprint == base.semantic_fingerprint
    assert mechanism.execution_fingerprint != base.execution_fingerprint


def test_evidence_provenance_does_not_change_fingerprints() -> None:
    first = resolve_trace_request(request(4, evidence=TraceEvidence(name="gate-a", version="1")))
    second = resolve_trace_request(request(4, evidence=TraceEvidence(name="gate-b", version="2")))
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert first.execution_fingerprint == second.execution_fingerprint


def test_prefix_view_target_and_mode_change_semantic_fingerprint() -> None:
    base = resolve_trace_request(request([1, 2, 3], output_position=1))
    independent = resolve_trace_request(
        request(
            [1, 2, 3],
            output_position=1,
            prefix_view=PrefixViewTarget(mode="independent_prefix", target_position=2),
        )
    )
    reused = resolve_trace_request(
        request(
            [1, 2, 3],
            output_position=1,
            prefix_view=PrefixViewTarget(
                mode="full_sequence_target_position",
                target_position=2,
            ),
        )
    )
    assert (
        len(
            {
                base.semantic_fingerprint,
                independent.semantic_fingerprint,
                reused.semantic_fingerprint,
            }
        )
        == 3
    )
    assert base.execution_fingerprint == independent.execution_fingerprint
    assert independent.execution_fingerprint == reused.execution_fingerprint


def test_observability_sink_output_fields_do_not_change_execution_fingerprint() -> None:
    base = resolve_trace_request(request(4))
    sink_only = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(
                observability=ObservabilityPolicy(
                    telemetry_jsonl_path="/different/output/telemetry.jsonl",
                    telemetry_context={"run_id": "different"},
                )
            ),
        )
    )
    behavior = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(
                observability=ObservabilityPolicy(telemetry_max_events=3),
            ),
        )
    )
    assert base.execution_fingerprint == sink_only.execution_fingerprint
    assert base.execution_fingerprint != behavior.execution_fingerprint


def test_explicit_telemetry_disable_overrides_compact_output_default() -> None:
    plan = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(
                compact_output=True,
                observability=ObservabilityPolicy(telemetry_enabled=False),
            ),
        )
    )
    observer, _ = _open_observability(plan)
    observer.observe(TraceEvent(scope="op", name="should.not.record"))
    assert observer.observe(DiagnosticSnapshot(object())) == {}
    export = observer.close_export()
    assert export["summary"]["enabled"] is False
    assert export["summary"]["event_count"] == 0
    assert export["events"] == []


def test_decoder_cache_policy_changes_only_execution_fingerprint() -> None:
    base = resolve_trace_request(request(4))
    cached = resolve_trace_request(
        request(
            4,
            execution=ExecutionConstraints(
                session=SessionPlan(decoder_cache=DecoderCachePolicy(enabled=True, max_bytes=4096))
            ),
        )
    )
    assert base.semantic_fingerprint == cached.semantic_fingerprint
    assert base.execution_fingerprint != cached.execution_fingerprint


def test_provider_semantic_identity_and_physical_capabilities_hash_separately() -> None:
    class Provider:
        def __init__(self, *, checkpoint: str, cache_bytes: int) -> None:
            self.scan = checkpoint
            self.capabilities = TranscoderCapabilities(
                architecture="clt",
                checkpoint_format="test",
                supports_exact_chunked_provider=True,
                supports_decoder_chunk_cache=True,
                default_decoder_chunk_size=8192,
                default_cross_batch_decoder_cache_bytes=cache_bytes,
            )

    class Model(FakeModel):
        def __init__(self, *, checkpoint: str, cache_bytes: int) -> None:
            self.transcoders = Provider(checkpoint=checkpoint, cache_bytes=cache_bytes)

    def resolved(checkpoint: str, cache_bytes: int):
        selected = TraceRequest(
            problem=AttributionProblem(
                prompt=[4],
                model=Model(checkpoint=checkpoint, cache_bytes=cache_bytes),
            )
        )
        return resolve_trace_request(selected)

    base = resolved("checkpoint-a", 1024)
    physical = resolved("checkpoint-a", 2048)
    semantic = resolved("checkpoint-b", 1024)
    assert base.semantic_fingerprint == physical.semantic_fingerprint
    assert base.execution_fingerprint != physical.execution_fingerprint
    assert base.semantic_fingerprint != semantic.semantic_fingerprint
    assert base.execution_fingerprint == semantic.execution_fingerprint


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
    def execute(
        problem,
        _plan,
        *,
        observer,
        forward_overrides,
        execution_identity,
        governor_runtime,
    ):
        del observer, governor_runtime
        assert forward_overrides is None
        if problem.prompt == [2]:
            raise LookupError("bad shape")
        execution_identity.mark_requested_as_effective()
        return list(problem.prompt)

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)
    results = trace_batch([request(1), request(2), request(3)], failure="return")
    assert [result.status for result in results] == [
        TraceStatus.SUCCEEDED,
        TraceStatus.FAILED,
        TraceStatus.SUCCEEDED,
    ]
    assert [result.output for result in results] == [[1], None, [3]]
    assert results[1].effective_execution_fingerprint is None
    assert results[1].execution_fingerprint == results[1].requested_execution_fingerprint
    with pytest.raises(LookupError, match="bad shape"):
        trace_batch([request(1), request(2), request(3)])

    cancelled = threading.Event()
    cancelled.set()
    results = trace_batch([request(1), request(3)], failure="return", cancellation=cancelled)
    assert [result.status for result in results] == [TraceStatus.CANCELLED] * 2


def test_batch_does_not_convert_base_exceptions(monkeypatch) -> None:
    def cancel(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", cancel)
    with pytest.raises(asyncio.CancelledError):
        trace_batch([request(1)], failure="return")


def test_canonical_observer_owns_telemetry_summary(monkeypatch) -> None:
    class Output:
        telemetry_summary = {"event_count": 9}

    def execute(*_args, observer, execution_identity, **_kwargs):
        execution_identity.mark_requested_as_effective()
        observer.observe(TraceEvent(scope="phase", name="test.event", phase="test"))
        return Output()

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace",
        execute,
    )
    selected = request(
        execution=ExecutionConstraints(observability=ObservabilityPolicy(profile=True))
    )
    assert trace_one(selected).telemetry_summary["event_count"] == 3


def test_probe_completion_persists_namespaced_diagnostic_metadata(monkeypatch) -> None:
    def execute(*_args, execution_identity, **_kwargs):
        execution_identity.mark_requested_as_effective()
        return ProbeCompletion(
            mode="phase0_probe",
            diagnostic_metadata={"mapped_row_bytes": 4096, "lifecycle": "sealed"},
        )

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute
    )

    result = trace_one(request())

    assert result.status is TraceStatus.PROBE_COMPLETED
    assert result.output is None
    assert result.telemetry_summary["diagnostic_metadata"] == {
        "mapped_row_bytes": 4096,
        "lifecycle": "sealed",
    }


def test_successful_trace_does_not_add_probe_diagnostic_metadata(monkeypatch) -> None:
    def execute(*_args, execution_identity, **_kwargs):
        execution_identity.mark_requested_as_effective()
        return {"graph": "unchanged"}

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute
    )

    result = trace_one(request())

    assert result.status is TraceStatus.SUCCEEDED
    assert result.output == {"graph": "unchanged"}
    assert "diagnostic_metadata" not in result.telemetry_summary


def test_session_reuse_reset_close_and_failure_recovery(monkeypatch) -> None:
    delegates = []

    class Delegate:
        def __init__(self, **_kwargs):
            self.closed = 0
            self.fail = True
            delegates.append(self)

        def prepare_target_position(self, position, selected):
            if self.fail:
                self.fail = False
                raise RuntimeError("injected")
            from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides

            return selected.problem, ForwardOverrides(target_logit_source=f"position-{position}")

        def close(self):
            self.closed += 1

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.forward_session.ForwardTraceSession",
        Delegate,
    )
    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace",
        lambda _problem, _plan, *, observer, forward_overrides, execution_identity, governor_runtime: (
            (
                execution_identity.mark_requested_as_effective(),
                governor_runtime is None,
                observer.observe(TraceEvent(scope="phase", name="window.test", phase="test")),
                f"graph-{forward_overrides.target_logit_source.removeprefix('position-')}",
            )[3]
        ),
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


def test_session_owns_bounded_decoder_cache_across_traces_and_failures(monkeypatch) -> None:
    class Provider:
        capabilities = TranscoderCapabilities(
            architecture="clt",
            checkpoint_format="test",
            supports_decoder_chunk_cache=True,
        )

        def __init__(self) -> None:
            self.created: list[tuple[object, int | None, object]] = []
            self.cleared: list[object] = []

        def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
            cache = object()
            self.created.append((cache, max_bytes, fingerprint))
            return cache

        def clear_decoder_block_cache(self, cache) -> None:
            self.cleared.append(cache)

    class Model(FakeModel):
        def __init__(self) -> None:
            self.transcoders = Provider()

    model = Model()
    selected = TraceRequest(problem=AttributionProblem(prompt=[1], model=model))
    seen: list[object] = []
    fail = False

    observers = []

    def execute(
        _problem,
        _plan,
        *,
        observer,
        forward_overrides,
        execution_identity,
        governor_runtime,
    ):
        assert governor_runtime is None
        execution_identity.mark_requested_as_effective()
        observers.append(observer)
        seen.append(forward_overrides.decoder_chunk_cache)
        if fail:
            raise RuntimeError("injected decoder-cache failure")
        return {"ok": True}

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)
    session = open_session(
        replace(
            selected,
            execution=ExecutionConstraints(
                session=SessionPlan(decoder_cache=DecoderCachePolicy(enabled=True, max_bytes=4096))
            ),
        ),
    )

    session.trace()
    session.trace()
    assert seen[0] is seen[1]
    assert observers[0] is not observers[1]
    assert model.transcoders.created[0][1] == 4096
    assert model.transcoders.created[0][2]["architecture"] == "clt"

    session.reset()
    assert model.transcoders.cleared == [seen[0]]
    session.trace()
    assert seen[2] is not seen[0]

    fail = True
    with pytest.raises(RuntimeError, match="decoder-cache failure"):
        session.trace()
    assert model.transcoders.cleared[-1] is seen[3]
    fail = False
    session.trace()
    assert seen[4] is not seen[3]
    session.close()
    assert model.transcoders.cleared[-1] is seen[4]


def test_trace_one_owns_and_clears_its_planned_decoder_cache(monkeypatch) -> None:
    class Provider:
        capabilities = TranscoderCapabilities(
            architecture="clt",
            checkpoint_format="test",
            supports_decoder_chunk_cache=True,
        )

        def __init__(self) -> None:
            self.created = []
            self.cleared = []

        def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
            cache = object()
            self.created.append((cache, max_bytes, fingerprint))
            return cache

        def clear_decoder_block_cache(self, cache) -> None:
            self.cleared.append(cache)

    class Model(FakeModel):
        def __init__(self) -> None:
            self.transcoders = Provider()

    model = Model()
    selected = TraceRequest(
        problem=AttributionProblem(prompt=[1], model=model),
        execution=ExecutionConstraints(
            session=SessionPlan(
                decoder_cache=DecoderCachePolicy(enabled=True, max_bytes=4096)
            )
        ),
    )
    seen = []

    def execute(
        _problem,
        _plan,
        *,
        observer,
        forward_overrides,
        execution_identity,
        governor_runtime,
    ):
        del observer
        assert governor_runtime is None
        execution_identity.mark_requested_as_effective()
        seen.append(forward_overrides.decoder_chunk_cache)
        return {"ok": True}

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)

    result = trace_one(selected)

    assert result.status is TraceStatus.SUCCEEDED
    assert seen == [model.transcoders.created[0][0]]
    assert model.transcoders.created[0][1] == 4096
    assert model.transcoders.cleared == seen


def test_session_window_resolves_effective_request(monkeypatch) -> None:
    calls = []

    class Delegate:
        def __init__(self, **_kwargs):
            pass

        def prepare_target_position(self, position, selected):
            from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides

            calls.append((position, selected))
            return selected.problem, ForwardOverrides()

        def close(self):
            pass

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.forward_session.ForwardTraceSession",
        Delegate,
    )
    backend_calls = []

    def execute(
        problem,
        plan,
        *,
        observer,
        forward_overrides,
        execution_identity,
        governor_runtime,
    ):
        assert governor_runtime is None
        execution_identity.mark_requested_as_effective()
        backend_calls.append((problem, plan, observer, forward_overrides))
        return "graph"

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace",
        execute,
    )
    base = request(
        [1, 2, 3],
        targets=["base"],
        semantics=TraceSemantics(source_batch_size=6),
        execution=ExecutionConstraints(storage=RowStoragePlan(influence_row_tile_size=2)),
    )
    session = open_session(base)
    first = session.trace_window(2, reuse=True)
    override = replace(base, problem=replace(base.problem, max_n_logits=3))
    second = session.trace_window(3, reuse=True, request=override)

    assert calls[0][1].problem.targets == ["base"]
    assert backend_calls[0][1].semantics.source_batch_size == 6
    assert backend_calls[0][1].execution.storage.influence_row_tile_size == 2
    assert calls[1][1].problem.max_n_logits == 3
    assert calls[0][1].problem.output_position == 1
    assert calls[1][1].problem.output_position == 2
    assert all(call[2] is not None for call in backend_calls)
    assert first.semantic_fingerprint != second.semantic_fingerprint


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_session_cleanup_attempts_delegate_and_cache_and_groups_failures(operation) -> None:
    calls = []

    class Delegate:
        def close(self):
            calls.append("delegate")
            raise RuntimeError("delegate cleanup")

    class CacheOwner:
        def reset(self):
            calls.append("cache-reset")
            raise LookupError("cache reset cleanup")

        def close(self):
            calls.append("cache-close")
            raise LookupError("cache close cleanup")

    session = open_session(request([1, 2, 3]))
    session._delegate = Delegate()
    session._decoder_cache = CacheOwner()

    exception_group_type = getattr(__import__("builtins"), "ExceptionGroup", None)
    if exception_group_type is None:
        exception_group_type = getattr(__import__("exceptiongroup"), "ExceptionGroup")
    with pytest.raises(exception_group_type) as captured:
        getattr(session, operation)()

    assert calls == ["delegate", f"cache-{operation}"]
    assert len(captured.value.exceptions) == 2


def test_session_context_preserves_primary_exception_and_attaches_cleanup_failures() -> None:
    calls = []

    class Delegate:
        def close(self):
            calls.append("delegate")
            raise RuntimeError("delegate cleanup")

    class CacheOwner:
        def close(self):
            calls.append("cache")
            raise LookupError("cache cleanup")

    session = open_session(request([1, 2, 3]))
    session._delegate = Delegate()
    session._decoder_cache = CacheOwner()

    with pytest.raises(ValueError, match="primary") as captured:
        with session:
            raise ValueError("primary")

    assert calls == ["delegate", "cache"]
    assert len(captured.value.__notes__) == 2
    assert all("cleanup also failed" in note for note in captured.value.__notes__)
