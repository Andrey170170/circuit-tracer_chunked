from __future__ import annotations

from types import SimpleNamespace

import pytest

from circuit_tracer.attribution.nnsight.execution import (
    AttributionExecution,
    ProbeCompletion,
)
from circuit_tracer.attribution.nnsight.phases.phase4_batches import (
    _record_replay_cuda_window,
)
from circuit_tracer.observability.run_scope import TraceRunScope
from circuit_tracer.governor.ledger import PhaseId
from circuit_tracer.tracing.plan import DiagnosticStopPolicy


def test_diagnostic_stop_policy_requires_positive_transition_batch_count() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        DiagnosticStopPolicy(mode="typo")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        DiagnosticStopPolicy(mode="transition_probe", phase4_batches=0)
    with pytest.raises(ValueError, match="only"):
        DiagnosticStopPolicy(mode="phase0_probe", phase4_batches=1)
    assert DiagnosticStopPolicy(mode="transition_probe", phase4_batches=3).phase4_batches == 3


def test_phase0_probe_stops_before_all_later_phases_and_releases_session() -> None:
    calls: list[str] = []
    execution = object.__new__(AttributionExecution)
    execution.prepared = SimpleNamespace(
        plan=SimpleNamespace(
            execution=SimpleNamespace(
                diagnostic_stop=DiagnosticStopPolicy(mode="phase0_probe")
            )
        ),
        frontier=SimpleNamespace(execution_metadata={"mapped_rows": 11}),
    )
    execution.row_store_grant = None
    execution._grant = lambda phase: calls.append(f"grant:{phase.value}") or phase
    execution._release = lambda grant: calls.append(
        f"release:{getattr(grant, 'value', None)}"
    )
    def run_phase0() -> None:
        calls.append("run:phase0")
        execution.prepared.frontier.execution_metadata["mapped_rows"] = 12

    execution.run_phase0_preparation = run_phase0
    execution.apply_active_universe_replan = lambda: calls.append("later")
    execution._run_with_grant = lambda phase, callback: callback()

    result = execution.run()

    assert result == ProbeCompletion(
        mode="phase0_probe", diagnostic_metadata={"mapped_rows": 12}
    )
    with pytest.raises(TypeError):
        result.diagnostic_metadata["mapped_rows"] = 13  # type: ignore[index]
    assert calls == [
        "grant:session",
        "run:phase0",
        "release:None",
        "release:session",
    ]


def test_transition_probe_stops_after_declared_physical_batch_count() -> None:
    calls: list[str] = []
    policy = DiagnosticStopPolicy(mode="transition_probe", phase4_batches=3)
    execution = object.__new__(AttributionExecution)
    execution.prepared = SimpleNamespace(
        plan=SimpleNamespace(execution=SimpleNamespace(diagnostic_stop=policy)),
        frontier=SimpleNamespace(execution_metadata={"lifecycle_released": True}),
    )
    execution.row_store_grant = None
    execution._grant = lambda phase: phase
    execution._release = lambda grant: None
    execution._run_with_grant = lambda phase, callback: callback()
    execution.run_phase0_preparation = lambda: calls.append("phase0")
    execution.apply_active_universe_replan = lambda: None
    execution.run_forward_pass = lambda: calls.append("phase1")
    execution.setup_active_features_and_storage = lambda: calls.append("phase2")
    execution.apply_phase3_entry_replan = lambda: None
    execution.finalize_active_decoder_row_admission = lambda: None
    execution.attribute_seed_nodes = lambda: calls.append("phase3")
    execution.apply_checkpoint_working_set_transition = lambda: None
    execution.apply_phase4_entry_replan = lambda: None

    def run_phase4() -> None:
        calls.append("phase4")
        execution.prepared.frontier.execution_metadata["lifecycle_released"] = False
        execution.phase4 = SimpleNamespace(
            phase4_execution_batch_count=policy.phase4_batches
        )

    execution.expand_feature_frontier = run_phase4
    execution.assemble_graph = lambda: calls.append("phase5")

    result = execution.run()

    assert result == ProbeCompletion(
        mode="transition_probe",
        phase4_batches_completed=3,
        diagnostic_metadata={"lifecycle_released": False},
    )
    assert calls == ["phase0", "phase1", "phase2", "phase3", "phase4"]


def test_transition_probe_refuses_incomplete_phase4_count() -> None:
    policy = DiagnosticStopPolicy(mode="transition_probe", phase4_batches=3)
    execution = object.__new__(AttributionExecution)
    execution.prepared = SimpleNamespace(
        plan=SimpleNamespace(execution=SimpleNamespace(diagnostic_stop=policy))
    )
    execution.row_store_grant = None
    execution._grant = lambda phase: phase
    execution._release = lambda grant: None
    execution._run_with_grant = lambda phase, callback: callback()
    execution.run_phase0_preparation = lambda: None
    execution.apply_active_universe_replan = lambda: None
    execution.run_forward_pass = lambda: None
    execution.setup_active_features_and_storage = lambda: None
    execution.apply_phase3_entry_replan = lambda: None
    execution.finalize_active_decoder_row_admission = lambda: None
    execution.attribute_seed_nodes = lambda: None
    execution.apply_checkpoint_working_set_transition = lambda: None
    execution.apply_phase4_entry_replan = lambda: None
    execution.expand_feature_frontier = lambda: setattr(
        execution, "phase4", SimpleNamespace(phase4_execution_batch_count=2)
    )
    execution.assemble_graph = lambda: pytest.fail("Phase 5 must not run")

    with pytest.raises(RuntimeError, match="transition probe incomplete"):
        execution.run()


@pytest.mark.parametrize("failure_index", [0, 2])
def test_resource_observer_failure_does_not_leak_phase_grant(
    failure_index: int,
) -> None:
    calls: list[str] = []

    class Observer:
        index = 0

        def observe(self, observation):
            index = self.index
            self.index += 1
            if index == failure_index:
                raise OSError(f"observer failure {index}")
            if index in {0, 2}:
                return {"proc_minor_faults": index}
            if index == 3:
                return {"memory_delta_proc_minor_faults": 2}
            return None

    execution = object.__new__(AttributionExecution)
    execution.prepared = SimpleNamespace(
        diagnostics=SimpleNamespace(observer=Observer()),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: calls.append("warning")),
        plan=SimpleNamespace(
            execution=SimpleNamespace(
                observability=SimpleNamespace(telemetry_context={})
            )
        ),
    )
    execution._grant = lambda phase: calls.append(f"grant:{phase.value}") or phase
    execution._release = lambda grant: calls.append(f"release:{grant.value}")

    result = execution._run_with_grant(
        PhaseId.PHASE4, lambda: calls.append("callback") or "result"
    )

    assert result == "result"
    assert calls[0] == "grant:phase4"
    assert "callback" in calls
    assert calls[-1] == "release:phase4"
    assert calls.count("release:phase4") == 1


def test_callback_failure_records_failed_interval_and_releases_grant() -> None:
    events = []

    class Observer:
        def observe(self, observation):
            events.append(observation)
            if observation.__class__.__name__ == "MemorySnapshot":
                return {"proc_minor_faults": len(events)}
            if observation.__class__.__name__ == "MemoryDelta":
                return {"memory_delta_proc_minor_faults": 1}
            return None

    released = []
    execution = object.__new__(AttributionExecution)
    execution.prepared = SimpleNamespace(
        diagnostics=SimpleNamespace(observer=Observer()),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        plan=SimpleNamespace(
            execution=SimpleNamespace(
                observability=SimpleNamespace(telemetry_context={})
            )
        ),
    )
    execution._grant = lambda phase: phase
    execution._release = released.append

    with pytest.raises(RuntimeError, match="kernel"):
        execution._run_with_grant(
            PhaseId.PHASE4,
            lambda: (_ for _ in ()).throw(RuntimeError("kernel")),
        )

    failed = [
        event
        for event in events
        if getattr(event, "name", "") == "phase4.resource_interval.failed"
    ]
    assert len(failed) == 1
    assert failed[0].attrs["error_type"] == "RuntimeError"
    assert failed[0].attrs["memory_delta_proc_minor_faults"] == 1
    assert released == [PhaseId.PHASE4]


def test_tape_cuda_time_is_emitted_once_at_window_scope() -> None:
    events = []
    state = SimpleNamespace(
        telemetry_observer=SimpleNamespace(observe=events.append),
        config=SimpleNamespace(
            cache_state="cold",
            cache_state_provenance="test",
        ),
    )

    _record_replay_cuda_window(state, elapsed_ms=12.5, batch_count=3)

    assert len(events) == 1
    assert events[0].name == "phase4.feature_vjp_tape_window"
    assert events[0].attrs["physical_batch_count"] == 3
    assert events[0].attrs["cuda_kernel_elapsed_ms"] == 12.5


class _Observer:
    def __init__(self) -> None:
        self.events = []
        self.closed = False

    def observe(self, event) -> None:
        self.events.append(event)

    def close_export(self):
        self.closed = True
        return {"summary": {"sink_status": "closed"}, "events": []}

    def render_human_summary(self, logger, export) -> None:
        return None


def test_probe_terminal_is_canonical_and_sink_is_closed() -> None:
    observer = _Observer()
    identity = SimpleNamespace(
        requested_fingerprint="requested",
        effective_fingerprint="effective",
        execution_fingerprint="effective",
    )
    scope = TraceRunScope(
        observer=observer,
        logger=SimpleNamespace(),
        compact_output=False,
        profile=False,
        execution_identity=identity,
    )

    evidence = scope.close(None, terminal_status="probe_completed")

    terminal = observer.events[-1]
    assert terminal.name == "attribute.probe_completed"
    assert terminal.attrs["status"] == "probe_completed"
    assert terminal.attrs["status"] not in {"succeeded", "failed", "unknown"}
    assert observer.closed is True
    assert evidence.summary["sink_status"] == "closed"
    with pytest.raises(RuntimeError, match="already closed"):
        scope.close(None)
