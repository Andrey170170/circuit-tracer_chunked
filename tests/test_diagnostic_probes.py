from __future__ import annotations

from types import SimpleNamespace

import pytest

from circuit_tracer.attribution.nnsight.execution import (
    AttributionExecution,
    ProbeCompletion,
)
from circuit_tracer.observability.run_scope import TraceRunScope
from circuit_tracer.tracing.plan import DiagnosticStopPolicy


def test_diagnostic_stop_policy_requires_positive_transition_batch_count() -> None:
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
        )
    )
    execution.row_store_grant = None
    execution._grant = lambda phase: calls.append(f"grant:{phase.value}") or phase
    execution._release = lambda grant: calls.append(
        f"release:{getattr(grant, 'value', None)}"
    )
    execution.run_phase0_preparation = lambda: calls.append("run:phase0")
    execution.apply_active_universe_replan = lambda: calls.append("later")
    execution._run_with_grant = lambda phase, callback: callback()

    result = execution.run()

    assert result == ProbeCompletion(mode="phase0_probe")
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
        plan=SimpleNamespace(execution=SimpleNamespace(diagnostic_stop=policy))
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
    execution.apply_phase4_entry_replan = lambda: None

    def run_phase4() -> None:
        calls.append("phase4")
        execution.phase4 = SimpleNamespace(
            phase4_execution_batch_count=policy.phase4_batches
        )

    execution.expand_feature_frontier = run_phase4
    execution.assemble_graph = lambda: calls.append("phase5")

    result = execution.run()

    assert result == ProbeCompletion(
        mode="transition_probe", phase4_batches_completed=3
    )
    assert calls == ["phase0", "phase1", "phase2", "phase3", "phase4"]


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
