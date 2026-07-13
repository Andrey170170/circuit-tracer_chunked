from __future__ import annotations

import logging

from circuit_tracer.observability.events import MemoryDelta
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.observability.recorder import TelemetryRecorder


def test_observer_records_paired_lifecycle_event_and_wall_clock() -> None:
    observer = TelemetryObserver(TelemetryRecorder())

    observer.phase(
        name="phase3.logit_attribution",
        phase="phase3",
        elapsed_ms=12.5,
        attrs={"batches": 2},
        wall_clock=True,
    )
    telemetry_export = observer.close_export()

    assert telemetry_export["events"] == [
        {
            "t_rel_ms": telemetry_export["events"][0]["t_rel_ms"],
            "scope": "phase",
            "name": "phase3.logit_attribution",
            "phase": "phase3",
            "elapsed_ms": 12.5,
            "attrs": {"batches": 2},
        }
    ]
    summary = telemetry_export["summary"]
    assert summary["event_count"] == 1
    assert summary["wall_clock_interval_count"] == 1
    assert summary["wall_clock_elapsed_ms_by_phase"] == {"phase3": 12.5}


def test_observer_adapts_memory_delta_through_keyword_only_resource_contract() -> None:
    observer = TelemetryObserver(TelemetryRecorder())

    attrs = observer.observe(
        MemoryDelta(
            before={"rss_gib": 2.0},
            after={"rss_gib": 3.5},
            keys=("rss_gib",),
        )
    )

    assert attrs == {
        "memory_before_rss_gib": 2.0,
        "memory_after_rss_gib": 3.5,
        "memory_delta_rss_gib": 1.5,
    }


def test_observer_terminal_success_does_not_mutate_scientific_result(caplog) -> None:
    observer = TelemetryObserver(TelemetryRecorder())
    observer.run(
        name="attribute.done",
        elapsed_ms=8.0,
        attrs={"compact_output": True},
        wall_clock=True,
    )
    telemetry_export = observer.close_export()

    with caplog.at_level(logging.INFO):
        observer.render_human_summary(logging.getLogger("observer-test"), telemetry_export)

    assert not hasattr(observer, "attach_compact_result")
    assert "event_count=1 | stored_event_count=1 | dropped_event_count=0" in caplog.text


def test_observer_terminal_failure_attaches_exception_telemetry() -> None:
    observer = TelemetryObserver(TelemetryRecorder())
    observer.run(
        name="attribute.failed",
        elapsed_ms=3.0,
        attrs={
            "compact_output": False,
            "error_type": "RuntimeError",
            "error_message": "synthetic failure",
        },
        wall_clock=True,
    )
    telemetry_export = observer.close_export()
    exc = RuntimeError("synthetic failure")

    observer.attach_exception(exc, telemetry_export)

    assert exc.circuit_tracer_telemetry_summary == telemetry_export["summary"]
    assert exc.circuit_tracer_telemetry_events == telemetry_export["events"]


class _FakeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def record_event(self, **kwargs) -> None:
        self.calls.append(("event", kwargs))

    def record_wall_clock_duration(self, **kwargs) -> None:
        self.calls.append(("wall_clock", kwargs))

    def close(self) -> None:
        self.calls.append(("close", {}))

    def export(self, *, include_events=True, max_events=None) -> dict[str, object]:
        self.calls.append(
            ("export", {"include_events": include_events, "max_events": max_events})
        )
        return {"summary": {}, "events": []}


def test_observer_fake_recorder_call_order_and_payload() -> None:
    recorder = _FakeRecorder()
    observer = TelemetryObserver(recorder)

    observer.batch(
        name="phase4.feature_batch",
        phase="phase4",
        batch_index=7,
        elapsed_ms=4.5,
        attrs={"batch_rows": 16},
        wall_clock=True,
    )
    observer.close_export()

    assert [name for name, _ in recorder.calls] == [
        "event",
        "wall_clock",
        "close",
        "export",
    ]
    assert recorder.calls[0][1] == {
        "scope": "batch",
        "name": "phase4.feature_batch",
        "phase": "phase4",
        "step_index": None,
        "batch_index": 7,
        "elapsed_ms": 4.5,
        "attrs": {"batch_rows": 16},
    }
    assert recorder.calls[1][1] == {
        "scope": "batch",
        "name": "phase4.feature_batch",
        "elapsed_ms": 4.5,
    }


def test_observer_phase_wall_clock_includes_phase() -> None:
    recorder = _FakeRecorder()
    observer = TelemetryObserver(recorder)

    observer.phase(
        name="phase3.logit_attribution",
        phase="phase3",
        elapsed_ms=12.5,
        wall_clock=True,
    )

    assert recorder.calls[1][1] == {
        "scope": "phase",
        "name": "phase3.logit_attribution",
        "phase": "phase3",
        "elapsed_ms": 12.5,
    }


def test_observer_run_wall_clock_omits_phase() -> None:
    recorder = _FakeRecorder()
    observer = TelemetryObserver(recorder)

    observer.run(name="attribute.done", elapsed_ms=8.0, wall_clock=True)

    assert recorder.calls[1][1] == {
        "scope": "run",
        "name": "attribute.done",
        "elapsed_ms": 8.0,
    }
