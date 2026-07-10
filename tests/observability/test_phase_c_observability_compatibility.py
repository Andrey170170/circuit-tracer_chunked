"""Phase C compatibility contracts for structured observability consumers.

These tests intentionally exercise the recorder as a serialized integration
boundary.  Unit-level scalar sanitization and individual summary counters live
in ``tests/utils/test_telemetry.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from circuit_tracer.observability import TelemetryRecorder


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_leaf_observability_package_is_the_direct_recorder_boundary() -> None:
    assert TelemetryRecorder.__module__ == "circuit_tracer.observability.recorder"


def test_serialized_events_preserve_consumer_field_shape_and_sequence(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(
        jsonl_path=path,
        static_context={"run_id": "compatibility-run", "schema_version": 1},
    )

    recorder.record_event(
        scope="phase",
        name="phase0.prepare",
        phase="phase0",
        step_index=2,
        batch_index=1,
        elapsed_ms=3.5,
        attrs={"candidate_count": 7},
    )
    recorder.record_event(scope="run", name="attribute.done")
    recorder.close()

    records = _read_jsonl(path)
    assert [record["sequence"] for record in records] == [1, 2]
    assert [record["event_index"] for record in records] == [0, 1]
    assert all(record["run_id"] == "compatibility-run" for record in records)
    assert all(record["schema_version"] == 1 for record in records)
    assert {
        "event_index",
        "sequence",
        "t_rel_ms",
        "scope",
        "name",
    }.issubset(records[0])
    assert records[0]["phase"] == "phase0"
    assert records[0]["step_index"] == 2
    assert records[0]["batch_index"] == 1
    assert records[0]["elapsed_ms"] == 3.5
    assert records[0]["attrs"] == {"candidate_count": 7}


def test_streamed_event_count_remains_complete_when_retention_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(max_events=2, jsonl_path=path)

    for index in range(5):
        recorder.record_event(scope="batch", name="phase3.batch", batch_index=index)
    recorder.close()

    exported = recorder.export(include_events=True)
    summary = exported["summary"]
    assert len(_read_jsonl(path)) == 5
    assert summary["event_count"] == 5
    assert summary["sink_event_count"] == 5
    assert summary["stored_event_count"] == 2
    assert summary["dropped_event_count"] == 3
    assert [event["batch_index"] for event in exported["events"]] == [0, 1]


def test_streamed_exception_event_keeps_lifecycle_context(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(jsonl_path=path, static_context={"run_id": "failed-run"})

    try:
        with recorder.timer(scope="phase", name="phase4.compute", phase="phase4"):
            raise RuntimeError("expected test failure")
    except RuntimeError:
        pass
    recorder.record_event(
        scope="run",
        name="attribute.failed",
        attrs={"error_type": "RuntimeError", "error_message": "expected test failure"},
    )
    recorder.close()

    records = _read_jsonl(path)
    assert [record["name"] for record in records] == ["phase4.compute", "attribute.failed"]
    assert records[0]["attrs"] == {"error_type": "RuntimeError"}
    assert records[1]["attrs"] == {
        "error_type": "RuntimeError",
        "error_message": "expected test failure",
    }
    assert records[0]["sequence"] < records[1]["sequence"]


def test_sink_write_failure_does_not_interrupt_subsequent_event_recording() -> None:
    class FailingSink:
        def write(self, _: str) -> int:
            raise OSError("simulated sink failure")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    recorder = TelemetryRecorder()
    recorder._sink = FailingSink()  # type: ignore[assignment]
    recorder._sink_status = "open"

    recorder.record_event(scope="run", name="attribute.start")
    recorder.record_event(scope="run", name="attribute.done")
    recorder.close()

    exported = recorder.export(include_events=True)
    assert [event["name"] for event in exported["events"]] == [
        "attribute.start",
        "attribute.done",
    ]
    assert exported["summary"]["sink_status"] == "error"
    assert exported["summary"]["sink_error_count"] == 2
