import json

import torch

from circuit_tracer.observability.recorder import TelemetryRecorder, sanitize_attrs


def test_recorder_sanitizes_and_streams_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(jsonl_path=path, static_context={"run": "direct"})

    recorder.record_event(
        scope="op", name="sample", attrs=sanitize_attrs({"value": torch.tensor(3)})
    )
    recorder.close()

    record = json.loads(path.read_text())
    assert record["run"] == "direct"
    assert record["attrs"] == {"value": 3}


def test_recorder_buffers_sink_with_bounded_event_loss_window(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(
        jsonl_path=path,
        sink_flush_interval_events=4,
    )

    for index in range(10):
        recorder.record_event(scope="op", name="sample", step_index=index)

    summary_before_close = recorder.build_summary()
    assert summary_before_close["sink_event_count"] == 10
    assert summary_before_close["sink_flush_count"] == 2
    assert summary_before_close["sink_pending_event_count"] == 2
    assert summary_before_close["sink_max_pending_event_count"] == 4
    assert summary_before_close["sink_max_crash_loss_events"] == 3

    recorder.close()

    assert len(path.read_text().splitlines()) == 10
    summary_after_close = recorder.build_summary()
    assert summary_after_close["sink_flush_count"] == 3
    assert summary_after_close["sink_pending_event_count"] == 0


def test_recorder_flushes_phase_and_run_boundaries(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = TelemetryRecorder(
        jsonl_path=path,
        sink_flush_interval_events=64,
    )

    recorder.record_event(scope="op", name="work")
    assert recorder.build_summary()["sink_flush_count"] == 0
    recorder.record_event(scope="phase", name="phase0.done", phase="phase0")
    assert recorder.build_summary()["sink_flush_count"] == 1
    assert len(path.read_text().splitlines()) == 2

    recorder.record_event(scope="op", name="more-work")
    recorder.record_event(scope="run", name="attribute.failed")
    assert recorder.build_summary()["sink_flush_count"] == 2
    assert len(path.read_text().splitlines()) == 4
    recorder.close()
