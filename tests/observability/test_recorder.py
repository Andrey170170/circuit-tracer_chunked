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
