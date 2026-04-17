import torch

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils.telemetry import TelemetryRecorder


def test_emit_trace_event_forwards_to_structured_recorder() -> None:
    recorder = TelemetryRecorder(enabled=True)
    transcoder = CrossLayerTranscoder(
        n_layers=1,
        d_transcoder=2,
        d_model=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_decoder=False,
        lazy_encoder=False,
    )
    transcoder.configure_trace_logging(logger=None, telemetry_recorder=recorder)

    transcoder.emit_trace_event(
        "phase0.custom_event",
        elapsed_ms=5.5,
        payload="ok",
    )

    exported = recorder.export(include_events=True)
    events = exported["events"]
    assert isinstance(events, list)
    assert len(events) == 1

    event = events[0]
    assert event["name"] == "transcoder.phase0.custom_event"
    assert event["phase"] == "phase0"
    assert event["elapsed_ms"] == 5.5
    assert event["attrs"]["payload"] == "ok"
