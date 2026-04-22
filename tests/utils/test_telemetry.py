import torch

from circuit_tracer.utils.telemetry import (
    TelemetryRecorder,
    get_memory_snapshot,
    sanitize_attrs,
)


def test_sanitize_attrs_scalarizes_complex_values() -> None:
    attrs = sanitize_attrs(
        {
            "int": 3,
            "float": 2.5,
            "inf": float("inf"),
            "tensor_scalar": torch.tensor(7),
            "tensor_vector": torch.tensor([1, 2, 3]),
            "sequence": [1, 2, 3],
            "mapping": {"nested": 4},
        }
    )

    assert attrs["int"] == 3
    assert attrs["float"] == 2.5
    assert attrs["inf"] == "inf"
    assert attrs["tensor_scalar"] == 7
    assert isinstance(attrs["tensor_vector"], str)
    assert "tensor(shape=" in attrs["tensor_vector"]
    assert isinstance(attrs["sequence"], str)
    assert isinstance(attrs["mapping"], str)


def test_telemetry_recorder_tracks_summary_and_dropped_events() -> None:
    recorder = TelemetryRecorder(enabled=True, max_events=2)
    recorder.record_event(
        scope="phase",
        name="phase0.precompute",
        phase="phase0",
        elapsed_ms=12.0,
        attrs={"active_features": 42},
    )
    recorder.record_event(scope="invalid_scope", name="scope.normalize")
    recorder.record_event(scope="op", name="dropped.third")

    summary = recorder.build_summary()
    assert summary["event_count"] == 3
    assert summary["stored_event_count"] == 2
    assert summary["dropped_event_count"] == 1
    assert summary["max_events"] == 2
    assert summary["counts_by_scope"]["op"] == 2
    assert summary["counts_by_scope"]["phase"] == 1

    exported = recorder.export(include_events=True)
    events = exported["events"]
    assert isinstance(events, list)
    assert len(events) == 2


def test_telemetry_recorder_tracks_wall_clock_durations_separately() -> None:
    recorder = TelemetryRecorder(enabled=True, max_events=1)
    recorder.record_event(scope="phase", name="phase4.refresh", phase="phase4", elapsed_ms=5.0)
    recorder.record_wall_clock_duration(
        scope="phase",
        name="phase4.refresh",
        phase="phase4",
        elapsed_ms=2.0,
    )

    summary = recorder.build_summary()
    assert summary["elapsed_ms_by_phase"]["phase4"] == 5.0
    assert summary["elapsed_ms_by_phase_aggregate"]["phase4"] == 5.0
    assert summary["wall_clock_elapsed_ms_by_phase"]["phase4"] == 2.0
    assert summary["wall_clock_elapsed_ms_total"] == 2.0


def test_telemetry_timer_records_elapsed_and_error_type() -> None:
    recorder = TelemetryRecorder(enabled=True)

    try:
        with recorder.timer(scope="phase", name="phase4.compute", phase="phase4"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    events = recorder.export(include_events=True)["events"]
    assert isinstance(events, list)
    assert len(events) == 1

    event = events[0]
    assert event["name"] == "phase4.compute"
    assert event["phase"] == "phase4"
    assert event["elapsed_ms"] >= 0
    assert event["attrs"]["error_type"] == "RuntimeError"


def test_get_memory_snapshot_reports_current_and_peak_rss_keys() -> None:
    snapshot = get_memory_snapshot(torch.device("cpu"))

    assert "rss_current_gib" in snapshot
    assert "rss_gib" in snapshot
    if snapshot["rss_current_gib"] is not None:
        assert snapshot["rss_current_gib"] >= 0
