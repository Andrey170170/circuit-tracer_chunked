import torch

from circuit_tracer.attribution.attribute_nnsight import _FileBackedFeatureRowStore
from circuit_tracer.utils.telemetry import TelemetryRecorder


def test_file_backed_feature_row_store_emits_structured_events() -> None:
    recorder = TelemetryRecorder(enabled=True)
    store = _FileBackedFeatureRowStore(
        n_rows=4,
        n_feature_columns=3,
        dtype=torch.float32,
        telemetry_recorder=recorder,
    )

    try:
        rows = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 0.75]], dtype=torch.float32)
        full_row_abs_sums = torch.tensor([6.0, 1.5], dtype=torch.float32)
        store.append_rows(
            row_start=0,
            feature_rows=rows,
            full_row_abs_sums=full_row_abs_sums,
            phase="phase3",
        )

        read_rows = store.read_feature_rows(0, 2, phase="phase4")
        assert read_rows.shape == (2, 3)

        dense = store.materialize_dense_feature_slice(
            row_start=0,
            row_end=2,
            selected_feature_columns=torch.tensor([0, 2]),
            phase="phase5",
        )
        assert dense.shape == (2, 2)
    finally:
        store.cleanup()

    stats = store.get_diagnostic_snapshot()
    assert stats["append_call_count"] == 1
    assert stats["read_call_count"] >= 1
    assert stats["materialize_call_count"] == 1

    summary = recorder.build_summary()
    assert summary["counts_by_scope"]["op"] >= 3
    events = recorder.export(include_events=True)["events"]
    names = {event["name"] for event in events}
    assert "feature_row_store.append_rows" in names
    assert "feature_row_store.read_rows" in names
    assert "feature_row_store.materialize_dense_slice" in names
