from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

import circuit_tracer.runtime as runtime


class FakeModel:
    backend = "nnsight"
    provider_id = "fake-v1"


def request(value=1, **kwargs):
    return runtime.TraceRequest(prompt=[value], model=FakeModel(), **kwargs)


def test_attribute_facade_and_trace_one_share_implementation(monkeypatch):
    calls = []

    def execute(prompt, model, **kwargs):
        calls.append((prompt, model, kwargs))
        return {"value": prompt[0], "telemetry_summary": {"events": 2}}

    monkeypatch.setattr("circuit_tracer.attribution.attribute_nnsight._attribute_impl", execute)
    from circuit_tracer.attribution.attribute_nnsight import attribute

    direct = runtime.trace_one(request(3, legacy_kwargs={"batch_size": 7}))
    facade = attribute([3], FakeModel(), batch_size=7)

    assert direct.output == facade
    assert direct.telemetry_summary == {"events": 2}
    assert len(calls) == 2
    assert calls[0][2]["batch_size"] == calls[1][2]["batch_size"] == 7
    assert calls[0][2]["telemetry_context"]["runtime_compatibility"]["translator"] == runtime.LEGACY_TRANSLATOR_VERSION


def test_legacy_coupled_mapping_and_refresh_cadence(monkeypatch):
    captured = {}

    def execute(prompt, model, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("circuit_tracer.attribution.attribute_nnsight._attribute_impl", execute)
    translated = runtime.request_from_legacy(
        [1], FakeModel(), feature_batch_size=12, update_interval=3
    )
    result = runtime.trace_one(translated)

    assert translated.logical.feature_group_size == 12
    assert translated.logical.phase4_reference_frontier_batch == 12
    assert translated.logical.phase4_refresh_stride == 3
    assert translated.physical.phase4_microbatch_max_rows == 12
    assert captured["phase4_compute_microbatch_max_rows"] == 12
    mapping = result.compatibility_metadata["translated"]
    assert mapping["logical"]["feature_batch_size"] == {
        "feature_group_size": 12,
        "phase4_reference_frontier_batch": 12,
    }
    assert mapping["logical"]["update_interval"] == 3


def test_legacy_feature_group_can_use_explicit_legacy_physical_split(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight._attribute_impl",
        lambda prompt, model, **kwargs: captured.update(kwargs) or {},
    )
    trace_request = runtime.request_from_legacy(
        [1],
        FakeModel(),
        feature_batch_size=12,
        phase4_compute_microbatch_max_rows=4,
    )
    runtime.trace_one(trace_request)
    assert captured["feature_batch_size"] == 12
    assert captured["phase4_compute_microbatch_max_rows"] == 4


def test_attribute_preserves_implementation_signature():
    from circuit_tracer.attribution.attribute_nnsight import _attribute_impl, attribute

    assert inspect.signature(attribute) == inspect.signature(_attribute_impl)
    assert inspect.signature(attribute).parameters["exact_trace_internal_dtype"].default == "fp32"


def test_exported_attribute_exposes_tiled_mechanism_controls():
    from circuit_tracer import attribute

    signature = inspect.signature(attribute)
    expected_defaults = {
        "full_retention_backend": "full_file",
        "feature_row_column_tile_size": 2048,
        "influence_row_tile_size": 4096,
        "influence_column_tile_size": 2048,
        "feature_row_retention": "full_file",
        "replay_tile_cache_bytes": None,
    }
    assert {
        name: signature.parameters[name].default for name in expected_defaults
    } == expected_defaults


def test_exported_attribute_forwards_tiled_mechanism_controls(monkeypatch):
    import circuit_tracer.attribution.attribute_nnsight as nnsight_module
    from circuit_tracer import attribute

    captured = {}
    monkeypatch.setattr(
        nnsight_module,
        "attribute",
        lambda prompt, model, **kwargs: captured.update(kwargs) or "graph",
    )
    assert (
        attribute(
            [1],
            FakeModel(),
            full_retention_backend="column_tiled_v1",
            feature_row_column_tile_size=32,
            influence_row_tile_size=64,
            influence_column_tile_size=16,
            feature_row_retention="none_recompute",
            replay_tile_cache_bytes=1024,
        )
        == "graph"
    )
    assert captured["full_retention_backend"] == "column_tiled_v1"
    assert captured["feature_row_column_tile_size"] == 32
    assert captured["influence_row_tile_size"] == 64
    assert captured["influence_column_tile_size"] == 16
    assert captured["feature_row_retention"] == "none_recompute"
    assert captured["replay_tile_cache_bytes"] == 1024


def test_fingerprints_are_stable_and_split_logical_from_physical():
    first = request(4)
    same = request(4)
    physical = request(4, physical=runtime.TracePhysicalControls(session_capacity=8))
    logical = request(4, logical=runtime.TraceLogicalSemantics(source_group_size=8))

    assert first.semantic_fingerprint == same.semantic_fingerprint
    assert first.execution_fingerprint == same.execution_fingerprint
    assert first.semantic_fingerprint == physical.semantic_fingerprint
    assert first.execution_fingerprint != physical.execution_fingerprint
    assert first.semantic_fingerprint != logical.semantic_fingerprint
    assert first.execution_fingerprint == logical.execution_fingerprint


def test_conflict_is_rejected_before_implementation(monkeypatch):
    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight._attribute_impl",
        lambda *args, **kwargs: pytest.fail("implementation reached before preflight"),
    )
    conflicting = request(
        logical=runtime.TraceLogicalSemantics(source_group_size=8),
        legacy_kwargs={"batch_size": 16},
    )
    with pytest.raises(ValueError, match="conflicting logical value"):
        runtime.trace_one(conflicting)


def test_mixed_batch_order_isolation_failure_and_cancellation(monkeypatch):
    def execute(prompt, model, **kwargs):
        if prompt == [2]:
            raise LookupError("bad shape")
        return list(prompt)

    monkeypatch.setattr("circuit_tracer.attribution.attribute_nnsight._attribute_impl", execute)
    results = runtime.trace_batch([request(1), request(2), request(3)], failure="return")
    assert [result.status for result in results] == [
        runtime.TraceStatus.SUCCEEDED,
        runtime.TraceStatus.FAILED,
        runtime.TraceStatus.SUCCEEDED,
    ]
    assert [result.output for result in results] == [[1], None, [3]]
    with pytest.raises(LookupError, match="bad shape"):
        runtime.trace_batch([request(1), request(2), request(3)])

    cancelled = threading.Event()
    cancelled.set()
    results = runtime.trace_batch([request(1), request(3)], failure="return", cancellation=cancelled)
    assert [result.status for result in results] == [runtime.TraceStatus.CANCELLED] * 2


def test_batch_does_not_convert_base_exceptions(monkeypatch):
    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight._attribute_impl",
        lambda *args, **kwargs: (_ for _ in ()).throw(asyncio.CancelledError()),
    )
    with pytest.raises(asyncio.CancelledError):
        runtime.trace_batch([request(1)], failure="return")


def test_graph_style_telemetry_summary_is_extracted(monkeypatch):
    class Output:
        telemetry_summary = {"event_count": 9}

    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight._attribute_impl",
        lambda *args, **kwargs: Output(),
    )
    assert runtime.trace_one(request()).telemetry_summary == {"event_count": 9}


def test_session_reuse_reset_close_and_failure_recovery(monkeypatch):
    delegates = []

    class Delegate:
        def __init__(self, **kwargs):
            self.cleaned = 0
            self.fail = True
            delegates.append(self)

        def attribute_target_position(self, position, **kwargs):
            if self.fail:
                self.fail = False
                raise RuntimeError("injected")
            return f"graph-{position}"

        def cleanup(self):
            self.cleaned += 1

    monkeypatch.setattr(
        "circuit_tracer.attribution.attribute_nnsight.FullSequenceWindowAttributionSession",
        Delegate,
    )
    session = runtime.open_session(request([1, 2, 3]), window_max_prefix_len=3)
    with pytest.raises(RuntimeError, match="injected"):
        session.trace_window(2, reuse=True)
    assert session.trace_window(2, reuse=True).output == "graph-2"
    assert len(delegates) == 1
    session.reset()
    assert delegates[0].cleaned == 1
    with pytest.raises(RuntimeError, match="injected"):
        session.trace_window(2, reuse=True)
    session.close()
    assert delegates[1].cleaned == 1
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.trace()
