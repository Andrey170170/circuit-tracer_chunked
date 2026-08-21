from __future__ import annotations

import json
import logging

import pytest

from circuit_tracer.attribution.nnsight.run_scope import AttributionRunScope
from circuit_tracer.execution_identity import ExecutionIdentityState
from circuit_tracer.observability.events import TraceEvent
from circuit_tracer.observability.errors import safe_exception_attrs
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.observability.recorder import TelemetryRecorder
from circuit_tracer.observability.run_scope import TraceRunScope


def test_incremental_sink_is_complete_without_final_event_copy(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    observer = TelemetryObserver(TelemetryRecorder(max_events=3, jsonl_path=path))
    for index in range(25):
        observer.observe(TraceEvent(scope="op", name="row", step_index=index))

    exported = observer.close_export()

    assert "events" not in exported
    assert exported["summary"]["event_count"] == 25
    assert exported["summary"]["stored_event_count"] == 3
    assert exported["summary"]["sink_event_count"] == 25
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 25
    assert records[-1]["sequence"] == 25


def test_terminal_failures_do_not_mask_primary_exception() -> None:
    primary = RuntimeError("scientific failure")

    class BrokenObserver:
        def observe(self, _event) -> None:
            raise OSError("terminal event")

        def close_export(self):
            raise OSError("sink close")

        def attach_exception(self, *_args) -> None:
            raise OSError("attachment")

    scope = TraceRunScope(
        observer=BrokenObserver(),
        logger=logging.getLogger("test"),
        compact_output=False,
        profile=False,
        execution_identity=ExecutionIdentityState("requested"),
    )

    scope.close(primary)

    notes = getattr(primary, "__notes__", [])
    assert any("without masking the primary" in note for note in notes)
    assert any("attachment failed" in note for note in notes)


def test_unprintable_primary_exception_is_serialized_without_masking() -> None:
    class UnprintableFailure(RuntimeError):
        def __str__(self) -> str:
            raise KeyError("unsafe exception formatting")

    events = []

    class Observer:
        def observe(self, event) -> None:
            events.append(event)

        def close_export(self):
            return {"summary": {}, "events": []}

        def attach_exception(self, *_args) -> None:
            return None

    primary = UnprintableFailure()
    scope = TraceRunScope(
        observer=Observer(),
        logger=logging.getLogger("test"),
        compact_output=False,
        profile=False,
        execution_identity=ExecutionIdentityState("requested"),
    )

    scope.close(primary)

    terminal = events[-1]
    assert terminal.attrs["error_type"] == "UnprintableFailure"
    assert terminal.attrs["error_message"].startswith("<unavailable: str raised")


def test_typed_failure_details_are_preserved_as_bounded_primitives() -> None:
    error = RuntimeError("admission refused")
    error.details = {
        "decoder_active_row_hbm_free_bytes": 100,
        "decoder_active_row_effective_budget_bytes": 84,
        "decoder_active_row_admission_reason": "estimated_bytes_exceed_dynamic_budget",
        "unsafe": {"nested": "value"},
    }

    attrs = safe_exception_attrs(error)

    assert attrs["error_details"] == {
        "decoder_active_row_hbm_free_bytes": 100,
        "decoder_active_row_effective_budget_bytes": 84,
        "decoder_active_row_admission_reason": "estimated_bytes_exceed_dynamic_budget",
    }


def test_typed_failure_details_stop_safely_when_mapping_iteration_breaks() -> None:
    class BrokenDetails(dict):
        def items(self):
            yield "preserved", 1
            raise RuntimeError("broken details iterator")

    error = RuntimeError("admission refused")
    error.details = BrokenDetails()

    attrs = safe_exception_attrs(error)

    assert attrs["error_details"] == {"preserved": 1}


def test_nnsight_resources_cleanup_independently_and_preserve_primary() -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def cleanup(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise OSError(self.name)

    primary = RuntimeError("trace")
    scope = AttributionRunScope(
        offload_handles=[lambda: calls.append("offload")],
        feature_row_store=Resource("feature", fail=True),
        nonfeature_row_store=Resource("nonfeature"),
        ctx=Resource("context", fail=True),
    )

    scope.close(primary)

    assert calls == ["feature", "nonfeature", "context", "offload"]
    assert len(getattr(primary, "__notes__", [])) == 2


def test_cleanup_only_failures_are_grouped_after_all_attempts() -> None:
    calls: list[str] = []

    def fail(name: str):
        def callback() -> None:
            calls.append(name)
            raise OSError(name)

        return callback

    scope = AttributionRunScope(offload_handles=[fail("first"), fail("second")])

    exception_group_type = getattr(__import__("builtins"), "ExceptionGroup")
    with pytest.raises(exception_group_type) as raised:
        scope.close(None)

    assert calls == ["first", "second"]
    assert len(raised.value.exceptions) == 2
