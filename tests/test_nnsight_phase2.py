from types import SimpleNamespace

import pytest

from circuit_tracer.attribution.nnsight.phases.phase2 import _make_replay_lifecycle
from circuit_tracer.attribution.nnsight.run_scope import raise_cleanup_failures


def test_replay_lifecycle_restores_offloaded_modules_once_before_rebuild() -> None:
    events: list[str] = []
    ctx = SimpleNamespace(
        reset_saved_graph_handles=lambda: events.append("reset"),
        rebuild_saved_graph_handles=lambda: events.append("rebuild"),
        release_saved_graph_handles=lambda: events.append("release"),
    )
    handles = [lambda: events.append("restore-1"), lambda: events.append("restore-2")]
    lifecycle = _make_replay_lifecycle(ctx, handles)

    lifecycle.begin_request()
    lifecycle.release()
    lifecycle.begin_request()
    lifecycle.release()

    assert events == [
        "reset",
        "restore-1",
        "restore-2",
        "rebuild",
        "release",
        "reset",
        "rebuild",
        "release",
    ]
    assert handles == []


def test_cleanup_only_failures_raise_exception_group() -> None:
    first = RuntimeError("feature cleanup failed")
    second = OSError("sink close failed")

    exception_group_type = getattr(__import__("builtins"), "ExceptionGroup")
    with pytest.raises(exception_group_type, match="lifecycle cleanup failed") as raised:
        raise_cleanup_failures([first, second])

    assert raised.value.exceptions == (first, second)
