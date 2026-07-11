"""Typed lifecycle facade over structured attribution telemetry."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Protocol

from circuit_tracer.observability.exception_export import (
    _attach_telemetry_export_to_exception,
)
from circuit_tracer.observability.recorder import TelemetryRecorder


class TelemetryRecorderLike(Protocol):
    """Recorder operations required by :class:`TelemetryObserver`."""

    def record_event(
        self,
        *,
        scope: str,
        name: str,
        phase: str | None = None,
        step_index: int | None = None,
        batch_index: int | None = None,
        elapsed_ms: float | int | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> None: ...

    def record_wall_clock_duration(
        self,
        *,
        scope: str,
        name: str,
        elapsed_ms: float | int | None,
        phase: str | None = None,
    ) -> None: ...

    def close(self) -> None: ...

    def export(
        self, *, include_events: bool = True, max_events: int | None = None
    ) -> dict[str, object]: ...


class TelemetryObserver:
    """Own paired lifecycle events, export, and terminal attachments."""

    def __init__(self, recorder: TelemetryRecorderLike) -> None:
        self.recorder = recorder

    @classmethod
    def create(
        cls,
        *,
        enabled: bool = True,
        max_events: int = 20000,
        jsonl_path: str | os.PathLike[str] | None = None,
        static_context: Mapping[str, object] | None = None,
    ) -> "TelemetryObserver":
        return cls(
            TelemetryRecorder(
                enabled=enabled,
                max_events=max_events,
                jsonl_path=jsonl_path,
                static_context=static_context,
            )
        )

    def event(
        self,
        *,
        scope: str,
        name: str,
        phase: str | None = None,
        step_index: int | None = None,
        batch_index: int | None = None,
        elapsed_ms: float | int | None = None,
        attrs: Mapping[str, object] | None = None,
        wall_clock: bool = False,
    ) -> None:
        self._record_lifecycle(
            scope=scope,
            name=name,
            phase=phase,
            step_index=step_index,
            batch_index=batch_index,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
            wall_clock=wall_clock,
            wall_clock_phase=phase,
        )

    def _record_lifecycle(
        self,
        *,
        scope: str,
        name: str,
        phase: str | None,
        step_index: int | None,
        batch_index: int | None,
        elapsed_ms: float | int | None,
        attrs: Mapping[str, object] | None,
        wall_clock: bool,
        wall_clock_phase: str | None,
    ) -> None:
        self.recorder.record_event(
            scope=scope,
            name=name,
            phase=phase,
            step_index=step_index,
            batch_index=batch_index,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
        )
        if wall_clock:
            duration_kwargs: dict[str, object] = {
                "scope": scope,
                "name": name,
                "elapsed_ms": elapsed_ms,
            }
            if wall_clock_phase is not None:
                duration_kwargs["phase"] = wall_clock_phase
            self.recorder.record_wall_clock_duration(**duration_kwargs)

    def run(
        self, *, name: str, elapsed_ms=None, attrs=None, wall_clock: bool = False
    ) -> None:
        self._record_lifecycle(
            scope="run",
            name=name,
            phase=None,
            step_index=None,
            batch_index=None,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
            wall_clock=wall_clock,
            wall_clock_phase=None,
        )

    def phase(
        self,
        *,
        name: str,
        phase: str,
        elapsed_ms=None,
        attrs=None,
        wall_clock: bool = False,
    ) -> None:
        self._record_lifecycle(
            scope="phase",
            name=name,
            phase=phase,
            step_index=None,
            batch_index=None,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
            wall_clock=wall_clock,
            wall_clock_phase=phase,
        )

    def batch(
        self,
        *,
        name: str,
        phase: str,
        batch_index: int,
        elapsed_ms=None,
        attrs=None,
        wall_clock: bool = False,
    ) -> None:
        self._record_lifecycle(
            scope="batch",
            name=name,
            phase=phase,
            step_index=None,
            batch_index=batch_index,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
            wall_clock=wall_clock,
            wall_clock_phase=None,
        )

    def close_export(self, *, include_events: bool = True) -> dict[str, object]:
        self.recorder.close()
        return self.recorder.export(include_events=include_events)

    @staticmethod
    def attach_compact_result(
        result: MutableMapping[str, object], telemetry_export: Mapping[str, object]
    ) -> None:
        result["telemetry_summary"] = telemetry_export["summary"]
        result["telemetry_events"] = telemetry_export.get("events", [])

    @staticmethod
    def attach_exception(exc: BaseException, telemetry_export: Mapping[str, object]) -> None:
        _attach_telemetry_export_to_exception(exc, telemetry_export)

    @staticmethod
    def render_human_summary(logger, telemetry_export: Mapping[str, object]) -> None:
        summary = telemetry_export["summary"]
        logger.info(
            "Telemetry summary | "
            f"event_count={summary.get('event_count')} | "
            f"stored_event_count={summary.get('stored_event_count')} | "
            f"dropped_event_count={summary.get('dropped_event_count')}"
        )


# Private compatibility name retained for the NNSight attribution migration.
_TelemetryObserver = TelemetryObserver
