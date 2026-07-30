"""Typed lifecycle facade over structured attribution telemetry."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any, Protocol

from circuit_tracer.observability.events import (
    BatchProfile,
    DiagnosticSnapshot,
    DiagnosticsMessage,
    MemoryBoundary,
    MemoryDelta,
    MemorySnapshot,
    MemorySnapshotAttrs,
    NumericDelta,
    Observation,
    PhaseMetrics,
    RuntimeSnapshot,
    SparsificationProfile,
    TraceEvent,
)

from circuit_tracer.observability.exception_export import (
    _attach_telemetry_export_to_exception,
)
from circuit_tracer.observability.recorder import TelemetryRecorder
from circuit_tracer.observability.human_logs import (
    _log_batch_profile,
    _log_memory_boundary,
    _log_phase_metrics,
    _log_sparsification_profile,
    _snapshot_diagnostics,
)
from circuit_tracer.observability.resources import (
    build_memory_before_after_attrs,
    build_memory_snapshot_attrs,
    diff_numeric_metrics,
    format_numeric_metrics,
    get_memory_snapshot,
)


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

    def __init__(
        self,
        recorder: TelemetryRecorderLike,
        *,
        logger: Any = None,
        enabled: bool = True,
    ) -> None:
        self._recorder = recorder
        self.logger = logger
        self._enabled = bool(enabled)

    @classmethod
    def create(
        cls,
        *,
        enabled: bool = True,
        max_events: int = 20000,
        jsonl_path: str | os.PathLike[str] | None = None,
        static_context: Mapping[str, object] | None = None,
        logger: Any = None,
    ) -> "TelemetryObserver":
        return cls(
            TelemetryRecorder(
                enabled=enabled,
                max_events=max_events,
                jsonl_path=jsonl_path,
                static_context=static_context,
            ),
            logger=logger,
            enabled=enabled,
        )

    def observe(self, observation: Observation) -> object | None:
        """Adapt one typed domain observation to recording, sampling, or rendering."""
        if not self._enabled:
            if isinstance(observation, RuntimeSnapshot):
                return {}, {}
            if isinstance(
                observation,
                (
                    DiagnosticSnapshot,
                    MemorySnapshot,
                    MemorySnapshotAttrs,
                    MemoryDelta,
                    NumericDelta,
                ),
            ):
                return {}
            return None
        if isinstance(observation, TraceEvent):
            self.event(
                scope=observation.scope,
                name=observation.name,
                phase=observation.phase,
                step_index=observation.step_index,
                batch_index=observation.batch_index,
                elapsed_ms=observation.elapsed_ms,
                attrs=observation.attrs,
                wall_clock=observation.wall_clock,
            )
            return None
        if isinstance(observation, MemoryBoundary):
            _log_memory_boundary(
                self.logger, observation.label, observation.device, **observation.extra
            )
            return None
        if isinstance(observation, PhaseMetrics):
            _log_phase_metrics(
                self.logger,
                observation.label,
                observation.started_at,
                observation.device,
                **observation.extra,
            )
            return None
        if isinstance(observation, BatchProfile):
            _log_batch_profile(
                self.logger,
                observation.label,
                observation.batch_index,
                observation.total_batches,
                observation.elapsed_seconds,
                dict(observation.context_before) if observation.context_before else None,
                dict(observation.context_after) if observation.context_after else None,
                dict(observation.transcoder_before)
                if observation.transcoder_before
                else None,
                dict(observation.transcoder_after) if observation.transcoder_after else None,
            )
            return None
        if isinstance(observation, SparsificationProfile):
            _log_sparsification_profile(self.logger, dict(observation.stats))
            return None
        if isinstance(observation, DiagnosticsMessage):
            self.logger.info(
                f"{observation.label} | "
                f"{format_numeric_metrics(observation.diagnostics, limit=observation.limit)}"
            )
            return None
        if isinstance(observation, DiagnosticSnapshot):
            return _snapshot_diagnostics(observation.source)
        if isinstance(observation, MemorySnapshot):
            return get_memory_snapshot(observation.device)
        if isinstance(observation, MemorySnapshotAttrs):
            return build_memory_snapshot_attrs(
                observation.snapshot,
                keys=observation.keys,
                prefix=observation.prefix,
            )
        if isinstance(observation, MemoryDelta):
            return build_memory_before_after_attrs(
                before=observation.before,
                after=observation.after,
                keys=observation.keys,
            )
        if isinstance(observation, NumericDelta):
            return diff_numeric_metrics(observation.before, observation.after)
        if isinstance(observation, RuntimeSnapshot):
            memory = get_memory_snapshot(observation.device)
            context = _snapshot_diagnostics(observation.context)
            transcoder = _snapshot_diagnostics(observation.transcoder)

            def digest(value: object) -> str | None:
                if value is None:
                    return None
                encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
                return hashlib.sha1(encoded).hexdigest()[:16]

            summary = {
                "memory_snapshot": memory,
                "ctx_diagnostic_snapshot": context,
                "transcoder_diagnostic_snapshot": transcoder,
                "ctx_diagnostic_snapshot_hash": digest(context),
                "transcoder_diagnostic_snapshot_hash": digest(transcoder),
            }
            stream = {
                key: memory.get(key)
                for key in (
                    "rss_current_gib",
                    "rss_gib",
                    "cuda_allocated_gib",
                    "cuda_reserved_gib",
                    "cuda_max_allocated_gib",
                    "cuda_max_reserved_gib",
                )
            }
            stream.update(
                ctx_diagnostic_snapshot_hash=summary["ctx_diagnostic_snapshot_hash"],
                transcoder_diagnostic_snapshot_hash=summary[
                    "transcoder_diagnostic_snapshot_hash"
                ],
            )
            return summary, stream
        raise TypeError(f"unsupported observation: {type(observation).__name__}")

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
        self._recorder.record_event(
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
            self._recorder.record_wall_clock_duration(**duration_kwargs)

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

    def close_export(self, *, include_events: bool | None = None) -> dict[str, object]:
        self._recorder.close()
        if include_events is None:
            include_events = not bool(
                getattr(self._recorder, "incremental_sink_enabled", False)
            )
        return self._recorder.export(include_events=include_events)

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
