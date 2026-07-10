from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from numbers import Number
from typing import TextIO, cast

import torch

TelemetryScalar = str | int | float | bool | None
_ALLOWED_TELEMETRY_SCOPES = {"run", "phase", "batch", "op"}


def _truncate_text(value: str, *, max_length: int = 256) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return f"{value[: max_length - 3]}..."


def sanitize_scalar_attr(value: object) -> TelemetryScalar:
    """Convert telemetry attribute values into scalar JSON-safe values."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return sanitize_scalar_attr(value.item())
        return _truncate_text(
            f"tensor(shape={tuple(value.shape)},dtype={value.dtype},device={value.device})"
        )

    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)

    if isinstance(value, Number):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
        return str(numeric_value)

    if isinstance(value, Mapping):
        preview_items = []
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 6:
                preview_items.append("...")
                break
            preview_items.append(f"{key}={sanitize_scalar_attr(item)}")
        return _truncate_text("{" + ", ".join(preview_items) + "}")

    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        preview = ", ".join(str(sanitize_scalar_attr(item)) for item in sequence[:6])
        if len(sequence) > 6:
            preview = f"{preview}, ..."
        return _truncate_text("[" + preview + "]")

    return _truncate_text(str(value))


def sanitize_attrs(attrs: Mapping[str, object] | None) -> dict[str, TelemetryScalar]:
    if not attrs:
        return {}
    sanitized: dict[str, TelemetryScalar] = {}
    for key, value in attrs.items():
        sanitized[str(key)] = sanitize_scalar_attr(value)
    return sanitized


class _NullTimer(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class _TelemetryTimer(AbstractContextManager[None]):
    recorder: "TelemetryRecorder"
    scope: str
    name: str
    phase: str | None
    step_index: int | None
    batch_index: int | None
    attrs: Mapping[str, object] | None
    _start_time: float | None = None

    def __enter__(self) -> None:
        self._start_time = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._start_time is None:
            return False
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        timer_attrs = dict(self.attrs or {})
        if exc_type is not None:
            timer_attrs["error_type"] = exc_type.__name__
        self.recorder.record_event(
            scope=self.scope,
            name=self.name,
            phase=self.phase,
            step_index=self.step_index,
            batch_index=self.batch_index,
            elapsed_ms=elapsed_ms,
            attrs=timer_attrs,
        )
        return False


class TelemetryRecorder:
    """Structured scalar telemetry recorder for attribution hot paths."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_events: int = 20000,
        jsonl_path: str | os.PathLike[str] | None = None,
        static_context: Mapping[str, object] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_events = max(0, int(max_events))
        self._start_time = time.perf_counter()
        self._events: list[dict[str, object]] = []
        self._event_count = 0
        self._dropped_event_count = 0
        self._counts_by_scope: dict[str, int] = defaultdict(int)
        self._counts_by_phase: dict[str, int] = defaultdict(int)
        self._counts_by_name: dict[str, int] = defaultdict(int)
        self._elapsed_ms_total = 0.0
        self._elapsed_ms_by_scope: dict[str, float] = defaultdict(float)
        self._elapsed_ms_by_phase: dict[str, float] = defaultdict(float)
        self._elapsed_ms_by_name: dict[str, float] = defaultdict(float)
        self._wall_clock_elapsed_ms_total = 0.0
        self._wall_clock_elapsed_ms_by_scope: dict[str, float] = defaultdict(float)
        self._wall_clock_elapsed_ms_by_phase: dict[str, float] = defaultdict(float)
        self._wall_clock_elapsed_ms_by_name: dict[str, float] = defaultdict(float)
        self._wall_clock_count = 0
        self._jsonl_path = os.fspath(jsonl_path) if jsonl_path is not None else None
        self._static_context = sanitize_attrs(static_context)
        self._sink: TextIO | None = None
        self._sink_status = "disabled" if self._jsonl_path is None else "pending"
        self._sink_event_count = 0
        self._sink_error_count = 0
        self._sink_last_error: str | None = None
        if self._jsonl_path is not None:
            self._open_sink()

    def _record_sink_error(self, exc: BaseException) -> None:
        self._sink_error_count += 1
        self._sink_last_error = _truncate_text(f"{type(exc).__name__}: {exc}")
        self._sink_status = "error"

    def _open_sink(self) -> None:
        if self._jsonl_path is None or self._sink is not None:
            return
        try:
            self._sink = open(self._jsonl_path, "a", encoding="utf-8", buffering=1)
            self._sink_status = "open"
        except Exception as exc:  # pragma: no cover - filesystem dependent
            self._record_sink_error(exc)

    def _stream_event(self, event: Mapping[str, object]) -> None:
        if self._sink is None:
            return
        record = {
            **self._static_context,
            "event_index": self._event_count - 1,
            "sequence": self._event_count,
            **event,
        }
        try:
            self._sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._sink.flush()
            self._sink_event_count += 1
        except Exception as exc:  # pragma: no cover - filesystem dependent
            self._record_sink_error(exc)

    def close(self) -> None:
        if self._sink is None:
            return
        sink = self._sink
        self._sink = None
        try:
            sink.flush()
            sink.close()
            if self._sink_status != "error":
                self._sink_status = "closed"
        except Exception as exc:  # pragma: no cover - filesystem dependent
            self._record_sink_error(exc)

    def _normalize_scope(self, scope: str) -> str:
        if scope in _ALLOWED_TELEMETRY_SCOPES:
            return scope
        return "op"

    @staticmethod
    def _normalize_elapsed_ms(elapsed_ms: float | int | None) -> float | None:
        if elapsed_ms is None:
            return None
        value = float(elapsed_ms)
        if not math.isfinite(value):
            return None
        return value

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
    ) -> None:
        if not self.enabled:
            return

        safe_scope = self._normalize_scope(scope)
        safe_name = str(name)
        safe_phase = None if phase is None else str(phase)
        safe_elapsed_ms = self._normalize_elapsed_ms(elapsed_ms)
        safe_attrs = sanitize_attrs(attrs)
        if safe_scope != scope:
            safe_attrs.setdefault("original_scope", sanitize_scalar_attr(scope))

        self._event_count += 1
        self._counts_by_scope[safe_scope] += 1
        self._counts_by_name[safe_name] += 1
        if safe_phase is not None:
            self._counts_by_phase[safe_phase] += 1

        if safe_elapsed_ms is not None:
            self._elapsed_ms_total += safe_elapsed_ms
            self._elapsed_ms_by_scope[safe_scope] += safe_elapsed_ms
            self._elapsed_ms_by_name[safe_name] += safe_elapsed_ms
            if safe_phase is not None:
                self._elapsed_ms_by_phase[safe_phase] += safe_elapsed_ms

        event: dict[str, object] = {
            "t_rel_ms": (time.perf_counter() - self._start_time) * 1000.0,
            "scope": safe_scope,
            "name": safe_name,
        }
        if safe_phase is not None:
            event["phase"] = safe_phase
        if step_index is not None:
            event["step_index"] = int(step_index)
        if batch_index is not None:
            event["batch_index"] = int(batch_index)
        if safe_elapsed_ms is not None:
            event["elapsed_ms"] = safe_elapsed_ms
        if safe_attrs:
            event["attrs"] = safe_attrs

        self._stream_event(event)
        if self.max_events > 0 and len(self._events) >= self.max_events:
            self._dropped_event_count += 1
            return
        self._events.append(event)

    def record_wall_clock_duration(
        self,
        *,
        scope: str,
        name: str,
        elapsed_ms: float | int | None,
        phase: str | None = None,
    ) -> None:
        """Record explicit wall-clock timing independent from event aggregates."""

        if not self.enabled:
            return

        safe_elapsed_ms = self._normalize_elapsed_ms(elapsed_ms)
        if safe_elapsed_ms is None:
            return

        safe_scope = self._normalize_scope(scope)
        safe_name = str(name)
        safe_phase = None if phase is None else str(phase)

        self._wall_clock_count += 1
        self._wall_clock_elapsed_ms_total += safe_elapsed_ms
        self._wall_clock_elapsed_ms_by_scope[safe_scope] += safe_elapsed_ms
        self._wall_clock_elapsed_ms_by_name[safe_name] += safe_elapsed_ms
        if safe_phase is not None:
            self._wall_clock_elapsed_ms_by_phase[safe_phase] += safe_elapsed_ms

    def timer(
        self,
        *,
        scope: str,
        name: str,
        phase: str | None = None,
        step_index: int | None = None,
        batch_index: int | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[None]:
        if not self.enabled:
            return _NullTimer()
        return _TelemetryTimer(
            recorder=self,
            scope=scope,
            name=name,
            phase=phase,
            step_index=step_index,
            batch_index=batch_index,
            attrs=attrs,
        )

    @staticmethod
    def _top_items_by_value(
        values: Mapping[str, int | float], *, limit: int = 25
    ) -> dict[str, int | float]:
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        return dict(ordered[:limit])

    def build_summary(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "max_events": int(self.max_events),
            "event_count": int(self._event_count),
            "stored_event_count": int(len(self._events)),
            "dropped_event_count": int(self._dropped_event_count),
            "total_elapsed_ms": float(self._elapsed_ms_total),
            "wall_clock_elapsed_ms_total": float(self._wall_clock_elapsed_ms_total),
            "wall_clock_interval_count": int(self._wall_clock_count),
            "sink_enabled": self._jsonl_path is not None,
            "sink_path": self._jsonl_path,
            "sink_status": self._sink_status,
            "sink_event_count": int(self._sink_event_count),
            "sink_error_count": int(self._sink_error_count),
            "sink_last_error": self._sink_last_error,
            "counts_by_scope": dict(sorted(self._counts_by_scope.items())),
            "counts_by_phase": dict(sorted(self._counts_by_phase.items())),
            "elapsed_ms_by_scope": dict(sorted(self._elapsed_ms_by_scope.items())),
            "elapsed_ms_by_phase": dict(sorted(self._elapsed_ms_by_phase.items())),
            "elapsed_ms_by_scope_aggregate": dict(sorted(self._elapsed_ms_by_scope.items())),
            "elapsed_ms_by_phase_aggregate": dict(sorted(self._elapsed_ms_by_phase.items())),
            "wall_clock_elapsed_ms_by_scope": dict(
                sorted(self._wall_clock_elapsed_ms_by_scope.items())
            ),
            "wall_clock_elapsed_ms_by_phase": dict(
                sorted(self._wall_clock_elapsed_ms_by_phase.items())
            ),
            "counts_by_name_top": self._top_items_by_value(self._counts_by_name),
            "elapsed_ms_by_name_top": self._top_items_by_value(self._elapsed_ms_by_name),
            "wall_clock_elapsed_ms_by_name_top": self._top_items_by_value(
                self._wall_clock_elapsed_ms_by_name
            ),
        }

    def export(
        self,
        *,
        include_events: bool = True,
        max_events: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"summary": self.build_summary()}
        if not include_events:
            return payload

        events = self._events
        truncated_count = 0
        if max_events is not None:
            max_events = max(0, int(max_events))
            if len(events) > max_events:
                truncated_count = len(events) - max_events
                events = events[:max_events]
        payload["events"] = list(events)
        if truncated_count:
            summary = cast(dict[str, object], payload["summary"])
            summary["export_truncated_event_count"] = truncated_count
        return payload

