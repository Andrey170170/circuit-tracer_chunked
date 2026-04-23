from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from numbers import Number
from typing import cast

import torch

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]


TelemetryScalar = str | int | float | bool | None
_ALLOWED_TELEMETRY_SCOPES = {"run", "phase", "batch", "op"}
_DEFAULT_MEMORY_ATTR_KEYS: tuple[str, ...] = (
    "rss_current_gib",
    "proc_rss_anon_gib",
    "proc_rss_file_gib",
    "cgroup_memory_current_gib",
    "cgroup_memory_anon_gib",
    "cgroup_memory_file_gib",
    "cuda_allocated_gib",
    "cuda_reserved_gib",
)


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

    def __init__(self, *, enabled: bool = True, max_events: int = 20000) -> None:
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

        if self.max_events > 0 and len(self._events) >= self.max_events:
            self._dropped_event_count += 1
            return

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


def _format_optional_gib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} GiB"


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def _kib_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**2)


def _read_file_first_line(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()
    except (FileNotFoundError, OSError):
        return None


def _parse_memory_bytes_value(raw_value: str | None) -> int | None:
    if not raw_value:
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"", "max"}:
        return None
    try:
        value = int(normalized)
    except ValueError:
        return None
    return value if value >= 0 else None


def _read_memory_bytes_file(path: str) -> int | None:
    return _parse_memory_bytes_value(_read_file_first_line(path))


def _read_memory_stat_file(path: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                key, value_text = parts
                parsed = _parse_memory_bytes_value(value_text)
                if parsed is not None:
                    stats[key] = parsed
    except (FileNotFoundError, OSError):
        return {}
    return stats


def _get_current_rss_gib_from_proc() -> float | None:
    """Best-effort Linux RSS snapshot using /proc/self/statm."""

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0:
        return None

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.readline().split()
        if len(fields) < 2:
            return None
        resident_pages = int(fields[1])
    except (FileNotFoundError, OSError, ValueError):
        return None

    return (resident_pages * page_size) / (1024**3)


def _get_process_status_memory_gib_from_proc() -> dict[str, float | None]:
    status_fields: dict[str, float | None] = {
        "proc_rss_gib": None,
        "proc_rss_anon_gib": None,
        "proc_rss_file_gib": None,
        "proc_rss_shmem_gib": None,
    }
    field_map = {
        "VmRSS": "proc_rss_gib",
        "RssAnon": "proc_rss_anon_gib",
        "RssFile": "proc_rss_file_gib",
        "RssShmem": "proc_rss_shmem_gib",
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                attr_name = field_map.get(key)
                if attr_name is None:
                    continue
                value_parts = raw_value.strip().split()
                if not value_parts:
                    continue
                try:
                    kib_value = int(value_parts[0])
                except ValueError:
                    continue
                if kib_value < 0:
                    continue
                status_fields[attr_name] = _kib_to_gib(kib_value)
    except (FileNotFoundError, OSError):
        return status_fields

    return status_fields


def _resolve_cgroup_memory_dir() -> str | None:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            cgroup_lines = handle.readlines()
    except (FileNotFoundError, OSError):
        cgroup_lines = []

    unified_path: str | None = None
    legacy_memory_path: str | None = None
    for line in cgroup_lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        if controllers == "":
            unified_path = rel_path
            break
        controller_set = set(filter(None, controllers.split(",")))
        if "memory" in controller_set:
            legacy_memory_path = rel_path

    if unified_path is not None:
        base = "/sys/fs/cgroup"
        candidate = (
            os.path.normpath(os.path.join(base, unified_path.lstrip("/")))
            if unified_path not in {"", "/"}
            else base
        )
        if os.path.isdir(candidate):
            return candidate

    if legacy_memory_path is not None:
        for base in ("/sys/fs/cgroup/memory", "/sys/fs/cgroup"):
            candidate = (
                os.path.normpath(os.path.join(base, legacy_memory_path.lstrip("/")))
                if legacy_memory_path not in {"", "/"}
                else base
            )
            if os.path.isdir(candidate):
                return candidate

    fallback = "/sys/fs/cgroup"
    if os.path.isfile(os.path.join(fallback, "memory.current")):
        return fallback
    return None


def _get_cgroup_memory_snapshot_gib() -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
        "cgroup_memory_current_gib": None,
        "cgroup_memory_peak_gib": None,
        "cgroup_memory_anon_gib": None,
        "cgroup_memory_file_gib": None,
        "cgroup_memory_active_file_gib": None,
        "cgroup_memory_inactive_file_gib": None,
        "cgroup_memory_shmem_gib": None,
        "cgroup_memory_slab_reclaimable_gib": None,
        "cgroup_memory_slab_unreclaimable_gib": None,
    }
    cgroup_dir = _resolve_cgroup_memory_dir()
    if cgroup_dir is None:
        return snapshot

    snapshot["cgroup_memory_current_gib"] = _bytes_to_gib(
        _read_memory_bytes_file(os.path.join(cgroup_dir, "memory.current"))
    )
    snapshot["cgroup_memory_peak_gib"] = _bytes_to_gib(
        _read_memory_bytes_file(os.path.join(cgroup_dir, "memory.peak"))
    )

    memory_stats = _read_memory_stat_file(os.path.join(cgroup_dir, "memory.stat"))
    stat_keys = {
        "anon": "cgroup_memory_anon_gib",
        "file": "cgroup_memory_file_gib",
        "active_file": "cgroup_memory_active_file_gib",
        "inactive_file": "cgroup_memory_inactive_file_gib",
        "shmem": "cgroup_memory_shmem_gib",
        "slab_reclaimable": "cgroup_memory_slab_reclaimable_gib",
        "slab_unreclaimable": "cgroup_memory_slab_unreclaimable_gib",
    }
    for stat_key, attr_name in stat_keys.items():
        snapshot[attr_name] = _bytes_to_gib(memory_stats.get(stat_key))

    return snapshot


def _coerce_finite_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def build_memory_snapshot_attrs(
    snapshot: Mapping[str, object] | None,
    *,
    keys: Sequence[str] | None = None,
    prefix: str = "memory",
) -> dict[str, float | None]:
    selected_keys = tuple(keys) if keys is not None else _DEFAULT_MEMORY_ATTR_KEYS
    attrs: dict[str, float | None] = {}
    for key in selected_keys:
        value = snapshot.get(key) if snapshot is not None else None
        attrs[f"{prefix}_{key}"] = _coerce_finite_float(value)
    return attrs


def build_memory_before_after_attrs(
    *,
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    keys: Sequence[str] | None = None,
    before_prefix: str = "memory_before",
    after_prefix: str = "memory_after",
    delta_prefix: str = "memory_delta",
) -> dict[str, float | None]:
    selected_keys = tuple(keys) if keys is not None else _DEFAULT_MEMORY_ATTR_KEYS
    attrs: dict[str, float | None] = {}
    attrs.update(build_memory_snapshot_attrs(before, keys=selected_keys, prefix=before_prefix))
    attrs.update(build_memory_snapshot_attrs(after, keys=selected_keys, prefix=after_prefix))
    for key in selected_keys:
        before_value = _coerce_finite_float(before.get(key) if before is not None else None)
        after_value = _coerce_finite_float(after.get(key) if after is not None else None)
        attrs[f"{delta_prefix}_{key}"] = (
            (after_value - before_value)
            if before_value is not None and after_value is not None
            else None
        )
    return attrs


def get_memory_snapshot(device: torch.device | None = None) -> dict[str, float | None]:
    rss_current_gib = _get_current_rss_gib_from_proc()
    rss_gib = None
    if resource is not None:
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    snapshot: dict[str, float | None] = {
        "rss_current_gib": rss_current_gib,
        "rss_gib": rss_gib,
        "proc_rss_gib": None,
        "proc_rss_anon_gib": None,
        "proc_rss_file_gib": None,
        "proc_rss_shmem_gib": None,
        "cgroup_memory_current_gib": None,
        "cgroup_memory_peak_gib": None,
        "cgroup_memory_anon_gib": None,
        "cgroup_memory_file_gib": None,
        "cgroup_memory_active_file_gib": None,
        "cgroup_memory_inactive_file_gib": None,
        "cgroup_memory_shmem_gib": None,
        "cgroup_memory_slab_reclaimable_gib": None,
        "cgroup_memory_slab_unreclaimable_gib": None,
        "cuda_allocated_gib": None,
        "cuda_reserved_gib": None,
        "cuda_max_allocated_gib": None,
        "cuda_max_reserved_gib": None,
    }
    snapshot.update(_get_process_status_memory_gib_from_proc())
    snapshot.update(_get_cgroup_memory_snapshot_gib())

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                "cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "cuda_max_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
        )

    return snapshot


def format_memory_snapshot(
    device: torch.device | None = None, extra: Mapping[str, object] | None = None
) -> str:
    snapshot = get_memory_snapshot(device)
    parts = [
        f"rss={_format_optional_gib(snapshot['rss_gib'])}",
        f"rss_current={_format_optional_gib(snapshot['rss_current_gib'])}",
        f"proc_anon={_format_optional_gib(snapshot['proc_rss_anon_gib'])}",
        f"proc_file={_format_optional_gib(snapshot['proc_rss_file_gib'])}",
        f"cg_current={_format_optional_gib(snapshot['cgroup_memory_current_gib'])}",
        f"cg_peak={_format_optional_gib(snapshot['cgroup_memory_peak_gib'])}",
        f"cg_anon={_format_optional_gib(snapshot['cgroup_memory_anon_gib'])}",
        f"cg_file={_format_optional_gib(snapshot['cgroup_memory_file_gib'])}",
        f"cuda_alloc={_format_optional_gib(snapshot['cuda_allocated_gib'])}",
        f"cuda_reserved={_format_optional_gib(snapshot['cuda_reserved_gib'])}",
        f"cuda_peak_alloc={_format_optional_gib(snapshot['cuda_max_allocated_gib'])}",
        f"cuda_peak_reserved={_format_optional_gib(snapshot['cuda_max_reserved_gib'])}",
    ]
    if extra:
        parts.extend(f"{key}={value}" for key, value in extra.items())
    return ", ".join(parts)


def flatten_numeric_metrics(
    metrics: Mapping[str, object], prefix: str = ""
) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_numeric_metrics(cast(Mapping[str, object], value), full_key))
        elif isinstance(value, Number) and not isinstance(value, bool):
            flat[full_key] = cast(float | int, value)
    return flat


def diff_numeric_metrics(
    before: Mapping[str, object] | None, after: Mapping[str, object]
) -> dict[str, float | int]:
    after_flat = flatten_numeric_metrics(after)
    if before is None:
        return after_flat

    before_flat = flatten_numeric_metrics(before)
    diff: dict[str, float | int] = {}
    for key, value in after_flat.items():
        baseline = before_flat.get(key, 0)
        delta = value - baseline
        if delta:
            diff[key] = delta
    return diff


def format_numeric_metrics(metrics: Mapping[str, object], limit: int | None = None) -> str:
    flat = flatten_numeric_metrics(metrics)
    items = list(flat.items())
    if limit is not None:
        items = items[:limit]
    return ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in items
    )
