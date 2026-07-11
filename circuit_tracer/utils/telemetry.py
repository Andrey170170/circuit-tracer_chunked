"""Compatibility facade for the observability subsystem.

New code should import from :mod:`circuit_tracer.observability` directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch

from circuit_tracer.observability import resources as _resources
from circuit_tracer.observability.recorder import (  # noqa: F401 - compatibility re-exports
    TelemetryRecorder,
    TelemetryScalar,
    _NullTimer,
    _TelemetryTimer,
    _truncate_text,
    sanitize_attrs,
    sanitize_scalar_attr,
)
from circuit_tracer.observability.resources import (  # noqa: F401 - compatibility re-exports
    _DEFAULT_MEMORY_ATTR_KEYS,
    _bytes_to_gib,
    _coerce_finite_float,
    _format_optional_gib,
    _get_cgroup_memory_snapshot_gib,
    _get_current_rss_gib_from_proc,
    _get_process_status_memory_gib_from_proc,
    _kib_to_gib,
    _parse_memory_bytes_value,
    _read_file_first_line,
    _read_memory_bytes_file,
    _read_memory_stat_file,
    _resolve_cgroup_memory_dir,
    build_memory_before_after_attrs,
    build_memory_snapshot_attrs,
    diff_numeric_metrics,
    flatten_numeric_metrics,
    format_numeric_metrics,
)

__all__ = [
    "TelemetryRecorder",
    "TelemetryScalar",
    "build_memory_before_after_attrs",
    "build_memory_snapshot_attrs",
    "diff_numeric_metrics",
    "flatten_numeric_metrics",
    "format_memory_snapshot",
    "format_numeric_metrics",
    "get_memory_snapshot",
    "sanitize_attrs",
    "sanitize_scalar_attr",
]


_T = TypeVar("_T")
_DEFAULT_CGROUP_MEMORY_DIR_RESOLVER = _resolve_cgroup_memory_dir


def _with_compat_resolver(callback: Callable[[], _T]) -> _T:
    if _resolve_cgroup_memory_dir is _DEFAULT_CGROUP_MEMORY_DIR_RESOLVER:
        return callback()

    resolver = _resources._resolve_cgroup_memory_dir
    _resources._resolve_cgroup_memory_dir = _resolve_cgroup_memory_dir
    try:
        return callback()
    finally:
        _resources._resolve_cgroup_memory_dir = resolver


def get_memory_snapshot(device: torch.device | None = None) -> dict[str, float | None]:
    """Return a resource snapshot while preserving facade monkeypatch behavior."""

    return _with_compat_resolver(lambda: _resources.get_memory_snapshot(device))


def format_memory_snapshot(
    device: torch.device | None = None, extra: Mapping[str, object] | None = None
) -> str:
    return _with_compat_resolver(lambda: _resources.format_memory_snapshot(device, extra))
