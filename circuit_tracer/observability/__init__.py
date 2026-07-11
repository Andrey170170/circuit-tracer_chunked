"""Observability foundations for circuit tracing."""

from circuit_tracer.observability.recorder import (
    TelemetryRecorder,
    TelemetryScalar,
    sanitize_attrs,
    sanitize_scalar_attr,
)
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.observability.resources import (
    build_memory_before_after_attrs,
    build_memory_snapshot_attrs,
    diff_numeric_metrics,
    flatten_numeric_metrics,
    format_memory_snapshot,
    format_numeric_metrics,
    get_memory_snapshot,
)

__all__ = [
    "TelemetryRecorder",
    "TelemetryObserver",
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
