"""Canonical public tracing API."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Event

from .planning import resolve_trace_request
from .request import TraceRequest
from .result import TraceResult, TraceStatus
from .runner import run_trace
from .session import SessionWindow, TraceSession


def trace_one(request: TraceRequest) -> TraceResult:
    plan = resolve_trace_request(request)
    return run_trace(request.problem, plan)


def trace_batch(
    requests: Sequence[TraceRequest],
    *,
    failure: str = "raise",
    cancellation: Event | None = None,
) -> list[TraceResult]:
    if failure not in {"raise", "return"}:
        raise ValueError("failure must be 'raise' or 'return'")
    results: list[TraceResult] = []
    for request in requests:
        if cancellation is not None and cancellation.is_set():
            if failure == "raise":
                raise RuntimeError("trace batch cancelled")
            results.append(_terminal_result(request, TraceStatus.CANCELLED, "cancelled"))
            continue
        try:
            results.append(trace_one(request))
        except Exception as error:
            if failure == "raise":
                raise
            results.append(_terminal_result(request, TraceStatus.FAILED, error))
    return results


def _terminal_result(request: TraceRequest, status: TraceStatus, error: object) -> TraceResult:
    plan = resolve_trace_request(request)
    return TraceResult(
        output=None,
        semantic_fingerprint=plan.semantic_fingerprint,
        execution_fingerprint=plan.execution_fingerprint,
        status=status,
        telemetry_summary={"error_type": type(error).__name__, "error_message": str(error)},
        admission_report=plan.admission_report,
    )


def open_session(
    request: TraceRequest, *, window: SessionWindow | None = None
) -> TraceSession:
    resolve_trace_request(request)
    return TraceSession(request, window)

