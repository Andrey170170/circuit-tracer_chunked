"""Owned full-sequence/window tracing sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .planning import resolve_trace_request
from .request import TraceRequest
from .result import TraceResult, TraceStatus
from .runner import run_trace


@dataclass(frozen=True)
class SessionWindow:
    """Full-sequence session capacity and deterministic prefix-reuse policy."""

    max_prefix_len: int | None = None

    def __post_init__(self) -> None:
        if self.max_prefix_len is not None and self.max_prefix_len <= 0:
            raise ValueError("max_prefix_len must be positive")


class TraceSession:
    def __init__(self, request: TraceRequest, window: SessionWindow | None = None) -> None:
        self.request = request
        self.window = window or SessionWindow()
        self._delegate: Any = None
        self._reuse: bool | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trace session is closed")

    def trace(self, request: TraceRequest | None = None) -> TraceResult:
        self._ensure_open()
        selected = request or self.request
        return run_trace(selected.problem, resolve_trace_request(selected))

    def trace_window(
        self,
        target_position: int,
        *,
        reuse: bool,
        request: TraceRequest | None = None,
    ) -> TraceResult:
        self._ensure_open()
        selected = request or self.request
        if selected.problem.model is not self.request.problem.model:
            raise ValueError("a session cannot change models")
        if self._delegate is None:
            if getattr(selected.problem.model, "backend", None) != "nnsight":
                raise ValueError("window tracing requires the NNSight backend")
            from circuit_tracer.attribution.nnsight.forward_session import ForwardTraceSession

            self._delegate = ForwardTraceSession(
                model=self.request.problem.model,
                full_token_ids=self.request.problem.prompt,
                window_max_prefix_len=self.window.max_prefix_len,
                reuse_phase0_window_state=reuse,
                reuse_target_logits=reuse,
            )
            self._reuse = reuse
        elif self._reuse != reuse:
            raise ValueError("reuse cannot change without reset()")
        window_problem = replace(selected.problem, output_position=target_position - 1)
        window_request = replace(selected, problem=window_problem)
        plan = resolve_trace_request(window_request)
        output = self._delegate.trace_target_position(target_position, window_request, plan)
        return TraceResult(
            output=output,
            semantic_fingerprint=plan.semantic_fingerprint,
            execution_fingerprint=plan.execution_fingerprint,
            status=TraceStatus.SUCCEEDED,
            telemetry_summary=getattr(output, "telemetry_summary", {}),
            admission_report=plan.admission_report,
        )

    def reset(self) -> None:
        self._ensure_open()
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            self._reuse = None
            delegate.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._delegate is not None:
                delegate, self._delegate = self._delegate, None
                delegate.close()
        finally:
            self._reuse = None
            self._closed = True

    def __enter__(self) -> "TraceSession":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

