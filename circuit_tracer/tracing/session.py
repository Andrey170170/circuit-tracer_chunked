"""Owned full-sequence/window tracing sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    provider_fingerprint,
)

from .planning import resolve_trace_request
from .plan import DecoderCachePolicy
from .request import TraceRequest
from .result import TraceResult, TraceStatus
from .runner import run_trace


@dataclass(frozen=True)
class SessionWindow:
    """Independent prefix-window and decoder-chunk reuse policies."""

    max_prefix_len: int | None = None

    def __post_init__(self) -> None:
        if self.max_prefix_len is not None and self.max_prefix_len <= 0:
            raise ValueError("max_prefix_len must be positive")


class _SessionDecoderCache:
    """Own one bounded provider cache across traces and clear it on failure."""

    def __init__(self, model: Any, policy: DecoderCachePolicy) -> None:
        self._provider = getattr(model, "transcoders", None)
        self._policy = policy
        self._cache: Any = None
        self._fingerprint: object | None = None
        self._closed = False
        if policy.enabled:
            if self._provider is None:
                raise ValueError("decoder cache reuse requires a transcoder provider")
            capabilities = get_transcoder_capabilities(self._provider)
            if not capabilities.supports_decoder_chunk_cache:
                raise ValueError("transcoder provider does not support decoder chunk caching")
            self._fingerprint = provider_fingerprint(self._provider)

    def acquire(self) -> tuple[Any | None, object | None]:
        if self._closed:
            raise RuntimeError("decoder cache owner is closed")
        if not self._policy.enabled:
            return None, None
        if self._cache is None:
            create = self._provider.create_decoder_block_cache
            self._cache = create(
                max_bytes=self._policy.max_bytes,
                fingerprint=self._fingerprint,
            )
        return self._cache, self._fingerprint

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("decoder cache owner is closed")
        self._clear()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._clear()
        finally:
            self._closed = True

    def _clear(self) -> None:
        if self._cache is not None:
            cache, self._cache = self._cache, None
            self._provider.clear_decoder_block_cache(cache)


class TraceSession:
    def __init__(self, request: TraceRequest, window: SessionWindow | None = None) -> None:
        self.request = request
        self.window = window or SessionWindow()
        self._delegate: Any = None
        self._reuse: bool | None = None
        self._closed = False
        self._decoder_cache = _SessionDecoderCache(
            request.problem.model,
            request.execution.session.decoder_cache,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trace session is closed")

    def trace(self, request: TraceRequest | None = None) -> TraceResult:
        self._ensure_open()
        selected = request or self.request
        if selected.problem.model is not self.request.problem.model:
            raise ValueError("a session cannot change models")
        if not self.request.execution.session.decoder_cache.enabled:
            return run_trace(selected.problem, resolve_trace_request(selected))
        from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides
        from circuit_tracer.attribution.nnsight.backend import run_nnsight_trace

        plan = resolve_trace_request(selected)
        cache, fingerprint = self._decoder_cache.acquire()
        try:
            output = run_nnsight_trace(
                selected.problem,
                plan,
                forward_overrides=ForwardOverrides(
                    decoder_chunk_cache=cache,
                    decoder_cache_fingerprint=fingerprint,
                ),
            )
        except BaseException:
            self._decoder_cache.reset()
            raise
        return TraceResult(
            output=output,
            semantic_fingerprint=plan.semantic_fingerprint,
            execution_fingerprint=plan.execution_fingerprint,
            status=TraceStatus.SUCCEEDED,
            telemetry_summary=getattr(output, "telemetry_summary", {}),
            admission_report=plan.admission_report,
        )

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
                decoder_cache_owner=self._decoder_cache,
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
        self._decoder_cache.reset()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._delegate is not None:
                delegate, self._delegate = self._delegate, None
                delegate.close()
        finally:
            try:
                self._decoder_cache.close()
            finally:
                self._reuse = None
                self._closed = True

    def __enter__(self) -> "TraceSession":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
