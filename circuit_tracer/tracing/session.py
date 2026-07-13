"""Owned full-sequence/window tracing sessions."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from circuit_tracer.transcoder.provider import (
    get_transcoder_capabilities,
    provider_fingerprint,
)

from .planning import resolve_trace_request
from .plan import DecoderCachePolicy
from .problem import PrefixViewTarget
from .request import TraceRequest
from .result import TraceResult
from .runner import run_trace

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


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


def _raise_cleanup_failures(failures: Sequence[BaseException]) -> None:
    if not failures:
        return
    if all(isinstance(error, Exception) for error in failures):
        raise ExceptionGroup(
            "Trace session cleanup failed",
            [error for error in failures if isinstance(error, Exception)],
        )
    raise BaseExceptionGroup("Trace session cleanup failed", list(failures))


def _attempt_cleanup(
    label: str,
    callback: Callable[[], object],
    *,
    primary_error: BaseException | None,
    failures: list[BaseException],
) -> None:
    try:
        callback()
    except BaseException as cleanup_error:
        failures.append(cleanup_error)
        if primary_error is not None:
            try:
                primary_error.add_note(
                    f"trace session {label} cleanup also failed: {cleanup_error!r}"
                )
            except BaseException:
                pass


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

        plan = resolve_trace_request(selected)
        cache, fingerprint = self._decoder_cache.acquire()
        try:
            return run_trace(
                selected.problem,
                plan,
                forward_overrides=ForwardOverrides(
                    decoder_chunk_cache=cache,
                    decoder_cache_fingerprint=fingerprint,
                ),
            )
        except BaseException as primary_error:
            self._cleanup_decoder_cache_after_failure(primary_error)
            raise

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
        window_problem = replace(
            selected.problem,
            output_position=target_position - 1,
            prefix_view=PrefixViewTarget(
                mode=("full_sequence_target_position" if reuse else "independent_prefix"),
                target_position=target_position,
            ),
        )
        window_request = replace(selected, problem=window_problem)
        problem, overrides = self._delegate.prepare_target_position(
            target_position,
            window_request,
        )
        plan = resolve_trace_request(replace(window_request, problem=problem))
        try:
            return run_trace(problem, plan, forward_overrides=overrides)
        except BaseException as primary_error:
            self._cleanup_decoder_cache_after_failure(primary_error)
            raise

    def reset(self) -> None:
        self._ensure_open()
        failures: list[BaseException] = []
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            self._reuse = None
            _attempt_cleanup(
                "forward delegate",
                delegate.close,
                primary_error=None,
                failures=failures,
            )
        _attempt_cleanup(
            "decoder cache",
            self._decoder_cache.reset,
            primary_error=None,
            failures=failures,
        )
        _raise_cleanup_failures(failures)

    def close(self) -> None:
        self._close(primary_error=None)

    def _close(self, primary_error: BaseException | None) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            _attempt_cleanup(
                "forward delegate",
                delegate.close,
                primary_error=primary_error,
                failures=failures,
            )
        _attempt_cleanup(
            "decoder cache",
            self._decoder_cache.close,
            primary_error=primary_error,
            failures=failures,
        )
        self._reuse = None
        self._closed = True
        if primary_error is None:
            _raise_cleanup_failures(failures)

    def _cleanup_decoder_cache_after_failure(self, primary_error: BaseException) -> None:
        failures: list[BaseException] = []
        _attempt_cleanup(
            "decoder cache reset",
            self._decoder_cache.reset,
            primary_error=primary_error,
            failures=failures,
        )

    def __enter__(self) -> "TraceSession":
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: Any,
    ) -> None:
        self._close(primary_error=exc)
