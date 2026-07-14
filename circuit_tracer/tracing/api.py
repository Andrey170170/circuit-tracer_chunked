"""Canonical public tracing API."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Event

from circuit_tracer.governor.contracts import ProviderProfile, ResourceEnvelope

from .planning import resolve_trace_request
from .request import TraceRequest
from .result import TraceResult, TraceStatus
from .runner import run_trace
from .session import SessionWindow, TraceSession, _SessionDecoderCache


def trace_one(
    request: TraceRequest,
    *,
    resources: ResourceEnvelope | None = None,
    provider_profile: ProviderProfile | None = None,
) -> TraceResult:
    plan = resolve_trace_request(request, resources=resources, provider_profile=provider_profile)
    cache_policy = plan.execution.session.decoder_cache
    if not cache_policy.enabled:
        return run_trace(request.problem, plan)

    from circuit_tracer.attribution.nnsight.forward_session import ForwardOverrides

    owner = _SessionDecoderCache(request.problem.model, cache_policy)
    cache, fingerprint = owner.acquire()
    try:
        result = run_trace(
            request.problem,
            plan,
            forward_overrides=ForwardOverrides(
                decoder_chunk_cache=cache,
                decoder_cache_fingerprint=fingerprint,
            ),
        )
    except BaseException as primary_error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            primary_error.add_note(f"one-shot decoder cache cleanup also failed: {cleanup_error!r}")
        raise
    owner.close()
    return result


def trace_batch(
    requests: Sequence[TraceRequest],
    *,
    failure: str = "raise",
    cancellation: Event | None = None,
    resources: ResourceEnvelope | None = None,
    provider_profile: ProviderProfile | None = None,
) -> list[TraceResult]:
    if (resources is None) != (provider_profile is None):
        raise ValueError("resources and provider_profile must be supplied together")
    if failure not in {"raise", "return"}:
        raise ValueError("failure must be 'raise' or 'return'")
    results: list[TraceResult] = []
    for request in requests:
        if cancellation is not None and cancellation.is_set():
            if failure == "raise":
                raise RuntimeError("trace batch cancelled")
            results.append(
                _terminal_result(
                    request,
                    TraceStatus.CANCELLED,
                    "cancelled",
                    resources=resources,
                    provider_profile=provider_profile,
                )
            )
            continue
        try:
            results.append(
                trace_one(
                    request,
                    resources=resources,
                    provider_profile=provider_profile,
                )
            )
        except Exception as error:
            if failure == "raise":
                raise
            results.append(
                _terminal_result(
                    request,
                    TraceStatus.FAILED,
                    error,
                    resources=resources,
                    provider_profile=provider_profile,
                )
            )
    return results


def _terminal_result(
    request: TraceRequest,
    status: TraceStatus,
    error: object,
    *,
    resources: ResourceEnvelope | None = None,
    provider_profile: ProviderProfile | None = None,
) -> TraceResult:
    plan = resolve_trace_request(request, resources=resources, provider_profile=provider_profile)
    return _terminal_result_from_plan(plan, status, error)


def _terminal_result_from_plan(plan, status: TraceStatus, error: object) -> TraceResult:
    return TraceResult(
        output=None,
        semantic_fingerprint=plan.semantic_fingerprint,
        requested_execution_fingerprint=plan.requested_execution_fingerprint,
        status=status,
        telemetry_summary={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "requested_execution_fingerprint": plan.requested_execution_fingerprint,
            "effective_execution_fingerprint": None,
            "execution_fingerprint": plan.requested_execution_fingerprint,
        },
        admission_report=plan.admission_report,
    )


def open_session(
    request: TraceRequest,
    *,
    window: SessionWindow | None = None,
    resources: ResourceEnvelope | None = None,
    provider_profile: ProviderProfile | None = None,
) -> TraceSession:
    return TraceSession(
        request,
        window,
        resources=resources,
        provider_profile=provider_profile,
    )
