"""Phase-1 trace-batch policy support.

This module owns Phase-1 trace-batch configuration, sizing, and metadata.
The Phase-1 execution body remains in :mod:`circuit_tracer.attribution.nnsight.backend`.
"""

from dataclasses import dataclass
from typing import Literal, cast


_PHASE1_TRACE_BATCH_POLICY_DEFAULT: Literal["legacy"] = "legacy"
_PHASE1_TRACE_BATCH_SIZE_MAX_DEFAULT: int | None = None
_PHASE1_TRACE_BATCH_POLICY_EFFECTIVE_POLICY_BY_POLICY: dict[str, str] = {
    "legacy": "legacy",
    "cap_effective_batches": "cap_effective_batches",
}


@dataclass(frozen=True)
class _Phase1TraceBatchConfig:
    requested_policy: Literal["legacy", "cap_effective_batches"]
    effective_policy: Literal["legacy", "cap_effective_batches"]
    requested_batch_size_max: int | None
    effective_batch_size_max: int | None
    default_policy: Literal["legacy"]
    default_batch_size_max: int | None
    effective_behavior: Literal["requested", "legacy_fallback_missing_batch_size_max"]
    fallback_reason: str | None


@dataclass(frozen=True)
class _Phase1TraceBatchSizing:
    requested_source_batch_size: int
    requested_feature_batch_size: int
    requested_logit_batch_size: int
    requested_phase4_max_feature_batch_size: int
    requested_feature_batch_size_defaulted: bool
    requested_logit_batch_size_defaulted: bool
    effective_source_batch_size: int
    effective_feature_batch_size: int
    effective_logit_batch_size: int
    effective_phase4_max_feature_batch_size: int
    source_batch_size_cap_applied: bool
    feature_batch_size_cap_applied: bool
    logit_batch_size_cap_applied: bool
    phase4_max_feature_batch_size_cap_applied: bool
    cap_applied: bool
    cap_reason: str
    trace_batch_size_legacy: int
    trace_batch_size_effective_pre_planner: int
    trace_batch_size_cap_applied: bool


def _resolve_phase1_trace_batch_policy(
    phase1_trace_batch_policy: str,
) -> Literal["legacy", "cap_effective_batches"]:
    normalized = str(phase1_trace_batch_policy).strip().lower()
    allowed_values = {"legacy", "cap_effective_batches"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            "phase1_trace_batch_policy must be one of: "
            f"{allowed} (got {phase1_trace_batch_policy!r})"
        )
    return cast(Literal["legacy", "cap_effective_batches"], normalized)


def _resolve_phase1_trace_batch_size_max(
    phase1_trace_batch_size_max: int | None,
) -> int | None:
    if phase1_trace_batch_size_max is None:
        return None
    resolved = int(phase1_trace_batch_size_max)
    if resolved <= 0:
        raise ValueError("phase1_trace_batch_size_max must be > 0 when provided")
    return resolved


def _resolve_phase1_trace_batch_config(
    *,
    phase1_trace_batch_policy: str,
    phase1_trace_batch_size_max: int | None,
) -> _Phase1TraceBatchConfig:
    requested_policy = _resolve_phase1_trace_batch_policy(phase1_trace_batch_policy)
    requested_batch_size_max = _resolve_phase1_trace_batch_size_max(phase1_trace_batch_size_max)
    cap_requested = requested_policy == "cap_effective_batches"
    fallback_missing_batch_size_max = cap_requested and requested_batch_size_max is None
    effective_policy = cast(
        Literal["legacy", "cap_effective_batches"],
        (
            _PHASE1_TRACE_BATCH_POLICY_DEFAULT
            if fallback_missing_batch_size_max
            else _PHASE1_TRACE_BATCH_POLICY_EFFECTIVE_POLICY_BY_POLICY[requested_policy]
        ),
    )
    effective_batch_size_max = (
        requested_batch_size_max
        if effective_policy == "cap_effective_batches"
        else _PHASE1_TRACE_BATCH_SIZE_MAX_DEFAULT
    )
    effective_behavior: Literal["requested", "legacy_fallback_missing_batch_size_max"] = (
        "legacy_fallback_missing_batch_size_max" if fallback_missing_batch_size_max else "requested"
    )
    fallback_reason = (
        "cap_effective_batches requested without phase1_trace_batch_size_max; "
        "falling back to legacy execution"
        if fallback_missing_batch_size_max
        else None
    )
    return _Phase1TraceBatchConfig(
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        requested_batch_size_max=requested_batch_size_max,
        effective_batch_size_max=effective_batch_size_max,
        default_policy=_PHASE1_TRACE_BATCH_POLICY_DEFAULT,
        default_batch_size_max=_PHASE1_TRACE_BATCH_SIZE_MAX_DEFAULT,
        effective_behavior=effective_behavior,
        fallback_reason=fallback_reason,
    )


def _build_phase1_trace_batch_metadata(
    phase1_trace_batch_config: _Phase1TraceBatchConfig,
) -> dict[str, object]:
    return {
        "trace_batch_policy_requested": phase1_trace_batch_config.requested_policy,
        "trace_batch_policy": phase1_trace_batch_config.requested_policy,
        "trace_batch_policy_default": phase1_trace_batch_config.default_policy,
        "trace_batch_policy_effective": phase1_trace_batch_config.effective_policy,
        "trace_batch_policy_effective_behavior": phase1_trace_batch_config.effective_behavior,
        "trace_batch_policy_fallback_reason": phase1_trace_batch_config.fallback_reason,
        "trace_batch_policy_reference_execution": bool(
            phase1_trace_batch_config.requested_policy != phase1_trace_batch_config.effective_policy
        ),
        "trace_batch_size_max_requested": phase1_trace_batch_config.requested_batch_size_max,
        "trace_batch_size_max": phase1_trace_batch_config.requested_batch_size_max,
        "trace_batch_size_max_default": phase1_trace_batch_config.default_batch_size_max,
        "trace_batch_size_max_effective": phase1_trace_batch_config.effective_batch_size_max,
        "trace_batch_size_max_reference_execution": bool(
            phase1_trace_batch_config.requested_batch_size_max
            != phase1_trace_batch_config.effective_batch_size_max
        ),
    }


def _resolve_phase1_trace_batch_sizing(
    *,
    batch_size: int,
    feature_batch_size: int | None,
    logit_batch_size: int | None,
    feature_batch_size_max: int | None,
    phase1_trace_batch_config: _Phase1TraceBatchConfig,
) -> _Phase1TraceBatchSizing:
    requested_feature_batch_size = batch_size if feature_batch_size is None else feature_batch_size
    requested_logit_batch_size = batch_size if logit_batch_size is None else logit_batch_size
    requested_phase4_max_feature_batch_size = (
        requested_feature_batch_size if feature_batch_size_max is None else feature_batch_size_max
    )

    cap_limit = (
        int(phase1_trace_batch_config.effective_batch_size_max)
        if phase1_trace_batch_config.effective_policy == "cap_effective_batches"
        and phase1_trace_batch_config.effective_batch_size_max is not None
        else None
    )

    if cap_limit is None:
        effective_source_batch_size = int(batch_size)
    else:
        # Phase-1-only cap decoupling: cap applies only to the source/invoke
        # trace batch size used to drive the Phase-1 forward/cache footprint.
        effective_source_batch_size = min(int(batch_size), cap_limit)

    # Keep downstream phase/requested batch knobs unchanged by the Phase-1 cap.
    effective_feature_batch_size = int(requested_feature_batch_size)
    effective_logit_batch_size = int(requested_logit_batch_size)
    effective_phase4_max_feature_batch_size = int(requested_phase4_max_feature_batch_size)

    source_batch_size_cap_applied = effective_source_batch_size < int(batch_size)
    feature_batch_size_cap_applied = effective_feature_batch_size < int(
        requested_feature_batch_size
    )
    logit_batch_size_cap_applied = effective_logit_batch_size < int(requested_logit_batch_size)
    phase4_max_feature_batch_size_cap_applied = effective_phase4_max_feature_batch_size < int(
        requested_phase4_max_feature_batch_size
    )
    cap_applied = (
        source_batch_size_cap_applied
        or feature_batch_size_cap_applied
        or logit_batch_size_cap_applied
        or phase4_max_feature_batch_size_cap_applied
    )

    if cap_limit is not None:
        cap_reason = (
            "cap_effective_batches_applied"
            if cap_applied
            else "cap_effective_batches_no_reduction_needed"
        )
    elif phase1_trace_batch_config.requested_policy == "cap_effective_batches":
        cap_reason = "cap_effective_batches_fallback_missing_batch_size_max"
    elif phase1_trace_batch_config.requested_batch_size_max is not None:
        cap_reason = "legacy_policy_ignores_phase1_trace_batch_size_max"
    else:
        cap_reason = "legacy_policy_no_cap"

    trace_batch_size_legacy = max(
        int(batch_size),
        int(requested_feature_batch_size),
        int(requested_logit_batch_size),
    )
    trace_batch_size_effective_pre_planner = effective_source_batch_size

    return _Phase1TraceBatchSizing(
        requested_source_batch_size=int(batch_size),
        requested_feature_batch_size=int(requested_feature_batch_size),
        requested_logit_batch_size=int(requested_logit_batch_size),
        requested_phase4_max_feature_batch_size=int(requested_phase4_max_feature_batch_size),
        requested_feature_batch_size_defaulted=(feature_batch_size is None),
        requested_logit_batch_size_defaulted=(logit_batch_size is None),
        effective_source_batch_size=effective_source_batch_size,
        effective_feature_batch_size=effective_feature_batch_size,
        effective_logit_batch_size=effective_logit_batch_size,
        effective_phase4_max_feature_batch_size=effective_phase4_max_feature_batch_size,
        source_batch_size_cap_applied=source_batch_size_cap_applied,
        feature_batch_size_cap_applied=feature_batch_size_cap_applied,
        logit_batch_size_cap_applied=logit_batch_size_cap_applied,
        phase4_max_feature_batch_size_cap_applied=phase4_max_feature_batch_size_cap_applied,
        cap_applied=cap_applied,
        cap_reason=cap_reason,
        trace_batch_size_legacy=trace_batch_size_legacy,
        trace_batch_size_effective_pre_planner=trace_batch_size_effective_pre_planner,
        trace_batch_size_cap_applied=source_batch_size_cap_applied,
    )


def _build_phase1_trace_batch_sizing_metadata(
    sizing: _Phase1TraceBatchSizing,
) -> dict[str, object]:
    return {
        "source_batch_size_requested": sizing.requested_source_batch_size,
        "source_batch_size_effective": sizing.effective_source_batch_size,
        "source_batch_size_cap_applied": sizing.source_batch_size_cap_applied,
        "feature_batch_size_requested": sizing.requested_feature_batch_size,
        "feature_batch_size_defaulted": sizing.requested_feature_batch_size_defaulted,
        "feature_batch_size_effective": sizing.effective_feature_batch_size,
        "feature_batch_size_cap_applied": sizing.feature_batch_size_cap_applied,
        "logit_batch_size_requested": sizing.requested_logit_batch_size,
        "logit_batch_size_defaulted": sizing.requested_logit_batch_size_defaulted,
        "logit_batch_size_effective": sizing.effective_logit_batch_size,
        "logit_batch_size_cap_applied": sizing.logit_batch_size_cap_applied,
        "phase4_feature_batch_size_max_requested": sizing.requested_phase4_max_feature_batch_size,
        "phase4_feature_batch_size_max_effective": sizing.effective_phase4_max_feature_batch_size,
        "phase4_feature_batch_size_max_cap_applied": sizing.phase4_max_feature_batch_size_cap_applied,
        "trace_batch_size_legacy": sizing.trace_batch_size_legacy,
        "trace_batch_size_effective_pre_planner": sizing.trace_batch_size_effective_pre_planner,
        "trace_batch_size_cap_applied": sizing.trace_batch_size_cap_applied,
        "trace_batch_cap_applied": sizing.cap_applied,
        "trace_batch_cap_reason": sizing.cap_reason,
    }
