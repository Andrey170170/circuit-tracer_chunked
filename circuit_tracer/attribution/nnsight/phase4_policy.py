"""Phase-4 attribution scheduling and policy support.

This module owns policy/configuration, frontier planning, rank selection, scheduler
telemetry summaries, exact-encoder residency setup, and feature-batch preflight.
The Phase-4 execution body remains in :mod:`circuit_tracer.attribution.nnsight.backend`.
"""

import math
import time
from dataclasses import dataclass
from typing import Literal, cast

import torch

from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.graph import compute_partial_feature_influences
from circuit_tracer.observability import CudaMemoryProbe, probe_cuda_memory
from circuit_tracer.observability.events import TraceEvent, TraceObserver
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder.provider import require_exact_chunked_provider

from circuit_tracer.attribution.nnsight.numerics import _exact_trace_internal_dtype_name
from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata,
    _resolve_prefix_view_trace_input_ids,
)
from circuit_tracer.attribution.nnsight.telemetry import _safe_float
from circuit_tracer.attribution.nnsight.replay import (
    _compute_row_denominator_scaled_l1,
    _hash_index_tensor,
)

_PHASE4_RANK_SELECTION_NEAR_CUTOFF_EPSILON = 1e-6


_PHASE4_REFRESH_MEMORY_ATTR_KEYS: tuple[str, ...] = (
    "rss_current_gib",
    "proc_rss_anon_gib",
    "proc_rss_file_gib",
    "cgroup_memory_current_gib",
    "cgroup_memory_anon_gib",
    "cgroup_memory_file_gib",
    "cuda_allocated_gib",
    "cuda_reserved_gib",
)


_PHASE4_SCHEDULER_MODE_ALIAS: dict[str, str] = {
    "legacy": "locality",
}


_PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS: dict[str, str] = {
    "compact": "summary",
    "full": "debug",
}


_PHASE4_SCHEDULER_VERSION_BY_MODE: dict[str, str] = {
    "locality": "locality_v1",
    "planner_v1": "planner_v1",
    "planner_v2": "planner_v2",
}


_PHASE4_SCHEDULER_POLICY_BY_MODE: dict[str, str] = {
    "locality": "fixed_frontier_locality",
    "planner_v1": "membership_preserving_locality",
    "planner_v2": "bounded_membership_selection",
}


_PHASE4_SCHEDULER_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "locality": "locality",
    "planner_v1": "planner_v1",
    "planner_v2": "planner_v2",
}


_PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE: dict[str, str] = {
    "off": "off_v1",
    "v1": "v1",
}


_PHASE4_REFRESH_OPTIMIZATION_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "off": "off",
    "v1": "v1",
}


_PHASE4_ROW_EXECUTOR_VERSION_BY_MODE: dict[str, str] = {
    "batched": "batched_v1",
    "streaming_v1": "streaming_v1",
}


_PHASE4_ROW_EXECUTOR_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "batched": "batched",
    "streaming_v1": "streaming_v1",
}


_PHASE4_ROW_REDUCTION_VERSION_BY_MODE: dict[str, str] = {
    "off": "off_v1",
    "gpu_v1": "gpu_v1_staged",
}


_PHASE4_REFRESH_POLICY_DEFAULT: Literal["standard"] = "standard"


_PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT = 1


_PHASE4_REFRESH_POLICY_EFFECTIVE_POLICY_BY_POLICY: dict[str, str] = {
    "standard": "standard",
    "deferred_v1": "deferred_v1",
}


_PHASE4_RANKER_DEFAULT: Literal["argsort"] = "argsort"


_PHASE4_RANKER_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "argsort": "argsort",
    "topk_v1": "topk_v1",
}


_PHASE4_RANKER_TIE_BEHAVIOR_BY_MODE: dict[str, str] = {
    "argsort": (
        "argsort preserves the current behavior; equal-score ordering follows torch.argsort "
        "backend semantics."
    ),
    "topk_v1": (
        "topk_v1 uses torch.topk for frontier membership; ties at the cutoff can select a "
        "different equal-score member than argsort. Selected members are then ordered by "
        "descending score (deterministic index tie-break for telemetry/debug stability)."
    ),
}


_EXACT_ENCODER_RESIDENCY_DEFAULT: Literal["lazy"] = "lazy"


_EXACT_ENCODER_RESIDENCY_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "lazy": "lazy",
    "active_cpu": "active_cpu",
}


_PHASE4_STREAMING_V1_MAX_MICROBATCH_SIZE = 64


_PHASE4_PLANNER_V2_POLICY_VERSION = "planner_v2_bounded_membership_v1"


_PHASE4_PLANNER_V2_CANDIDATE_WINDOW_MULTIPLIER = 2.0


_PHASE4_PLANNER_V2_LOCKED_PREFIX_FRACTION = 0.5


_PHASE4_PLANNER_V2_MAX_REPLACEMENT_FRACTION = 0.25


_PHASE4_PLANNER_V2_MIN_SCORE_RATIO = 0.995


@dataclass(frozen=True)
class _Phase4SchedulerConfig:
    requested_mode: Literal["locality", "planner_v1", "planner_v2"]
    effective_mode: Literal["locality", "planner_v1", "planner_v2"]
    version: str
    policy: str
    effective_version: str
    effective_policy: str
    effective_behavior: Literal["requested", "planner_v1_reference_execution"]
    debug: bool
    telemetry_detail: Literal["summary", "normal", "debug"]


@dataclass(frozen=True)
class _Phase4RefreshOptimizationConfig:
    requested_mode: Literal["off", "v1"]
    effective_mode: Literal["off", "v1"]
    version: str
    effective_version: str
    effective_behavior: Literal["requested", "off_reference_execution"]


@dataclass(frozen=True)
class _Phase4RowExecutorConfig:
    requested_mode: Literal["batched", "streaming_v1"]
    effective_mode: Literal["batched", "streaming_v1"]
    version: str
    effective_version: str
    effective_behavior: Literal["requested", "batched_reference_execution"]


@dataclass(frozen=True)
class _Phase4RowReductionConfig:
    requested_mode: Literal["off", "gpu_v1"]
    effective_mode: Literal["off", "gpu_v1"]
    version: str
    effective_version: str
    effective_behavior: Literal["requested", "off_reference_execution"]


@dataclass(frozen=True)
class _Phase4RefreshPolicyConfig:
    requested_policy: Literal["standard", "deferred_v1"]
    effective_policy: Literal["standard", "deferred_v1"]
    requested_interval_multiplier: int
    effective_interval_multiplier: int
    effective_queue_multiplier: int
    default_policy: Literal["standard"]
    default_interval_multiplier: int
    policy_applicable: bool
    effective_behavior: Literal["requested", "standard_reference_execution"]
    fallback_reason: str | None


@dataclass(frozen=True)
class _Phase4RankerConfig:
    requested_mode: Literal["argsort", "topk_v1"]
    effective_mode: Literal["argsort", "topk_v1"]
    default_mode: Literal["argsort"]
    effective_behavior: Literal["requested", "argsort_reference_execution"]


@dataclass(frozen=True)
class _ExactEncoderResidencyConfig:
    requested_mode: Literal["lazy", "active_cpu"]
    effective_mode: Literal["lazy", "active_cpu"]
    default_mode: Literal["lazy"]
    mode_applicable: bool
    effective_behavior: Literal["requested", "lazy_reference_execution"]
    fallback_reason: str | None


@dataclass(frozen=True)
class _Phase4FrontierPlan:
    selected_frontier: torch.Tensor
    batch_boundaries: list[tuple[int, int]]
    selected_membership_hash: str | None
    selected_order_hash: str | None
    locality_fragmentation_summary: dict[str, object]
    boundary_reason_counts: dict[str, int]
    invariant_summary: dict[str, object]


@dataclass(frozen=True)
class _Phase4FrontierRankSelection:
    selected_frontier: torch.Tensor
    selected_scores: torch.Tensor
    candidate_count: int
    selected_count: int
    selected_order_hash: str | None
    selected_membership_hash: str | None
    cutoff_score: float | None
    cutoff_gap: float | None
    relative_cutoff_gap: float | None
    near_cutoff_epsilon: float | None
    near_cutoff_count: int
    tie_count_at_cutoff: int
    tie_at_cutoff: bool
    tie_behavior: str


def _reorder_pending_for_phase4_locality(
    pending: torch.Tensor,
    *,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> torch.Tensor:
    """Stable-reorder a fixed frontier for better Phase-4 locality.

    Frontier membership stays unchanged; only execution order changes.
    The current priority is:
      1. source layer
      2. decoder chunk id (when exact chunked + chunk size available)
      3. position

    For equal keys, Python's stable sort preserves the original influence-rank order.
    """

    if pending.numel() <= 1:
        return pending

    use_chunk_key = bool(exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0)
    pending_list = pending.detach().cpu().tolist()
    pending_list.sort(
        key=lambda idx: (
            int(feat_layers[idx]),
            (int(feat_ids[idx]) // int(decoder_chunk_size)) if use_chunk_key else -1,
            int(feat_positions[idx]),
        )
    )
    return torch.tensor(pending_list, dtype=pending.dtype, device=pending.device)


def _compute_phase4_locality_shaped_batch_end_with_reason(
    pending: torch.Tensor,
    *,
    pending_offset: int,
    max_batch_size: int,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> tuple[int, str]:
    """Pick a Phase-4 batch end that prefers layer/chunk run boundaries.

    This keeps the frontier membership fixed and preserves ordering while avoiding
    unnecessary splits of contiguous ``(source_layer, decoder_chunk)`` runs when
    a boundary is available within the current max-size slice.

    To avoid over-splitting, only take the earlier boundary when the resulting
    split batch is not too small and the preserved suffix run is short.
    """

    total_pending = int(pending.numel())
    if pending_offset >= total_pending:
        return total_pending, "pending_exhausted"

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")

    if max_batch_size == 1:
        return min(pending_offset + max_batch_size, total_pending), "max_batch_size_one"

    baseline_end = min(pending_offset + max_batch_size, total_pending)
    if baseline_end >= total_pending:
        return baseline_end, "tail_complete"

    use_chunk_key = bool(exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0)
    probe = pending[pending_offset : baseline_end + 1]
    probe_layers = feat_layers[probe]
    if use_chunk_key:
        probe_chunks = torch.div(
            feat_ids[probe],
            int(decoder_chunk_size),
            rounding_mode="floor",
        )
    else:
        probe_chunks = torch.zeros_like(probe_layers)

    split_index = max_batch_size - 1
    if int(probe_layers[split_index].item()) != int(probe_layers[split_index + 1].item()):
        return baseline_end, "boundary_aligned"
    if int(probe_chunks[split_index].item()) != int(probe_chunks[split_index + 1].item()):
        return baseline_end, "boundary_aligned"

    prefix_layers = probe_layers[:max_batch_size]
    prefix_chunks = probe_chunks[:max_batch_size]
    boundaries = (prefix_layers[1:] != prefix_layers[:-1]) | (
        prefix_chunks[1:] != prefix_chunks[:-1]
    )
    boundary_positions = torch.nonzero(boundaries, as_tuple=False)
    if boundary_positions.numel() == 0:
        return baseline_end, "split_unavailable"

    last_boundary = int(boundary_positions[-1].item())
    split_batch_size = last_boundary + 1
    preserved_suffix_run = max_batch_size - split_batch_size

    # Keep the shaping heuristic intentionally conservative so easy prompts do
    # not fragment into many tiny refresh batches.
    min_split_batch_size = max(2, max_batch_size // 2)
    max_preserved_suffix_run = max(1, max_batch_size // 3)
    if split_batch_size < min_split_batch_size:
        return baseline_end, "split_too_small"
    if preserved_suffix_run > max_preserved_suffix_run:
        return baseline_end, "preserved_suffix_too_long"

    return pending_offset + split_batch_size, "split_at_last_boundary"


def _compute_phase4_locality_shaped_batch_end(
    pending: torch.Tensor,
    *,
    pending_offset: int,
    max_batch_size: int,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> int:
    batch_end, _ = _compute_phase4_locality_shaped_batch_end_with_reason(
        pending,
        pending_offset=pending_offset,
        max_batch_size=max_batch_size,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=exact_chunked_decoder,
        decoder_chunk_size=decoder_chunk_size,
    )
    return batch_end


def _compute_phase4_locality_shaped_frontier_size(
    pending: torch.Tensor,
    *,
    max_batch_size: int,
    max_batches: int,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> int:
    """Return the pending-prefix size covering at most ``max_batches`` shaped batches."""

    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")

    pending_offset = 0
    pending_size = int(pending.numel())
    for _ in range(max_batches):
        if pending_offset >= pending_size:
            break
        batch_end = _compute_phase4_locality_shaped_batch_end(
            pending,
            pending_offset=pending_offset,
            max_batch_size=max_batch_size,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        if batch_end <= pending_offset:
            raise RuntimeError("Phase 4 locality shaping produced a non-advancing batch boundary")
        pending_offset = batch_end
    return pending_offset


def _compute_phase4_refresh_cycle_batches(
    *,
    update_interval: int,
    queue_multiplier: int,
) -> int:
    resolved_update_interval = int(update_interval)
    resolved_queue_multiplier = int(queue_multiplier)
    if resolved_update_interval <= 0:
        raise ValueError("update_interval must be > 0")
    if resolved_queue_multiplier <= 0:
        raise ValueError("queue_multiplier must be > 0")
    return resolved_update_interval * resolved_queue_multiplier


def _compute_phase4_refresh_queue_window_size(
    *,
    update_interval: int,
    phase4_feature_batch_size: int,
    queue_multiplier: int,
    remaining_feature_count: int | None = None,
) -> int:
    resolved_feature_batch_size = int(phase4_feature_batch_size)
    if resolved_feature_batch_size <= 0:
        raise ValueError("phase4_feature_batch_size must be > 0")

    refresh_cycle_batches = _compute_phase4_refresh_cycle_batches(
        update_interval=update_interval,
        queue_multiplier=queue_multiplier,
    )
    queue_window_size = refresh_cycle_batches * resolved_feature_batch_size

    if remaining_feature_count is None:
        return int(queue_window_size)

    resolved_remaining = int(remaining_feature_count)
    if resolved_remaining < 0:
        raise ValueError("remaining_feature_count must be >= 0 when provided")
    return min(int(queue_window_size), resolved_remaining)


def _resolve_phase4_scheduler_mode(
    phase4_scheduler_mode: str,
) -> Literal["locality", "planner_v1", "planner_v2"]:
    normalized = str(phase4_scheduler_mode).strip().lower()
    normalized = _PHASE4_SCHEDULER_MODE_ALIAS.get(normalized, normalized)
    if normalized not in _PHASE4_SCHEDULER_POLICY_BY_MODE:
        allowed = ", ".join(
            sorted(set(_PHASE4_SCHEDULER_POLICY_BY_MODE) | set(_PHASE4_SCHEDULER_MODE_ALIAS))
        )
        raise ValueError(
            f"phase4_scheduler_mode must be one of: {allowed} (got {phase4_scheduler_mode!r})"
        )
    return cast(Literal["locality", "planner_v1", "planner_v2"], normalized)


def _resolve_phase4_scheduler_telemetry_detail(
    phase4_scheduler_telemetry_detail: str,
) -> Literal["summary", "normal", "debug"]:
    normalized = str(phase4_scheduler_telemetry_detail).strip().lower()
    normalized = _PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS.get(normalized, normalized)
    allowed_values = {"summary", "normal", "debug"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values | set(_PHASE4_SCHEDULER_TELEMETRY_DETAIL_ALIAS)))
        raise ValueError(
            "phase4_scheduler_telemetry_detail must be one of: "
            f"{allowed} (got {phase4_scheduler_telemetry_detail!r})"
        )
    return cast(Literal["summary", "normal", "debug"], normalized)


def _resolve_phase4_scheduler_config(
    *,
    phase4_scheduler_mode: str,
    phase4_scheduler_debug: bool,
    phase4_scheduler_telemetry_detail: str,
) -> _Phase4SchedulerConfig:
    requested_mode = _resolve_phase4_scheduler_mode(phase4_scheduler_mode)
    effective_mode = cast(
        Literal["locality", "planner_v1", "planner_v2"],
        _PHASE4_SCHEDULER_EFFECTIVE_MODE_BY_MODE[requested_mode],
    )
    effective_behavior: Literal["requested", "planner_v1_reference_execution"] = (
        "planner_v1_reference_execution" if requested_mode != effective_mode else "requested"
    )
    return _Phase4SchedulerConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        version=_PHASE4_SCHEDULER_VERSION_BY_MODE[requested_mode],
        policy=_PHASE4_SCHEDULER_POLICY_BY_MODE[requested_mode],
        effective_version=_PHASE4_SCHEDULER_VERSION_BY_MODE[effective_mode],
        effective_policy=_PHASE4_SCHEDULER_POLICY_BY_MODE[effective_mode],
        effective_behavior=effective_behavior,
        debug=bool(phase4_scheduler_debug),
        telemetry_detail=_resolve_phase4_scheduler_telemetry_detail(
            phase4_scheduler_telemetry_detail
        ),
    )


def _build_phase4_scheduler_metadata(
    phase4_scheduler_config: _Phase4SchedulerConfig,
) -> dict[str, object]:
    return {
        "scheduler_requested_mode": phase4_scheduler_config.requested_mode,
        "scheduler_mode_requested": phase4_scheduler_config.requested_mode,
        "scheduler_mode": phase4_scheduler_config.requested_mode,
        "scheduler_version": phase4_scheduler_config.version,
        "scheduler_version_requested": phase4_scheduler_config.version,
        "scheduler_policy": phase4_scheduler_config.policy,
        "scheduler_policy_requested": phase4_scheduler_config.policy,
        "scheduler_effective_mode": phase4_scheduler_config.effective_mode,
        "scheduler_mode_effective": phase4_scheduler_config.effective_mode,
        "scheduler_effective_version": phase4_scheduler_config.effective_version,
        "scheduler_version_effective": phase4_scheduler_config.effective_version,
        "scheduler_effective_policy": phase4_scheduler_config.effective_policy,
        "scheduler_policy_effective": phase4_scheduler_config.effective_policy,
        "scheduler_effective_behavior": phase4_scheduler_config.effective_behavior,
        "scheduler_reference_execution": bool(
            phase4_scheduler_config.requested_mode != phase4_scheduler_config.effective_mode
        ),
        "scheduler_debug": bool(phase4_scheduler_config.debug),
        "scheduler_telemetry_detail": phase4_scheduler_config.telemetry_detail,
    }


def _resolve_phase4_refresh_optimization_mode(
    phase4_refresh_optimization: str,
) -> Literal["off", "v1"]:
    normalized = str(phase4_refresh_optimization).strip().lower()
    allowed_values = {"off", "v1"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            "phase4_refresh_optimization must be one of: "
            f"{allowed} (got {phase4_refresh_optimization!r})"
        )
    return cast(Literal["off", "v1"], normalized)


def _resolve_phase4_refresh_optimization_config(
    phase4_refresh_optimization: str,
    *,
    compact_output: bool,
    exact_chunked_provider_enabled: bool,
) -> _Phase4RefreshOptimizationConfig:
    requested_mode = _resolve_phase4_refresh_optimization_mode(phase4_refresh_optimization)
    refresh_optimization_applicable = bool(compact_output and exact_chunked_provider_enabled)
    effective_mode = cast(
        Literal["off", "v1"],
        requested_mode if requested_mode == "off" or refresh_optimization_applicable else "off",
    )
    effective_behavior: Literal["requested", "off_reference_execution"] = (
        "off_reference_execution" if requested_mode != effective_mode else "requested"
    )
    return _Phase4RefreshOptimizationConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        version=_PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE[requested_mode],
        effective_version=_PHASE4_REFRESH_OPTIMIZATION_VERSION_BY_MODE[effective_mode],
        effective_behavior=effective_behavior,
    )


def _build_phase4_refresh_optimization_metadata(
    phase4_refresh_optimization_config: _Phase4RefreshOptimizationConfig,
) -> dict[str, object]:
    return {
        "refresh_optimization_requested": phase4_refresh_optimization_config.requested_mode,
        "refresh_optimization_mode_requested": phase4_refresh_optimization_config.requested_mode,
        "refresh_optimization": phase4_refresh_optimization_config.requested_mode,
        "refresh_optimization_version": phase4_refresh_optimization_config.version,
        "refresh_optimization_version_requested": phase4_refresh_optimization_config.version,
        "refresh_optimization_effective": phase4_refresh_optimization_config.effective_mode,
        "refresh_optimization_mode_effective": phase4_refresh_optimization_config.effective_mode,
        "refresh_optimization_effective_version": phase4_refresh_optimization_config.effective_version,
        "refresh_optimization_version_effective": phase4_refresh_optimization_config.effective_version,
        "refresh_optimization_effective_behavior": phase4_refresh_optimization_config.effective_behavior,
        "refresh_optimization_reference_execution": bool(
            phase4_refresh_optimization_config.requested_mode
            != phase4_refresh_optimization_config.effective_mode
        ),
    }


def _resolve_phase4_row_executor_mode(
    phase4_row_executor: str,
) -> Literal["batched", "streaming_v1"]:
    normalized = str(phase4_row_executor).strip().lower()
    allowed_values = {"batched", "streaming_v1"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"phase4_row_executor must be one of: {allowed} (got {phase4_row_executor!r})"
        )
    return cast(Literal["batched", "streaming_v1"], normalized)


def _resolve_phase4_row_executor_config(
    phase4_row_executor: str,
    *,
    compact_output: bool,
    exact_chunked_provider_enabled: bool,
) -> _Phase4RowExecutorConfig:
    requested_mode = _resolve_phase4_row_executor_mode(phase4_row_executor)
    streaming_executor_applicable = bool(compact_output and exact_chunked_provider_enabled)
    effective_mode = cast(
        Literal["batched", "streaming_v1"],
        (
            _PHASE4_ROW_EXECUTOR_EFFECTIVE_MODE_BY_MODE[requested_mode]
            if requested_mode == "batched" or streaming_executor_applicable
            else "batched"
        ),
    )
    effective_behavior: Literal["requested", "batched_reference_execution"] = (
        "batched_reference_execution" if requested_mode != effective_mode else "requested"
    )
    return _Phase4RowExecutorConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        version=_PHASE4_ROW_EXECUTOR_VERSION_BY_MODE[requested_mode],
        effective_version=_PHASE4_ROW_EXECUTOR_VERSION_BY_MODE[effective_mode],
        effective_behavior=effective_behavior,
    )


def _resolve_phase4_streaming_v1_microbatch_size(reference_batch_size: int) -> int:
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size must be > 0")
    return min(reference_batch_size, _PHASE4_STREAMING_V1_MAX_MICROBATCH_SIZE)


def _build_phase4_row_executor_metadata(
    phase4_row_executor_config: _Phase4RowExecutorConfig,
) -> dict[str, object]:
    return {
        "row_executor_requested": phase4_row_executor_config.requested_mode,
        "row_executor_mode_requested": phase4_row_executor_config.requested_mode,
        "row_executor": phase4_row_executor_config.requested_mode,
        "row_executor_version": phase4_row_executor_config.version,
        "row_executor_version_requested": phase4_row_executor_config.version,
        "row_executor_effective": phase4_row_executor_config.effective_mode,
        "row_executor_mode_effective": phase4_row_executor_config.effective_mode,
        "row_executor_effective_version": phase4_row_executor_config.effective_version,
        "row_executor_version_effective": phase4_row_executor_config.effective_version,
        "row_executor_effective_behavior": phase4_row_executor_config.effective_behavior,
        "row_executor_reference_execution": bool(
            phase4_row_executor_config.requested_mode != phase4_row_executor_config.effective_mode
        ),
    }


def _resolve_phase4_row_reduction_mode(
    phase4_row_reduction: str,
) -> Literal["off", "gpu_v1"]:
    normalized = str(phase4_row_reduction).strip().lower()
    allowed_values = {"off", "gpu_v1"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"phase4_row_reduction must be one of: {allowed} (got {phase4_row_reduction!r})"
        )
    return cast(Literal["off", "gpu_v1"], normalized)


def _resolve_phase4_row_reduction_config(
    phase4_row_reduction: str,
    *,
    compact_output: bool,
    exact_chunked_provider_enabled: bool,
) -> _Phase4RowReductionConfig:
    requested_mode = _resolve_phase4_row_reduction_mode(phase4_row_reduction)
    gpu_v1_applicable = bool(compact_output and exact_chunked_provider_enabled)
    effective_mode = cast(
        Literal["off", "gpu_v1"],
        requested_mode if requested_mode == "off" or gpu_v1_applicable else "off",
    )
    effective_behavior: Literal["requested", "off_reference_execution"] = (
        "off_reference_execution" if requested_mode != effective_mode else "requested"
    )
    return _Phase4RowReductionConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        version=_PHASE4_ROW_REDUCTION_VERSION_BY_MODE[requested_mode],
        effective_version=_PHASE4_ROW_REDUCTION_VERSION_BY_MODE[effective_mode],
        effective_behavior=effective_behavior,
    )


def _build_phase4_row_reduction_metadata(
    phase4_row_reduction_config: _Phase4RowReductionConfig,
) -> dict[str, object]:
    return {
        "row_reduction_requested": phase4_row_reduction_config.requested_mode,
        "row_reduction_mode_requested": phase4_row_reduction_config.requested_mode,
        "row_reduction": phase4_row_reduction_config.requested_mode,
        "row_reduction_version": phase4_row_reduction_config.version,
        "row_reduction_version_requested": phase4_row_reduction_config.version,
        "row_reduction_effective": phase4_row_reduction_config.effective_mode,
        "row_reduction_mode_effective": phase4_row_reduction_config.effective_mode,
        "row_reduction_effective_version": phase4_row_reduction_config.effective_version,
        "row_reduction_version_effective": phase4_row_reduction_config.effective_version,
        "row_reduction_effective_behavior": phase4_row_reduction_config.effective_behavior,
        "row_reduction_reference_execution": bool(
            phase4_row_reduction_config.requested_mode != phase4_row_reduction_config.effective_mode
        ),
    }


def _resolve_phase4_refresh_policy(
    phase4_refresh_policy: str,
) -> Literal["standard", "deferred_v1"]:
    normalized = str(phase4_refresh_policy).strip().lower()
    allowed_values = {"standard", "deferred_v1"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"phase4_refresh_policy must be one of: {allowed} (got {phase4_refresh_policy!r})"
        )
    return cast(Literal["standard", "deferred_v1"], normalized)


def _resolve_phase4_refresh_interval_multiplier(
    phase4_refresh_interval_multiplier: int,
) -> int:
    resolved = int(phase4_refresh_interval_multiplier)
    if resolved <= 0:
        raise ValueError("phase4_refresh_interval_multiplier must be > 0")
    return resolved


def _resolve_phase4_refresh_policy_config(
    *,
    phase4_refresh_policy: str,
    phase4_refresh_interval_multiplier: int,
    compact_output: bool,
    exact_chunked_provider_enabled: bool,
) -> _Phase4RefreshPolicyConfig:
    requested_policy = _resolve_phase4_refresh_policy(phase4_refresh_policy)
    requested_interval_multiplier = _resolve_phase4_refresh_interval_multiplier(
        phase4_refresh_interval_multiplier
    )
    policy_applicable = bool(compact_output and exact_chunked_provider_enabled)

    fallback_reason: str | None = None
    if requested_policy == "deferred_v1" and not policy_applicable:
        effective_policy = cast(Literal["standard", "deferred_v1"], "standard")
        effective_interval_multiplier = _PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT
        effective_behavior: Literal["requested", "standard_reference_execution"] = (
            "standard_reference_execution"
        )
        fallback_reason = (
            "deferred_v1 requires compact_output=True and exact chunked provider support; "
            "falling back to standard execution"
        )
    elif requested_policy == "standard":
        effective_policy = cast(
            Literal["standard", "deferred_v1"],
            _PHASE4_REFRESH_POLICY_EFFECTIVE_POLICY_BY_POLICY[requested_policy],
        )
        effective_interval_multiplier = _PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT
        effective_behavior = (
            "requested"
            if requested_interval_multiplier == _PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT
            else "standard_reference_execution"
        )
    else:
        effective_policy = cast(
            Literal["standard", "deferred_v1"],
            _PHASE4_REFRESH_POLICY_EFFECTIVE_POLICY_BY_POLICY[requested_policy],
        )
        effective_interval_multiplier = requested_interval_multiplier
        effective_behavior = "requested"

    effective_queue_multiplier = int(effective_interval_multiplier)
    return _Phase4RefreshPolicyConfig(
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        requested_interval_multiplier=requested_interval_multiplier,
        effective_interval_multiplier=effective_interval_multiplier,
        effective_queue_multiplier=effective_queue_multiplier,
        default_policy=_PHASE4_REFRESH_POLICY_DEFAULT,
        default_interval_multiplier=_PHASE4_REFRESH_INTERVAL_MULTIPLIER_DEFAULT,
        policy_applicable=policy_applicable,
        effective_behavior=effective_behavior,
        fallback_reason=fallback_reason,
    )


def _build_phase4_refresh_policy_metadata(
    phase4_refresh_policy_config: _Phase4RefreshPolicyConfig,
) -> dict[str, object]:
    return {
        "refresh_policy_requested": phase4_refresh_policy_config.requested_policy,
        "refresh_policy": phase4_refresh_policy_config.requested_policy,
        "refresh_policy_default": phase4_refresh_policy_config.default_policy,
        "refresh_policy_effective": phase4_refresh_policy_config.effective_policy,
        "refresh_policy_applicable": bool(phase4_refresh_policy_config.policy_applicable),
        "refresh_policy_effective_behavior": phase4_refresh_policy_config.effective_behavior,
        "refresh_policy_fallback_reason": phase4_refresh_policy_config.fallback_reason,
        "refresh_policy_reference_execution": bool(
            phase4_refresh_policy_config.requested_policy
            != phase4_refresh_policy_config.effective_policy
        ),
        "refresh_interval_multiplier_requested": (
            phase4_refresh_policy_config.requested_interval_multiplier
        ),
        "refresh_interval_multiplier": phase4_refresh_policy_config.requested_interval_multiplier,
        "refresh_interval_multiplier_default": phase4_refresh_policy_config.default_interval_multiplier,
        "refresh_interval_multiplier_effective": (
            phase4_refresh_policy_config.effective_interval_multiplier
        ),
        "refresh_queue_multiplier_effective": (
            phase4_refresh_policy_config.effective_queue_multiplier
        ),
        "refresh_interval_multiplier_reference_execution": bool(
            phase4_refresh_policy_config.requested_interval_multiplier
            != phase4_refresh_policy_config.effective_interval_multiplier
        ),
    }


def _resolve_phase4_ranker(
    phase4_ranker: str,
) -> Literal["argsort", "topk_v1"]:
    normalized = str(phase4_ranker).strip().lower()
    allowed_values = {"argsort", "topk_v1"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"phase4_ranker must be one of: {allowed} (got {phase4_ranker!r})")
    return cast(Literal["argsort", "topk_v1"], normalized)


def _resolve_phase4_ranker_config(
    phase4_ranker: str,
) -> _Phase4RankerConfig:
    requested_mode = _resolve_phase4_ranker(phase4_ranker)
    effective_mode = cast(
        Literal["argsort", "topk_v1"],
        _PHASE4_RANKER_EFFECTIVE_MODE_BY_MODE[requested_mode],
    )
    effective_behavior: Literal["requested", "argsort_reference_execution"] = (
        "requested" if requested_mode == effective_mode else "argsort_reference_execution"
    )
    return _Phase4RankerConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        default_mode=_PHASE4_RANKER_DEFAULT,
        effective_behavior=effective_behavior,
    )


def _build_phase4_ranker_metadata(
    phase4_ranker_config: _Phase4RankerConfig,
) -> dict[str, object]:
    return {
        "ranker_requested": phase4_ranker_config.requested_mode,
        "ranker": phase4_ranker_config.requested_mode,
        "ranker_default": phase4_ranker_config.default_mode,
        "ranker_effective": phase4_ranker_config.effective_mode,
        "ranker_effective_behavior": phase4_ranker_config.effective_behavior,
        "ranker_reference_execution": bool(
            phase4_ranker_config.requested_mode != phase4_ranker_config.effective_mode
        ),
    }


def _rank_phase4_unvisited_features_argsort(
    feature_influences: torch.Tensor,
    visited: torch.Tensor,
) -> torch.Tensor:
    feature_rank = (
        torch.argsort(feature_influences, descending=True)
        .detach()
        .to(
            device="cpu",
            dtype=torch.long,
        )
    )
    visited_cpu = visited.detach().to(device="cpu", dtype=torch.bool).flatten()
    if visited_cpu.numel() != feature_rank.numel():
        raise ValueError(
            "feature_influences and visited must have matching flattened lengths "
            f"(got {feature_rank.numel()} and {visited_cpu.numel()})"
        )
    return feature_rank[~visited_cpu[feature_rank]]


def _compute_phase4_rank_selection_cutoff_metadata(
    *,
    unvisited_scores: torch.Tensor,
    selected_scores: torch.Tensor,
    selected_count: int,
    candidate_count: int,
) -> tuple[float | None, float | None, float | None, float | None, int, int, bool]:
    if selected_count <= 0 or candidate_count <= 0 or unvisited_scores.numel() <= 0:
        return None, None, None, None, 0, 0, False

    scores_cpu = unvisited_scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    selected_scores_cpu = selected_scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    if scores_cpu.numel() < candidate_count:
        candidate_count = int(scores_cpu.numel())
        selected_count = min(selected_count, candidate_count)
    if selected_scores_cpu.numel() < selected_count:
        selected_count = int(selected_scores_cpu.numel())

    if selected_count <= 0:
        return None, None, None, None, 0, 0, False

    scores_cpu = scores_cpu[:candidate_count]
    selected_scores_cpu = selected_scores_cpu[:selected_count]
    cutoff_score = float(selected_scores_cpu[selected_count - 1].item())
    selection_bound = bool(selected_count < candidate_count)
    cutoff_gap: float | None = None
    relative_cutoff_gap: float | None = None
    near_cutoff_epsilon: float | None = None
    near_cutoff_count = 0
    tie_count_at_cutoff = int((scores_cpu == cutoff_score).sum().item())
    strictly_greater_count = int((scores_cpu > cutoff_score).sum().item())
    if selection_bound:
        selected_at_cutoff = selected_count - strictly_greater_count
        unselected_ties = tie_count_at_cutoff - max(selected_at_cutoff, 0)
        if unselected_ties > 0:
            cutoff_gap = 0.0
        else:
            below_cutoff_scores = scores_cpu[scores_cpu < cutoff_score]
            if below_cutoff_scores.numel() > 0:
                next_score = float(below_cutoff_scores.max().item())
                cutoff_gap = float(cutoff_score - next_score)
        if cutoff_score > 0:
            relative_cutoff_gap = (
                float(cutoff_gap / cutoff_score) if cutoff_gap is not None else None
            )
            near_cutoff_epsilon = float(_PHASE4_RANK_SELECTION_NEAR_CUTOFF_EPSILON)
            near_boundary = cutoff_score * (1.0 - near_cutoff_epsilon)
            near_cutoff_count = unselected_ties + int(
                ((scores_cpu < cutoff_score) & (scores_cpu >= near_boundary)).sum().item()
            )
    tie_at_cutoff = bool(
        selected_count < candidate_count
        and tie_count_at_cutoff > 1
        and strictly_greater_count < selected_count
    )
    return (
        cutoff_score,
        cutoff_gap,
        relative_cutoff_gap,
        near_cutoff_epsilon,
        near_cutoff_count,
        tie_count_at_cutoff,
        tie_at_cutoff,
    )


def _compute_phase4_rank_selection_max_feature_nodes_cap_bound(
    *,
    candidate_count: int,
    actual_max_feature_nodes: int,
    n_visited: int,
    max_frontier_size: int,
) -> bool:
    remaining_feature_budget = max(0, int(actual_max_feature_nodes - n_visited))
    return bool(
        int(candidate_count) > remaining_feature_budget
        and int(max_frontier_size) >= remaining_feature_budget
    )


def _select_phase4_frontier_rank_selection(
    *,
    feature_influences: torch.Tensor,
    visited: torch.Tensor,
    frontier_size: int,
    ranker_mode: Literal["argsort", "topk_v1"],
) -> _Phase4FrontierRankSelection:
    if frontier_size < 0:
        raise ValueError("frontier_size must be >= 0")

    visited_cpu = visited.detach().to(device="cpu", dtype=torch.bool).flatten()
    if feature_influences.numel() != visited_cpu.numel():
        raise ValueError(
            "feature_influences and visited must have matching flattened lengths "
            f"(got {feature_influences.numel()} and {visited_cpu.numel()})"
        )

    candidate_count = int((~visited_cpu).sum().item())
    selected_count = min(int(frontier_size), candidate_count)
    if selected_count <= 0:
        return _Phase4FrontierRankSelection(
            selected_frontier=torch.empty(0, dtype=torch.long),
            selected_scores=torch.empty(0, dtype=torch.float64),
            candidate_count=candidate_count,
            selected_count=0,
            selected_order_hash=None,
            selected_membership_hash=None,
            cutoff_score=None,
            cutoff_gap=None,
            relative_cutoff_gap=None,
            near_cutoff_epsilon=None,
            near_cutoff_count=0,
            tie_count_at_cutoff=0,
            tie_at_cutoff=False,
            tie_behavior=_PHASE4_RANKER_TIE_BEHAVIOR_BY_MODE[ranker_mode],
        )

    if ranker_mode == "argsort":
        unvisited_rank = _rank_phase4_unvisited_features_argsort(feature_influences, visited_cpu)
        unvisited_scores = (
            feature_influences[unvisited_rank.to(feature_influences.device)]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        selected_frontier = unvisited_rank[:selected_count]
        selected_scores = unvisited_scores[:selected_count]
    else:
        unvisited_indices = (
            torch.nonzero(~visited_cpu, as_tuple=False).flatten().to(dtype=torch.long)
        )
        if unvisited_indices.numel() != candidate_count:
            raise RuntimeError(
                "Phase-4 topk_v1 candidate_count mismatch against unvisited index selection "
                f"(count={candidate_count}, selected={int(unvisited_indices.numel())})"
            )
        unvisited_scores_device = feature_influences[
            unvisited_indices.to(feature_influences.device)
        ]
        top_scores, top_positions = torch.topk(
            unvisited_scores_device,
            k=selected_count,
            largest=True,
            sorted=False,
        )
        top_scores_cpu = top_scores.detach().to(device="cpu", dtype=torch.float64)
        top_indices_cpu = unvisited_indices[
            top_positions.detach().to(device="cpu", dtype=torch.long)
        ]

        selected_entries = sorted(
            zip(top_indices_cpu.tolist(), top_scores_cpu.tolist(), strict=False),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        selected_frontier = torch.tensor(
            [int(index) for index, _ in selected_entries],
            dtype=torch.long,
        )
        selected_scores = torch.tensor(
            [float(score) for _, score in selected_entries],
            dtype=torch.float64,
        )
        unvisited_scores = unvisited_scores_device.detach().to(device="cpu", dtype=torch.float64)

    (
        cutoff_score,
        cutoff_gap,
        relative_cutoff_gap,
        near_cutoff_epsilon,
        near_cutoff_count,
        tie_count_at_cutoff,
        tie_at_cutoff,
    ) = _compute_phase4_rank_selection_cutoff_metadata(
        unvisited_scores=unvisited_scores,
        selected_scores=selected_scores,
        selected_count=selected_count,
        candidate_count=candidate_count,
    )

    selected_membership_hash = (
        _hash_index_tensor(torch.sort(selected_frontier).values)
        if selected_frontier.numel() > 0
        else None
    )
    selected_order_hash = (
        _hash_index_tensor(selected_frontier) if selected_frontier.numel() > 0 else None
    )
    return _Phase4FrontierRankSelection(
        selected_frontier=selected_frontier,
        selected_scores=selected_scores,
        candidate_count=candidate_count,
        selected_count=selected_count,
        selected_order_hash=selected_order_hash,
        selected_membership_hash=selected_membership_hash,
        cutoff_score=cutoff_score,
        cutoff_gap=cutoff_gap,
        relative_cutoff_gap=relative_cutoff_gap,
        near_cutoff_epsilon=near_cutoff_epsilon,
        near_cutoff_count=near_cutoff_count,
        tie_count_at_cutoff=tie_count_at_cutoff,
        tie_at_cutoff=tie_at_cutoff,
        tie_behavior=_PHASE4_RANKER_TIE_BEHAVIOR_BY_MODE[ranker_mode],
    )


def _resolve_exact_encoder_residency(
    exact_encoder_residency: str,
) -> Literal["lazy", "active_cpu"]:
    normalized = str(exact_encoder_residency).strip().lower()
    allowed_values = {"lazy", "active_cpu"}
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"exact_encoder_residency must be one of: {allowed} (got {exact_encoder_residency!r})"
        )
    return cast(Literal["lazy", "active_cpu"], normalized)


def _resolve_exact_encoder_residency_config(
    exact_encoder_residency: str,
    *,
    supports_exact_encoder_residency: bool,
) -> _ExactEncoderResidencyConfig:
    requested_mode = _resolve_exact_encoder_residency(exact_encoder_residency)
    mode_applicable = bool(supports_exact_encoder_residency)
    fallback_reason: str | None = None
    if requested_mode != "lazy" and not mode_applicable:
        effective_mode = cast(Literal["lazy", "active_cpu"], "lazy")
        fallback_reason = (
            "active encoder residency requires exact encoder-residency provider support; "
            "falling back to lazy execution"
        )
    else:
        effective_mode = cast(
            Literal["lazy", "active_cpu"],
            _EXACT_ENCODER_RESIDENCY_EFFECTIVE_MODE_BY_MODE[requested_mode],
        )
    effective_behavior: Literal["requested", "lazy_reference_execution"] = (
        "requested" if requested_mode == effective_mode else "lazy_reference_execution"
    )
    return _ExactEncoderResidencyConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        default_mode=_EXACT_ENCODER_RESIDENCY_DEFAULT,
        mode_applicable=mode_applicable,
        effective_behavior=effective_behavior,
        fallback_reason=fallback_reason,
    )


def _build_exact_encoder_residency_metadata(
    exact_encoder_residency_config: _ExactEncoderResidencyConfig,
) -> dict[str, object]:
    effective_mode = exact_encoder_residency_config.effective_mode
    return {
        "exact_encoder_residency_requested": exact_encoder_residency_config.requested_mode,
        "exact_encoder_residency": exact_encoder_residency_config.requested_mode,
        "exact_encoder_residency_default": exact_encoder_residency_config.default_mode,
        "exact_encoder_residency_effective": exact_encoder_residency_config.effective_mode,
        "exact_encoder_residency_applicable": bool(exact_encoder_residency_config.mode_applicable),
        "exact_encoder_residency_effective_behavior": (
            exact_encoder_residency_config.effective_behavior
        ),
        "exact_encoder_residency_fallback_reason": exact_encoder_residency_config.fallback_reason,
        "exact_encoder_materialize_phase0": bool(effective_mode != "lazy"),
        "exact_encoder_staging_destination_planned": (
            "none" if effective_mode == "lazy" else "cpu"
        ),
        "exact_encoder_pinned_requested": False,
        "exact_encoder_pinned_planned": False,
        "exact_encoder_pinned_effective": None,
        "exact_encoder_pinning_success": None,
        "exact_encoder_pinning_failure_reason": None,
        "exact_encoder_residency_reference_execution": bool(
            exact_encoder_residency_config.requested_mode
            != exact_encoder_residency_config.effective_mode
        ),
    }


def _build_phase4_scheduler_plan_telemetry(
    *,
    phase4_frontier_plan: _Phase4FrontierPlan | None,
    telemetry_detail: Literal["summary", "normal", "debug"],
) -> dict[str, object]:
    if phase4_frontier_plan is None:
        return {
            "scheduler_plan_frontier_size": None,
            "scheduler_plan_membership_hash": None,
            "scheduler_plan_order_hash": None,
            "scheduler_plan_batch_count": None,
            "scheduler_plan_boundary_reason_counts": None,
            "scheduler_plan_invariants": None,
            "scheduler_plan_layer_chunk_run_count": None,
            "scheduler_plan_layer_chunk_transition_count": None,
            "scheduler_plan_layer_chunk_fragmentation_ratio": None,
            "scheduler_plan_batch_fragmentation_ratio": None,
        }

    locality_summary = phase4_frontier_plan.locality_fragmentation_summary
    boundary_reason_counts = {
        str(key): int(value) for key, value in phase4_frontier_plan.boundary_reason_counts.items()
    }
    invariant_summary = {
        str(key): value for key, value in phase4_frontier_plan.invariant_summary.items()
    }
    if telemetry_detail == "summary":
        invariant_summary = {
            "membership_preserved": bool(invariant_summary.get("membership_preserved")),
            "duplicate_count": int(invariant_summary.get("duplicate_count", 0)),
            "missing_count": int(invariant_summary.get("missing_count", 0)),
            "unexpected_count": int(invariant_summary.get("unexpected_count", 0)),
            "non_advancing_boundary_count": int(
                invariant_summary.get("non_advancing_boundary_count", 0)
            ),
        }

    payload: dict[str, object] = {
        "scheduler_plan_frontier_size": int(phase4_frontier_plan.selected_frontier.numel()),
        "scheduler_plan_membership_hash": phase4_frontier_plan.selected_membership_hash,
        "scheduler_plan_order_hash": phase4_frontier_plan.selected_order_hash,
        "scheduler_plan_batch_count": int(len(phase4_frontier_plan.batch_boundaries)),
        "scheduler_plan_boundary_reason_counts": boundary_reason_counts,
        "scheduler_plan_invariants": invariant_summary,
        "scheduler_plan_layer_chunk_run_count": int(
            locality_summary.get("layer_chunk_run_count", 0)
        ),
        "scheduler_plan_layer_chunk_transition_count": int(
            locality_summary.get("layer_chunk_transition_count", 0)
        ),
        "scheduler_plan_layer_chunk_fragmentation_ratio": _safe_float(
            locality_summary.get("layer_chunk_fragmentation_ratio")
        ),
        "scheduler_plan_batch_fragmentation_ratio": _safe_float(
            locality_summary.get("batch_fragmentation_ratio")
        ),
    }
    if telemetry_detail in {"normal", "debug"}:
        payload["scheduler_plan_locality_fragmentation"] = dict(locality_summary)
    if telemetry_detail == "debug":
        boundary_sample = phase4_frontier_plan.batch_boundaries[:8]
        payload["scheduler_plan_batch_boundaries_sample"] = [
            [int(start), int(end)] for start, end in boundary_sample
        ]
        payload["scheduler_plan_batch_boundaries_sample_count"] = int(len(boundary_sample))
    return payload


def _build_phase4_planner_v2_refresh_telemetry_disabled() -> dict[str, object]:
    return {
        "scheduler_planner_v2_enabled": False,
        "scheduler_planner_v2_policy_version": None,
        "scheduler_planner_v2_reference_frontier_size": None,
        "scheduler_planner_v2_candidate_window_size": None,
        "scheduler_planner_v2_candidate_window_multiplier": None,
        "scheduler_planner_v2_locked_prefix_fraction": None,
        "scheduler_planner_v2_locked_prefix_size": None,
        "scheduler_planner_v2_max_replacement_fraction": None,
        "scheduler_planner_v2_max_replacement_count": None,
        "scheduler_planner_v2_min_score_ratio": None,
        "scheduler_planner_v2_score_cutoff": None,
        "scheduler_planner_v2_score_threshold": None,
        "scheduler_planner_v2_score_threshold_applied": None,
        "scheduler_planner_v2_candidate_window_order_hash": None,
        "scheduler_planner_v2_candidate_window_membership_hash": None,
        "scheduler_planner_v2_candidate_window_includes_reference": None,
        "scheduler_planner_v2_selection_attempted": None,
        "scheduler_planner_v2_selection_applied": None,
        "scheduler_planner_v2_selection_changed_membership": None,
        "scheduler_planner_v2_fallback_to_reference": None,
        "scheduler_planner_v2_fallback_reason": None,
        "scheduler_planner_v2_reference_membership_hash": None,
        "scheduler_planner_v2_selected_membership_hash": None,
        "scheduler_planner_v2_locked_prefix_membership_hash": None,
        "scheduler_planner_v2_replacement_count": None,
        "scheduler_planner_v2_replacement_fraction_realized": None,
        "scheduler_planner_v2_reference_score_sum": None,
        "scheduler_planner_v2_selected_score_sum": None,
        "scheduler_planner_v2_selected_score_ratio": None,
        "scheduler_planner_v2_reference_group_count": None,
        "scheduler_planner_v2_selected_group_count": None,
        "scheduler_planner_v2_group_count_delta": None,
        "scheduler_planner_v2_rank_displacement_sum": None,
    }


def _build_phase4_planner_v2_candidate_window(
    unvisited_feature_rank: torch.Tensor,
    *,
    reference_frontier: torch.Tensor,
    reference_frontier_size: int,
    candidate_scores: torch.Tensor,
    window_multiplier: float = _PHASE4_PLANNER_V2_CANDIDATE_WINDOW_MULTIPLIER,
    locked_prefix_fraction: float = _PHASE4_PLANNER_V2_LOCKED_PREFIX_FRACTION,
    max_replacement_fraction: float = _PHASE4_PLANNER_V2_MAX_REPLACEMENT_FRACTION,
    min_score_ratio: float = _PHASE4_PLANNER_V2_MIN_SCORE_RATIO,
    max_window_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if reference_frontier_size < 0:
        raise ValueError("reference_frontier_size must be >= 0")

    ranked = unvisited_feature_rank.detach().to(device="cpu", dtype=torch.long)
    reference = reference_frontier.detach().to(device="cpu", dtype=torch.long)
    scores = candidate_scores.detach().to(device="cpu", dtype=torch.float64).flatten()

    available_count = int(ranked.numel())
    reference_size = min(
        int(reference_frontier_size),
        int(reference.numel()),
        available_count,
    )

    multiplier = max(1.0, float(window_multiplier))
    locked_fraction = min(max(0.0, float(locked_prefix_fraction)), 1.0)
    replacement_fraction = max(0.0, float(max_replacement_fraction))
    score_ratio = min(max(0.0, float(min_score_ratio)), 1.0)

    locked_prefix_size = min(reference_size, int(math.floor(reference_size * locked_fraction)))
    max_replacement_count = int(math.ceil(reference_size * replacement_fraction))

    multiplier_target_size = (
        max(reference_size, int(math.ceil(reference_size * multiplier)))
        if reference_size > 0
        else 0
    )
    replacement_target_size = reference_size + max_replacement_count
    bounded_target_size = max(
        reference_size,
        min(multiplier_target_size, replacement_target_size),
    )
    if max_window_size is not None:
        bounded_target_size = min(bounded_target_size, int(max_window_size))
    bounded_target_size = min(bounded_target_size, available_count)

    score_cutoff = None
    score_threshold = None
    score_threshold_applied = False
    if (
        reference_size > 0
        and scores.numel() >= reference_size
        and scores.numel() >= available_count
    ):
        score_cutoff_value = float(scores[reference_size - 1].item())
        score_cutoff = score_cutoff_value
        if math.isfinite(score_cutoff_value) and score_cutoff_value > 0.0:
            score_threshold = float(score_cutoff_value * score_ratio)
            score_threshold_applied = True
            ratio_eligible_size = int((scores[:available_count] >= score_threshold).sum().item())
            bounded_target_size = min(
                bounded_target_size,
                max(reference_size, ratio_eligible_size),
            )

    window_size = bounded_target_size

    missing_reference_nodes: set[int] = set()
    if reference_size > 0:
        reference_nodes = reference[:reference_size]
        if window_size > 0:
            in_window = torch.isin(reference_nodes, ranked[:window_size])
        else:
            in_window = torch.zeros(reference_nodes.shape, dtype=torch.bool)
        missing_reference_nodes = {int(value) for value in reference_nodes[~in_window].tolist()}
        if missing_reference_nodes:
            max_reference_rank = window_size - 1
            for rank_idx, node_idx in enumerate(ranked.tolist()):
                if int(node_idx) in missing_reference_nodes:
                    max_reference_rank = max(max_reference_rank, rank_idx)
                    missing_reference_nodes.remove(int(node_idx))
                    if not missing_reference_nodes:
                        break
            if missing_reference_nodes:
                raise RuntimeError(
                    "Planner v2 candidate window missing reference frontier nodes "
                    "outside unvisited rank ordering"
                )
            window_size = max(window_size, max_reference_rank + 1)

    candidate_window = ranked[:window_size]
    candidate_window_sorted = (
        torch.sort(candidate_window).values if candidate_window.numel() > 0 else candidate_window
    )

    includes_reference = True
    if reference_size > 0:
        includes_reference = bool(
            torch.isin(reference[:reference_size], candidate_window).all().item()
        )

    telemetry: dict[str, object] = {
        "scheduler_planner_v2_enabled": True,
        "scheduler_planner_v2_policy_version": _PHASE4_PLANNER_V2_POLICY_VERSION,
        "scheduler_planner_v2_reference_frontier_size": int(reference_size),
        "scheduler_planner_v2_candidate_window_size": int(candidate_window.numel()),
        "scheduler_planner_v2_candidate_window_multiplier": float(multiplier),
        "scheduler_planner_v2_locked_prefix_fraction": float(locked_fraction),
        "scheduler_planner_v2_locked_prefix_size": int(locked_prefix_size),
        "scheduler_planner_v2_max_replacement_fraction": float(replacement_fraction),
        "scheduler_planner_v2_max_replacement_count": int(max_replacement_count),
        "scheduler_planner_v2_min_score_ratio": float(score_ratio),
        "scheduler_planner_v2_score_cutoff": score_cutoff,
        "scheduler_planner_v2_score_threshold": score_threshold,
        "scheduler_planner_v2_score_threshold_applied": bool(score_threshold_applied),
        "scheduler_planner_v2_candidate_window_order_hash": (
            _hash_index_tensor(candidate_window) if candidate_window.numel() > 0 else None
        ),
        "scheduler_planner_v2_candidate_window_membership_hash": (
            _hash_index_tensor(candidate_window_sorted) if candidate_window.numel() > 0 else None
        ),
        "scheduler_planner_v2_candidate_window_includes_reference": bool(includes_reference),
    }
    return candidate_window, telemetry


def _phase4_planner_v2_group_key(
    feature_idx: int,
    *,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> tuple[int, int]:
    layer_value = int(feat_layers[feature_idx].item())
    use_chunk_key = bool(exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0)
    if use_chunk_key:
        chunk_value = int(feat_ids[feature_idx].item()) // int(decoder_chunk_size)
    else:
        chunk_value = -1
    return layer_value, chunk_value


def _select_phase4_planner_v2_membership(
    *,
    unvisited_feature_rank: torch.Tensor,
    reference_frontier: torch.Tensor,
    reference_frontier_size: int,
    candidate_window: torch.Tensor,
    candidate_scores: torch.Tensor,
    visited: torch.Tensor,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
    locked_prefix_fraction: float = _PHASE4_PLANNER_V2_LOCKED_PREFIX_FRACTION,
    max_replacement_fraction: float = _PHASE4_PLANNER_V2_MAX_REPLACEMENT_FRACTION,
    min_score_ratio: float = _PHASE4_PLANNER_V2_MIN_SCORE_RATIO,
) -> tuple[torch.Tensor, dict[str, object]]:
    ranked = unvisited_feature_rank.detach().to(device="cpu", dtype=torch.long)
    reference = reference_frontier.detach().to(device="cpu", dtype=torch.long)
    window = candidate_window.detach().to(device="cpu", dtype=torch.long)
    scores = candidate_scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    visited_cpu = visited.detach().to(device="cpu", dtype=torch.bool).flatten()

    available_count = int(ranked.numel())
    reference_size = min(
        int(reference_frontier_size),
        int(reference.numel()),
        available_count,
    )
    locked_fraction = min(max(0.0, float(locked_prefix_fraction)), 1.0)
    replacement_fraction = max(0.0, float(max_replacement_fraction))
    required_score_ratio = min(max(0.0, float(min_score_ratio)), 1.0)
    locked_prefix_size = min(reference_size, int(math.floor(reference_size * locked_fraction)))
    max_replacement_count = int(math.ceil(reference_size * replacement_fraction))

    telemetry: dict[str, object] = {
        "scheduler_planner_v2_selection_attempted": True,
        "scheduler_planner_v2_selection_applied": True,
        "scheduler_planner_v2_selection_changed_membership": False,
        "scheduler_planner_v2_fallback_to_reference": False,
        "scheduler_planner_v2_fallback_reason": None,
        "scheduler_planner_v2_reference_membership_hash": None,
        "scheduler_planner_v2_selected_membership_hash": None,
        "scheduler_planner_v2_locked_prefix_membership_hash": None,
        "scheduler_planner_v2_replacement_count": 0,
        "scheduler_planner_v2_replacement_fraction_realized": 0.0,
        "scheduler_planner_v2_reference_score_sum": None,
        "scheduler_planner_v2_selected_score_sum": None,
        "scheduler_planner_v2_selected_score_ratio": None,
        "scheduler_planner_v2_reference_group_count": 0,
        "scheduler_planner_v2_selected_group_count": 0,
        "scheduler_planner_v2_group_count_delta": 0,
        "scheduler_planner_v2_rank_displacement_sum": 0,
    }

    def _fallback(reason: str) -> tuple[torch.Tensor, dict[str, object]]:
        telemetry["scheduler_planner_v2_selection_applied"] = False
        telemetry["scheduler_planner_v2_selection_changed_membership"] = False
        telemetry["scheduler_planner_v2_fallback_to_reference"] = True
        telemetry["scheduler_planner_v2_fallback_reason"] = reason
        telemetry["scheduler_planner_v2_replacement_count"] = 0
        telemetry["scheduler_planner_v2_replacement_fraction_realized"] = 0.0
        reference_ranked = reference[:reference_size]
        reference_ranked_sorted = (
            torch.sort(reference_ranked).values
            if reference_ranked.numel() > 0
            else reference_ranked
        )
        telemetry["scheduler_planner_v2_selected_membership_hash"] = (
            _hash_index_tensor(reference_ranked_sorted)
            if reference_ranked_sorted.numel() > 0
            else None
        )
        return reference_ranked, telemetry

    if reference_size <= 0:
        return torch.empty(0, dtype=torch.long), telemetry

    if scores.numel() < available_count:
        return _fallback("score_metrics_unavailable")

    rank_lookup: dict[int, int] = {}
    score_lookup: dict[int, float] = {}
    for rank_idx, node_idx in enumerate(ranked.tolist()):
        node_int = int(node_idx)
        rank_lookup[node_int] = int(rank_idx)
        score_value = float(scores[rank_idx].item())
        if math.isfinite(score_value):
            score_lookup[node_int] = score_value

    reference_nodes = [int(value) for value in reference[:reference_size].tolist()]
    if len(reference_nodes) != len(set(reference_nodes)):
        return _fallback("reference_contains_duplicates")

    for node_idx in reference_nodes:
        if node_idx >= int(visited_cpu.numel()):
            return _fallback("reference_index_out_of_range")
        if bool(visited_cpu[node_idx].item()):
            return _fallback("reference_contains_visited_feature")
        if node_idx not in rank_lookup or node_idx not in score_lookup:
            return _fallback("score_metrics_unavailable")

    reference_ranked_nodes = sorted(reference_nodes, key=lambda node: (rank_lookup[node], node))
    locked_nodes = reference_ranked_nodes[:locked_prefix_size]
    locked_node_set = set(locked_nodes)
    unlocked_reference_nodes = [
        node for node in reference_ranked_nodes if node not in locked_node_set
    ]
    reference_set = set(reference_ranked_nodes)

    reference_score_sum = float(sum(score_lookup[node] for node in reference_ranked_nodes))
    if not math.isfinite(reference_score_sum) or reference_score_sum <= 0.0:
        return _fallback("score_metrics_unavailable")

    telemetry["scheduler_planner_v2_reference_score_sum"] = reference_score_sum
    telemetry["scheduler_planner_v2_reference_membership_hash"] = _hash_index_tensor(
        torch.sort(torch.tensor(reference_ranked_nodes, dtype=torch.long)).values
    )
    telemetry["scheduler_planner_v2_locked_prefix_membership_hash"] = (
        _hash_index_tensor(torch.sort(torch.tensor(locked_nodes, dtype=torch.long)).values)
        if locked_nodes
        else None
    )

    reference_group_counts: dict[tuple[int, int], int] = {}
    for node in reference_ranked_nodes:
        group_key = _phase4_planner_v2_group_key(
            node,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        reference_group_counts[group_key] = reference_group_counts.get(group_key, 0) + 1

    telemetry["scheduler_planner_v2_reference_group_count"] = int(len(reference_group_counts))
    locked_group_set = {
        _phase4_planner_v2_group_key(
            node,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        for node in locked_nodes
    }

    outsider_entries: list[dict[str, object]] = []
    seen_outsiders: set[int] = set()
    for node_idx in window.tolist():
        node = int(node_idx)
        if node in reference_set or node in seen_outsiders:
            continue
        seen_outsiders.add(node)
        if node >= int(visited_cpu.numel()):
            return _fallback("candidate_window_index_out_of_range")
        if bool(visited_cpu[node].item()):
            return _fallback("candidate_window_contains_visited_feature")
        rank_value = rank_lookup.get(node)
        score_value = score_lookup.get(node)
        if rank_value is None or score_value is None:
            continue
        group_key = _phase4_planner_v2_group_key(
            node,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        outsider_entries.append(
            {
                "node": node,
                "rank": rank_value,
                "score": score_value,
                "group": group_key,
                "locked_group": int(group_key in locked_group_set),
                "reference_group_count": int(reference_group_counts.get(group_key, 0)),
            }
        )

    outsider_entries.sort(
        key=lambda item: (
            int(item["locked_group"]),
            int((item["reference_group_count"] or 0) > 0),
            int(item["reference_group_count"]),
            float(item["score"]),
            -int(item["rank"]),
            -int(item["node"]),
        ),
        reverse=True,
    )

    removable_entries: list[dict[str, object]] = []
    for node in unlocked_reference_nodes:
        group_key = _phase4_planner_v2_group_key(
            node,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        removable_entries.append(
            {
                "node": node,
                "rank": int(rank_lookup[node]),
                "score": float(score_lookup[node]),
                "group": group_key,
                "reference_group_count": int(reference_group_counts.get(group_key, 0)),
            }
        )

    removable_entries.sort(
        key=lambda item: (
            int(item["reference_group_count"]),
            float(item["score"]),
            -int(item["rank"]),
            int(item["node"]),
        )
    )

    max_k = min(max_replacement_count, len(outsider_entries), len(removable_entries))
    score_ratio_rejected = False
    best_candidate: dict[str, object] | None = None
    reference_rank_sum = int(sum(rank_lookup[node] for node in reference_ranked_nodes))

    for replacement_count in range(1, max_k + 1):
        dropped_nodes = {int(item["node"]) for item in removable_entries[:replacement_count]}
        added_nodes = [int(item["node"]) for item in outsider_entries[:replacement_count]]
        candidate_nodes = [
            node for node in reference_ranked_nodes if node not in dropped_nodes
        ] + added_nodes

        if len(candidate_nodes) != reference_size:
            continue
        if len(set(candidate_nodes)) != reference_size:
            continue

        candidate_ranked_nodes = sorted(candidate_nodes, key=lambda node: (rank_lookup[node], node))
        candidate_score_sum = float(sum(score_lookup[node] for node in candidate_ranked_nodes))
        score_ratio = candidate_score_sum / reference_score_sum
        if (not math.isfinite(score_ratio)) or score_ratio < required_score_ratio:
            score_ratio_rejected = True
            continue

        candidate_group_count = len(
            {
                _phase4_planner_v2_group_key(
                    node,
                    feat_layers=feat_layers,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                for node in candidate_ranked_nodes
            }
        )
        group_delta = int(len(reference_group_counts) - candidate_group_count)
        if group_delta <= 0:
            continue

        candidate_rank_sum = int(sum(rank_lookup[node] for node in candidate_ranked_nodes))
        rank_displacement_sum = int(candidate_rank_sum - reference_rank_sum)
        objective = (
            int(group_delta),
            int(-rank_displacement_sum),
            float(score_ratio),
            int(replacement_count),
        )
        if best_candidate is None or objective > cast(
            tuple[int, int, float, int], best_candidate["objective"]
        ):
            best_candidate = {
                "nodes": candidate_ranked_nodes,
                "score_sum": candidate_score_sum,
                "score_ratio": float(score_ratio),
                "group_count": int(candidate_group_count),
                "group_delta": int(group_delta),
                "replacement_count": int(replacement_count),
                "rank_displacement_sum": int(rank_displacement_sum),
                "objective": objective,
            }

    if best_candidate is None:
        reference_ranked_tensor = torch.tensor(reference_ranked_nodes, dtype=torch.long)
        reference_sorted_tensor = torch.sort(reference_ranked_tensor).values
        telemetry["scheduler_planner_v2_selected_membership_hash"] = _hash_index_tensor(
            reference_sorted_tensor
        )
        telemetry["scheduler_planner_v2_selected_score_sum"] = reference_score_sum
        telemetry["scheduler_planner_v2_selected_score_ratio"] = 1.0
        telemetry["scheduler_planner_v2_selected_group_count"] = int(len(reference_group_counts))
        telemetry["scheduler_planner_v2_group_count_delta"] = 0
        telemetry["scheduler_planner_v2_rank_displacement_sum"] = 0
        if max_k > 0 and score_ratio_rejected:
            return _fallback("score_ratio_below_threshold")
        return reference_ranked_tensor, telemetry

    selected_nodes = cast(list[int], best_candidate["nodes"])
    selected_tensor = torch.tensor(selected_nodes, dtype=torch.long)
    selected_sorted = torch.sort(selected_tensor).values

    if selected_tensor.numel() != reference_size:
        return _fallback("selected_count_mismatch")
    if int(torch.unique(selected_tensor).numel()) != reference_size:
        return _fallback("selected_membership_not_unique")
    if not set(locked_nodes).issubset(set(selected_nodes)):
        return _fallback("locked_prefix_not_preserved")
    if bool(visited_cpu[selected_tensor].any().item()):
        return _fallback("selected_membership_contains_visited_feature")

    selected_score_ratio = float(best_candidate["score_ratio"])
    if (not math.isfinite(selected_score_ratio)) or selected_score_ratio < required_score_ratio:
        return _fallback("score_ratio_below_threshold")

    replacement_count = int(best_candidate["replacement_count"])
    if replacement_count > max_replacement_count:
        return _fallback("replacement_fraction_exceeded")

    telemetry["scheduler_planner_v2_selection_changed_membership"] = True
    telemetry["scheduler_planner_v2_selected_membership_hash"] = _hash_index_tensor(selected_sorted)
    telemetry["scheduler_planner_v2_replacement_count"] = replacement_count
    telemetry["scheduler_planner_v2_replacement_fraction_realized"] = (
        float(replacement_count / reference_size) if reference_size > 0 else 0.0
    )
    telemetry["scheduler_planner_v2_selected_score_sum"] = float(best_candidate["score_sum"])
    telemetry["scheduler_planner_v2_selected_score_ratio"] = selected_score_ratio
    telemetry["scheduler_planner_v2_selected_group_count"] = int(best_candidate["group_count"])
    telemetry["scheduler_planner_v2_group_count_delta"] = int(best_candidate["group_delta"])
    telemetry["scheduler_planner_v2_rank_displacement_sum"] = int(
        best_candidate["rank_displacement_sum"]
    )

    return selected_tensor, telemetry


def _apply_phase4_planner_v2_refresh_plan(
    *,
    reference_plan: _Phase4FrontierPlan,
    unvisited_feature_rank: torch.Tensor,
    candidate_scores: torch.Tensor,
    visited: torch.Tensor,
    max_batch_size: int,
    max_batches: int | None,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> tuple[_Phase4FrontierPlan, torch.Tensor, dict[str, object]]:
    reference_frontier = reference_plan.selected_frontier
    reference_size = int(reference_frontier.numel())

    candidate_window = torch.empty(0, dtype=torch.long)
    telemetry = _build_phase4_planner_v2_refresh_telemetry_disabled()
    selected_membership = reference_frontier.detach().to(device="cpu", dtype=torch.long)
    try:
        candidate_window, telemetry = _build_phase4_planner_v2_candidate_window(
            unvisited_feature_rank,
            reference_frontier=reference_frontier,
            reference_frontier_size=reference_size,
            candidate_scores=candidate_scores,
        )
        selected_membership, selection_telemetry = _select_phase4_planner_v2_membership(
            unvisited_feature_rank=unvisited_feature_rank,
            reference_frontier=reference_frontier,
            reference_frontier_size=reference_size,
            candidate_window=candidate_window,
            candidate_scores=candidate_scores,
            visited=visited,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        telemetry.update(selection_telemetry)
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        reference_sorted = (
            torch.sort(selected_membership).values
            if selected_membership.numel() > 0
            else selected_membership
        )
        telemetry.update(
            {
                "scheduler_planner_v2_enabled": True,
                "scheduler_planner_v2_policy_version": _PHASE4_PLANNER_V2_POLICY_VERSION,
                "scheduler_planner_v2_reference_frontier_size": int(reference_size),
                "scheduler_planner_v2_selection_attempted": False,
                "scheduler_planner_v2_selection_applied": False,
                "scheduler_planner_v2_selection_changed_membership": False,
                "scheduler_planner_v2_fallback_to_reference": True,
                "scheduler_planner_v2_fallback_reason": (
                    f"planner_v2_selection_error:{type(exc).__name__}"
                ),
                "scheduler_planner_v2_reference_membership_hash": (
                    _hash_index_tensor(reference_sorted) if reference_sorted.numel() > 0 else None
                ),
                "scheduler_planner_v2_selected_membership_hash": (
                    _hash_index_tensor(reference_sorted) if reference_sorted.numel() > 0 else None
                ),
                "scheduler_planner_v2_replacement_count": 0,
                "scheduler_planner_v2_replacement_fraction_realized": 0.0,
                "scheduler_planner_v2_selected_score_ratio": 1.0,
                "scheduler_planner_v2_group_count_delta": 0,
                "scheduler_planner_v2_rank_displacement_sum": 0,
            }
        )

    fallback_to_reference = bool(telemetry.get("scheduler_planner_v2_fallback_to_reference", False))
    fallback_reason = cast(str | None, telemetry.get("scheduler_planner_v2_fallback_reason"))
    changed_membership = bool(
        telemetry.get("scheduler_planner_v2_selection_changed_membership", False)
    )

    selected_plan = reference_plan
    if (not fallback_to_reference) and changed_membership:
        try:
            candidate_plan = _plan_phase4_frontier_membership_preserving_v1(
                selected_membership,
                max_batch_size=max_batch_size,
                max_batches=max_batches,
                feat_layers=feat_layers,
                feat_positions=feat_positions,
                feat_ids=feat_ids,
                exact_chunked_decoder=exact_chunked_decoder,
                decoder_chunk_size=decoder_chunk_size,
                apply_locality_reorder=True,
            )
            candidate_sorted = torch.sort(
                candidate_plan.selected_frontier.detach().to(device="cpu", dtype=torch.long)
            ).values
            expected_sorted = torch.sort(
                selected_membership.detach().to(device="cpu", dtype=torch.long)
            ).values
            if (
                candidate_plan.selected_frontier.numel() != reference_size
                or candidate_sorted.numel() != expected_sorted.numel()
                or not torch.equal(candidate_sorted, expected_sorted)
            ):
                fallback_to_reference = True
                fallback_reason = "planner_v1_execution_membership_mismatch"
            elif bool(
                visited.detach()
                .to(device="cpu", dtype=torch.bool)
                .flatten()[
                    candidate_plan.selected_frontier.detach().to(device="cpu", dtype=torch.long)
                ]
                .any()
                .item()
            ):
                fallback_to_reference = True
                fallback_reason = "planner_v1_execution_contains_visited_feature"
            else:
                selected_plan = candidate_plan
        except Exception as exc:  # pragma: no cover - defensive fail-closed path
            fallback_to_reference = True
            fallback_reason = f"planner_v1_execution_error:{type(exc).__name__}"

    if fallback_to_reference:
        selected_plan = reference_plan
        telemetry["scheduler_planner_v2_selection_applied"] = False
        telemetry["scheduler_planner_v2_selection_changed_membership"] = False
        telemetry["scheduler_planner_v2_fallback_to_reference"] = True
        telemetry["scheduler_planner_v2_fallback_reason"] = fallback_reason

    planner_v2_invariants: dict[str, object] = {
        "planner_v2_attempted": True,
        "planner_v2_selection_applied": bool(
            telemetry.get("scheduler_planner_v2_selection_applied", False)
        ),
        "planner_v2_changed_membership": bool(
            telemetry.get("scheduler_planner_v2_selection_changed_membership", False)
        ),
        "planner_v2_fallback_to_reference": bool(
            telemetry.get("scheduler_planner_v2_fallback_to_reference", False)
        ),
        "planner_v2_fallback_reason": telemetry.get("scheduler_planner_v2_fallback_reason"),
        "planner_v2_replacement_count": int(
            telemetry.get("scheduler_planner_v2_replacement_count", 0)
        ),
        "planner_v2_selected_score_ratio": _safe_float(
            telemetry.get("scheduler_planner_v2_selected_score_ratio")
        ),
        "planner_v2_group_count_delta": int(
            telemetry.get("scheduler_planner_v2_group_count_delta", 0)
        ),
    }

    selected_plan = _Phase4FrontierPlan(
        selected_frontier=selected_plan.selected_frontier,
        batch_boundaries=selected_plan.batch_boundaries,
        selected_membership_hash=selected_plan.selected_membership_hash,
        selected_order_hash=selected_plan.selected_order_hash,
        locality_fragmentation_summary=selected_plan.locality_fragmentation_summary,
        boundary_reason_counts=selected_plan.boundary_reason_counts,
        invariant_summary={**selected_plan.invariant_summary, **planner_v2_invariants},
    )

    return selected_plan, candidate_window, telemetry


def _build_phase4_batch_locality_summary(
    idx_batch: torch.Tensor,
    *,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> dict[str, object]:
    if idx_batch.numel() <= 0:
        return {
            "scheduler_batch_hash": None,
            "scheduler_batch_distinct_source_layer_count": 0,
            "scheduler_batch_source_layer_min": None,
            "scheduler_batch_source_layer_max": None,
            "scheduler_batch_distinct_decoder_chunk_count": None,
            "scheduler_batch_decoder_chunk_min": None,
            "scheduler_batch_decoder_chunk_max": None,
            "scheduler_batch_monotonic_chunk_order": None,
        }

    layer_values = feat_layers[idx_batch].detach().to(device="cpu", dtype=torch.long)
    distinct_layers = torch.unique(layer_values)
    batch_hash = _hash_index_tensor(idx_batch)

    use_decoder_chunks = bool(
        exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0
    )
    if use_decoder_chunks:
        chunk_values = (
            torch.div(
                feat_ids[idx_batch],
                int(decoder_chunk_size),
                rounding_mode="floor",
            )
            .detach()
            .to(device="cpu", dtype=torch.long)
        )
        distinct_chunks = torch.unique(chunk_values)
        if chunk_values.numel() > 1:
            next_layers = layer_values[1:]
            prev_layers = layer_values[:-1]
            next_chunks = chunk_values[1:]
            prev_chunks = chunk_values[:-1]
            monotonic_chunk_order = bool(
                torch.all(
                    (next_layers > prev_layers)
                    | ((next_layers == prev_layers) & (next_chunks >= prev_chunks))
                ).item()
            )
        else:
            monotonic_chunk_order = True
        distinct_chunk_count = int(distinct_chunks.numel())
        chunk_min = int(chunk_values.min().item())
        chunk_max = int(chunk_values.max().item())
    else:
        monotonic_chunk_order = None
        distinct_chunk_count = None
        chunk_min = None
        chunk_max = None

    return {
        "scheduler_batch_hash": batch_hash,
        "scheduler_batch_distinct_source_layer_count": int(distinct_layers.numel()),
        "scheduler_batch_source_layer_min": int(layer_values.min().item()),
        "scheduler_batch_source_layer_max": int(layer_values.max().item()),
        "scheduler_batch_distinct_decoder_chunk_count": distinct_chunk_count,
        "scheduler_batch_decoder_chunk_min": chunk_min,
        "scheduler_batch_decoder_chunk_max": chunk_max,
        "scheduler_batch_monotonic_chunk_order": monotonic_chunk_order,
    }


def _build_phase4_frontier_locality_fragmentation_summary(
    selected_frontier: torch.Tensor,
    *,
    feat_layers: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
    batch_count: int,
) -> dict[str, object]:
    selected_count = int(selected_frontier.numel())
    if selected_count <= 0:
        return {
            "selected_count": 0,
            "layer_chunk_run_count": 0,
            "layer_chunk_transition_count": 0,
            "layer_chunk_fragmentation_ratio": 0.0,
            "batch_count": int(batch_count),
            "batch_fragmentation_ratio": 0.0,
        }

    layers = feat_layers[selected_frontier]
    use_chunk_key = bool(exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0)
    if use_chunk_key:
        chunks = torch.div(
            feat_ids[selected_frontier],
            int(decoder_chunk_size),
            rounding_mode="floor",
        )
    else:
        chunks = torch.zeros_like(layers)

    transitions = (layers[1:] != layers[:-1]) | (chunks[1:] != chunks[:-1])
    transition_count = int(transitions.sum().item()) if transitions.numel() > 0 else 0
    run_count = 1 + transition_count

    return {
        "selected_count": selected_count,
        "layer_chunk_run_count": int(run_count),
        "layer_chunk_transition_count": int(transition_count),
        "layer_chunk_fragmentation_ratio": float(transition_count / max(1, selected_count - 1)),
        "batch_count": int(batch_count),
        "batch_fragmentation_ratio": float(batch_count / max(1, run_count)),
    }


def _plan_phase4_frontier_membership_preserving_v1(
    pending_candidates: torch.Tensor,
    *,
    max_batch_size: int,
    max_batches: int | None,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
    apply_locality_reorder: bool = True,
) -> _Phase4FrontierPlan:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be > 0 when provided")

    if apply_locality_reorder:
        planned_candidates = _reorder_pending_for_phase4_locality(
            pending_candidates,
            feat_layers=feat_layers,
            feat_positions=feat_positions,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
    else:
        planned_candidates = pending_candidates

    if max_batches is None:
        selected_count = int(planned_candidates.numel())
    else:
        selected_count = _compute_phase4_locality_shaped_frontier_size(
            planned_candidates,
            max_batch_size=max_batch_size,
            max_batches=max_batches,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
    selected_frontier = planned_candidates[:selected_count]

    if apply_locality_reorder:
        expected_candidates = _reorder_pending_for_phase4_locality(
            pending_candidates,
            feat_layers=feat_layers,
            feat_positions=feat_positions,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
    else:
        expected_candidates = pending_candidates
    if max_batches is None:
        expected_selected = expected_candidates
    else:
        expected_count = _compute_phase4_locality_shaped_frontier_size(
            expected_candidates,
            max_batch_size=max_batch_size,
            max_batches=max_batches,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        expected_selected = expected_candidates[:expected_count]
    expected_sorted = torch.sort(
        expected_selected.detach().to(device="cpu", dtype=torch.long)
    ).values
    selected_sorted = torch.sort(
        selected_frontier.detach().to(device="cpu", dtype=torch.long)
    ).values
    expected_set = set(expected_sorted.tolist())
    selected_set = set(selected_sorted.tolist())
    missing_count = int(len(expected_set - selected_set))
    unexpected_count = int(len(selected_set - expected_set))
    duplicate_count = int(selected_frontier.numel() - torch.unique(selected_frontier).numel())
    if duplicate_count > 0:
        raise RuntimeError(
            "Planner v1 selected frontier contains duplicate nodes "
            f"(duplicate_count={duplicate_count})"
        )
    if selected_frontier.numel() != expected_selected.numel() or not torch.equal(
        selected_sorted,
        expected_sorted,
    ):
        raise RuntimeError(
            "Planner v1 selected frontier membership mismatch against locality semantics "
            f"(missing={missing_count}, unexpected={unexpected_count})"
        )

    batch_boundaries: list[tuple[int, int]] = []
    boundary_reason_counts: dict[str, int] = {}
    pending_offset = 0
    while pending_offset < int(selected_frontier.numel()):
        batch_end, boundary_reason = _compute_phase4_locality_shaped_batch_end_with_reason(
            selected_frontier,
            pending_offset=pending_offset,
            max_batch_size=max_batch_size,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        boundary_reason_counts[boundary_reason] = boundary_reason_counts.get(boundary_reason, 0) + 1
        if batch_end <= pending_offset:
            raise RuntimeError(
                "Planner v1 produced a non-advancing batch boundary "
                f"(offset={pending_offset}, batch_end={batch_end})"
            )
        batch_boundaries.append((pending_offset, batch_end))
        pending_offset = batch_end

    selected_membership_hash = (
        _hash_index_tensor(selected_sorted) if selected_frontier.numel() > 0 else None
    )
    selected_order_hash = (
        _hash_index_tensor(selected_frontier) if selected_frontier.numel() > 0 else None
    )
    membership_preserved = bool(
        duplicate_count == 0
        and missing_count == 0
        and unexpected_count == 0
        and selected_frontier.numel() == expected_selected.numel()
    )
    invariant_summary: dict[str, object] = {
        "candidate_count": int(pending_candidates.numel()),
        "selected_count": int(selected_frontier.numel()),
        "batch_count": int(len(batch_boundaries)),
        "membership_preserved": membership_preserved,
        "duplicate_count": int(duplicate_count),
        "missing_count": int(missing_count),
        "unexpected_count": int(unexpected_count),
        "non_advancing_boundary_count": 0,
    }

    return _Phase4FrontierPlan(
        selected_frontier=selected_frontier,
        batch_boundaries=batch_boundaries,
        selected_membership_hash=selected_membership_hash,
        selected_order_hash=selected_order_hash,
        locality_fragmentation_summary=_build_phase4_frontier_locality_fragmentation_summary(
            selected_frontier,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
            batch_count=len(batch_boundaries),
        ),
        boundary_reason_counts=boundary_reason_counts,
        invariant_summary=invariant_summary,
    )


def _resolve_phase4_feature_batch_planner_enabled(
    *,
    plan_feature_batch_size: bool,
    auto_scale_feature_batch_size: bool,
) -> bool:
    # Backward compatibility: keep legacy flag as an alias for fixed preflight planning.
    return bool(plan_feature_batch_size or auto_scale_feature_batch_size)


def _resolve_phase4_feature_batch_planner_status(
    *,
    planner_enabled: bool,
    effective_feature_batch_size: int,
    max_feature_batch_size: int,
) -> tuple[str, str | None]:
    if not planner_enabled:
        return "disabled", None
    if max_feature_batch_size <= effective_feature_batch_size:
        return (
            "skipped_no_headroom",
            "feature_batch_size_max does not exceed initial feature_batch_size",
        )
    return "pending", None


def _compute_phase4_planned_feature_batch_size(
    observed_feature_batch_size: int,
    *,
    max_feature_batch_size: int,
    observed_reserved_bytes: int | None,
    total_cuda_bytes: int | None,
    target_reserved_fraction: float,
    min_free_fraction: float,
) -> int:
    """Compute a fixed Phase-4 feature batch size from probe telemetry."""

    if observed_feature_batch_size <= 0:
        raise ValueError("observed_feature_batch_size must be > 0")
    if max_feature_batch_size <= 0:
        raise ValueError("max_feature_batch_size must be > 0")
    if not 0.0 < target_reserved_fraction < 1.0:
        raise ValueError("target_reserved_fraction must be in (0, 1)")
    if not 0.0 <= min_free_fraction < 1.0:
        raise ValueError("min_free_fraction must be in [0, 1)")

    baseline = min(observed_feature_batch_size, max_feature_batch_size)
    if observed_reserved_bytes is None or total_cuda_bytes is None or total_cuda_bytes <= 0:
        return baseline
    if observed_reserved_bytes <= 0:
        return baseline

    observed_reserved_fraction = observed_reserved_bytes / total_cuda_bytes
    if observed_reserved_fraction <= 0:
        return baseline

    reserved_budget_fraction = min(target_reserved_fraction, 1.0 - min_free_fraction)
    if reserved_budget_fraction <= 0:
        return 1

    scaled_batch_size = int(
        math.floor(
            observed_feature_batch_size * reserved_budget_fraction / observed_reserved_fraction
        )
    )
    if scaled_batch_size < 1:
        scaled_batch_size = 1
    return min(max_feature_batch_size, scaled_batch_size)


def _build_phase4_probe_pending_frontier(
    *,
    feature_influences: torch.Tensor | None,
    total_active_feats: int,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
    initial_feature_batch_size: int,
    feature_batch_probe_batches: int,
    update_interval: int,
    max_feature_nodes: int | None,
) -> torch.Tensor:
    """Build a representative fixed Phase-4 frontier for preflight probes."""

    actual_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    if actual_max_feature_nodes <= 0:
        return torch.empty(0, dtype=torch.long)

    if feature_influences is None or actual_max_feature_nodes == total_active_feats:
        pending = torch.arange(total_active_feats)
        probe_frontier_size = _compute_phase4_locality_shaped_frontier_size(
            pending,
            max_batch_size=initial_feature_batch_size,
            max_batches=feature_batch_probe_batches,
            feat_layers=feat_layers,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
        )
        return pending[:probe_frontier_size]

    feature_rank = torch.argsort(feature_influences, descending=True).cpu()
    queue_size = min(update_interval * initial_feature_batch_size, actual_max_feature_nodes)
    pending = feature_rank[:queue_size]

    pending = _reorder_pending_for_phase4_locality(
        pending,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=exact_chunked_decoder,
        decoder_chunk_size=decoder_chunk_size,
    )

    probe_frontier_size = _compute_phase4_locality_shaped_frontier_size(
        pending,
        max_batch_size=initial_feature_batch_size,
        max_batches=feature_batch_probe_batches,
        feat_layers=feat_layers,
        feat_ids=feat_ids,
        exact_chunked_decoder=exact_chunked_decoder,
        decoder_chunk_size=decoder_chunk_size,
    )
    return pending[:probe_frontier_size]


def _plan_phase4_feature_batch_size_preflight(
    *,
    model: NNSightReplacementModel,
    prompt,
    attribution_targets,
    batch_size: int,
    initial_feature_batch_size: int,
    effective_logit_batch_size: int,
    max_feature_batch_size: int,
    max_feature_nodes: int | None,
    update_interval: int,
    max_n_logits: int,
    desired_logit_prob: float,
    exact_trace_internal_dtype: torch.dtype,
    logger,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    exact_encoder_residency: Literal["lazy", "active_cpu"] = "lazy",
    diagnostic_feature_cap: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    internal_precision_requested: str = "float64",
    resolved_dtype_map: dict[str, str] | None = None,
    row_abs_sum_dtype: torch.dtype = torch.float64,
    planner_compute_dtype: torch.dtype = torch.float64,
    trace_observer: TraceObserver | None = None,
    prefix_view_metadata: PrefixViewMetadata | None = None,
) -> int:
    # Runtime import avoids the phase_support -> phase4_policy module cycle.
    from circuit_tracer.attribution.nnsight.phase_support import (
        _copy_rows_to_cpu_staging,
    )

    planner_start = time.perf_counter()
    exact_trace_internal_dtype_name = _exact_trace_internal_dtype_name(exact_trace_internal_dtype)

    def _finalize_planner(
        *,
        planned_feature_batch_size: int,
        planner_status: str,
        attrs: dict[str, object] | None = None,
    ) -> int:
        if trace_observer is not None:
            payload = {
                "planner_status": planner_status,
                "planned_feature_batch_size": planned_feature_batch_size,
            }
            if attrs:
                payload.update(attrs)
            trace_observer.observe(
                TraceEvent(
                    scope="phase",
                    name="phase4.planner.preflight",
                    phase="phase4",
                    elapsed_ms=(time.perf_counter() - planner_start) * 1000.0,
                    attrs=payload,
                )
            )
        return planned_feature_batch_size

    cuda_snapshot = probe_cuda_memory(CudaMemoryProbe("snapshot"))
    if not cuda_snapshot.available:
        logger.info(
            "Phase 4 planner skipped (CUDA unavailable); using fixed feature batch size "
            f"{min(initial_feature_batch_size, max_feature_batch_size)}"
        )
        return _finalize_planner(
            planned_feature_batch_size=min(initial_feature_batch_size, max_feature_batch_size),
            planner_status="skipped_cuda_unavailable",
        )

    input_ids = model.ensure_tokenized(prompt)
    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(
        input_ids, prefix_view_metadata
    )
    ctx = None
    observed_peak_reserved_bytes = 0
    total_cuda_bytes: int | None = None

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging) and trace_observer is not None:
        configure_trace_logging(None, trace_observer=trace_observer)

    try:
        logger.info(
            "Phase 4 planner preflight | "
            f"initial_feature_batch_size={initial_feature_batch_size} | "
            f"max_feature_batch_size={max_feature_batch_size} | "
            f"max_feature_nodes={max_feature_nodes} | "
            f"update_interval={update_interval} | "
            f"probe_batches={feature_batch_probe_batches} | "
            f"exact_trace_internal_dtype={exact_trace_internal_dtype_name} | "
            f"target_reserved_fraction={feature_batch_target_reserved_fraction:.3f} | "
            f"min_free_fraction={feature_batch_min_free_fraction:.3f}"
        )

        ctx = model.setup_attribution(
            input_ids,
            sparsification=sparsification,
            retain_full_logits=False,
            chunked_feature_replay_window=chunked_feature_replay_window,
            error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
            row_subchunk_size=row_subchunk_size,
            exact_encoder_residency=exact_encoder_residency,
            internal_precision_requested=internal_precision_requested,
            resolved_dtype_map=resolved_dtype_map,
            prefix_view_length=prefix_view_length,
            trace_observer=trace_observer,
        )
        if hasattr(ctx, "set_diagnostic_mode"):
            ctx.set_diagnostic_mode(False)
        configure_ctx_trace_logging = getattr(ctx, "configure_trace_logging", None)
        if callable(configure_ctx_trace_logging) and trace_observer is not None:
            configure_ctx_trace_logging(None, trace_observer=trace_observer)

        if diagnostic_feature_cap is not None and diagnostic_feature_cap > 0:
            ctx.apply_diagnostic_feature_cap(diagnostic_feature_cap)

        activation_matrix = ctx.activation_matrix
        total_active_feats = int(activation_matrix._nnz())
        if total_active_feats <= 0:
            logger.info(
                "Phase 4 planner preflight observed no active features; "
                f"using feature batch size {min(initial_feature_batch_size, max_feature_batch_size)}"
            )
            return _finalize_planner(
                planned_feature_batch_size=min(initial_feature_batch_size, max_feature_batch_size),
                planner_status="skipped_no_active_features",
            )

        feat_layers, feat_pos, feat_ids = activation_matrix.indices()
        n_layers, n_pos, _ = activation_matrix.shape
        logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
        trace_batch_size = max(batch_size, initial_feature_batch_size, effective_logit_batch_size)

        with model.trace() as tracer:
            with tracer.invoke(
                trace_input_ids.expand(trace_batch_size, -1),
                **ctx.resolve_phase1_invoke_kwargs(model),
            ):
                pass

            detach_barrier = tracer.barrier(2)
            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            ctx.cache_residual(model, tracer, barrier=detach_barrier)

        exact_chunked_decoder = require_exact_chunked_provider(model.transcoders)
        decoder_chunk_size = getattr(model.transcoders, "decoder_chunk_size", None)

        # Build probe candidates using the same Phase-3 attribution targets and
        # first Phase-4 frontier ranking semantics used by the real run.
        feature_influences: torch.Tensor | None = None
        targets = AttributionTargets(
            attribution_targets=attribution_targets,
            logits=ctx.get_last_token_logits()[0],
            unembed_proj=cast(torch.Tensor, model.unembed_weight),
            tokenizer=model.tokenizer,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )
        n_logits = len(targets)
        if n_logits > 0 and total_active_feats > 0:
            logit_feature_rows = torch.zeros(
                (n_logits, total_active_feats),
                dtype=exact_trace_internal_dtype,
            )
            logit_row_abs_max = torch.zeros(n_logits, dtype=exact_trace_internal_dtype)
            logit_row_l1_scaled = torch.zeros(n_logits, dtype=exact_trace_internal_dtype)
            row_to_node_index = torch.arange(n_logits, dtype=torch.long) + int(logit_offset)
            rows_cpu_staging: torch.Tensor | None = None
            for i in range(0, n_logits, effective_logit_batch_size):
                batch = targets.logit_vectors[i : i + effective_logit_batch_size]
                rows = ctx.compute_batch(
                    layers=torch.full((batch.shape[0],), n_layers),
                    positions=torch.full((batch.shape[0],), n_pos - 1),
                    inject_values=batch,
                    retain_graph=True,
                    phase_label="phase3_logits_probe",
                )
                rows_cpu, rows_cpu_staging = _copy_rows_to_cpu_staging(
                    rows,
                    staging_buffer=rows_cpu_staging,
                    dtype=exact_trace_internal_dtype,
                )
                end = i + batch.shape[0]
                logit_feature_rows[i:end] = rows_cpu[:, :total_active_feats]
                row_abs_max_chunk, row_l1_scaled_chunk = _compute_row_denominator_scaled_l1(
                    rows_cpu[:, :logit_offset],
                    dtype=exact_trace_internal_dtype,
                )
                logit_row_abs_max[i:end] = row_abs_max_chunk
                logit_row_l1_scaled[i:end] = row_l1_scaled_chunk

            feature_influences = compute_partial_feature_influences(
                logit_feature_rows,
                (logit_row_abs_max, logit_row_l1_scaled),
                targets.logit_probabilities.detach().cpu().to(dtype=exact_trace_internal_dtype),
                row_to_node_index,
                n_feature_nodes=total_active_feats,
                n_logits=n_logits,
                device=logit_feature_rows.device,
            )

        reset_decoder_cache = getattr(ctx, "reset_decoder_cache", None)
        if callable(reset_decoder_cache):
            reset_decoder_cache()

        pending = _build_phase4_probe_pending_frontier(
            feature_influences=feature_influences,
            total_active_feats=total_active_feats,
            feat_layers=feat_layers,
            feat_positions=feat_pos,
            feat_ids=feat_ids,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
            initial_feature_batch_size=initial_feature_batch_size,
            feature_batch_probe_batches=feature_batch_probe_batches,
            update_interval=update_interval,
            max_feature_nodes=max_feature_nodes,
        )

        pending_offset = 0
        probe_batches_ran = 0
        observed_feature_batch_size = 0
        for probe_idx in range(feature_batch_probe_batches):
            idx_batch = pending[pending_offset : pending_offset + initial_feature_batch_size]
            if idx_batch.numel() == 0:
                break
            pending_offset += int(idx_batch.numel())
            observed_feature_batch_size = max(observed_feature_batch_size, int(idx_batch.numel()))

            probe_cuda_memory(CudaMemoryProbe("reset_peak"))
            probe_batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.materialize_encoder_vectors(idx_batch),
                retain_graph=(probe_idx + 1) < feature_batch_probe_batches,
                phase_label="phase4_probe",
            )
            del rows
            probe_snapshot = probe_cuda_memory(CudaMemoryProbe("synchronize"))
            observed_batch_peak_reserved_bytes = int(probe_snapshot.peak_reserved_bytes or 0)
            observed_peak_reserved_bytes = max(
                observed_peak_reserved_bytes,
                observed_batch_peak_reserved_bytes,
            )
            probe_batches_ran += 1
            if trace_observer is not None:
                trace_observer.observe(
                    TraceEvent(
                        scope="batch",
                        name="phase4.planner.probe_batch",
                        phase="phase4",
                        batch_index=probe_batches_ran,
                        elapsed_ms=(time.perf_counter() - probe_batch_start) * 1000.0,
                        attrs={
                            "batch_nodes": int(idx_batch.numel()),
                            "observed_peak_reserved_bytes": observed_batch_peak_reserved_bytes,
                        },
                    )
                )

        if probe_batches_ran <= 0 or observed_feature_batch_size <= 0:
            logger.info(
                "Phase 4 planner preflight observed no representative probe batches; "
                f"using feature batch size {min(initial_feature_batch_size, max_feature_batch_size)}"
            )
            return _finalize_planner(
                planned_feature_batch_size=min(initial_feature_batch_size, max_feature_batch_size),
                planner_status="skipped_no_probe_batches",
            )

        cuda_snapshot = probe_cuda_memory(CudaMemoryProbe("snapshot"))
        observed_reserved_bytes: int | None = None
        if cuda_snapshot.available:
            total_cuda_bytes = cuda_snapshot.total_bytes
            observed_reserved_bytes = max(
                int(cuda_snapshot.current_reserved_bytes or 0), observed_peak_reserved_bytes
            )

        planned_feature_batch_size = _compute_phase4_planned_feature_batch_size(
            observed_feature_batch_size,
            max_feature_batch_size=max_feature_batch_size,
            observed_reserved_bytes=observed_reserved_bytes,
            total_cuda_bytes=total_cuda_bytes,
            target_reserved_fraction=feature_batch_target_reserved_fraction,
            min_free_fraction=feature_batch_min_free_fraction,
        )

        planned_reserved_fraction = (
            None
            if observed_reserved_bytes is None or total_cuda_bytes in (None, 0)
            else observed_reserved_bytes / total_cuda_bytes
        )
        logger.info(
            "Phase 4 planner result | "
            f"probes_ran={probe_batches_ran} | "
            f"observed_probe_feature_batch_size={observed_feature_batch_size} | "
            f"probe_frontier_candidates={int(pending.numel())} | "
            f"probe_reserved_fraction={planned_reserved_fraction if planned_reserved_fraction is not None else 'n/a'} | "
            f"planned_feature_batch_size={planned_feature_batch_size}"
        )
        return _finalize_planner(
            planned_feature_batch_size=planned_feature_batch_size,
            planner_status="executed",
            attrs={
                "probes_ran": probe_batches_ran,
                "observed_probe_feature_batch_size": observed_feature_batch_size,
                "probe_frontier_candidates": int(pending.numel()),
                "probe_reserved_fraction": planned_reserved_fraction,
            },
        )
    finally:
        if ctx is not None:
            cleanup = getattr(ctx, "cleanup", None)
            if callable(cleanup):
                cleanup()
            else:
                clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
                if callable(clear_decoder_cache):
                    clear_decoder_cache()
