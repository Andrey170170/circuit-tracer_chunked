"""Shared execution support for NNSight attribution phases."""

import os
import warnings

import numpy as np
import torch

from circuit_tracer.attribution.nnsight.numerics import _row_abs_sums_to_scaled_l1
from circuit_tracer.attribution.nnsight.phase4_policy import _reorder_pending_for_phase4_locality
from circuit_tracer.attribution.nnsight.replay import (
    _compute_row_denominator_scaled_l1,
    _hash_index_tensor,
)

_NNSIGHT_BACKEND_FILE = __file__.replace("phase_support.py", "backend.py")

def _resolve_internal_precision_requested(
    internal_precision: str | None,
    *,
    exact_trace_internal_dtype: torch.dtype = torch.float32,
) -> str:
    if internal_precision is None:
        return _dtype_to_name(exact_trace_internal_dtype)

    normalized = str(internal_precision).strip().lower()
    if normalized not in {"float32", "float64"}:
        raise ValueError("internal_precision must be one of {'float32', 'float64'}")
    return normalized


def _warn_internal_precision_deprecated() -> None:
    warnings.warn(
        "internal_precision is deprecated; use exact_trace_internal_dtype instead. "
        "When internal_precision is omitted, it is derived from exact_trace_internal_dtype.",
        DeprecationWarning,
        stacklevel=3,
    )


def _dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype for precision contract: {dtype}")


def _resolve_internal_dtype_map(
    *,
    internal_precision_requested: str,
    phase4_anomaly_debug_enabled: bool,
) -> dict[str, str]:
    """Resolve auditable dtype choices from the public precision contract.

    Notes:
        - ``float64`` mode preserves prior default behavior as closely as possible:
          row storage remains float32, while normalization/influence math is float64.
        - ``float32`` mode keeps both storage and runtime compute in float32.
        - shadow debug precision remains explicit and independently auditable.
    """

    if internal_precision_requested == "float64":
        feature_row_storage_dtype = torch.float32
        row_abs_sum_dtype = torch.float64
        influence_compute_dtype = torch.float64
        planner_compute_dtype = torch.float64
    else:
        feature_row_storage_dtype = torch.float32
        row_abs_sum_dtype = torch.float32
        influence_compute_dtype = torch.float32
        planner_compute_dtype = torch.float32

    shadow_debug_compute_dtype = (
        torch.float64 if phase4_anomaly_debug_enabled else influence_compute_dtype
    )

    return {
        "internal_precision_requested": internal_precision_requested,
        "feature_row_storage_dtype": _dtype_to_name(feature_row_storage_dtype),
        "row_abs_sum_dtype": _dtype_to_name(row_abs_sum_dtype),
        "influence_compute_dtype": _dtype_to_name(influence_compute_dtype),
        "planner_compute_dtype": _dtype_to_name(planner_compute_dtype),
        "shadow_debug_compute_dtype": _dtype_to_name(shadow_debug_compute_dtype),
    }

def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype name: {dtype_name}")


def _build_vector_stats(
    vector: torch.Tensor,
    *,
    epsilon: float = 1e-12,
    top_k: int = 8,
) -> dict[str, object]:
    values = vector.detach().to(device="cpu", dtype=torch.float64).flatten()
    count = int(values.numel())
    if count == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "nonfinite_count": 0,
            "nonzero_count": 0,
            "effective_nonzero_count": 0,
            "zero_count": 0,
            "effective_zero_count": 0,
            "min": None,
            "max": None,
            "sum": 0.0,
            "abs_sum": 0.0,
            "mean": None,
            "abs_mean": None,
            "epsilon": float(epsilon),
            "all_zero": True,
            "effectively_all_zero": True,
            "top_abs_values": [],
        }

    abs_values = values.abs()
    finite_mask = torch.isfinite(values)
    nan_count = int(torch.isnan(values).sum().item())
    posinf_count = int(torch.isposinf(values).sum().item())
    neginf_count = int(torch.isneginf(values).sum().item())
    finite_count = int(finite_mask.sum().item())
    nonzero_count = int((values != 0).sum().item())
    effective_nonzero_count = int((abs_values > epsilon).sum().item())
    top_k_actual = min(max(0, int(top_k)), count)
    top_abs_values = []
    if top_k_actual > 0:
        top_abs, top_indices = torch.topk(abs_values, k=top_k_actual)
        for rank, (abs_value, idx_tensor) in enumerate(
            zip(top_abs.tolist(), top_indices.tolist(), strict=False),
            start=1,
        ):
            idx = int(idx_tensor)
            top_abs_values.append(
                {
                    "rank": rank,
                    "index": idx,
                    "value": float(values[idx].item()),
                    "abs_value": float(abs_value),
                }
            )

    sum_value = float(values.sum().item())
    abs_sum_value = float(abs_values.sum().item())
    return {
        "count": count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": count - finite_count,
        "nonzero_count": nonzero_count,
        "effective_nonzero_count": effective_nonzero_count,
        "zero_count": count - nonzero_count,
        "effective_zero_count": count - effective_nonzero_count,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "sum": sum_value,
        "abs_sum": abs_sum_value,
        "mean": float(sum_value / count),
        "abs_mean": float(abs_sum_value / count),
        "epsilon": float(epsilon),
        "all_zero": nonzero_count == 0,
        "effectively_all_zero": effective_nonzero_count == 0,
        "top_abs_values": top_abs_values,
    }


def _resolve_phase3_effective_row_state(
    *,
    rows_cpu: torch.Tensor,
    row_input_column_count: int,
    total_active_features: int,
    dtype: torch.dtype,
    donor_feature_rows: torch.Tensor | None = None,
    donor_row_abs_sums: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Resolve Phase-3 rows used by capture and compact row storage.

    The returned rows, feature slice, denominator, and materialized row sums all
    describe the same effective state. In donor row replay, feature columns and row
    normalizers are donor-effective while dense token/error columns remain host
    computed.
    """

    if rows_cpu.ndim != 2:
        raise ValueError("rows_cpu must be rank-2")
    row_input_column_count = int(row_input_column_count)
    total_active_features = int(total_active_features)
    if row_input_column_count < 0 or row_input_column_count > int(rows_cpu.shape[1]):
        raise ValueError("row_input_column_count is out of bounds")
    if total_active_features < 0 or total_active_features > row_input_column_count:
        raise ValueError("total_active_features is out of bounds")
    if (donor_feature_rows is None) != (donor_row_abs_sums is None):
        raise ValueError("donor_feature_rows and donor_row_abs_sums must be provided together")

    effective_rows_cpu = rows_cpu
    if donor_feature_rows is not None and donor_row_abs_sums is not None:
        if donor_feature_rows.ndim != 2:
            raise ValueError("donor_feature_rows must be rank-2")
        if tuple(donor_feature_rows.shape) != (int(rows_cpu.shape[0]), total_active_features):
            raise ValueError(
                "donor_feature_rows shape must match current row batch and active feature count"
            )
        if donor_row_abs_sums.ndim != 1 or int(donor_row_abs_sums.numel()) != int(
            rows_cpu.shape[0]
        ):
            raise ValueError("donor_row_abs_sums length must match current row batch")
        effective_rows_cpu = rows_cpu.clone()
        effective_rows_cpu[:, :total_active_features] = donor_feature_rows.to(
            device=effective_rows_cpu.device,
            dtype=effective_rows_cpu.dtype,
        )
        row_denominator_scaled_l1 = _row_abs_sums_to_scaled_l1(
            donor_row_abs_sums,
            dtype=dtype,
        )
        row_abs_sums_cpu = donor_row_abs_sums.detach()
        if row_abs_sums_cpu.device.type != "cpu" or row_abs_sums_cpu.dtype != torch.float64:
            row_abs_sums_cpu = row_abs_sums_cpu.to(device="cpu", dtype=torch.float64)
        row_abs_sums_cpu = row_abs_sums_cpu.contiguous()
    else:
        row_denominator_scaled_l1 = _compute_row_denominator_scaled_l1(
            effective_rows_cpu[:, :row_input_column_count],
            dtype=dtype,
        )
        row_abs_sums_cpu = (
            row_denominator_scaled_l1[0].to(dtype=torch.float64)
            * row_denominator_scaled_l1[1].to(dtype=torch.float64)
        ).contiguous()

    row_input_slice = effective_rows_cpu[:, :row_input_column_count]
    feature_row_slice = effective_rows_cpu[:, :total_active_features]
    return (
        effective_rows_cpu,
        row_input_slice,
        feature_row_slice,
        row_denominator_scaled_l1,
        row_abs_sums_cpu,
    )


def _copy_rows_to_cpu_staging(
    rows: torch.Tensor,
    *,
    staging_buffer: torch.Tensor | None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy a row block into a reusable CPU tensor and return a row-aligned view."""

    target_dtype = rows.dtype if dtype is None else dtype
    if rows.ndim != 2:
        raise ValueError("rows must be rank-2")

    if rows.device.type == "cpu" and rows.dtype == target_dtype:
        return rows, staging_buffer

    source = rows.detach()

    n_rows = int(rows.shape[0])
    n_cols = int(rows.shape[1])
    needs_new_buffer = (
        staging_buffer is None
        or staging_buffer.device.type != "cpu"
        or staging_buffer.dtype != target_dtype
        or int(staging_buffer.shape[0]) < n_rows
        or int(staging_buffer.shape[1]) < n_cols
    )
    if needs_new_buffer:
        staging_buffer = torch.empty((n_rows, n_cols), dtype=target_dtype, device="cpu")

    rows_cpu = staging_buffer[:n_rows, :n_cols]
    rows_cpu.copy_(source, non_blocking=False)
    return rows_cpu, staging_buffer


def _copy_feature_rows_to_cpu_staging(
    rows: torch.Tensor,
    *,
    total_active_feats: int,
    staging_buffer: torch.Tensor | None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy only active feature columns into a reusable CPU staging tensor."""

    target_dtype = rows.dtype if dtype is None else dtype
    if rows.ndim != 2:
        raise ValueError("rows must be rank-2")
    if total_active_feats < 0 or total_active_feats > int(rows.shape[1]):
        raise ValueError("total_active_feats must fit within rows columns")

    n_rows = int(rows.shape[0])
    n_cols = int(total_active_feats)
    source = rows.detach()[:, :n_cols]
    if source.device.type == "cpu" and source.dtype == target_dtype and source.is_contiguous():
        return source, staging_buffer

    needs_new_buffer = (
        staging_buffer is None
        or staging_buffer.device.type != "cpu"
        or staging_buffer.dtype != target_dtype
        or int(staging_buffer.shape[0]) < n_rows
        or int(staging_buffer.shape[1]) < n_cols
    )
    if needs_new_buffer:
        staging_buffer = torch.empty((n_rows, n_cols), dtype=target_dtype, device="cpu")

    feature_rows_cpu = staging_buffer[:n_rows, :n_cols]
    feature_rows_cpu.copy_(source, non_blocking=False)
    return feature_rows_cpu, staging_buffer


def _row_denominator_to_row_abs_sums(
    row_denominator: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(row_denominator, torch.Tensor):
        return row_denominator

    row_abs_max, row_l1_scaled = row_denominator
    return row_abs_max * row_l1_scaled


def _build_matrix_abs_stats(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
    top_k: int = 8,
) -> dict[str, object]:
    values = matrix.detach().to(device="cpu", dtype=torch.float64)
    flat = values.flatten()
    abs_values = flat.abs()
    finite_mask = torch.isfinite(flat)
    nan_count = int(torch.isnan(flat).sum().item())
    posinf_count = int(torch.isposinf(flat).sum().item())
    neginf_count = int(torch.isneginf(flat).sum().item())
    finite_count = int(finite_mask.sum().item())

    row_l1 = values.abs().sum(dim=1)
    row_max_abs = (
        values.abs().amax(dim=1)
        if values.ndim == 2 and values.shape[0] > 0
        else torch.empty(0, dtype=torch.float64)
    )

    top_entries: list[dict[str, object]] = []
    if flat.numel() > 0:
        top_k_actual = min(max(int(top_k), 0), int(flat.numel()))
        if top_k_actual > 0:
            top_abs, top_indices = torch.topk(abs_values, k=top_k_actual)
            n_cols = values.shape[1] if values.ndim == 2 and values.shape else 1
            for rank, (abs_value, flat_idx) in enumerate(
                zip(top_abs.tolist(), top_indices.tolist(), strict=False),
                start=1,
            ):
                flat_idx_int = int(flat_idx)
                row_idx = flat_idx_int // n_cols
                col_idx = flat_idx_int % n_cols
                top_entries.append(
                    {
                        "rank": rank,
                        "flat_index": flat_idx_int,
                        "row_index": int(row_idx),
                        "col_index": int(col_idx),
                        "value": float(flat[flat_idx_int].item()),
                        "abs_value": float(abs_value),
                    }
                )

    finite_abs_values = abs_values[finite_mask]
    return {
        "shape": list(values.shape),
        "count": int(flat.numel()),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": int(flat.numel()) - finite_count,
        "finite_max_abs": (
            float(finite_abs_values.max().item()) if finite_abs_values.numel() else None
        ),
        "finite_mean_abs": (
            float(finite_abs_values.mean().item()) if finite_abs_values.numel() else None
        ),
        "row_l1_stats": _build_vector_stats(row_l1, epsilon=max(epsilon, 1e-8), top_k=top_k),
        "row_max_abs_stats": _build_vector_stats(
            row_max_abs,
            epsilon=max(epsilon, 1e-8),
            top_k=top_k,
        ),
        "top_abs_entries": top_entries,
    }


def _build_phase4_normalization_stats(
    row_abs_sums: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    clamp_epsilon: float = 1e-8,
) -> dict[str, object]:
    if isinstance(row_abs_sums, tuple):
        row_abs_max, row_l1_scaled = row_abs_sums
        row_abs_max_cpu = row_abs_max.detach().to(device="cpu", dtype=torch.float64).flatten()
        row_l1_scaled_cpu = row_l1_scaled.detach().to(device="cpu", dtype=torch.float64).flatten()
        materialized_row_l1 = row_abs_max_cpu * row_l1_scaled_cpu
        stats = _build_vector_stats(materialized_row_l1, epsilon=clamp_epsilon)
        count = int(row_abs_max_cpu.numel())
        scaled_threshold = torch.where(
            row_abs_max_cpu > 0,
            torch.full_like(row_abs_max_cpu, clamp_epsilon) / row_abs_max_cpu,
            torch.full_like(row_abs_max_cpu, float("inf")),
        )
        clamped_mask = (
            ~torch.isfinite(row_abs_max_cpu)
            | ~torch.isfinite(row_l1_scaled_cpu)
            | (row_abs_max_cpu <= 0)
            | (row_l1_scaled_cpu <= 0)
            | (row_l1_scaled_cpu < scaled_threshold)
        )
        clamped_row_count = int(clamped_mask.sum().item())
        clamped_fraction = (clamped_row_count / count) if count else 0.0
        stats["representation"] = "scaled_row_l1"
        stats["effective_zero_count"] = clamped_row_count
        stats["effective_nonzero_count"] = int(count - clamped_row_count)
        stats["effectively_all_zero"] = bool(count == 0 or clamped_row_count == count)
        stats["clamp_epsilon"] = float(clamp_epsilon)
        stats["clamped_row_count"] = clamped_row_count
        stats["clamped_row_fraction"] = float(clamped_fraction)
        stats["row_abs_max_stats"] = _build_vector_stats(row_abs_max_cpu, epsilon=clamp_epsilon)
        stats["row_l1_scaled_stats"] = _build_vector_stats(row_l1_scaled_cpu, epsilon=clamp_epsilon)
        return stats

    stats = _build_vector_stats(row_abs_sums, epsilon=clamp_epsilon)
    count = int(stats.get("count", 0) or 0)
    effective_zero_count = int(stats.get("effective_zero_count", 0) or 0)
    clamped_fraction = (effective_zero_count / count) if count else 0.0
    stats["clamp_epsilon"] = float(clamp_epsilon)
    stats["clamped_row_count"] = effective_zero_count
    stats["clamped_row_fraction"] = float(clamped_fraction)
    stats["representation"] = "raw_l1"
    return stats


def _build_phase4_cutoff_debug(
    candidate_scores: torch.Tensor,
    *,
    queue_size: int,
    window_radius: int = 8,
) -> dict[str, object]:
    if queue_size <= 0 or candidate_scores.numel() == 0:
        return {
            "queue_size": int(queue_size),
            "candidate_count": int(candidate_scores.numel()),
            "cutoff_rank": None,
            "cutoff_score": None,
            "next_score": None,
            "cutoff_margin": None,
            "near_cutoff_epsilon": None,
            "near_cutoff_count": 0,
            "exact_cutoff_count": 0,
            "window_scores": [],
        }

    cutoff_rank = min(queue_size - 1, candidate_scores.numel() - 1)
    cutoff_score = float(candidate_scores[cutoff_rank].item())
    next_score = (
        float(candidate_scores[cutoff_rank + 1].item())
        if cutoff_rank + 1 < candidate_scores.numel()
        else None
    )
    cutoff_margin = None if next_score is None else float(cutoff_score - next_score)
    epsilon = max(abs(cutoff_score) * 1e-6, 1e-8)
    near_cutoff_count = int(((candidate_scores - cutoff_score).abs() <= epsilon).sum().item())
    exact_cutoff_count = int((candidate_scores == cutoff_score).sum().item())
    window_start = max(0, cutoff_rank - window_radius)
    window_end = min(candidate_scores.numel(), cutoff_rank + window_radius + 1)
    window_scores = [float(value) for value in candidate_scores[window_start:window_end].tolist()]
    return {
        "queue_size": int(queue_size),
        "candidate_count": int(candidate_scores.numel()),
        "cutoff_rank": int(cutoff_rank),
        "cutoff_score": cutoff_score,
        "next_score": next_score,
        "cutoff_margin": cutoff_margin,
        "near_cutoff_epsilon": float(epsilon),
        "near_cutoff_count": near_cutoff_count,
        "exact_cutoff_count": exact_cutoff_count,
        "window_scores": window_scores,
    }


def _build_phase3_frontier_buffer_metadata(
    *,
    seed_feature_influences: torch.Tensor | None,
    base_max_feature_nodes: int,
    total_active_features: int,
    relative_epsilon: float | None,
    max_extra: int,
) -> dict[str, object]:
    requested = relative_epsilon is not None or max_extra > 0
    metadata: dict[str, object] = {
        "schema_version": 1,
        "requested": bool(requested),
        "enabled": False,
        "effective": False,
        "base_max_feature_nodes": int(base_max_feature_nodes),
        "actual_max_feature_nodes": int(base_max_feature_nodes),
        "total_active_features": int(total_active_features),
        "relative_epsilon": None if relative_epsilon is None else float(relative_epsilon),
        "max_extra": int(max_extra),
        "extra_feature_count": 0,
        "cutoff_rank": None,
        "cutoff_score": None,
        "next_score": None,
        "cutoff_gap": None,
        "relative_cutoff_gap": None,
        "near_cutoff_counts": {"0.001": 0, "0.01": 0, "0.05": 0},
        "status": "disabled",
        "fallback_reason": None,
    }
    if relative_epsilon is None or max_extra <= 0:
        metadata["fallback_reason"] = "epsilon_or_max_extra_not_enabled"
        return metadata
    metadata["enabled"] = True
    if relative_epsilon < 0:
        metadata["status"] = "fallback"
        metadata["fallback_reason"] = "relative_epsilon_negative"
        return metadata
    if base_max_feature_nodes >= total_active_features:
        metadata["status"] = "skipped_all_features_included"
        metadata["fallback_reason"] = "all_features_included"
        return metadata
    if seed_feature_influences is None or seed_feature_influences.numel() == 0:
        metadata["status"] = "fallback"
        metadata["fallback_reason"] = "seed_feature_influences_unavailable"
        return metadata

    scores = torch.sort(
        seed_feature_influences.detach().to(device="cpu", dtype=torch.float64).flatten(),
        descending=True,
    ).values
    if scores.numel() <= 0 or base_max_feature_nodes <= 0:
        metadata["status"] = "fallback"
        metadata["fallback_reason"] = "empty_or_nonpositive_base_budget"
        return metadata
    cutoff_rank = min(base_max_feature_nodes - 1, int(scores.numel()) - 1)
    cutoff_score = float(scores[cutoff_rank].item())
    next_score = float(scores[cutoff_rank + 1].item()) if cutoff_rank + 1 < scores.numel() else None
    cutoff_gap = None if next_score is None else float(cutoff_score - next_score)
    relative_cutoff_gap = None
    if cutoff_gap is not None and cutoff_score > 0:
        relative_cutoff_gap = float(cutoff_gap / cutoff_score)
    metadata.update(
        {
            "cutoff_rank": int(cutoff_rank),
            "cutoff_score": cutoff_score,
            "next_score": next_score,
            "cutoff_gap": cutoff_gap,
            "relative_cutoff_gap": relative_cutoff_gap,
        }
    )

    below = scores[base_max_feature_nodes:]
    near_counts: dict[str, int] = {}
    for eps in (0.001, 0.01, 0.05):
        near_counts[str(eps)] = (
            int((below >= cutoff_score * (1.0 - eps)).sum().item()) if cutoff_score > 0 else 0
        )
    metadata["near_cutoff_counts"] = near_counts
    # Relative thresholds around zero/negative cutoffs can include an unbounded tail
    # of weakly negative candidates. Keep the experiment conservative and record a
    # fallback rather than switching to an absolute threshold with different meaning.
    if cutoff_score <= 0:
        metadata["status"] = "fallback"
        metadata["fallback_reason"] = "nonpositive_cutoff_score"
        return metadata
    extra_count = min(
        int((below >= cutoff_score * (1.0 - float(relative_epsilon))).sum().item()),
        int(max_extra),
        int(total_active_features - base_max_feature_nodes),
    )
    metadata["extra_feature_count"] = int(extra_count)
    metadata["actual_max_feature_nodes"] = int(base_max_feature_nodes + extra_count)
    metadata["effective"] = bool(extra_count > 0)
    metadata["status"] = "expanded" if extra_count > 0 else "no_extra_candidates"
    return metadata


def _build_phase4_frontier_buffer_decision(
    *,
    candidate_scores: torch.Tensor,
    base_frontier_size: int,
    actual_max_feature_nodes: int,
    capacity_feature_nodes: int,
    total_active_features: int,
    used_total: int,
    epsilon: float | None,
    max_per_refresh: int,
    max_total: int,
    refresh_index: int,
    visited_before: int,
) -> dict[str, object]:
    requested = epsilon is not None or max_per_refresh > 0 or max_total > 0
    event: dict[str, object] = {
        "refresh_index": int(refresh_index),
        "visited_before": int(visited_before),
        "base_frontier_size": int(base_frontier_size),
        "expanded_frontier_size": int(base_frontier_size),
        "extra_feature_count": 0,
        "cutoff_score": None,
        "next_score": None,
        "relative_cutoff_gap": None,
        "near_cutoff_counts": {"0.001": 0, "0.01": 0, "0.05": 0},
        "fallback_reason": None,
    }
    decision: dict[str, object] = {
        "requested": bool(requested),
        "enabled": bool(epsilon is not None and max_per_refresh > 0 and max_total > 0),
        "effective": False,
        "extra_feature_count": 0,
        "expanded_frontier_size": int(base_frontier_size),
        "event": event,
    }
    if not decision["enabled"]:
        event["fallback_reason"] = "epsilon_or_budget_not_enabled"
        return decision
    if base_frontier_size <= 0:
        event["fallback_reason"] = "empty_base_frontier"
        return decision

    scores = torch.sort(
        candidate_scores.detach().to(device="cpu", dtype=torch.float64).flatten(),
        descending=True,
    ).values
    if scores.numel() <= 0:
        event["fallback_reason"] = "empty_candidates"
        return decision
    cutoff_rank = min(int(base_frontier_size) - 1, int(scores.numel()) - 1)
    cutoff_score = float(scores[cutoff_rank].item())
    next_score = float(scores[cutoff_rank + 1].item()) if cutoff_rank + 1 < scores.numel() else None
    cutoff_gap = None if next_score is None else float(cutoff_score - next_score)
    relative_cutoff_gap = None
    if cutoff_gap is not None and cutoff_score > 0:
        relative_cutoff_gap = float(cutoff_gap / cutoff_score)
    below = scores[int(base_frontier_size) :]
    near_counts = {
        str(eps): int((below >= cutoff_score * (1.0 - float(eps))).sum().item())
        if cutoff_score > 0
        else 0
        for eps in (0.001, 0.01, 0.05)
    }
    event.update(
        {
            "cutoff_score": cutoff_score,
            "next_score": next_score,
            "relative_cutoff_gap": relative_cutoff_gap,
            "near_cutoff_counts": near_counts,
        }
    )
    if cutoff_score <= 0:
        event["fallback_reason"] = "nonpositive_cutoff_score"
        return decision
    remaining = min(
        int(max_total - used_total),
        int(total_active_features - actual_max_feature_nodes),
        int(capacity_feature_nodes - actual_max_feature_nodes),
    )
    if remaining <= 0:
        event["fallback_reason"] = "capacity_or_budget_exhausted"
        return decision
    near_count = int((below >= cutoff_score * (1.0 - float(epsilon))).sum().item())
    extra = min(near_count, int(max_per_refresh), remaining)
    if extra <= 0:
        event["fallback_reason"] = "no_extra_candidates"
        return decision
    expanded = int(base_frontier_size + extra)
    event["extra_feature_count"] = int(extra)
    event["expanded_frontier_size"] = expanded
    decision.update(
        {"effective": True, "extra_feature_count": int(extra), "expanded_frontier_size": expanded}
    )
    return decision


def _build_semantic_sketch_fallback(
    *,
    candidate_features: torch.Tensor,
    activation_value: torch.Tensor,
    seed_influence: torch.Tensor,
    seed_rank: torch.Tensor,
    frontier_pre_rank: torch.Tensor,
    frontier_post_rank: torch.Tensor,
    is_top_seed: torch.Tensor,
    is_frontier_pre: torch.Tensor,
    is_frontier_post: torch.Tensor,
    descriptor_dim: int,
) -> torch.Tensor:
    candidate_cpu = candidate_features.detach().to(device="cpu", dtype=torch.int64)
    candidate_count = int(candidate_cpu.shape[0])
    descriptor_dim = int(descriptor_dim)
    if descriptor_dim <= 0:
        raise ValueError("semantic_descriptor_dim must be > 0")
    if candidate_count == 0:
        return torch.empty((0, descriptor_dim), dtype=torch.float32)

    candidate_np = candidate_cpu.numpy().astype(np.uint64, copy=False)
    layer_ids = candidate_np[:, 0]
    position_ids = candidate_np[:, 1]
    feature_ids = candidate_np[:, 2]

    seeds = (
        layer_ids * np.uint64(0x9E3779B185EBCA87)
        ^ position_ids * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ feature_ids * np.uint64(0x165667B19E3779F9)
        ^ np.uint64(0xD6E8FEB86659FD93)
    )
    dim_offsets = np.arange(descriptor_dim, dtype=np.uint64)[None, :]
    z = seeds[:, None] + dim_offsets * np.uint64(0x9E3779B97F4A7C15)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    base = ((z >> np.uint64(11)).astype(np.float64) / float(1 << 53)) * 2.0 - 1.0

    activation_np = activation_value.detach().to(device="cpu", dtype=torch.float64).numpy()
    seed_influence_np = seed_influence.detach().to(device="cpu", dtype=torch.float64).numpy()
    seed_rank_np = seed_rank.detach().to(device="cpu", dtype=torch.int64).numpy()
    frontier_pre_rank_np = frontier_pre_rank.detach().to(device="cpu", dtype=torch.int64).numpy()
    frontier_post_rank_np = frontier_post_rank.detach().to(device="cpu", dtype=torch.int64).numpy()
    is_top_seed_np = is_top_seed.detach().to(device="cpu", dtype=torch.float64).numpy()
    is_frontier_pre_np = is_frontier_pre.detach().to(device="cpu", dtype=torch.float64).numpy()
    is_frontier_post_np = is_frontier_post.detach().to(device="cpu", dtype=torch.float64).numpy()

    def _inverse_rank(rank_values: np.ndarray) -> np.ndarray:
        out = np.zeros(rank_values.shape[0], dtype=np.float64)
        valid = rank_values >= 0
        out[valid] = 1.0 / (rank_values[valid].astype(np.float64) + 1.0)
        return out

    inverse_seed_rank = _inverse_rank(seed_rank_np)
    inverse_pre_rank = _inverse_rank(frontier_pre_rank_np)
    inverse_post_rank = _inverse_rank(frontier_post_rank_np)

    metadata = np.stack(
        [
            np.tanh(activation_np),
            np.tanh(seed_influence_np),
            inverse_seed_rank,
            inverse_pre_rank,
            inverse_post_rank,
            is_top_seed_np,
            is_frontier_pre_np,
            is_frontier_post_np,
        ],
        axis=1,
    )
    dim_positions = np.arange(descriptor_dim, dtype=np.float64) + 1.0
    projection = np.vstack(
        [
            np.cos(dim_positions * 0.17),
            np.sin(dim_positions * 0.31),
            np.cos(dim_positions * 0.47),
            np.sin(dim_positions * 0.73),
            np.cos(dim_positions * 0.89),
            np.sin(dim_positions * 1.07),
            np.cos(dim_positions * 1.21),
            np.sin(dim_positions * 1.37),
        ]
    )
    sketch = np.tanh(base + 0.25 * (metadata @ projection)).astype(np.float32, copy=False)

    if descriptor_dim > 0:
        sketch[:, 0] = np.tanh(activation_np).astype(np.float32, copy=False)
    if descriptor_dim > 1:
        sketch[:, 1] = np.tanh(seed_influence_np).astype(np.float32, copy=False)
    if descriptor_dim > 2:
        sketch[:, 2] = inverse_seed_rank.astype(np.float32, copy=False)
    if descriptor_dim > 3:
        sketch[:, 3] = inverse_pre_rank.astype(np.float32, copy=False)
    if descriptor_dim > 4:
        sketch[:, 4] = inverse_post_rank.astype(np.float32, copy=False)
    if descriptor_dim > 5:
        sketch[:, 5] = is_top_seed_np.astype(np.float32, copy=False)
    if descriptor_dim > 6:
        sketch[:, 6] = is_frontier_pre_np.astype(np.float32, copy=False)
    if descriptor_dim > 7:
        sketch[:, 7] = is_frontier_post_np.astype(np.float32, copy=False)

    return torch.from_numpy(np.ascontiguousarray(sketch))


def _build_feature_semantic_descriptors_payload(
    *,
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    seed_feature_influences: torch.Tensor,
    frontier_pre_locality: torch.Tensor,
    frontier_post_locality: torch.Tensor,
    total_active_features: int,
    status: str,
    semantic_descriptor_top_k: int,
    semantic_descriptor_dim: int,
) -> dict[str, object]:
    top_k = int(semantic_descriptor_top_k)
    descriptor_dim = int(semantic_descriptor_dim)
    if top_k <= 0:
        raise ValueError("semantic_descriptor_top_k must be > 0")
    if descriptor_dim <= 0:
        raise ValueError("semantic_descriptor_dim must be > 0")

    active_features_cpu = active_features.detach().to(device="cpu", dtype=torch.int64)
    activation_values_cpu = activation_values.detach().to(device="cpu", dtype=torch.float32)
    feature_count = int(active_features_cpu.shape[0])
    if activation_values_cpu.numel() != feature_count:
        raise ValueError(
            "activation_values length must match active_features row count "
            f"({activation_values_cpu.numel()} != {feature_count})"
        )

    seed_influences_cpu = seed_feature_influences.detach().to(device="cpu", dtype=torch.float64)
    seed_influence_available = seed_influences_cpu.numel() == feature_count
    if not seed_influence_available:
        seed_influences_cpu = torch.zeros(feature_count, dtype=torch.float64)

    frontier_pre_cpu = frontier_pre_locality.detach().to(device="cpu", dtype=torch.int64)
    frontier_post_cpu = frontier_post_locality.detach().to(device="cpu", dtype=torch.int64)
    valid_frontier_pre = frontier_pre_cpu[
        (frontier_pre_cpu >= 0) & (frontier_pre_cpu < feature_count)
    ]
    valid_frontier_post = frontier_post_cpu[
        (frontier_post_cpu >= 0) & (frontier_post_cpu < feature_count)
    ]

    seed_rank_full = torch.full((feature_count,), -1, dtype=torch.int64)
    top_seed_mask_full = torch.zeros(feature_count, dtype=torch.bool)
    if seed_influence_available and feature_count > 0:
        ranked = torch.argsort(seed_influences_cpu, descending=True)
        seed_rank_full[ranked] = torch.arange(ranked.numel(), dtype=torch.int64)
        top_seed_indices = ranked[: min(feature_count, top_k)]
        top_seed_mask_full[top_seed_indices] = True

    frontier_pre_rank_full = torch.full((feature_count,), -1, dtype=torch.int64)
    for rank, feature_idx in enumerate(valid_frontier_pre.tolist()):
        if frontier_pre_rank_full[feature_idx] < 0:
            frontier_pre_rank_full[feature_idx] = int(rank)

    frontier_post_rank_full = torch.full((feature_count,), -1, dtype=torch.int64)
    for rank, feature_idx in enumerate(valid_frontier_post.tolist()):
        if frontier_post_rank_full[feature_idx] < 0:
            frontier_post_rank_full[feature_idx] = int(rank)

    candidate_mask = torch.zeros(feature_count, dtype=torch.bool)
    candidate_mask[top_seed_mask_full] = True
    candidate_mask[frontier_pre_rank_full >= 0] = True
    candidate_mask[frontier_post_rank_full >= 0] = True
    candidate_row_indices = torch.where(candidate_mask)[0].to(dtype=torch.int64)

    if candidate_row_indices.numel() > top_k:
        max_rank = max(feature_count, top_k) + 1
        candidate_rows = [int(value) for value in candidate_row_indices.tolist()]

        def _candidate_sort_key(feature_idx: int) -> tuple[int, int, int, int, int, int, int]:
            seed_rank = int(seed_rank_full[feature_idx].item())
            pre_rank = int(frontier_pre_rank_full[feature_idx].item())
            post_rank = int(frontier_post_rank_full[feature_idx].item())
            return (
                0 if seed_rank >= 0 else 1,
                seed_rank if seed_rank >= 0 else max_rank,
                0 if pre_rank >= 0 else 1,
                pre_rank if pre_rank >= 0 else max_rank,
                0 if post_rank >= 0 else 1,
                post_rank if post_rank >= 0 else max_rank,
                feature_idx,
            )

        candidate_rows = sorted(candidate_rows, key=_candidate_sort_key)[:top_k]
        candidate_row_indices = torch.tensor(candidate_rows, dtype=torch.int64)

    candidate_features = active_features_cpu[candidate_row_indices]
    candidate_activation = activation_values_cpu[candidate_row_indices]
    candidate_seed_influence = seed_influences_cpu[candidate_row_indices]
    candidate_seed_rank = seed_rank_full[candidate_row_indices]
    candidate_is_top_seed = top_seed_mask_full[candidate_row_indices]
    candidate_frontier_pre_rank = frontier_pre_rank_full[candidate_row_indices]
    candidate_is_frontier_pre = candidate_frontier_pre_rank >= 0
    candidate_frontier_post_rank = frontier_post_rank_full[candidate_row_indices]
    candidate_is_frontier_post = candidate_frontier_post_rank >= 0
    candidate_count = int(candidate_row_indices.numel())

    semantic_sketch = _build_semantic_sketch_fallback(
        candidate_features=candidate_features,
        activation_value=candidate_activation,
        seed_influence=candidate_seed_influence,
        seed_rank=candidate_seed_rank,
        frontier_pre_rank=candidate_frontier_pre_rank,
        frontier_post_rank=candidate_frontier_post_rank,
        is_top_seed=candidate_is_top_seed,
        is_frontier_pre=candidate_is_frontier_pre,
        is_frontier_post=candidate_is_frontier_post,
        descriptor_dim=descriptor_dim,
    )

    return {
        "status": str(status),
        "descriptor_version": "v1",
        "descriptor_kind": "fallback_identity_metadata_v1",
        "descriptor_dim": int(descriptor_dim),
        "semantic_descriptor_top_k": int(top_k),
        "candidate_count": int(candidate_count),
        "total_active_features": int(total_active_features),
        "candidate_features": candidate_features,
        "candidate_row_indices": candidate_row_indices,
        "activation_value": candidate_activation,
        "seed_influence": candidate_seed_influence,
        "seed_rank": candidate_seed_rank,
        "is_top_seed": candidate_is_top_seed,
        "is_frontier_pre": candidate_is_frontier_pre,
        "frontier_pre_rank": candidate_frontier_pre_rank,
        "is_frontier_post": candidate_is_frontier_post,
        "frontier_post_rank": candidate_frontier_post_rank,
        "is_selected_phase4": torch.zeros(candidate_count, dtype=torch.bool),
        "phase4_selected_rank": torch.full((candidate_count,), -1, dtype=torch.int64),
        "phase4_selection_available": False,
        "seed_influence_available": bool(seed_influence_available),
        "semantic_sketch": semantic_sketch,
    }


def _annotate_phase4_selection_on_feature_semantic_descriptors(
    payload: dict[str, object], *, selected_features: torch.Tensor
) -> None:
    candidate_row_indices = payload.get("candidate_row_indices")
    if not isinstance(candidate_row_indices, torch.Tensor):
        return

    candidate_row_indices_cpu = candidate_row_indices.detach().to(device="cpu", dtype=torch.int64)
    selected_features_cpu = selected_features.detach().to(device="cpu", dtype=torch.int64)
    selected_rank_lookup = {
        int(feature_idx): int(rank)
        for rank, feature_idx in enumerate(selected_features_cpu.tolist())
    }

    selected_mask = torch.tensor(
        [
            int(feature_idx) in selected_rank_lookup
            for feature_idx in candidate_row_indices_cpu.tolist()
        ],
        dtype=torch.bool,
    )
    selected_rank = torch.tensor(
        [
            selected_rank_lookup.get(int(feature_idx), -1)
            for feature_idx in candidate_row_indices_cpu
        ],
        dtype=torch.int64,
    )
    payload["is_selected_phase4"] = selected_mask
    payload["phase4_selected_rank"] = selected_rank
    payload["phase4_selection_available"] = True


def _record_phase4_refresh_debug(
    anomaly_debug_result: dict[str, object] | None,
    *,
    refresh_index: int,
    n_visited: int,
    queue_size: int,
    pending: torch.Tensor,
    previous_pending: torch.Tensor | None,
    first_pending: torch.Tensor | None,
    candidate_scores: torch.Tensor,
    refresh_elapsed_ms: float,
    rank_signal_stats: dict[str, object] | None,
    logit_probability_stats: dict[str, object] | None,
    normalization_input_stats: dict[str, object] | None,
    feature_row_store_read_stats: dict[str, object] | None,
    streaming_chunk_reuse_stats: dict[str, object] | None,
) -> None:
    if anomaly_debug_result is None:
        return

    pending_cpu = pending.detach().to(device="cpu", dtype=torch.int64)
    pending_set = set(int(value) for value in pending_cpu.tolist())
    previous_overlap = None
    if previous_pending is not None:
        previous_set = set(int(value) for value in previous_pending.tolist())
        if previous_set:
            previous_overlap = len(pending_set & previous_set) / len(previous_set)
    first_overlap = None
    if first_pending is not None:
        first_set = set(int(value) for value in first_pending.tolist())
        if first_set:
            first_overlap = len(pending_set & first_set) / len(first_set)

    record = {
        "refresh_index": int(refresh_index),
        "refresh_elapsed_ms": float(refresh_elapsed_ms),
        "n_visited": int(n_visited),
        "pending_size": int(pending_cpu.numel()),
        "queue_size": int(queue_size),
        "pending_hash": _hash_index_tensor(pending_cpu),
        "pending_sample": [int(value) for value in pending_cpu[:16].tolist()],
        "overlap_with_previous": previous_overlap,
        "overlap_with_first": first_overlap,
        "cutoff": _build_phase4_cutoff_debug(candidate_scores, queue_size=queue_size),
    }
    if rank_signal_stats is not None:
        record["rank_signal_stats"] = rank_signal_stats
        record["rank_signal_all_zero"] = bool(rank_signal_stats.get("all_zero", False))
        record["rank_signal_effectively_all_zero"] = bool(
            rank_signal_stats.get("effectively_all_zero", False)
        )
    if logit_probability_stats is not None:
        record["logit_probability_stats"] = logit_probability_stats
    if normalization_input_stats is not None:
        record["normalization_input_stats"] = normalization_input_stats
    if feature_row_store_read_stats is not None:
        record["feature_row_store_read_stats"] = feature_row_store_read_stats
    if streaming_chunk_reuse_stats is not None:
        record["streaming_chunk_reuse_stats"] = streaming_chunk_reuse_stats
    records = anomaly_debug_result.setdefault("records", [])
    assert isinstance(records, list)
    records.append(record)


def _compare_phase4_frontiers(
    actual_pending: torch.Tensor,
    shadow_pending: torch.Tensor,
) -> dict[str, object]:
    actual_cpu = actual_pending.detach().to(device="cpu", dtype=torch.int64)
    shadow_cpu = shadow_pending.detach().to(device="cpu", dtype=torch.int64)
    actual_list = [int(value) for value in actual_cpu.tolist()]
    shadow_list = [int(value) for value in shadow_cpu.tolist()]
    actual_set = set(actual_list)
    shadow_set = set(shadow_list)
    overlap_count = len(actual_set & shadow_set)
    union_count = len(actual_set | shadow_set)
    overlap_fraction = overlap_count / len(actual_set) if actual_set else None
    jaccard_similarity = (overlap_count / union_count) if union_count else None
    first_differing_rank = None
    for idx, (actual_value, shadow_value) in enumerate(zip(actual_list, shadow_list)):
        if actual_value != shadow_value:
            first_differing_rank = int(idx)
            break

    prefix_match_count = 0
    for actual_value, shadow_value in zip(actual_list, shadow_list):
        if actual_value != shadow_value:
            break
        prefix_match_count += 1

    shared_rank_count = min(len(actual_list), len(shadow_list))
    rank_disagreement_count = sum(
        1 for idx in range(shared_rank_count) if actual_list[idx] != shadow_list[idx]
    ) + abs(len(actual_list) - len(shadow_list))

    added_nodes = sorted(shadow_set - actual_set)
    removed_nodes = sorted(actual_set - shadow_set)
    overlap_nodes = sorted(actual_set & shadow_set)
    return {
        "actual_hash": _hash_index_tensor(actual_cpu),
        "shadow_hash": _hash_index_tensor(shadow_cpu),
        "overlap_count": int(overlap_count),
        "overlap_fraction": overlap_fraction,
        "jaccard_similarity": jaccard_similarity,
        "prefix_match_count": int(prefix_match_count),
        "rank_disagreement_count": int(rank_disagreement_count),
        "changed_selected_nodes": int(len(actual_set ^ shadow_set)),
        "first_differing_rank": first_differing_rank,
        "added_nodes_sample": [int(value) for value in added_nodes[:16]],
        "removed_nodes_sample": [int(value) for value in removed_nodes[:16]],
        "overlap_nodes_sample": [int(value) for value in overlap_nodes[:16]],
        "actual_sample": [int(value) for value in actual_cpu[:16].tolist()],
        "shadow_sample": [int(value) for value in shadow_cpu[:16].tolist()],
    }


def _build_phase4_deterministic_shadow_pending(
    candidate_indices: torch.Tensor,
    feature_influences: torch.Tensor,
    *,
    queue_size: int,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    exact_chunked_decoder: bool,
    decoder_chunk_size: int | None,
) -> torch.Tensor:
    use_chunk_key = bool(exact_chunked_decoder and decoder_chunk_size and decoder_chunk_size > 0)
    ranked = sorted(
        candidate_indices.detach().to(device="cpu", dtype=torch.int64).tolist(),
        key=lambda idx: (
            -float(feature_influences[idx].item()),
            int(feat_layers[idx]),
            (int(feat_ids[idx]) // int(decoder_chunk_size)) if use_chunk_key else -1,
            int(feat_positions[idx]),
            int(feat_ids[idx]),
            int(idx),
        ),
    )
    pending = torch.tensor(ranked[:queue_size], dtype=torch.long)
    return _reorder_pending_for_phase4_locality(
        pending,
        feat_layers=feat_layers,
        feat_positions=feat_positions,
        feat_ids=feat_ids,
        exact_chunked_decoder=exact_chunked_decoder,
        decoder_chunk_size=decoder_chunk_size,
    )


def _build_phase4_environment_fingerprint() -> dict[str, object]:
    return {
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "workspace_root": os.getenv("WORKSPACE_ROOT"),
        "lib_workspace_root": os.getenv("LIB_WORKSPACE_ROOT"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nnsight_backend_file": _NNSIGHT_BACKEND_FILE,
    }
