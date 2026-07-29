"""NNSight attribution telemetry payload and checkpoint helpers."""

import hashlib
import json
import math
from typing import Literal, TypedDict, cast

import torch

from circuit_tracer.observability.events import RuntimeSnapshot, TraceObserver


def _safe_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.item())
    return float(value)


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return _safe_int(value.item())
    return None


def _tensor_nbytes_estimate(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _dtype_element_size(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _build_tensor_transfer_estimate(
    *,
    prefix: str,
    source: torch.Tensor,
    destination_device: torch.device | str,
    destination_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    dest_device = torch.device(destination_device)
    dest_dtype = source.dtype if destination_dtype is None else destination_dtype
    estimated_bytes = int(source.numel() * _dtype_element_size(dest_dtype))
    transfer_bytes = (
        estimated_bytes
        if source.device.type != dest_device.type or source.dtype != dest_dtype
        else 0
    )
    return {
        f"{prefix}_source": str(source.device.type),
        f"{prefix}_destination": str(dest_device.type),
        f"{prefix}_dtype_source": str(source.dtype),
        f"{prefix}_dtype_destination": str(dest_dtype),
        f"{prefix}_bytes": int(estimated_bytes),
        f"{prefix}_transfer_bytes": int(transfer_bytes),
    }


class _RowTransferTelemetry(TypedDict):
    row_transfer_source: str
    row_transfer_destination: str
    row_transfer_count: int
    row_transfer_bytes: int
    row_input_bytes: int
    feature_row_bytes: int


def _build_row_transfer_telemetry(
    *,
    rows: torch.Tensor,
    rows_cpu: torch.Tensor,
    row_input_slice: torch.Tensor,
    feature_row_slice: torch.Tensor,
) -> _RowTransferTelemetry:
    source = str(rows.device.type)
    destination = str(rows_cpu.device.type)
    transferred = source != destination or rows.dtype != rows_cpu.dtype
    return {
        "row_transfer_source": source,
        "row_transfer_destination": destination,
        "row_transfer_count": int(rows.shape[0]),
        "row_transfer_bytes": int(_tensor_nbytes_estimate(rows_cpu) if transferred else 0),
        "row_input_bytes": int(_tensor_nbytes_estimate(row_input_slice)),
        "feature_row_bytes": int(_tensor_nbytes_estimate(feature_row_slice)),
    }


def _build_phase4_gpu_row_reduction_transfer_telemetry(
    *,
    rows: torch.Tensor,
    feature_row_slice: torch.Tensor,
    row_abs_max: torch.Tensor,
    row_l1_scaled: torch.Tensor,
) -> dict[str, object]:
    compact_transfer_bytes = (
        _tensor_nbytes_estimate(feature_row_slice)
        + _tensor_nbytes_estimate(row_abs_max)
        + _tensor_nbytes_estimate(row_l1_scaled)
    )
    baseline_full_row_transfer_bytes = _tensor_nbytes_estimate(rows)
    return {
        "row_transfer_source": str(rows.device.type),
        "row_transfer_destination": "cpu",
        "row_transfer_count": int(rows.shape[0]),
        "row_transfer_bytes": int(compact_transfer_bytes),
        "row_input_bytes": int(
            _tensor_nbytes_estimate(row_abs_max) + _tensor_nbytes_estimate(row_l1_scaled)
        ),
        "feature_row_bytes": int(_tensor_nbytes_estimate(feature_row_slice)),
        "row_reduction_backend": "gpu_v1",
        "row_reduction_baseline_full_row_transfer_bytes": int(baseline_full_row_transfer_bytes),
        "row_reduction_compact_transfer_bytes": int(compact_transfer_bytes),
        "row_reduction_gpu_to_cpu_bytes_saved": int(
            max(0, baseline_full_row_transfer_bytes - compact_transfer_bytes)
        ),
    }


def _build_phase4_refresh_substage_telemetry(
    *,
    telemetry_detail: Literal["summary", "normal", "debug"],
    partial_influence_elapsed_ms: float,
    rank_topk_elapsed_ms: float,
    frontier_plan_elapsed_ms: float,
    row_store_read_elapsed_ms: float | None,
    influence_normalization_elapsed_ms: float | None,
    influence_matmul_elapsed_ms: float | None,
    chunk_request_count: int | None,
    active_row_chunk_count: int | None,
    row_reader_row_count: int | None,
    solver_iteration_count: int | None,
    row_chunk_strategy: str | None = None,
    row_weight_nonzero_row_count: int | None = None,
    row_weight_zero_row_count: int | None = None,
    row_reader_overread_zero_row_count: int | None = None,
    active_row_range_count: int | None = None,
    streaming_chunk_reuse_stats: dict[str, object] | None = None,
    feature_row_store_read_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "refresh_partial_influence_elapsed_ms": float(partial_influence_elapsed_ms),
        "refresh_rank_topk_elapsed_ms": float(rank_topk_elapsed_ms),
        "refresh_frontier_plan_elapsed_ms": float(frontier_plan_elapsed_ms),
    }
    if telemetry_detail in {"normal", "debug"}:
        payload.update(
            {
                "refresh_row_store_read_elapsed_ms": _safe_float(row_store_read_elapsed_ms),
                "refresh_influence_normalization_elapsed_ms": _safe_float(
                    influence_normalization_elapsed_ms
                ),
                "refresh_influence_matmul_elapsed_ms": _safe_float(influence_matmul_elapsed_ms),
                "refresh_chunk_request_count": _safe_int(chunk_request_count),
                "refresh_active_row_chunk_count": _safe_int(active_row_chunk_count),
                "refresh_rows_touched": _safe_int(row_reader_row_count),
                "refresh_solver_iteration_count": _safe_int(solver_iteration_count),
                "refresh_row_chunk_strategy": row_chunk_strategy,
                "refresh_row_weight_nonzero_rows": _safe_int(row_weight_nonzero_row_count),
                "refresh_row_weight_zero_rows": _safe_int(row_weight_zero_row_count),
                "refresh_row_reader_overread_zero_rows": _safe_int(
                    row_reader_overread_zero_row_count
                ),
                "refresh_active_row_range_count": _safe_int(active_row_range_count),
            }
        )
        if streaming_chunk_reuse_stats is not None:
            for source_key, telemetry_key in {
                "active_row_scan_elapsed_ms_total": "refresh_active_row_scan_elapsed_ms",
                "chunk_allocation_zero_fill_elapsed_ms_total": "refresh_chunk_allocation_zero_fill_elapsed_ms",
                "direct_accumulation_elapsed_ms_total": "refresh_direct_accumulation_elapsed_ms",
                "transfer_cast_abs_elapsed_ms_total": "refresh_transfer_cast_abs_elapsed_ms",
                "cache_lookup_elapsed_ms_total": "refresh_cache_lookup_elapsed_ms",
                "cache_store_elapsed_ms_total": "refresh_cache_store_elapsed_ms",
                "cache_eviction_elapsed_ms_total": "refresh_cache_eviction_elapsed_ms",
                "row_weight_update_elapsed_ms_total": "refresh_row_weight_update_elapsed_ms",
                "accounted_elapsed_ms_total": "refresh_accounted_elapsed_ms",
                "unaccounted_elapsed_ms_total": "refresh_unaccounted_elapsed_ms",
            }.items():
                payload[telemetry_key] = _safe_float(streaming_chunk_reuse_stats.get(source_key))
            for source_key, telemetry_key in {
                "active_row_accumulation_mode": "refresh_active_row_accumulation_mode",
                "active_row_direct_accumulation": "refresh_active_row_direct_accumulation",
                "direct_accumulation_subrange_count": "refresh_direct_accumulation_subrange_count",
                "prepared_row_reader_enabled": "refresh_prepared_row_reader_enabled",
            }.items():
                value = streaming_chunk_reuse_stats.get(source_key)
                payload[telemetry_key] = value if isinstance(value, str) else _safe_int(value)
        if feature_row_store_read_stats is not None:
            for source_key, telemetry_key in {
                "prepared_read_cache_hit_count": "feature_row_store_prepared_read_cache_hits",
                "prepared_read_cache_miss_count": "feature_row_store_prepared_read_cache_misses",
                "prepared_read_cache_hit_row_count": "feature_row_store_prepared_read_cache_hit_rows",
                "prepared_read_cache_miss_row_count": "feature_row_store_prepared_read_cache_miss_rows",
                "prepared_read_cache_eviction_count": "feature_row_store_prepared_read_cache_evictions",
                "prepared_read_cache_invalidation_count": "feature_row_store_prepared_read_cache_invalidations",
                "prepared_read_cache_invalidation_entry_count": "feature_row_store_prepared_read_cache_invalidation_entries",
                "prepared_read_cache_store_attempt_count": "feature_row_store_prepared_read_cache_store_attempts",
                "prepared_read_cache_store_success_count": "feature_row_store_prepared_read_cache_store_success",
                "prepared_read_cache_store_skip_disabled_count": "feature_row_store_prepared_read_cache_store_skip_disabled",
                "prepared_read_cache_store_skip_too_large_count": "feature_row_store_prepared_read_cache_store_skip_too_large",
                "prepared_read_cache_entry_count": "feature_row_store_prepared_read_cache_entry_count",
                "prepared_read_cache_nbytes": "feature_row_store_prepared_read_cache_nbytes",
                "gpu_row_tier_read_hits": "feature_row_store_gpu_tier_read_hits",
                "gpu_row_tier_read_hit_rows": "feature_row_store_gpu_tier_read_hit_rows",
                "gpu_row_tier_read_hit_bytes": "feature_row_store_gpu_tier_read_hit_bytes",
                "gpu_row_tier_read_fallbacks": "feature_row_store_gpu_tier_read_fallbacks",
                "gpu_row_tier_read_fallback_rows": "feature_row_store_gpu_tier_read_fallback_rows",
                "gpu_row_tier_avoided_file_read_bytes": (
                    "feature_row_store_gpu_tier_avoided_file_read_bytes"
                ),
                "gpu_row_tier_avoided_h2d_bytes": (
                    "feature_row_store_gpu_tier_avoided_h2d_bytes"
                ),
                "gpu_row_tier_d2h_bytes": "feature_row_store_gpu_tier_d2h_bytes",
                "gpu_row_tier_copy_failures": "feature_row_store_gpu_tier_copy_failures",
                "gpu_row_tier_append_calls": "feature_row_store_gpu_tier_append_calls",
                "gpu_row_tier_append_rows": "feature_row_store_gpu_tier_append_rows",
                "gpu_row_tier_append_bytes": "feature_row_store_gpu_tier_append_bytes",
                "gpu_row_tier_high_water_bytes": "feature_row_store_gpu_tier_high_water_bytes",
                "gpu_row_tier_owned_bytes": "feature_row_store_gpu_tier_owned_bytes",
            }.items():
                payload[telemetry_key] = _safe_int(feature_row_store_read_stats.get(source_key))
            payload["feature_row_store_gpu_tier_read_transfer_elapsed_ms"] = _safe_float(
                feature_row_store_read_stats.get("gpu_row_tier_read_transfer_elapsed_ms")
            )
            payload["feature_row_store_prepared_read_cache_prepare_elapsed_ms"] = _safe_float(
                feature_row_store_read_stats.get("prepared_read_cache_prepare_elapsed_ms_total")
            )
    return payload


def _build_phase4_executor_substage_telemetry(
    *,
    telemetry_detail: Literal["summary", "normal", "debug"],
    encoder_materialize_elapsed_ms: float,
    compute_batch_elapsed_ms: float,
    cpu_staging_elapsed_ms: float,
    denominator_elapsed_ms: float,
    row_store_write_elapsed_ms: float,
    batch_elapsed_ms: float,
) -> dict[str, object]:
    accounted_elapsed_ms = (
        encoder_materialize_elapsed_ms
        + compute_batch_elapsed_ms
        + cpu_staging_elapsed_ms
        + denominator_elapsed_ms
        + row_store_write_elapsed_ms
    )
    payload: dict[str, object] = {
        "executor_encoder_materialize_elapsed_ms": float(encoder_materialize_elapsed_ms),
        "executor_compute_batch_elapsed_ms": float(compute_batch_elapsed_ms),
        "executor_accounted_elapsed_ms": float(accounted_elapsed_ms),
        "executor_overhead_elapsed_ms": float(max(0.0, batch_elapsed_ms - accounted_elapsed_ms)),
    }
    if telemetry_detail in {"normal", "debug"}:
        payload.update(
            {
                "executor_cpu_staging_elapsed_ms": float(cpu_staging_elapsed_ms),
                "executor_denominator_elapsed_ms": float(denominator_elapsed_ms),
                "executor_row_store_write_elapsed_ms": float(row_store_write_elapsed_ms),
            }
        )
    return payload


def _build_phase4_executor_batch_telemetry(
    *,
    semantic_batch_count: int,
    semantic_batch_max_rows: int,
    semantic_batch_index_start: int | None = None,
    semantic_batch_index_end: int | None = None,
    semantic_batch_rows: tuple[int, ...] = (),
    execution_batch_index: int | None = None,
    execution_batch_count: int | None = None,
    execution_batch_rows: int | None = None,
    execution_batch_max_rows: int | None = None,
    execution_batch_coalesced: bool = False,
    execution_batch_split: bool = False,
) -> dict[str, object]:
    return {
        "phase4_semantic_batch_count": int(semantic_batch_count),
        "phase4_semantic_batch_max_rows": int(semantic_batch_max_rows),
        "phase4_semantic_batch_index_start": semantic_batch_index_start,
        "phase4_semantic_batch_index_end": semantic_batch_index_end,
        "phase4_semantic_batch_rows": semantic_batch_rows,
        "phase4_execution_batch_index": execution_batch_index,
        "phase4_execution_batch_count": execution_batch_count,
        "phase4_execution_batch_rows": execution_batch_rows,
        "phase4_execution_batch_max_rows": execution_batch_max_rows,
        "phase4_execution_batch_coalesced": bool(execution_batch_coalesced),
        "phase4_execution_batch_split": bool(execution_batch_split),
    }


def _record_cross_cluster_checkpoint(
    *,
    cross_cluster_debug_summary: dict[str, object] | None,
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None,
    checkpoint_name: str,
    phase: str,
    summary_payload: dict[str, object] | None,
    stream_payload: dict[str, object] | None = None,
) -> None:
    if cross_cluster_debug_summary is not None and summary_payload is not None:
        checkpoints = cross_cluster_debug_summary.setdefault("checkpoints", {})
        assert isinstance(checkpoints, dict)
        checkpoints[checkpoint_name] = summary_payload

    if cross_cluster_debug_checkpoints is None:
        return

    payload = stream_payload if stream_payload is not None else summary_payload
    if payload is None:
        payload = {}
    record: dict[str, object] = {
        "checkpoint_name": checkpoint_name,
        "phase": phase,
    }
    record.update(payload)
    cross_cluster_debug_checkpoints.append(record)


def _record_cross_cluster_batch_event(
    *,
    cross_cluster_debug_batches: list[dict[str, object]] | None,
    event_name: str,
    phase: str,
    event_index: int,
    payload: dict[str, object],
) -> None:
    if cross_cluster_debug_batches is None:
        return

    record: dict[str, object] = {
        "event_name": event_name,
        "phase": phase,
        "event_index": int(event_index),
    }
    record.update(payload)
    cross_cluster_debug_batches.append(record)


def _hash_json_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _build_cross_cluster_runtime_snapshot(
    *,
    observer: TraceObserver,
    device: torch.device | None,
    ctx=None,
    transcoder=None,
) -> tuple[dict[str, object], dict[str, object]]:
    return cast(
        tuple[dict[str, object], dict[str, object]],
        observer.observe(
            RuntimeSnapshot(device=device, context=ctx, transcoder=transcoder)
        ),
    )
