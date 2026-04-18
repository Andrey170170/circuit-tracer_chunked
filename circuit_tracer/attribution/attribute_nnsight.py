"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import hashlib
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal, cast

import numpy as np
import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.graph import (
    Graph,
    compute_partial_feature_influences,
    compute_partial_feature_influences_streaming,
    compute_partial_influences,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.utils.telemetry import (
    TelemetryRecorder,
    diff_numeric_metrics,
    format_memory_snapshot,
    format_numeric_metrics,
)


def _log_phase_metrics(logger, label: str, phase_start: float, device, **extra):
    logger.info(
        f"{label} completed in {time.perf_counter() - phase_start:.2f}s | "
        f"{format_memory_snapshot(device=device, extra=extra)}"
    )


def _log_memory_boundary(logger, label: str, device, **extra) -> None:
    logger.info(f"{label} | {format_memory_snapshot(device=device, extra=extra)}")


def _snapshot_diagnostics(obj) -> dict[str, object] | None:
    if obj is None or not hasattr(obj, "get_diagnostic_snapshot"):
        return None
    return obj.get_diagnostic_snapshot()


def _log_batch_profile(
    logger,
    label: str,
    batch_idx: int,
    total_batches: int,
    elapsed: float,
    ctx_before: dict[str, object] | None,
    ctx_after: dict[str, object] | None,
    transcoder_before: dict[str, object] | None,
    transcoder_after: dict[str, object] | None,
):
    parts = [f"{label} batch {batch_idx}/{total_batches} in {elapsed:.2f}s"]
    ctx_delta = diff_numeric_metrics(ctx_before, ctx_after) if ctx_after is not None else {}
    transcoder_delta = (
        diff_numeric_metrics(transcoder_before, transcoder_after)
        if transcoder_after is not None
        else {}
    )
    if ctx_delta:
        parts.append(f"ctx[{format_numeric_metrics(ctx_delta, limit=12)}]")
    if transcoder_delta:
        parts.append(f"transcoder[{format_numeric_metrics(transcoder_delta, limit=12)}]")
    logger.info(" | ".join(parts))


def _log_sparsification_profile(logger, stats: dict[str, object]) -> None:
    retained = stats.get("per_layer_retained_counts", {})
    logger.info(
        "Sparsification screening | "
        f"candidates={stats.get('candidate_count_before')}->{stats.get('candidate_count_after')} | "
        f"per_layer_position_topk={stats.get('per_layer_position_topk')} | "
        f"global_cap={stats.get('global_cap')} | "
        f"retained_activation_mass={stats.get('retained_activation_mass', 1.0):.4f} | "
        f"screen_seconds={stats.get('screen_seconds', 0.0):.4f} | "
        f"per_layer_retained={retained}"
    )


class _FileBackedFeatureRowStore:
    """Append-only dense feature-row store backed by a temporary memmap file."""

    def __init__(
        self,
        *,
        n_rows: int,
        n_feature_columns: int,
        dtype: torch.dtype,
        telemetry_recorder: TelemetryRecorder | None = None,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported feature row store dtype: {dtype}")

        self.n_rows = n_rows
        self.n_feature_columns = n_feature_columns
        self.row_abs_sums = torch.zeros(n_rows, dtype=dtype)

        self._dtype = dtype
        self._np_dtype = np.float32 if dtype == torch.float32 else np.float64
        self._tmpdir = tempfile.TemporaryDirectory(prefix="ct_feature_rows_")
        self._path = f"{self._tmpdir.name}/feature_rows.memmap"
        self._rows: np.memmap | None = np.memmap(
            self._path,
            mode="w+",
            dtype=self._np_dtype,
            shape=(n_rows, n_feature_columns),
        )
        self._telemetry_recorder = telemetry_recorder
        self._closed = False
        self._diagnostic_stats: dict[str, float | int | None] = {
            "append_call_count": 0,
            "append_row_count": 0,
            "read_call_count": 0,
            "read_row_count": 0,
            "read_last_row_start": None,
            "read_last_row_end": None,
            "materialize_call_count": 0,
            "materialize_row_count": 0,
            "materialize_column_count": 0,
            "materialize_last_row_start": None,
            "materialize_last_row_end": None,
        }

    def _telemetry_timer(
        self,
        *,
        name: str,
        phase: str | None,
        attrs: dict[str, object],
    ):
        if self._telemetry_recorder is None:
            return nullcontext()
        return self._telemetry_recorder.timer(
            scope="op",
            name=name,
            phase=phase,
            attrs=attrs,
        )

    @property
    def path(self) -> str:
        return self._path

    @property
    def nbytes(self) -> int:
        rows = self._require_open_rows()
        return int(rows.size * rows.dtype.itemsize)

    def _require_open_rows(self) -> np.memmap:
        if self._closed or self._rows is None:
            raise RuntimeError("feature row store has been cleaned up")
        return self._rows

    def append_rows(
        self,
        *,
        row_start: int,
        feature_rows: torch.Tensor,
        full_row_abs_sums: torch.Tensor,
        phase: str | None = None,
    ) -> None:
        if feature_rows.ndim != 2:
            raise ValueError("feature_rows must be rank-2")
        row_count, n_feature_cols = feature_rows.shape
        if n_feature_cols != self.n_feature_columns:
            raise ValueError(
                "feature_rows second dimension must equal configured n_feature_columns"
            )
        if full_row_abs_sums.numel() != row_count:
            raise ValueError("full_row_abs_sums length must equal number of feature_rows")
        row_end = row_start + row_count
        if row_start < 0 or row_end > self.n_rows:
            raise ValueError("row range is out of bounds for file-backed store")

        with self._telemetry_timer(
            name="feature_row_store.append_rows",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_count,
                "feature_columns": n_feature_cols,
            },
        ):
            rows = self._require_open_rows()
            feature_rows_cpu = feature_rows.to(
                device=self.row_abs_sums.device,
                dtype=self._dtype,
            ).contiguous()
            rows[row_start:row_end] = feature_rows_cpu.numpy()

            self.row_abs_sums[row_start:row_end] = full_row_abs_sums.to(
                device=self.row_abs_sums.device,
                dtype=self.row_abs_sums.dtype,
            )

        self._diagnostic_stats["append_call_count"] = (
            int(self._diagnostic_stats["append_call_count"] or 0) + 1
        )
        self._diagnostic_stats["append_row_count"] = int(
            self._diagnostic_stats["append_row_count"] or 0
        ) + int(row_count)

    def read_feature_rows(
        self,
        row_start: int,
        row_end: int,
        *,
        phase: str | None = None,
    ) -> torch.Tensor:
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for file-backed store")

        with self._telemetry_timer(
            name="feature_row_store.read_rows",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_end - row_start,
            },
        ):
            rows = self._require_open_rows()
            result = torch.from_numpy(np.asarray(rows[row_start:row_end], dtype=self._np_dtype))

        self._diagnostic_stats["read_call_count"] = (
            int(self._diagnostic_stats["read_call_count"] or 0) + 1
        )
        self._diagnostic_stats["read_row_count"] = int(
            self._diagnostic_stats["read_row_count"] or 0
        ) + int(row_end - row_start)
        self._diagnostic_stats["read_last_row_start"] = int(row_start)
        self._diagnostic_stats["read_last_row_end"] = int(row_end)
        return result

    def materialize_dense_feature_slice(
        self,
        *,
        row_start: int,
        row_end: int,
        selected_feature_columns: torch.Tensor,
        col_chunk_size: int = 2048,
        phase: str | None = None,
    ) -> torch.Tensor:
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for file-backed store")
        if col_chunk_size <= 0:
            raise ValueError("col_chunk_size must be > 0")

        n_rows = row_end - row_start
        n_cols = selected_feature_columns.numel()
        dense = torch.zeros((n_rows, n_cols), dtype=self.row_abs_sums.dtype)
        if n_rows == 0 or n_cols == 0:
            return dense

        selected_cols = selected_feature_columns.to(dtype=torch.long, device="cpu")
        if selected_cols.min() < 0 or selected_cols.max() >= self.n_feature_columns:
            raise ValueError("selected feature column indices must be in [0, n_feature_columns)")

        with self._telemetry_timer(
            name="feature_row_store.materialize_dense_slice",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": n_rows,
                "selected_columns": n_cols,
                "col_chunk_size": col_chunk_size,
            },
        ):
            rows = self._require_open_rows()
            for col_start in range(0, n_cols, col_chunk_size):
                col_end = min(col_start + col_chunk_size, n_cols)
                cols_np = selected_cols[col_start:col_end].numpy()
                chunk_np = np.asarray(rows[row_start:row_end, cols_np], dtype=self._np_dtype)
                dense[:, col_start:col_end] = torch.from_numpy(chunk_np)

        self._diagnostic_stats["materialize_call_count"] = (
            int(self._diagnostic_stats["materialize_call_count"] or 0) + 1
        )
        self._diagnostic_stats["materialize_row_count"] = int(
            self._diagnostic_stats["materialize_row_count"] or 0
        ) + int(n_rows)
        self._diagnostic_stats["materialize_column_count"] = int(
            self._diagnostic_stats["materialize_column_count"] or 0
        ) + int(n_cols)
        self._diagnostic_stats["materialize_last_row_start"] = int(row_start)
        self._diagnostic_stats["materialize_last_row_end"] = int(row_end)

        return dense

    def get_diagnostic_snapshot(self) -> dict[str, float | int | None]:
        return dict(self._diagnostic_stats)

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True

        rows = self._rows
        self._rows = None
        if rows is not None:
            try:
                rows.flush()
            except Exception:
                pass

        self._tmpdir.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


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


def _resolve_phase4_feature_batch_planner_enabled(
    *,
    plan_feature_batch_size: bool,
    auto_scale_feature_batch_size: bool,
) -> bool:
    # Backward compatibility: keep legacy flag as an alias for fixed preflight planning.
    return bool(plan_feature_batch_size or auto_scale_feature_batch_size)


def _parse_env_bool(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _resolve_telemetry_max_events(
    *,
    telemetry_max_events: int | None,
    compact_output: bool,
    exact_chunked_decoder: bool,
    profile: bool,
    phase4_anomaly_debug_enabled: bool,
) -> int:
    if telemetry_max_events is not None and telemetry_max_events > 0:
        return int(telemetry_max_events)

    env_override = _parse_env_int("CIRCUIT_TRACER_TELEMETRY_MAX_EVENTS")
    if env_override is not None:
        return env_override

    if compact_output and exact_chunked_decoder:
        return 120_000
    if profile or phase4_anomaly_debug_enabled:
        return 60_000
    return 20_000


def _resolve_phase4_anomaly_debug_enabled(phase4_anomaly_debug: bool) -> bool:
    return bool(phase4_anomaly_debug or _parse_env_bool("PHASE4_ANOMALY_DEBUG"))


def _hash_index_tensor(indices: torch.Tensor) -> str:
    indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.blake2s(indices_cpu.numpy().tobytes(), digest_size=8).hexdigest()


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


def _build_phase4_normalization_stats(
    row_abs_sums: torch.Tensor,
    *,
    clamp_epsilon: float = 1e-8,
) -> dict[str, object]:
    stats = _build_vector_stats(row_abs_sums, epsilon=clamp_epsilon)
    count = int(stats.get("count", 0) or 0)
    effective_zero_count = int(stats.get("effective_zero_count", 0) or 0)
    clamped_fraction = (effective_zero_count / count) if count else 0.0
    stats["clamp_epsilon"] = float(clamp_epsilon)
    stats["clamped_row_count"] = effective_zero_count
    stats["clamped_row_fraction"] = float(clamped_fraction)
    return stats


def _safe_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.item())
    return float(value)


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
    records = anomaly_debug_result.setdefault("records", [])
    assert isinstance(records, list)
    records.append(record)


def _compare_phase4_frontiers(
    actual_pending: torch.Tensor,
    shadow_pending: torch.Tensor,
) -> dict[str, object]:
    actual_cpu = actual_pending.detach().to(device="cpu", dtype=torch.int64)
    shadow_cpu = shadow_pending.detach().to(device="cpu", dtype=torch.int64)
    actual_set = set(int(value) for value in actual_cpu.tolist())
    shadow_set = set(int(value) for value in shadow_cpu.tolist())
    overlap_count = len(actual_set & shadow_set)
    overlap_fraction = overlap_count / len(actual_set) if actual_set else None
    first_differing_rank = None
    for idx, (actual_value, shadow_value) in enumerate(
        zip(actual_cpu.tolist(), shadow_cpu.tolist())
    ):
        if actual_value != shadow_value:
            first_differing_rank = int(idx)
            break
    return {
        "actual_hash": _hash_index_tensor(actual_cpu),
        "shadow_hash": _hash_index_tensor(shadow_cpu),
        "overlap_count": int(overlap_count),
        "overlap_fraction": overlap_fraction,
        "changed_selected_nodes": int(len(actual_set ^ shadow_set)),
        "first_differing_rank": first_differing_rank,
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
        "attribute_nnsight_file": __file__,
    }


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


def _get_cuda_reserved_snapshot() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    peak_reserved = int(torch.cuda.memory_reserved(device_index))
    total_cuda_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    return peak_reserved, total_cuda_bytes


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
        max_probe_candidates = min(
            int(pending.numel()),
            initial_feature_batch_size * feature_batch_probe_batches,
        )
        return pending[:max_probe_candidates]

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

    max_probe_candidates = min(
        int(pending.numel()),
        initial_feature_batch_size * feature_batch_probe_batches,
    )
    return pending[:max_probe_candidates]


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
    logger,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    diagnostic_feature_cap: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    telemetry_recorder: TelemetryRecorder | None = None,
) -> int:
    planner_start = time.perf_counter()

    def _finalize_planner(
        *,
        planned_feature_batch_size: int,
        planner_status: str,
        attrs: dict[str, object] | None = None,
    ) -> int:
        if telemetry_recorder is not None:
            payload = {
                "planner_status": planner_status,
                "planned_feature_batch_size": planned_feature_batch_size,
            }
            if attrs:
                payload.update(attrs)
            telemetry_recorder.record_event(
                scope="phase",
                name="phase4.planner.preflight",
                phase="phase4",
                elapsed_ms=(time.perf_counter() - planner_start) * 1000.0,
                attrs=payload,
            )
        return planned_feature_batch_size

    if not torch.cuda.is_available():
        logger.info(
            "Phase 4 planner skipped (CUDA unavailable); using fixed feature batch size "
            f"{min(initial_feature_batch_size, max_feature_batch_size)}"
        )
        return _finalize_planner(
            planned_feature_batch_size=min(initial_feature_batch_size, max_feature_batch_size),
            planner_status="skipped_cuda_unavailable",
        )

    input_ids = model.ensure_tokenized(prompt)
    ctx = None
    observed_peak_reserved_bytes = 0
    total_cuda_bytes: int | None = None

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(None, telemetry_recorder=telemetry_recorder)

    try:
        logger.info(
            "Phase 4 planner preflight | "
            f"initial_feature_batch_size={initial_feature_batch_size} | "
            f"max_feature_batch_size={max_feature_batch_size} | "
            f"max_feature_nodes={max_feature_nodes} | "
            f"update_interval={update_interval} | "
            f"probe_batches={feature_batch_probe_batches} | "
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
        )
        if hasattr(ctx, "set_diagnostic_mode"):
            ctx.set_diagnostic_mode(False)
        configure_ctx_trace_logging = getattr(ctx, "configure_trace_logging", None)
        if callable(configure_ctx_trace_logging):
            configure_ctx_trace_logging(None, telemetry_recorder=telemetry_recorder)

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
            with tracer.invoke(input_ids.expand(trace_batch_size, -1)):
                pass

            detach_barrier = tracer.barrier(2)
            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            ctx.cache_residual(model, tracer, barrier=detach_barrier)

        exact_chunked_decoder = bool(getattr(model.transcoders, "exact_chunked_decoder", False))
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
            logit_feature_rows = torch.zeros((n_logits, total_active_feats), dtype=torch.float32)
            logit_row_abs_sums = torch.zeros(n_logits, dtype=torch.float32)
            row_to_node_index = torch.arange(n_logits, dtype=torch.long) + int(logit_offset)
            for i in range(0, n_logits, effective_logit_batch_size):
                batch = targets.logit_vectors[i : i + effective_logit_batch_size]
                rows = ctx.compute_batch(
                    layers=torch.full((batch.shape[0],), n_layers),
                    positions=torch.full((batch.shape[0],), n_pos - 1),
                    inject_values=batch,
                    retain_graph=True,
                    phase_label="phase3_logits_probe",
                )
                rows_cpu = rows.to(device="cpu", dtype=torch.float32)
                end = i + batch.shape[0]
                logit_feature_rows[i:end] = rows_cpu[:, :total_active_feats]
                logit_row_abs_sums[i:end] = rows_cpu[:, :logit_offset].abs().sum(dim=1)

            feature_influences = compute_partial_feature_influences(
                logit_feature_rows,
                logit_row_abs_sums,
                targets.logit_probabilities.detach().cpu().to(dtype=torch.float32),
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

            device_index = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device_index)
            probe_batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.materialize_encoder_vectors(idx_batch),
                retain_graph=(probe_idx + 1) < feature_batch_probe_batches,
                phase_label="phase4_probe",
            )
            del rows
            torch.cuda.synchronize(device_index)
            observed_peak_reserved_bytes = max(
                observed_peak_reserved_bytes,
                int(torch.cuda.max_memory_reserved(device_index)),
            )
            probe_batches_ran += 1
            if telemetry_recorder is not None:
                telemetry_recorder.record_event(
                    scope="batch",
                    name="phase4.planner.probe_batch",
                    phase="phase4",
                    batch_index=probe_batches_ran,
                    elapsed_ms=(time.perf_counter() - probe_batch_start) * 1000.0,
                    attrs={
                        "batch_nodes": int(idx_batch.numel()),
                        "observed_peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device_index)
                        ),
                    },
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

        cuda_snapshot = _get_cuda_reserved_snapshot()
        observed_reserved_bytes: int | None = None
        if cuda_snapshot is not None:
            current_reserved_bytes, total_cuda_bytes = cuda_snapshot
            observed_reserved_bytes = max(current_reserved_bytes, observed_peak_reserved_bytes)

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


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: NNSightReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    plan_feature_batch_size: bool = False,
    auto_scale_feature_batch_size: bool = False,
    feature_batch_size_max: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    phase4_anomaly_debug: bool = False,
    telemetry_max_events: int | None = None,
    compact_output: bool = False,
) -> Graph:
    """Compute an attribution graph for *prompt* using NNSight backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``NNSightReplacementModel``
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or tuple)
                            with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        feature_batch_size: Optional override for feature-attribution batches.
            Defaults to ``batch_size`` when omitted.
        logit_batch_size: Optional override for logit-attribution batches.
            Defaults to ``batch_size`` when omitted.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        profile: Whether to emit batch-level diagnostic profiling logs.
        profile_log_interval: Log every N batches when profiling.
        diagnostic_feature_cap: Optional debug-only early cap on active features.
            This changes attribution semantics and should only be used for profiling.
        sparsification: Optional candidate-screening config applied before
            reconstruction and reused by later attribution phases.
        chunked_feature_replay_window: Exact-mode knob controlling how many
            layer grads are buffered before chunked feature replay flush.
        error_vector_prefetch_lookahead: Exact-mode knob controlling staged
            error-vector lookahead window size.
        stage_encoder_vecs_on_cpu: Exact-mode knob to force/disable CPU staging
            of encoder vectors. ``None`` preserves backend default behavior.
        stage_error_vectors_on_cpu: Exact-mode knob to force/disable CPU
            staging of error vectors. ``None`` preserves backend defaults.
        row_subchunk_size: Optional exact-mode knob controlling inner replay
            row subchunk size. ``None`` preserves current behavior (equal to
            decoder chunk size).
        plan_feature_batch_size: Whether to run a probe-based preflight and
            choose a single fixed Phase-4 feature microbatch size for the run.
        auto_scale_feature_batch_size: Legacy alias for
            ``plan_feature_batch_size`` (kept for backward compatibility).
        feature_batch_size_max: Optional upper bound for the preflight-planned
            Phase-4 feature microbatch size.
        feature_batch_target_reserved_fraction: Reserved-memory utilization
            target used by the planner (0-1).
        feature_batch_min_free_fraction: Minimum free-memory fraction to keep
            unused (0-1), applied as a stricter cap than target utilization.
        feature_batch_probe_batches: Number of preflight Phase-4 probe batches
            to run before the real attribution pass.
        phase4_anomaly_debug: Enable opt-in Phase-4 anomaly debug scaffolding.
            Can also be activated via ``PHASE4_ANOMALY_DEBUG=1``.
        telemetry_max_events: Optional cap for in-memory telemetry event storage.
            If omitted, an environment/default policy is used.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if (verbose or profile) and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            feature_batch_size=feature_batch_size,
            logit_batch_size=logit_batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            profile=profile,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            sparsification=sparsification,
            chunked_feature_replay_window=chunked_feature_replay_window,
            error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
            row_subchunk_size=row_subchunk_size,
            plan_feature_batch_size=plan_feature_batch_size,
            auto_scale_feature_batch_size=auto_scale_feature_batch_size,
            feature_batch_size_max=feature_batch_size_max,
            feature_batch_target_reserved_fraction=feature_batch_target_reserved_fraction,
            feature_batch_min_free_fraction=feature_batch_min_free_fraction,
            feature_batch_probe_batches=feature_batch_probe_batches,
            phase4_anomaly_debug=phase4_anomaly_debug,
            telemetry_max_events=telemetry_max_events,
            compact_output=compact_output,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _run_attribution(
    model: NNSightReplacementModel,
    prompt,
    attribution_targets,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
    feature_batch_size: int | None,
    logit_batch_size: int | None,
    max_feature_nodes: int | None,
    offload: Literal["cpu", "disk", None],
    verbose: bool,
    offload_handles,
    logger,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    plan_feature_batch_size: bool = False,
    auto_scale_feature_batch_size: bool = False,
    feature_batch_size_max: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    phase4_anomaly_debug: bool = False,
    telemetry_max_events: int | None = None,
    compact_output: bool = False,
):
    start_time = time.time()
    run_start = time.perf_counter()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if feature_batch_size is not None and feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be > 0 when provided")
    if logit_batch_size is not None and logit_batch_size <= 0:
        raise ValueError("logit_batch_size must be > 0 when provided")
    if chunked_feature_replay_window <= 0:
        raise ValueError("chunked_feature_replay_window must be > 0")
    if error_vector_prefetch_lookahead <= 0:
        raise ValueError("error_vector_prefetch_lookahead must be > 0")
    if row_subchunk_size is not None and row_subchunk_size <= 0:
        raise ValueError("row_subchunk_size must be > 0 when provided")
    if feature_batch_size_max is not None and feature_batch_size_max <= 0:
        raise ValueError("feature_batch_size_max must be > 0 when provided")
    if not 0.0 < feature_batch_target_reserved_fraction < 1.0:
        raise ValueError("feature_batch_target_reserved_fraction must be in (0, 1)")
    if not 0.0 <= feature_batch_min_free_fraction < 1.0:
        raise ValueError("feature_batch_min_free_fraction must be in [0, 1)")
    if feature_batch_probe_batches <= 0:
        raise ValueError("feature_batch_probe_batches must be > 0")

    phase4_anomaly_debug_enabled = _resolve_phase4_anomaly_debug_enabled(phase4_anomaly_debug)
    telemetry_max_events_resolved = _resolve_telemetry_max_events(
        telemetry_max_events=telemetry_max_events,
        compact_output=compact_output,
        exact_chunked_decoder=bool(getattr(model.transcoders, "exact_chunked_decoder", False)),
        profile=profile,
        phase4_anomaly_debug_enabled=phase4_anomaly_debug_enabled,
    )
    telemetry_recorder = TelemetryRecorder(
        enabled=(profile or compact_output or phase4_anomaly_debug_enabled),
        max_events=telemetry_max_events_resolved,
    )
    telemetry_recorder.record_event(
        scope="run",
        name="attribute.start",
        attrs={
            "profile": profile,
            "compact_output": compact_output,
            "batch_size": batch_size,
            "feature_batch_size": feature_batch_size,
            "logit_batch_size": logit_batch_size,
            "telemetry_max_events": telemetry_max_events_resolved,
        },
    )

    effective_feature_batch_size = batch_size if feature_batch_size is None else feature_batch_size
    max_phase4_feature_batch_size = (
        effective_feature_batch_size if feature_batch_size_max is None else feature_batch_size_max
    )
    planner_enabled = _resolve_phase4_feature_batch_planner_enabled(
        plan_feature_batch_size=plan_feature_batch_size,
        auto_scale_feature_batch_size=auto_scale_feature_batch_size,
    )
    if auto_scale_feature_batch_size and not plan_feature_batch_size:
        logger.info(
            "Phase-4 feature batch planning | "
            "legacy auto_scale_feature_batch_size flag detected; "
            "using fixed preflight planner semantics"
        )
    if (not planner_enabled) and max_phase4_feature_batch_size < effective_feature_batch_size:
        raise ValueError("feature_batch_size_max must be >= the effective feature batch size")
    effective_logit_batch_size = batch_size if logit_batch_size is None else logit_batch_size

    exact_chunked_decoder = bool(getattr(model.transcoders, "exact_chunked_decoder", False))
    planner_status, planner_skip_reason = _resolve_phase4_feature_batch_planner_status(
        planner_enabled=planner_enabled,
        effective_feature_batch_size=effective_feature_batch_size,
        max_feature_batch_size=max_phase4_feature_batch_size,
    )
    anomaly_debug_result: dict[str, object] | None = None
    if phase4_anomaly_debug_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "Phase-4 anomaly debug requires compact_output=True and exact_chunked_decoder=True"
        )
    if phase4_anomaly_debug_enabled:
        anomaly_debug_result = {
            "schema_version": 2,
            "enabled": True,
            "mode": "phase4_shadow_debug",
            "status": "scaffold",
            "shadow_execution": False,
            "refresh_count": 0,
            "environment": _build_phase4_environment_fingerprint(),
            "summary": {},
            "records": [],
        }
    if planner_enabled and not (compact_output and exact_chunked_decoder):
        raise ValueError(
            "Phase-4 feature batch planner requires compact_output=True and exact_chunked_decoder=True"
        )
    if planner_enabled:
        if planner_status == "skipped_no_headroom":
            logger.info(
                "Phase-4 feature batch planner skipped | "
                f"status={planner_status} | "
                f"initial_feature_batch_size={effective_feature_batch_size} | "
                f"feature_batch_size_max={max_phase4_feature_batch_size} | "
                f"reason={planner_skip_reason}"
            )
            telemetry_recorder.record_event(
                scope="phase",
                name="phase4.planner.preflight",
                phase="phase4",
                attrs={
                    "planner_status": planner_status,
                    "planned_feature_batch_size": effective_feature_batch_size,
                    "planner_skip_reason": planner_skip_reason,
                },
            )
        else:
            planner_probe_feature_batch_size = min(
                effective_feature_batch_size,
                max_phase4_feature_batch_size,
            )
            effective_feature_batch_size = _plan_phase4_feature_batch_size_preflight(
                model=model,
                prompt=prompt,
                attribution_targets=attribution_targets,
                batch_size=batch_size,
                initial_feature_batch_size=planner_probe_feature_batch_size,
                effective_logit_batch_size=effective_logit_batch_size,
                max_feature_batch_size=max_phase4_feature_batch_size,
                max_feature_nodes=max_feature_nodes,
                update_interval=update_interval,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                logger=logger,
                sparsification=sparsification,
                chunked_feature_replay_window=chunked_feature_replay_window,
                error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
                stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
                stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
                row_subchunk_size=row_subchunk_size,
                diagnostic_feature_cap=diagnostic_feature_cap,
                feature_batch_target_reserved_fraction=feature_batch_target_reserved_fraction,
                feature_batch_min_free_fraction=feature_batch_min_free_fraction,
                feature_batch_probe_batches=feature_batch_probe_batches,
                telemetry_recorder=telemetry_recorder,
            )
            planner_status = "executed"

    trace_batch_size = max(
        batch_size,
        effective_feature_batch_size,
        effective_logit_batch_size,
    )
    ctx = None
    feature_row_store: _FileBackedFeatureRowStore | None = None
    compact_output_result: dict[str, object] | None = None

    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.perf_counter()
    input_ids = model.ensure_tokenized(prompt)
    _log_memory_boundary(logger, "Phase 0 start", model.device)

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(
            logger.info if profile else None,
            telemetry_recorder=telemetry_recorder,
        )

    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()

    if profile:
        logger.info(
            "Profiling enabled | "
            f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
            f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
            f"exact_chunked_decoder={getattr(model.transcoders, 'exact_chunked_decoder', False)} | "
            f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
            f"decoder_cache_bytes={getattr(model.transcoders, 'cross_batch_decoder_cache_bytes', 0)} | "
            f"chunked_feature_replay_window={chunked_feature_replay_window} | "
            f"error_vector_prefetch_lookahead={error_vector_prefetch_lookahead} | "
            f"stage_encoder_vecs_on_cpu={stage_encoder_vecs_on_cpu} | "
            f"stage_error_vectors_on_cpu={stage_error_vectors_on_cpu} | "
            f"row_subchunk_size={row_subchunk_size} | "
            f"planner_enabled={planner_enabled} | "
            f"feature_batch_size_max={max_phase4_feature_batch_size} | "
            f"prompt_tokens={input_ids.shape[-1]} | feature_batch_size={effective_feature_batch_size} | "
            f"logit_batch_size={effective_logit_batch_size}"
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
    )
    if hasattr(ctx, "set_diagnostic_mode"):
        ctx.set_diagnostic_mode(profile)
    configure_ctx_trace_logging = getattr(ctx, "configure_trace_logging", None)
    if callable(configure_ctx_trace_logging):
        configure_ctx_trace_logging(
            logger.info if profile else None,
            telemetry_recorder=telemetry_recorder,
        )

    if diagnostic_feature_cap is not None and diagnostic_feature_cap > 0:
        before_cap, after_cap = ctx.apply_diagnostic_feature_cap(diagnostic_feature_cap)
        logger.info(
            f"Diagnostic feature cap applied before attribution rows: {before_cap} -> {after_cap} active features"
        )
    if profile and getattr(ctx, "sparsification_stats", None):
        _log_sparsification_profile(logger, ctx.sparsification_stats)

    try:
        activation_matrix = ctx.activation_matrix

        _log_phase_metrics(
            logger,
            "Precomputation",
            phase_start,
            model.device,
            active_features=ctx.activation_matrix._nnz(),
            logit_retention=getattr(ctx, "logit_retention", "full"),
        )
        phase0_elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase0.precompute",
            phase="phase0",
            elapsed_ms=phase0_elapsed_ms,
            attrs={
                "active_features": int(ctx.activation_matrix._nnz()),
                "logit_retention": getattr(ctx, "logit_retention", "full"),
            },
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase0.precompute",
            phase="phase0",
            elapsed_ms=phase0_elapsed_ms,
        )
        if profile:
            if getattr(ctx, "setup_diagnostic_stats", None):
                logger.info(
                    f"Phase 0 setup diagnostics | {format_numeric_metrics(ctx.setup_diagnostic_stats, limit=20)}"
                )
            transcoder_snapshot = _snapshot_diagnostics(model.transcoders)
            if transcoder_snapshot:
                logger.info(
                    f"Precompute diagnostics | {format_numeric_metrics(transcoder_snapshot, limit=20)}"
                )
        logger.info(f"Found {ctx.activation_matrix._nnz()} active features")

        if (
            offload
            and not model.skip_transcoder
            and not getattr(model.transcoders, "exact_chunked_decoder", False)
        ):
            offload_handles += offload_modules(model.transcoders, offload)

        # Phase 1: forward pass
        logger.info("Phase 1: Running forward pass")
        phase_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 1 start", model.device)
        with model.trace() as tracer:
            with tracer.invoke(input_ids.expand(trace_batch_size, -1)):
                pass

            detach_barrier = tracer.barrier(2)

            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            ctx.cache_residual(model, tracer, barrier=detach_barrier)

        _log_phase_metrics(logger, "Forward pass", phase_start, model.device)
        phase1_elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase1.forward_pass",
            phase="phase1",
            elapsed_ms=phase1_elapsed_ms,
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase1.forward_pass",
            phase="phase1",
            elapsed_ms=phase1_elapsed_ms,
        )

        if offload:
            offload_handles += offload_modules(
                [layer.mlp for layer in getattr(model.pre_logit_location, "layers")], offload
            )
            if model.skip_transcoder and not getattr(
                model.transcoders, "exact_chunked_decoder", False
            ):
                offload_handles += offload_modules(model.transcoders, offload)

        # Phase 2: build input vector list
        logger.info("Phase 2: Building input vectors")
        phase2_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 2 start", model.device)
        feat_layers, feat_pos, feat_ids = activation_matrix.indices()
        n_layers, n_pos, _ = activation_matrix.shape
        total_active_feats = activation_matrix._nnz()

        # Create AttributionTargets using NNSight's unembed_weight accessor
        targets = AttributionTargets(
            attribution_targets=attribution_targets,
            logits=ctx.get_last_token_logits()[0],
            unembed_proj=cast(torch.Tensor, model.unembed_weight),  # NNSight uses unembed_weight
            tokenizer=model.tokenizer,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )

        log_attribution_target_info(targets, attribution_targets, logger)

        if offload:
            offload_handles += offload_modules([model.embed_location], offload)
            tied_embeds = (
                model.embed_weight.untyped_storage().data_ptr()  # type:ignore
                == model.unembed_weight.untyped_storage().data_ptr()  # type:ignore
            )
            if not tied_embeds:
                offload_handles += offload_modules([model.lm_head], offload)

        logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
        n_logits = len(targets)
        total_nodes = logit_offset + n_logits

        actual_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
        logger.info(
            f"Will include {actual_max_feature_nodes} of {total_active_feats} feature nodes"
        )

        use_compact_feature_row_store = compact_output and bool(
            getattr(model.transcoders, "exact_chunked_decoder", False)
        )
        if use_compact_feature_row_store:
            # Benchmark-critical path only: exact chunked decoder + compact output.
            # Keep dense full-row behavior unchanged for non-compact Graph outputs.
            assert compact_output
            assert bool(getattr(model.transcoders, "exact_chunked_decoder", False))
            feature_row_store = _FileBackedFeatureRowStore(
                n_rows=actual_max_feature_nodes + n_logits,
                n_feature_columns=total_active_feats,
                dtype=torch.float32,
                telemetry_recorder=telemetry_recorder,
            )
        else:
            edge_matrix = torch.zeros(actual_max_feature_nodes + n_logits, total_nodes)

        # Maps stored row indices to original feature/node indices.
        # First populated with logit node IDs, then feature IDs in attribution order
        row_to_node_index = torch.zeros(actual_max_feature_nodes + n_logits, dtype=torch.int32)

        phase2_extra: dict[str, object] = {
            "row_store_mode": (
                "compact_feature_file_backed_dense"
                if use_compact_feature_row_store
                else "dense_full"
            )
        }
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            phase2_extra.update(
                feature_row_store="dense_memmap",
                feature_row_store_path=feature_row_store.path,
                row_abs_sums_shape=f"{tuple(feature_row_store.row_abs_sums.shape)}",
                feature_edge_columns=total_active_feats,
            )
        else:
            phase2_extra.update(
                edge_matrix_shape=f"{tuple(edge_matrix.shape)}",
                edge_matrix_dtype=edge_matrix.dtype,
            )

        _log_phase_metrics(
            logger,
            "Input vector build",
            phase2_start,
            model.device,
            **phase2_extra,
        )
        phase2_elapsed_ms = (time.perf_counter() - phase2_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase2.input_vector_build",
            phase="phase2",
            elapsed_ms=phase2_elapsed_ms,
            attrs=phase2_extra,
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase2.input_vector_build",
            phase="phase2",
            elapsed_ms=phase2_elapsed_ms,
        )

        # Phase 3: logit attribution
        logger.info("Phase 3: Computing logit attributions")
        phase3_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 3 start", model.device)
        i = -1
        total_logit_batches = max(
            (len(targets) + effective_logit_batch_size - 1) // effective_logit_batch_size,
            1,
        )
        for i in range(0, len(targets), effective_logit_batch_size):
            batch = targets.logit_vectors[i : i + effective_logit_batch_size]
            ctx_before = _snapshot_diagnostics(ctx) if profile else None
            transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
            batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), n_layers),
                positions=torch.full((batch.shape[0],), n_pos - 1),
                inject_values=batch,
                phase_label="phase3_logits",
            )
            rows_cpu = rows.cpu()
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                end = i + batch.shape[0]
                feature_row_store.append_rows(
                    row_start=i,
                    feature_rows=rows_cpu[:, :total_active_feats],
                    full_row_abs_sums=rows_cpu[:, :logit_offset].abs().sum(dim=1),
                    phase="phase3",
                )
            else:
                edge_matrix[i : i + batch.shape[0], :logit_offset] = rows_cpu
            row_to_node_index[i : i + batch.shape[0]] = (
                torch.arange(i, i + batch.shape[0]) + logit_offset
            )
            batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            telemetry_recorder.record_event(
                scope="batch",
                name="phase3.logit_batch",
                phase="phase3",
                batch_index=(i // effective_logit_batch_size) + 1,
                elapsed_ms=batch_elapsed_ms,
                attrs={
                    "batch_rows": int(batch.shape[0]),
                    "batch_start_index": int(i),
                    "total_logit_batches": int(total_logit_batches),
                },
            )
            telemetry_recorder.record_wall_clock_duration(
                scope="batch",
                name="phase3.logit_batch",
                elapsed_ms=batch_elapsed_ms,
            )
            if profile and ((i // effective_logit_batch_size) + 1) % profile_log_interval == 0:
                _log_batch_profile(
                    logger,
                    "Phase 3",
                    (i // effective_logit_batch_size) + 1,
                    total_logit_batches,
                    batch_elapsed_ms / 1000.0,
                    ctx_before,
                    _snapshot_diagnostics(ctx),
                    transcoder_before,
                    _snapshot_diagnostics(model.transcoders),
                )

        _log_phase_metrics(
            logger,
            f"{i + 1} logit attribution(s)",
            phase3_start,
            model.device,
        )
        phase3_elapsed_ms = (time.perf_counter() - phase3_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase3.logit_attribution",
            phase="phase3",
            elapsed_ms=phase3_elapsed_ms,
            attrs={"logit_count": int(len(targets)), "batches": int(total_logit_batches)},
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase3.logit_attribution",
            phase="phase3",
            elapsed_ms=phase3_elapsed_ms,
        )
        reset_decoder_cache = getattr(ctx, "reset_decoder_cache", None)
        if callable(reset_decoder_cache):
            reset_decoder_cache()

        # Phase 4: feature attribution
        logger.info("Phase 4: Computing feature attributions")
        phase4_start = time.perf_counter()
        _log_memory_boundary(logger, "Phase 4 start", model.device)
        decoder_chunk_size = getattr(model.transcoders, "decoder_chunk_size", None)
        phase4_feature_batch_size = effective_feature_batch_size
        logger.info(
            "Phase 4 frontier order | "
            "mode=fixed_frontier_locality | "
            f"exact_chunked_decoder={exact_chunked_decoder} | "
            f"decoder_chunk_size={decoder_chunk_size}"
        )
        logger.info(
            "Phase 4 feature batch mode | "
            f"planner_enabled={planner_enabled} | "
            f"planner_status={planner_status} | "
            f"fixed_feature_batch_size={phase4_feature_batch_size} | "
            f"max_feature_batch_size={max_phase4_feature_batch_size}"
            + (
                f" | planner_skip_reason={planner_skip_reason}"
                if planner_skip_reason is not None
                else ""
            )
        )
        st = n_logits
        visited = torch.zeros(total_active_feats, dtype=torch.bool)
        n_visited = 0
        phase4_batch_count = 0
        phase4_refresh_count = 0
        phase4_refresh_elapsed_ms_total = 0.0
        previous_phase4_pending: torch.Tensor | None = None
        first_phase4_pending: torch.Tensor | None = None
        phase4_logit_probability_stats = _build_vector_stats(
            targets.logit_probabilities.detach().cpu(),
            epsilon=1e-12,
            top_k=8,
        )
        if anomaly_debug_result is not None:
            anomaly_debug_result["logit_probability_stats"] = phase4_logit_probability_stats

        pbar = tqdm(
            total=actual_max_feature_nodes,
            desc="Feature influence computation",
            disable=not verbose,
        )

        while n_visited < actual_max_feature_nodes:
            if actual_max_feature_nodes == total_active_feats:
                pending = torch.arange(total_active_feats)
            else:
                refresh_start = time.perf_counter()
                feature_row_store_snapshot_before = (
                    feature_row_store.get_diagnostic_snapshot()
                    if use_compact_feature_row_store and feature_row_store is not None
                    else None
                )
                if use_compact_feature_row_store:
                    assert feature_row_store is not None
                    feature_influences = compute_partial_feature_influences_streaming(
                        lambda row_start, row_end: feature_row_store.read_feature_rows(
                            row_start,
                            row_end,
                            phase="phase4",
                        ),
                        feature_row_store.row_abs_sums[:st],
                        targets.logit_probabilities,
                        row_to_node_index[:st],
                        n_feature_nodes=total_active_feats,
                        n_logits=n_logits,
                        device=feature_row_store.row_abs_sums.device,
                    )
                else:
                    influences = compute_partial_influences(
                        edge_matrix[:st],
                        targets.logit_probabilities,
                        row_to_node_index[:st],
                        device=edge_matrix.device,
                    )
                    feature_influences = influences[:total_active_feats]

                feature_rank = torch.argsort(feature_influences, descending=True).cpu()
                unvisited_feature_rank = feature_rank[~visited[feature_rank]]
                candidate_scores = feature_influences[unvisited_feature_rank].detach().cpu()
                rank_signal_stats = _build_vector_stats(
                    candidate_scores,
                    epsilon=1e-12,
                    top_k=8,
                )
                if use_compact_feature_row_store:
                    assert feature_row_store is not None
                    normalization_input_stats = _build_phase4_normalization_stats(
                        feature_row_store.row_abs_sums[:st].detach().cpu(),
                    )
                else:
                    normalization_input_stats = _build_phase4_normalization_stats(
                        edge_matrix[:st, :logit_offset].abs().sum(dim=1).detach().cpu(),
                    )
                feature_row_store_snapshot_after = (
                    feature_row_store.get_diagnostic_snapshot()
                    if use_compact_feature_row_store and feature_row_store is not None
                    else None
                )
                feature_row_store_read_stats = (
                    diff_numeric_metrics(
                        feature_row_store_snapshot_before,
                        feature_row_store_snapshot_after,
                    )
                    if feature_row_store_snapshot_after is not None
                    else None
                )
                queue_size = min(
                    update_interval * phase4_feature_batch_size,
                    actual_max_feature_nodes - n_visited,
                )
                pending = unvisited_feature_rank[:queue_size]
                pending = _reorder_pending_for_phase4_locality(
                    pending,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=decoder_chunk_size,
                )
                refresh_elapsed_ms = (time.perf_counter() - refresh_start) * 1000.0
                phase4_refresh_elapsed_ms_total += refresh_elapsed_ms
                telemetry_recorder.record_event(
                    scope="batch",
                    name="phase4.refresh",
                    phase="phase4",
                    batch_index=phase4_refresh_count + 1,
                    elapsed_ms=refresh_elapsed_ms,
                    attrs={
                        "refresh_index": int(phase4_refresh_count),
                        "stored_rows": int(st),
                        "visited_features": int(n_visited),
                        "frontier_candidate_count": int(candidate_scores.numel()),
                        "queue_size": int(queue_size),
                        "rank_nonzero_count": int(rank_signal_stats["nonzero_count"]),
                        "rank_effective_nonzero_count": int(
                            rank_signal_stats["effective_nonzero_count"]
                        ),
                        "rank_max": _safe_float(rank_signal_stats.get("max")),
                        "rank_abs_sum": _safe_float(rank_signal_stats.get("abs_sum")),
                        "rank_all_zero": bool(rank_signal_stats["all_zero"]),
                        "rank_effectively_all_zero": bool(
                            rank_signal_stats["effectively_all_zero"]
                        ),
                        "normalization_clamped_row_count": int(
                            normalization_input_stats["clamped_row_count"]
                        ),
                        "normalization_clamped_row_fraction": _safe_float(
                            normalization_input_stats.get("clamped_row_fraction")
                        ),
                        "feature_row_store_read_calls": _safe_float(
                            (feature_row_store_read_stats or {}).get("read_call_count")
                        ),
                        "feature_row_store_read_rows": _safe_float(
                            (feature_row_store_read_stats or {}).get("read_row_count")
                        ),
                    },
                )
                telemetry_recorder.record_wall_clock_duration(
                    scope="batch",
                    name="phase4.refresh",
                    elapsed_ms=refresh_elapsed_ms,
                )
                if anomaly_debug_result is not None:
                    _record_phase4_refresh_debug(
                        anomaly_debug_result,
                        refresh_index=phase4_refresh_count,
                        n_visited=n_visited,
                        queue_size=queue_size,
                        pending=pending,
                        previous_pending=previous_phase4_pending,
                        first_pending=first_phase4_pending,
                        candidate_scores=candidate_scores,
                        refresh_elapsed_ms=refresh_elapsed_ms,
                        rank_signal_stats=rank_signal_stats,
                        logit_probability_stats=phase4_logit_probability_stats,
                        normalization_input_stats=normalization_input_stats,
                        feature_row_store_read_stats=feature_row_store_read_stats,
                    )
                    debug_records = anomaly_debug_result.get("records", [])
                    assert isinstance(debug_records, list) and debug_records
                    current_debug_record = debug_records[-1]
                    assert isinstance(current_debug_record, dict)
                    deterministic_pending = _build_phase4_deterministic_shadow_pending(
                        unvisited_feature_rank,
                        feature_influences.detach().cpu(),
                        queue_size=queue_size,
                        feat_layers=feat_layers,
                        feat_positions=feat_pos,
                        feat_ids=feat_ids,
                        exact_chunked_decoder=exact_chunked_decoder,
                        decoder_chunk_size=decoder_chunk_size,
                    )
                    current_debug_record["deterministic_shadow"] = _compare_phase4_frontiers(
                        pending,
                        deterministic_pending,
                    )
                    if phase4_refresh_count == 0:
                        if use_compact_feature_row_store:
                            assert feature_row_store is not None
                            float64_feature_influences = (
                                compute_partial_feature_influences_streaming(
                                    lambda row_start, row_end: feature_row_store.read_feature_rows(
                                        row_start,
                                        row_end,
                                        phase="phase4_anomaly_debug",
                                    ),
                                    feature_row_store.row_abs_sums[:st].to(dtype=torch.float64),
                                    targets.logit_probabilities.to(dtype=torch.float64),
                                    row_to_node_index[:st],
                                    n_feature_nodes=total_active_feats,
                                    n_logits=n_logits,
                                    device=torch.device("cpu"),
                                )
                            )
                        else:
                            float64_influences = compute_partial_influences(
                                edge_matrix[:st].to(dtype=torch.float64),
                                targets.logit_probabilities.to(dtype=torch.float64),
                                row_to_node_index[:st],
                                device=torch.device("cpu"),
                            )
                            float64_feature_influences = float64_influences[:total_active_feats]
                        float32_signal_stats = _build_vector_stats(
                            feature_influences.detach().cpu(),
                            epsilon=1e-12,
                            top_k=8,
                        )
                        float64_signal_stats = _build_vector_stats(
                            float64_feature_influences.detach().cpu(),
                            epsilon=1e-12,
                            top_k=8,
                        )
                        float64_feature_rank = torch.argsort(
                            float64_feature_influences,
                            descending=True,
                        ).cpu()
                        float64_pending = float64_feature_rank[~visited[float64_feature_rank]][
                            :queue_size
                        ]
                        float64_pending = _reorder_pending_for_phase4_locality(
                            float64_pending,
                            feat_layers=feat_layers,
                            feat_positions=feat_pos,
                            feat_ids=feat_ids,
                            exact_chunked_decoder=exact_chunked_decoder,
                            decoder_chunk_size=decoder_chunk_size,
                        )
                        current_debug_record["float64_shadow"] = _compare_phase4_frontiers(
                            pending,
                            float64_pending,
                        )
                        current_debug_record["float_precision_signal_compare"] = {
                            "float32": float32_signal_stats,
                            "float64": float64_signal_stats,
                            "float32_all_zero": bool(float32_signal_stats["all_zero"]),
                            "float64_all_zero": bool(float64_signal_stats["all_zero"]),
                            "float32_effectively_all_zero": bool(
                                float32_signal_stats["effectively_all_zero"]
                            ),
                            "float64_effectively_all_zero": bool(
                                float64_signal_stats["effectively_all_zero"]
                            ),
                        }
                    current_pending_cpu = pending.detach().to(device="cpu", dtype=torch.int64)
                    if first_phase4_pending is None:
                        first_phase4_pending = current_pending_cpu.clone()
                    previous_phase4_pending = current_pending_cpu
                phase4_refresh_count += 1

            pending_offset = 0
            while pending_offset < len(pending):
                idx_batch = pending[pending_offset : pending_offset + phase4_feature_batch_size]
                pending_offset += len(idx_batch)
                n_visited += len(idx_batch)
                phase4_batch_count += 1

                ctx_before = _snapshot_diagnostics(ctx) if profile else None
                transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
                batch_start = time.perf_counter()
                rows = ctx.compute_batch(
                    layers=feat_layers[idx_batch],
                    positions=feat_pos[idx_batch],
                    inject_values=ctx.materialize_encoder_vectors(idx_batch),
                    retain_graph=n_visited < actual_max_feature_nodes,
                    phase_label="phase4_features",
                )

                row_count = rows.shape[0]
                end = st + row_count
                rows_cpu = rows.cpu()
                if use_compact_feature_row_store:
                    assert feature_row_store is not None
                    feature_row_store.append_rows(
                        row_start=st,
                        feature_rows=rows_cpu[:row_count, :total_active_feats],
                        full_row_abs_sums=rows_cpu[:row_count, :logit_offset].abs().sum(dim=1),
                        phase="phase4",
                    )
                else:
                    edge_matrix[st:end, :logit_offset] = rows_cpu
                row_to_node_index[st:end] = idx_batch
                visited[idx_batch] = True
                st = end
                pbar.update(len(idx_batch))

                if profile:
                    batch_number = phase4_batch_count
                    if batch_number % profile_log_interval == 0:
                        batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
                        _log_batch_profile(
                            logger,
                            "Phase 4",
                            batch_number,
                            max(
                                (actual_max_feature_nodes + phase4_feature_batch_size - 1)
                                // phase4_feature_batch_size,
                                1,
                            ),
                            batch_elapsed_ms / 1000.0,
                            ctx_before,
                            _snapshot_diagnostics(ctx),
                            transcoder_before,
                            _snapshot_diagnostics(model.transcoders),
                        )
                batch_number = phase4_batch_count
                batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
                telemetry_recorder.record_event(
                    scope="batch",
                    name="phase4.feature_batch",
                    phase="phase4",
                    batch_index=batch_number,
                    elapsed_ms=batch_elapsed_ms,
                    attrs={
                        "batch_rows": int(row_count),
                        "visited_features": int(n_visited),
                        "target_feature_count": int(actual_max_feature_nodes),
                    },
                )
                telemetry_recorder.record_wall_clock_duration(
                    scope="batch",
                    name="phase4.feature_batch",
                    elapsed_ms=batch_elapsed_ms,
                )

        pbar.close()
        _log_phase_metrics(
            logger,
            "Feature attributions",
            phase4_start,
            model.device,
            selected_features=int(visited.sum().item()),
            final_feature_batch_size=phase4_feature_batch_size,
            phase4_batches=phase4_batch_count,
        )
        phase4_elapsed_ms = (time.perf_counter() - phase4_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase4.feature_attribution",
            phase="phase4",
            elapsed_ms=phase4_elapsed_ms,
            attrs={
                "selected_features": int(visited.sum().item()),
                "feature_batch_size": int(phase4_feature_batch_size),
                "phase4_batches": int(phase4_batch_count),
                "phase4_refreshes": int(phase4_refresh_count),
                "phase4_refresh_elapsed_ms_total": float(phase4_refresh_elapsed_ms_total),
            },
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase4.feature_attribution",
            phase="phase4",
            elapsed_ms=phase4_elapsed_ms,
        )
        if anomaly_debug_result is not None:
            records = anomaly_debug_result.get("records", [])
            cutoff_margins = [
                float(record["cutoff"]["cutoff_margin"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("cutoff"), dict)
                and record["cutoff"].get("cutoff_margin") is not None
            ]
            previous_overlaps = [
                float(record["overlap_with_previous"])
                for record in records
                if isinstance(record, dict) and record.get("overlap_with_previous") is not None
            ]
            first_overlaps = [
                float(record["overlap_with_first"])
                for record in records
                if isinstance(record, dict) and record.get("overlap_with_first") is not None
            ]
            deterministic_overlaps = [
                float(record["deterministic_shadow"]["overlap_fraction"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("deterministic_shadow"), dict)
                and record["deterministic_shadow"].get("overlap_fraction") is not None
            ]
            float64_overlaps = [
                float(record["float64_shadow"]["overlap_fraction"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("float64_shadow"), dict)
                and record["float64_shadow"].get("overlap_fraction") is not None
            ]
            refresh_elapsed_values = [
                float(record["refresh_elapsed_ms"])
                for record in records
                if isinstance(record, dict) and record.get("refresh_elapsed_ms") is not None
            ]
            rank_nonzero_counts = [
                int(record["rank_signal_stats"]["nonzero_count"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("rank_signal_stats"), dict)
                and record["rank_signal_stats"].get("nonzero_count") is not None
            ]
            rank_effective_nonzero_counts = [
                int(record["rank_signal_stats"]["effective_nonzero_count"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("rank_signal_stats"), dict)
                and record["rank_signal_stats"].get("effective_nonzero_count") is not None
            ]
            rank_abs_sums = [
                float(record["rank_signal_stats"]["abs_sum"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("rank_signal_stats"), dict)
                and record["rank_signal_stats"].get("abs_sum") is not None
            ]
            rank_max_values = [
                float(record["rank_signal_stats"]["max"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("rank_signal_stats"), dict)
                and record["rank_signal_stats"].get("max") is not None
            ]
            rank_all_zero_count = sum(
                1
                for record in records
                if isinstance(record, dict) and bool(record.get("rank_signal_all_zero"))
            )
            rank_effectively_all_zero_count = sum(
                1
                for record in records
                if isinstance(record, dict) and bool(record.get("rank_signal_effectively_all_zero"))
            )
            normalization_clamped_counts = [
                int(record["normalization_input_stats"]["clamped_row_count"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("normalization_input_stats"), dict)
                and record["normalization_input_stats"].get("clamped_row_count") is not None
            ]
            normalization_clamped_fractions = [
                float(record["normalization_input_stats"]["clamped_row_fraction"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("normalization_input_stats"), dict)
                and record["normalization_input_stats"].get("clamped_row_fraction") is not None
            ]
            feature_row_store_read_calls = [
                float(record["feature_row_store_read_stats"]["read_call_count"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("feature_row_store_read_stats"), dict)
                and record["feature_row_store_read_stats"].get("read_call_count") is not None
            ]
            feature_row_store_read_rows = [
                float(record["feature_row_store_read_stats"]["read_row_count"])
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("feature_row_store_read_stats"), dict)
                and record["feature_row_store_read_stats"].get("read_row_count") is not None
            ]
            first_float_precision = None
            if records and isinstance(records[0], dict):
                precision_compare = records[0].get("float_precision_signal_compare")
                if isinstance(precision_compare, dict):
                    first_float_precision = precision_compare
            anomaly_debug_result["refresh_count"] = int(len(records))
            anomaly_debug_result["status"] = "captured_refresh_debug"
            anomaly_debug_result["summary"] = {
                "refresh_count": int(len(records)),
                "pending_size_first": (
                    int(records[0]["pending_size"])
                    if records and isinstance(records[0], dict)
                    else 0
                ),
                "cutoff_margin_min": min(cutoff_margins) if cutoff_margins else None,
                "cutoff_margin_mean": (
                    sum(cutoff_margins) / len(cutoff_margins) if cutoff_margins else None
                ),
                "overlap_with_previous_mean": (
                    sum(previous_overlaps) / len(previous_overlaps) if previous_overlaps else None
                ),
                "overlap_with_first_mean": (
                    sum(first_overlaps) / len(first_overlaps) if first_overlaps else None
                ),
                "deterministic_shadow_overlap_mean": (
                    sum(deterministic_overlaps) / len(deterministic_overlaps)
                    if deterministic_overlaps
                    else None
                ),
                "float64_shadow_overlap_mean": (
                    sum(float64_overlaps) / len(float64_overlaps) if float64_overlaps else None
                ),
                "refresh_elapsed_ms_total": (
                    sum(refresh_elapsed_values) if refresh_elapsed_values else None
                ),
                "refresh_elapsed_ms_mean": (
                    (sum(refresh_elapsed_values) / len(refresh_elapsed_values))
                    if refresh_elapsed_values
                    else None
                ),
                "rank_signal_all_zero_refresh_count": int(rank_all_zero_count),
                "rank_signal_effectively_all_zero_refresh_count": int(
                    rank_effectively_all_zero_count
                ),
                "rank_signal_nonzero_count_min": (
                    min(rank_nonzero_counts) if rank_nonzero_counts else None
                ),
                "rank_signal_nonzero_count_mean": (
                    (sum(rank_nonzero_counts) / len(rank_nonzero_counts))
                    if rank_nonzero_counts
                    else None
                ),
                "rank_signal_effective_nonzero_count_min": (
                    min(rank_effective_nonzero_counts) if rank_effective_nonzero_counts else None
                ),
                "rank_signal_effective_nonzero_count_mean": (
                    (sum(rank_effective_nonzero_counts) / len(rank_effective_nonzero_counts))
                    if rank_effective_nonzero_counts
                    else None
                ),
                "rank_signal_abs_sum_mean": (
                    (sum(rank_abs_sums) / len(rank_abs_sums)) if rank_abs_sums else None
                ),
                "rank_signal_max_max": max(rank_max_values) if rank_max_values else None,
                "normalization_clamped_row_count_max": (
                    max(normalization_clamped_counts) if normalization_clamped_counts else None
                ),
                "normalization_clamped_row_fraction_mean": (
                    (sum(normalization_clamped_fractions) / len(normalization_clamped_fractions))
                    if normalization_clamped_fractions
                    else None
                ),
                "feature_row_store_read_calls_per_refresh_mean": (
                    (sum(feature_row_store_read_calls) / len(feature_row_store_read_calls))
                    if feature_row_store_read_calls
                    else None
                ),
                "feature_row_store_read_rows_per_refresh_mean": (
                    (sum(feature_row_store_read_rows) / len(feature_row_store_read_rows))
                    if feature_row_store_read_rows
                    else None
                ),
                "first_refresh_float32_effectively_all_zero": (
                    bool(first_float_precision.get("float32_effectively_all_zero"))
                    if isinstance(first_float_precision, dict)
                    else None
                ),
                "first_refresh_float64_effectively_all_zero": (
                    bool(first_float_precision.get("float64_effectively_all_zero"))
                    if isinstance(first_float_precision, dict)
                    else None
                ),
            }

        # Phase 5: packaging graph / compact output
        phase5_start = time.perf_counter()
        selected_features = torch.where(visited)[0]
        if compact_output:
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                feature_feature_edges = feature_row_store.materialize_dense_feature_slice(
                    row_start=n_logits,
                    row_end=st,
                    selected_feature_columns=selected_features,
                    phase="phase5",
                )
                logit_feature_edges = feature_row_store.materialize_dense_feature_slice(
                    row_start=0,
                    row_end=n_logits,
                    selected_feature_columns=selected_features,
                    phase="phase5",
                )
            else:
                feature_feature_edges = edge_matrix[n_logits:st, selected_features].detach().cpu()
                logit_feature_edges = edge_matrix[:n_logits, selected_features].detach().cpu()

            compact_output_result = {
                "input_string": model.tokenizer.decode(input_ids),
                "input_tokens": input_ids.detach().cpu(),
                "logit_targets": targets.logit_targets,
                "logit_probabilities": targets.logit_probabilities.detach().cpu(),
                "vocab_size": targets.vocab_size,
                "active_features": activation_matrix.indices().T.detach().cpu(),
                "activation_values": activation_matrix.values().detach().cpu(),
                "selected_features": selected_features.detach().cpu(),
                "feature_row_node_indices": row_to_node_index[n_logits:st].detach().cpu(),
                "logit_row_node_indices": row_to_node_index[:n_logits].detach().cpu(),
                "feature_feature_edges": feature_feature_edges,
                "logit_feature_edges": logit_feature_edges,
                "phase4_feature_batch_size": int(phase4_feature_batch_size),
                "phase4_feature_batch_size_initial": int(
                    batch_size if feature_batch_size is None else feature_batch_size
                ),
                "phase4_feature_batch_size_max": int(max_phase4_feature_batch_size),
                "phase4_feature_batch_planner_enabled": bool(planner_enabled),
                "phase4_feature_batch_planner_status": planner_status,
                "phase4_feature_batch_planner_skip_reason": planner_skip_reason,
                "phase4_anomaly_debug_enabled": bool(phase4_anomaly_debug_enabled),
                "phase4_refresh_count": int(phase4_refresh_count),
                "phase4_batch_count": int(phase4_batch_count),
                "phase4_refresh_elapsed_seconds_total": round(
                    phase4_refresh_elapsed_ms_total / 1000.0,
                    6,
                ),
                "telemetry_max_events": int(telemetry_max_events_resolved),
                "cfg": model.config,
                "scan": model.scan,
            }
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                file_backed_store_bytes = feature_row_store.nbytes
            else:
                del edge_matrix
                file_backed_store_bytes = None
            logger.info(
                "Attribution completed in "
                f"{time.time() - start_time:.2f}s | "
                f"compact_feature_edge_shape={tuple(compact_output_result['feature_feature_edges'].shape)} | "
                f"compact_logit_edge_shape={tuple(compact_output_result['logit_feature_edges'].shape)}"
                + (
                    f" | feature_row_store_bytes={file_backed_store_bytes}"
                    if file_backed_store_bytes is not None
                    else ""
                )
            )
            phase5_elapsed_ms = (time.perf_counter() - phase5_start) * 1000.0
            telemetry_recorder.record_event(
                scope="phase",
                name="phase5.packaging",
                phase="phase5",
                elapsed_ms=phase5_elapsed_ms,
                attrs={
                    "compact_output": True,
                    "selected_features": int(selected_features.numel()),
                    "feature_edge_rows": int(
                        compact_output_result["feature_feature_edges"].shape[0]
                    ),
                    "feature_edge_cols": int(
                        compact_output_result["feature_feature_edges"].shape[1]
                    ),
                },
            )
            telemetry_recorder.record_wall_clock_duration(
                scope="phase",
                name="phase5.packaging",
                phase="phase5",
                elapsed_ms=phase5_elapsed_ms,
            )
            return compact_output_result

        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        if actual_max_feature_nodes < total_active_feats:
            col_read = torch.cat([selected_features, non_feature_nodes])
        else:
            col_read = torch.arange(total_nodes)

        final_node_count = len(col_read)
        full_edge_matrix = torch.zeros(final_node_count, final_node_count, dtype=edge_matrix.dtype)
        feature_row_order = row_to_node_index[n_logits:st].argsort()
        full_edge_matrix[:actual_max_feature_nodes] = edge_matrix[n_logits:st][feature_row_order][
            :, col_read
        ]
        full_edge_matrix[-n_logits:] = edge_matrix[:n_logits, :][:, col_read]

        graph = Graph(
            input_string=model.tokenizer.decode(input_ids),
            input_tokens=input_ids,
            logit_targets=targets.logit_targets,
            logit_probabilities=targets.logit_probabilities,
            vocab_size=targets.vocab_size,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            selected_features=selected_features,
            adjacency_matrix=full_edge_matrix.detach(),
            cfg=model.config,
            scan=model.scan,
        )

        logger.info(
            f"Attribution completed in {time.time() - start_time:.2f}s | "
            f"{format_memory_snapshot(device=model.device, extra={'adjacency_shape': tuple(full_edge_matrix.shape)})}"
        )
        phase5_elapsed_ms = (time.perf_counter() - phase5_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="phase5.packaging",
            phase="phase5",
            elapsed_ms=phase5_elapsed_ms,
            attrs={
                "compact_output": False,
                "adjacency_rows": int(full_edge_matrix.shape[0]),
                "adjacency_cols": int(full_edge_matrix.shape[1]),
            },
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="phase5.packaging",
            phase="phase5",
            elapsed_ms=phase5_elapsed_ms,
        )

        return graph
    finally:
        teardown_start = time.perf_counter()
        if feature_row_store is not None:
            feature_row_store.cleanup()
        if ctx is not None:
            _log_memory_boundary(logger, "Teardown start", model.device)
            cleanup = getattr(ctx, "cleanup", None)
            if callable(cleanup):
                cleanup()
            else:
                clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
                if callable(clear_decoder_cache):
                    clear_decoder_cache()
            _log_memory_boundary(logger, "Teardown done", model.device)
        teardown_elapsed_ms = (time.perf_counter() - teardown_start) * 1000.0
        telemetry_recorder.record_event(
            scope="phase",
            name="teardown.cleanup",
            phase="teardown",
            elapsed_ms=teardown_elapsed_ms,
            attrs={
                "ctx_present": ctx is not None,
                "feature_row_store": feature_row_store is not None,
            },
        )
        telemetry_recorder.record_wall_clock_duration(
            scope="phase",
            name="teardown.cleanup",
            phase="teardown",
            elapsed_ms=teardown_elapsed_ms,
        )

        exc_type, exc, _ = sys.exc_info()
        if exc_type is None:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            telemetry_recorder.record_event(
                scope="run",
                name="attribute.done",
                elapsed_ms=run_elapsed_ms,
                attrs={"compact_output": compact_output},
            )
            telemetry_recorder.record_wall_clock_duration(
                scope="run",
                name="attribute.done",
                elapsed_ms=run_elapsed_ms,
            )
        else:
            run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
            telemetry_recorder.record_event(
                scope="run",
                name="attribute.failed",
                elapsed_ms=run_elapsed_ms,
                attrs={
                    "compact_output": compact_output,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc) if exc is not None else None,
                },
            )
            telemetry_recorder.record_wall_clock_duration(
                scope="run",
                name="attribute.failed",
                elapsed_ms=run_elapsed_ms,
            )

        if compact_output_result is not None:
            telemetry_export = telemetry_recorder.export(include_events=True)
            compact_output_result["telemetry_summary"] = telemetry_export["summary"]
            compact_output_result["telemetry_events"] = telemetry_export.get("events", [])
            if anomaly_debug_result is not None:
                compact_output_result["phase4_anomaly_debug"] = anomaly_debug_result
        elif profile:
            telemetry_summary = telemetry_recorder.build_summary()
            logger.info(
                "Telemetry summary | "
                f"event_count={telemetry_summary.get('event_count')} | "
                f"stored_event_count={telemetry_summary.get('stored_event_count')} | "
                f"dropped_event_count={telemetry_summary.get('dropped_event_count')}"
            )
