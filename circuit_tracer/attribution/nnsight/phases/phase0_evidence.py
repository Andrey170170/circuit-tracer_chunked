"""Diagnostics and cross-cluster evidence for NNSight Phase 0."""

from __future__ import annotations

from collections.abc import Mapping, Sized
import time
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import _build_vector_stats
from circuit_tracer.attribution.nnsight.replay import _hash_sparse_membership_indices
from circuit_tracer.attribution.nnsight.telemetry import (
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.observability.events import (
    DiagnosticSnapshot,
    DiagnosticsMessage,
    PhaseMetrics,
    RuntimeSnapshot,
    TraceEvent,
    TraceObserver,
)


def _value_stats_int_entry(value_stats: dict[str, object], key: str) -> int:
    return int(cast(int, value_stats[key]))


def _value_stats_safe_float_entry(value_stats: dict[str, object], key: str) -> float | None:
    return _safe_float(cast(torch.Tensor | float | int | None, value_stats.get(key)))


def _diagnostic_int_entry(diagnostics: Mapping[str, object], key: str, default: int = 0) -> int:
    return int(cast(int, diagnostics.get(key, default)))


def _diagnostic_mapping_entry(diagnostics: Mapping[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], diagnostics.get(key, {}))


def _diagnostic_sized_entry(diagnostics: Mapping[str, object], key: str) -> Sized:
    return cast(Sized, diagnostics.get(key, {}))


def observe_phase0_completion(
    *,
    ctx: Any,
    model: Any,
    logger: Any,
    observer: TraceObserver,
    phase_start: float,
    profile: bool,
) -> None:
    """Emit Phase 0 timing and diagnostic observations in their established order."""
    observer.observe(
        PhaseMetrics(
            "Precomputation",
            phase_start,
            model.device,
            {
                "active_features": ctx.activation_matrix._nnz(),
                "logit_retention": getattr(ctx, "logit_retention", "full"),
            },
        )
    )
    elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
    observer.observe(
        TraceEvent(
            scope="phase",
            name="phase0.precompute",
            phase="phase0",
            elapsed_ms=elapsed_ms,
            attrs={
                "active_features": int(ctx.activation_matrix._nnz()),
                "logit_retention": getattr(ctx, "logit_retention", "full"),
            },
            wall_clock=True,
        )
    )
    if profile:
        if getattr(ctx, "setup_diagnostic_stats", None):
            observer.observe(
                DiagnosticsMessage("Phase 0 setup diagnostics", ctx.setup_diagnostic_stats)
            )
        transcoder_snapshot = cast(
            dict[str, object] | None,
            observer.observe(DiagnosticSnapshot(model.transcoders)),
        )
        if transcoder_snapshot:
            observer.observe(DiagnosticsMessage("Precompute diagnostics", transcoder_snapshot))
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")


def record_phase0_cross_cluster_evidence(
    *,
    ctx: Any,
    model: Any,
    activation_matrix: torch.Tensor,
    observer: TraceObserver,
    activation_threshold_compare_mode: str,
    stage_encoder_vecs_on_cpu: bool | None,
    stage_error_vectors_on_cpu: bool | None,
    cross_cluster_debug_summary: dict[str, object],
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None,
) -> torch.Tensor:
    """Record the sparse-setup checkpoint without exposing recorder mechanics."""
    runtime_summary, runtime_stream = cast(
        tuple[dict[str, object], dict[str, object]],
        observer.observe(RuntimeSnapshot(model.device, context=ctx, transcoder=model.transcoders)),
    )
    activation_matrix = activation_matrix.coalesce()
    indices = activation_matrix.indices().detach().cpu()
    values = activation_matrix.values().detach().cpu()
    raw_hash = _hash_sparse_membership_indices(
        indices, shape=activation_matrix.shape, canonicalize=False
    )
    canonical_hash = _hash_sparse_membership_indices(
        indices, shape=activation_matrix.shape, canonicalize=True
    )
    n_layers = int(activation_matrix.shape[0])
    layer_counts = (
        torch.bincount(indices[0], minlength=n_layers).tolist()
        if indices.numel() > 0
        else [0] * n_layers
    )
    transcoder_snapshot = runtime_summary.get("transcoder_diagnostic_snapshot")
    threshold_membership = (
        transcoder_snapshot.get("phase0_threshold_membership")
        if isinstance(transcoder_snapshot, dict)
        else None
    )
    if not isinstance(threshold_membership, dict):
        threshold_membership = None
    boundary_fingerprints = (
        transcoder_snapshot.get("phase0_boundary_fingerprints")
        if isinstance(transcoder_snapshot, dict)
        else None
    )
    if not isinstance(boundary_fingerprints, dict):
        boundary_fingerprints = None
    setup_stats = getattr(ctx, "setup_diagnostic_stats", None)
    pre_clt = (
        setup_stats.get("phase0_pre_clt_input_fingerprints")
        if isinstance(setup_stats, dict)
        else None
    )
    if not isinstance(pre_clt, dict):
        pre_clt = None
    global_hashes = (
        boundary_fingerprints.get("global_hashes")
        if isinstance(boundary_fingerprints, dict)
        else None
    )
    if not isinstance(global_hashes, dict):
        global_hashes = None
    value_stats = _build_vector_stats(values, epsilon=1e-12, top_k=8)
    summary = {
        "active_feature_count": int(activation_matrix._nnz()),
        "per_layer_retained_counts": [int(v) for v in layer_counts],
        "active_feature_indices_hash": raw_hash,
        "active_feature_indices_hash_raw_order": raw_hash,
        "active_feature_membership_hash_canonical": canonical_hash,
        "activation_value_stats": value_stats,
        "phase0_activation_threshold_compare_mode": activation_threshold_compare_mode,
        "phase0_threshold_membership": threshold_membership,
        "phase0_boundary_fingerprints": boundary_fingerprints,
        "phase0_pre_clt_input_fingerprints": pre_clt,
        "phase0_pre_clt_input_global_hash": pre_clt.get("global_hash")
        if isinstance(pre_clt, dict)
        else None,
        "logit_retention": getattr(ctx, "logit_retention", None),
        "staging_flags": {
            "stage_encoder_vecs_on_cpu": bool(stage_encoder_vecs_on_cpu),
            "stage_error_vectors_on_cpu": bool(stage_error_vectors_on_cpu),
        },
        "setup_diagnostic_stats": setup_stats,
        **runtime_summary,
    }
    stream = {
        "active_feature_count": int(activation_matrix._nnz()),
        "retained_layer_count": n_layers,
        "retained_nonzero_layer_count": sum(1 for value in layer_counts if int(value) > 0),
        "active_feature_indices_hash": summary["active_feature_indices_hash"],
        "active_feature_membership_hash_canonical": canonical_hash,
        "phase0_activation_threshold_compare_mode": activation_threshold_compare_mode,
        "activation_value_count": _value_stats_int_entry(value_stats, "count"),
        "activation_value_nonfinite_count": _value_stats_int_entry(value_stats, "nonfinite_count"),
        "activation_value_abs_sum": _value_stats_safe_float_entry(value_stats, "abs_sum"),
        "activation_value_max": _value_stats_safe_float_entry(value_stats, "max"),
        "activation_value_effectively_all_zero": bool(value_stats["effectively_all_zero"]),
        "phase0_threshold_membership_layer_count": len(
            _diagnostic_sized_entry(threshold_membership, "per_layer")
        )
        if isinstance(threshold_membership, dict)
        else None,
        "phase0_threshold_membership_borderline_sample_count": _diagnostic_int_entry(
            threshold_membership, "borderline_sample_count"
        )
        if isinstance(threshold_membership, dict)
        else None,
        "phase0_threshold_membership_near_count_abs_lte_1e_04": _diagnostic_int_entry(
            _diagnostic_mapping_entry(threshold_membership, "near_counts_by_epsilon"),
            "abs_lte_1e-04",
        )
        if isinstance(threshold_membership, dict)
        else None,
        "phase0_pre_clt_input_global_hash": summary.get("phase0_pre_clt_input_global_hash"),
        "phase0_pre_clt_input_layer_count": int(pre_clt.get("layer_count", 0))
        if isinstance(pre_clt, dict)
        else None,
        "phase0_boundary_layer_count": len(
            _diagnostic_sized_entry(boundary_fingerprints, "per_layer")
        )
        if isinstance(boundary_fingerprints, dict)
        else None,
        "phase0_boundary_transcoder_constants_global_hash": _diagnostic_mapping_entry(
            boundary_fingerprints, "transcoder_constant_fingerprints"
        ).get("global_hash")
        if isinstance(boundary_fingerprints, dict)
        else None,
        "phase0_boundary_preactivation_hash_global": global_hashes.get("pre_activation_hash_global")
        if isinstance(global_hashes, dict)
        else None,
        "phase0_boundary_margin_hash_global": global_hashes.get("compare_margin_hash_global")
        if isinstance(global_hashes, dict)
        else None,
        "phase0_boundary_mask_membership_hash_global": global_hashes.get(
            "mask_membership_hash_global"
        )
        if isinstance(global_hashes, dict)
        else None,
        "phase0_boundary_post_activation_hash_global": global_hashes.get(
            "post_activation_hash_global"
        )
        if isinstance(global_hashes, dict)
        else None,
        "logit_retention": getattr(ctx, "logit_retention", None),
        "stage_encoder_vecs_on_cpu": bool(stage_encoder_vecs_on_cpu),
        "stage_error_vectors_on_cpu": bool(stage_error_vectors_on_cpu),
        "setup_diagnostic_stats_present": bool(getattr(ctx, "setup_diagnostic_stats", None)),
        **runtime_stream,
    }
    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=cross_cluster_debug_summary,
        cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
        checkpoint_name="phase0_sparse_setup",
        phase="phase0",
        summary_payload=summary,
        stream_payload=stream,
    )
    return activation_matrix
