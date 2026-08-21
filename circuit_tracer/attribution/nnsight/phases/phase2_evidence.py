"""Phase 2 telemetry and cross-cluster evidence assembly."""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import _build_vector_stats
from circuit_tracer.attribution.nnsight.phases.phase2_storage import (
    ActiveFeaturePlan,
    OpenedRowStorage,
    Phase2ExecutionPolicy,
    RowStoreLayout,
    RowStoreRuntime,
)
from circuit_tracer.attribution.nnsight.phases.phase2_targets import TargetSelection
from circuit_tracer.attribution.nnsight.replay import _hash_float_tensor, _hash_index_tensor
from circuit_tracer.attribution.nnsight.telemetry import (
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.observability.events import (
    PhaseMetrics,
    RuntimeSnapshot,
    TraceEvent,
    TraceObserver,
)


def record_target_replay_evidence(
    *,
    model: Any,
    ctx: Any,
    targets: TargetSelection,
    phase0_replay_metadata: dict[str, object],
    observer: TraceObserver,
    debug_summary: dict[str, object] | None,
    debug_checkpoints: list[dict[str, object]] | None,
) -> None:
    """Record target-logit and Phase-0 replay evidence before storage mutation."""
    if debug_summary is None:
        return
    runtime_summary, runtime_stream = cast(
        tuple[dict[str, object], dict[str, object]],
        observer.observe(RuntimeSnapshot(model.device, context=ctx, transcoder=model.transcoders)),
    )
    token_ids = [int(target.vocab_idx) for target in targets.targets.logit_targets]
    probabilities = targets.targets.logit_probabilities.detach().cpu()
    stats = _build_vector_stats(probabilities, epsilon=1e-12, top_k=8)
    token_hash = (
        _hash_index_tensor(torch.tensor(token_ids, dtype=torch.int64)) if token_ids else None
    )
    state_hash = _hash_float_tensor(probabilities, dtype=torch.float64)
    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=debug_summary,
        cross_cluster_debug_checkpoints=debug_checkpoints,
        checkpoint_name="phase1_target_logits", phase="phase1",
        summary_payload={
            "target_count": len(targets.targets), "target_token_ids": token_ids,
            "target_token_ids_hash": token_hash, "target_probability_stats": stats,
            "target_logit_state_hash": state_hash, **runtime_summary,
        },
        stream_payload={
            "target_count": len(targets.targets), "target_token_ids_hash": token_hash,
            "target_probability_count": int(cast(int, stats["count"])),
            "target_probability_nonfinite_count": int(cast(int, stats["nonfinite_count"])),
            "target_probability_abs_sum": _safe_float(cast(Any, stats.get("abs_sum"))),
            "target_probability_max": _safe_float(cast(Any, stats.get("max"))),
            "target_probability_effectively_all_zero": bool(stats["effectively_all_zero"]),
            "target_logit_state_hash": state_hash, **runtime_stream,
        },
    )
    debug_summary["phase0_replay_metadata"] = phase0_replay_metadata
    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=debug_summary,
        cross_cluster_debug_checkpoints=debug_checkpoints,
        checkpoint_name="phase2_phase0_replay", phase="phase2",
        summary_payload=phase0_replay_metadata,
        stream_payload={
            "phase0_replay_mode": phase0_replay_metadata.get("mode"),
            "phase0_replay_status": phase0_replay_metadata.get("status"),
            "validation_warning_count": phase0_replay_metadata.get("validation_warning_count"),
            "dtype_roundtrip_loss": cast(
                dict[str, object], phase0_replay_metadata.get("dtype_metadata", {})
            ).get("dtype_roundtrip_loss"),
        },
    )


def _storage_metrics(
    *, plan: ActiveFeaturePlan, storage: OpenedRowStorage,
    layout: RowStoreLayout, runtime: RowStoreRuntime,
    execution: Phase2ExecutionPolicy, phase0_metadata: dict[str, object],
) -> dict[str, object]:
    extra: dict[str, object] = {
        "full_retention_backend_requested": layout.backend,
        "full_retention_backend_effective": layout.backend,
        "feature_row_retention_requested": layout.retention,
        "feature_row_retention_effective": layout.retention,
        "replay_tile_cache_bytes_requested": runtime.replay_tile_cache_bytes,
        "replay_tile_cache_bytes_effective": runtime.replay_tile_cache_bytes,
        "row_store_preallocate_requested": runtime.preallocate,
        "row_store_preallocate_effective": runtime.preallocate and layout.retention == "full_file",
        "row_store_mode": (
            "compact_none_recompute"
            if execution.use_compact_feature_row_store and layout.retention == "none_recompute"
            else "compact_feature_file_backed_dense"
            if execution.use_compact_feature_row_store else "dense_full"
        ),
        "phase0_replay_mode": phase0_metadata.get("mode"),
        "phase0_replay_status": phase0_metadata.get("status"),
        "phase0_replay_validation_warning_count": phase0_metadata.get(
            "validation_warning_count"
        ),
    }
    if execution.use_compact_feature_row_store:
        assert storage.feature_row_store is not None and storage.nonfeature_row_store is not None
        extra.update(
            feature_row_store=type(storage.feature_row_store).__name__,
            feature_row_store_path=getattr(storage.feature_row_store, "path", None),
            nonfeature_row_store_path=getattr(storage.nonfeature_row_store, "path", None),
            row_abs_sums_shape=f"{tuple(storage.feature_row_store.row_abs_max.shape)}",
            row_abs_max_shape=f"{tuple(storage.feature_row_store.row_abs_max.shape)}",
            row_l1_scaled_shape=f"{tuple(storage.feature_row_store.row_l1_scaled.shape)}",
            feature_edge_columns=plan.total_active_feats,
            nonfeature_edge_columns=plan.logit_offset - plan.total_active_feats,
            **storage.feature_row_store.get_diagnostic_snapshot(),
        )
    else:
        assert storage.edge_matrix is not None
        extra.update(edge_matrix_shape=f"{tuple(storage.edge_matrix.shape)}",
                     edge_matrix_dtype=storage.edge_matrix.dtype)
    return extra


def record_storage_evidence(
    *, phase2_start: float, model: Any, ctx: Any, plan: ActiveFeaturePlan,
    storage: OpenedRowStorage, layout: RowStoreLayout, runtime: RowStoreRuntime,
    execution: Phase2ExecutionPolicy, phase0_metadata: dict[str, object],
    observer: TraceObserver, debug_summary: dict[str, object] | None,
    debug_checkpoints: list[dict[str, object]] | None,
) -> None:
    """Publish storage metrics and feature-ordering evidence after stores are opened."""
    extra = _storage_metrics(
        plan=plan, storage=storage, layout=layout, runtime=runtime,
        execution=execution, phase0_metadata=phase0_metadata,
    )
    observer.observe(PhaseMetrics("Input vector build", phase2_start, model.device, extra))
    observer.observe(TraceEvent(
        scope="phase", name="phase2.input_vector_build", phase="phase2",
        elapsed_ms=(time.perf_counter() - phase2_start) * 1000.0,
        attrs=extra, wall_clock=True,
    ))
    if debug_summary is None:
        return
    runtime_summary, runtime_stream = cast(
        tuple[dict[str, object], dict[str, object]],
        observer.observe(RuntimeSnapshot(model.device, context=ctx, transcoder=model.transcoders)),
    )
    row_dtype = execution.exact_dtype if execution.use_compact_feature_row_store else layout.feature_dtype
    abs_dtype = execution.exact_dtype if execution.use_compact_feature_row_store else layout.row_abs_sum_dtype
    denominator_components = 2 if execution.use_compact_feature_row_store else 1
    row_count = plan.actual_max_feature_nodes + plan.n_logits
    expected_store_bytes = 0 if layout.retention == "none_recompute" else (
        row_count * plan.total_active_feats * torch.empty((), dtype=row_dtype).element_size()
    )
    expected_abs_bytes = denominator_components * row_count * torch.empty((), dtype=abs_dtype).element_size()
    common = {
        "feat_layers_hash": _hash_index_tensor(plan.feat_layers),
        "feat_pos_hash": _hash_index_tensor(plan.feat_pos),
        "feat_ids_hash": _hash_index_tensor(plan.feat_ids),
        "feature_count": plan.total_active_feats,
        "phase0_replay_mode": phase0_metadata.get("mode"),
        "phase0_replay_status": phase0_metadata.get("status"),
        "phase0_replay_validation_warning_count": phase0_metadata.get("validation_warning_count"),
        "decoder_chunk_size": (int(model.transcoders.decoder_chunk_size)
                               if getattr(model.transcoders, "decoder_chunk_size", None) is not None
                               else None),
        "row_store_mode": extra.get("row_store_mode"),
        "row_denominator_component_count": denominator_components,
        "row_store_expected_bytes": expected_store_bytes,
        "row_abs_sums_expected_bytes": expected_abs_bytes,
        "row_denominator_expected_bytes": expected_abs_bytes,
        "phase4_feature_batch_size_initial": execution.effective_feature_batch_size,
    }
    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=debug_summary,
        cross_cluster_debug_checkpoints=debug_checkpoints,
        checkpoint_name="phase2_feature_ordering", phase="phase2",
        summary_payload={**common, **runtime_summary},
        stream_payload={**common, **runtime_stream},
    )
