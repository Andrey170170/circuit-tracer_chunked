"""Phase 4 frontier selection from completed Phase 3 rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase4_policy import (
    _compute_phase4_refresh_queue_window_size,
    _reorder_pending_for_phase4_locality,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_feature_semantic_descriptors_payload,
    _build_phase3_frontier_buffer_metadata,
    _build_phase4_cutoff_debug,
    _build_phase4_deterministic_shadow_pending,
    _build_phase4_normalization_stats,
    _build_vector_stats,
    _compare_phase4_frontiers,
    _dtype_to_name,
)
from circuit_tracer.attribution.nnsight.replay import (
    _build_phase3_seed_bundle_payload,
    _build_phase3_seed_influence_topk,
    _hash_float_tensor,
    _hash_index_tensor,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _hash_json_payload,
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_feature_influences_tiled,
    compute_partial_influences,
)
from circuit_tracer.observability.events import RuntimeSnapshot, TraceObserver


@dataclass(frozen=True)
class Phase3FrontierResult:
    """Phase 4 capacity and optional immutable evidence selected from Phase 3."""

    actual_max_feature_nodes: int
    buffer_metadata: dict[str, object]
    seed_bundle: dict[str, object] | None
    semantic_descriptors: dict[str, object] | None


@dataclass(frozen=True)
class RankedFrontier:
    influences: torch.Tensor
    ranked_features: torch.Tensor
    pre_locality: torch.Tensor
    post_locality: torch.Tensor
    queue_size: int
    actual_max_feature_nodes: int
    buffer_metadata: dict[str, object]
    normalization_stats: dict[str, object] | None
    row_store_snapshot: dict[str, float | int | None] | None


def select_phase3_frontier(*, inputs: Any, config: Any) -> Phase3FrontierResult:
    """Rank seed influences, apply locality, and package requested evidence."""
    enabled = (
        inputs.cross_cluster_debug_summary is not None
        or config.capture_phase3_seed_bundle_enabled
        or config.capture_feature_semantic_descriptors_enabled
        or inputs.phase3_frontier_buffer_metadata["enabled"]
    )
    if not enabled:
        return Phase3FrontierResult(
            config.actual_max_feature_nodes,
            inputs.phase3_frontier_buffer_metadata,
            inputs.phase3_seed_bundle_payload,
            inputs.feature_semantic_descriptors_payload,
        )
    summary, stream = _runtime_snapshot(inputs)
    phase_summary: dict[str, object] = {
        "stored_row_count_before_phase4": int(config.n_logits),
        "actual_max_feature_nodes": int(config.actual_max_feature_nodes),
        "total_active_features": int(config.total_active_feats),
        "update_interval": int(config.update_interval),
        "feature_batch_size": int(config.effective_feature_batch_size),
        "planner_compute_dtype": _dtype_to_name(config.planner_compute_dtype),
        "influence_compute_dtype": _dtype_to_name(config.influence_compute_dtype),
        **summary,
    }
    if config.actual_max_feature_nodes < config.total_active_feats:
        ranked = _rank_frontier(inputs, config)
        phase_summary.update({
            "actual_max_feature_nodes": ranked.actual_max_feature_nodes,
            "status": "captured", "queue_size": ranked.queue_size,
            "phase3_frontier_buffer_metadata": ranked.buffer_metadata,
        })
        seed_bundle, descriptors = _package_selected_evidence(inputs, config, ranked)
        if inputs.cross_cluster_debug_summary is not None:
            _add_frontier_debug(phase_summary, inputs, config, ranked)
    else:
        metadata = inputs.phase3_frontier_buffer_metadata
        metadata["status"] = "skipped_all_features_included"
        metadata["fallback_reason"] = "all_features_included"
        phase_summary.update({
            "status": "skipped_all_features_included",
            "queue_size": int(config.actual_max_feature_nodes),
        })
        seed_bundle, descriptors = _package_skipped_evidence(inputs, config)
        ranked = None
    if inputs.cross_cluster_debug_summary is not None:
        _record_frontier_checkpoint(inputs, config, phase_summary, stream)
    return Phase3FrontierResult(
        ranked.actual_max_feature_nodes if ranked is not None else config.actual_max_feature_nodes,
        ranked.buffer_metadata if ranked is not None else inputs.phase3_frontier_buffer_metadata,
        seed_bundle,
        descriptors,
    )


def _rank_frontier(inputs: Any, config: Any) -> RankedFrontier:
    influences, normalization, store_snapshot = _compute_seed_influences(inputs, config)
    rank = torch.argsort(influences, descending=True).cpu()
    metadata = _build_phase3_frontier_buffer_metadata(
        seed_feature_influences=influences,
        base_max_feature_nodes=int(config.base_max_feature_nodes),
        total_active_features=int(config.total_active_feats),
        relative_epsilon=config.phase3_frontier_buffer_relative_epsilon,
        max_extra=int(config.phase3_frontier_buffer_max_extra),
    )
    actual_max = int(metadata["actual_max_feature_nodes"])
    queue_size = min(
        _compute_phase4_refresh_queue_window_size(
            update_interval=config.update_interval,
            phase4_feature_batch_size=config.effective_feature_batch_size,
            queue_multiplier=config.phase4_refresh_policy_config.effective_queue_multiplier,
        ),
        actual_max,
    )
    pre_locality = rank[:queue_size]
    post_locality = _reorder_pending_for_phase4_locality(
        pre_locality,
        feat_layers=inputs.feat_layers,
        feat_positions=inputs.feat_pos,
        feat_ids=inputs.feat_ids,
        exact_chunked_decoder=config.exact_chunked_decoder,
        decoder_chunk_size=getattr(inputs.model.transcoders, "decoder_chunk_size", None),
    )
    return RankedFrontier(
        influences, rank, pre_locality, post_locality, queue_size, actual_max,
        metadata, normalization, store_snapshot
    )


def _compute_seed_influences(
    inputs: Any, config: Any
) -> tuple[torch.Tensor, dict[str, object] | None, dict[str, float | int | None] | None]:
    pre_phase4_st = int(config.n_logits)
    if config.use_compact_feature_row_store:
        store = inputs.feature_row_store
        assert store is not None
        denominator = (store.row_abs_max[:pre_phase4_st], store.row_l1_scaled[:pre_phase4_st])
        if (
            config.full_retention_backend == "column_tiled_v1"
            or config.feature_row_retention == "none_recompute"
        ):
            influences = compute_partial_feature_influences_tiled(
                store.read_tile,
                denominator,
                inputs.targets.logit_probabilities,
                inputs.row_to_node_index[:pre_phase4_st],
                n_feature_nodes=config.total_active_feats,
                n_logits=config.n_logits,
                row_tile_size=config.influence_row_tile_size,
                column_tile_size=config.influence_column_tile_size,
                device=store.row_abs_max.device,
                compute_dtype=config.planner_compute_dtype,
            )
        else:
            influences = compute_partial_feature_influences_streaming(
                lambda start, end: store.read_feature_rows(start, end, phase="phase3_seed_ranking"),
                denominator,
                inputs.targets.logit_probabilities,
                inputs.row_to_node_index[:pre_phase4_st],
                n_feature_nodes=config.total_active_feats,
                n_logits=config.n_logits,
                device=store.row_abs_max.device,
                compute_dtype=config.planner_compute_dtype,
            )
        if inputs.cross_cluster_debug_summary is None:
            return influences, None, None
        normalization = _build_phase4_normalization_stats(
            (denominator[0].detach().cpu(), denominator[1].detach().cpu())
        )
        return influences, normalization, store.get_diagnostic_snapshot()
    assert inputs.edge_matrix is not None
    planner = compute_partial_influences(
        inputs.edge_matrix[:pre_phase4_st].to(dtype=config.planner_compute_dtype),
        inputs.targets.logit_probabilities.to(dtype=config.planner_compute_dtype),
        inputs.row_to_node_index[:pre_phase4_st],
        device=torch.device("cpu"),
    )
    normalization = None
    if inputs.cross_cluster_debug_summary is not None:
        normalization = _build_phase4_normalization_stats(
            inputs.edge_matrix[:pre_phase4_st, : config.logit_offset].abs().sum(dim=1).detach().cpu()
        )
    return planner[: config.total_active_feats], normalization, None


def _package_selected_evidence(
    inputs: Any, config: Any, ranked: RankedFrontier
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    active_features = inputs.activation_matrix.indices().T
    activation_values = inputs.activation_matrix.values()
    seed_bundle = inputs.phase3_seed_bundle_payload
    descriptors = inputs.feature_semantic_descriptors_payload
    if config.capture_phase3_seed_bundle_enabled:
        seed_bundle = _build_phase3_seed_bundle_payload(
            active_features=active_features, activation_values=activation_values,
            seed_feature_influences=ranked.influences,
            frontier_pre_locality=ranked.pre_locality,
            frontier_post_locality=ranked.post_locality, queue_size=ranked.queue_size,
            actual_max_feature_nodes=ranked.actual_max_feature_nodes,
            total_active_features=config.total_active_feats, status="captured",
            planner_compute_dtype=config.planner_compute_dtype,
            influence_compute_dtype=config.influence_compute_dtype,
        )
    if config.capture_feature_semantic_descriptors_enabled:
        descriptors = _build_feature_semantic_descriptors_payload(
            active_features=active_features, activation_values=activation_values,
            seed_feature_influences=ranked.influences,
            frontier_pre_locality=ranked.pre_locality,
            frontier_post_locality=ranked.post_locality,
            total_active_features=config.total_active_feats, status="captured",
            semantic_descriptor_top_k=config.semantic_descriptor_top_k,
            semantic_descriptor_dim=config.semantic_descriptor_dim,
        )
    return seed_bundle, descriptors


def _package_skipped_evidence(
    inputs: Any, config: Any
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    empty_values = torch.empty(0, dtype=config.planner_compute_dtype)
    empty_rank = torch.empty(0, dtype=torch.long)
    seed_bundle = inputs.phase3_seed_bundle_payload
    descriptors = inputs.feature_semantic_descriptors_payload
    if config.capture_phase3_seed_bundle_enabled:
        seed_bundle = _build_phase3_seed_bundle_payload(
            active_features=inputs.activation_matrix.indices().T,
            activation_values=inputs.activation_matrix.values(),
            seed_feature_influences=empty_values, frontier_pre_locality=empty_rank,
            frontier_post_locality=empty_rank, queue_size=config.actual_max_feature_nodes,
            actual_max_feature_nodes=config.actual_max_feature_nodes,
            total_active_features=config.total_active_feats,
            status="skipped_all_features_included",
            planner_compute_dtype=config.planner_compute_dtype,
            influence_compute_dtype=config.influence_compute_dtype,
        )
    if config.capture_feature_semantic_descriptors_enabled:
        descriptors = _build_feature_semantic_descriptors_payload(
            active_features=inputs.activation_matrix.indices().T,
            activation_values=inputs.activation_matrix.values(),
            seed_feature_influences=empty_values, frontier_pre_locality=empty_rank,
            frontier_post_locality=empty_rank, total_active_features=config.total_active_feats,
            status="skipped_all_features_included",
            semantic_descriptor_top_k=config.semantic_descriptor_top_k,
            semantic_descriptor_dim=config.semantic_descriptor_dim,
        )
    return seed_bundle, descriptors


def _add_frontier_debug(
    summary: dict[str, object], inputs: Any, config: Any, ranked: RankedFrontier
) -> None:
    deterministic = _build_phase4_deterministic_shadow_pending(
        ranked.ranked_features, ranked.influences.detach().cpu(), queue_size=ranked.queue_size,
        feat_layers=inputs.feat_layers, feat_positions=inputs.feat_pos, feat_ids=inputs.feat_ids,
        exact_chunked_decoder=config.exact_chunked_decoder,
        decoder_chunk_size=getattr(inputs.model.transcoders, "decoder_chunk_size", None),
    )
    topk = _build_phase3_seed_influence_topk(
        ranked_feature_indices=ranked.ranked_features,
        seed_feature_influences=ranked.influences,
        feat_layers=inputs.feat_layers, feat_positions=inputs.feat_pos,
        feat_ids=inputs.feat_ids, top_k=8,
    )
    summary.update({
        "feature_influence_stats": _build_vector_stats(ranked.influences.detach().cpu(), epsilon=1e-12, top_k=8),
        "feature_influence_hash": _hash_float_tensor(ranked.influences.detach().cpu(), dtype=torch.float64),
        "frontier_pre_locality_hash": _hash_index_tensor(ranked.pre_locality),
        "frontier_post_locality_hash": _hash_index_tensor(ranked.post_locality),
        "frontier_pre_locality_sample": [int(v) for v in ranked.pre_locality[:16].tolist()],
        "frontier_post_locality_sample": [int(v) for v in ranked.post_locality[:16].tolist()],
        "seed_influence_topk": topk, "seed_influence_topk_hash": _hash_json_payload(topk),
        "seed_cutoff": _build_phase4_cutoff_debug(
            ranked.influences[ranked.ranked_features].detach().cpu(), queue_size=ranked.queue_size
        ),
        "deterministic_shadow": _compare_phase4_frontiers(ranked.post_locality, deterministic),
        "normalization_input_stats": ranked.normalization_stats,
        "feature_row_store_summary": ranked.row_store_snapshot,
    })
    if config.shadow_debug_compute_dtype != config.planner_compute_dtype:
        shadow = _compute_shadow_frontier(inputs, config, ranked.queue_size)
        summary["shadow_debug"] = _compare_phase4_frontiers(ranked.post_locality, shadow)


def _compute_shadow_frontier(inputs: Any, config: Any, queue_size: int) -> torch.Tensor:
    pre_phase4_st = int(config.n_logits)
    if config.use_compact_feature_row_store:
        store = inputs.feature_row_store
        assert store is not None
        influences = compute_partial_feature_influences_streaming(
            lambda start, end: store.read_feature_rows(start, end, phase="phase3_seed_ranking_shadow"),
            (store.row_abs_max[:pre_phase4_st], store.row_l1_scaled[:pre_phase4_st]),
            inputs.targets.logit_probabilities, inputs.row_to_node_index[:pre_phase4_st],
            n_feature_nodes=config.total_active_feats, n_logits=config.n_logits,
            device=torch.device("cpu"), compute_dtype=config.shadow_debug_compute_dtype,
        )
    else:
        assert inputs.edge_matrix is not None
        all_influences = compute_partial_influences(
            inputs.edge_matrix[:pre_phase4_st].to(dtype=config.shadow_debug_compute_dtype),
            inputs.targets.logit_probabilities.to(dtype=config.shadow_debug_compute_dtype),
            inputs.row_to_node_index[:pre_phase4_st], device=torch.device("cpu"),
        )
        influences = all_influences[: config.total_active_feats]
    rank = torch.argsort(influences, descending=True).cpu()
    return _reorder_pending_for_phase4_locality(
        rank[:queue_size], feat_layers=inputs.feat_layers, feat_positions=inputs.feat_pos,
        feat_ids=inputs.feat_ids, exact_chunked_decoder=config.exact_chunked_decoder,
        decoder_chunk_size=getattr(inputs.model.transcoders, "decoder_chunk_size", None),
    )


def _record_frontier_checkpoint(
    inputs: Any, config: Any, summary: dict[str, object], runtime_stream: dict[str, object]
) -> None:
    deterministic = summary.get("deterministic_shadow")
    shadow = summary.get("shadow_debug")
    normalization = summary.get("normalization_input_stats")
    influence_stats = summary.get("feature_influence_stats")
    cutoff = summary.get("seed_cutoff")
    stream = {
        "status": summary.get("status"), "stored_row_count_before_phase4": config.n_logits,
        "actual_max_feature_nodes": summary.get("actual_max_feature_nodes"),
        "total_active_features": config.total_active_feats, "update_interval": config.update_interval,
        "feature_batch_size": config.effective_feature_batch_size, "queue_size": summary.get("queue_size"),
        "feature_influence_hash": summary.get("feature_influence_hash"),
        "frontier_pre_locality_hash": summary.get("frontier_pre_locality_hash"),
        "frontier_post_locality_hash": summary.get("frontier_post_locality_hash"),
        "deterministic_shadow_overlap_fraction": _dict_float(deterministic, "overlap_fraction"),
        "deterministic_shadow_jaccard": _dict_float(deterministic, "jaccard_similarity"),
        "deterministic_shadow_prefix_match_count": _dict_int(deterministic, "prefix_match_count"),
        "shadow_debug_overlap_fraction": _dict_float(shadow, "overlap_fraction"),
        "seed_influence_topk_hash": summary.get("seed_influence_topk_hash"),
        "seed_cutoff_margin": _dict_float(cutoff, "cutoff_margin"),
        "seed_cutoff_near_tie_count": _dict_int(cutoff, "near_cutoff_count"),
        "seed_cutoff_exact_tie_count": _dict_int(cutoff, "exact_cutoff_count"),
        "feature_influence_nonfinite_count": _dict_int(influence_stats, "nonfinite_count"),
        "feature_influence_abs_sum": _dict_float(influence_stats, "abs_sum"),
        "normalization_clamped_row_count": _dict_int(normalization, "clamped_row_count"),
        "normalization_clamped_row_fraction": _dict_float(normalization, "clamped_row_fraction"),
        **runtime_stream,
    }
    _record_cross_cluster_checkpoint(
        cross_cluster_debug_summary=inputs.cross_cluster_debug_summary,
        cross_cluster_debug_checkpoints=inputs.cross_cluster_debug_checkpoints,
        checkpoint_name="phase3_seed_ranking_pre_phase4", phase="phase3",
        summary_payload=summary, stream_payload=stream,
    )


def _runtime_snapshot(inputs: Any) -> tuple[dict[str, object], dict[str, object]]:
    if inputs.cross_cluster_debug_summary is None:
        return {}, {}
    observer: TraceObserver = inputs.telemetry_observer
    return cast(
        tuple[dict[str, object], dict[str, object]],
        observer.observe(RuntimeSnapshot(
            inputs.model.device, context=inputs.ctx, transcoder=inputs.model.transcoders
        )),
    )


def _dict_float(value: object, key: str) -> float | None:
    return _safe_float(value.get(key)) if isinstance(value, dict) else None


def _dict_int(value: object, key: str) -> int | None:
    return int(value.get(key, 0)) if isinstance(value, dict) else None
