"""Phase 5 result packaging for the NNSight attribution backend."""

from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Callable
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import (
    _annotate_phase4_selection_on_feature_semantic_descriptors,
)
from circuit_tracer.attribution.nnsight.prefix_view import (
    _compact_nonfeature_column_counts,
    _compact_selected_feature_columns,
    validate_compact_prefix_view_output,
)
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_cross_cluster_runtime_snapshot,
    _record_cross_cluster_checkpoint,
)
from circuit_tracer.graph import Graph
from circuit_tracer.utils.telemetry import format_memory_snapshot


@dataclass(frozen=True)
class Phase5Inputs:
    logger: Any
    model: Any
    ctx: Any
    targets: Any
    telemetry_observer: Any
    activation_matrix: torch.Tensor
    visited: torch.Tensor
    edge_matrix: torch.Tensor | None
    row_to_node_index: torch.Tensor
    input_ids: torch.Tensor
    feature_row_store: Any | None
    nonfeature_row_store: Any | None
    phase0_replay_metadata: dict[str, object]
    phase3_gradient_replay_metadata: dict[str, object]
    phase3_row_replay_metadata: dict[str, object]
    phase3_frontier_buffer_metadata: dict[str, object]
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    phase0_donor_bundle_payload: dict[str, object] | None
    phase3_seed_bundle_payload: dict[str, object] | None
    phase3_gradient_bundle_payload: dict[str, object] | None
    phase3_row_bundle_payload: dict[str, object] | None
    feature_semantic_descriptors_payload: dict[str, object] | None
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    prefix_view_metadata: Any | None
    publish_compact_output_result: Callable[[dict[str, object]], None]
    release_dense_edge_matrix: Callable[[], None]


@dataclass(frozen=True)
class Phase5Config:
    compact_output: bool
    use_compact_feature_row_store: bool
    capture_feature_semantic_descriptors_enabled: bool
    capture_phase0_donor_bundle_enabled: bool
    capture_phase3_seed_bundle_enabled: bool
    capture_phase3_gradient_bundle_enabled: bool
    capture_phase3_row_bundle_enabled: bool
    cross_cluster_debug_enabled: bool
    phase4_anomaly_debug_enabled: bool
    n_pos: int
    n_logits: int
    st: int
    total_active_feats: int
    total_nodes: int
    actual_max_feature_nodes: int
    batch_size: int
    feature_batch_size: int | None
    max_phase4_feature_batch_size: int
    planner_enabled: bool
    planner_status: str
    planner_skip_reason: str | None
    phase4_scheduler_config: Any
    phase4_refresh_optimization_config: Any
    phase4_row_executor_config: Any
    phase4_row_reduction_config: Any
    phase1_trace_batch_metadata: dict[str, object]
    internal_precision_requested: str
    resolved_dtype_map: dict[str, object]
    phase0_activation_threshold_compare_mode_resolved: str
    exact_trace_internal_dtype_name: str
    telemetry_max_events_resolved: int
    semantic_descriptor_top_k: int
    semantic_descriptor_dim: int
    phase4_feature_batch_size: int
    phase4_executor_reference_batch_size: int
    phase4_executor_microbatch_size: int
    phase4_refresh_count: int
    phase4_scheduler_reference_batch_count: int
    phase4_executor_microbatch_count: int
    phase4_elapsed_ms: float
    phase4_refresh_elapsed_ms_total: float
    phase4_feature_batch_elapsed_ms_total: float
    phase4_refresh_partial_influence_elapsed_ms_total: float
    phase4_refresh_rank_topk_elapsed_ms_total: float
    phase4_refresh_frontier_plan_elapsed_ms_total: float
    phase4_refresh_row_store_read_elapsed_ms_total: float
    phase4_refresh_prepared_chunk_cache_bytes: int
    phase4_refresh_prepared_chunk_cache_bytes_effective: int
    phase4_refresh_active_row_accumulation: str
    phase4_refresh_active_row_accumulation_effective: str
    phase4_refresh_aux_fallback_reason: str | None
    phase4_refresh_aux_applicable: bool
    start_time: float
    phase0_context_override: Any | None
    target_logit_source: str | None
    target_logits_override: torch.Tensor | None


@dataclass(frozen=True)
class Phase5Result:
    output: dict[str, object] | Graph
    compact_output_result: dict[str, object] | None
    edge_matrix: torch.Tensor | None


def run_phase5(*, inputs: Phase5Inputs, config: Phase5Config) -> Phase5Result:
    logger = inputs.logger
    model = inputs.model
    ctx = inputs.ctx
    targets = inputs.targets
    telemetry_observer = inputs.telemetry_observer
    activation_matrix = inputs.activation_matrix
    visited = inputs.visited
    edge_matrix = inputs.edge_matrix
    row_to_node_index = inputs.row_to_node_index
    input_ids = inputs.input_ids
    feature_row_store = inputs.feature_row_store
    nonfeature_row_store = inputs.nonfeature_row_store
    phase0_replay_metadata = inputs.phase0_replay_metadata
    phase3_gradient_replay_metadata = inputs.phase3_gradient_replay_metadata
    phase3_row_replay_metadata = inputs.phase3_row_replay_metadata
    phase3_frontier_buffer_metadata = inputs.phase3_frontier_buffer_metadata
    phase4_frontier_buffer_metadata = inputs.phase4_frontier_buffer_metadata
    phase4_execution_metadata = inputs.phase4_execution_metadata
    phase0_donor_bundle_payload = inputs.phase0_donor_bundle_payload
    phase3_seed_bundle_payload = inputs.phase3_seed_bundle_payload
    phase3_gradient_bundle_payload = inputs.phase3_gradient_bundle_payload
    phase3_row_bundle_payload = inputs.phase3_row_bundle_payload
    feature_semantic_descriptors_payload = inputs.feature_semantic_descriptors_payload
    cross_cluster_debug_summary = inputs.cross_cluster_debug_summary
    cross_cluster_debug_checkpoints = inputs.cross_cluster_debug_checkpoints
    cross_cluster_debug_batches = inputs.cross_cluster_debug_batches
    prefix_view_metadata = inputs.prefix_view_metadata
    publish_compact_output_result = inputs.publish_compact_output_result
    release_dense_edge_matrix = inputs.release_dense_edge_matrix
    compact_output = config.compact_output
    use_compact_feature_row_store = config.use_compact_feature_row_store
    capture_feature_semantic_descriptors_enabled = (
        config.capture_feature_semantic_descriptors_enabled
    )
    capture_phase0_donor_bundle_enabled = config.capture_phase0_donor_bundle_enabled
    capture_phase3_seed_bundle_enabled = config.capture_phase3_seed_bundle_enabled
    capture_phase3_gradient_bundle_enabled = config.capture_phase3_gradient_bundle_enabled
    capture_phase3_row_bundle_enabled = config.capture_phase3_row_bundle_enabled
    cross_cluster_debug_enabled = config.cross_cluster_debug_enabled
    phase4_anomaly_debug_enabled = config.phase4_anomaly_debug_enabled
    n_pos = config.n_pos
    n_logits = config.n_logits
    st = config.st
    total_active_feats = config.total_active_feats
    total_nodes = config.total_nodes
    actual_max_feature_nodes = config.actual_max_feature_nodes
    batch_size = config.batch_size
    feature_batch_size = config.feature_batch_size
    max_phase4_feature_batch_size = config.max_phase4_feature_batch_size
    planner_enabled = config.planner_enabled
    planner_status = config.planner_status
    planner_skip_reason = config.planner_skip_reason
    phase4_scheduler_config = config.phase4_scheduler_config
    phase4_refresh_optimization_config = config.phase4_refresh_optimization_config
    phase4_row_executor_config = config.phase4_row_executor_config
    phase4_row_reduction_config = config.phase4_row_reduction_config
    phase1_trace_batch_metadata = config.phase1_trace_batch_metadata
    internal_precision_requested = config.internal_precision_requested
    resolved_dtype_map = config.resolved_dtype_map
    phase0_activation_threshold_compare_mode_resolved = (
        config.phase0_activation_threshold_compare_mode_resolved
    )
    exact_trace_internal_dtype_name = config.exact_trace_internal_dtype_name
    telemetry_max_events_resolved = config.telemetry_max_events_resolved
    semantic_descriptor_top_k = config.semantic_descriptor_top_k
    semantic_descriptor_dim = config.semantic_descriptor_dim
    phase4_feature_batch_size = config.phase4_feature_batch_size
    phase4_executor_reference_batch_size = config.phase4_executor_reference_batch_size
    phase4_executor_microbatch_size = config.phase4_executor_microbatch_size
    phase4_refresh_count = config.phase4_refresh_count
    phase4_scheduler_reference_batch_count = config.phase4_scheduler_reference_batch_count
    phase4_executor_microbatch_count = config.phase4_executor_microbatch_count
    phase4_elapsed_ms = config.phase4_elapsed_ms
    phase4_refresh_elapsed_ms_total = config.phase4_refresh_elapsed_ms_total
    phase4_feature_batch_elapsed_ms_total = config.phase4_feature_batch_elapsed_ms_total
    phase4_refresh_partial_influence_elapsed_ms_total = (
        config.phase4_refresh_partial_influence_elapsed_ms_total
    )
    phase4_refresh_rank_topk_elapsed_ms_total = config.phase4_refresh_rank_topk_elapsed_ms_total
    phase4_refresh_frontier_plan_elapsed_ms_total = (
        config.phase4_refresh_frontier_plan_elapsed_ms_total
    )
    phase4_refresh_row_store_read_elapsed_ms_total = (
        config.phase4_refresh_row_store_read_elapsed_ms_total
    )
    phase4_refresh_prepared_chunk_cache_bytes = config.phase4_refresh_prepared_chunk_cache_bytes
    phase4_refresh_prepared_chunk_cache_bytes_effective = (
        config.phase4_refresh_prepared_chunk_cache_bytes_effective
    )
    phase4_refresh_active_row_accumulation = config.phase4_refresh_active_row_accumulation
    phase4_refresh_active_row_accumulation_effective = (
        config.phase4_refresh_active_row_accumulation_effective
    )
    phase4_refresh_aux_fallback_reason = config.phase4_refresh_aux_fallback_reason
    phase4_refresh_aux_applicable = config.phase4_refresh_aux_applicable
    start_time = config.start_time
    phase0_context_override = config.phase0_context_override
    target_logit_source = config.target_logit_source
    target_logits_override = config.target_logits_override
    phase5_start = time.perf_counter()
    selected_features = torch.where(visited)[0]
    if use_compact_feature_row_store:
        assert feature_row_store is not None
        selected_features = _compact_selected_feature_columns(
            selected_features,
            n_feature_columns=feature_row_store.n_feature_columns,
        )
    if capture_feature_semantic_descriptors_enabled and isinstance(
        feature_semantic_descriptors_payload, dict
    ):
        _annotate_phase4_selection_on_feature_semantic_descriptors(
            feature_semantic_descriptors_payload,
            selected_features=selected_features,
        )
    selected_features_cpu = (
        selected_features.detach().to(device="cpu", dtype=torch.long) if compact_output else None
    )
    if compact_output:
        active_features_cpu = activation_matrix.indices().T.detach().cpu()
        n_active_features = int(active_features_cpu.shape[0])
        compact_token_count = int(n_pos)
        n_error_nodes, _n_token_nodes, n_nonfeature_nodes = _compact_nonfeature_column_counts(
            n_layers=int(model.cfg.n_layers),
            compact_token_count=compact_token_count,
        )
        error_col_start = n_active_features
        token_col_start = error_col_start + n_error_nodes
        logit_col_start = token_col_start + compact_token_count
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            assert nonfeature_row_store is not None
            assert selected_features_cpu is not None
            feature_feature_edges = feature_row_store.materialize_dense_feature_slice(
                row_start=n_logits,
                row_end=st,
                selected_feature_columns=selected_features_cpu,
                phase="phase5",
            )
            logit_feature_edges = feature_row_store.materialize_dense_feature_slice(
                row_start=0,
                row_end=n_logits,
                selected_feature_columns=selected_features_cpu,
                phase="phase5",
            )
            if int(nonfeature_row_store.n_feature_columns) != int(n_nonfeature_nodes):
                raise ValueError(
                    "compact nonfeature row-store width does not match "
                    "prefix-visible error/token column count"
                )
            nonfeature_columns = torch.arange(n_nonfeature_nodes, dtype=torch.long)
            feature_nonfeature_edges = nonfeature_row_store.materialize_dense_feature_slice(
                row_start=n_logits,
                row_end=st,
                selected_feature_columns=nonfeature_columns,
                phase="phase5",
            )
            logit_nonfeature_edges = nonfeature_row_store.materialize_dense_feature_slice(
                row_start=0,
                row_end=n_logits,
                selected_feature_columns=nonfeature_columns,
                phase="phase5",
            )
            n_error_columns = int(token_col_start - error_col_start)
            feature_error_edges = feature_nonfeature_edges[:, :n_error_columns]
            feature_token_edges = feature_nonfeature_edges[:, n_error_columns:]
            logit_error_edges = logit_nonfeature_edges[:, :n_error_columns]
            logit_token_edges = logit_nonfeature_edges[:, n_error_columns:]
        else:
            feature_feature_edges = edge_matrix[n_logits:st, selected_features].detach().cpu()
            logit_feature_edges = edge_matrix[:n_logits, selected_features].detach().cpu()
            feature_error_edges = (
                edge_matrix[n_logits:st, error_col_start:token_col_start].detach().cpu()
            )
            feature_token_edges = (
                edge_matrix[n_logits:st, token_col_start:logit_col_start].detach().cpu()
            )
            logit_error_edges = (
                edge_matrix[:n_logits, error_col_start:token_col_start].detach().cpu()
            )
            logit_token_edges = (
                edge_matrix[:n_logits, token_col_start:logit_col_start].detach().cpu()
            )

        assert selected_features_cpu is not None
        compact_output_result = {
            "input_string": model.tokenizer.decode(input_ids),
            "input_tokens": input_ids[:n_pos].detach().cpu(),
            "full_input_tokens": input_ids.detach().cpu(),
            "logit_targets": targets.logit_targets,
            "logit_probabilities": targets.logit_probabilities.detach().cpu(),
            "vocab_size": targets.vocab_size,
            "active_features": active_features_cpu,
            "activation_values": activation_matrix.values().detach().cpu(),
            "selected_features": selected_features_cpu,
            "feature_row_node_indices": row_to_node_index[n_logits:st].detach().cpu(),
            "logit_row_node_indices": row_to_node_index[:n_logits].detach().cpu(),
            "feature_feature_edges": feature_feature_edges,
            "logit_feature_edges": logit_feature_edges,
            "feature_error_edges": feature_error_edges,
            "feature_token_edges": feature_token_edges,
            "logit_error_edges": logit_error_edges,
            "logit_token_edges": logit_token_edges,
            "n_error_nodes": n_error_nodes,
            "n_token_nodes": int(n_pos),
            "phase4_feature_batch_size": int(phase4_feature_batch_size),
            "phase4_feature_batch_size_initial": int(
                batch_size if feature_batch_size is None else feature_batch_size
            ),
            "phase4_feature_batch_size_max": int(max_phase4_feature_batch_size),
            "phase4_feature_batch_planner_enabled": bool(planner_enabled),
            "phase4_feature_batch_planner_status": planner_status,
            "phase4_feature_batch_planner_skip_reason": planner_skip_reason,
            "phase4_scheduler_requested_mode": phase4_scheduler_config.requested_mode,
            "phase4_scheduler_mode": phase4_scheduler_config.requested_mode,
            "phase4_scheduler_mode_requested": phase4_scheduler_config.requested_mode,
            "phase4_scheduler_version": phase4_scheduler_config.version,
            "phase4_scheduler_version_requested": phase4_scheduler_config.version,
            "phase4_scheduler_policy": phase4_scheduler_config.policy,
            "phase4_scheduler_policy_requested": phase4_scheduler_config.policy,
            "phase4_scheduler_effective_mode": phase4_scheduler_config.effective_mode,
            "phase4_scheduler_mode_effective": phase4_scheduler_config.effective_mode,
            "phase4_scheduler_effective_version": phase4_scheduler_config.effective_version,
            "phase4_scheduler_version_effective": phase4_scheduler_config.effective_version,
            "phase4_scheduler_effective_policy": phase4_scheduler_config.effective_policy,
            "phase4_scheduler_policy_effective": phase4_scheduler_config.effective_policy,
            "phase4_scheduler_effective_behavior": phase4_scheduler_config.effective_behavior,
            "phase4_scheduler_reference_execution": bool(
                phase4_scheduler_config.requested_mode != phase4_scheduler_config.effective_mode
            ),
            "phase4_scheduler_debug": bool(phase4_scheduler_config.debug),
            "phase4_scheduler_telemetry_detail": phase4_scheduler_config.telemetry_detail,
            "phase4_refresh_optimization_requested": phase4_refresh_optimization_config.requested_mode,
            "phase4_refresh_optimization": phase4_refresh_optimization_config.requested_mode,
            "phase4_refresh_optimization_mode_requested": phase4_refresh_optimization_config.requested_mode,
            "phase4_refresh_optimization_effective": phase4_refresh_optimization_config.effective_mode,
            "phase4_refresh_optimization_mode_effective": phase4_refresh_optimization_config.effective_mode,
            "phase4_refresh_optimization_version": phase4_refresh_optimization_config.version,
            "phase4_refresh_optimization_version_requested": phase4_refresh_optimization_config.version,
            "phase4_refresh_optimization_effective_version": phase4_refresh_optimization_config.effective_version,
            "phase4_refresh_optimization_version_effective": phase4_refresh_optimization_config.effective_version,
            "phase4_refresh_optimization_effective_behavior": phase4_refresh_optimization_config.effective_behavior,
            "phase4_refresh_optimization_reference_execution": bool(
                phase4_refresh_optimization_config.requested_mode
                != phase4_refresh_optimization_config.effective_mode
            ),
            "phase4_refresh_prepared_chunk_cache_bytes_requested": int(
                phase4_refresh_prepared_chunk_cache_bytes
            ),
            "phase4_refresh_prepared_chunk_cache_bytes_effective": int(
                phase4_refresh_prepared_chunk_cache_bytes_effective
            ),
            "phase4_refresh_prepared_chunk_cache_enabled": bool(
                phase4_refresh_prepared_chunk_cache_bytes_effective > 0
            ),
            "phase4_refresh_active_row_accumulation_requested": phase4_refresh_active_row_accumulation,
            "phase4_refresh_active_row_accumulation_effective": phase4_refresh_active_row_accumulation_effective,
            "phase4_refresh_active_row_accumulation_fallback_reason": phase4_refresh_aux_fallback_reason,
            "phase4_refresh_active_row_accumulation_applicable": bool(
                phase4_refresh_aux_applicable
            ),
            "phase4_row_executor_requested": phase4_row_executor_config.requested_mode,
            "phase4_row_executor": phase4_row_executor_config.requested_mode,
            "phase4_row_executor_mode_requested": phase4_row_executor_config.requested_mode,
            "phase4_row_executor_effective": phase4_row_executor_config.effective_mode,
            "phase4_row_executor_mode_effective": phase4_row_executor_config.effective_mode,
            "phase4_row_executor_version": phase4_row_executor_config.version,
            "phase4_row_executor_version_requested": phase4_row_executor_config.version,
            "phase4_row_executor_effective_version": phase4_row_executor_config.effective_version,
            "phase4_row_executor_version_effective": phase4_row_executor_config.effective_version,
            "phase4_row_executor_effective_behavior": phase4_row_executor_config.effective_behavior,
            "phase4_row_executor_reference_execution": bool(
                phase4_row_executor_config.requested_mode
                != phase4_row_executor_config.effective_mode
            ),
            "phase4_row_reduction_requested": phase4_row_reduction_config.requested_mode,
            "phase4_row_reduction": phase4_row_reduction_config.requested_mode,
            "phase4_row_reduction_mode_requested": phase4_row_reduction_config.requested_mode,
            "phase4_row_reduction_effective": phase4_row_reduction_config.effective_mode,
            "phase4_row_reduction_mode_effective": phase4_row_reduction_config.effective_mode,
            "phase4_row_reduction_version": phase4_row_reduction_config.version,
            "phase4_row_reduction_version_requested": phase4_row_reduction_config.version,
            "phase4_row_reduction_effective_version": phase4_row_reduction_config.effective_version,
            "phase4_row_reduction_version_effective": phase4_row_reduction_config.effective_version,
            "phase4_row_reduction_effective_behavior": phase4_row_reduction_config.effective_behavior,
            "phase4_row_reduction_reference_execution": bool(
                phase4_row_reduction_config.requested_mode
                != phase4_row_reduction_config.effective_mode
            ),
            **{f"phase1_{key}": value for key, value in phase1_trace_batch_metadata.items()},
            "phase4_executor_configured_reference_batch_size": int(
                phase4_executor_reference_batch_size
            ),
            "phase4_executor_reference_batch_size": int(phase4_executor_reference_batch_size),
            "phase4_executor_microbatch_size": int(phase4_executor_microbatch_size),
            "internal_precision_requested": internal_precision_requested,
            "resolved_dtype_map": resolved_dtype_map,
            "phase4_anomaly_debug_enabled": bool(phase4_anomaly_debug_enabled),
            "cross_cluster_debug_enabled": bool(cross_cluster_debug_enabled),
            "phase0_replay_mode": phase0_replay_metadata.get("mode"),
            "phase0_replay_status": phase0_replay_metadata.get("status"),
            "phase0_replay_context_policy": phase0_replay_metadata.get("context_policy"),
            "phase0_replay_donor_bundle_path": phase0_replay_metadata.get("donor_bundle_path"),
            "phase0_replay_donor_bundle_basename": phase0_replay_metadata.get(
                "donor_bundle_basename"
            ),
            "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                "validation_warning_count"
            ),
            "phase0_replay_validation_warnings": phase0_replay_metadata.get("validation_warnings"),
            "phase0_replay_dtype_roundtrip_loss": cast(
                dict[str, object],
                phase0_replay_metadata.get("dtype_metadata", {}),
            ).get("dtype_roundtrip_loss"),
            "phase3_gradient_replay_mode": phase3_gradient_replay_metadata.get("mode"),
            "phase3_gradient_replay_status": phase3_gradient_replay_metadata.get("status"),
            "phase3_gradient_replay_donor_bundle_path": phase3_gradient_replay_metadata.get(
                "donor_bundle_path"
            ),
            "phase3_gradient_replay_donor_bundle_basename": (
                phase3_gradient_replay_metadata.get("donor_bundle_basename")
            ),
            "phase3_gradient_replay_validation_failure_count": (
                phase3_gradient_replay_metadata.get("validation_failure_count")
            ),
            "phase3_gradient_replay_error": phase3_gradient_replay_metadata.get("error"),
            "phase3_row_replay_mode": phase3_row_replay_metadata.get("mode"),
            "phase3_row_replay_status": phase3_row_replay_metadata.get("status"),
            "phase3_row_replay_donor_bundle_path": phase3_row_replay_metadata.get(
                "donor_bundle_path"
            ),
            "phase3_row_replay_donor_bundle_basename": phase3_row_replay_metadata.get(
                "donor_bundle_basename"
            ),
            "phase3_row_replay_validation_failure_count": phase3_row_replay_metadata.get(
                "validation_failure_count"
            ),
            "phase3_row_replay_error": phase3_row_replay_metadata.get("error"),
            "phase3_row_replay_source": phase3_row_replay_metadata.get("source"),
            "capture_phase0_donor_bundle_enabled": bool(capture_phase0_donor_bundle_enabled),
            "capture_phase3_seed_bundle_enabled": bool(capture_phase3_seed_bundle_enabled),
            "capture_phase3_gradient_bundle_enabled": bool(capture_phase3_gradient_bundle_enabled),
            "capture_phase3_row_bundle_enabled": bool(capture_phase3_row_bundle_enabled),
            "capture_feature_semantic_descriptors_enabled": bool(
                capture_feature_semantic_descriptors_enabled
            ),
            "phase0_donor_bundle_schema_version": (
                int(phase0_donor_bundle_payload.get("schema_version", 1))
                if isinstance(phase0_donor_bundle_payload, dict)
                else 1
            ),
            "phase0_donor_bundle_replay_kind": (
                str(phase0_donor_bundle_payload.get("replay_kind", "phase0_active_features_v1"))
                if isinstance(phase0_donor_bundle_payload, dict)
                else "phase0_active_features_v1"
            ),
            "phase0_donor_bundle_status": (
                str(phase0_donor_bundle_payload.get("status", "captured"))
                if isinstance(phase0_donor_bundle_payload, dict)
                else None
            ),
            "semantic_descriptor_top_k": int(semantic_descriptor_top_k),
            "semantic_descriptor_dim": int(semantic_descriptor_dim),
            "phase4_refresh_count": int(phase4_refresh_count),
            "phase3_frontier_buffer_metadata": phase3_frontier_buffer_metadata,
            "phase4_frontier_buffer_metadata": phase4_frontier_buffer_metadata,
            "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
            "phase4_batches": int(phase4_scheduler_reference_batch_count),
            "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
            "phase4_refresh_elapsed_seconds_total": round(
                phase4_refresh_elapsed_ms_total / 1000.0,
                6,
            ),
            "phase4_feature_batch_elapsed_seconds_total": round(
                phase4_feature_batch_elapsed_ms_total / 1000.0,
                6,
            ),
            "phase4_refresh_partial_influence_elapsed_seconds_total": round(
                phase4_refresh_partial_influence_elapsed_ms_total / 1000.0,
                6,
            ),
            "phase4_refresh_rank_topk_elapsed_seconds_total": round(
                phase4_refresh_rank_topk_elapsed_ms_total / 1000.0,
                6,
            ),
            "phase4_refresh_frontier_plan_elapsed_seconds_total": round(
                phase4_refresh_frontier_plan_elapsed_ms_total / 1000.0,
                6,
            ),
            "phase4_refresh_row_store_read_elapsed_seconds_total": round(
                phase4_refresh_row_store_read_elapsed_ms_total / 1000.0,
                6,
            ),
            "exact_trace_internal_dtype": exact_trace_internal_dtype_name,
            "phase0_activation_threshold_compare_mode": (
                phase0_activation_threshold_compare_mode_resolved
            ),
            "telemetry_max_events": int(telemetry_max_events_resolved),
            "cfg": model.config,
            "scan": model.scan,
        }
        publish_compact_output_result(compact_output_result)
        compact_output_result["phase0_replay_metadata"] = phase0_replay_metadata
        compact_output_result["phase3_gradient_replay_metadata"] = phase3_gradient_replay_metadata
        compact_output_result["phase3_row_replay_metadata"] = phase3_row_replay_metadata
        compact_output_result["phase3_frontier_buffer_metadata"] = phase3_frontier_buffer_metadata
        compact_output_result["phase4_frontier_buffer_metadata"] = phase4_frontier_buffer_metadata
        if capture_phase0_donor_bundle_enabled:
            compact_output_result["phase0_donor_bundle"] = phase0_donor_bundle_payload
        if capture_phase3_seed_bundle_enabled:
            compact_output_result["phase3_seed_bundle"] = phase3_seed_bundle_payload
        if capture_phase3_gradient_bundle_enabled:
            compact_output_result["phase3_gradient_bundle"] = phase3_gradient_bundle_payload
        if capture_phase3_row_bundle_enabled:
            compact_output_result["phase3_row_bundle"] = phase3_row_bundle_payload
        if capture_feature_semantic_descriptors_enabled:
            compact_output_result["feature_semantic_descriptors"] = (
                feature_semantic_descriptors_payload
            )
        if cross_cluster_debug_summary is not None:
            cross_cluster_debug_summary["status"] = "captured"
            phase4_runtime_summary, phase4_runtime_stream = _build_cross_cluster_runtime_snapshot(
                device=model.device,
                ctx=ctx,
                transcoder=model.transcoders,
            )
            phase4_entry_summary_checkpoint = {
                "phase4_refresh_count": int(phase4_refresh_count),
                "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                "phase4_batches": int(phase4_scheduler_reference_batch_count),
                "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                **phase4_execution_metadata,
                **phase4_runtime_summary,
            }
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase4_entry",
                phase="phase4",
                summary_payload=phase4_entry_summary_checkpoint,
                stream_payload={
                    "checkpoint_stage": "post_phase4",
                    "phase4_refresh_count": int(phase4_refresh_count),
                    "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                    "phase4_batches": int(phase4_scheduler_reference_batch_count),
                    "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                    **phase4_execution_metadata,
                    **phase4_runtime_stream,
                },
            )
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase4_run_summary",
                phase="phase4",
                summary_payload=None,
                stream_payload={
                    "selected_feature_count": int(visited.sum().item()),
                    "phase4_feature_batch_size": int(phase4_feature_batch_size),
                    "phase4_refresh_count": int(phase4_refresh_count),
                    "phase4_batch_count": int(phase4_scheduler_reference_batch_count),
                    "phase4_batches": int(phase4_scheduler_reference_batch_count),
                    "phase4_executor_microbatch_count": int(phase4_executor_microbatch_count),
                    "phase4_elapsed_ms": float(phase4_elapsed_ms),
                    "phase4_refresh_elapsed_ms_total": float(phase4_refresh_elapsed_ms_total),
                    "phase4_feature_batch_elapsed_ms_total": float(
                        phase4_feature_batch_elapsed_ms_total
                    ),
                    "phase4_refresh_partial_influence_elapsed_ms_total": float(
                        phase4_refresh_partial_influence_elapsed_ms_total
                    ),
                    "phase4_refresh_rank_topk_elapsed_ms_total": float(
                        phase4_refresh_rank_topk_elapsed_ms_total
                    ),
                    "phase4_refresh_frontier_plan_elapsed_ms_total": float(
                        phase4_refresh_frontier_plan_elapsed_ms_total
                    ),
                    **phase4_execution_metadata,
                    **phase4_runtime_stream,
                },
            )
            cross_cluster_debug_summary["checkpoint_stream_count"] = int(
                len(cross_cluster_debug_checkpoints or [])
            )
            cross_cluster_debug_summary["batch_event_stream_count"] = int(
                len(cross_cluster_debug_batches or [])
            )
            compact_output_result["cross_cluster_debug_summary"] = cross_cluster_debug_summary
        if cross_cluster_debug_checkpoints is not None:
            compact_output_result["cross_cluster_debug_checkpoints"] = (
                cross_cluster_debug_checkpoints
            )
        if cross_cluster_debug_batches is not None:
            compact_output_result["cross_cluster_debug_batches"] = cross_cluster_debug_batches
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            file_backed_store_bytes = feature_row_store.nbytes
        else:
            edge_matrix = None
            release_dense_edge_matrix()
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
        telemetry_observer.phase(
            name="phase5.packaging",
            phase="phase5",
            elapsed_ms=phase5_elapsed_ms,
            attrs={
                "compact_output": True,
                "selected_features": int(selected_features.numel()),
                "feature_edge_rows": int(compact_output_result["feature_feature_edges"].shape[0]),
                "feature_edge_cols": int(compact_output_result["feature_feature_edges"].shape[1]),
            },
            wall_clock=True,
        )
        if prefix_view_metadata is not None:
            compact_output_result["prefix_view_metadata"] = dict(prefix_view_metadata)
            validate_compact_prefix_view_output(
                compact_output_result, n_layers=int(model.cfg.n_layers)
            )
        compact_output_result["phase0_window_state_reuse_requested"] = (
            phase0_context_override is not None
        )
        compact_output_result["phase0_window_state_reuse_effective"] = (
            phase0_context_override is not None
        )
        compact_output_result["target_logit_source"] = target_logit_source or (
            "override" if target_logits_override is not None else "context"
        )
        return Phase5Result(
            output=compact_output_result,
            compact_output_result=compact_output_result,
            edge_matrix=edge_matrix,
        )

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
        input_string=model.tokenizer.decode(input_ids[:n_pos]),
        input_tokens=input_ids[:n_pos],
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
    telemetry_observer.phase(
        name="phase5.packaging",
        phase="phase5",
        elapsed_ms=phase5_elapsed_ms,
        attrs={
            "compact_output": False,
            "adjacency_rows": int(full_edge_matrix.shape[0]),
            "adjacency_cols": int(full_edge_matrix.shape[1]),
        },
        wall_clock=True,
    )

    return Phase5Result(output=graph, compact_output_result=None, edge_matrix=edge_matrix)
