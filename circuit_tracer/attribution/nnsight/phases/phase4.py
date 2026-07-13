"""Phase 4 feature attribution execution for the NNSight backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.nnsight.phases.phase4_batches import execute_pending_frontier
from circuit_tracer.attribution.nnsight.phases.phase4_cleanup import finish_phase4
from circuit_tracer.attribution.nnsight.phases.phase4_diagnostics import (
    summarize_phase4_diagnostics,
)
from circuit_tracer.attribution.nnsight.phases.phase4_frontier import (
    initialize_phase4,
    prepare_feature_frontier,
)
from circuit_tracer.attribution.nnsight.phases.phase4_state import FeatureAttributionRun
from circuit_tracer.observability.events import TraceObserver


@dataclass(frozen=True)
class Phase4Inputs:
    logger: Any
    model: Any
    ctx: Any
    targets: AttributionTargets
    edge_matrix: torch.Tensor | None
    feat_ids: torch.Tensor
    feat_layers: torch.Tensor
    feat_pos: torch.Tensor
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
    row_to_node_index: torch.Tensor
    telemetry_observer: TraceObserver
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    anomaly_debug_result: dict[str, object] | None
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    rows_cpu_staging: torch.Tensor | None


@dataclass(frozen=True)
class Phase4Config:
    actual_max_feature_nodes: int
    total_active_feats: int
    n_logits: int
    logit_offset: int
    effective_feature_batch_size: int
    compute_microbatch_max_rows: int
    max_phase4_feature_batch_size: int
    update_interval: int
    row_store_capacity_feature_nodes: int
    exact_trace_internal_dtype_resolved: torch.dtype
    influence_compute_dtype: torch.dtype
    shadow_debug_compute_dtype: torch.dtype
    exact_chunked_decoder: bool
    use_compact_feature_row_store: bool
    planner_enabled: bool
    planner_status: str
    planner_skip_reason: str | None
    phase4_debug_summary_enabled: bool
    cross_cluster_debug_enabled: bool
    phase4_frontier_buffer_relative_epsilon: float | None
    phase4_frontier_buffer_max_extra_per_refresh: int
    phase4_frontier_buffer_max_extra_total: int
    phase4_refresh_prepared_chunk_cache_bytes_effective: int
    phase4_refresh_active_row_accumulation_effective: str
    phase4_scheduler_config: Any
    phase4_refresh_optimization_config: Any
    phase4_refresh_policy_config: Any
    phase4_ranker_config: Any
    phase4_row_executor_config: Any
    phase4_row_reduction_config: Any
    row_store_cache_control_config: Any
    exact_encoder_residency_config: Any
    profile: bool
    profile_log_interval: int
    verbose: bool
    full_retention_backend: str = "full_file"
    influence_row_tile_size: int = 4096
    influence_column_tile_size: int = 2048
    feature_row_column_tile_size: int = 2048
    feature_row_retention: str = "full_file"


@dataclass(frozen=True)
class Phase4Result:
    visited: torch.Tensor
    actual_max_feature_nodes: int
    edge_matrix: torch.Tensor | None
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
    row_to_node_index: torch.Tensor
    rows_cpu_staging: torch.Tensor | None
    st: int
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    anomaly_debug_result: dict[str, object] | None
    phase4_elapsed_ms: float
    phase4_feature_batch_size: int
    phase4_executor_reference_batch_size: int
    phase4_executor_microbatch_size: int
    phase4_refresh_count: int
    phase4_scheduler_reference_batch_count: int
    phase4_executor_microbatch_count: int
    phase4_refresh_elapsed_ms_total: float
    phase4_feature_batch_elapsed_ms_total: float
    phase4_refresh_partial_influence_elapsed_ms_total: float
    phase4_refresh_rank_topk_elapsed_ms_total: float
    phase4_refresh_frontier_plan_elapsed_ms_total: float
    phase4_refresh_row_store_read_elapsed_ms_total: float


def run_phase4(*, inputs: Phase4Inputs, config: Phase4Config) -> Phase4Result:
    """Sequence Phase 4 domain operations without owning their subsystem logic."""
    state = FeatureAttributionRun(inputs=inputs, config=config)
    initialize_phase4(state)
    while state.n_visited < state.actual_max_feature_nodes:
        prepare_feature_frontier(state)
        execute_pending_frontier(state)
    finish_phase4(state)
    summarize_phase4_diagnostics(state)
    return Phase4Result(
        visited=state.visited,
        actual_max_feature_nodes=state.actual_max_feature_nodes,
        edge_matrix=state.edge_matrix,
        feature_row_store=state.feature_row_store,
        nonfeature_row_store=state.nonfeature_row_store,
        row_to_node_index=state.row_to_node_index,
        rows_cpu_staging=state.rows_cpu_staging,
        st=state.st,
        phase4_frontier_buffer_metadata=state.phase4_frontier_buffer_metadata,
        phase4_execution_metadata=state.phase4_execution_metadata,
        cross_cluster_debug_summary=state.cross_cluster_debug_summary,
        cross_cluster_debug_checkpoints=state.cross_cluster_debug_checkpoints,
        cross_cluster_debug_batches=state.cross_cluster_debug_batches,
        anomaly_debug_result=state.anomaly_debug_result,
        phase4_elapsed_ms=state.phase4_elapsed_ms,
        phase4_feature_batch_size=state.phase4_feature_batch_size,
        phase4_executor_reference_batch_size=state.phase4_executor_reference_batch_size,
        phase4_executor_microbatch_size=state.phase4_executor_microbatch_size,
        phase4_refresh_count=state.phase4_refresh_count,
        phase4_scheduler_reference_batch_count=state.phase4_scheduler_reference_batch_count,
        phase4_executor_microbatch_count=state.phase4_executor_microbatch_count,
        phase4_refresh_elapsed_ms_total=state.phase4_refresh_elapsed_ms_total,
        phase4_feature_batch_elapsed_ms_total=state.phase4_feature_batch_elapsed_ms_total,
        phase4_refresh_partial_influence_elapsed_ms_total=state.phase4_refresh_partial_influence_elapsed_ms_total,
        phase4_refresh_rank_topk_elapsed_ms_total=state.phase4_refresh_rank_topk_elapsed_ms_total,
        phase4_refresh_frontier_plan_elapsed_ms_total=state.phase4_refresh_frontier_plan_elapsed_ms_total,
        phase4_refresh_row_store_read_elapsed_ms_total=state.phase4_refresh_row_store_read_elapsed_ms_total,
    )
