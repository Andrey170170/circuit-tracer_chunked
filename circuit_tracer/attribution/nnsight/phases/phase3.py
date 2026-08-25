"""Phase 3 logit attribution orchestration for NNSight attribution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from circuit_tracer.attribution.nnsight.row_denominator_evidence import (
    enable_row_denominator_audit,
)

from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.observability.events import MemoryBoundary, TraceObserver

from .phase3_batches import run_logit_batches
from .phase3_evidence import package_phase3_replay_evidence
from .phase3_frontier import select_phase3_frontier


@dataclass(frozen=True)
class Phase3Inputs:
    logger: Any
    model: Any
    ctx: Any
    targets: AttributionTargets
    activation_matrix: torch.Tensor
    feat_layers: torch.Tensor
    feat_pos: torch.Tensor
    feat_ids: torch.Tensor
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
    edge_matrix: torch.Tensor | None
    row_to_node_index: torch.Tensor
    telemetry_observer: TraceObserver
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None
    anomaly_debug_result: dict[str, object] | None
    loaded_phase3_row_donor_bundle: dict[str, object] | None
    phase3_frontier_buffer_metadata: dict[str, object]
    phase3_gradient_bundle_payload: dict[str, object] | None
    phase3_row_bundle_payload: dict[str, object] | None
    phase3_seed_bundle_payload: dict[str, object] | None
    feature_semantic_descriptors_payload: dict[str, object] | None


@dataclass(frozen=True)
class Phase3Config:
    effective_logit_batch_size: int
    compute_microbatch_max_rows: int
    effective_feature_batch_size: int
    output_position: int | None
    n_layers: int
    n_pos: int
    n_logits: int
    logit_offset: int
    total_active_feats: int
    base_max_feature_nodes: int
    actual_max_feature_nodes: int
    exact_trace_internal_dtype_resolved: torch.dtype
    phase3_gradient_replay_mode_resolved: str
    phase3_row_replay_mode_resolved: str
    capture_phase3_gradient_bundle_enabled: bool
    capture_phase3_row_bundle_enabled: bool
    capture_phase3_seed_bundle_enabled: bool
    capture_feature_semantic_descriptors_enabled: bool
    phase3_frontier_buffer_relative_epsilon: float | None
    phase3_frontier_buffer_max_extra: int
    update_interval: int
    planner_compute_dtype: torch.dtype
    influence_compute_dtype: torch.dtype
    shadow_debug_compute_dtype: torch.dtype
    phase4_refresh_policy_config: Any
    exact_chunked_decoder: bool
    use_compact_feature_row_store: bool
    semantic_descriptor_top_k: int
    semantic_descriptor_dim: int
    profile: bool
    profile_log_interval: int
    full_retention_backend: str = "full_file"
    influence_row_tile_size: int = 4096
    influence_column_tile_size: int = 2048
    feature_row_column_tile_size: int = 2048
    feature_row_retention: str = "full_file"


@dataclass(frozen=True)
class Phase3Result:
    stored_row_count: int
    row_to_node_index: torch.Tensor
    rows_cpu_staging: torch.Tensor | None
    actual_max_feature_nodes: int
    phase3_frontier_buffer_metadata: dict[str, object]
    phase3_gradient_bundle_payload: dict[str, object] | None
    phase3_row_bundle_payload: dict[str, object] | None
    phase3_seed_bundle_payload: dict[str, object] | None
    feature_semantic_descriptors_payload: dict[str, object] | None
    anomaly_debug_result: dict[str, object] | None
    compute_batch_elapsed_ms_total: float
    cpu_staging_elapsed_ms_total: float
    denominator_elapsed_ms_total: float
    row_store_write_elapsed_ms_total: float
    gpu_to_cpu_bytes_total: int
    cpu_to_gpu_bytes_total: int
    copy_count: int


def run_phase3(*, inputs: Phase3Inputs, config: Phase3Config) -> Phase3Result:
    """Run ordered logit attribution, replay capture, and frontier selection."""
    inputs.logger.info("Phase 3: Computing logit attributions")
    if config.capture_feature_semantic_descriptors_enabled:
        enable_row_denominator_audit(inputs.feature_row_store)
    phase_start = time.perf_counter()
    inputs.telemetry_observer.observe(MemoryBoundary("Phase 3 start", inputs.model.device))
    batches = run_logit_batches(inputs=inputs, config=config, phase_start=phase_start)
    evidence = package_phase3_replay_evidence(
        inputs=inputs, config=config, rows=batches.replay_rows
    )
    frontier = select_phase3_frontier(inputs=inputs, config=config)
    metrics = batches.metrics
    return Phase3Result(
        stored_row_count=int(config.n_logits),
        row_to_node_index=inputs.row_to_node_index,
        rows_cpu_staging=batches.rows_cpu_staging,
        actual_max_feature_nodes=frontier.actual_max_feature_nodes,
        phase3_frontier_buffer_metadata=frontier.buffer_metadata,
        phase3_gradient_bundle_payload=evidence.gradient_bundle,
        phase3_row_bundle_payload=evidence.row_bundle,
        phase3_seed_bundle_payload=frontier.seed_bundle,
        feature_semantic_descriptors_payload=frontier.semantic_descriptors,
        anomaly_debug_result=inputs.anomaly_debug_result,
        compute_batch_elapsed_ms_total=metrics.compute_batch_elapsed_ms,
        cpu_staging_elapsed_ms_total=metrics.cpu_staging_elapsed_ms,
        denominator_elapsed_ms_total=metrics.denominator_elapsed_ms,
        row_store_write_elapsed_ms_total=metrics.row_store_write_elapsed_ms,
        gpu_to_cpu_bytes_total=metrics.gpu_to_cpu_bytes,
        cpu_to_gpu_bytes_total=metrics.cpu_to_gpu_bytes,
        copy_count=metrics.copy_count,
    )
