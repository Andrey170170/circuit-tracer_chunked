"""Domain contracts shared by Phase 5 graph assembly operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.graph import Graph


@dataclass(frozen=True)
class GraphAssemblyRuntime:
    logger: Any
    model: Any
    ctx: Any
    targets: Any
    observer: Any
    input_ids: torch.Tensor


@dataclass(frozen=True)
class GraphAssemblyState:
    activation_matrix: torch.Tensor
    visited: torch.Tensor
    edge_matrix: torch.Tensor | None
    row_to_node_index: torch.Tensor
    feature_row_store: Any | None
    nonfeature_row_store: Any | None

    def __post_init__(self) -> None:
        if self.activation_matrix.layout != torch.sparse_coo:
            raise ValueError("Phase 5 activation matrix must be sparse COO")
        if self.visited.ndim != 1 or self.row_to_node_index.ndim != 1:
            raise ValueError("Phase 5 visited and row mappings must be vectors")


@dataclass(frozen=True)
class ReplayArtifacts:
    phase0_replay_metadata: dict[str, object]
    phase3_gradient_replay_metadata: dict[str, object]
    phase3_row_replay_metadata: dict[str, object]
    phase0_donor_bundle_payload: dict[str, object] | None
    phase3_seed_bundle_payload: dict[str, object] | None
    phase3_gradient_bundle_payload: dict[str, object] | None
    phase3_row_bundle_payload: dict[str, object] | None


@dataclass(frozen=True)
class DiagnosticArtifacts:
    phase3_frontier_buffer_metadata: dict[str, object]
    phase4_frontier_buffer_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    feature_semantic_descriptors_payload: dict[str, object] | None
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cross_cluster_debug_batches: list[dict[str, object]] | None


@dataclass(frozen=True)
class GraphOutputOwnership:
    prefix_view_metadata: Any | None
    publish_compact_output_result: Callable[[dict[str, object]], None]
    release_dense_edge_matrix: Callable[[], None]


@dataclass(frozen=True)
class Phase5Inputs:
    runtime: GraphAssemblyRuntime
    graph: GraphAssemblyState
    replay: ReplayArtifacts
    diagnostics: DiagnosticArtifacts
    output: GraphOutputOwnership


@dataclass(frozen=True)
class OutputArtifactPolicy:
    compact_output: bool
    use_compact_feature_row_store: bool
    capture_feature_semantic_descriptors: bool
    capture_phase0_donor_bundle: bool
    capture_phase3_seed_bundle: bool
    capture_phase3_gradient_bundle: bool
    capture_phase3_row_bundle: bool
    cross_cluster_debug_enabled: bool
    phase4_anomaly_debug_enabled: bool


@dataclass(frozen=True)
class GraphAssemblyLimits:
    n_pos: int
    n_logits: int
    st: int
    total_active_feats: int
    total_nodes: int
    actual_max_feature_nodes: int

    def __post_init__(self) -> None:
        if min(self.n_pos, self.n_logits, self.total_nodes) <= 0:
            raise ValueError("Phase 5 graph dimensions must be positive")
        if not self.n_logits <= self.st <= self.n_logits + self.actual_max_feature_nodes:
            raise ValueError("Phase 5 row frontier is outside the allocated graph rows")


@dataclass(frozen=True)
class BatchExecutionSummary:
    batch_size: int
    feature_batch_size: int | None
    max_phase4_feature_batch_size: int
    planner_enabled: bool
    planner_status: str
    planner_skip_reason: str | None
    phase1_trace_batch_metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_phase4_feature_batch_size <= 0:
            raise ValueError("Phase 5 batch sizes must be positive")


@dataclass(frozen=True)
class Phase4PolicySummary:
    phase4_scheduler_config: Any
    phase4_refresh_optimization_config: Any
    phase4_row_executor_config: Any
    phase4_row_reduction_config: Any
    prepared_chunk_cache_bytes: int
    prepared_chunk_cache_bytes_effective: int
    active_row_accumulation: str
    active_row_accumulation_effective: str
    refresh_aux_fallback_reason: str | None
    refresh_aux_applicable: bool


@dataclass(frozen=True)
class NumericExecutionSummary:
    internal_precision_requested: str
    resolved_dtype_map: dict[str, object]
    activation_compare_mode: str
    exact_dtype_name: str
    telemetry_max_events: int


@dataclass(frozen=True)
class Phase4WorkSummary:
    semantic_descriptor_top_k: int
    semantic_descriptor_dim: int
    feature_batch_size: int
    executor_reference_batch_size: int
    executor_microbatch_size: int
    refresh_count: int
    scheduler_reference_batch_count: int
    executor_microbatch_count: int


@dataclass(frozen=True)
class Phase4TimingSummary:
    elapsed_ms: float
    refresh_elapsed_ms: float
    feature_batch_elapsed_ms: float
    partial_influence_elapsed_ms: float
    rank_topk_elapsed_ms: float
    frontier_plan_elapsed_ms: float
    row_store_read_elapsed_ms: float


@dataclass(frozen=True)
class RunProvenance:
    start_time: float
    phase0_context_override: Any | None
    target_logit_source: str | None
    target_logits_override: torch.Tensor | None


@dataclass(frozen=True)
class Phase5Config:
    output_policy: OutputArtifactPolicy
    graph_limits: GraphAssemblyLimits
    batches: BatchExecutionSummary
    phase4_policy: Phase4PolicySummary
    numerics: NumericExecutionSummary
    phase4_work: Phase4WorkSummary
    phase4_timings: Phase4TimingSummary
    provenance: RunProvenance


@dataclass(frozen=True)
class Phase5Result:
    output: dict[str, object] | Graph
    compact_output_result: dict[str, object] | None
    edge_matrix: torch.Tensor | None
