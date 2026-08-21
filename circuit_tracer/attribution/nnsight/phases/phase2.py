"""Phase 2 orchestration for targets, replay, storage, and evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.attribution.targets import AttributionTargets, TargetSpec
from circuit_tracer.attribution.nnsight.phases.phase2_evidence import (
    record_storage_evidence,
    record_target_replay_evidence,
)
from circuit_tracer.attribution.nnsight.phases.phase2_replay import (
    Phase0ReplayPolicy,
    Phase3ReplayPolicy,
    apply_phase0_replay,
    load_phase3_replay,
)
from circuit_tracer.attribution.nnsight.phases.phase2_storage import (
    FeatureRowInfluencePolicy,  # noqa: F401 - typed Phase-2 public surface
    FrontierBufferPolicy,
    FeatureRowStore,
    Phase2ExecutionPolicy,
    RowStoreLayout,
    RowStoreRuntime,
    _make_replay_lifecycle,  # noqa: F401 - retained public test seam
    open_row_storage,
    plan_active_feature_storage,
)
from circuit_tracer.attribution.nnsight.phases.phase2_targets import (
    TargetSelectionPolicy,
    select_attribution_targets,
)
from circuit_tracer.observability.events import MemoryBoundary, TraceObserver


@dataclass(frozen=True)
class Phase2Inputs:
    logger: Any
    model: Any
    ctx: Any
    input_ids: torch.Tensor
    activation_matrix: torch.Tensor
    telemetry_observer: TraceObserver
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    offload_handles: list[Any]
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None
    target_logits_override: torch.Tensor | None
    resource_owner: "Phase2ResourceOwner"


@dataclass
class Phase2ResourceOwner:
    """Expose Phase 2 row stores to the orchestrator if setup exits early."""

    feature_row_store: FeatureRowStore | None = None
    nonfeature_row_store: FeatureRowStore | None = None


@dataclass(frozen=True)
class Phase2Config:
    targets: TargetSelectionPolicy
    phase0_replay: Phase0ReplayPolicy
    phase3_replay: Phase3ReplayPolicy
    frontier: FrontierBufferPolicy
    storage_layout: RowStoreLayout
    storage_runtime: RowStoreRuntime
    execution: Phase2ExecutionPolicy


@dataclass(frozen=True)
class Phase2Result:
    targets: AttributionTargets
    activation_matrix: torch.Tensor
    feat_layers: torch.Tensor
    feat_pos: torch.Tensor
    feat_ids: torch.Tensor
    n_layers: int
    n_pos: int
    total_active_feats: int
    logit_offset: int
    n_logits: int
    total_nodes: int
    base_max_feature_nodes: int
    actual_max_feature_nodes: int
    row_store_capacity_feature_nodes: int
    feature_row_store: FeatureRowStore | None
    nonfeature_row_store: FeatureRowStore | None
    edge_matrix: torch.Tensor | None
    row_to_node_index: torch.Tensor
    phase0_donor_bundle_payload: dict[str, object] | None
    phase0_replay_metadata: dict[str, object]
    phase3_frontier_buffer_metadata: dict[str, object]
    phase4_frontier_buffer_metadata: dict[str, object]
    phase3_gradient_replay_metadata: dict[str, object]
    phase3_row_replay_metadata: dict[str, object]
    loaded_phase3_row_donor_bundle: dict[str, object] | None


def run_phase2(*, inputs: Phase2Inputs, config: Phase2Config) -> Phase2Result:
    """Sequence Phase 2 domain operations without owning their subsystem logic."""
    import time

    phase2_start = time.perf_counter()
    inputs.logger.info("Phase 2: Building input vectors")
    inputs.telemetry_observer.observe(MemoryBoundary("Phase 2 start", inputs.model.device))

    target_selection = select_attribution_targets(
        logger=inputs.logger,
        model=inputs.model,
        ctx=inputs.ctx,
        policy=config.targets,
        attribution_targets=inputs.attribution_targets,
        target_logits_override=inputs.target_logits_override,
    )
    phase0 = apply_phase0_replay(
        model=inputs.model,
        ctx=inputs.ctx,
        input_ids=inputs.input_ids,
        host_activation_matrix=inputs.activation_matrix,
        targets=target_selection,
        observer=inputs.telemetry_observer,
        policy=config.phase0_replay,
    )
    record_target_replay_evidence(
        model=inputs.model,
        ctx=inputs.ctx,
        targets=target_selection,
        phase0_replay_metadata=phase0.metadata,
        observer=inputs.telemetry_observer,
        debug_summary=inputs.cross_cluster_debug_summary,
        debug_checkpoints=inputs.cross_cluster_debug_checkpoints,
    )
    feature_plan = plan_active_feature_storage(
        logger=inputs.logger,
        model=inputs.model,
        activation_matrix=phase0.activation_matrix,
        n_logits=len(target_selection.targets),
        offload_handles=inputs.offload_handles,
        frontier=config.frontier,
        execution=config.execution,
    )
    storage = open_row_storage(
        ctx=inputs.ctx,
        model=inputs.model,
        plan=feature_plan,
        layout=config.storage_layout,
        runtime=config.storage_runtime,
        execution=config.execution,
        observer=inputs.telemetry_observer,
        offload_handles=inputs.offload_handles,
        owner=inputs.resource_owner,
    )
    record_storage_evidence(
        phase2_start=phase2_start,
        model=inputs.model,
        ctx=inputs.ctx,
        plan=feature_plan,
        storage=storage,
        layout=config.storage_layout,
        runtime=config.storage_runtime,
        execution=config.execution,
        phase0_metadata=phase0.metadata,
        observer=inputs.telemetry_observer,
        debug_summary=inputs.cross_cluster_debug_summary,
        debug_checkpoints=inputs.cross_cluster_debug_checkpoints,
    )
    phase3 = load_phase3_replay(
        ctx=inputs.ctx,
        targets=target_selection,
        activation_matrix=phase0.activation_matrix,
        n_layers=feature_plan.n_layers,
        n_pos=feature_plan.n_pos,
        trace_batch_size=config.execution.trace_batch_size,
        policy=config.phase3_replay,
        validation_context=phase0.validation_context,
    )
    return Phase2Result(
        targets=target_selection.targets,
        activation_matrix=phase0.activation_matrix,
        feat_layers=feature_plan.feat_layers,
        feat_pos=feature_plan.feat_pos,
        feat_ids=feature_plan.feat_ids,
        n_layers=feature_plan.n_layers,
        n_pos=feature_plan.n_pos,
        total_active_feats=feature_plan.total_active_feats,
        logit_offset=feature_plan.logit_offset,
        n_logits=feature_plan.n_logits,
        total_nodes=feature_plan.total_nodes,
        base_max_feature_nodes=feature_plan.base_max_feature_nodes,
        actual_max_feature_nodes=feature_plan.actual_max_feature_nodes,
        row_store_capacity_feature_nodes=feature_plan.row_store_capacity_feature_nodes,
        feature_row_store=storage.feature_row_store,
        nonfeature_row_store=storage.nonfeature_row_store,
        edge_matrix=storage.edge_matrix,
        row_to_node_index=storage.row_to_node_index,
        phase0_donor_bundle_payload=phase0.donor_bundle_payload,
        phase0_replay_metadata=phase0.metadata,
        phase3_frontier_buffer_metadata=feature_plan.phase3_frontier_buffer_metadata,
        phase4_frontier_buffer_metadata=feature_plan.phase4_frontier_buffer_metadata,
        phase3_gradient_replay_metadata=phase3.gradient_metadata,
        phase3_row_replay_metadata=phase3.row_metadata,
        loaded_phase3_row_donor_bundle=phase3.loaded_row_donor_bundle,
    )
