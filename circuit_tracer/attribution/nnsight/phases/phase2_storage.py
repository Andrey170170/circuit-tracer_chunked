"""Active-feature planning and row-storage ownership for attribution Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Any, Literal, Protocol, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import (
    _build_phase3_frontier_buffer_metadata,
)
from circuit_tracer.attribution.nnsight.row_replay import (
    NonfeatureProjectionLedger,
    ReplayGraphLifecycle,
    RowRecipe,
    RowRecipeLedger,
)
from circuit_tracer.attribution.nnsight.row_store import (
    FeatureRowStore,
    _ColumnTiledFeatureRowStore,
    _FileBackedFeatureRowStore,
    _GpuResidentFeatureRowStore,
)
from circuit_tracer.transcoder.provider import provider_fingerprint
from circuit_tracer.utils.disk_offload import offload_modules


class RowStoreOwnership(Protocol):
    feature_row_store: FeatureRowStore | None
    nonfeature_row_store: FeatureRowStore | None


@dataclass(frozen=True)
class FrontierBufferPolicy:
    phase3_relative_epsilon: float | None
    phase3_max_extra: int
    phase4_relative_epsilon: float | None
    phase4_max_extra_per_refresh: int
    phase4_max_extra_total: int

    def __post_init__(self) -> None:
        if min(self.phase3_max_extra, self.phase4_max_extra_per_refresh,
               self.phase4_max_extra_total) < 0:
            raise ValueError("frontier buffer limits must be non-negative")


@dataclass(frozen=True)
class RowStoreLayout:
    retention: Literal["full_file", "none_recompute"]
    backend: Literal["full_file", "column_tiled_v1"]
    feature_column_tile_size: int
    influence_row_tile_size: int
    influence_column_tile_size: int
    feature_dtype: torch.dtype
    row_abs_sum_dtype: torch.dtype

    def __post_init__(self) -> None:
        if min(self.feature_column_tile_size, self.influence_row_tile_size,
               self.influence_column_tile_size) <= 0:
            raise ValueError("row-store tile sizes must be positive")
        if self.retention == "none_recompute" and self.backend != "full_file":
            raise ValueError("recomputed rows cannot select a file-retention backend")


@dataclass(frozen=True)
class RowStoreRuntime:
    cache_control: Any
    temp_root_policy: Literal["default", "env_node_local"]
    temp_root: str | os.PathLike[str] | None
    preallocate: bool
    prepared_chunk_cache_bytes: int
    replay_tile_cache_bytes: int
    gpu_resident_max_bytes: int = 0
    gpu_resident_safety_margin_bytes: int = 0
    gpu_resident_device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        if (
            self.prepared_chunk_cache_bytes < 0
            or self.replay_tile_cache_bytes < 0
            or self.gpu_resident_max_bytes < 0
            or self.gpu_resident_safety_margin_bytes < 0
        ):
            raise ValueError("row-store cache sizes must be non-negative")


@dataclass(frozen=True)
class Phase2ExecutionPolicy:
    offload: Literal["cpu", "disk", None]
    max_feature_nodes: int | None
    compact_output: bool
    exact_chunked_decoder: bool
    use_compact_feature_row_store: bool
    exact_dtype: torch.dtype
    effective_feature_batch_size: int
    trace_batch_size: int

    def __post_init__(self) -> None:
        if self.effective_feature_batch_size <= 0 or self.trace_batch_size <= 0:
            raise ValueError("Phase 2 batch sizes must be positive")
        if self.max_feature_nodes is not None and self.max_feature_nodes <= 0:
            raise ValueError("max_feature_nodes must be positive when provided")


@dataclass(frozen=True)
class ActiveFeaturePlan:
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
    phase3_frontier_buffer_metadata: dict[str, object]
    phase4_frontier_buffer_metadata: dict[str, object]


@dataclass(frozen=True)
class OpenedRowStorage:
    feature_row_store: FeatureRowStore | None
    nonfeature_row_store: FeatureRowStore | None
    edge_matrix: torch.Tensor | None
    row_to_node_index: torch.Tensor


def _make_replay_lifecycle(ctx: Any, offload_handles: list[Any]) -> ReplayGraphLifecycle:
    def rebuild_forward() -> None:
        while offload_handles:
            offload_handles[0]()
            del offload_handles[0]
        ctx.rebuild_saved_graph_handles()

    return ReplayGraphLifecycle(
        reset=ctx.reset_saved_graph_handles,
        rebuild_forward=rebuild_forward,
        release=ctx.release_saved_graph_handles,
    )


def plan_active_feature_storage(
    *,
    logger: Any,
    model: Any,
    activation_matrix: torch.Tensor,
    n_logits: int,
    offload_handles: list[Any],
    frontier: FrontierBufferPolicy,
    execution: Phase2ExecutionPolicy,
) -> ActiveFeaturePlan:
    """Freeze feature order, offload embeddings, and size the Phase-3/4 frontier."""
    feat_layers, feat_pos, feat_ids = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = int(activation_matrix._nnz())
    if execution.offload:
        offload_handles += offload_modules([model.embed_location], execution.offload)
        tied_embeds = (
            model.embed_weight.untyped_storage().data_ptr()
            == model.unembed_weight.untyped_storage().data_ptr()
        )
        if not tied_embeds:
            offload_handles += offload_modules([model.lm_head], execution.offload)
    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    total_nodes = logit_offset + n_logits
    base_max = min(execution.max_feature_nodes or total_active_feats, total_active_feats)
    phase3_metadata = _build_phase3_frontier_buffer_metadata(
        seed_feature_influences=None,
        base_max_feature_nodes=base_max,
        total_active_features=total_active_feats,
        relative_epsilon=frontier.phase3_relative_epsilon,
        max_extra=frontier.phase3_max_extra,
    )
    capacity = min(
        base_max
        + (frontier.phase3_max_extra if frontier.phase3_relative_epsilon is not None else 0)
        + (frontier.phase4_max_extra_total if frontier.phase4_relative_epsilon is not None else 0),
        total_active_feats,
    )
    phase4_metadata: dict[str, object] = {
        "schema_version": 1,
        "requested": bool(frontier.phase4_relative_epsilon is not None
                          or frontier.phase4_max_extra_per_refresh > 0
                          or frontier.phase4_max_extra_total > 0),
        "enabled": bool(frontier.phase4_relative_epsilon is not None
                        and frontier.phase4_max_extra_per_refresh > 0
                        and frontier.phase4_max_extra_total > 0),
        "effective": False,
        "relative_epsilon": (None if frontier.phase4_relative_epsilon is None
                             else float(frontier.phase4_relative_epsilon)),
        "max_extra_per_refresh": frontier.phase4_max_extra_per_refresh,
        "max_extra_total": frontier.phase4_max_extra_total,
        "extra_feature_count_total": 0, "expanded_refresh_count": 0,
        "fallback_count": 0, "capacity_feature_nodes": capacity,
        "initial_target_feature_nodes": base_max,
        "final_actual_max_feature_nodes": base_max, "events": [],
    }
    logger.info(f"Will include {base_max} of {total_active_feats} feature nodes")
    return ActiveFeaturePlan(
        feat_layers, feat_pos, feat_ids, int(n_layers), int(n_pos), total_active_feats,
        int(logit_offset), n_logits, int(total_nodes), int(base_max), int(base_max),
        int(capacity), phase3_metadata, phase4_metadata,
    )


def _open_replay_ledgers(
    *, ctx: Any, model: Any, plan: ActiveFeaturePlan, layout: RowStoreLayout,
    runtime: RowStoreRuntime, execution: Phase2ExecutionPolicy,
    offload_handles: list[Any], owner: RowStoreOwnership,
) -> tuple[FeatureRowStore, FeatureRowStore]:
    def feature_producer(recipes: Sequence[RowRecipe], start: int, end: int) -> torch.Tensor:
        assert all(recipe.injection is not None for recipe in recipes)
        return ctx.compute_batch(
            layers=torch.tensor([recipe.layer for recipe in recipes]),
            positions=torch.tensor([recipe.position for recipe in recipes]),
            inject_values=torch.stack([cast(torch.Tensor, recipe.injection) for recipe in recipes]),
            feature_column_range=(start, end), include_nonfeature=False,
            phase_label="row_replay_feature",
        ).detach().to(device="cpu", dtype=execution.exact_dtype)

    def nonfeature_producer(recipes: Sequence[RowRecipe], start: int, end: int) -> torch.Tensor:
        assert all(recipe.injection is not None for recipe in recipes)
        values = ctx.compute_batch(
            layers=torch.tensor([recipe.layer for recipe in recipes]),
            positions=torch.tensor([recipe.position for recipe in recipes]),
            inject_values=torch.stack([cast(torch.Tensor, recipe.injection) for recipe in recipes]),
            feature_column_range=(0, 0), include_nonfeature=True,
            phase_label="row_replay_nonfeature",
        )
        return values[:, start:end].detach().to(device="cpu", dtype=execution.exact_dtype)

    common = dict(
        n_rows=plan.row_store_capacity_feature_nodes + plan.n_logits,
        dtype=execution.exact_dtype,
        semantic_fingerprint={"active_features": plan.total_active_feats},
        execution_fingerprint={"trace_batch_size": execution.trace_batch_size},
        provider_fingerprint=provider_fingerprint(model.transcoders),
        tile_cache_bytes=runtime.replay_tile_cache_bytes,
        replay_batch_rows=execution.trace_batch_size,
        max_request_rows=layout.influence_row_tile_size,
        max_request_columns=layout.influence_column_tile_size,
    )
    feature_store = RowRecipeLedger(
        n_feature_columns=plan.total_active_feats, producer=feature_producer,
        lifecycle=_make_replay_lifecycle(ctx, offload_handles), **common,
    )
    owner.feature_row_store = feature_store
    nonfeature_store = NonfeatureProjectionLedger(
        n_feature_columns=plan.logit_offset - plan.total_active_feats,
        producer=nonfeature_producer,
        lifecycle=_make_replay_lifecycle(ctx, offload_handles), **common,
    )
    owner.nonfeature_row_store = nonfeature_store
    return feature_store, nonfeature_store


def _open_file_stores(
    *, plan: ActiveFeaturePlan, layout: RowStoreLayout, runtime: RowStoreRuntime,
    execution: Phase2ExecutionPolicy, observer: Any, owner: RowStoreOwnership,
) -> tuple[FeatureRowStore, FeatureRowStore]:
    store_class = (_ColumnTiledFeatureRowStore
                   if layout.backend == "column_tiled_v1" else _FileBackedFeatureRowStore)
    tiled_kwargs: dict[str, int] = {}
    if layout.backend == "column_tiled_v1":
        tiled_kwargs = {
            "column_tile_size": layout.feature_column_tile_size,
            "max_request_rows": max(layout.influence_row_tile_size,
                                    execution.effective_feature_batch_size),
            "max_request_columns": max(4096, layout.influence_column_tile_size,
                                       layout.feature_column_tile_size),
        }
    common: dict[str, object] = dict(
        n_rows=plan.row_store_capacity_feature_nodes + plan.n_logits,
        dtype=execution.exact_dtype, row_abs_sum_dtype=execution.exact_dtype,
        read_chunk_cache_bytes=256 * 1024 * 1024,
        row_store_cache_control_mode=runtime.cache_control.effective_mode,
        temp_root_policy=runtime.temp_root_policy, temp_root=runtime.temp_root,
        preallocate=runtime.preallocate, trace_observer=observer, **tiled_kwargs,
    )
    feature_store = cast(FeatureRowStore, store_class(
        n_feature_columns=plan.total_active_feats,
        prepared_read_cache_bytes=runtime.prepared_chunk_cache_bytes, **common,
    ))
    if runtime.gpu_resident_max_bytes > 0:
        if layout.backend != "full_file":
            raise ValueError("GPU full-residency tier requires full_file row storage")
        assert isinstance(feature_store, _FileBackedFeatureRowStore)
        feature_store = _GpuResidentFeatureRowStore(
            backing_store=feature_store,
            max_bytes=runtime.gpu_resident_max_bytes,
            safety_margin_bytes=runtime.gpu_resident_safety_margin_bytes,
            device=runtime.gpu_resident_device,
        )
    owner.feature_row_store = feature_store
    nonfeature_store = cast(FeatureRowStore, store_class(
        n_feature_columns=plan.logit_offset - plan.total_active_feats,
        prepared_read_cache_bytes=0, **common,
    ))
    owner.nonfeature_row_store = nonfeature_store
    return feature_store, nonfeature_store


def open_row_storage(
    *, ctx: Any, model: Any, plan: ActiveFeaturePlan, layout: RowStoreLayout,
    runtime: RowStoreRuntime, execution: Phase2ExecutionPolicy, observer: Any,
    offload_handles: list[Any], owner: RowStoreOwnership,
) -> OpenedRowStorage:
    """Open the selected storage backend and transfer cleanup ownership immediately."""
    feature_store = nonfeature_store = None
    edge_matrix = None
    if execution.use_compact_feature_row_store:
        assert execution.compact_output and execution.exact_chunked_decoder
        if layout.retention == "none_recompute":
            feature_store, nonfeature_store = _open_replay_ledgers(
                ctx=ctx, model=model, plan=plan, layout=layout, runtime=runtime,
                execution=execution, offload_handles=offload_handles, owner=owner,
            )
        else:
            feature_store, nonfeature_store = _open_file_stores(
                plan=plan, layout=layout, runtime=runtime, execution=execution,
                observer=observer, owner=owner,
            )
    else:
        edge_matrix = torch.zeros(
            plan.row_store_capacity_feature_nodes + plan.n_logits, plan.total_nodes
        )
    row_to_node_index = torch.zeros(
        plan.row_store_capacity_feature_nodes + plan.n_logits, dtype=torch.int32
    )
    return OpenedRowStorage(feature_store, nonfeature_store, edge_matrix, row_to_node_index)
