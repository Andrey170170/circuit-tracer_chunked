"""Influence recomputation and ranking operations for NNSight Phase 4."""

from __future__ import annotations
import time
from typing import cast
import torch
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _compute_phase4_rank_selection_max_feature_nodes_cap_bound,
    _compute_phase4_refresh_queue_window_size,
    _rank_phase4_unvisited_features_argsort,
    _select_phase4_frontier_rank_selection,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_phase4_frontier_buffer_decision,
    _build_phase4_normalization_stats,
    _build_vector_stats,
)
from circuit_tracer.attribution.nnsight.telemetry import _safe_float, _safe_int
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_feature_influences_tiled,
    compute_partial_influences,
)


def recompute_feature_influences(state):
    """Recompute partial feature influences without changing reference numerics."""
    state.refresh_index = int(state.phase4_refresh_count)
    state.pending_refresh_index = state.refresh_index
    state.refresh_start = time.perf_counter()
    state.refresh_memory_before = state.memory_snapshot()
    state.feature_row_store_snapshot_before = (
        state.feature_row_store.get_diagnostic_snapshot()
        if state.use_compact_feature_row_store and state.feature_row_store is not None
        else None
    )
    state.streaming_chunk_reuse_stats: dict[str, int | float | str] | None = None
    state.refresh_row_store_read_elapsed_ms: float | None = None
    state.refresh_influence_normalization_elapsed_ms: float | None = None
    state.refresh_influence_matmul_elapsed_ms: float | None = None
    state.refresh_chunk_request_count: int | None = None
    state.refresh_active_row_chunk_count: int | None = None
    state.refresh_rows_touched: int | None = None
    state.refresh_solver_iteration_count: int | None = None
    state.refresh_row_chunk_strategy: str | None = None
    state.refresh_row_weight_nonzero_rows: int | None = None
    state.refresh_row_weight_zero_rows: int | None = None
    state.refresh_row_reader_overread_zero_rows: int | None = None
    state.refresh_active_row_range_count: int | None = None
    state.partial_influence_start = time.perf_counter()
    if state.use_compact_feature_row_store:
        assert state.feature_row_store is not None
        state.streaming_chunk_reuse_stats = {}
        state.refresh_active_row_only_chunks = (
            state.phase4_refresh_optimization_config.effective_mode == "v1"
        )
        state.row_denominator_prefix = (
            state.feature_row_store.row_abs_max[: state.st],
            state.feature_row_store.row_l1_scaled[: state.st],
        )
        state.refresh_prepared_row_reader = bool(
            state.phase4_refresh_prepared_chunk_cache_bytes_effective > 0
        )
        if state.refresh_prepared_row_reader:

            def refresh_row_reader(row_start: int, row_end: int) -> torch.Tensor:
                return state.feature_row_store.read_prepared_feature_rows(
                    row_start,
                    row_end,
                    device=state.feature_row_store.row_abs_max.device,
                    dtype=state.influence_compute_dtype,
                    phase="phase4",
                )

            state.refresh_row_reader = refresh_row_reader
        else:

            def refresh_row_reader(row_start: int, row_end: int) -> torch.Tensor:
                return state.feature_row_store.read_feature_rows(row_start, row_end, phase="phase4")

            state.refresh_row_reader = refresh_row_reader
        if (
            state.config.full_retention_backend == "column_tiled_v1"
            or state.config.feature_row_retention == "none_recompute"
        ):
            state.feature_influences = compute_partial_feature_influences_tiled(
                state.feature_row_store.read_tile,
                state.row_denominator_prefix,
                state.phase4_logit_probabilities,
                state.row_to_node_index[: state.st],
                n_feature_nodes=state.total_active_feats,
                n_logits=state.n_logits,
                row_tile_size=state.config.influence_row_tile_size,
                column_tile_size=state.config.influence_column_tile_size,
                device=state.feature_row_store.row_abs_max.device,
                compute_dtype=state.influence_compute_dtype,
                telemetry=state.streaming_chunk_reuse_stats,
            )
        else:
            state.feature_influences = compute_partial_feature_influences_streaming(
                state.refresh_row_reader,
                state.row_denominator_prefix,
                state.phase4_logit_probabilities,
                state.row_to_node_index[: state.st],
                n_feature_nodes=state.total_active_feats,
                n_logits=state.n_logits,
                device=state.feature_row_store.row_abs_max.device,
                chunk_reuse_stats=state.streaming_chunk_reuse_stats,
                compute_dtype=state.influence_compute_dtype,
                active_row_only_chunks=state.refresh_active_row_only_chunks,
                row_reader_returns_prepared=state.refresh_prepared_row_reader,
                active_row_accumulation=state.phase4_refresh_active_row_accumulation_effective,
            )
        state.refresh_row_store_read_elapsed_ms = _safe_float(
            state.streaming_chunk_reuse_stats.get("row_reader_elapsed_ms_total")
        )
        state.refresh_influence_normalization_elapsed_ms = _safe_float(
            state.streaming_chunk_reuse_stats.get("normalization_elapsed_ms_total")
        )
        state.refresh_influence_matmul_elapsed_ms = _safe_float(
            state.streaming_chunk_reuse_stats.get("matmul_elapsed_ms_total")
        )
        state.refresh_direct_accumulation_elapsed_ms = _safe_float(
            state.streaming_chunk_reuse_stats.get("direct_accumulation_elapsed_ms_total")
        )
        if state.refresh_direct_accumulation_elapsed_ms is not None:
            state.refresh_influence_matmul_elapsed_ms = float(
                state.refresh_influence_matmul_elapsed_ms or 0.0
            ) + float(state.refresh_direct_accumulation_elapsed_ms)
        state.refresh_chunk_request_count = _safe_int(
            state.streaming_chunk_reuse_stats.get("chunk_request_count")
        )
        state.refresh_active_row_chunk_count = _safe_int(
            state.streaming_chunk_reuse_stats.get("active_row_chunk_count")
        )
        state.refresh_rows_touched = _safe_int(
            state.streaming_chunk_reuse_stats.get("row_reader_row_count")
        )
        state.refresh_solver_iteration_count = _safe_int(
            state.streaming_chunk_reuse_stats.get("iteration_count")
        )
        state.row_chunk_strategy_value = state.streaming_chunk_reuse_stats.get("row_chunk_strategy")
        if isinstance(state.row_chunk_strategy_value, str):
            state.refresh_row_chunk_strategy = state.row_chunk_strategy_value
        state.refresh_row_weight_nonzero_rows = _safe_int(
            state.streaming_chunk_reuse_stats.get("row_weight_nonzero_row_count")
        )
        state.refresh_row_weight_zero_rows = _safe_int(
            state.streaming_chunk_reuse_stats.get("row_weight_zero_row_count")
        )
        state.refresh_row_reader_overread_zero_rows = _safe_int(
            state.streaming_chunk_reuse_stats.get("row_reader_overread_zero_row_count")
        )
        state.refresh_active_row_range_count = _safe_int(
            state.streaming_chunk_reuse_stats.get("active_row_range_count")
        )
    else:
        state.influences = compute_partial_influences(
            state.edge_matrix[: state.st],
            state.phase4_logit_probabilities,
            state.row_to_node_index[: state.st],
            device=state.edge_matrix.device,
        )
        state.feature_influences = state.influences[: state.total_active_feats]
    state.refresh_partial_influence_elapsed_ms = (
        time.perf_counter() - state.partial_influence_start
    ) * 1000.0
    state.phase4_refresh_partial_influence_elapsed_ms_total += (
        state.refresh_partial_influence_elapsed_ms
    )
    if state.refresh_row_store_read_elapsed_ms is not None:
        state.phase4_refresh_row_store_read_elapsed_ms_total += (
            state.refresh_row_store_read_elapsed_ms
        )
    if state.refresh_influence_normalization_elapsed_ms is not None:
        state.phase4_refresh_influence_normalization_elapsed_ms_total += (
            state.refresh_influence_normalization_elapsed_ms
        )
    if state.refresh_influence_matmul_elapsed_ms is not None:
        state.phase4_refresh_influence_matmul_elapsed_ms_total += (
            state.refresh_influence_matmul_elapsed_ms
        )


def rank_feature_frontier(state):
    """Rank unvisited features and apply bounded frontier expansion."""
    state.max_frontier_size = min(
        _compute_phase4_refresh_queue_window_size(
            update_interval=state.update_interval,
            phase4_feature_batch_size=state.phase4_feature_batch_size,
            queue_multiplier=state.phase4_refresh_queue_multiplier,
        ),
        int(state.actual_max_feature_nodes - state.n_visited),
    )
    state.phase4_frontier_buffer_event: dict[str, object] | None = None
    if bool(state.phase4_frontier_buffer_metadata["enabled"]):
        state.unvisited_scores_for_buffer = state.feature_influences[
            _rank_phase4_unvisited_features_argsort(state.feature_influences, state.visited)
        ]
        state.buffer_decision = _build_phase4_frontier_buffer_decision(
            candidate_scores=state.unvisited_scores_for_buffer,
            base_frontier_size=int(state.max_frontier_size),
            actual_max_feature_nodes=int(state.actual_max_feature_nodes),
            capacity_feature_nodes=int(state.row_store_capacity_feature_nodes),
            total_active_features=int(state.total_active_feats),
            used_total=int(state.phase4_frontier_buffer_extra_used_total),
            epsilon=state.phase4_frontier_buffer_relative_epsilon,
            max_per_refresh=int(state.phase4_frontier_buffer_max_extra_per_refresh),
            max_total=int(state.phase4_frontier_buffer_max_extra_total),
            refresh_index=state.refresh_index,
            visited_before=int(state.n_visited),
        )
        state.extra = int(state.buffer_decision["extra_feature_count"])
        state.phase4_frontier_buffer_event = cast(dict[str, object], state.buffer_decision["event"])
        cast(list[dict[str, object]], state.phase4_frontier_buffer_metadata["events"]).append(
            state.phase4_frontier_buffer_event
        )
        if state.extra > 0:
            state.phase4_frontier_buffer_extra_used_total += state.extra
            state.actual_max_feature_nodes += state.extra
            state.max_frontier_size = int(state.buffer_decision["expanded_frontier_size"])
            state.phase4_frontier_buffer_metadata["extra_feature_count_total"] = int(
                state.phase4_frontier_buffer_extra_used_total
            )
            state.phase4_frontier_buffer_metadata["expanded_refresh_count"] = (
                int(state.phase4_frontier_buffer_metadata["expanded_refresh_count"]) + 1
            )
            state.phase4_frontier_buffer_metadata["effective"] = True
            state.phase4_frontier_buffer_metadata["final_actual_max_feature_nodes"] = int(
                state.actual_max_feature_nodes
            )
            if getattr(state.pbar, "total", None) is not None:
                state.pbar.total = int(state.actual_max_feature_nodes)
                state.pbar.refresh()
        elif state.phase4_frontier_buffer_event.get("fallback_reason") is not None:
            state.phase4_frontier_buffer_metadata["fallback_count"] = (
                int(state.phase4_frontier_buffer_metadata["fallback_count"]) + 1
            )
    state.rank_topk_start = time.perf_counter()
    state.rank_selection = _select_phase4_frontier_rank_selection(
        feature_influences=state.feature_influences,
        visited=state.visited,
        frontier_size=state.max_frontier_size,
        ranker_mode=state.phase4_ranker_config.effective_mode,
    )
    state.pending_candidates = state.rank_selection.selected_frontier
    state.unvisited_feature_rank: torch.Tensor | None = None
    if (
        state.phase4_scheduler_config.requested_mode == "planner_v2"
        or state.phase4_debug_summary_enabled
    ):
        state.unvisited_feature_rank = _rank_phase4_unvisited_features_argsort(
            state.feature_influences, state.visited
        )
    state.max_feature_nodes_cap_bound = _compute_phase4_rank_selection_max_feature_nodes_cap_bound(
        candidate_count=int(state.rank_selection.candidate_count),
        actual_max_feature_nodes=int(state.actual_max_feature_nodes),
        n_visited=int(state.n_visited),
        max_frontier_size=int(state.max_frontier_size),
    )
    state.ranker_refresh_telemetry = {
        "ranker_frontier_candidate_count": int(state.rank_selection.candidate_count),
        "ranker_frontier_selected_count": int(state.rank_selection.selected_count),
        "ranker_frontier_selected_hash": state.rank_selection.selected_order_hash,
        "ranker_frontier_selected_order_hash": state.rank_selection.selected_order_hash,
        "ranker_frontier_selected_membership_hash": state.rank_selection.selected_membership_hash,
        "ranker_frontier_cutoff_score": state.rank_selection.cutoff_score,
        "ranker_frontier_cutoff_gap": state.rank_selection.cutoff_gap,
        "ranker_frontier_relative_cutoff_gap": state.rank_selection.relative_cutoff_gap,
        "ranker_frontier_near_cutoff_epsilon": state.rank_selection.near_cutoff_epsilon,
        "ranker_frontier_near_cutoff_count": int(state.rank_selection.near_cutoff_count),
        "ranker_frontier_max_feature_nodes_cap_bound": bool(state.max_feature_nodes_cap_bound),
        "ranker_frontier_tie_count_at_cutoff": int(state.rank_selection.tie_count_at_cutoff),
        "ranker_frontier_tie_at_cutoff": bool(state.rank_selection.tie_at_cutoff),
        "ranker_frontier_tie_behavior": state.rank_selection.tie_behavior,
    }
    if (
        (
            state.cross_cluster_debug_enabled
            or state.phase4_scheduler_config.telemetry_detail == "debug"
        )
        and state.rank_selection.cutoff_score is not None
        and (state.rank_selection.cutoff_score > 0)
    ):
        state.unvisited_scores_for_cutoff = (
            state.feature_influences[
                _rank_phase4_unvisited_features_argsort(state.feature_influences, state.visited)
            ]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        state.cutoff_score_for_profile = float(state.rank_selection.cutoff_score)
        state.below_cutoff_scores = state.unvisited_scores_for_cutoff[
            int(state.rank_selection.selected_count) :
        ]
        state.near_cutoff_counts = {
            str(state.eps): int(
                (
                    state.below_cutoff_scores
                    >= state.cutoff_score_for_profile * (1.0 - float(state.eps))
                )
                .sum()
                .item()
            )
            for state.eps in (0.001, 0.01, 0.05)
        }
        state.ranker_refresh_telemetry["ranker_frontier_near_cutoff_counts"] = (
            state.near_cutoff_counts
        )
    state.candidate_scores: torch.Tensor | None = None
    state.rank_signal_stats: dict[str, object] | None = None
    state.normalization_input_stats: dict[str, object] | None = None
    if state.phase4_debug_summary_enabled:
        if state.unvisited_feature_rank is not None:
            state.candidate_scores = (
                state.feature_influences[state.unvisited_feature_rank].detach().cpu()
            )
        else:
            state.candidate_scores = state.rank_selection.selected_scores
        state.rank_signal_stats = _build_vector_stats(
            state.candidate_scores, epsilon=1e-12, top_k=8
        )
        if state.use_compact_feature_row_store:
            assert state.feature_row_store is not None
            state.normalization_input_stats = _build_phase4_normalization_stats(
                (
                    state.feature_row_store.row_abs_max[: state.st].detach().cpu(),
                    state.feature_row_store.row_l1_scaled[: state.st].detach().cpu(),
                )
            )
        else:
            state.normalization_input_stats = _build_phase4_normalization_stats(
                state.edge_matrix[: state.st, : state.logit_offset].abs().sum(dim=1).detach().cpu()
            )
    state.feature_row_store_snapshot_after = (
        state.feature_row_store.get_diagnostic_snapshot()
        if state.use_compact_feature_row_store and state.feature_row_store is not None
        else None
    )
    state.feature_row_store_read_stats = (
        state.numeric_delta(
            state.feature_row_store_snapshot_before, state.feature_row_store_snapshot_after
        )
        if state.feature_row_store_snapshot_after is not None
        else None
    )
    state.refresh_rank_topk_elapsed_ms = (time.perf_counter() - state.rank_topk_start) * 1000.0
    state.phase4_refresh_rank_topk_elapsed_ms_total += state.refresh_rank_topk_elapsed_ms
