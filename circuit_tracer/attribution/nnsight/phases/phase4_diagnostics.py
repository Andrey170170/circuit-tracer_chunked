"""Diagnostics and evidence operations for NNSight Phase 4."""

from __future__ import annotations
from typing import cast
import torch
from circuit_tracer.attribution.nnsight.phase4_policy import (
    _PHASE4_REFRESH_MEMORY_ATTR_KEYS,
    _reorder_pending_for_phase4_locality,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_phase4_deterministic_shadow_pending,
    _build_vector_stats,
    _compare_phase4_frontiers,
    _record_phase4_refresh_debug,
)
from circuit_tracer.attribution.nnsight.replay import _hash_index_tensor
from circuit_tracer.attribution.nnsight.telemetry import (
    _record_cross_cluster_batch_event,
    _safe_float,
    _safe_int,
)
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_influences,
)
from circuit_tracer.observability.events import (
    DiagnosticSnapshot,
    MemoryDelta,
    MemorySnapshot,
    NumericDelta,
    TraceEvent,
)


def _bind_phase4_contract(state):
    """Bind the public phase boundary to the operation run."""
    state.logger = state.inputs.logger
    state.model = state.inputs.model
    state.ctx = state.inputs.ctx
    state.targets = state.inputs.targets
    state.edge_matrix = state.inputs.edge_matrix
    state.feat_ids = state.inputs.feat_ids
    state.feat_layers = state.inputs.feat_layers
    state.feat_pos = state.inputs.feat_pos
    state.feature_row_store = state.inputs.feature_row_store
    state.nonfeature_row_store = state.inputs.nonfeature_row_store
    state.row_to_node_index = state.inputs.row_to_node_index
    state.telemetry_observer = state.inputs.telemetry_observer

    def diagnostic_snapshot(source: object) -> dict[str, object] | None:
        return cast(
            dict[str, object] | None, state.telemetry_observer.observe(DiagnosticSnapshot(source))
        )

    state.diagnostic_snapshot = diagnostic_snapshot

    def memory_snapshot() -> dict[str, object]:
        return cast(
            dict[str, object], state.telemetry_observer.observe(MemorySnapshot(state.model.device))
        )

    state.memory_snapshot = memory_snapshot

    def memory_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
        return cast(
            dict[str, object],
            state.telemetry_observer.observe(
                MemoryDelta(before, after, _PHASE4_REFRESH_MEMORY_ATTR_KEYS)
            ),
        )

    state.memory_delta = memory_delta

    def numeric_delta(
        before: dict[str, object] | None, after: dict[str, object] | None
    ) -> dict[str, int | float]:
        return cast(
            dict[str, int | float], state.telemetry_observer.observe(NumericDelta(before, after))
        )

    state.numeric_delta = numeric_delta
    state.cross_cluster_debug_summary = state.inputs.cross_cluster_debug_summary
    state.cross_cluster_debug_checkpoints = state.inputs.cross_cluster_debug_checkpoints
    state.cross_cluster_debug_batches = state.inputs.cross_cluster_debug_batches
    state.anomaly_debug_result = state.inputs.anomaly_debug_result
    state.phase4_frontier_buffer_metadata = state.inputs.phase4_frontier_buffer_metadata
    state.phase4_execution_metadata = state.inputs.phase4_execution_metadata
    state.rows_cpu_staging = state.inputs.rows_cpu_staging
    state.actual_max_feature_nodes = state.config.actual_max_feature_nodes
    state.total_active_feats = state.config.total_active_feats
    state.eligible_feature_indices = state.config.eligible_feature_indices
    state.eligible_feature_count = (
        state.total_active_feats
        if state.eligible_feature_indices is None
        else int(state.eligible_feature_indices.numel())
    )
    state.n_logits = state.config.n_logits
    state.logit_offset = state.config.logit_offset
    state.effective_feature_batch_size = state.config.effective_feature_batch_size
    state.max_phase4_feature_batch_size = state.config.max_phase4_feature_batch_size
    state.update_interval = state.config.update_interval
    state.row_store_capacity_feature_nodes = state.config.row_store_capacity_feature_nodes
    state.exact_trace_internal_dtype_resolved = state.config.exact_trace_internal_dtype_resolved
    state.influence_compute_dtype = state.config.influence_compute_dtype
    state.shadow_debug_compute_dtype = state.config.shadow_debug_compute_dtype
    state.exact_chunked_decoder = state.config.exact_chunked_decoder
    state.use_compact_feature_row_store = state.config.use_compact_feature_row_store
    state.planner_enabled = state.config.planner_enabled
    state.planner_status = state.config.planner_status
    state.planner_skip_reason = state.config.planner_skip_reason
    state.phase4_debug_summary_enabled = state.config.phase4_debug_summary_enabled
    state.cross_cluster_debug_enabled = state.config.cross_cluster_debug_enabled
    state.phase4_frontier_buffer_relative_epsilon = (
        state.config.phase4_frontier_buffer_relative_epsilon
    )
    state.phase4_frontier_buffer_max_extra_per_refresh = (
        state.config.phase4_frontier_buffer_max_extra_per_refresh
    )
    state.phase4_frontier_buffer_max_extra_total = (
        state.config.phase4_frontier_buffer_max_extra_total
    )
    state.phase4_refresh_prepared_chunk_cache_bytes_effective = (
        state.config.phase4_refresh_prepared_chunk_cache_bytes_effective
    )
    state.phase4_refresh_active_row_accumulation_effective = (
        state.config.phase4_refresh_active_row_accumulation_effective
    )
    state.phase4_scheduler_config = state.config.phase4_scheduler_config
    state.phase4_refresh_optimization_config = state.config.phase4_refresh_optimization_config
    state.phase4_refresh_policy_config = state.config.phase4_refresh_policy_config
    state.phase4_ranker_config = state.config.phase4_ranker_config
    state.phase4_row_executor_config = state.config.phase4_row_executor_config
    state.phase4_row_reduction_config = state.config.phase4_row_reduction_config
    state.row_store_cache_control_config = state.config.row_store_cache_control_config
    state.exact_encoder_residency_config = state.config.exact_encoder_residency_config
    state.profile = state.config.profile
    state.profile_log_interval = state.config.profile_log_interval
    state.verbose = state.config.verbose


def record_refresh_trace(state):
    """Emit refresh telemetry after planning completes."""
    state.telemetry_observer.observe(
        TraceEvent(
            scope="batch",
            name="phase4.refresh",
            phase="phase4",
            batch_index=state.phase4_refresh_count + 1,
            elapsed_ms=state.refresh_elapsed_ms,
            attrs={
                "refresh_index": state.refresh_index,
                "resource_sampled": state.refresh_resource_sampled,
                "stored_rows": int(state.st),
                "visited_features": int(state.n_visited),
                "frontier_candidate_count": int(state.rank_selection.candidate_count),
                "queue_size": int(state.queue_size),
                "phase4_frontier_buffer_extra_count": int(
                    0
                    if state.phase4_frontier_buffer_event is None
                    else state.phase4_frontier_buffer_event.get("extra_feature_count", 0)
                ),
                "phase4_frontier_buffer_extra_used_total": int(
                    state.phase4_frontier_buffer_extra_used_total
                ),
                "phase4_frontier_buffer_expanded_frontier_size": int(state.max_frontier_size),
                "pending_count": int(state.pending.numel()),
                "pending_hash": _hash_index_tensor(state.pending)
                if state.pending.numel() > 0
                else None,
                **state.phase4_execution_metadata,
                **state.ranker_refresh_telemetry,
                **state.planner_v2_refresh_telemetry,
                **state.phase4_plan_telemetry,
                **state.refresh_substage_telemetry,
                "rank_nonzero_count": int(state.rank_signal_stats["nonzero_count"])
                if state.rank_signal_stats is not None
                else None,
                "rank_effective_nonzero_count": int(
                    state.rank_signal_stats["effective_nonzero_count"]
                )
                if state.rank_signal_stats is not None
                else None,
                "rank_max": _safe_float(state.rank_signal_stats.get("max"))
                if state.rank_signal_stats is not None
                else None,
                "rank_abs_sum": _safe_float(state.rank_signal_stats.get("abs_sum"))
                if state.rank_signal_stats is not None
                else None,
                "rank_all_zero": bool(state.rank_signal_stats["all_zero"])
                if state.rank_signal_stats is not None
                else None,
                "rank_effectively_all_zero": bool(state.rank_signal_stats["effectively_all_zero"])
                if state.rank_signal_stats is not None
                else None,
                "normalization_clamped_row_count": int(
                    state.normalization_input_stats["clamped_row_count"]
                )
                if state.normalization_input_stats is not None
                else None,
                "normalization_clamped_row_fraction": _safe_float(
                    state.normalization_input_stats.get("clamped_row_fraction")
                )
                if state.normalization_input_stats is not None
                else None,
                "feature_row_store_read_calls": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_call_count")
                ),
                "feature_row_store_read_rows": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_row_count")
                ),
                "feature_row_store_read_bytes": int(
                    float((state.feature_row_store_read_stats or {}).get("read_row_count") or 0)
                    * int(state.total_active_feats)
                    * torch.empty(
                        (), dtype=state.exact_trace_internal_dtype_resolved
                    ).element_size()
                )
                if state.use_compact_feature_row_store
                else None,
                "feature_row_store_read_cache_hits": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_cache_hit_count")
                ),
                "feature_row_store_read_cache_misses": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_cache_miss_count")
                ),
                "feature_row_store_read_cache_store_success": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_cache_store_success_count")
                ),
                "feature_row_store_read_cache_store_skip_disabled": _safe_float(
                    (state.feature_row_store_read_stats or {}).get(
                        "read_cache_store_skip_disabled_count"
                    )
                ),
                "feature_row_store_read_cache_store_skip_too_large": _safe_float(
                    (state.feature_row_store_read_stats or {}).get(
                        "read_cache_store_skip_too_large_count"
                    )
                ),
                "streaming_chunk_cache_requests": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_request_count")
                ),
                "streaming_chunk_cache_enabled": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_cache_enabled")
                ),
                "streaming_chunk_cache_max_bytes": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_cache_max_bytes")
                ),
                "streaming_chunk_cache_hits": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_cache_hit_count")
                ),
                "streaming_chunk_cache_misses": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_cache_miss_count")
                ),
                "streaming_row_reader_calls": _safe_int(
                    (state.streaming_chunk_reuse_stats or {}).get("row_reader_call_count")
                ),
                "streaming_row_reader_rows": _safe_int(
                    (state.streaming_chunk_reuse_stats or {}).get("row_reader_row_count")
                ),
                "streaming_row_reader_estimated_bytes": int(
                    float(
                        (state.streaming_chunk_reuse_stats or {}).get("row_reader_row_count") or 0
                    )
                    * int(state.total_active_feats)
                    * torch.empty(
                        (), dtype=state.exact_trace_internal_dtype_resolved
                    ).element_size()
                )
                if state.streaming_chunk_reuse_stats is not None
                else None,
                "streaming_chunk_cache_store_success": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get("chunk_cache_store_success_count")
                ),
                "streaming_chunk_cache_store_skip_disabled": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get(
                        "chunk_cache_store_skip_disabled_count"
                    )
                ),
                "streaming_chunk_cache_store_skip_too_large": _safe_float(
                    (state.streaming_chunk_reuse_stats or {}).get(
                        "chunk_cache_store_skip_too_large_count"
                    )
                ),
                "feature_row_store_materialize_calls": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("materialize_call_count")
                ),
                "feature_row_store_materialize_rows": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("materialize_row_count")
                ),
                "feature_row_store_materialize_columns": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("materialize_column_count")
                ),
                **state.memory_delta(state.refresh_memory_before, state.refresh_memory_after),
            },
            wall_clock=True,
        )
    )


def _record_cross_cluster_refresh_debug(state):
    """Record cross-cluster refresh evidence."""
    if state.cross_cluster_debug_batches is not None:
        assert state.rank_signal_stats is not None
        assert state.normalization_input_stats is not None
        _record_cross_cluster_batch_event(
            cross_cluster_debug_batches=state.cross_cluster_debug_batches,
            event_name="phase4.refresh",
            phase="phase4",
            event_index=state.phase4_refresh_count + 1,
            payload={
                "refresh_index": state.refresh_index,
                "stored_rows": int(state.st),
                "visited_features": int(state.n_visited),
                "frontier_candidate_count": int(state.rank_selection.candidate_count),
                "queue_size": int(state.queue_size),
                "pending_count": int(state.pending.numel()),
                "pending_hash": _hash_index_tensor(state.pending)
                if state.pending.numel() > 0
                else None,
                "pending_sample": [
                    int(state.value) for state.value in state.pending.detach().cpu()[:16].tolist()
                ],
                "pending_full": [
                    int(state.value) for state.value in state.pending.detach().cpu().tolist()
                ]
                if state.phase4_scheduler_config.telemetry_detail == "debug"
                else None,
                "planner_v2_candidate_window_size": int(state.planner_v2_candidate_window.numel()),
                "planner_v2_candidate_window_hash": _hash_index_tensor(
                    state.planner_v2_candidate_window
                )
                if state.planner_v2_candidate_window.numel() > 0
                else None,
                **state.phase4_execution_metadata,
                **state.ranker_refresh_telemetry,
                **state.planner_v2_refresh_telemetry,
                **state.phase4_plan_telemetry,
                **state.refresh_substage_telemetry,
                "rank_nonzero_count": int(state.rank_signal_stats["nonzero_count"]),
                "rank_effective_nonzero_count": int(
                    state.rank_signal_stats["effective_nonzero_count"]
                ),
                "rank_nonfinite_count": int(state.rank_signal_stats["nonfinite_count"]),
                "rank_max": _safe_float(state.rank_signal_stats.get("max")),
                "rank_abs_sum": _safe_float(state.rank_signal_stats.get("abs_sum")),
                "rank_effectively_all_zero": bool(state.rank_signal_stats["effectively_all_zero"]),
                "normalization_clamped_row_count": int(
                    state.normalization_input_stats["clamped_row_count"]
                ),
                "normalization_clamped_row_fraction": _safe_float(
                    state.normalization_input_stats.get("clamped_row_fraction")
                ),
                "feature_row_store_read_calls": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_call_count")
                ),
                "feature_row_store_read_rows": _safe_float(
                    (state.feature_row_store_read_stats or {}).get("read_row_count")
                ),
                "refresh_elapsed_ms": float(state.refresh_elapsed_ms),
                "resource_sampled": state.refresh_resource_sampled,
                **state.refresh_memory_after,
            },
        )


def _record_anomaly_refresh_debug(state):
    """Record anomaly and precision-shadow refresh evidence."""
    if state.anomaly_debug_result is not None:
        assert state.candidate_scores is not None
        assert state.phase4_logit_probability_stats is not None
        _record_phase4_refresh_debug(
            state.anomaly_debug_result,
            refresh_index=state.refresh_index,
            n_visited=state.n_visited,
            queue_size=state.queue_size,
            pending=state.pending,
            previous_pending=state.previous_phase4_pending,
            first_pending=state.first_phase4_pending,
            candidate_scores=state.candidate_scores,
            refresh_elapsed_ms=state.refresh_elapsed_ms,
            rank_signal_stats=state.rank_signal_stats,
            logit_probability_stats=state.phase4_logit_probability_stats,
            normalization_input_stats=state.normalization_input_stats,
            feature_row_store_read_stats=state.feature_row_store_read_stats,
            streaming_chunk_reuse_stats=state.streaming_chunk_reuse_stats,
        )
        state.debug_records = state.anomaly_debug_result.get("records", [])
        assert isinstance(state.debug_records, list) and state.debug_records
        state.current_debug_record = state.debug_records[-1]
        assert isinstance(state.current_debug_record, dict)
        assert state.unvisited_feature_rank is not None
        state.deterministic_pending = _build_phase4_deterministic_shadow_pending(
            state.unvisited_feature_rank,
            state.feature_influences.detach().cpu(),
            queue_size=state.queue_size,
            feat_layers=state.feat_layers,
            feat_positions=state.feat_pos,
            feat_ids=state.feat_ids,
            exact_chunked_decoder=state.exact_chunked_decoder,
            decoder_chunk_size=state.decoder_chunk_size,
        )
        state.current_debug_record["deterministic_shadow"] = _compare_phase4_frontiers(
            state.pending, state.deterministic_pending
        )
        if state.phase4_refresh_count == 0:
            if state.use_compact_feature_row_store:
                assert state.feature_row_store is not None
                state.shadow_row_denominator = (
                    state.feature_row_store.row_abs_max[: state.st].to(
                        dtype=state.shadow_debug_compute_dtype
                    ),
                    state.feature_row_store.row_l1_scaled[: state.st].to(
                        dtype=state.shadow_debug_compute_dtype
                    ),
                )
                state.float64_feature_influences = compute_partial_feature_influences_streaming(
                    lambda row_start, row_end: state.feature_row_store.read_feature_rows(
                        row_start, row_end, phase="phase4_anomaly_debug"
                    ),
                    state.shadow_row_denominator,
                    state.phase4_logit_probabilities.to(dtype=state.shadow_debug_compute_dtype),
                    state.row_to_node_index[: state.st],
                    n_feature_nodes=state.total_active_feats,
                    n_logits=state.n_logits,
                    device=torch.device("cpu"),
                    compute_dtype=state.shadow_debug_compute_dtype,
                )
            else:
                state.float64_influences = compute_partial_influences(
                    state.edge_matrix[: state.st].to(dtype=state.shadow_debug_compute_dtype),
                    state.phase4_logit_probabilities.to(dtype=state.shadow_debug_compute_dtype),
                    state.row_to_node_index[: state.st],
                    device=torch.device("cpu"),
                )
                state.float64_feature_influences = state.float64_influences[
                    : state.total_active_feats
                ]
            if state.exact_trace_internal_dtype_resolved == torch.float32:
                state.float32_feature_influences = state.feature_influences
            elif state.use_compact_feature_row_store:
                assert state.feature_row_store is not None
                state.float32_row_denominator = (
                    state.feature_row_store.row_abs_max[: state.st].to(dtype=torch.float32),
                    state.feature_row_store.row_l1_scaled[: state.st].to(dtype=torch.float32),
                )
                state.float32_feature_influences = compute_partial_feature_influences_streaming(
                    lambda row_start, row_end: state.feature_row_store.read_feature_rows(
                        row_start, row_end, phase="phase4_anomaly_debug"
                    ),
                    state.float32_row_denominator,
                    state.phase4_logit_probabilities.to(dtype=torch.float32),
                    state.row_to_node_index[: state.st],
                    n_feature_nodes=state.total_active_feats,
                    n_logits=state.n_logits,
                    device=torch.device("cpu"),
                )
            else:
                state.float32_influences = compute_partial_influences(
                    state.edge_matrix[: state.st].to(dtype=torch.float32),
                    state.phase4_logit_probabilities.to(dtype=torch.float32),
                    state.row_to_node_index[: state.st],
                    device=torch.device("cpu"),
                )
                state.float32_feature_influences = state.float32_influences[
                    : state.total_active_feats
                ]
            state.float32_signal_stats = _build_vector_stats(
                state.float32_feature_influences.detach().cpu(), epsilon=1e-12, top_k=8
            )
            state.float64_signal_stats = _build_vector_stats(
                state.float64_feature_influences.detach().cpu(), epsilon=1e-12, top_k=8
            )
            state.float64_feature_rank = torch.argsort(
                state.float64_feature_influences, descending=True
            ).cpu()
            state.float64_pending = state.float64_feature_rank[
                ~state.selection_visited[state.float64_feature_rank]
            ][: state.queue_size]
            state.float64_pending = _reorder_pending_for_phase4_locality(
                state.float64_pending,
                feat_layers=state.feat_layers,
                feat_positions=state.feat_pos,
                feat_ids=state.feat_ids,
                exact_chunked_decoder=state.exact_chunked_decoder,
                decoder_chunk_size=state.decoder_chunk_size,
            )
            state.current_debug_record["float64_shadow"] = _compare_phase4_frontiers(
                state.pending, state.float64_pending
            )
            state.current_debug_record["float_precision_signal_compare"] = {
                "float32": state.float32_signal_stats,
                "float64": state.float64_signal_stats,
                "float32_all_zero": bool(state.float32_signal_stats["all_zero"]),
                "float64_all_zero": bool(state.float64_signal_stats["all_zero"]),
                "float32_effectively_all_zero": bool(
                    state.float32_signal_stats["effectively_all_zero"]
                ),
                "float64_effectively_all_zero": bool(
                    state.float64_signal_stats["effectively_all_zero"]
                ),
            }
        state.current_pending_cpu = state.pending.detach().to(device="cpu", dtype=torch.int64)
        if state.first_phase4_pending is None:
            state.first_phase4_pending = state.current_pending_cpu.clone()
        state.previous_phase4_pending = state.current_pending_cpu


def record_refresh_debug(state):
    """Record refresh evidence in reference order."""
    _record_cross_cluster_refresh_debug(state)
    _record_anomaly_refresh_debug(state)
    state.phase4_refresh_count += 1


def _collect_refresh_diagnostic_series(state):
    """Collect frontier, timing, rank, and normalization series."""
    state.records = state.anomaly_debug_result.get("records", [])
    state.cutoff_margins = [
        float(state.record["cutoff"]["cutoff_margin"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("cutoff"), dict)
        and (state.record["cutoff"].get("cutoff_margin") is not None)
    ]
    state.previous_overlaps = [
        float(state.record["overlap_with_previous"])
        for state.record in state.records
        if isinstance(state.record, dict) and state.record.get("overlap_with_previous") is not None
    ]
    state.first_overlaps = [
        float(state.record["overlap_with_first"])
        for state.record in state.records
        if isinstance(state.record, dict) and state.record.get("overlap_with_first") is not None
    ]
    state.deterministic_overlaps = [
        float(state.record["deterministic_shadow"]["overlap_fraction"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("deterministic_shadow"), dict)
        and (state.record["deterministic_shadow"].get("overlap_fraction") is not None)
    ]
    state.float64_overlaps = [
        float(state.record["float64_shadow"]["overlap_fraction"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("float64_shadow"), dict)
        and (state.record["float64_shadow"].get("overlap_fraction") is not None)
    ]
    state.refresh_elapsed_values = [
        float(state.record["refresh_elapsed_ms"])
        for state.record in state.records
        if isinstance(state.record, dict) and state.record.get("refresh_elapsed_ms") is not None
    ]
    state.rank_nonzero_counts = [
        int(state.record["rank_signal_stats"]["nonzero_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("rank_signal_stats"), dict)
        and (state.record["rank_signal_stats"].get("nonzero_count") is not None)
    ]
    state.rank_effective_nonzero_counts = [
        int(state.record["rank_signal_stats"]["effective_nonzero_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("rank_signal_stats"), dict)
        and (state.record["rank_signal_stats"].get("effective_nonzero_count") is not None)
    ]
    state.rank_abs_sums = [
        float(state.record["rank_signal_stats"]["abs_sum"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("rank_signal_stats"), dict)
        and (state.record["rank_signal_stats"].get("abs_sum") is not None)
    ]
    state.rank_max_values = [
        float(state.record["rank_signal_stats"]["max"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("rank_signal_stats"), dict)
        and (state.record["rank_signal_stats"].get("max") is not None)
    ]
    state.rank_all_zero_count = sum(
        (
            1
            for state.record in state.records
            if isinstance(state.record, dict) and bool(state.record.get("rank_signal_all_zero"))
        )
    )
    state.rank_effectively_all_zero_count = sum(
        (
            1
            for state.record in state.records
            if isinstance(state.record, dict)
            and bool(state.record.get("rank_signal_effectively_all_zero"))
        )
    )
    state.normalization_clamped_counts = [
        int(state.record["normalization_input_stats"]["clamped_row_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("normalization_input_stats"), dict)
        and (state.record["normalization_input_stats"].get("clamped_row_count") is not None)
    ]
    state.normalization_clamped_fractions = [
        float(state.record["normalization_input_stats"]["clamped_row_fraction"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("normalization_input_stats"), dict)
        and (state.record["normalization_input_stats"].get("clamped_row_fraction") is not None)
    ]


def _collect_storage_diagnostic_series(state):
    """Collect row-store, cache, and replay diagnostic series."""
    state.feature_row_store_read_calls = [
        float(state.record["feature_row_store_read_stats"]["read_call_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("feature_row_store_read_stats"), dict)
        and (state.record["feature_row_store_read_stats"].get("read_call_count") is not None)
    ]
    state.feature_row_store_read_rows = [
        float(state.record["feature_row_store_read_stats"]["read_row_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("feature_row_store_read_stats"), dict)
        and (state.record["feature_row_store_read_stats"].get("read_row_count") is not None)
    ]
    state.feature_row_store_cache_store_success = [
        float(state.record["feature_row_store_read_stats"]["read_cache_store_success_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("feature_row_store_read_stats"), dict)
        and (
            state.record["feature_row_store_read_stats"].get("read_cache_store_success_count")
            is not None
        )
    ]
    state.feature_row_store_cache_skip_disabled = [
        float(state.record["feature_row_store_read_stats"]["read_cache_store_skip_disabled_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("feature_row_store_read_stats"), dict)
        and (
            state.record["feature_row_store_read_stats"].get("read_cache_store_skip_disabled_count")
            is not None
        )
    ]
    state.feature_row_store_cache_skip_too_large = [
        float(state.record["feature_row_store_read_stats"]["read_cache_store_skip_too_large_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("feature_row_store_read_stats"), dict)
        and (
            state.record["feature_row_store_read_stats"].get(
                "read_cache_store_skip_too_large_count"
            )
            is not None
        )
    ]
    state.streaming_chunk_cache_hits = [
        float(state.record["streaming_chunk_reuse_stats"]["chunk_cache_hit_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("streaming_chunk_reuse_stats"), dict)
        and (state.record["streaming_chunk_reuse_stats"].get("chunk_cache_hit_count") is not None)
    ]
    state.streaming_chunk_cache_misses = [
        float(state.record["streaming_chunk_reuse_stats"]["chunk_cache_miss_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("streaming_chunk_reuse_stats"), dict)
        and (state.record["streaming_chunk_reuse_stats"].get("chunk_cache_miss_count") is not None)
    ]
    state.streaming_chunk_cache_store_success = [
        float(state.record["streaming_chunk_reuse_stats"]["chunk_cache_store_success_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("streaming_chunk_reuse_stats"), dict)
        and (
            state.record["streaming_chunk_reuse_stats"].get("chunk_cache_store_success_count")
            is not None
        )
    ]
    state.streaming_chunk_cache_skip_disabled = [
        float(state.record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_disabled_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("streaming_chunk_reuse_stats"), dict)
        and (
            state.record["streaming_chunk_reuse_stats"].get("chunk_cache_store_skip_disabled_count")
            is not None
        )
    ]
    state.streaming_chunk_cache_skip_too_large = [
        float(state.record["streaming_chunk_reuse_stats"]["chunk_cache_store_skip_too_large_count"])
        for state.record in state.records
        if isinstance(state.record, dict)
        and isinstance(state.record.get("streaming_chunk_reuse_stats"), dict)
        and (
            state.record["streaming_chunk_reuse_stats"].get(
                "chunk_cache_store_skip_too_large_count"
            )
            is not None
        )
    ]
    state.first_float_precision = None
    if state.records and isinstance(state.records[0], dict):
        state.precision_compare = state.records[0].get("float_precision_signal_compare")
        if isinstance(state.precision_compare, dict):
            state.first_float_precision = state.precision_compare
    state.phase3_logit_row_batches = state.anomaly_debug_result.get("phase3_logit_row_batches", [])
    state.first_phase3_logit_batch = (
        state.phase3_logit_row_batches[0]
        if isinstance(state.phase3_logit_row_batches, list) and state.phase3_logit_row_batches
        else None
    )
    state.phase4_feature_row_batches = state.anomaly_debug_result.get(
        "phase4_feature_row_batches", []
    )
    state.anomaly_debug_result["refresh_count"] = int(len(state.records))
    state.anomaly_debug_result["status"] = "captured_refresh_debug"


def _collect_diagnostic_series(state):
    """Collect diagnostic series in reference order."""
    _collect_refresh_diagnostic_series(state)
    _collect_storage_diagnostic_series(state)


def _commit_diagnostic_summary(state):
    """Commit the final diagnostic summary."""
    state.anomaly_debug_result["summary"] = {
        "refresh_count": int(len(state.records)),
        "pending_size_first": int(state.records[0]["pending_size"])
        if state.records and isinstance(state.records[0], dict)
        else 0,
        "cutoff_margin_min": min(state.cutoff_margins) if state.cutoff_margins else None,
        "cutoff_margin_mean": sum(state.cutoff_margins) / len(state.cutoff_margins)
        if state.cutoff_margins
        else None,
        "overlap_with_previous_mean": sum(state.previous_overlaps) / len(state.previous_overlaps)
        if state.previous_overlaps
        else None,
        "overlap_with_first_mean": sum(state.first_overlaps) / len(state.first_overlaps)
        if state.first_overlaps
        else None,
        "deterministic_shadow_overlap_mean": sum(state.deterministic_overlaps)
        / len(state.deterministic_overlaps)
        if state.deterministic_overlaps
        else None,
        "float64_shadow_overlap_mean": sum(state.float64_overlaps) / len(state.float64_overlaps)
        if state.float64_overlaps
        else None,
        "refresh_elapsed_ms_total": sum(state.refresh_elapsed_values)
        if state.refresh_elapsed_values
        else None,
        "refresh_elapsed_ms_mean": sum(state.refresh_elapsed_values)
        / len(state.refresh_elapsed_values)
        if state.refresh_elapsed_values
        else None,
        "rank_signal_all_zero_refresh_count": int(state.rank_all_zero_count),
        "rank_signal_effectively_all_zero_refresh_count": int(
            state.rank_effectively_all_zero_count
        ),
        "rank_signal_nonzero_count_min": min(state.rank_nonzero_counts)
        if state.rank_nonzero_counts
        else None,
        "rank_signal_nonzero_count_mean": sum(state.rank_nonzero_counts)
        / len(state.rank_nonzero_counts)
        if state.rank_nonzero_counts
        else None,
        "rank_signal_effective_nonzero_count_min": min(state.rank_effective_nonzero_counts)
        if state.rank_effective_nonzero_counts
        else None,
        "rank_signal_effective_nonzero_count_mean": sum(state.rank_effective_nonzero_counts)
        / len(state.rank_effective_nonzero_counts)
        if state.rank_effective_nonzero_counts
        else None,
        "rank_signal_abs_sum_mean": sum(state.rank_abs_sums) / len(state.rank_abs_sums)
        if state.rank_abs_sums
        else None,
        "rank_signal_max_max": max(state.rank_max_values) if state.rank_max_values else None,
        "normalization_clamped_row_count_max": max(state.normalization_clamped_counts)
        if state.normalization_clamped_counts
        else None,
        "normalization_clamped_row_fraction_mean": sum(state.normalization_clamped_fractions)
        / len(state.normalization_clamped_fractions)
        if state.normalization_clamped_fractions
        else None,
        "feature_row_store_read_calls_per_refresh_mean": sum(state.feature_row_store_read_calls)
        / len(state.feature_row_store_read_calls)
        if state.feature_row_store_read_calls
        else None,
        "feature_row_store_read_rows_per_refresh_mean": sum(state.feature_row_store_read_rows)
        / len(state.feature_row_store_read_rows)
        if state.feature_row_store_read_rows
        else None,
        "feature_row_store_cache_store_success_per_refresh_mean": sum(
            state.feature_row_store_cache_store_success
        )
        / len(state.feature_row_store_cache_store_success)
        if state.feature_row_store_cache_store_success
        else None,
        "feature_row_store_cache_skip_disabled_per_refresh_mean": sum(
            state.feature_row_store_cache_skip_disabled
        )
        / len(state.feature_row_store_cache_skip_disabled)
        if state.feature_row_store_cache_skip_disabled
        else None,
        "feature_row_store_cache_skip_too_large_per_refresh_mean": sum(
            state.feature_row_store_cache_skip_too_large
        )
        / len(state.feature_row_store_cache_skip_too_large)
        if state.feature_row_store_cache_skip_too_large
        else None,
        "streaming_chunk_cache_hits_per_refresh_mean": sum(state.streaming_chunk_cache_hits)
        / len(state.streaming_chunk_cache_hits)
        if state.streaming_chunk_cache_hits
        else None,
        "streaming_chunk_cache_misses_per_refresh_mean": sum(state.streaming_chunk_cache_misses)
        / len(state.streaming_chunk_cache_misses)
        if state.streaming_chunk_cache_misses
        else None,
        "streaming_chunk_cache_store_success_per_refresh_mean": sum(
            state.streaming_chunk_cache_store_success
        )
        / len(state.streaming_chunk_cache_store_success)
        if state.streaming_chunk_cache_store_success
        else None,
        "streaming_chunk_cache_skip_disabled_per_refresh_mean": sum(
            state.streaming_chunk_cache_skip_disabled
        )
        / len(state.streaming_chunk_cache_skip_disabled)
        if state.streaming_chunk_cache_skip_disabled
        else None,
        "streaming_chunk_cache_skip_too_large_per_refresh_mean": sum(
            state.streaming_chunk_cache_skip_too_large
        )
        / len(state.streaming_chunk_cache_skip_too_large)
        if state.streaming_chunk_cache_skip_too_large
        else None,
        "phase3_logit_row_batch_count": int(
            len(state.phase3_logit_row_batches)
            if isinstance(state.phase3_logit_row_batches, list)
            else 0
        ),
        "phase4_feature_row_batch_count": int(
            len(state.phase4_feature_row_batches)
            if isinstance(state.phase4_feature_row_batches, list)
            else 0
        ),
        "first_refresh_float32_effectively_all_zero": bool(
            state.first_float_precision.get("float32_effectively_all_zero")
        )
        if isinstance(state.first_float_precision, dict)
        else None,
        "first_refresh_float64_effectively_all_zero": bool(
            state.first_float_precision.get("float64_effectively_all_zero")
        )
        if isinstance(state.first_float_precision, dict)
        else None,
        "phase3_logit_row_batch_0_abs_sum": state.first_phase3_logit_batch.get(
            "row_abs_sum_stats", {}
        ).get("abs_sum")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
        "phase3_logit_row_batch_0_max_abs": state.first_phase3_logit_batch.get(
            "row_input_stats", {}
        ).get("finite_max_abs")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
        "phase3_logit_row_batch_0_nonfinite_count": state.first_phase3_logit_batch.get(
            "row_input_stats", {}
        ).get("nonfinite_count")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
        "phase3_logit_row_batch_0_row_l1_max": state.first_phase3_logit_batch.get(
            "row_abs_sum_stats", {}
        ).get("max")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
        "phase3_logit_row_batch_0_row_l1_effectively_all_zero": state.first_phase3_logit_batch.get(
            "row_abs_sum_stats", {}
        ).get("effectively_all_zero")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
        "phase3_logit_row_batch_0_row_l1_nonfinite_count": state.first_phase3_logit_batch.get(
            "row_abs_sum_stats", {}
        ).get("nonfinite_count")
        if isinstance(state.first_phase3_logit_batch, dict)
        else None,
    }


def summarize_phase4_diagnostics(state):
    if state.anomaly_debug_result is not None:
        _collect_diagnostic_series(state)
        _commit_diagnostic_summary(state)
