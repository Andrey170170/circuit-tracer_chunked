"""Phase 3 logit attribution for NNSight attribution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase4_policy import (
    _PHASE4_REFRESH_MEMORY_ATTR_KEYS,
    _compute_phase4_refresh_queue_window_size,
    _reorder_pending_for_phase4_locality,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_feature_semantic_descriptors_payload,
    _build_matrix_abs_stats,
    _build_phase3_frontier_buffer_metadata,
    _build_phase4_cutoff_debug,
    _build_phase4_deterministic_shadow_pending,
    _build_phase4_normalization_stats,
    _build_vector_stats,
    _compare_phase4_frontiers,
    _copy_rows_to_cpu_staging,
    _dtype_to_name,
    _resolve_phase3_effective_row_state,
)
from circuit_tracer.attribution.nnsight.replay import (
    _build_phase3_gradient_bundle_payload,
    _build_phase3_row_bundle_payload,
    _build_phase3_seed_bundle_payload,
    _build_phase3_seed_influence_topk,
    _compute_row_abs_sums,
    _hash_float_tensor,
    _hash_index_tensor,
)
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_cross_cluster_runtime_snapshot,
    _build_row_transfer_telemetry,
    _build_tensor_transfer_estimate,
    _hash_json_payload,
    _record_cross_cluster_batch_event,
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_influences,
)
from circuit_tracer.observability.human_logs import (
    _log_batch_profile,
    _log_memory_boundary,
    _log_phase_metrics,
    _snapshot_diagnostics,
)
from circuit_tracer.utils.telemetry import build_memory_before_after_attrs, get_memory_snapshot


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
    telemetry_observer: Any
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
    """Run logit attribution and prepare the Phase 4 seed state."""
    logger = inputs.logger
    model = inputs.model
    ctx = inputs.ctx
    targets = inputs.targets
    activation_matrix = inputs.activation_matrix
    feat_layers = inputs.feat_layers
    feat_pos = inputs.feat_pos
    feat_ids = inputs.feat_ids
    feature_row_store = inputs.feature_row_store
    nonfeature_row_store = inputs.nonfeature_row_store
    edge_matrix = inputs.edge_matrix
    row_to_node_index = inputs.row_to_node_index
    telemetry_observer = inputs.telemetry_observer
    cross_cluster_debug_summary = inputs.cross_cluster_debug_summary
    cross_cluster_debug_checkpoints = inputs.cross_cluster_debug_checkpoints
    cross_cluster_debug_batches = inputs.cross_cluster_debug_batches
    anomaly_debug_result = inputs.anomaly_debug_result
    loaded_phase3_row_donor_bundle = inputs.loaded_phase3_row_donor_bundle
    phase3_frontier_buffer_metadata = inputs.phase3_frontier_buffer_metadata
    phase3_gradient_bundle_payload = inputs.phase3_gradient_bundle_payload
    phase3_row_bundle_payload = inputs.phase3_row_bundle_payload
    phase3_seed_bundle_payload = inputs.phase3_seed_bundle_payload
    feature_semantic_descriptors_payload = inputs.feature_semantic_descriptors_payload
    effective_logit_batch_size = config.effective_logit_batch_size
    effective_feature_batch_size = config.effective_feature_batch_size
    output_position = config.output_position
    n_layers = config.n_layers
    n_pos = config.n_pos
    n_logits = config.n_logits
    logit_offset = config.logit_offset
    total_active_feats = config.total_active_feats
    base_max_feature_nodes = config.base_max_feature_nodes
    actual_max_feature_nodes = config.actual_max_feature_nodes
    exact_trace_internal_dtype_resolved = config.exact_trace_internal_dtype_resolved
    phase3_gradient_replay_mode_resolved = config.phase3_gradient_replay_mode_resolved
    phase3_row_replay_mode_resolved = config.phase3_row_replay_mode_resolved
    capture_phase3_gradient_bundle_enabled = config.capture_phase3_gradient_bundle_enabled
    capture_phase3_row_bundle_enabled = config.capture_phase3_row_bundle_enabled
    capture_phase3_seed_bundle_enabled = config.capture_phase3_seed_bundle_enabled
    capture_feature_semantic_descriptors_enabled = (
        config.capture_feature_semantic_descriptors_enabled
    )
    phase3_frontier_buffer_relative_epsilon = config.phase3_frontier_buffer_relative_epsilon
    phase3_frontier_buffer_max_extra = config.phase3_frontier_buffer_max_extra
    update_interval = config.update_interval
    planner_compute_dtype = config.planner_compute_dtype
    influence_compute_dtype = config.influence_compute_dtype
    shadow_debug_compute_dtype = config.shadow_debug_compute_dtype
    phase4_refresh_policy_config = config.phase4_refresh_policy_config
    exact_chunked_decoder = config.exact_chunked_decoder
    use_compact_feature_row_store = config.use_compact_feature_row_store
    semantic_descriptor_top_k = config.semantic_descriptor_top_k
    semantic_descriptor_dim = config.semantic_descriptor_dim
    profile = config.profile
    profile_log_interval = config.profile_log_interval
    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase3_start = time.perf_counter()
    _log_memory_boundary(logger, "Phase 3 start", model.device)
    i = -1
    total_logit_batches = max(
        (len(targets) + effective_logit_batch_size - 1) // effective_logit_batch_size,
        1,
    )
    phase3_feature_row_batches: list[torch.Tensor] = []
    phase3_row_abs_sum_batches: list[torch.Tensor] = []
    phase3_feature_abs_sum_batches: list[torch.Tensor] = []
    phase3_error_abs_sum_batches: list[torch.Tensor] = []
    phase3_token_abs_sum_batches: list[torch.Tensor] = []
    rows_cpu_staging: torch.Tensor | None = None
    phase3_compute_batch_elapsed_ms_total = 0.0
    phase3_cpu_staging_elapsed_ms_total = 0.0
    phase3_denominator_elapsed_ms_total = 0.0
    phase3_row_store_write_elapsed_ms_total = 0.0
    phase3_gpu_to_cpu_bytes_total = 0
    phase3_cpu_to_gpu_bytes_total = 0
    phase3_copy_count = 0
    for i in range(0, len(targets), effective_logit_batch_size):
        batch = targets.logit_vectors[i : i + effective_logit_batch_size]
        ctx_before = _snapshot_diagnostics(ctx) if profile else None
        transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
        batch_start = time.perf_counter()
        batch_memory_before = get_memory_snapshot(model.device)
        if phase3_gradient_replay_mode_resolved == "donor":
            setattr(ctx, "phase3_gradient_replay_column_offset", int(i))
        phase3_inject_transfer_telemetry = _build_tensor_transfer_estimate(
            prefix="inject_values",
            source=batch,
            destination_device=model.device,
        )
        if (
            phase3_inject_transfer_telemetry["inject_values_source"] == "cpu"
            and phase3_inject_transfer_telemetry["inject_values_destination"] == "cuda"
        ):
            phase3_cpu_to_gpu_bytes_total += int(
                phase3_inject_transfer_telemetry["inject_values_transfer_bytes"]
            )
        compute_batch_start = time.perf_counter()
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full(
                (batch.shape[0],),
                output_position if output_position is not None else n_pos - 1,
            ),
            inject_values=batch,
            phase_label="phase3_logits",
        )
        phase3_compute_batch_elapsed_ms = (time.perf_counter() - compute_batch_start) * 1000.0
        phase3_compute_batch_elapsed_ms_total += phase3_compute_batch_elapsed_ms
        cpu_staging_start = time.perf_counter()
        rows_cpu, rows_cpu_staging = _copy_rows_to_cpu_staging(
            rows,
            staging_buffer=rows_cpu_staging,
        )
        phase3_cpu_staging_elapsed_ms = (time.perf_counter() - cpu_staging_start) * 1000.0
        phase3_cpu_staging_elapsed_ms_total += phase3_cpu_staging_elapsed_ms
        donor_feature_rows: torch.Tensor | None = None
        donor_row_abs_sums: torch.Tensor | None = None
        donor_feature_abs_sums: torch.Tensor | None = None
        donor_error_abs_sums: torch.Tensor | None = None
        donor_token_abs_sums: torch.Tensor | None = None
        if loaded_phase3_row_donor_bundle is not None:
            end = i + batch.shape[0]
            donor_feature_rows = cast(
                torch.Tensor,
                loaded_phase3_row_donor_bundle["phase3_feature_rows"],
            )[i:end]
            donor_row_abs_sums = cast(
                torch.Tensor,
                loaded_phase3_row_donor_bundle["row_abs_sums"],
            )[i:end]
            donor_feature_abs_sums = cast(
                torch.Tensor,
                loaded_phase3_row_donor_bundle["feature_abs_sums"],
            )[i:end]
            donor_error_abs_sums = cast(
                torch.Tensor,
                loaded_phase3_row_donor_bundle["error_abs_sums"],
            )[i:end]
            donor_token_abs_sums = cast(
                torch.Tensor,
                loaded_phase3_row_donor_bundle["token_abs_sums"],
            )[i:end]
        denominator_start = time.perf_counter()
        (
            rows_cpu,
            row_input_slice,
            feature_row_slice,
            (row_abs_max_cpu, row_l1_scaled_cpu),
            row_abs_sums_cpu,
        ) = _resolve_phase3_effective_row_state(
            rows_cpu=rows_cpu,
            row_input_column_count=int(logit_offset),
            total_active_features=int(total_active_feats),
            dtype=exact_trace_internal_dtype_resolved,
            donor_feature_rows=donor_feature_rows,
            donor_row_abs_sums=donor_row_abs_sums,
        )
        phase3_denominator_elapsed_ms = (time.perf_counter() - denominator_start) * 1000.0
        phase3_denominator_elapsed_ms_total += phase3_denominator_elapsed_ms
        phase3_row_transfer_telemetry = _build_row_transfer_telemetry(
            rows=rows,
            rows_cpu=rows_cpu,
            row_input_slice=row_input_slice,
            feature_row_slice=feature_row_slice,
        )
        if phase3_row_transfer_telemetry["row_transfer_source"] == "cuda":
            phase3_gpu_to_cpu_bytes_total += int(
                phase3_row_transfer_telemetry["row_transfer_bytes"]
            )
        if phase3_row_transfer_telemetry["row_transfer_destination"] == "cuda":
            phase3_cpu_to_gpu_bytes_total += int(
                phase3_row_transfer_telemetry["row_transfer_bytes"]
            )
        if int(phase3_row_transfer_telemetry["row_transfer_bytes"]) > 0:
            phase3_copy_count += 1
        if capture_phase3_row_bundle_enabled:
            feature_rows_cpu = feature_row_slice.contiguous()
            error_start = int(total_active_feats)
            error_end = int(total_active_feats + n_layers * n_pos)
            token_end = int(logit_offset)
            phase3_feature_row_batches.append(feature_rows_cpu)
            phase3_row_abs_sum_batches.append(row_abs_sums_cpu.contiguous())
            if (
                donor_feature_abs_sums is not None
                and donor_error_abs_sums is not None
                and donor_token_abs_sums is not None
            ):
                phase3_feature_abs_sum_batches.append(donor_feature_abs_sums.contiguous())
                phase3_error_abs_sum_batches.append(donor_error_abs_sums.contiguous())
                phase3_token_abs_sum_batches.append(donor_token_abs_sums.contiguous())
            else:
                phase3_feature_abs_sum_batches.append(
                    _compute_row_abs_sums(
                        feature_rows_cpu,
                        dtype=torch.float64,
                    ).contiguous()
                )
                phase3_error_abs_sum_batches.append(
                    _compute_row_abs_sums(
                        rows_cpu[:, error_start:error_end],
                        dtype=torch.float64,
                    ).contiguous()
                )
                phase3_token_abs_sum_batches.append(
                    _compute_row_abs_sums(
                        rows_cpu[:, error_end:token_end],
                        dtype=torch.float64,
                    ).contiguous()
                )
        if anomaly_debug_result is not None:
            logit_row_batches = anomaly_debug_result.setdefault(
                "phase3_logit_row_batches",
                [],
            )
            assert isinstance(logit_row_batches, list)
            logit_row_batches.append(
                {
                    "batch_index": int((i // effective_logit_batch_size) + 1),
                    "batch_row_count": int(batch.shape[0]),
                    "row_input_stats": _build_matrix_abs_stats(
                        row_input_slice,
                        epsilon=1e-12,
                        top_k=8,
                    ),
                    "row_abs_sum_stats": _build_phase4_normalization_stats(
                        (row_abs_max_cpu, row_l1_scaled_cpu),
                        clamp_epsilon=1e-8,
                    ),
                }
            )
        if use_compact_feature_row_store:
            assert feature_row_store is not None
            assert nonfeature_row_store is not None
            end = i + batch.shape[0]
            row_store_write_start = time.perf_counter()
            feature_row_store.append_rows(
                row_start=i,
                feature_rows=feature_row_slice,
                row_denominator_scaled_l1=(row_abs_max_cpu, row_l1_scaled_cpu),
                phase="phase3",
            )
            nonfeature_row_store.append_rows(
                row_start=i,
                feature_rows=rows_cpu[:, total_active_feats:logit_offset],
                row_denominator_scaled_l1=(row_abs_max_cpu, row_l1_scaled_cpu),
                phase="phase3",
            )
            phase3_row_store_write_elapsed_ms = (
                time.perf_counter() - row_store_write_start
            ) * 1000.0
        else:
            row_store_write_start = time.perf_counter()
            edge_matrix[i : i + batch.shape[0], :logit_offset] = rows_cpu
            phase3_row_store_write_elapsed_ms = (
                time.perf_counter() - row_store_write_start
            ) * 1000.0
        phase3_row_store_write_elapsed_ms_total += phase3_row_store_write_elapsed_ms
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
        batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        batch_memory_after = get_memory_snapshot(model.device)
        telemetry_observer.batch(
            name="phase3.logit_batch",
            phase="phase3",
            batch_index=(i // effective_logit_batch_size) + 1,
            elapsed_ms=batch_elapsed_ms,
            attrs={
                "batch_rows": int(batch.shape[0]),
                "batch_start_index": int(i),
                "total_logit_batches": int(total_logit_batches),
                "compute_batch_elapsed_ms": float(phase3_compute_batch_elapsed_ms),
                "cpu_staging_elapsed_ms": float(phase3_cpu_staging_elapsed_ms),
                "denominator_elapsed_ms": float(phase3_denominator_elapsed_ms),
                "row_store_write_elapsed_ms": float(phase3_row_store_write_elapsed_ms),
                **phase3_inject_transfer_telemetry,
                **phase3_row_transfer_telemetry,
                **build_memory_before_after_attrs(
                    before=batch_memory_before,
                    after=batch_memory_after,
                    keys=_PHASE4_REFRESH_MEMORY_ATTR_KEYS,
                ),
            },
            wall_clock=True,
        )
        if cross_cluster_debug_batches is not None:
            row_input_stats = _build_matrix_abs_stats(
                row_input_slice,
                epsilon=1e-12,
                top_k=0,
            )
            row_abs_sum_stats = _build_phase4_normalization_stats(
                (row_abs_max_cpu, row_l1_scaled_cpu),
                clamp_epsilon=1e-8,
            )
            _record_cross_cluster_batch_event(
                cross_cluster_debug_batches=cross_cluster_debug_batches,
                event_name="phase3.logit_batch",
                phase="phase3",
                event_index=(i // effective_logit_batch_size) + 1,
                payload={
                    "batch_rows": int(batch.shape[0]),
                    "batch_start_index": int(i),
                    "total_logit_batches": int(total_logit_batches),
                    "row_input_nonfinite_count": int(row_input_stats["nonfinite_count"]),
                    "row_input_finite_max_abs": _safe_float(
                        row_input_stats.get("finite_max_abs")
                    ),
                    "row_l1_abs_sum": _safe_float(row_abs_sum_stats.get("abs_sum")),
                    "row_l1_max": _safe_float(row_abs_sum_stats.get("max")),
                    "row_l1_nonfinite_count": int(row_abs_sum_stats["nonfinite_count"]),
                    "row_l1_effectively_all_zero": bool(
                        row_abs_sum_stats["effectively_all_zero"]
                    ),
                    **get_memory_snapshot(model.device),
                },
            )
        if profile and ((i // effective_logit_batch_size) + 1) % profile_log_interval == 0:
            _log_batch_profile(
                logger,
                "Phase 3",
                (i // effective_logit_batch_size) + 1,
                total_logit_batches,
                batch_elapsed_ms / 1000.0,
                ctx_before,
                _snapshot_diagnostics(ctx),
                transcoder_before,
                _snapshot_diagnostics(model.transcoders),
            )

    _log_phase_metrics(
        logger,
        f"{i + 1} logit attribution(s)",
        phase3_start,
        model.device,
    )
    phase3_elapsed_ms = (time.perf_counter() - phase3_start) * 1000.0
    telemetry_observer.phase(
        name="phase3.logit_attribution",
        phase="phase3",
        elapsed_ms=phase3_elapsed_ms,
        attrs={
            "logit_count": int(len(targets)),
            "batches": int(total_logit_batches),
            "phase3_compute_batch_elapsed_ms_total": float(
                phase3_compute_batch_elapsed_ms_total
            ),
            "phase3_cpu_staging_elapsed_ms_total": float(phase3_cpu_staging_elapsed_ms_total),
            "phase3_denominator_elapsed_ms_total": float(phase3_denominator_elapsed_ms_total),
            "phase3_row_store_write_elapsed_ms_total": float(
                phase3_row_store_write_elapsed_ms_total
            ),
            "phase3_gpu_to_cpu_bytes_total": int(phase3_gpu_to_cpu_bytes_total),
            "phase3_cpu_to_gpu_bytes_total": int(phase3_cpu_to_gpu_bytes_total),
            "phase3_copy_count": int(phase3_copy_count),
        },
        wall_clock=True,
    )
    reset_decoder_cache = getattr(ctx, "reset_decoder_cache", None)
    if callable(reset_decoder_cache):
        reset_decoder_cache()

    phase3_target_token_ids = torch.tensor(
        [int(target.vocab_idx) for target in targets.logit_targets],
        dtype=torch.int64,
    )
    if capture_phase3_gradient_bundle_enabled:
        gradient_captures = getattr(ctx, "phase3_gradient_captures", [])
        phase3_gradient_bundle_payload = _build_phase3_gradient_bundle_payload(
            gradient_captures=(
                gradient_captures if isinstance(gradient_captures, list) else []
            ),
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            target_token_ids=phase3_target_token_ids,
            target_probabilities=targets.logit_probabilities,
            status=(
                "captured_replayed_effective_state"
                if phase3_gradient_replay_mode_resolved != "disabled"
                else "captured"
            ),
        )
    if capture_phase3_row_bundle_enabled:
        phase3_row_bundle_payload = _build_phase3_row_bundle_payload(
            feature_rows=phase3_feature_row_batches,
            row_abs_sums=phase3_row_abs_sum_batches,
            feature_abs_sums=phase3_feature_abs_sum_batches,
            error_abs_sums=phase3_error_abs_sum_batches,
            token_abs_sums=phase3_token_abs_sum_batches,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            target_token_ids=phase3_target_token_ids,
            target_probabilities=targets.logit_probabilities,
            total_active_features=int(total_active_feats),
            error_column_count=int(n_layers * n_pos),
            token_column_count=int(n_pos),
            status=(
                "captured_replayed_effective_state"
                if (
                    phase3_gradient_replay_mode_resolved != "disabled"
                    or phase3_row_replay_mode_resolved != "disabled"
                )
                else "captured"
            ),
        )

    if (
        cross_cluster_debug_summary is not None
        or capture_phase3_seed_bundle_enabled
        or capture_feature_semantic_descriptors_enabled
        or phase3_frontier_buffer_metadata["enabled"]
    ):
        phase3_runtime_summary: dict[str, object] = {}
        phase3_runtime_stream: dict[str, object] = {}
        if cross_cluster_debug_summary is not None:
            phase3_runtime_summary, phase3_runtime_stream = (
                _build_cross_cluster_runtime_snapshot(
                    device=model.device,
                    ctx=ctx,
                    transcoder=model.transcoders,
                )
            )
        pre_phase4_st = int(n_logits)
        phase3_seed_summary: dict[str, object] = {
            "stored_row_count_before_phase4": pre_phase4_st,
            "actual_max_feature_nodes": int(actual_max_feature_nodes),
            "total_active_features": int(total_active_feats),
            "update_interval": int(update_interval),
            "feature_batch_size": int(effective_feature_batch_size),
            "planner_compute_dtype": _dtype_to_name(planner_compute_dtype),
            "influence_compute_dtype": _dtype_to_name(influence_compute_dtype),
            **phase3_runtime_summary,
        }
        if actual_max_feature_nodes < total_active_feats:
            normalization_input_stats: dict[str, object] | None = None
            row_store_snapshot: dict[str, float | int | None] | None = None
            if use_compact_feature_row_store:
                assert feature_row_store is not None
                row_denominator_prefix = (
                    feature_row_store.row_abs_max[:pre_phase4_st],
                    feature_row_store.row_l1_scaled[:pre_phase4_st],
                )
                seed_feature_influences = compute_partial_feature_influences_streaming(
                    lambda row_start, row_end: feature_row_store.read_feature_rows(
                        row_start,
                        row_end,
                        phase="phase3_seed_ranking",
                    ),
                    row_denominator_prefix,
                    targets.logit_probabilities,
                    row_to_node_index[:pre_phase4_st],
                    n_feature_nodes=total_active_feats,
                    n_logits=n_logits,
                    device=feature_row_store.row_abs_max.device,
                    compute_dtype=planner_compute_dtype,
                )
                if cross_cluster_debug_summary is not None:
                    normalization_input_stats = _build_phase4_normalization_stats(
                        (
                            row_denominator_prefix[0].detach().cpu(),
                            row_denominator_prefix[1].detach().cpu(),
                        ),
                    )
                    row_store_snapshot = feature_row_store.get_diagnostic_snapshot()
            else:
                planner_influences = compute_partial_influences(
                    edge_matrix[:pre_phase4_st].to(dtype=planner_compute_dtype),
                    targets.logit_probabilities.to(dtype=planner_compute_dtype),
                    row_to_node_index[:pre_phase4_st],
                    device=torch.device("cpu"),
                )
                seed_feature_influences = planner_influences[:total_active_feats]
                if cross_cluster_debug_summary is not None:
                    normalization_input_stats = _build_phase4_normalization_stats(
                        edge_matrix[:pre_phase4_st, :logit_offset]
                        .abs()
                        .sum(dim=1)
                        .detach()
                        .cpu(),
                    )

            unvisited_feature_rank = torch.argsort(
                seed_feature_influences,
                descending=True,
            ).cpu()
            candidate_scores = seed_feature_influences[unvisited_feature_rank].detach().cpu()
            phase3_frontier_buffer_metadata = _build_phase3_frontier_buffer_metadata(
                seed_feature_influences=seed_feature_influences,
                base_max_feature_nodes=int(base_max_feature_nodes),
                total_active_features=int(total_active_feats),
                relative_epsilon=phase3_frontier_buffer_relative_epsilon,
                max_extra=int(phase3_frontier_buffer_max_extra),
            )
            actual_max_feature_nodes = int(
                phase3_frontier_buffer_metadata["actual_max_feature_nodes"]
            )
            phase3_seed_summary["phase3_frontier_buffer_metadata"] = (
                phase3_frontier_buffer_metadata
            )
            queue_size = min(
                _compute_phase4_refresh_queue_window_size(
                    update_interval=update_interval,
                    phase4_feature_batch_size=effective_feature_batch_size,
                    queue_multiplier=phase4_refresh_policy_config.effective_queue_multiplier,
                ),
                int(actual_max_feature_nodes),
            )
            pre_locality_pending = unvisited_feature_rank[:queue_size]
            post_locality_pending = _reorder_pending_for_phase4_locality(
                pre_locality_pending,
                feat_layers=feat_layers,
                feat_positions=feat_pos,
                feat_ids=feat_ids,
                exact_chunked_decoder=exact_chunked_decoder,
                decoder_chunk_size=getattr(model.transcoders, "decoder_chunk_size", None),
            )
            phase3_seed_summary.update(
                {
                    "status": "captured",
                    "queue_size": int(queue_size),
                }
            )

            if capture_phase3_seed_bundle_enabled:
                phase3_seed_bundle_payload = _build_phase3_seed_bundle_payload(
                    active_features=activation_matrix.indices().T,
                    activation_values=activation_matrix.values(),
                    seed_feature_influences=seed_feature_influences,
                    frontier_pre_locality=pre_locality_pending,
                    frontier_post_locality=post_locality_pending,
                    queue_size=queue_size,
                    actual_max_feature_nodes=int(actual_max_feature_nodes),
                    total_active_features=int(total_active_feats),
                    status="captured",
                    planner_compute_dtype=planner_compute_dtype,
                    influence_compute_dtype=influence_compute_dtype,
                )
            if capture_feature_semantic_descriptors_enabled:
                feature_semantic_descriptors_payload = (
                    _build_feature_semantic_descriptors_payload(
                        active_features=activation_matrix.indices().T,
                        activation_values=activation_matrix.values(),
                        seed_feature_influences=seed_feature_influences,
                        frontier_pre_locality=pre_locality_pending,
                        frontier_post_locality=post_locality_pending,
                        total_active_features=int(total_active_feats),
                        status="captured",
                        semantic_descriptor_top_k=semantic_descriptor_top_k,
                        semantic_descriptor_dim=semantic_descriptor_dim,
                    )
                )

            if cross_cluster_debug_summary is not None:
                deterministic_pending = _build_phase4_deterministic_shadow_pending(
                    unvisited_feature_rank,
                    seed_feature_influences.detach().cpu(),
                    queue_size=queue_size,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    exact_chunked_decoder=exact_chunked_decoder,
                    decoder_chunk_size=getattr(model.transcoders, "decoder_chunk_size", None),
                )
                seed_cutoff_debug = _build_phase4_cutoff_debug(
                    candidate_scores,
                    queue_size=queue_size,
                )
                seed_influence_topk = _build_phase3_seed_influence_topk(
                    ranked_feature_indices=unvisited_feature_rank,
                    seed_feature_influences=seed_feature_influences,
                    feat_layers=feat_layers,
                    feat_positions=feat_pos,
                    feat_ids=feat_ids,
                    top_k=8,
                )

                phase3_seed_summary.update(
                    {
                        "feature_influence_stats": _build_vector_stats(
                            seed_feature_influences.detach().cpu(),
                            epsilon=1e-12,
                            top_k=8,
                        ),
                        "feature_influence_hash": _hash_float_tensor(
                            seed_feature_influences.detach().cpu(),
                            dtype=torch.float64,
                        ),
                        "frontier_pre_locality_hash": _hash_index_tensor(pre_locality_pending),
                        "frontier_post_locality_hash": _hash_index_tensor(
                            post_locality_pending
                        ),
                        "frontier_pre_locality_sample": [
                            int(v) for v in pre_locality_pending[:16].tolist()
                        ],
                        "frontier_post_locality_sample": [
                            int(v) for v in post_locality_pending[:16].tolist()
                        ],
                        "seed_influence_topk": seed_influence_topk,
                        "seed_influence_topk_hash": _hash_json_payload(seed_influence_topk),
                        "seed_cutoff": seed_cutoff_debug,
                        "deterministic_shadow": _compare_phase4_frontiers(
                            post_locality_pending,
                            deterministic_pending,
                        ),
                        "normalization_input_stats": normalization_input_stats,
                        "feature_row_store_summary": row_store_snapshot,
                    }
                )

                if shadow_debug_compute_dtype != planner_compute_dtype:
                    if use_compact_feature_row_store:
                        assert feature_row_store is not None
                        shadow_feature_influences = (
                            compute_partial_feature_influences_streaming(
                                lambda row_start, row_end: feature_row_store.read_feature_rows(
                                    row_start,
                                    row_end,
                                    phase="phase3_seed_ranking_shadow",
                                ),
                                (
                                    feature_row_store.row_abs_max[:pre_phase4_st],
                                    feature_row_store.row_l1_scaled[:pre_phase4_st],
                                ),
                                targets.logit_probabilities,
                                row_to_node_index[:pre_phase4_st],
                                n_feature_nodes=total_active_feats,
                                n_logits=n_logits,
                                device=torch.device("cpu"),
                                compute_dtype=shadow_debug_compute_dtype,
                            )
                        )
                    else:
                        shadow_influences = compute_partial_influences(
                            edge_matrix[:pre_phase4_st].to(dtype=shadow_debug_compute_dtype),
                            targets.logit_probabilities.to(dtype=shadow_debug_compute_dtype),
                            row_to_node_index[:pre_phase4_st],
                            device=torch.device("cpu"),
                        )
                        shadow_feature_influences = shadow_influences[:total_active_feats]
                    shadow_rank = torch.argsort(
                        shadow_feature_influences,
                        descending=True,
                    ).cpu()
                    shadow_pending = _reorder_pending_for_phase4_locality(
                        shadow_rank[:queue_size],
                        feat_layers=feat_layers,
                        feat_positions=feat_pos,
                        feat_ids=feat_ids,
                        exact_chunked_decoder=exact_chunked_decoder,
                        decoder_chunk_size=getattr(
                            model.transcoders, "decoder_chunk_size", None
                        ),
                    )
                    phase3_seed_summary["shadow_debug"] = _compare_phase4_frontiers(
                        post_locality_pending,
                        shadow_pending,
                    )
        else:
            phase3_frontier_buffer_metadata["status"] = "skipped_all_features_included"
            phase3_frontier_buffer_metadata["fallback_reason"] = "all_features_included"
            phase3_seed_summary.update(
                {
                    "status": "skipped_all_features_included",
                    "queue_size": int(actual_max_feature_nodes),
                }
            )
            if capture_phase3_seed_bundle_enabled:
                phase3_seed_bundle_payload = _build_phase3_seed_bundle_payload(
                    active_features=activation_matrix.indices().T,
                    activation_values=activation_matrix.values(),
                    seed_feature_influences=torch.empty(
                        0,
                        dtype=planner_compute_dtype,
                    ),
                    frontier_pre_locality=torch.empty(0, dtype=torch.long),
                    frontier_post_locality=torch.empty(0, dtype=torch.long),
                    queue_size=int(actual_max_feature_nodes),
                    actual_max_feature_nodes=int(actual_max_feature_nodes),
                    total_active_features=int(total_active_feats),
                    status="skipped_all_features_included",
                    planner_compute_dtype=planner_compute_dtype,
                    influence_compute_dtype=influence_compute_dtype,
                )
            if capture_feature_semantic_descriptors_enabled:
                feature_semantic_descriptors_payload = (
                    _build_feature_semantic_descriptors_payload(
                        active_features=activation_matrix.indices().T,
                        activation_values=activation_matrix.values(),
                        seed_feature_influences=torch.empty(0, dtype=planner_compute_dtype),
                        frontier_pre_locality=torch.empty(0, dtype=torch.long),
                        frontier_post_locality=torch.empty(0, dtype=torch.long),
                        total_active_features=int(total_active_feats),
                        status="skipped_all_features_included",
                        semantic_descriptor_top_k=semantic_descriptor_top_k,
                        semantic_descriptor_dim=semantic_descriptor_dim,
                    )
                )
        if cross_cluster_debug_summary is not None:
            deterministic_shadow = phase3_seed_summary.get("deterministic_shadow")
            shadow_debug = phase3_seed_summary.get("shadow_debug")
            normalization_input_stats = phase3_seed_summary.get("normalization_input_stats")
            feature_influence_stats = phase3_seed_summary.get("feature_influence_stats")
            phase3_stream_checkpoint = {
                "status": phase3_seed_summary.get("status"),
                "stored_row_count_before_phase4": int(pre_phase4_st),
                "actual_max_feature_nodes": int(actual_max_feature_nodes),
                "total_active_features": int(total_active_feats),
                "update_interval": int(update_interval),
                "feature_batch_size": int(effective_feature_batch_size),
                "queue_size": phase3_seed_summary.get("queue_size"),
                "feature_influence_hash": phase3_seed_summary.get("feature_influence_hash"),
                "frontier_pre_locality_hash": phase3_seed_summary.get(
                    "frontier_pre_locality_hash"
                ),
                "frontier_post_locality_hash": phase3_seed_summary.get(
                    "frontier_post_locality_hash"
                ),
                "deterministic_shadow_overlap_fraction": (
                    _safe_float(deterministic_shadow.get("overlap_fraction"))
                    if isinstance(deterministic_shadow, dict)
                    else None
                ),
                "deterministic_shadow_jaccard": (
                    _safe_float(deterministic_shadow.get("jaccard_similarity"))
                    if isinstance(deterministic_shadow, dict)
                    else None
                ),
                "deterministic_shadow_prefix_match_count": (
                    int(deterministic_shadow.get("prefix_match_count", 0))
                    if isinstance(deterministic_shadow, dict)
                    else None
                ),
                "shadow_debug_overlap_fraction": (
                    _safe_float(shadow_debug.get("overlap_fraction"))
                    if isinstance(shadow_debug, dict)
                    else None
                ),
                "seed_influence_topk_hash": phase3_seed_summary.get("seed_influence_topk_hash"),
                "seed_cutoff_margin": (
                    _safe_float(phase3_seed_summary.get("seed_cutoff", {}).get("cutoff_margin"))
                    if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                    else None
                ),
                "seed_cutoff_near_tie_count": (
                    int(phase3_seed_summary.get("seed_cutoff", {}).get("near_cutoff_count", 0))
                    if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                    else None
                ),
                "seed_cutoff_exact_tie_count": (
                    int(phase3_seed_summary.get("seed_cutoff", {}).get("exact_cutoff_count", 0))
                    if isinstance(phase3_seed_summary.get("seed_cutoff"), dict)
                    else None
                ),
                "feature_influence_nonfinite_count": (
                    int(feature_influence_stats.get("nonfinite_count", 0))
                    if isinstance(feature_influence_stats, dict)
                    else None
                ),
                "feature_influence_abs_sum": (
                    _safe_float(feature_influence_stats.get("abs_sum"))
                    if isinstance(feature_influence_stats, dict)
                    else None
                ),
                "normalization_clamped_row_count": (
                    int(normalization_input_stats.get("clamped_row_count", 0))
                    if isinstance(normalization_input_stats, dict)
                    else None
                ),
                "normalization_clamped_row_fraction": (
                    _safe_float(normalization_input_stats.get("clamped_row_fraction"))
                    if isinstance(normalization_input_stats, dict)
                    else None
                ),
                **phase3_runtime_stream,
            }
            _record_cross_cluster_checkpoint(
                cross_cluster_debug_summary=cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
                checkpoint_name="phase3_seed_ranking_pre_phase4",
                phase="phase3",
                summary_payload=phase3_seed_summary,
                stream_payload=phase3_stream_checkpoint,
            )


    return Phase3Result(
        stored_row_count=int(n_logits),
        row_to_node_index=row_to_node_index,
        rows_cpu_staging=rows_cpu_staging,
        actual_max_feature_nodes=int(actual_max_feature_nodes),
        phase3_frontier_buffer_metadata=phase3_frontier_buffer_metadata,
        phase3_gradient_bundle_payload=phase3_gradient_bundle_payload,
        phase3_row_bundle_payload=phase3_row_bundle_payload,
        phase3_seed_bundle_payload=phase3_seed_bundle_payload,
        feature_semantic_descriptors_payload=feature_semantic_descriptors_payload,
        anomaly_debug_result=anomaly_debug_result,
        compute_batch_elapsed_ms_total=phase3_compute_batch_elapsed_ms_total,
        cpu_staging_elapsed_ms_total=phase3_cpu_staging_elapsed_ms_total,
        denominator_elapsed_ms_total=phase3_denominator_elapsed_ms_total,
        row_store_write_elapsed_ms_total=phase3_row_store_write_elapsed_ms_total,
        gpu_to_cpu_bytes_total=phase3_gpu_to_cpu_bytes_total,
        cpu_to_gpu_bytes_total=phase3_cpu_to_gpu_bytes_total,
        copy_count=phase3_copy_count,
    )
