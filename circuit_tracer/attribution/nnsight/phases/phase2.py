"""Phase 2 input-vector, replay, and row-store setup for NNSight attribution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import time
from typing import Any, Literal, cast

import torch

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.attribution.nnsight.phase_support import (
    _build_phase3_frontier_buffer_metadata,
    _build_vector_stats,
)
from circuit_tracer.attribution.nnsight.replay import (
    _build_phase0_activation_matrix_from_loaded_bundle,
    _build_phase0_donor_bundle_payload,
    _build_phase0_replay_metadata,
    _build_phase0_replay_validation_context,
    _build_phase3_replay_metadata,
    _extract_clt_constants_hash_from_snapshot,
    _hash_float_tensor,
    _hash_index_tensor,
    _hash_tensor_raw_bytes,
    _load_phase0_donor_bundle_npz,
    _load_phase3_gradient_donor_bundle_npz,
    _load_phase3_row_donor_bundle_npz,
)
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_cross_cluster_runtime_snapshot,
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.observability.human_logs import (
    _log_memory_boundary,
    _log_phase_metrics,
    _snapshot_diagnostics,
)
from circuit_tracer.utils.disk_offload import offload_modules


@dataclass(frozen=True)
class Phase2Inputs:
    logger: Any
    model: Any
    ctx: Any
    input_ids: torch.Tensor
    activation_matrix: torch.Tensor
    telemetry_observer: Any
    telemetry_recorder: Any
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    offload_handles: list[Any]
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None
    target_logits_override: torch.Tensor | None
    resource_owner: "Phase2ResourceOwner"


@dataclass
class Phase2ResourceOwner:
    """Expose Phase 2 row stores to the orchestrator if setup exits early."""

    feature_row_store: _FileBackedFeatureRowStore | None = None
    nonfeature_row_store: _FileBackedFeatureRowStore | None = None


@dataclass(frozen=True)
class Phase2Config:
    output_position: int | None
    n_input_pos: int
    max_n_logits: int
    desired_logit_prob: float
    phase0_replay_mode_resolved: str
    phase0_donor_bundle_path: str | os.PathLike[str] | None
    phase0_donor_context_policy_resolved: str
    capture_phase0_donor_bundle_enabled: bool
    offload: Literal["cpu", "disk", None]
    max_feature_nodes: int | None
    phase3_frontier_buffer_relative_epsilon: float | None
    phase3_frontier_buffer_max_extra: int
    phase4_frontier_buffer_relative_epsilon: float | None
    phase4_frontier_buffer_max_extra_per_refresh: int
    phase4_frontier_buffer_max_extra_total: int
    compact_output: bool
    exact_chunked_decoder: bool
    use_compact_feature_row_store: bool
    exact_trace_internal_dtype_resolved: torch.dtype
    phase4_refresh_prepared_chunk_cache_bytes_effective: int
    row_store_cache_control_config: Any
    row_store_temp_root_policy_resolved: Literal["default", "env_node_local"]
    row_store_temp_root: str | os.PathLike[str] | None
    row_store_preallocate: bool
    feature_row_storage_dtype: torch.dtype
    row_abs_sum_dtype: torch.dtype
    effective_feature_batch_size: int
    phase3_gradient_replay_mode_resolved: str
    phase3_gradient_donor_bundle_path: str | os.PathLike[str] | None
    phase3_replay_validation_policy_resolved: str
    trace_batch_size: int
    phase3_row_replay_mode_resolved: str
    phase3_row_donor_bundle_path: str | os.PathLike[str] | None


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
    feature_row_store: _FileBackedFeatureRowStore | None
    nonfeature_row_store: _FileBackedFeatureRowStore | None
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
    """Run the complete Phase 2 setup contract."""
    logger = inputs.logger
    model = inputs.model
    ctx = inputs.ctx
    input_ids = inputs.input_ids
    activation_matrix = inputs.activation_matrix
    telemetry_observer = inputs.telemetry_observer
    telemetry_recorder = inputs.telemetry_recorder
    cross_cluster_debug_summary = inputs.cross_cluster_debug_summary
    cross_cluster_debug_checkpoints = inputs.cross_cluster_debug_checkpoints
    offload_handles = inputs.offload_handles
    attribution_targets = inputs.attribution_targets
    target_logits_override = inputs.target_logits_override
    output_position = config.output_position
    n_input_pos = config.n_input_pos
    max_n_logits = config.max_n_logits
    desired_logit_prob = config.desired_logit_prob
    phase0_replay_mode_resolved = config.phase0_replay_mode_resolved
    phase0_donor_bundle_path = config.phase0_donor_bundle_path
    phase0_donor_context_policy_resolved = config.phase0_donor_context_policy_resolved
    capture_phase0_donor_bundle_enabled = config.capture_phase0_donor_bundle_enabled
    offload = config.offload
    max_feature_nodes = config.max_feature_nodes
    phase3_frontier_buffer_relative_epsilon = config.phase3_frontier_buffer_relative_epsilon
    phase3_frontier_buffer_max_extra = config.phase3_frontier_buffer_max_extra
    phase4_frontier_buffer_relative_epsilon = config.phase4_frontier_buffer_relative_epsilon
    phase4_frontier_buffer_max_extra_per_refresh = (
        config.phase4_frontier_buffer_max_extra_per_refresh
    )
    phase4_frontier_buffer_max_extra_total = config.phase4_frontier_buffer_max_extra_total
    compact_output = config.compact_output
    exact_chunked_decoder = config.exact_chunked_decoder
    use_compact_feature_row_store = config.use_compact_feature_row_store
    exact_trace_internal_dtype_resolved = config.exact_trace_internal_dtype_resolved
    phase4_refresh_prepared_chunk_cache_bytes_effective = (
        config.phase4_refresh_prepared_chunk_cache_bytes_effective
    )
    row_store_cache_control_config = config.row_store_cache_control_config
    row_store_temp_root_policy_resolved = config.row_store_temp_root_policy_resolved
    row_store_temp_root = config.row_store_temp_root
    row_store_preallocate = config.row_store_preallocate
    feature_row_storage_dtype = config.feature_row_storage_dtype
    row_abs_sum_dtype = config.row_abs_sum_dtype
    effective_feature_batch_size = config.effective_feature_batch_size
    phase3_gradient_replay_mode_resolved = config.phase3_gradient_replay_mode_resolved
    phase3_gradient_donor_bundle_path = config.phase3_gradient_donor_bundle_path
    phase3_replay_validation_policy_resolved = config.phase3_replay_validation_policy_resolved
    trace_batch_size = config.trace_batch_size
    phase3_row_replay_mode_resolved = config.phase3_row_replay_mode_resolved
    phase3_row_donor_bundle_path = config.phase3_row_donor_bundle_path
    feature_row_store: _FileBackedFeatureRowStore | None = None
    nonfeature_row_store: _FileBackedFeatureRowStore | None = None
    edge_matrix: torch.Tensor | None = None
    phase0_donor_bundle_payload: dict[str, object] | None = None
    loaded_phase3_row_donor_bundle: dict[str, object] | None = None

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase2_start = time.perf_counter()
    _log_memory_boundary(logger, "Phase 2 start", model.device)

    # Create AttributionTargets using NNSight's unembed_weight accessor
    output_logits = (
        ctx.get_logits_at_position(output_position)[0]
        if output_position is not None and output_position != n_input_pos - 1
        else ctx.get_last_token_logits()[0]
    )
    if target_logits_override is not None:
        output_logits = target_logits_override.to(device=output_logits.device)
    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=output_logits,
        unembed_proj=cast(torch.Tensor, model.unembed_weight),  # NNSight uses unembed_weight
        tokenizer=model.tokenizer,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    log_attribution_target_info(targets, attribution_targets, logger)
    target_token_ids_tensor = torch.tensor(
        [int(target.vocab_idx) for target in targets.logit_targets],
        dtype=torch.int64,
        device=output_logits.device,
    )

    host_activation_matrix = activation_matrix.coalesce()
    host_transcoder_snapshot_for_replay = _snapshot_diagnostics(model.transcoders)
    host_clt_constants_hash = _extract_clt_constants_hash_from_snapshot(
        host_transcoder_snapshot_for_replay
    )
    host_validation_context = _build_phase0_replay_validation_context(
        input_tokens=input_ids,
        target_token_ids=target_token_ids_tensor,
        activation_matrix=host_activation_matrix,
        clt_constants_hash=host_clt_constants_hash,
    )
    host_hashes_for_replay_metadata = {
        "input_tokens_hash": host_validation_context.get("input_tokens_hash"),
        "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
        "active_feature_membership_hash_raw_order": host_validation_context.get(
            "active_feature_membership_hash_raw_order"
        ),
        "active_feature_membership_hash_canonical": host_validation_context.get(
            "active_feature_membership_hash_canonical"
        ),
        "clt_constants_hash": host_validation_context.get("clt_constants_hash"),
    }

    if phase0_replay_mode_resolved == "donor_phase0":
        assert phase0_donor_bundle_path is not None
        loaded_phase0_donor_bundle = _load_phase0_donor_bundle_npz(
            phase0_donor_bundle_path,
            context_policy=cast(
                Literal["strict", "warn"],
                phase0_donor_context_policy_resolved,
            ),
            validation_context=host_validation_context,
        )
        donor_activation_matrix = _build_phase0_activation_matrix_from_loaded_bundle(
            loaded_phase0_donor_bundle,
            device=host_activation_matrix.device,
        )
        replace_phase0_activation_state = getattr(ctx, "replace_phase0_activation_state", None)
        if callable(replace_phase0_activation_state):
            replace_phase0_activation_state(donor_activation_matrix)
        else:
            raise RuntimeError(
                "Attribution context does not support Phase-0 activation-state replacement"
            )

        activation_matrix = ctx.activation_matrix.coalesce()
        donor_validation_metadata = cast(
            dict[str, object],
            loaded_phase0_donor_bundle.get("validation_metadata", {}),
        )
        donor_dtype_metadata = cast(
            dict[str, object],
            loaded_phase0_donor_bundle.get("dtype_metadata", {}),
        )
        donor_computed_hashes = cast(
            dict[str, object],
            donor_validation_metadata.get("computed_hashes", {}),
        )
        donor_stored_hashes = cast(
            dict[str, object],
            donor_validation_metadata.get("stored_hashes", {}),
        )
        donor_warning_list = [
            str(item) for item in cast(list[object], donor_validation_metadata.get("warnings", []))
        ]
        donor_warning_count = int(
            cast(
                int,
                donor_validation_metadata.get(
                    "validation_failure_count",
                    len(donor_warning_list),
                ),
            )
        )
        phase0_replay_status = "applied_with_warnings" if donor_warning_list else "applied"
        phase0_replay_metadata = _build_phase0_replay_metadata(
            mode=phase0_replay_mode_resolved,
            status=phase0_replay_status,
            donor_bundle_path=phase0_donor_bundle_path,
            context_policy=phase0_donor_context_policy_resolved,
            validation_warnings=donor_warning_list,
            validation_failure_count=donor_warning_count,
            dtype_metadata=donor_dtype_metadata,
            host_hashes=host_hashes_for_replay_metadata,
            donor_hashes={
                "computed": donor_computed_hashes,
                "stored": donor_stored_hashes,
            },
            host_active_feature_count=int(host_activation_matrix._nnz()),
            donor_active_feature_count=int(activation_matrix._nnz()),
            replay_single_step_intended=True,
            note="single-step intended replay mode",
        )
        telemetry_recorder.record_event(
            scope="phase",
            name="phase2.phase0_replay",
            phase="phase2",
            attrs={
                "phase0_replay_mode": phase0_replay_mode_resolved,
                "phase0_replay_status": phase0_replay_status,
                "context_policy": phase0_donor_context_policy_resolved,
                "validation_warning_count": int(len(donor_warning_list)),
                "dtype_roundtrip_loss": bool(
                    donor_dtype_metadata.get("dtype_roundtrip_loss", False)
                ),
                "host_active_feature_count": int(host_activation_matrix._nnz()),
                "donor_active_feature_count": int(activation_matrix._nnz()),
            },
        )
    else:
        phase0_replay_metadata = _build_phase0_replay_metadata(
            mode=phase0_replay_mode_resolved,
            status="disabled",
            donor_bundle_path=None,
            context_policy=phase0_donor_context_policy_resolved,
            host_hashes=host_hashes_for_replay_metadata,
            host_active_feature_count=int(host_activation_matrix._nnz()),
            replay_single_step_intended=True,
            note="single-step intended replay mode",
        )
        telemetry_recorder.record_event(
            scope="phase",
            name="phase2.phase0_replay",
            phase="phase2",
            attrs={
                "phase0_replay_mode": phase0_replay_mode_resolved,
                "phase0_replay_status": "disabled",
            },
        )

    if capture_phase0_donor_bundle_enabled:
        valid_target_mask = (target_token_ids_tensor >= 0) & (
            target_token_ids_tensor < int(output_logits.shape[0])
        )
        target_logits = (
            output_logits[target_token_ids_tensor[valid_target_mask]]
            if bool(valid_target_mask.any().item())
            else None
        )
        capture_status = (
            "captured_replayed_effective_state"
            if phase0_replay_mode_resolved != "disabled"
            else "captured"
        )
        phase0_donor_bundle_payload = _build_phase0_donor_bundle_payload(
            activation_matrix=activation_matrix,
            input_tokens=input_ids,
            target_token_ids=target_token_ids_tensor,
            target_probabilities=targets.logit_probabilities,
            target_logits=target_logits,
            transcoder_diagnostic_snapshot=_snapshot_diagnostics(model.transcoders),
            status=capture_status,
        )
        phase0_donor_bundle_payload["replayed_effective_state"] = bool(
            phase0_replay_mode_resolved != "disabled"
        )
        phase0_donor_bundle_payload["phase0_replay_mode"] = phase0_replay_mode_resolved

    feat_layers, feat_pos, feat_ids = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    if cross_cluster_debug_summary is not None:
        phase1_runtime_summary, phase1_runtime_stream = _build_cross_cluster_runtime_snapshot(
            device=model.device,
            ctx=ctx,
            transcoder=model.transcoders,
        )
        target_token_ids = [int(target.vocab_idx) for target in targets.logit_targets]
        target_probabilities = targets.logit_probabilities.detach().cpu()
        target_probability_stats = _build_vector_stats(
            target_probabilities,
            epsilon=1e-12,
            top_k=8,
        )
        phase1_summary_checkpoint = {
            "target_count": int(len(targets)),
            "target_token_ids": target_token_ids,
            "target_token_ids_hash": _hash_index_tensor(
                torch.tensor(target_token_ids, dtype=torch.int64)
            )
            if target_token_ids
            else None,
            "target_probability_stats": target_probability_stats,
            "target_logit_state_hash": _hash_float_tensor(
                target_probabilities,
                dtype=torch.float64,
            ),
            **phase1_runtime_summary,
        }
        phase1_stream_checkpoint = {
            "target_count": int(len(targets)),
            "target_token_ids_hash": phase1_summary_checkpoint["target_token_ids_hash"],
            "target_probability_count": int(cast(int, target_probability_stats["count"])),
            "target_probability_nonfinite_count": int(
                cast(int, target_probability_stats["nonfinite_count"])
            ),
            "target_probability_abs_sum": _safe_float(
                cast(torch.Tensor | float | int | None, target_probability_stats.get("abs_sum"))
            ),
            "target_probability_max": _safe_float(
                cast(torch.Tensor | float | int | None, target_probability_stats.get("max"))
            ),
            "target_probability_effectively_all_zero": bool(
                target_probability_stats["effectively_all_zero"]
            ),
            "target_logit_state_hash": phase1_summary_checkpoint["target_logit_state_hash"],
            **phase1_runtime_stream,
        }
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
            checkpoint_name="phase1_target_logits",
            phase="phase1",
            summary_payload=phase1_summary_checkpoint,
            stream_payload=phase1_stream_checkpoint,
        )
        cross_cluster_debug_summary["phase0_replay_metadata"] = phase0_replay_metadata
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
            checkpoint_name="phase2_phase0_replay",
            phase="phase2",
            summary_payload=phase0_replay_metadata,
            stream_payload={
                "phase0_replay_mode": phase0_replay_metadata.get("mode"),
                "phase0_replay_status": phase0_replay_metadata.get("status"),
                "validation_warning_count": phase0_replay_metadata.get("validation_warning_count"),
                "dtype_roundtrip_loss": cast(
                    dict[str, object],
                    phase0_replay_metadata.get("dtype_metadata", {}),
                ).get("dtype_roundtrip_loss"),
            },
        )

    if offload:
        offload_handles += offload_modules([model.embed_location], offload)
        tied_embeds = (
            model.embed_weight.untyped_storage().data_ptr()
            == model.unembed_weight.untyped_storage().data_ptr()
        )
        if not tied_embeds:
            offload_handles += offload_modules([model.lm_head], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(targets)
    total_nodes = logit_offset + n_logits

    base_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    actual_max_feature_nodes = base_max_feature_nodes
    phase3_frontier_buffer_metadata = _build_phase3_frontier_buffer_metadata(
        seed_feature_influences=None,
        base_max_feature_nodes=int(base_max_feature_nodes),
        total_active_features=int(total_active_feats),
        relative_epsilon=phase3_frontier_buffer_relative_epsilon,
        max_extra=int(phase3_frontier_buffer_max_extra),
    )
    row_store_capacity_feature_nodes = min(
        base_max_feature_nodes
        + (
            int(phase3_frontier_buffer_max_extra)
            if phase3_frontier_buffer_relative_epsilon is not None
            else 0
        )
        + (
            int(phase4_frontier_buffer_max_extra_total)
            if phase4_frontier_buffer_relative_epsilon is not None
            else 0
        ),
        total_active_feats,
    )
    phase4_frontier_buffer_metadata: dict[str, object] = {
        "schema_version": 1,
        "requested": bool(
            phase4_frontier_buffer_relative_epsilon is not None
            or phase4_frontier_buffer_max_extra_per_refresh > 0
            or phase4_frontier_buffer_max_extra_total > 0
        ),
        "enabled": bool(
            phase4_frontier_buffer_relative_epsilon is not None
            and phase4_frontier_buffer_max_extra_per_refresh > 0
            and phase4_frontier_buffer_max_extra_total > 0
        ),
        "effective": False,
        "relative_epsilon": None
        if phase4_frontier_buffer_relative_epsilon is None
        else float(phase4_frontier_buffer_relative_epsilon),
        "max_extra_per_refresh": int(phase4_frontier_buffer_max_extra_per_refresh),
        "max_extra_total": int(phase4_frontier_buffer_max_extra_total),
        "extra_feature_count_total": 0,
        "expanded_refresh_count": 0,
        "fallback_count": 0,
        "capacity_feature_nodes": int(row_store_capacity_feature_nodes),
        "initial_target_feature_nodes": int(base_max_feature_nodes),
        "final_actual_max_feature_nodes": int(actual_max_feature_nodes),
        "events": [],
    }
    logger.info(f"Will include {actual_max_feature_nodes} of {total_active_feats} feature nodes")

    if use_compact_feature_row_store:
        # Benchmark-critical path only: exact chunked decoder + compact output.
        # Keep dense full-row behavior unchanged for non-compact Graph outputs.
        assert compact_output
        assert exact_chunked_decoder
        n_nonfeature_columns = int(logit_offset - total_active_feats)
        feature_row_store = _FileBackedFeatureRowStore(
            n_rows=row_store_capacity_feature_nodes + n_logits,
            n_feature_columns=total_active_feats,
            dtype=exact_trace_internal_dtype_resolved,
            row_abs_sum_dtype=exact_trace_internal_dtype_resolved,
            read_chunk_cache_bytes=256 * 1024 * 1024,
            prepared_read_cache_bytes=phase4_refresh_prepared_chunk_cache_bytes_effective,
            row_store_cache_control_mode=row_store_cache_control_config.effective_mode,
            temp_root_policy=row_store_temp_root_policy_resolved,
            temp_root=row_store_temp_root,
            preallocate=row_store_preallocate,
            telemetry_recorder=telemetry_recorder,
        )
        inputs.resource_owner.feature_row_store = feature_row_store
        nonfeature_row_store = _FileBackedFeatureRowStore(
            n_rows=row_store_capacity_feature_nodes + n_logits,
            n_feature_columns=n_nonfeature_columns,
            dtype=exact_trace_internal_dtype_resolved,
            row_abs_sum_dtype=exact_trace_internal_dtype_resolved,
            read_chunk_cache_bytes=256 * 1024 * 1024,
            prepared_read_cache_bytes=0,
            row_store_cache_control_mode=row_store_cache_control_config.effective_mode,
            temp_root_policy=row_store_temp_root_policy_resolved,
            temp_root=row_store_temp_root,
            preallocate=row_store_preallocate,
            telemetry_recorder=telemetry_recorder,
        )
        inputs.resource_owner.nonfeature_row_store = nonfeature_row_store
    else:
        edge_matrix = torch.zeros(row_store_capacity_feature_nodes + n_logits, total_nodes)

    # Maps stored row indices to original feature/node indices.
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(row_store_capacity_feature_nodes + n_logits, dtype=torch.int32)

    phase2_extra: dict[str, object] = {
        "row_store_mode": (
            "compact_feature_file_backed_dense" if use_compact_feature_row_store else "dense_full"
        ),
        "phase0_replay_mode": phase0_replay_metadata.get("mode"),
        "phase0_replay_status": phase0_replay_metadata.get("status"),
        "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
            "validation_warning_count"
        ),
    }
    if use_compact_feature_row_store:
        assert feature_row_store is not None
        assert nonfeature_row_store is not None
        phase2_extra.update(
            feature_row_store="dense_memmap",
            feature_row_store_path=feature_row_store.path,
            nonfeature_row_store_path=nonfeature_row_store.path,
            row_abs_sums_shape=f"{tuple(feature_row_store.row_abs_max.shape)}",
            row_abs_max_shape=f"{tuple(feature_row_store.row_abs_max.shape)}",
            row_l1_scaled_shape=f"{tuple(feature_row_store.row_l1_scaled.shape)}",
            feature_edge_columns=total_active_feats,
            nonfeature_edge_columns=n_nonfeature_columns,
            **feature_row_store.get_diagnostic_snapshot(),
        )
    else:
        assert edge_matrix is not None
        phase2_extra.update(
            # The non-compact branch allocates this immediately above.
            edge_matrix_shape=f"{tuple(edge_matrix.shape)}",
            edge_matrix_dtype=edge_matrix.dtype,
        )

    _log_phase_metrics(
        logger,
        "Input vector build",
        phase2_start,
        model.device,
        **phase2_extra,
    )
    phase2_elapsed_ms = (time.perf_counter() - phase2_start) * 1000.0
    telemetry_observer.phase(
        name="phase2.input_vector_build",
        phase="phase2",
        elapsed_ms=phase2_elapsed_ms,
        attrs=phase2_extra,
        wall_clock=True,
    )
    if cross_cluster_debug_summary is not None:
        phase2_runtime_summary, phase2_runtime_stream = _build_cross_cluster_runtime_snapshot(
            device=model.device,
            ctx=ctx,
            transcoder=model.transcoders,
        )
        row_store_dtype_for_metrics = (
            exact_trace_internal_dtype_resolved
            if use_compact_feature_row_store
            else feature_row_storage_dtype
        )
        row_abs_sum_dtype_for_metrics = (
            exact_trace_internal_dtype_resolved
            if use_compact_feature_row_store
            else row_abs_sum_dtype
        )
        row_denominator_component_count = 2 if use_compact_feature_row_store else 1
        row_count = int(actual_max_feature_nodes + n_logits)
        row_store_expected_bytes = (
            row_count
            * int(total_active_feats)
            * torch.empty((), dtype=row_store_dtype_for_metrics).element_size()
        )
        row_abs_sums_expected_bytes = (
            row_denominator_component_count
            * row_count
            * torch.empty((), dtype=row_abs_sum_dtype_for_metrics).element_size()
        )
        phase2_summary_checkpoint = {
            "feat_layers_hash": _hash_index_tensor(feat_layers),
            "feat_pos_hash": _hash_index_tensor(feat_pos),
            "feat_ids_hash": _hash_index_tensor(feat_ids),
            "feature_count": int(total_active_feats),
            "phase0_replay_mode": phase0_replay_metadata.get("mode"),
            "phase0_replay_status": phase0_replay_metadata.get("status"),
            "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                "validation_warning_count"
            ),
            "decoder_chunk_size": (
                int(getattr(model.transcoders, "decoder_chunk_size", 0))
                if getattr(model.transcoders, "decoder_chunk_size", None) is not None
                else None
            ),
            "row_store_mode": phase2_extra.get("row_store_mode"),
            "row_denominator_component_count": int(row_denominator_component_count),
            "row_store_expected_bytes": int(row_store_expected_bytes),
            "row_abs_sums_expected_bytes": int(row_abs_sums_expected_bytes),
            "row_denominator_expected_bytes": int(row_abs_sums_expected_bytes),
            "phase4_feature_batch_size_initial": int(effective_feature_batch_size),
            **phase2_runtime_summary,
        }
        phase2_stream_checkpoint = {
            "feat_layers_hash": phase2_summary_checkpoint["feat_layers_hash"],
            "feat_pos_hash": phase2_summary_checkpoint["feat_pos_hash"],
            "feat_ids_hash": phase2_summary_checkpoint["feat_ids_hash"],
            "feature_count": int(total_active_feats),
            "phase0_replay_mode": phase0_replay_metadata.get("mode"),
            "phase0_replay_status": phase0_replay_metadata.get("status"),
            "phase0_replay_validation_warning_count": phase0_replay_metadata.get(
                "validation_warning_count"
            ),
            "decoder_chunk_size": phase2_summary_checkpoint["decoder_chunk_size"],
            "row_store_mode": phase2_summary_checkpoint["row_store_mode"],
            "row_denominator_component_count": int(row_denominator_component_count),
            "row_store_expected_bytes": int(row_store_expected_bytes),
            "row_abs_sums_expected_bytes": int(row_abs_sums_expected_bytes),
            "row_denominator_expected_bytes": int(row_abs_sums_expected_bytes),
            "phase4_feature_batch_size_initial": int(effective_feature_batch_size),
            **phase2_runtime_stream,
        }
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=cross_cluster_debug_checkpoints,
            checkpoint_name="phase2_feature_ordering",
            phase="phase2",
            summary_payload=phase2_summary_checkpoint,
            stream_payload=phase2_stream_checkpoint,
        )

    if phase3_gradient_replay_mode_resolved == "donor":
        assert phase3_gradient_donor_bundle_path is not None
        loaded_gradient_bundle = _load_phase3_gradient_donor_bundle_npz(
            phase3_gradient_donor_bundle_path,
            target_token_ids=target_token_ids_tensor,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            expected_n_layers=int(n_layers),
            expected_gradient_batch_size=int(trace_batch_size),
            expected_n_positions=int(n_pos),
            expected_d_model=int(targets.logit_vectors.shape[-1]),
            validation_policy=cast(Literal["strict"], phase3_replay_validation_policy_resolved),
        )
        gradient_tensor = cast(torch.Tensor, loaded_gradient_bundle["gradients"])
        setattr(ctx, "phase3_gradient_replay_tensor", gradient_tensor)
        setattr(ctx, "phase3_gradient_replay_status", "applied")
        gradient_validation_metadata = cast(
            dict[str, object], loaded_gradient_bundle.get("validation_metadata", {})
        )
        phase3_gradient_replay_metadata = _build_phase3_replay_metadata(
            replay_kind="phase3_gradient_replay_v1",
            mode=phase3_gradient_replay_mode_resolved,
            status="applied",
            donor_bundle_path=phase3_gradient_donor_bundle_path,
            validation_policy=phase3_replay_validation_policy_resolved,
            validation_failure_count=int(
                cast(int, gradient_validation_metadata.get("validation_failure_count", 0))
            ),
            donor_hashes=cast(
                dict[str, object], gradient_validation_metadata.get("stored_hashes", {})
            ),
            host_hashes={
                "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
                "active_features_hash": _hash_index_tensor(
                    activation_matrix.indices().T.detach().cpu().reshape(-1)
                ),
                "activation_values_hash": _hash_tensor_raw_bytes(activation_matrix.values()),
                "active_feature_count": int(total_active_feats),
            },
            source="donor_gradient_bundle",
            note="feature/error gradients replayed from donor; token gradient remains host-computed",
        )
    else:
        setattr(ctx, "phase3_gradient_replay_tensor", None)
        setattr(ctx, "phase3_gradient_replay_status", "disabled")

    if phase3_row_replay_mode_resolved == "donor":
        assert phase3_row_donor_bundle_path is not None
        loaded_phase3_row_donor_bundle = _load_phase3_row_donor_bundle_npz(
            phase3_row_donor_bundle_path,
            target_token_ids=target_token_ids_tensor,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            expected_total_active_features=int(total_active_feats),
            validation_policy=cast(Literal["strict"], phase3_replay_validation_policy_resolved),
        )
        row_validation_metadata = cast(
            dict[str, object], loaded_phase3_row_donor_bundle.get("validation_metadata", {})
        )
        phase3_row_replay_metadata = _build_phase3_replay_metadata(
            replay_kind="phase3_row_replay_v1",
            mode=phase3_row_replay_mode_resolved,
            status="applied",
            donor_bundle_path=phase3_row_donor_bundle_path,
            validation_policy=phase3_replay_validation_policy_resolved,
            validation_failure_count=int(
                cast(int, row_validation_metadata.get("validation_failure_count", 0))
            ),
            donor_hashes=cast(dict[str, object], row_validation_metadata.get("stored_hashes", {})),
            host_hashes={
                "target_token_ids_hash": host_validation_context.get("target_token_ids_hash"),
                "active_features_hash": _hash_index_tensor(
                    activation_matrix.indices().T.detach().cpu().reshape(-1)
                ),
                "activation_values_hash": _hash_tensor_raw_bytes(activation_matrix.values()),
                "active_feature_count": int(total_active_feats),
            },
            source="donor_row_bundle_override",
            note=(
                "donor row bundle overrides feature rows and row normalizers; "
                "dense token/error columns remain host-computed"
            ),
        )

    return Phase2Result(
        targets=targets,
        activation_matrix=activation_matrix,
        feat_layers=feat_layers,
        feat_pos=feat_pos,
        feat_ids=feat_ids,
        n_layers=n_layers,
        n_pos=n_pos,
        total_active_feats=total_active_feats,
        logit_offset=logit_offset,
        n_logits=n_logits,
        total_nodes=total_nodes,
        base_max_feature_nodes=base_max_feature_nodes,
        actual_max_feature_nodes=actual_max_feature_nodes,
        row_store_capacity_feature_nodes=row_store_capacity_feature_nodes,
        feature_row_store=feature_row_store,
        nonfeature_row_store=nonfeature_row_store,
        edge_matrix=edge_matrix,
        row_to_node_index=row_to_node_index,
        phase0_donor_bundle_payload=phase0_donor_bundle_payload,
        phase0_replay_metadata=phase0_replay_metadata,
        phase3_frontier_buffer_metadata=phase3_frontier_buffer_metadata,
        phase4_frontier_buffer_metadata=phase4_frontier_buffer_metadata,
        phase3_gradient_replay_metadata=phase3_gradient_replay_metadata,
        phase3_row_replay_metadata=phase3_row_replay_metadata,
        loaded_phase3_row_donor_bundle=loaded_phase3_row_donor_bundle,
    )
