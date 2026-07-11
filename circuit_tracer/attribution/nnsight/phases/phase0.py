"""Phase 0 precomputation for NNSight attribution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sized
import time
from dataclasses import dataclass
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import _build_vector_stats
from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata,
    _apply_prefix_view_activation_mask,
    _resolve_prefix_view_trace_input_ids,
)
from circuit_tracer.attribution.nnsight.replay import _hash_sparse_membership_indices
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_cross_cluster_runtime_snapshot,
    _record_cross_cluster_checkpoint,
    _safe_float,
)
from circuit_tracer.observability.human_logs import (
    _log_memory_boundary,
    _log_phase_metrics,
    _log_sparsification_profile,
    _snapshot_diagnostics,
)
from circuit_tracer.utils.telemetry import format_numeric_metrics


@dataclass(frozen=True)
class Phase0Inputs:
    logger: Any
    model: Any
    prompt: str | torch.Tensor | list[int]
    sparsification: Any
    telemetry_observer: Any
    telemetry_recorder: Any
    phase0_context_override: Any | None
    prefix_view_metadata: PrefixViewMetadata | None
    exact_encoder_residency_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cleanup_owner: "Phase0CleanupOwner"


@dataclass
class Phase0CleanupOwner:
    """Expose a successfully-created Phase 0 context to the orchestrator on failure."""

    ctx: Any | None = None


class Phase0ExecutionError(RuntimeError):
    """Phase 0 failed after creating a context that the orchestrator must clean up."""

    def __init__(self, ctx: Any, cause: BaseException) -> None:
        super().__init__("Phase 0 failed after attribution context creation")
        self.ctx = ctx
        self.cause = cause


def _value_stats_int_entry(value_stats: dict[str, object], key: str) -> int:
    """Read an integer vector-stat entry without changing its runtime coercion."""
    return int(cast(int, value_stats[key]))


def _value_stats_safe_float_entry(value_stats: dict[str, object], key: str) -> float | None:
    """Read a numeric vector-stat entry without changing its runtime coercion."""
    return _safe_float(cast(torch.Tensor | float | int | None, value_stats.get(key)))


def _diagnostic_int_entry(diagnostics: Mapping[str, object], key: str, default: int = 0) -> int:
    """Read an integer diagnostic entry without changing its runtime coercion."""
    return int(cast(int, diagnostics.get(key, default)))


def _diagnostic_mapping_entry(diagnostics: Mapping[str, object], key: str) -> dict[str, object]:
    """Narrow a nested diagnostic mapping while preserving invalid-payload errors."""
    return cast(dict[str, object], diagnostics.get(key, {}))


def _diagnostic_sized_entry(diagnostics: Mapping[str, object], key: str) -> Sized:
    """Narrow a diagnostic collection while preserving invalid-payload errors."""
    return cast(Sized, diagnostics.get(key, {}))


def _with_phase0_cleanup_ownership(
    function: Callable[..., "Phase0Result"],
) -> Callable[..., "Phase0Result"]:
    """Wrap only errors raised after Phase 0 has created its context."""

    def wrapped(*, inputs: Phase0Inputs, config: "Phase0Config") -> "Phase0Result":
        try:
            return function(inputs=inputs, config=config)
        except BaseException as exc:
            if inputs.cleanup_owner.ctx is None:
                raise
            raise Phase0ExecutionError(inputs.cleanup_owner.ctx, exc) from exc

    return wrapped


@dataclass(frozen=True)
class Phase0Config:
    output_position: int | None
    profile: bool
    phase0_activation_threshold_compare_mode: str
    cross_cluster_debug_enabled: bool
    exact_chunked_provider_enabled: bool
    exact_chunked_decoder: bool
    chunked_feature_replay_window: int
    error_vector_prefetch_lookahead: int
    stage_encoder_vecs_on_cpu: bool | None
    stage_error_vectors_on_cpu: bool | None
    row_subchunk_size: int | None
    planner_enabled: bool
    max_phase4_feature_batch_size: int
    phase1_trace_batch_config: Any
    phase1_trace_batch_metadata: dict[str, object]
    phase4_refresh_policy_config: Any
    phase4_ranker_config: Any
    row_store_cache_control_config: Any
    exact_encoder_residency_config: Any
    exact_trace_internal_dtype_name: str
    effective_source_batch_size: int
    effective_feature_batch_size: int
    effective_logit_batch_size: int
    internal_precision_requested: str
    resolved_dtype_map: dict[str, str]
    decoder_chunk_cache: Any | None
    decoder_cache_fingerprint: object | None
    capture_phase3_gradient_bundle_enabled: bool
    diagnostic_feature_cap: int | None


@dataclass(frozen=True)
class Phase0Result:
    ctx: Any
    input_ids: torch.Tensor
    n_input_pos: int
    output_position: int | None
    trace_input_ids: torch.Tensor
    activation_matrix: torch.Tensor
    prefix_view_length: int | None
    prefix_view_activation_mask_metadata: dict[str, int] | None
    exact_encoder_residency_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]


@_with_phase0_cleanup_ownership
def run_phase0(*, inputs: Phase0Inputs, config: Phase0Config) -> Phase0Result:
    """Run the complete Phase 0 precompute contract."""
    logger = inputs.logger
    model = inputs.model
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.perf_counter()
    input_ids = model.ensure_tokenized(inputs.prompt)
    n_input_pos = int(input_ids.shape[-1])
    output_position = config.output_position
    if output_position is not None:
        output_position = int(output_position)
        if output_position < 0 or output_position >= n_input_pos:
            raise ValueError(
                f"output_position must be in [0, {n_input_pos}) (got {output_position})"
            )
    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(
        input_ids, inputs.prefix_view_metadata
    )
    _log_memory_boundary(logger, "Phase 0 start", model.device)

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(
            logger.info if config.profile else None,
            telemetry_recorder=inputs.telemetry_recorder,
        )
    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()
    configure_compare = getattr(
        model.transcoders, "configure_phase0_activation_threshold_compare", None
    )
    if callable(configure_compare):
        configure_compare(
            mode=config.phase0_activation_threshold_compare_mode,
            collect_diagnostics=config.cross_cluster_debug_enabled,
            sample_limit_per_layer=3,
        )

    if config.profile:
        logger.info(
            "Profiling enabled | "
            f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
            f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
            f"exact_chunked_provider_enabled={config.exact_chunked_provider_enabled} | "
            f"exact_chunked_decoder={config.exact_chunked_decoder} | "
            f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
            f"decoder_cache_bytes={getattr(model.transcoders, 'cross_batch_decoder_cache_bytes', 0)} | "
            f"chunked_feature_replay_window={config.chunked_feature_replay_window} | "
            f"error_vector_prefetch_lookahead={config.error_vector_prefetch_lookahead} | "
            f"stage_encoder_vecs_on_cpu={config.stage_encoder_vecs_on_cpu} | "
            f"stage_error_vectors_on_cpu={config.stage_error_vectors_on_cpu} | "
            f"row_subchunk_size={config.row_subchunk_size} | "
            f"planner_enabled={config.planner_enabled} | "
            f"feature_batch_size_max={config.max_phase4_feature_batch_size} | "
            f"phase1_trace_batch_policy={config.phase1_trace_batch_config.requested_policy} "
            f"(effective={config.phase1_trace_batch_config.effective_policy}, "
            f"size_max={config.phase1_trace_batch_config.requested_batch_size_max}, "
            f"size_max_effective={config.phase1_trace_batch_config.effective_batch_size_max}) | "
            f"phase4_refresh_policy={config.phase4_refresh_policy_config.requested_policy} "
            f"(effective={config.phase4_refresh_policy_config.effective_policy}, "
            f"interval_multiplier={config.phase4_refresh_policy_config.requested_interval_multiplier}, "
            f"interval_multiplier_effective={config.phase4_refresh_policy_config.effective_interval_multiplier}, "
            f"queue_multiplier_effective={config.phase4_refresh_policy_config.effective_queue_multiplier}) | "
            f"phase4_ranker={config.phase4_ranker_config.requested_mode} "
            f"(effective={config.phase4_ranker_config.effective_mode}) | "
            f"row_store_cache_control={config.row_store_cache_control_config.requested_mode} "
            f"(effective={config.row_store_cache_control_config.effective_mode}) | "
            f"exact_encoder_residency={config.exact_encoder_residency_config.requested_mode} "
            f"(effective={config.exact_encoder_residency_config.effective_mode}) | "
            f"exact_trace_internal_dtype={config.exact_trace_internal_dtype_name} | "
            f"prompt_tokens={input_ids.shape[-1]} | "
            f"source_batch_size={config.effective_source_batch_size} | "
            f"feature_batch_size={config.effective_feature_batch_size} | "
            f"logit_batch_size={config.effective_logit_batch_size} | "
            f"trace_batch_cap_reason={config.phase1_trace_batch_metadata.get('trace_batch_cap_reason')}"
        )

    if inputs.phase0_context_override is not None:
        ctx = inputs.phase0_context_override
    else:
        ctx = model.setup_attribution(
            input_ids,
            sparsification=inputs.sparsification,
            retain_full_logits=output_position is not None and output_position != n_input_pos - 1,
            chunked_feature_replay_window=config.chunked_feature_replay_window,
            error_vector_prefetch_lookahead=config.error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=config.stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=config.stage_error_vectors_on_cpu,
            row_subchunk_size=config.row_subchunk_size,
            exact_encoder_residency=config.exact_encoder_residency_config.effective_mode,
            internal_precision_requested=config.internal_precision_requested,
            resolved_dtype_map=config.resolved_dtype_map,
            prefix_view_length=prefix_view_length,
            decoder_chunk_cache=config.decoder_chunk_cache,
            decoder_cache_fingerprint=config.decoder_cache_fingerprint,
        )
    inputs.cleanup_owner.ctx = ctx
    runtime_metadata = {
        "exact_encoder_staging_destination": getattr(
            ctx, "exact_encoder_staging_destination", "none"
        ),
        "exact_encoder_materialized_during_phase0": bool(
            getattr(ctx, "exact_encoder_materialized_during_phase0", False)
        ),
        "active_encoder_shape": tuple(getattr(ctx, "encoder_vecs").shape),
        "active_encoder_bytes": int(
            getattr(ctx, "encoder_vecs").numel() * getattr(ctx, "encoder_vecs").element_size()
        ),
        "exact_encoder_pinned_effective": bool(
            getattr(ctx, "exact_encoder_pinned_effective", False)
        ),
        "exact_encoder_pinning_success": getattr(ctx, "exact_encoder_pinning_success", None),
        "exact_encoder_pinning_failure_reason": getattr(
            ctx, "exact_encoder_pinning_failure_reason", None
        ),
    }
    inputs.exact_encoder_residency_metadata.update(runtime_metadata)
    inputs.phase4_execution_metadata.update(runtime_metadata)
    if hasattr(ctx, "set_diagnostic_mode"):
        ctx.set_diagnostic_mode(config.profile)
    if config.capture_phase3_gradient_bundle_enabled:
        setattr(ctx, "capture_phase3_gradients", True)
    configure_ctx_logging = getattr(ctx, "configure_trace_logging", None)
    if callable(configure_ctx_logging):
        configure_ctx_logging(
            logger.info if config.profile else None,
            telemetry_recorder=inputs.telemetry_recorder,
        )
    if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
        ctx.setup_diagnostic_stats.update(
            {
                "phase1_trace_batch": dict(config.phase1_trace_batch_metadata),
                "phase4_execution": dict(inputs.phase4_execution_metadata),
            }
        )

    prefix_mask_metadata: dict[str, int] | None = None
    if (
        inputs.prefix_view_metadata is not None
        and inputs.prefix_view_metadata.get("mode") == "full_sequence_target_position"
    ):
        replace_state = getattr(ctx, "replace_phase0_activation_state", None)
        if not callable(replace_state):
            raise RuntimeError(
                "Attribution context does not support Phase-0 activation-state replacement"
            )
        prefix_mask_metadata = _apply_prefix_view_activation_mask(
            ctx, int(inputs.prefix_view_metadata["target_position"])
        )
        if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
            ctx.setup_diagnostic_stats["prefix_view_activation_mask"] = dict(prefix_mask_metadata)

    if config.diagnostic_feature_cap is not None and config.diagnostic_feature_cap > 0:
        before_cap, after_cap = ctx.apply_diagnostic_feature_cap(config.diagnostic_feature_cap)
        logger.info(
            f"Diagnostic feature cap applied before attribution rows: {before_cap} -> {after_cap} active features"
        )
    if config.profile and getattr(ctx, "sparsification_stats", None):
        _log_sparsification_profile(logger, ctx.sparsification_stats)

    activation_matrix = ctx.activation_matrix
    _log_phase_metrics(
        logger,
        "Precomputation",
        phase_start,
        model.device,
        active_features=ctx.activation_matrix._nnz(),
        logit_retention=getattr(ctx, "logit_retention", "full"),
    )
    elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
    inputs.telemetry_observer.phase(
        name="phase0.precompute",
        phase="phase0",
        elapsed_ms=elapsed_ms,
        attrs={
            "active_features": int(ctx.activation_matrix._nnz()),
            "logit_retention": getattr(ctx, "logit_retention", "full"),
        },
        wall_clock=True,
    )
    if config.profile:
        if getattr(ctx, "setup_diagnostic_stats", None):
            logger.info(
                f"Phase 0 setup diagnostics | {format_numeric_metrics(ctx.setup_diagnostic_stats, limit=20)}"
            )
        transcoder_snapshot = _snapshot_diagnostics(model.transcoders)
        if transcoder_snapshot:
            logger.info(
                f"Precompute diagnostics | {format_numeric_metrics(transcoder_snapshot, limit=20)}"
            )
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")

    if inputs.cross_cluster_debug_summary is not None:
        runtime_summary, runtime_stream = _build_cross_cluster_runtime_snapshot(
            device=model.device, ctx=ctx, transcoder=model.transcoders
        )
        activation_matrix = activation_matrix.coalesce()
        indices = activation_matrix.indices().detach().cpu()
        values = activation_matrix.values().detach().cpu()
        raw_hash = _hash_sparse_membership_indices(
            indices, shape=activation_matrix.shape, canonicalize=False
        )
        canonical_hash = _hash_sparse_membership_indices(
            indices, shape=activation_matrix.shape, canonicalize=True
        )
        n_layers = int(activation_matrix.shape[0])
        layer_counts = (
            torch.bincount(indices[0], minlength=n_layers).tolist()
            if indices.numel() > 0
            else [0] * n_layers
        )
        transcoder_snapshot = runtime_summary.get("transcoder_diagnostic_snapshot")
        threshold_membership = (
            transcoder_snapshot.get("phase0_threshold_membership")
            if isinstance(transcoder_snapshot, dict)
            else None
        )
        if isinstance(threshold_membership, dict):
            threshold_membership = cast(dict[str, object], threshold_membership)
        else:
            threshold_membership = None
        boundary_fingerprints = (
            transcoder_snapshot.get("phase0_boundary_fingerprints")
            if isinstance(transcoder_snapshot, dict)
            else None
        )
        if isinstance(boundary_fingerprints, dict):
            boundary_fingerprints = cast(dict[str, object], boundary_fingerprints)
        else:
            boundary_fingerprints = None
        setup_stats = getattr(ctx, "setup_diagnostic_stats", None)
        pre_clt = (
            setup_stats.get("phase0_pre_clt_input_fingerprints")
            if isinstance(setup_stats, dict)
            else None
        )
        if not isinstance(pre_clt, dict):
            pre_clt = None
        global_hashes = (
            boundary_fingerprints.get("global_hashes")
            if isinstance(boundary_fingerprints, dict)
            else None
        )
        if not isinstance(global_hashes, dict):
            global_hashes = None
        value_stats = _build_vector_stats(values, epsilon=1e-12, top_k=8)
        summary = {
            "active_feature_count": int(activation_matrix._nnz()),
            "per_layer_retained_counts": [int(v) for v in layer_counts],
            "active_feature_indices_hash": raw_hash,
            "active_feature_indices_hash_raw_order": raw_hash,
            "active_feature_membership_hash_canonical": canonical_hash,
            "activation_value_stats": value_stats,
            "phase0_activation_threshold_compare_mode": config.phase0_activation_threshold_compare_mode,
            "phase0_threshold_membership": threshold_membership,
            "phase0_boundary_fingerprints": boundary_fingerprints,
            "phase0_pre_clt_input_fingerprints": pre_clt,
            "phase0_pre_clt_input_global_hash": pre_clt.get("global_hash")
            if isinstance(pre_clt, dict)
            else None,
            "logit_retention": getattr(ctx, "logit_retention", None),
            "staging_flags": {
                "stage_encoder_vecs_on_cpu": bool(config.stage_encoder_vecs_on_cpu),
                "stage_error_vectors_on_cpu": bool(config.stage_error_vectors_on_cpu),
            },
            "setup_diagnostic_stats": setup_stats,
            **runtime_summary,
        }
        stream = {
            "active_feature_count": int(activation_matrix._nnz()),
            "retained_layer_count": n_layers,
            "retained_nonzero_layer_count": sum(1 for value in layer_counts if int(value) > 0),
            "active_feature_indices_hash": summary["active_feature_indices_hash"],
            "active_feature_membership_hash_canonical": canonical_hash,
            "phase0_activation_threshold_compare_mode": config.phase0_activation_threshold_compare_mode,
            "activation_value_count": _value_stats_int_entry(value_stats, "count"),
            "activation_value_nonfinite_count": _value_stats_int_entry(
                value_stats, "nonfinite_count"
            ),
            "activation_value_abs_sum": _value_stats_safe_float_entry(value_stats, "abs_sum"),
            "activation_value_max": _value_stats_safe_float_entry(value_stats, "max"),
            "activation_value_effectively_all_zero": bool(value_stats["effectively_all_zero"]),
            "phase0_threshold_membership_layer_count": len(
                _diagnostic_sized_entry(threshold_membership, "per_layer")
            )
            if isinstance(threshold_membership, dict)
            else None,
            "phase0_threshold_membership_borderline_sample_count": _diagnostic_int_entry(
                threshold_membership, "borderline_sample_count"
            )
            if isinstance(threshold_membership, dict)
            else None,
            "phase0_threshold_membership_near_count_abs_lte_1e_04": _diagnostic_int_entry(
                _diagnostic_mapping_entry(threshold_membership, "near_counts_by_epsilon"),
                "abs_lte_1e-04",
            )
            if isinstance(threshold_membership, dict)
            else None,
            "phase0_pre_clt_input_global_hash": summary.get("phase0_pre_clt_input_global_hash"),
            "phase0_pre_clt_input_layer_count": int(pre_clt.get("layer_count", 0))
            if isinstance(pre_clt, dict)
            else None,
            "phase0_boundary_layer_count": len(
                _diagnostic_sized_entry(boundary_fingerprints, "per_layer")
            )
            if isinstance(boundary_fingerprints, dict)
            else None,
            "phase0_boundary_transcoder_constants_global_hash": _diagnostic_mapping_entry(
                boundary_fingerprints, "transcoder_constant_fingerprints"
            ).get("global_hash")
            if isinstance(boundary_fingerprints, dict)
            else None,
            "phase0_boundary_preactivation_hash_global": global_hashes.get(
                "pre_activation_hash_global"
            )
            if isinstance(global_hashes, dict)
            else None,
            "phase0_boundary_margin_hash_global": global_hashes.get("compare_margin_hash_global")
            if isinstance(global_hashes, dict)
            else None,
            "phase0_boundary_mask_membership_hash_global": global_hashes.get(
                "mask_membership_hash_global"
            )
            if isinstance(global_hashes, dict)
            else None,
            "phase0_boundary_post_activation_hash_global": global_hashes.get(
                "post_activation_hash_global"
            )
            if isinstance(global_hashes, dict)
            else None,
            "logit_retention": getattr(ctx, "logit_retention", None),
            "stage_encoder_vecs_on_cpu": bool(config.stage_encoder_vecs_on_cpu),
            "stage_error_vectors_on_cpu": bool(config.stage_error_vectors_on_cpu),
            "setup_diagnostic_stats_present": bool(getattr(ctx, "setup_diagnostic_stats", None)),
            **runtime_stream,
        }
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=inputs.cross_cluster_debug_summary,
            cross_cluster_debug_checkpoints=inputs.cross_cluster_debug_checkpoints,
            checkpoint_name="phase0_sparse_setup",
            phase="phase0",
            summary_payload=summary,
            stream_payload=stream,
        )

    return Phase0Result(
        ctx=ctx,
        input_ids=input_ids,
        n_input_pos=n_input_pos,
        output_position=output_position,
        trace_input_ids=trace_input_ids,
        activation_matrix=activation_matrix,
        prefix_view_length=prefix_view_length,
        prefix_view_activation_mask_metadata=prefix_mask_metadata,
        exact_encoder_residency_metadata=inputs.exact_encoder_residency_metadata,
        phase4_execution_metadata=inputs.phase4_execution_metadata,
    )
