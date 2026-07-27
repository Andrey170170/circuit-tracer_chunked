"""Phase 0 precomputation coordinator for NNSight attribution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.attribution.nnsight.phases.phase0_activation import (
    prepare_phase0_activation_state,
)
from circuit_tracer.attribution.nnsight.phases.phase0_cleanup import (
    Phase0CleanupOwner,
    Phase0ExecutionError,
    phase0_cleanup_error,
    transfer_phase0_cleanup_ownership,
)
from circuit_tracer.attribution.nnsight.phases.phase0_context import (
    Phase0AttributionPolicy,
    Phase0ProfileSettings,
    configure_phase0_context,
    configure_phase0_transcoders,
    create_phase0_context,
    log_phase0_profile,
)
from circuit_tracer.attribution.nnsight.phases.phase0_evidence import (
    observe_phase0_completion,
    record_phase0_cross_cluster_evidence,
)
from circuit_tracer.attribution.nnsight.phases.phase0_tokens import prepare_phase0_tokens
from circuit_tracer.attribution.nnsight.prefix_view import PrefixViewMetadata
from circuit_tracer.observability.events import MemoryBoundary, TraceObserver

__all__ = [
    "Phase0CleanupOwner",
    "Phase0Config",
    "Phase0ExecutionError",
    "Phase0Inputs",
    "Phase0Result",
    "run_phase0",
]


@dataclass(frozen=True)
class Phase0Inputs:
    logger: Any
    model: Any
    prompt: str | torch.Tensor | list[int]
    sparsification: Any
    telemetry_observer: TraceObserver
    phase0_context_override: Any | None
    prefix_view_metadata: PrefixViewMetadata | None
    exact_encoder_residency_metadata: dict[str, object]
    phase4_execution_metadata: dict[str, object]
    cross_cluster_debug_summary: dict[str, object] | None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None
    cleanup_owner: Phase0CleanupOwner


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
    decoder_active_row_residency: bool = False
    decoder_active_row_max_bytes: int = 0


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


def _attribution_policy(config: Phase0Config) -> Phase0AttributionPolicy:
    return Phase0AttributionPolicy(
        chunked_feature_replay_window=config.chunked_feature_replay_window,
        error_vector_prefetch_lookahead=config.error_vector_prefetch_lookahead,
        stage_encoder_vecs_on_cpu=config.stage_encoder_vecs_on_cpu,
        stage_error_vectors_on_cpu=config.stage_error_vectors_on_cpu,
        row_subchunk_size=config.row_subchunk_size,
        exact_encoder_residency=config.exact_encoder_residency_config.effective_mode,
        internal_precision_requested=config.internal_precision_requested,
        resolved_dtype_map=config.resolved_dtype_map,
        decoder_chunk_cache=config.decoder_chunk_cache,
        decoder_cache_fingerprint=config.decoder_cache_fingerprint,
        decoder_active_row_residency=config.decoder_active_row_residency,
        decoder_active_row_max_bytes=config.decoder_active_row_max_bytes,
    )


def _profile_settings(config: Phase0Config) -> Phase0ProfileSettings:
    return Phase0ProfileSettings(
        exact_chunked_provider_enabled=config.exact_chunked_provider_enabled,
        exact_chunked_decoder=config.exact_chunked_decoder,
        planner_enabled=config.planner_enabled,
        max_phase4_feature_batch_size=config.max_phase4_feature_batch_size,
        phase1_trace_batch_config=config.phase1_trace_batch_config,
        phase1_trace_batch_metadata=config.phase1_trace_batch_metadata,
        phase4_refresh_policy_config=config.phase4_refresh_policy_config,
        phase4_ranker_config=config.phase4_ranker_config,
        row_store_cache_control_config=config.row_store_cache_control_config,
        exact_encoder_residency_config=config.exact_encoder_residency_config,
        exact_trace_internal_dtype_name=config.exact_trace_internal_dtype_name,
        effective_source_batch_size=config.effective_source_batch_size,
        effective_feature_batch_size=config.effective_feature_batch_size,
        effective_logit_batch_size=config.effective_logit_batch_size,
    )


def run_phase0(*, inputs: Phase0Inputs, config: Phase0Config) -> Phase0Result:
    """Run the complete Phase 0 precompute contract."""
    try:
        inputs.logger.info("Phase 0: Precomputing activations and vectors")
        phase_start = time.perf_counter()
        tokens = prepare_phase0_tokens(
            model=inputs.model,
            prompt=inputs.prompt,
            output_position=config.output_position,
            prefix_view_metadata=inputs.prefix_view_metadata,
        )
        inputs.telemetry_observer.observe(MemoryBoundary("Phase 0 start", inputs.model.device))
        configure_phase0_transcoders(
            model=inputs.model,
            logger=inputs.logger,
            observer=inputs.telemetry_observer,
            profile=config.profile,
            activation_threshold_compare_mode=config.phase0_activation_threshold_compare_mode,
            cross_cluster_debug_enabled=config.cross_cluster_debug_enabled,
        )
        attribution_policy = _attribution_policy(config)
        if config.profile:
            log_phase0_profile(
                logger=inputs.logger,
                model=inputs.model,
                tokens=tokens,
                attribution=attribution_policy,
                settings=_profile_settings(config),
            )
        ctx = create_phase0_context(
            model=inputs.model,
            tokens=tokens,
            sparsification=inputs.sparsification,
            context_override=inputs.phase0_context_override,
            policy=attribution_policy,
            observer=inputs.telemetry_observer,
        )
        transfer_phase0_cleanup_ownership(owner=inputs.cleanup_owner, ctx=ctx)
        configure_phase0_context(
            ctx=ctx,
            logger=inputs.logger,
            observer=inputs.telemetry_observer,
            profile=config.profile,
            capture_phase3_gradient_bundle_enabled=config.capture_phase3_gradient_bundle_enabled,
            phase1_trace_batch_metadata=config.phase1_trace_batch_metadata,
            phase4_execution_metadata=inputs.phase4_execution_metadata,
            exact_encoder_residency_metadata=inputs.exact_encoder_residency_metadata,
        )
        activation = prepare_phase0_activation_state(
            ctx=ctx,
            prefix_view_metadata=inputs.prefix_view_metadata,
            diagnostic_feature_cap=config.diagnostic_feature_cap,
            profile=config.profile,
            logger=inputs.logger,
            observer=inputs.telemetry_observer,
        )
        observe_phase0_completion(
            ctx=ctx,
            model=inputs.model,
            logger=inputs.logger,
            observer=inputs.telemetry_observer,
            phase_start=phase_start,
            profile=config.profile,
        )
        result_activation_matrix = activation.activation_matrix
        if inputs.cross_cluster_debug_summary is not None:
            result_activation_matrix = record_phase0_cross_cluster_evidence(
                ctx=ctx,
                model=inputs.model,
                activation_matrix=activation.activation_matrix,
                observer=inputs.telemetry_observer,
                activation_threshold_compare_mode=config.phase0_activation_threshold_compare_mode,
                stage_encoder_vecs_on_cpu=config.stage_encoder_vecs_on_cpu,
                stage_error_vectors_on_cpu=config.stage_error_vectors_on_cpu,
                cross_cluster_debug_summary=inputs.cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=inputs.cross_cluster_debug_checkpoints,
            )
        return Phase0Result(
            ctx=ctx,
            input_ids=tokens.input_ids,
            n_input_pos=tokens.n_input_pos,
            output_position=tokens.output_position,
            trace_input_ids=tokens.trace_input_ids,
            activation_matrix=result_activation_matrix,
            prefix_view_length=tokens.prefix_view_length,
            prefix_view_activation_mask_metadata=(activation.prefix_view_activation_mask_metadata),
            exact_encoder_residency_metadata=inputs.exact_encoder_residency_metadata,
            phase4_execution_metadata=inputs.phase4_execution_metadata,
        )
    except BaseException as exc:
        cleanup_error = phase0_cleanup_error(owner=inputs.cleanup_owner, cause=exc)
        if cleanup_error is None:
            raise
        raise cleanup_error from exc
