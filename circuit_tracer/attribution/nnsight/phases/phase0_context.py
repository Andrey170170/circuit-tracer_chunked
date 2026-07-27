"""Attribution-context setup operations for NNSight Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from circuit_tracer.attribution.nnsight.phases.phase0_tokens import Phase0TokenPreparation
from circuit_tracer.observability.events import TraceObserver


@dataclass(frozen=True)
class Phase0AttributionPolicy:
    """Runtime choices passed to attribution-context construction."""

    chunked_feature_replay_window: int
    error_vector_prefetch_lookahead: int
    stage_encoder_vecs_on_cpu: bool | None
    stage_error_vectors_on_cpu: bool | None
    row_subchunk_size: int | None
    exact_encoder_residency: str
    internal_precision_requested: str
    resolved_dtype_map: dict[str, str]
    decoder_chunk_cache: Any | None
    decoder_cache_fingerprint: object | None
    decoder_active_row_residency: bool
    decoder_active_row_max_bytes: int
    phase0_decoder_row_ranges: bool


@dataclass(frozen=True)
class Phase0ProfileSettings:
    """Resolved settings rendered by the Phase 0 profiling message."""

    exact_chunked_provider_enabled: bool
    exact_chunked_decoder: bool
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


@dataclass(frozen=True)
class Phase0ContextConfiguration:
    """Metadata produced while configuring a created attribution context."""

    runtime_metadata: dict[str, object]


def configure_phase0_transcoders(
    *,
    model: Any,
    logger: Any,
    observer: TraceObserver,
    profile: bool,
    activation_threshold_compare_mode: str,
    cross_cluster_debug_enabled: bool,
) -> None:
    """Configure transcoder logging and Phase 0 diagnostic collection."""
    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(logger.info if profile else None, trace_observer=observer)
    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()
    configure_compare = getattr(
        model.transcoders, "configure_phase0_activation_threshold_compare", None
    )
    if callable(configure_compare):
        configure_compare(
            mode=activation_threshold_compare_mode,
            collect_diagnostics=cross_cluster_debug_enabled,
            sample_limit_per_layer=3,
        )


def log_phase0_profile(
    *,
    logger: Any,
    model: Any,
    tokens: Phase0TokenPreparation,
    attribution: Phase0AttributionPolicy,
    settings: Phase0ProfileSettings,
) -> None:
    """Render the established Phase 0 profiling configuration message."""
    logger.info(
        "Profiling enabled | "
        f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
        f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
        f"exact_chunked_provider_enabled={settings.exact_chunked_provider_enabled} | "
        f"exact_chunked_decoder={settings.exact_chunked_decoder} | "
        f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
        f"decoder_cache_bytes={getattr(model.transcoders, 'cross_batch_decoder_cache_bytes', 0)} | "
        f"chunked_feature_replay_window={attribution.chunked_feature_replay_window} | "
        f"error_vector_prefetch_lookahead={attribution.error_vector_prefetch_lookahead} | "
        f"stage_encoder_vecs_on_cpu={attribution.stage_encoder_vecs_on_cpu} | "
        f"stage_error_vectors_on_cpu={attribution.stage_error_vectors_on_cpu} | "
        f"row_subchunk_size={attribution.row_subchunk_size} | "
        f"phase0_decoder_row_ranges={attribution.phase0_decoder_row_ranges} | "
        f"planner_enabled={settings.planner_enabled} | "
        f"feature_batch_size_max={settings.max_phase4_feature_batch_size} | "
        f"phase1_trace_batch_policy={settings.phase1_trace_batch_config.requested_policy} "
        f"(effective={settings.phase1_trace_batch_config.effective_policy}, "
        f"size_max={settings.phase1_trace_batch_config.requested_batch_size_max}, "
        f"size_max_effective={settings.phase1_trace_batch_config.effective_batch_size_max}) | "
        f"phase4_refresh_policy={settings.phase4_refresh_policy_config.requested_policy} "
        f"(effective={settings.phase4_refresh_policy_config.effective_policy}, "
        f"interval_multiplier={settings.phase4_refresh_policy_config.requested_interval_multiplier}, "
        f"interval_multiplier_effective={settings.phase4_refresh_policy_config.effective_interval_multiplier}, "
        f"queue_multiplier_effective={settings.phase4_refresh_policy_config.effective_queue_multiplier}) | "
        f"phase4_ranker={settings.phase4_ranker_config.requested_mode} "
        f"(effective={settings.phase4_ranker_config.effective_mode}) | "
        f"row_store_cache_control={settings.row_store_cache_control_config.requested_mode} "
        f"(effective={settings.row_store_cache_control_config.effective_mode}) | "
        f"exact_encoder_residency={settings.exact_encoder_residency_config.requested_mode} "
        f"(effective={settings.exact_encoder_residency_config.effective_mode}) | "
        f"exact_trace_internal_dtype={settings.exact_trace_internal_dtype_name} | "
        f"prompt_tokens={tokens.input_ids.shape[-1]} | "
        f"source_batch_size={settings.effective_source_batch_size} | "
        f"feature_batch_size={settings.effective_feature_batch_size} | "
        f"logit_batch_size={settings.effective_logit_batch_size} | "
        f"trace_batch_cap_reason={settings.phase1_trace_batch_metadata.get('trace_batch_cap_reason')}"
    )


def create_phase0_context(
    *,
    model: Any,
    tokens: Phase0TokenPreparation,
    sparsification: Any,
    context_override: Any | None,
    policy: Phase0AttributionPolicy,
    observer: TraceObserver,
) -> Any:
    """Create or reuse the attribution context before ownership transfer."""
    if context_override is not None:
        return context_override
    return model.setup_attribution(
        tokens.input_ids,
        sparsification=sparsification,
        retain_full_logits=tokens.output_position is not None
        and tokens.output_position != tokens.n_input_pos - 1,
        chunked_feature_replay_window=policy.chunked_feature_replay_window,
        error_vector_prefetch_lookahead=policy.error_vector_prefetch_lookahead,
        stage_encoder_vecs_on_cpu=policy.stage_encoder_vecs_on_cpu,
        stage_error_vectors_on_cpu=policy.stage_error_vectors_on_cpu,
        row_subchunk_size=policy.row_subchunk_size,
        exact_encoder_residency=policy.exact_encoder_residency,
        internal_precision_requested=policy.internal_precision_requested,
        resolved_dtype_map=policy.resolved_dtype_map,
        prefix_view_length=tokens.prefix_view_length,
        decoder_chunk_cache=policy.decoder_chunk_cache,
        decoder_cache_fingerprint=policy.decoder_cache_fingerprint,
        decoder_active_row_residency=policy.decoder_active_row_residency,
        decoder_active_row_max_bytes=policy.decoder_active_row_max_bytes,
        phase0_decoder_row_ranges=policy.phase0_decoder_row_ranges,
        trace_observer=observer,
    )


def configure_phase0_context(
    *,
    ctx: Any,
    logger: Any,
    observer: TraceObserver,
    profile: bool,
    capture_phase3_gradient_bundle_enabled: bool,
    phase1_trace_batch_metadata: dict[str, object],
    phase4_execution_metadata: dict[str, object],
    exact_encoder_residency_metadata: dict[str, object],
) -> Phase0ContextConfiguration:
    """Configure the created context and publish its encoder residency metadata."""
    encoder_vecs = getattr(ctx, "encoder_vecs")
    runtime_metadata = {
        "exact_encoder_staging_destination": getattr(
            ctx, "exact_encoder_staging_destination", "none"
        ),
        "exact_encoder_materialized_during_phase0": bool(
            getattr(ctx, "exact_encoder_materialized_during_phase0", False)
        ),
        "active_encoder_shape": tuple(encoder_vecs.shape),
        "active_encoder_bytes": int(encoder_vecs.numel() * encoder_vecs.element_size()),
        "exact_encoder_pinned_effective": bool(
            getattr(ctx, "exact_encoder_pinned_effective", False)
        ),
        "exact_encoder_pinning_success": getattr(ctx, "exact_encoder_pinning_success", None),
        "exact_encoder_pinning_failure_reason": getattr(
            ctx, "exact_encoder_pinning_failure_reason", None
        ),
    }
    exact_encoder_residency_metadata.update(runtime_metadata)
    phase4_execution_metadata.update(runtime_metadata)
    if hasattr(ctx, "set_diagnostic_mode"):
        ctx.set_diagnostic_mode(profile)
    if capture_phase3_gradient_bundle_enabled:
        setattr(ctx, "capture_phase3_gradients", True)
    configure_trace_logging = getattr(ctx, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(logger.info if profile else None, trace_observer=observer)
    if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
        ctx.setup_diagnostic_stats.update(
            {
                "phase1_trace_batch": dict(phase1_trace_batch_metadata),
                "phase4_execution": dict(phase4_execution_metadata),
            }
        )
    return Phase0ContextConfiguration(runtime_metadata=runtime_metadata)
