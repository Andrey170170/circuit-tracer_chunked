"""Artifact metadata and replay payload packaging for Phase 5."""

from __future__ import annotations

from typing import Any, cast

from circuit_tracer.attribution.nnsight.phases.phase5_compact import CompactGraphAssembly
from circuit_tracer.attribution.nnsight.phases.phase5_types import Phase5Config, Phase5Inputs


def _execution_policy_metadata(prefix: str, policy: Any) -> dict[str, object]:
    return {
        f"{prefix}_requested": policy.requested_mode,
        f"{prefix}_mode_requested": policy.requested_mode,
        f"{prefix}_effective": policy.effective_mode,
        f"{prefix}_mode_effective": policy.effective_mode,
        f"{prefix}_version": policy.version,
        f"{prefix}_version_requested": policy.version,
        f"{prefix}_effective_version": policy.effective_version,
        f"{prefix}_version_effective": policy.effective_version,
        f"{prefix}_effective_behavior": policy.effective_behavior,
        f"{prefix}_reference_execution": policy.requested_mode != policy.effective_mode,
    }


def package_compact_artifacts(
    *, assembly: CompactGraphAssembly, inputs: Phase5Inputs, config: Phase5Config
) -> dict[str, object]:
    """Attach execution, replay, timing, and capture metadata to compact graph tensors."""
    artifact = assembly.artifact
    output, batches = config.output_policy, config.batches
    policy, numerics, work, timings = (
        config.phase4_policy, config.numerics, config.phase4_work, config.phase4_timings
    )
    scheduler = policy.phase4_scheduler_config
    refresh = policy.phase4_refresh_optimization_config
    artifact.update({
        "phase4_feature_batch_size": work.feature_batch_size,
        "phase4_feature_batch_size_initial": batches.batch_size if batches.feature_batch_size is None else batches.feature_batch_size,
        "phase4_feature_batch_size_max": batches.max_phase4_feature_batch_size,
        "phase4_feature_batch_planner_enabled": batches.planner_enabled,
        "phase4_feature_batch_planner_status": batches.planner_status,
        "phase4_feature_batch_planner_skip_reason": batches.planner_skip_reason,
        "phase4_scheduler_requested_mode": scheduler.requested_mode,
        "phase4_scheduler_mode": scheduler.requested_mode,
        "phase4_scheduler_mode_requested": scheduler.requested_mode,
        "phase4_scheduler_version": scheduler.version,
        "phase4_scheduler_version_requested": scheduler.version,
        "phase4_scheduler_policy": scheduler.policy,
        "phase4_scheduler_policy_requested": scheduler.policy,
        "phase4_scheduler_effective_mode": scheduler.effective_mode,
        "phase4_scheduler_mode_effective": scheduler.effective_mode,
        "phase4_scheduler_effective_version": scheduler.effective_version,
        "phase4_scheduler_version_effective": scheduler.effective_version,
        "phase4_scheduler_effective_policy": scheduler.effective_policy,
        "phase4_scheduler_policy_effective": scheduler.effective_policy,
        "phase4_scheduler_effective_behavior": scheduler.effective_behavior,
        "phase4_scheduler_reference_execution": scheduler.requested_mode != scheduler.effective_mode,
        "phase4_scheduler_debug": bool(scheduler.debug),
        "phase4_scheduler_telemetry_detail": scheduler.telemetry_detail,
        "phase4_refresh_optimization": refresh.requested_mode,
        "phase4_refresh_optimization_requested": refresh.requested_mode,
        "phase4_refresh_optimization_mode_requested": refresh.requested_mode,
        "phase4_refresh_optimization_effective": refresh.effective_mode,
        "phase4_refresh_optimization_mode_effective": refresh.effective_mode,
        "phase4_refresh_optimization_version": refresh.version,
        "phase4_refresh_optimization_version_requested": refresh.version,
        "phase4_refresh_optimization_effective_version": refresh.effective_version,
        "phase4_refresh_optimization_version_effective": refresh.effective_version,
        "phase4_refresh_optimization_effective_behavior": refresh.effective_behavior,
        "phase4_refresh_optimization_reference_execution": refresh.requested_mode != refresh.effective_mode,
        "phase4_refresh_prepared_chunk_cache_bytes_requested": policy.prepared_chunk_cache_bytes,
        "phase4_refresh_prepared_chunk_cache_bytes_effective": policy.prepared_chunk_cache_bytes_effective,
        "phase4_refresh_prepared_chunk_cache_enabled": policy.prepared_chunk_cache_bytes_effective > 0,
        "phase4_refresh_active_row_accumulation_requested": policy.active_row_accumulation,
        "phase4_refresh_active_row_accumulation_effective": policy.active_row_accumulation_effective,
        "phase4_refresh_active_row_accumulation_fallback_reason": policy.refresh_aux_fallback_reason,
        "phase4_refresh_active_row_accumulation_applicable": policy.refresh_aux_applicable,
        **_execution_policy_metadata("phase4_row_executor", policy.phase4_row_executor_config),
        **_execution_policy_metadata("phase4_row_reduction", policy.phase4_row_reduction_config),
        **{f"phase1_{key}": value for key, value in batches.phase1_trace_batch_metadata.items()},
        "phase4_executor_configured_reference_batch_size": work.executor_reference_batch_size,
        "phase4_executor_reference_batch_size": work.executor_reference_batch_size,
        "phase4_executor_microbatch_size": work.executor_microbatch_size,
        "internal_precision_requested": numerics.internal_precision_requested,
        "resolved_dtype_map": numerics.resolved_dtype_map,
        "phase4_anomaly_debug_enabled": output.phase4_anomaly_debug_enabled,
        "cross_cluster_debug_enabled": output.cross_cluster_debug_enabled,
        "capture_phase0_donor_bundle_enabled": output.capture_phase0_donor_bundle,
        "capture_phase3_seed_bundle_enabled": output.capture_phase3_seed_bundle,
        "capture_phase3_gradient_bundle_enabled": output.capture_phase3_gradient_bundle,
        "capture_phase3_row_bundle_enabled": output.capture_phase3_row_bundle,
        "capture_feature_semantic_descriptors_enabled": output.capture_feature_semantic_descriptors,
        "semantic_descriptor_top_k": work.semantic_descriptor_top_k,
        "semantic_descriptor_dim": work.semantic_descriptor_dim,
        "phase4_refresh_count": work.refresh_count,
        "phase3_frontier_buffer_metadata": inputs.diagnostics.phase3_frontier_buffer_metadata,
        "phase4_frontier_buffer_metadata": inputs.diagnostics.phase4_frontier_buffer_metadata,
        "phase4_batch_count": work.scheduler_reference_batch_count,
        "phase4_batches": work.scheduler_reference_batch_count,
        "phase4_executor_microbatch_count": work.executor_microbatch_count,
        "phase4_refresh_elapsed_seconds_total": round(timings.refresh_elapsed_ms / 1000.0, 6),
        "phase4_feature_batch_elapsed_seconds_total": round(timings.feature_batch_elapsed_ms / 1000.0, 6),
        "phase4_refresh_partial_influence_elapsed_seconds_total": round(timings.partial_influence_elapsed_ms / 1000.0, 6),
        "phase4_refresh_rank_topk_elapsed_seconds_total": round(timings.rank_topk_elapsed_ms / 1000.0, 6),
        "phase4_refresh_frontier_plan_elapsed_seconds_total": round(timings.frontier_plan_elapsed_ms / 1000.0, 6),
        "phase4_refresh_row_store_read_elapsed_seconds_total": round(timings.row_store_read_elapsed_ms / 1000.0, 6),
        "exact_trace_internal_dtype": numerics.exact_dtype_name,
        "phase0_activation_threshold_compare_mode": numerics.activation_compare_mode,
        "telemetry_max_events": numerics.telemetry_max_events,
        "cfg": inputs.runtime.model.config, "scan": inputs.runtime.model.scan,
    })
    for prefix, metadata in (
        ("phase0_replay", inputs.replay.phase0_replay_metadata),
        ("phase3_gradient_replay", inputs.replay.phase3_gradient_replay_metadata),
        ("phase3_row_replay", inputs.replay.phase3_row_replay_metadata),
    ):
        artifact[f"{prefix}_mode"] = metadata.get("mode")
        artifact[f"{prefix}_status"] = metadata.get("status")
        artifact[f"{prefix}_donor_bundle_path"] = metadata.get("donor_bundle_path")
        artifact[f"{prefix}_donor_bundle_basename"] = metadata.get("donor_bundle_basename")
    artifact["phase0_replay_context_policy"] = inputs.replay.phase0_replay_metadata.get("context_policy")
    artifact["phase0_replay_validation_warning_count"] = inputs.replay.phase0_replay_metadata.get("validation_warning_count")
    artifact["phase0_replay_validation_warnings"] = inputs.replay.phase0_replay_metadata.get("validation_warnings")
    artifact["phase0_replay_dtype_roundtrip_loss"] = cast(dict[str, object], inputs.replay.phase0_replay_metadata.get("dtype_metadata", {})).get("dtype_roundtrip_loss")
    for prefix, metadata in (("phase3_gradient_replay", inputs.replay.phase3_gradient_replay_metadata), ("phase3_row_replay", inputs.replay.phase3_row_replay_metadata)):
        artifact[f"{prefix}_validation_failure_count"] = metadata.get("validation_failure_count")
        artifact[f"{prefix}_error"] = metadata.get("error")
    artifact["phase3_row_replay_source"] = inputs.replay.phase3_row_replay_metadata.get("source")
    payload = inputs.replay.phase0_donor_bundle_payload
    artifact["phase0_donor_bundle_schema_version"] = int(payload.get("schema_version", 1)) if isinstance(payload, dict) else 1
    artifact["phase0_donor_bundle_replay_kind"] = str(payload.get("replay_kind", "phase0_active_features_v1")) if isinstance(payload, dict) else "phase0_active_features_v1"
    artifact["phase0_donor_bundle_status"] = str(payload.get("status", "captured")) if isinstance(payload, dict) else None
    return artifact


def attach_optional_artifacts(*, artifact: dict[str, object], inputs: Phase5Inputs, config: Phase5Config) -> None:
    """Attach nested replay, donor, descriptor, and frontier payloads after publication."""
    replay, policy = inputs.replay, config.output_policy
    artifact.update(
        phase0_replay_metadata=replay.phase0_replay_metadata,
        phase3_gradient_replay_metadata=replay.phase3_gradient_replay_metadata,
        phase3_row_replay_metadata=replay.phase3_row_replay_metadata,
        phase3_frontier_buffer_metadata=inputs.diagnostics.phase3_frontier_buffer_metadata,
        phase4_frontier_buffer_metadata=inputs.diagnostics.phase4_frontier_buffer_metadata,
    )
    optional = (
        (policy.capture_phase0_donor_bundle, "phase0_donor_bundle", replay.phase0_donor_bundle_payload),
        (policy.capture_phase3_seed_bundle, "phase3_seed_bundle", replay.phase3_seed_bundle_payload),
        (policy.capture_phase3_gradient_bundle, "phase3_gradient_bundle", replay.phase3_gradient_bundle_payload),
        (policy.capture_phase3_row_bundle, "phase3_row_bundle", replay.phase3_row_bundle_payload),
        (policy.capture_feature_semantic_descriptors, "feature_semantic_descriptors", inputs.diagnostics.feature_semantic_descriptors_payload),
    )
    for enabled, name, value in optional:
        if enabled:
            artifact[name] = value
