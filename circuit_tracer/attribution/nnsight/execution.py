"""Phase operations and explicit execution state for NNSight attribution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from circuit_tracer.execution_identity import ExecutionIdentityState
from circuit_tracer.governor.ledger import PhaseId, ResourceGrant
from circuit_tracer.governor.runtime import (
    ActiveUniverseObservation,
    PlanRevision,
    TraceGovernorRuntime,
)

from circuit_tracer.attribution.nnsight.phases.phase0 import (
    Phase0CleanupOwner,
    Phase0Config,
    Phase0ExecutionError,
    Phase0Inputs,
    Phase0Result,
)
from circuit_tracer.attribution.nnsight.phases.phase2 import (
    FeatureRowInfluencePolicy,
    FrontierBufferPolicy,
    Phase2Config,
    Phase2ExecutionPolicy,
    Phase2Inputs,
    Phase2ResourceOwner,
    Phase2Result,
    Phase0ReplayPolicy,
    Phase3ReplayPolicy,
    RowStoreLayout,
    RowStoreRuntime,
    TargetSelectionPolicy,
)
from circuit_tracer.attribution.nnsight.phases.phase3 import (
    Phase3Config,
    Phase3Inputs,
    Phase3Result,
)
from circuit_tracer.attribution.nnsight.phases.phase4 import (
    Phase4Config,
    Phase4Inputs,
    Phase4Result,
)
from circuit_tracer.attribution.nnsight.phases.phase5 import (
    BatchExecutionSummary,
    DiagnosticArtifacts,
    GraphAssemblyLimits,
    GraphAssemblyRuntime,
    GraphAssemblyState,
    GraphOutputOwnership,
    NumericExecutionSummary,
    OutputArtifactPolicy,
    Phase4PolicySummary,
    Phase4TimingSummary,
    Phase4WorkSummary,
    Phase5Config,
    Phase5Inputs,
    Phase5Result,
    ReplayArtifacts,
    RunProvenance,
)
from circuit_tracer.attribution.nnsight.preparation import (
    PreparedBackend,
    finalize_active_decoder_row_admission,
    finalize_feature_row_influence_execution,
    finalize_phase0_decoder_row_range_execution,
    reprepare_after_active_universe,
)
from circuit_tracer.attribution.nnsight.run_scope import AttributionRunScope
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.observability.events import MemoryDelta, MemorySnapshot, TraceEvent
from circuit_tracer.observability.errors import safe_exception_attrs, safe_exception_message
from circuit_tracer.diagnostic import ProbeCompletion
from circuit_tracer.transcoder.checkpoint_assets import (
    CheckpointPageLifecycle,
    CheckpointPageTelemetry,
)
from circuit_tracer.transcoder.checkpoint_working_set import PhaseWorkingSetPlan
from circuit_tracer.transcoder.provider import get_checkpoint_lifecycle_provider


def _decoder_row_execution_metadata(snapshot: dict[str, object]) -> dict[str, object]:
    """Keep active-row and Phase-0 range evidence in final execution metadata."""

    prefixes = ("decoder_active_row_", "phase0_decoder_row_ranges_")
    return {key: value for key, value in snapshot.items() if key.startswith(prefixes)}


def _memory_headroom_bytes(snapshot: object) -> int | None:
    if not isinstance(snapshot, dict):
        return None
    raw = snapshot.get("cgroup_memory_headroom_gib")
    if not isinstance(raw, (int, float)):
        return None
    return max(0, int(float(raw) * 1024**3))


def _checkpoint_page_trace_event(event: CheckpointPageTelemetry) -> TraceEvent:
    return TraceEvent(
        scope="op",
        name=f"checkpoint.page.{event.advice.value}",
        phase="phase3",
        attrs={
            "outcome": event.outcome,
            "asset_id": event.asset_id,
            "path": event.path,
            "device": event.device,
            "inode": event.inode,
            "asset_role": event.role,
            "offset": event.offset,
            "requested_bytes": event.length,
            "page_size": event.page_size,
            "page_span_offset": event.page_span_offset,
            "page_span_length": event.page_span_length,
            "kernel_effect_granularity": event.kernel_effect_granularity,
            "kernel_effect_verified": event.kernel_effect_verified,
            "supported": event.supported,
            "effective": event.effective,
            "issued": event.issued,
            "refused": event.refused,
            "attempted": event.attempted,
            "idempotent": event.idempotent,
            "reason": event.reason,
            "error": event.error,
        },
    )


@dataclass(frozen=True)
class BackendOperations:
    """Concrete phase operations, injectable at the backend compatibility seam."""

    run_phase0: Callable[..., Phase0Result]
    run_phase1: Callable[..., None]
    run_phase2: Callable[..., Phase2Result]
    run_phase3: Callable[..., Phase3Result]
    run_phase4: Callable[..., Phase4Result]
    run_phase5: Callable[..., Phase5Result]


@dataclass
class AttributionExecution:
    """Own intermediate phase results for one prepared attribution run."""

    prepared: PreparedBackend
    scope: AttributionRunScope
    operations: BackendOperations
    phase0: Phase0Result | None = None
    phase2: Phase2Result | None = None
    phase3: Phase3Result | None = None
    phase4: Phase4Result | None = None
    governor_runtime: TraceGovernorRuntime | None = None
    execution_identity: ExecutionIdentityState | None = None
    row_store_grant: ResourceGrant | None = None

    def run(self) -> Any:
        """Execute the canonical attribution lifecycle in readable domain order."""
        session_grant = self._grant(PhaseId.SESSION)
        try:
            self._run_with_grant(PhaseId.PHASE0, self.run_phase0_preparation)
            if self._diagnostic_stop_mode() == "phase0_probe":
                return ProbeCompletion(
                    mode="phase0_probe",
                    diagnostic_metadata=self._probe_diagnostic_metadata(),
                )
            self.apply_active_universe_replan()
            self._run_with_grant(PhaseId.PHASE1, self.run_forward_pass)
            self.row_store_grant = self._grant(PhaseId.PHASE2)
            self.setup_active_features_and_storage()
            self.apply_phase3_entry_replan()
            self.finalize_active_decoder_row_admission()
            self._run_with_grant(PhaseId.PHASE3, self.attribute_seed_nodes)
            if self._diagnostic_stop_mode() == "phase3_probe":
                return ProbeCompletion(
                    mode="phase3_probe",
                    diagnostic_metadata=self._probe_diagnostic_metadata(),
                    diagnostic_artifacts=self._probe_diagnostic_artifacts(),
                )
            self.apply_checkpoint_working_set_transition()
            self.apply_phase4_entry_replan()
            self._run_with_grant(PhaseId.PHASE4, self.expand_feature_frontier)
            if self._diagnostic_stop_mode() == "transition_probe":
                completed = self._phase4().phase4_execution_batch_count
                requested = self.prepared.plan.execution.diagnostic_stop.phase4_batches
                if requested is None or completed != requested:
                    raise RuntimeError(
                        "transition probe incomplete: "
                        f"requested {requested} physical Phase 4 batches, "
                        f"completed {completed}"
                    )
                return ProbeCompletion(
                    mode="transition_probe",
                    phase4_batches_completed=completed,
                    diagnostic_metadata=self._probe_diagnostic_metadata(),
                    diagnostic_artifacts=self._probe_diagnostic_artifacts(),
                )
            return self._run_with_grant(PhaseId.PHASE5, self.assemble_graph)
        finally:
            self._release(self.row_store_grant)
            self._release(session_grant)

    def _diagnostic_stop_mode(self) -> str:
        prepared = getattr(self, "prepared", None)
        plan = getattr(prepared, "plan", None)
        execution = getattr(plan, "execution", None)
        policy = getattr(execution, "diagnostic_stop", None)
        return str(getattr(policy, "mode", "none"))

    def _probe_diagnostic_metadata(self) -> Mapping[str, object]:
        """Snapshot execution diagnostics before a probe releases its resources."""

        frontier = getattr(getattr(self, "prepared", None), "frontier", None)
        metadata = getattr(frontier, "execution_metadata", {})
        return MappingProxyType(dict(metadata))

    def _probe_diagnostic_artifacts(self) -> Mapping[str, object]:
        """Retain requested bounded captures when a probe skips Phase 5."""

        phase2 = self._phase2()
        phase3 = self._phase3()
        candidates = {
            "phase0_donor_bundle": phase2.phase0_donor_bundle_payload,
            "phase3_seed_bundle": phase3.phase3_seed_bundle_payload,
            "phase3_gradient_bundle": phase3.phase3_gradient_bundle_payload,
            "phase3_row_bundle": phase3.phase3_row_bundle_payload,
            "feature_semantic_descriptors": (phase3.feature_semantic_descriptors_payload),
        }
        return MappingProxyType(
            {name: payload for name, payload in candidates.items() if payload is not None}
        )

    def apply_active_universe_replan(self) -> None:
        if self.governor_runtime is None:
            return
        observation = ActiveUniverseObservation.from_sparse_tensor(self._phase0().activation_matrix)
        revision = self.governor_runtime.active_universe_replan(observation)
        self._apply_governor_revision(revision)

    def apply_phase3_entry_replan(self) -> None:
        if self.governor_runtime is not None:
            self._apply_governor_revision(self.governor_runtime.phase3_entry_replan())

    def apply_phase4_entry_replan(self) -> None:
        if self.governor_runtime is not None:
            self._apply_governor_revision(self.governor_runtime.phase4_entry_replan())
        self._refresh_active_decoder_row_execution_metadata()

    def finalize_active_decoder_row_admission(self) -> None:
        """Resolve active-state-dependent residency before the Phase-3 grant."""

        frontier = self.prepared.frontier
        if not frontier.decoder_active_row_residency_effective:
            return
        estimated_bytes = self._phase0().ctx.estimate_active_decoder_row_bytes()
        self.prepared = finalize_active_decoder_row_admission(
            self.prepared,
            estimated_bytes=estimated_bytes,
        )
        if self.execution_identity is None:
            raise RuntimeError("active-row admission requires execution identity state")
        self.execution_identity.revise_effective(self.prepared.effective_execution)

    def _refresh_active_decoder_row_execution_metadata(self) -> None:
        if self.phase0 is None:
            return
        snapshot = self.phase0.ctx.get_diagnostic_snapshot()
        diagnostics = _decoder_row_execution_metadata(snapshot)
        self.prepared.frontier.execution_metadata.update(diagnostics)
        self.prepared = finalize_phase0_decoder_row_range_execution(
            self.prepared,
            diagnostics=diagnostics,
        )
        if self.execution_identity is not None:
            self.execution_identity.revise_effective(self.prepared.effective_execution)

    def apply_checkpoint_working_set_transition(self) -> None:
        """Safely replace raw decoder assets with sealed active-row residency."""

        # Some compatibility seams construct a partial execution to exercise
        # orchestration only. A real transition is possible only after Phase 0.
        if getattr(self, "phase0", None) is None:
            return
        ctx = self._phase0().ctx
        provider = get_checkpoint_lifecycle_provider(getattr(ctx, "decoder_provider", None))
        observer = self.prepared.diagnostics.observer
        if provider is None:
            observer.observe(
                TraceEvent(
                    scope="op",
                    name="checkpoint.working_set.skipped",
                    phase="phase3",
                    attrs={"reason": "provider_lifecycle_unavailable"},
                )
            )
            return

        try:
            active_row_bytes = int(ctx.seal_active_decoder_rows_for_checkpoint_transition())
        except RuntimeError as exc:
            observer.observe(
                TraceEvent(
                    scope="op",
                    name="checkpoint.working_set.refused",
                    phase="phase3",
                    attrs={
                        "reason": "active_decoder_rows_not_sealed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return

        before = observer.observe(MemorySnapshot(device=None))
        try:
            ctx.close_owned_decoder_resources_for_checkpoint_transition()
            provider.close_decoder_checkpoint_handles()
        except Exception as exc:
            observer.observe(
                TraceEvent(
                    scope="op",
                    name="checkpoint.working_set.refused",
                    phase="phase3",
                    attrs={
                        "reason": "owned_decoder_resource_close_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return

        snapshot = ctx.get_diagnostic_snapshot()
        retain_bytes = active_row_bytes + int(snapshot.get("active_encoder_bytes", 0))
        lifecycle_capability = provider.checkpoint_lifecycle
        plan = PhaseWorkingSetPlan.admit(
            lifecycle_capability.manifest,
            retain_bytes=retain_bytes,
            byte_budget=lifecycle_capability.prefault_budget_bytes,
            available_headroom_bytes=_memory_headroom_bytes(before),
        )
        observer.observe(
            TraceEvent(
                scope="op",
                name="checkpoint.working_set.admitted",
                phase="phase3",
                attrs={
                    "retain_bytes": plan.retain_bytes,
                    "release_requested_bytes": sum(item.length for item in plan.release),
                    "prefault_requested_bytes": plan.prefault_requested_bytes,
                    "prefault_admitted_bytes": plan.prefault_admitted_bytes,
                    "prefault_refused_bytes": plan.prefault_refused_bytes,
                    "byte_budget": plan.byte_budget,
                    "available_headroom_bytes": plan.available_headroom_bytes,
                    "fallback_reason": plan.fallback_reason,
                },
            )
        )

        page_lifecycle = CheckpointPageLifecycle(
            lifecycle_capability.manifest,
            telemetry=lambda event: observer.observe(_checkpoint_page_trace_event(event)),
        )
        for byte_range in plan.release:
            page_lifecycle.release(byte_range)
        for byte_range in plan.prefault:
            page_lifecycle.prefault(byte_range)

        after = observer.observe(MemorySnapshot(device=None))
        delta = observer.observe(
            MemoryDelta(
                before=before if isinstance(before, dict) else {},
                after=after if isinstance(after, dict) else {},
            )
        )
        observer.observe(
            TraceEvent(
                scope="op",
                name="checkpoint.working_set.transitioned",
                phase="phase3",
                attrs={
                    "release_range_count": len(plan.release),
                    "prefault_range_count": len(plan.prefault),
                    **(delta if isinstance(delta, dict) else {}),
                },
            )
        )
        self._refresh_active_decoder_row_execution_metadata()

    def _apply_governor_revision(self, revision: PlanRevision) -> None:
        from circuit_tracer.tracing.governor_bridge import recompile_governed_plan

        plan = recompile_governed_plan(self.prepared.problem, self.prepared.plan, revision.plan)
        plan = replace(
            plan,
            planning_profile=self.governor_runtime.profile,
            planning_envelope=self.governor_runtime.envelope,
            planning_workload=self.governor_runtime.workload,
            planning_requirements=self.governor_runtime.requirements,
            planning_parent_fingerprint=revision.parent_execution_fingerprint,
            planning_epoch_fingerprint=revision.execution_fingerprint,
        )
        self.prepared = reprepare_after_active_universe(self.prepared, plan)
        if self.execution_identity is None:
            raise RuntimeError("governed execution requires execution identity state")
        self.execution_identity.revise_effective(self.prepared.effective_execution)

    def _grant(self, phase: PhaseId) -> ResourceGrant | None:
        if self.governor_runtime is None:
            return None
        return self.governor_runtime.grant(phase)

    def _release(self, grant: ResourceGrant | None) -> None:
        if self.governor_runtime is not None:
            self.governor_runtime.release(grant)

    def _run_with_grant(self, phase: PhaseId, callback: Callable[[], Any]) -> Any:
        grant = self._grant(phase)
        primary_error: BaseException | None = None
        try:
            before = self._observe_diagnostic(MemorySnapshot(device=None))
            asset_role = {
                PhaseId.PHASE0: "decoder",
                PhaseId.PHASE1: "model_forward",
                PhaseId.PHASE2: "row_storage",
                PhaseId.PHASE3: "encoder_and_model",
                PhaseId.PHASE4: "decoder_encoder_model",
                PhaseId.PHASE5: "graph_assembly",
            }.get(phase, "runtime")
            context = self.prepared.plan.execution.observability.telemetry_context
            cache_state = context.get("cache_state", "unavailable")
            self._observe_diagnostic(
                TraceEvent(
                    scope="phase",
                    name=f"{phase.value}.resource_interval.start",
                    phase=phase.value,
                    attrs={
                        "asset_role": asset_role,
                        "cache_state": cache_state,
                        "cache_state_provenance": (
                            "telemetry_context" if "cache_state" in context else "unavailable"
                        ),
                    },
                )
            )
            try:
                result = callback()
            except BaseException as error:
                primary_error = error
                self._record_resource_interval_end(
                    phase=phase,
                    status="failed",
                    before=before,
                    asset_role=asset_role,
                    cache_state=cache_state,
                    cache_state_provenance=(
                        "telemetry_context" if "cache_state" in context else "unavailable"
                    ),
                    error=error,
                )
                raise
            else:
                self._record_resource_interval_end(
                    phase=phase,
                    status="done",
                    before=before,
                    asset_role=asset_role,
                    cache_state=cache_state,
                    cache_state_provenance=(
                        "telemetry_context" if "cache_state" in context else "unavailable"
                    ),
                )
                return result
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self._release(grant)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"governor {phase.value} release also failed: {cleanup_error!r}"
                )

    def _record_resource_interval_end(
        self,
        *,
        phase: PhaseId,
        status: str,
        before: object,
        asset_role: str,
        cache_state: object,
        cache_state_provenance: str,
        error: BaseException | None = None,
    ) -> None:
        after = self._observe_diagnostic(MemorySnapshot(device=None))
        delta = self._observe_diagnostic(
            MemoryDelta(
                before=before if isinstance(before, dict) else {},
                after=after if isinstance(after, dict) else {},
            )
        )
        attrs: dict[str, object] = {
            "asset_role": asset_role,
            "cache_state": cache_state,
            "cache_state_provenance": cache_state_provenance,
            **(delta if isinstance(delta, dict) else {}),
        }
        if error is not None:
            attrs.update(safe_exception_attrs(error))
        self._observe_diagnostic(
            TraceEvent(
                scope="phase",
                name=f"{phase.value}.resource_interval.{status}",
                phase=phase.value,
                attrs=attrs,
            )
        )

    def _observe_diagnostic(self, observation: object) -> object | None:
        """Keep Stage-B diagnostic sampling best-effort and non-semantic."""

        try:
            return self.prepared.diagnostics.observer.observe(observation)
        except Exception as error:
            self.prepared.logger.warning(
                "Diagnostic resource observation failed: %s: %s",
                type(error).__name__,
                safe_exception_message(error),
            )
            return None

    def run_phase0_preparation(self) -> None:
        p = self.prepared
        plan = p.plan
        replay = plan.execution.replay
        policy = plan.execution.observability
        try:
            self.phase0 = self.operations.run_phase0(
                inputs=Phase0Inputs(
                    logger=p.logger,
                    model=p.problem.model,
                    prompt=p.problem.prompt,
                    sparsification=plan.semantics.sparsification,
                    telemetry_observer=p.diagnostics.observer,
                    phase0_context_override=p.forward_overrides.phase0_context,
                    prefix_view_metadata=p.prefix_view_metadata,
                    exact_encoder_residency_metadata=(p.frontier.exact_encoder_residency_metadata),
                    phase4_execution_metadata=p.frontier.execution_metadata,
                    cross_cluster_debug_summary=p.diagnostics.cross_cluster_summary,
                    cross_cluster_debug_checkpoints=p.diagnostics.cross_cluster_checkpoints,
                    cleanup_owner=Phase0CleanupOwner(),
                ),
                config=Phase0Config(
                    output_position=p.output_position,
                    profile=policy.profile,
                    phase0_activation_threshold_compare_mode=p.numerics.activation_compare_mode,
                    cross_cluster_debug_enabled=policy.cross_cluster_debug,
                    exact_chunked_provider_enabled=p.provider.exact_chunked,
                    exact_chunked_decoder=p.provider.exact_chunked,
                    chunked_feature_replay_window=replay.feature_window,
                    error_vector_prefetch_lookahead=replay.error_vector_prefetch_lookahead,
                    stage_encoder_vecs_on_cpu=replay.stage_encoder_vecs_on_cpu,
                    stage_error_vectors_on_cpu=replay.stage_error_vectors_on_cpu,
                    row_subchunk_size=replay.decoder_contraction_tile,
                    planner_enabled=p.batches.planner_enabled,
                    max_phase4_feature_batch_size=p.batches.max_phase4_feature_batch_size,
                    phase1_trace_batch_config=p.batches.phase1_config,
                    phase1_trace_batch_metadata=p.batches.phase1_metadata,
                    phase4_refresh_policy_config=p.frontier.refresh_policy,
                    phase4_ranker_config=p.frontier.ranker,
                    row_store_cache_control_config=p.frontier.row_store_cache_control,
                    exact_encoder_residency_config=p.frontier.exact_encoder_residency,
                    exact_trace_internal_dtype_name=p.numerics.exact_dtype_name,
                    effective_source_batch_size=p.batches.source_batch_size,
                    effective_feature_batch_size=p.batches.feature_batch_size,
                    effective_logit_batch_size=p.batches.logit_batch_size,
                    backward_engine_mode=p.batches.backward_engine_mode,
                    backward_batch_capacity=p.batches.backward_batch_capacity,
                    internal_precision_requested=p.numerics.internal_precision_requested,
                    resolved_dtype_map=p.numerics.dtype_map,
                    decoder_chunk_cache=p.forward_overrides.decoder_chunk_cache,
                    decoder_cache_fingerprint=p.forward_overrides.decoder_cache_fingerprint,
                    capture_phase3_gradient_bundle_enabled=(policy.capture_phase3_gradient_bundle),
                    diagnostic_feature_cap=plan.semantics.diagnostic_feature_cap,
                    decoder_active_row_residency=(
                        p.frontier.decoder_active_row_residency_effective
                    ),
                    decoder_active_row_max_bytes=int(
                        plan.execution.frontier.decoder_active_row_max_bytes
                    ),
                    phase0_decoder_row_ranges=bool(p.frontier.phase0_decoder_row_ranges_effective),
                ),
            )
        except Phase0ExecutionError as exc:
            self.scope.ctx = exc.ctx
            raise exc.cause
        self.scope.ctx = self.phase0.ctx
        self._refresh_active_decoder_row_execution_metadata()

    def run_forward_pass(self) -> None:
        p = self.prepared
        phase0 = self._phase0()
        model = p.problem.model
        execution = p.plan.execution
        if execution.offload and not model.skip_transcoder and not p.provider.exact_chunked:
            p.offload_handles.extend(offload_modules(model.transcoders, execution.offload))
        self.operations.run_phase1(
            logger=p.logger,
            model=model,
            ctx=phase0.ctx,
            trace_input_ids=phase0.trace_input_ids,
            trace_batch_size=p.batches.forward_lane_count,
            trace_batch_config=p.batches.phase1_config,
            trace_batch_metadata=p.batches.phase1_metadata,
            effective_source_batch_size=p.batches.source_batch_size,
            effective_feature_batch_size=p.batches.feature_batch_size,
            effective_logit_batch_size=p.batches.logit_batch_size,
            telemetry_observer=p.diagnostics.observer,
        )
        if execution.offload:
            p.offload_handles.extend(
                offload_modules(
                    [layer.mlp for layer in getattr(model.pre_logit_location, "layers")],
                    execution.offload,
                )
            )
            if model.skip_transcoder and not p.provider.exact_chunked:
                p.offload_handles.extend(offload_modules(model.transcoders, execution.offload))

    def setup_active_features_and_storage(self) -> None:
        p = self.prepared
        plan = p.plan
        phase0 = self._phase0()
        storage = plan.execution.storage
        frontier = plan.semantics.frontier
        owner = Phase2ResourceOwner()
        self.scope.phase2_resource_owner = owner
        self.phase2 = self.operations.run_phase2(
            inputs=Phase2Inputs(
                logger=p.logger,
                model=p.problem.model,
                ctx=phase0.ctx,
                input_ids=phase0.input_ids,
                activation_matrix=phase0.activation_matrix,
                telemetry_observer=p.diagnostics.observer,
                cross_cluster_debug_summary=p.diagnostics.cross_cluster_summary,
                cross_cluster_debug_checkpoints=p.diagnostics.cross_cluster_checkpoints,
                offload_handles=p.offload_handles,
                attribution_targets=p.problem.targets,
                target_logits_override=p.forward_overrides.target_logits,
                resource_owner=owner,
            ),
            config=Phase2Config(
                targets=TargetSelectionPolicy(
                    output_position=phase0.output_position,
                    n_input_pos=phase0.n_input_pos,
                    max_n_logits=p.problem.max_n_logits,
                    desired_logit_prob=p.problem.desired_logit_prob,
                ),
                phase0_replay=Phase0ReplayPolicy(
                    mode=p.replay.phase0_mode,
                    donor_bundle_path=p.replay.phase0_bundle_path,
                    context_policy=p.replay.phase0_context_policy,
                    capture_bundle=(plan.execution.observability.capture_phase0_donor_bundle),
                ),
                phase3_replay=Phase3ReplayPolicy(
                    gradient_mode=p.replay.phase3_gradient_mode,
                    gradient_bundle_path=p.replay.phase3_gradient_bundle_path,
                    row_mode=p.replay.phase3_row_mode,
                    row_bundle_path=p.replay.phase3_row_bundle_path,
                    validation_policy=p.replay.phase3_validation_policy,
                ),
                frontier=FrontierBufferPolicy(
                    phase3_relative_epsilon=frontier.phase3_buffer_relative_epsilon,
                    phase3_max_extra=frontier.phase3_buffer_max_extra,
                    phase4_relative_epsilon=frontier.phase4_buffer_relative_epsilon,
                    phase4_max_extra_per_refresh=frontier.phase4_buffer_max_extra_per_refresh,
                    phase4_max_extra_total=frontier.phase4_buffer_max_extra_total,
                ),
                storage_layout=RowStoreLayout(
                    retention=storage.retention,
                    backend=storage.full_retention_backend,
                    feature_column_tile_size=storage.feature_column_tile_size,
                    influence_row_tile_size=storage.influence_row_tile_size,
                    influence_column_tile_size=storage.influence_column_tile_size,
                    feature_dtype=p.numerics.feature_row_storage_dtype,
                    row_abs_sum_dtype=p.numerics.row_abs_sum_dtype,
                ),
                storage_runtime=RowStoreRuntime(
                    cache_control=p.frontier.row_store_cache_control,
                    temp_root_policy=_resolve_storage_temp_policy(storage),
                    temp_root=storage.temp_root,
                    preallocate=storage.preallocate,
                    prepared_chunk_cache_bytes=(p.frontier.prepared_chunk_cache_bytes_effective),
                    replay_tile_cache_bytes=int(storage.replay_tile_cache_bytes or 0),
                    influence=FeatureRowInfluencePolicy(
                        mode=storage.feature_row_influence_mode,
                        requirement=storage.feature_row_influence_requirement,
                        resident_max_bytes=storage.gpu_resident_max_bytes,
                        window_max_bytes=storage.gpu_window_max_bytes,
                        safety_margin_bytes=storage.gpu_resident_safety_margin_bytes,
                        device=p.problem.model.device,
                    ),
                ),
                execution=Phase2ExecutionPolicy(
                    offload=plan.execution.offload,
                    max_feature_nodes=plan.semantics.max_feature_nodes,
                    compact_output=plan.execution.compact_output,
                    exact_chunked_decoder=p.provider.exact_chunked,
                    use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                    exact_dtype=p.numerics.exact_dtype,
                    effective_feature_batch_size=p.batches.feature_batch_size,
                    trace_batch_size=p.batches.trace_batch_size,
                    source_selection=p.problem.source_selection,
                    target_position=(
                        phase0.output_position
                        if phase0.output_position is not None
                        else phase0.n_input_pos - 1
                    ),
                ),
            ),
        )
        self.scope.feature_row_store = self.phase2.feature_row_store
        self.scope.nonfeature_row_store = self.phase2.nonfeature_row_store
        self._refresh_feature_row_influence_execution_identity()

    def _refresh_feature_row_influence_execution_identity(self) -> None:
        phase2 = self._phase2()
        feature_store = phase2.feature_row_store
        if feature_store is not None:
            storage = self.prepared.plan.execution.storage
            snapshot = feature_store.get_diagnostic_snapshot()
            self.prepared = finalize_feature_row_influence_execution(
                self.prepared,
                resolved_mode=str(
                    snapshot.get(
                        "feature_row_influence_mode_resolved",
                        storage.feature_row_influence_mode,
                    )
                ),
                reason=str(snapshot.get("gpu_row_tier_reason", "requested_mode_effective")),
            )
            if self.execution_identity is not None:
                self.execution_identity.revise_effective(self.prepared.effective_execution)

    def attribute_seed_nodes(self) -> None:
        p = self.prepared
        plan = p.plan
        phase2 = self._phase2()
        policy = plan.execution.observability
        storage = plan.execution.storage
        ctx = self._phase0().ctx
        ctx.prepare_active_decoder_rows(
            requested=p.frontier.decoder_active_row_residency_requested,
            enabled=p.frontier.decoder_active_row_residency_effective,
            max_bytes=p.frontier.decoder_active_row_max_bytes_effective,
            fallback_reason=p.frontier.decoder_active_row_fallback_reason,
            admitted_estimated_bytes=p.frontier.decoder_active_row_estimated_bytes,
        )
        self._refresh_active_decoder_row_execution_metadata()
        self.phase3 = self.operations.run_phase3(
            inputs=Phase3Inputs(
                logger=p.logger,
                model=p.problem.model,
                ctx=self._phase0().ctx,
                targets=phase2.targets,
                activation_matrix=phase2.activation_matrix,
                feat_layers=phase2.feat_layers,
                feat_pos=phase2.feat_pos,
                feat_ids=phase2.feat_ids,
                feature_row_store=phase2.feature_row_store,
                nonfeature_row_store=phase2.nonfeature_row_store,
                edge_matrix=phase2.edge_matrix,
                row_to_node_index=phase2.row_to_node_index,
                telemetry_observer=p.diagnostics.observer,
                cross_cluster_debug_summary=p.diagnostics.cross_cluster_summary,
                cross_cluster_debug_checkpoints=p.diagnostics.cross_cluster_checkpoints,
                cross_cluster_debug_batches=p.diagnostics.cross_cluster_batches,
                anomaly_debug_result=p.diagnostics.anomaly_debug_result,
                loaded_phase3_row_donor_bundle=phase2.loaded_phase3_row_donor_bundle,
                phase3_frontier_buffer_metadata=phase2.phase3_frontier_buffer_metadata,
                phase3_gradient_bundle_payload=None,
                phase3_row_bundle_payload=None,
                phase3_seed_bundle_payload=None,
                feature_semantic_descriptors_payload=None,
            ),
            config=Phase3Config(
                effective_logit_batch_size=p.batches.logit_batch_size,
                compute_microbatch_max_rows=(p.batches.session_controls.phase3_microbatch_max_rows),
                effective_feature_batch_size=p.batches.feature_batch_size,
                output_position=self._phase0().output_position,
                n_layers=phase2.n_layers,
                n_pos=phase2.n_pos,
                n_logits=phase2.n_logits,
                logit_offset=phase2.logit_offset,
                total_active_feats=phase2.total_active_feats,
                base_max_feature_nodes=phase2.base_max_feature_nodes,
                actual_max_feature_nodes=phase2.actual_max_feature_nodes,
                exact_trace_internal_dtype_resolved=p.numerics.exact_dtype,
                phase3_gradient_replay_mode_resolved=p.replay.phase3_gradient_mode,
                phase3_row_replay_mode_resolved=p.replay.phase3_row_mode,
                capture_phase3_gradient_bundle_enabled=policy.capture_phase3_gradient_bundle,
                capture_phase3_row_bundle_enabled=policy.capture_phase3_row_bundle,
                capture_phase3_seed_bundle_enabled=policy.capture_phase3_seed_bundle,
                capture_feature_semantic_descriptors_enabled=(
                    policy.capture_feature_semantic_descriptors
                ),
                phase3_frontier_buffer_relative_epsilon=(
                    plan.semantics.frontier.phase3_buffer_relative_epsilon
                ),
                phase3_frontier_buffer_max_extra=(plan.semantics.frontier.phase3_buffer_max_extra),
                update_interval=plan.semantics.update_interval,
                planner_compute_dtype=p.numerics.planner_compute_dtype,
                influence_compute_dtype=p.numerics.influence_compute_dtype,
                shadow_debug_compute_dtype=p.numerics.shadow_debug_compute_dtype,
                phase4_refresh_policy_config=p.frontier.refresh_policy,
                exact_chunked_decoder=p.provider.exact_chunked,
                use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                semantic_descriptor_top_k=policy.semantic_descriptor_top_k,
                semantic_descriptor_dim=policy.semantic_descriptor_dim,
                profile=policy.profile,
                profile_log_interval=policy.profile_log_interval,
                full_retention_backend=storage.full_retention_backend,
                influence_row_tile_size=storage.influence_row_tile_size,
                influence_column_tile_size=storage.influence_column_tile_size,
                feature_row_column_tile_size=storage.feature_column_tile_size,
                feature_row_retention=storage.retention,
                eligible_feature_indices=phase2.eligible_feature_indices,
            ),
        )

    def expand_feature_frontier(self) -> None:
        p = self.prepared
        plan = p.plan
        phase2 = self._phase2()
        phase3 = self._phase3()
        policy = plan.execution.observability
        storage = plan.execution.storage
        frontier = plan.semantics.frontier
        self.phase4 = self.operations.run_phase4(
            inputs=Phase4Inputs(
                logger=p.logger,
                model=p.problem.model,
                ctx=self._phase0().ctx,
                targets=phase2.targets,
                edge_matrix=phase2.edge_matrix,
                feat_ids=phase2.feat_ids,
                feat_layers=phase2.feat_layers,
                feat_pos=phase2.feat_pos,
                feature_row_store=phase2.feature_row_store,
                nonfeature_row_store=phase2.nonfeature_row_store,
                row_to_node_index=phase3.row_to_node_index,
                telemetry_observer=p.diagnostics.observer,
                cross_cluster_debug_summary=p.diagnostics.cross_cluster_summary,
                cross_cluster_debug_checkpoints=p.diagnostics.cross_cluster_checkpoints,
                cross_cluster_debug_batches=p.diagnostics.cross_cluster_batches,
                anomaly_debug_result=phase3.anomaly_debug_result,
                phase4_frontier_buffer_metadata=phase2.phase4_frontier_buffer_metadata,
                phase4_execution_metadata=p.frontier.execution_metadata,
                rows_cpu_staging=phase3.rows_cpu_staging,
            ),
            config=Phase4Config(
                actual_max_feature_nodes=phase3.actual_max_feature_nodes,
                total_active_feats=phase2.total_active_feats,
                n_logits=phase2.n_logits,
                logit_offset=phase2.logit_offset,
                effective_feature_batch_size=p.batches.feature_batch_size,
                execution_batch_max_rows=(
                    p.batches.session_controls.phase4_execution_batch_max_rows
                ),
                max_phase4_feature_batch_size=p.batches.max_phase4_feature_batch_size,
                update_interval=plan.semantics.update_interval,
                row_store_capacity_feature_nodes=phase2.row_store_capacity_feature_nodes,
                exact_trace_internal_dtype_resolved=p.numerics.exact_dtype,
                influence_compute_dtype=p.numerics.influence_compute_dtype,
                shadow_debug_compute_dtype=p.numerics.shadow_debug_compute_dtype,
                exact_chunked_decoder=p.provider.exact_chunked,
                use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                planner_enabled=p.batches.planner_enabled,
                planner_status=p.batches.planner_status,
                planner_skip_reason=p.batches.planner_skip_reason,
                phase4_debug_summary_enabled=(
                    policy.phase4_anomaly_debug or policy.cross_cluster_debug
                ),
                cross_cluster_debug_enabled=policy.cross_cluster_debug,
                phase4_frontier_buffer_relative_epsilon=frontier.phase4_buffer_relative_epsilon,
                phase4_frontier_buffer_max_extra_per_refresh=(
                    frontier.phase4_buffer_max_extra_per_refresh
                ),
                phase4_frontier_buffer_max_extra_total=frontier.phase4_buffer_max_extra_total,
                phase4_refresh_prepared_chunk_cache_bytes_effective=(
                    p.frontier.prepared_chunk_cache_bytes_effective
                ),
                phase4_refresh_active_row_accumulation_effective=(
                    p.frontier.active_row_accumulation_effective
                ),
                phase4_scheduler_config=p.frontier.scheduler,
                phase4_refresh_optimization_config=p.frontier.refresh_optimization,
                phase4_refresh_policy_config=p.frontier.refresh_policy,
                phase4_ranker_config=p.frontier.ranker,
                phase4_row_executor_config=p.frontier.row_executor,
                phase4_row_reduction_config=p.frontier.row_reduction,
                row_store_cache_control_config=p.frontier.row_store_cache_control,
                full_retention_backend=storage.full_retention_backend,
                influence_row_tile_size=storage.influence_row_tile_size,
                influence_column_tile_size=storage.influence_column_tile_size,
                feature_row_column_tile_size=storage.feature_column_tile_size,
                feature_row_retention=storage.retention,
                feature_vjp_tape_batch_window=(p.frontier.feature_vjp_tape_batch_window_effective),
                feature_vjp_tape_max_bytes=(p.frontier.feature_vjp_tape_max_bytes_effective),
                feature_vjp_tape_enabled=p.frontier.feature_vjp_tape_enabled,
                feature_vjp_tape_fallback_reason=(p.frontier.feature_vjp_tape_fallback_reason),
                decoder_page_prefetch_depth=(p.frontier.decoder_page_prefetch_depth_effective),
                diagnostic_stop_after_batches=(
                    plan.execution.diagnostic_stop.phase4_batches
                    if plan.execution.diagnostic_stop.mode == "transition_probe"
                    else None
                ),
                cache_state=str(policy.telemetry_context.get("cache_state", "unavailable")),
                cache_state_provenance=(
                    "telemetry_context"
                    if "cache_state" in policy.telemetry_context
                    else "unavailable"
                ),
                exact_encoder_residency_config=p.frontier.exact_encoder_residency,
                profile=policy.profile,
                profile_log_interval=policy.profile_log_interval,
                verbose=policy.verbose,
                eligible_feature_indices=phase2.eligible_feature_indices,
            ),
        )
        self.scope.feature_row_store = self.phase4.feature_row_store
        self.scope.nonfeature_row_store = self.phase4.nonfeature_row_store
        self._refresh_feature_row_influence_execution_identity()

    def assemble_graph(self) -> Any:
        p = self.prepared
        plan = p.plan
        phase0 = self._phase0()
        phase2 = self._phase2()
        phase3 = self._phase3()
        phase4 = self._phase4()
        policy = plan.execution.observability
        frontier = plan.execution.frontier
        edge_matrix = phase4.edge_matrix

        def release_dense_matrix() -> None:
            nonlocal edge_matrix
            edge_matrix = None

        result = self.operations.run_phase5(
            inputs=Phase5Inputs(
                runtime=GraphAssemblyRuntime(
                    logger=p.logger,
                    model=p.problem.model,
                    ctx=phase0.ctx,
                    targets=phase2.targets,
                    observer=p.diagnostics.observer,
                    input_ids=phase0.input_ids,
                ),
                graph=GraphAssemblyState(
                    activation_matrix=phase2.activation_matrix,
                    visited=phase4.visited,
                    edge_matrix=edge_matrix,
                    row_to_node_index=phase4.row_to_node_index,
                    feature_row_store=phase4.feature_row_store,
                    nonfeature_row_store=phase4.nonfeature_row_store,
                ),
                replay=ReplayArtifacts(
                    phase0_replay_metadata=phase2.phase0_replay_metadata,
                    phase3_gradient_replay_metadata=phase2.phase3_gradient_replay_metadata,
                    phase3_row_replay_metadata=phase2.phase3_row_replay_metadata,
                    phase0_donor_bundle_payload=phase2.phase0_donor_bundle_payload,
                    phase3_seed_bundle_payload=phase3.phase3_seed_bundle_payload,
                    phase3_gradient_bundle_payload=phase3.phase3_gradient_bundle_payload,
                    phase3_row_bundle_payload=phase3.phase3_row_bundle_payload,
                ),
                diagnostics=DiagnosticArtifacts(
                    phase3_frontier_buffer_metadata=phase3.phase3_frontier_buffer_metadata,
                    phase4_frontier_buffer_metadata=phase4.phase4_frontier_buffer_metadata,
                    phase4_execution_metadata=phase4.phase4_execution_metadata,
                    feature_semantic_descriptors_payload=(
                        phase3.feature_semantic_descriptors_payload
                    ),
                    cross_cluster_debug_summary=phase4.cross_cluster_debug_summary,
                    cross_cluster_debug_checkpoints=phase4.cross_cluster_debug_checkpoints,
                    cross_cluster_debug_batches=phase4.cross_cluster_debug_batches,
                ),
                output=GraphOutputOwnership(
                    prefix_view_metadata=p.prefix_view_metadata,
                    publish_compact_output_result=lambda _result: None,
                    release_dense_edge_matrix=release_dense_matrix,
                ),
            ),
            config=Phase5Config(
                output_policy=OutputArtifactPolicy(
                    compact_output=plan.execution.compact_output,
                    use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                    capture_feature_semantic_descriptors=(
                        policy.capture_feature_semantic_descriptors
                    ),
                    capture_phase0_donor_bundle=policy.capture_phase0_donor_bundle,
                    capture_phase3_seed_bundle=policy.capture_phase3_seed_bundle,
                    capture_phase3_gradient_bundle=policy.capture_phase3_gradient_bundle,
                    capture_phase3_row_bundle=policy.capture_phase3_row_bundle,
                    cross_cluster_debug_enabled=policy.cross_cluster_debug,
                    phase4_anomaly_debug_enabled=policy.phase4_anomaly_debug,
                ),
                graph_limits=GraphAssemblyLimits(
                    n_pos=phase2.n_pos,
                    n_logits=phase2.n_logits,
                    st=phase4.st,
                    total_active_feats=phase2.total_active_feats,
                    total_nodes=phase2.total_nodes,
                    actual_max_feature_nodes=phase4.actual_max_feature_nodes,
                ),
                batches=BatchExecutionSummary(
                    batch_size=plan.semantics.source_batch_size,
                    feature_batch_size=plan.semantics.feature_batch_size,
                    max_phase4_feature_batch_size=p.batches.max_phase4_feature_batch_size,
                    planner_enabled=p.batches.planner_enabled,
                    planner_status=p.batches.planner_status,
                    planner_skip_reason=p.batches.planner_skip_reason,
                    phase1_trace_batch_metadata=p.batches.phase1_metadata,
                ),
                phase4_policy=Phase4PolicySummary(
                    phase4_scheduler_config=p.frontier.scheduler,
                    phase4_refresh_optimization_config=p.frontier.refresh_optimization,
                    phase4_row_executor_config=p.frontier.row_executor,
                    phase4_row_reduction_config=p.frontier.row_reduction,
                    prepared_chunk_cache_bytes=frontier.refresh_prepared_chunk_cache_bytes,
                    prepared_chunk_cache_bytes_effective=(
                        p.frontier.prepared_chunk_cache_bytes_effective
                    ),
                    active_row_accumulation=frontier.refresh_active_row_accumulation,
                    active_row_accumulation_effective=(
                        p.frontier.active_row_accumulation_effective
                    ),
                    refresh_aux_fallback_reason=p.frontier.refresh_aux_fallback_reason,
                    refresh_aux_applicable=p.frontier.refresh_aux_applicable,
                ),
                numerics=NumericExecutionSummary(
                    internal_precision_requested=p.numerics.internal_precision_requested,
                    resolved_dtype_map=p.numerics.dtype_map,
                    activation_compare_mode=p.numerics.activation_compare_mode,
                    exact_dtype_name=p.numerics.exact_dtype_name,
                    telemetry_max_events=p.diagnostics.telemetry_max_events,
                ),
                phase4_work=Phase4WorkSummary(
                    semantic_descriptor_top_k=policy.semantic_descriptor_top_k,
                    semantic_descriptor_dim=policy.semantic_descriptor_dim,
                    feature_batch_size=phase4.phase4_feature_batch_size,
                    semantic_batch_max_rows=phase4.phase4_semantic_batch_max_rows,
                    execution_batch_max_rows=phase4.phase4_execution_batch_max_rows,
                    refresh_count=phase4.phase4_refresh_count,
                    scheduler_reference_batch_count=(phase4.phase4_scheduler_reference_batch_count),
                    execution_batch_count=phase4.phase4_execution_batch_count,
                ),
                phase4_timings=Phase4TimingSummary(
                    elapsed_ms=phase4.phase4_elapsed_ms,
                    refresh_elapsed_ms=phase4.phase4_refresh_elapsed_ms_total,
                    feature_batch_elapsed_ms=phase4.phase4_feature_batch_elapsed_ms_total,
                    partial_influence_elapsed_ms=(
                        phase4.phase4_refresh_partial_influence_elapsed_ms_total
                    ),
                    rank_topk_elapsed_ms=phase4.phase4_refresh_rank_topk_elapsed_ms_total,
                    frontier_plan_elapsed_ms=(phase4.phase4_refresh_frontier_plan_elapsed_ms_total),
                    row_store_read_elapsed_ms=(
                        phase4.phase4_refresh_row_store_read_elapsed_ms_total
                    ),
                ),
                provenance=RunProvenance(
                    start_time=p.start_time,
                    phase0_context_override=p.forward_overrides.phase0_context,
                    target_logit_source=p.forward_overrides.target_logit_source,
                    target_logits_override=p.forward_overrides.target_logits,
                ),
            ),
        )
        return result.output

    def _phase0(self) -> Phase0Result:
        assert self.phase0 is not None
        return self.phase0

    def _phase2(self) -> Phase2Result:
        assert self.phase2 is not None
        return self.phase2

    def _phase3(self) -> Phase3Result:
        assert self.phase3 is not None
        return self.phase3

    def _phase4(self) -> Phase4Result:
        assert self.phase4 is not None
        return self.phase4


def _resolve_storage_temp_policy(storage: Any) -> Any:
    from circuit_tracer.attribution.nnsight.row_store import _resolve_row_store_temp_root_policy

    return _resolve_row_store_temp_root_policy(storage.temp_root_policy)
