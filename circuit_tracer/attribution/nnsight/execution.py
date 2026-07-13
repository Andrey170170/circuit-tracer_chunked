"""Phase operations and explicit execution state for NNSight attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from circuit_tracer.attribution.nnsight.phases.phase0 import (
    Phase0CleanupOwner,
    Phase0Config,
    Phase0ExecutionError,
    Phase0Inputs,
    Phase0Result,
)
from circuit_tracer.attribution.nnsight.phases.phase2 import (
    Phase2Config,
    Phase2Inputs,
    Phase2ResourceOwner,
    Phase2Result,
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
    Phase5Config,
    Phase5Inputs,
    Phase5Result,
)
from circuit_tracer.attribution.nnsight.preparation import PreparedBackend
from circuit_tracer.attribution.nnsight.run_scope import AttributionRunScope
from circuit_tracer.utils.disk_offload import offload_modules


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

    def run(self) -> Any:
        """Execute the canonical attribution lifecycle in readable domain order."""
        self.run_phase0_preparation()
        self.run_forward_pass()
        self.setup_active_features_and_storage()
        self.attribute_seed_nodes()
        self.expand_feature_frontier()
        return self.assemble_graph()

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
                    telemetry_recorder=p.diagnostics.recorder,
                    phase0_context_override=p.forward_overrides.phase0_context,
                    prefix_view_metadata=p.prefix_view_metadata,
                    exact_encoder_residency_metadata=(
                        p.frontier.exact_encoder_residency_metadata
                    ),
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
                    internal_precision_requested=p.numerics.internal_precision_requested,
                    resolved_dtype_map=p.numerics.dtype_map,
                    decoder_chunk_cache=p.forward_overrides.decoder_chunk_cache,
                    decoder_cache_fingerprint=p.forward_overrides.decoder_cache_fingerprint,
                    capture_phase3_gradient_bundle_enabled=(
                        policy.capture_phase3_gradient_bundle
                    ),
                    diagnostic_feature_cap=plan.semantics.diagnostic_feature_cap,
                ),
            )
        except Phase0ExecutionError as exc:
            self.scope.ctx = exc.ctx
            raise exc.cause
        self.scope.ctx = self.phase0.ctx

    def run_forward_pass(self) -> None:
        p = self.prepared
        phase0 = self._phase0()
        model = p.problem.model
        execution = p.plan.execution
        if execution.offload and not model.skip_transcoder and not p.provider.exact_chunked:
            p.offload_handles += offload_modules(model.transcoders, execution.offload)
        self.operations.run_phase1(
            logger=p.logger,
            model=model,
            ctx=phase0.ctx,
            trace_input_ids=phase0.trace_input_ids,
            trace_batch_size=p.batches.trace_batch_size,
            trace_batch_config=p.batches.phase1_config,
            trace_batch_metadata=p.batches.phase1_metadata,
            effective_source_batch_size=p.batches.source_batch_size,
            effective_feature_batch_size=p.batches.feature_batch_size,
            effective_logit_batch_size=p.batches.logit_batch_size,
            telemetry_observer=p.diagnostics.observer,
        )
        if execution.offload:
            p.offload_handles += offload_modules(
                [layer.mlp for layer in getattr(model.pre_logit_location, "layers")],
                execution.offload,
            )
            if model.skip_transcoder and not p.provider.exact_chunked:
                p.offload_handles += offload_modules(model.transcoders, execution.offload)

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
                telemetry_recorder=p.diagnostics.recorder,
                cross_cluster_debug_summary=p.diagnostics.cross_cluster_summary,
                cross_cluster_debug_checkpoints=p.diagnostics.cross_cluster_checkpoints,
                offload_handles=p.offload_handles,
                attribution_targets=p.problem.targets,
                target_logits_override=p.forward_overrides.target_logits,
                resource_owner=owner,
            ),
            config=Phase2Config(
                output_position=phase0.output_position,
                n_input_pos=phase0.n_input_pos,
                max_n_logits=p.problem.max_n_logits,
                desired_logit_prob=p.problem.desired_logit_prob,
                phase0_replay_mode_resolved=p.replay.phase0_mode,
                phase0_donor_bundle_path=p.replay.phase0_bundle_path,
                phase0_donor_context_policy_resolved=p.replay.phase0_context_policy,
                capture_phase0_donor_bundle_enabled=(
                    plan.execution.observability.capture_phase0_donor_bundle
                ),
                offload=plan.execution.offload,
                max_feature_nodes=plan.semantics.max_feature_nodes,
                phase3_frontier_buffer_relative_epsilon=frontier.phase3_buffer_relative_epsilon,
                phase3_frontier_buffer_max_extra=frontier.phase3_buffer_max_extra,
                phase4_frontier_buffer_relative_epsilon=frontier.phase4_buffer_relative_epsilon,
                phase4_frontier_buffer_max_extra_per_refresh=(
                    frontier.phase4_buffer_max_extra_per_refresh
                ),
                phase4_frontier_buffer_max_extra_total=frontier.phase4_buffer_max_extra_total,
                compact_output=plan.execution.compact_output,
                exact_chunked_decoder=p.provider.exact_chunked,
                use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                exact_trace_internal_dtype_resolved=p.numerics.exact_dtype,
                phase4_refresh_prepared_chunk_cache_bytes_effective=(
                    p.frontier.prepared_chunk_cache_bytes_effective
                ),
                row_store_cache_control_config=p.frontier.row_store_cache_control,
                row_store_temp_root_policy_resolved=_resolve_storage_temp_policy(storage),
                row_store_temp_root=storage.temp_root,
                row_store_preallocate=storage.preallocate,
                full_retention_backend=storage.full_retention_backend,
                feature_row_column_tile_size=storage.feature_column_tile_size,
                influence_row_tile_size=storage.influence_row_tile_size,
                influence_column_tile_size=storage.influence_column_tile_size,
                feature_row_retention=storage.retention,
                replay_tile_cache_bytes=int(storage.replay_tile_cache_bytes or 0),
                feature_row_storage_dtype=p.numerics.feature_row_storage_dtype,
                row_abs_sum_dtype=p.numerics.row_abs_sum_dtype,
                effective_feature_batch_size=p.batches.feature_batch_size,
                phase3_gradient_replay_mode_resolved=p.replay.phase3_gradient_mode,
                phase3_gradient_donor_bundle_path=p.replay.phase3_gradient_bundle_path,
                phase3_replay_validation_policy_resolved=p.replay.phase3_validation_policy,
                trace_batch_size=p.batches.trace_batch_size,
                phase3_row_replay_mode_resolved=p.replay.phase3_row_mode,
                phase3_row_donor_bundle_path=p.replay.phase3_row_bundle_path,
            ),
        )
        self.scope.feature_row_store = self.phase2.feature_row_store
        self.scope.nonfeature_row_store = self.phase2.nonfeature_row_store

    def attribute_seed_nodes(self) -> None:
        p = self.prepared
        plan = p.plan
        phase2 = self._phase2()
        policy = plan.execution.observability
        storage = plan.execution.storage
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
                compute_microbatch_max_rows=(
                    p.batches.session_controls.phase3_microbatch_max_rows
                ),
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
                phase3_frontier_buffer_max_extra=(
                    plan.semantics.frontier.phase3_buffer_max_extra
                ),
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
            ),
        )
        self.scope.anomaly_debug_result = self.phase3.anomaly_debug_result

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
                compute_microbatch_max_rows=(
                    p.batches.session_controls.phase4_microbatch_max_rows
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
                exact_encoder_residency_config=p.frontier.exact_encoder_residency,
                profile=policy.profile,
                profile_log_interval=policy.profile_log_interval,
                verbose=policy.verbose,
            ),
        )
        self.scope.feature_row_store = self.phase4.feature_row_store
        self.scope.nonfeature_row_store = self.phase4.nonfeature_row_store
        self.scope.anomaly_debug_result = self.phase4.anomaly_debug_result
        self.scope.cross_cluster_debug_summary = self.phase4.cross_cluster_debug_summary
        self.scope.cross_cluster_debug_checkpoints = (
            self.phase4.cross_cluster_debug_checkpoints
        )
        self.scope.cross_cluster_debug_batches = self.phase4.cross_cluster_debug_batches

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

        def publish(result: dict[str, object]) -> None:
            self.scope.compact_output_result = result

        def release_dense_matrix() -> None:
            nonlocal edge_matrix
            edge_matrix = None

        result = self.operations.run_phase5(
            inputs=Phase5Inputs(
                logger=p.logger,
                model=p.problem.model,
                ctx=phase0.ctx,
                targets=phase2.targets,
                telemetry_observer=p.diagnostics.observer,
                activation_matrix=phase2.activation_matrix,
                visited=phase4.visited,
                edge_matrix=edge_matrix,
                row_to_node_index=phase4.row_to_node_index,
                input_ids=phase0.input_ids,
                feature_row_store=phase4.feature_row_store,
                nonfeature_row_store=phase4.nonfeature_row_store,
                phase0_replay_metadata=phase2.phase0_replay_metadata,
                phase3_gradient_replay_metadata=phase2.phase3_gradient_replay_metadata,
                phase3_row_replay_metadata=phase2.phase3_row_replay_metadata,
                phase3_frontier_buffer_metadata=phase3.phase3_frontier_buffer_metadata,
                phase4_frontier_buffer_metadata=phase4.phase4_frontier_buffer_metadata,
                phase4_execution_metadata=phase4.phase4_execution_metadata,
                phase0_donor_bundle_payload=phase2.phase0_donor_bundle_payload,
                phase3_seed_bundle_payload=phase3.phase3_seed_bundle_payload,
                phase3_gradient_bundle_payload=phase3.phase3_gradient_bundle_payload,
                phase3_row_bundle_payload=phase3.phase3_row_bundle_payload,
                feature_semantic_descriptors_payload=(
                    phase3.feature_semantic_descriptors_payload
                ),
                cross_cluster_debug_summary=phase4.cross_cluster_debug_summary,
                cross_cluster_debug_checkpoints=phase4.cross_cluster_debug_checkpoints,
                cross_cluster_debug_batches=phase4.cross_cluster_debug_batches,
                prefix_view_metadata=p.prefix_view_metadata,
                publish_compact_output_result=publish,
                release_dense_edge_matrix=release_dense_matrix,
            ),
            config=Phase5Config(
                compact_output=plan.execution.compact_output,
                use_compact_feature_row_store=p.provider.use_compact_feature_row_store,
                capture_feature_semantic_descriptors_enabled=(
                    policy.capture_feature_semantic_descriptors
                ),
                capture_phase0_donor_bundle_enabled=policy.capture_phase0_donor_bundle,
                capture_phase3_seed_bundle_enabled=policy.capture_phase3_seed_bundle,
                capture_phase3_gradient_bundle_enabled=policy.capture_phase3_gradient_bundle,
                capture_phase3_row_bundle_enabled=policy.capture_phase3_row_bundle,
                cross_cluster_debug_enabled=policy.cross_cluster_debug,
                phase4_anomaly_debug_enabled=policy.phase4_anomaly_debug,
                n_pos=phase2.n_pos,
                n_logits=phase2.n_logits,
                st=phase4.st,
                total_active_feats=phase2.total_active_feats,
                total_nodes=phase2.total_nodes,
                actual_max_feature_nodes=phase4.actual_max_feature_nodes,
                batch_size=plan.semantics.source_batch_size,
                feature_batch_size=plan.semantics.feature_batch_size,
                max_phase4_feature_batch_size=p.batches.max_phase4_feature_batch_size,
                planner_enabled=p.batches.planner_enabled,
                planner_status=p.batches.planner_status,
                planner_skip_reason=p.batches.planner_skip_reason,
                phase4_scheduler_config=p.frontier.scheduler,
                phase4_refresh_optimization_config=p.frontier.refresh_optimization,
                phase4_row_executor_config=p.frontier.row_executor,
                phase4_row_reduction_config=p.frontier.row_reduction,
                phase1_trace_batch_metadata=p.batches.phase1_metadata,
                internal_precision_requested=p.numerics.internal_precision_requested,
                resolved_dtype_map=p.numerics.dtype_map,
                phase0_activation_threshold_compare_mode_resolved=(
                    p.numerics.activation_compare_mode
                ),
                exact_trace_internal_dtype_name=p.numerics.exact_dtype_name,
                telemetry_max_events_resolved=p.diagnostics.telemetry_max_events,
                semantic_descriptor_top_k=policy.semantic_descriptor_top_k,
                semantic_descriptor_dim=policy.semantic_descriptor_dim,
                phase4_feature_batch_size=phase4.phase4_feature_batch_size,
                phase4_executor_reference_batch_size=(
                    phase4.phase4_executor_reference_batch_size
                ),
                phase4_executor_microbatch_size=phase4.phase4_executor_microbatch_size,
                phase4_refresh_count=phase4.phase4_refresh_count,
                phase4_scheduler_reference_batch_count=(
                    phase4.phase4_scheduler_reference_batch_count
                ),
                phase4_executor_microbatch_count=phase4.phase4_executor_microbatch_count,
                phase4_elapsed_ms=phase4.phase4_elapsed_ms,
                phase4_refresh_elapsed_ms_total=phase4.phase4_refresh_elapsed_ms_total,
                phase4_feature_batch_elapsed_ms_total=(
                    phase4.phase4_feature_batch_elapsed_ms_total
                ),
                phase4_refresh_partial_influence_elapsed_ms_total=(
                    phase4.phase4_refresh_partial_influence_elapsed_ms_total
                ),
                phase4_refresh_rank_topk_elapsed_ms_total=(
                    phase4.phase4_refresh_rank_topk_elapsed_ms_total
                ),
                phase4_refresh_frontier_plan_elapsed_ms_total=(
                    phase4.phase4_refresh_frontier_plan_elapsed_ms_total
                ),
                phase4_refresh_row_store_read_elapsed_ms_total=(
                    phase4.phase4_refresh_row_store_read_elapsed_ms_total
                ),
                phase4_refresh_prepared_chunk_cache_bytes=(
                    frontier.refresh_prepared_chunk_cache_bytes
                ),
                phase4_refresh_prepared_chunk_cache_bytes_effective=(
                    p.frontier.prepared_chunk_cache_bytes_effective
                ),
                phase4_refresh_active_row_accumulation=(
                    frontier.refresh_active_row_accumulation
                ),
                phase4_refresh_active_row_accumulation_effective=(
                    p.frontier.active_row_accumulation_effective
                ),
                phase4_refresh_aux_fallback_reason=p.frontier.refresh_aux_fallback_reason,
                phase4_refresh_aux_applicable=p.frontier.refresh_aux_applicable,
                start_time=p.start_time,
                phase0_context_override=p.forward_overrides.phase0_context,
                target_logit_source=p.forward_overrides.target_logit_source,
                target_logits_override=p.forward_overrides.target_logits,
            ),
        )
        self.scope.compact_output_result = result.compact_output_result
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
