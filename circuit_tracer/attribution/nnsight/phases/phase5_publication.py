"""Phase 5 evidence recording, validation, and output publication."""

from __future__ import annotations

import time
from typing import cast

import torch

from circuit_tracer.attribution.nnsight.phases.phase5_types import Phase5Config, Phase5Inputs
from circuit_tracer.attribution.nnsight.prefix_view import validate_compact_prefix_view_output
from circuit_tracer.attribution.nnsight.telemetry import _record_cross_cluster_checkpoint
from circuit_tracer.observability.events import MemoryBoundary, RuntimeSnapshot, TraceEvent


def record_phase4_evidence(
    *, artifact: dict[str, object], inputs: Phase5Inputs, config: Phase5Config
) -> None:
    """Capture the final Phase-4 runtime/checkpoint streams into the compact artifact."""
    diagnostics = inputs.diagnostics
    summary = diagnostics.cross_cluster_debug_summary
    if summary is not None:
        summary["status"] = "captured"
        runtime_summary, runtime_stream = cast(
            tuple[dict[str, object], dict[str, object]],
            inputs.runtime.observer.observe(RuntimeSnapshot(
                inputs.runtime.model.device, context=inputs.runtime.ctx,
                transcoder=inputs.runtime.model.transcoders,
            )),
        )
        work, timings = config.phase4_work, config.phase4_timings
        counts = {
            "phase4_refresh_count": work.refresh_count,
            "phase4_batch_count": work.scheduler_reference_batch_count,
            "phase4_batches": work.scheduler_reference_batch_count,
            "phase4_executor_microbatch_count": work.executor_microbatch_count,
        }
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=summary,
            cross_cluster_debug_checkpoints=diagnostics.cross_cluster_debug_checkpoints,
            checkpoint_name="phase4_entry", phase="phase4",
            summary_payload={**counts, **diagnostics.phase4_execution_metadata, **runtime_summary},
            stream_payload={"checkpoint_stage": "post_phase4", **counts,
                            **diagnostics.phase4_execution_metadata, **runtime_stream},
        )
        _record_cross_cluster_checkpoint(
            cross_cluster_debug_summary=summary,
            cross_cluster_debug_checkpoints=diagnostics.cross_cluster_debug_checkpoints,
            checkpoint_name="phase4_run_summary", phase="phase4", summary_payload=None,
            stream_payload={
                "selected_feature_count": int(inputs.graph.visited.sum().item()),
                "phase4_feature_batch_size": work.feature_batch_size, **counts,
                "phase4_elapsed_ms": timings.elapsed_ms,
                "phase4_refresh_elapsed_ms_total": timings.refresh_elapsed_ms,
                "phase4_feature_batch_elapsed_ms_total": timings.feature_batch_elapsed_ms,
                "phase4_refresh_partial_influence_elapsed_ms_total": timings.partial_influence_elapsed_ms,
                "phase4_refresh_rank_topk_elapsed_ms_total": timings.rank_topk_elapsed_ms,
                "phase4_refresh_frontier_plan_elapsed_ms_total": timings.frontier_plan_elapsed_ms,
                **diagnostics.phase4_execution_metadata, **runtime_stream,
            },
        )
        summary["checkpoint_stream_count"] = len(diagnostics.cross_cluster_debug_checkpoints or [])
        summary["batch_event_stream_count"] = len(diagnostics.cross_cluster_debug_batches or [])
        artifact["cross_cluster_debug_summary"] = summary
    if diagnostics.cross_cluster_debug_checkpoints is not None:
        artifact["cross_cluster_debug_checkpoints"] = diagnostics.cross_cluster_debug_checkpoints
    if diagnostics.cross_cluster_debug_batches is not None:
        artifact["cross_cluster_debug_batches"] = diagnostics.cross_cluster_debug_batches


def finalize_compact_publication(
    *, artifact: dict[str, object], selected_features: torch.Tensor,
    inputs: Phase5Inputs, config: Phase5Config, phase5_start: float,
) -> torch.Tensor | None:
    """Publish identity, attach evidence, release dense ownership, validate, and finalize."""
    inputs.output.publish_compact_output_result(artifact)
    from circuit_tracer.attribution.nnsight.phases.phase5_artifacts import attach_optional_artifacts

    attach_optional_artifacts(artifact=artifact, inputs=inputs, config=config)
    record_phase4_evidence(artifact=artifact, inputs=inputs, config=config)
    edge_matrix = inputs.graph.edge_matrix
    if config.output_policy.use_compact_feature_row_store:
        assert inputs.graph.feature_row_store is not None
        store_bytes = inputs.graph.feature_row_store.nbytes
    else:
        edge_matrix = None
        inputs.output.release_dense_edge_matrix()
        store_bytes = None
    inputs.runtime.logger.info(
        "Attribution completed in "
        f"{time.time() - config.provenance.start_time:.2f}s | "
        f"compact_feature_edge_shape={tuple(artifact['feature_feature_edges'].shape)} | "
        f"compact_logit_edge_shape={tuple(artifact['logit_feature_edges'].shape)}"
        + (f" | feature_row_store_bytes={store_bytes}" if store_bytes is not None else "")
    )
    inputs.runtime.observer.observe(TraceEvent(
        scope="phase", name="phase5.packaging", phase="phase5",
        elapsed_ms=(time.perf_counter() - phase5_start) * 1000.0,
        attrs={
            "compact_output": True,
            "selected_features": int(selected_features.numel()),
            "feature_edge_rows": int(cast(torch.Tensor, artifact["feature_feature_edges"]).shape[0]),
            "feature_edge_cols": int(cast(torch.Tensor, artifact["feature_feature_edges"]).shape[1]),
        }, wall_clock=True,
    ))
    if inputs.output.prefix_view_metadata is not None:
        artifact["prefix_view_metadata"] = dict(inputs.output.prefix_view_metadata)
        validate_compact_prefix_view_output(
            artifact, n_layers=int(inputs.runtime.model.cfg.n_layers)
        )
    reused = config.provenance.phase0_context_override is not None
    artifact["phase0_window_state_reuse_requested"] = reused
    artifact["phase0_window_state_reuse_effective"] = reused
    artifact["target_logit_source"] = config.provenance.target_logit_source or (
        "override" if config.provenance.target_logits_override is not None else "context"
    )
    return edge_matrix


def finalize_full_publication(
    *, full_edge_matrix: torch.Tensor, inputs: Phase5Inputs,
    config: Phase5Config, phase5_start: float,
) -> None:
    """Emit final dense-graph memory and publication telemetry."""
    inputs.runtime.observer.observe(MemoryBoundary(
        f"Attribution completed in {time.time() - config.provenance.start_time:.2f}s",
        inputs.runtime.model.device, {"adjacency_shape": tuple(full_edge_matrix.shape)},
    ))
    inputs.runtime.observer.observe(TraceEvent(
        scope="phase", name="phase5.packaging", phase="phase5",
        elapsed_ms=(time.perf_counter() - phase5_start) * 1000.0,
        attrs={"compact_output": False,
               "adjacency_rows": int(full_edge_matrix.shape[0]),
               "adjacency_cols": int(full_edge_matrix.shape[1])},
        wall_clock=True,
    ))
