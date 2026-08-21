"""Phase 1 forward-pass execution for NNSight attribution."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol

import torch

from circuit_tracer.observability.events import (
    MemoryBoundary,
    PhaseMetrics,
    TraceEvent,
    TraceObserver,
)


class _Logger(Protocol):
    def info(self, message: object, *args: object, **kwargs: object) -> None: ...


class _Phase1Model(Protocol):
    device: torch.device | str


class _Phase1Context(Protocol):
    def run_forward_pass(
        self,
        model: _Phase1Model,
        trace_input_ids: torch.Tensor,
        *,
        trace_batch_size: int,
    ) -> None: ...


class _Phase1TraceBatchConfig(Protocol):
    requested_policy: str
    effective_policy: str
    requested_batch_size_max: int | None
    effective_batch_size_max: int | None
    effective_behavior: str


def _run_phase1_forward_pass(
    *,
    logger: _Logger,
    model: _Phase1Model,
    ctx: _Phase1Context,
    trace_input_ids: torch.Tensor,
    trace_batch_size: int,
    trace_batch_config: _Phase1TraceBatchConfig,
    trace_batch_metadata: Mapping[str, object],
    effective_source_batch_size: int,
    effective_feature_batch_size: int,
    effective_logit_batch_size: int,
    telemetry_observer: TraceObserver,
) -> None:
    """Run Phase 1 while preserving its logging and telemetry contract."""
    logger.info("Phase 1: Running forward pass")
    backward_detail = ""
    if "backward_engine_mode" in trace_batch_metadata:
        backward_detail = (
            f"backward_engine_mode={trace_batch_metadata['backward_engine_mode']} | "
            f"backward_batch_capacity={trace_batch_metadata.get('backward_batch_capacity')} | "
            f"forward_lane_count={trace_batch_metadata.get('forward_lane_count')} | "
        )
    logger.info(
        "Phase 1 trace-batch policy | "
        f"requested_policy={trace_batch_config.requested_policy} | "
        f"effective_policy={trace_batch_config.effective_policy} | "
        f"requested_size_max={trace_batch_config.requested_batch_size_max} | "
        f"effective_size_max={trace_batch_config.effective_batch_size_max} | "
        f"effective_behavior={trace_batch_config.effective_behavior} | "
        f"source_batch_size={effective_source_batch_size} | "
        f"feature_batch_size={effective_feature_batch_size} | "
        f"logit_batch_size={effective_logit_batch_size} | "
        f"cap_reason={trace_batch_metadata.get('trace_batch_cap_reason')} | "
        f"{backward_detail}"
        f"trace_batch_size={trace_batch_size}"
    )
    phase_start = time.perf_counter()
    telemetry_observer.observe(MemoryBoundary("Phase 1 start", model.device))
    ctx.run_forward_pass(
        model,
        trace_input_ids,
        trace_batch_size=trace_batch_size,
    )
    logit_materialization = getattr(ctx, "phase1_logit_materialization_metadata", {})
    if logit_materialization:
        logger.info(
            "Phase 1 logit materialization | "
            + " | ".join(f"{key}={value}" for key, value in logit_materialization.items())
        )

    telemetry_observer.observe(PhaseMetrics("Forward pass", phase_start, model.device))
    phase1_elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
    telemetry_observer.observe(
        TraceEvent(
            scope="phase",
            name="phase1.forward_pass",
            phase="phase1",
            elapsed_ms=phase1_elapsed_ms,
            attrs={
                "trace_batch_size": int(trace_batch_size),
                **trace_batch_metadata,
                **{
                    f"logit_materialization_{key}": value
                    for key, value in logit_materialization.items()
                },
            },
            wall_clock=True,
        )
    )
