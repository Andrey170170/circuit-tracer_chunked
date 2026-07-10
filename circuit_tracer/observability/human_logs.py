"""Human-readable attribution log rendering helpers."""

import time

from circuit_tracer.observability.resources import (
    diff_numeric_metrics,
    format_memory_snapshot,
    format_numeric_metrics,
)


def _log_phase_metrics(logger, label: str, phase_start: float, device, **extra):
    logger.info(
        f"{label} completed in {time.perf_counter() - phase_start:.2f}s | "
        f"{format_memory_snapshot(device=device, extra=extra)}"
    )


def _log_memory_boundary(logger, label: str, device, **extra) -> None:
    logger.info(f"{label} | {format_memory_snapshot(device=device, extra=extra)}")


def _snapshot_diagnostics(obj) -> dict[str, object] | None:
    if obj is None or not hasattr(obj, "get_diagnostic_snapshot"):
        return None
    return obj.get_diagnostic_snapshot()


def _log_batch_profile(
    logger,
    label: str,
    batch_idx: int,
    total_batches: int | None,
    elapsed: float,
    ctx_before: dict[str, object] | None,
    ctx_after: dict[str, object] | None,
    transcoder_before: dict[str, object] | None,
    transcoder_after: dict[str, object] | None,
):
    batch_progress = (
        f"batch {batch_idx}/{total_batches}" if total_batches is not None else f"batch {batch_idx}"
    )
    parts = [f"{label} {batch_progress} in {elapsed:.2f}s"]
    ctx_delta = diff_numeric_metrics(ctx_before, ctx_after) if ctx_after is not None else {}
    transcoder_delta = (
        diff_numeric_metrics(transcoder_before, transcoder_after)
        if transcoder_after is not None
        else {}
    )
    if ctx_delta:
        parts.append(f"ctx[{format_numeric_metrics(ctx_delta, limit=12)}]")
    if transcoder_delta:
        parts.append(f"transcoder[{format_numeric_metrics(transcoder_delta, limit=12)}]")
    logger.info(" | ".join(parts))


def _log_sparsification_profile(logger, stats: dict[str, object]) -> None:
    retained = stats.get("per_layer_retained_counts", {})
    logger.info(
        "Sparsification screening | "
        f"candidates={stats.get('candidate_count_before')}->{stats.get('candidate_count_after')} | "
        f"per_layer_position_topk={stats.get('per_layer_position_topk')} | "
        f"global_cap={stats.get('global_cap')} | "
        f"retained_activation_mass={stats.get('retained_activation_mass', 1.0):.4f} | "
        f"screen_seconds={stats.get('screen_seconds', 0.0):.4f} | "
        f"per_layer_retained={retained}"
    )
