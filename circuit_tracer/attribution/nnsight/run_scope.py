"""Lifecycle and resource ownership for one NNSight attribution run."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup

from circuit_tracer.observability.human_logs import _log_memory_boundary


def raise_cleanup_failures(cleanup_failures: Sequence[BaseException]) -> None:
    """Raise every cleanup failure without losing cancellation-like exceptions."""
    if not cleanup_failures:
        return
    if all(isinstance(error, Exception) for error in cleanup_failures):
        raise ExceptionGroup(
            "Attribution lifecycle cleanup failed",
            [error for error in cleanup_failures if isinstance(error, Exception)],
        )
    raise BaseExceptionGroup("Attribution lifecycle cleanup failed", list(cleanup_failures))


@dataclass
class AttributionRunScope:
    """Own resources and terminal telemetry for exactly one attribution run."""

    logger: logging.Logger
    model: Any
    telemetry_observer: Any
    compact_output: bool
    profile: bool
    prefix_view_metadata: dict[str, object] | None
    log_memory_boundary: Callable[[Any, str, Any], None] = _log_memory_boundary
    run_start: float = field(default_factory=time.perf_counter)
    ctx: Any | None = None
    phase2_resource_owner: Any | None = None
    feature_row_store: Any | None = None
    nonfeature_row_store: Any | None = None
    compact_output_result: dict[str, object] | None = None
    anomaly_debug_result: dict[str, object] | None = None
    cross_cluster_debug_summary: dict[str, object] | None = None
    cross_cluster_debug_checkpoints: list[dict[str, object]] | None = None
    cross_cluster_debug_batches: list[dict[str, object]] | None = None

    def close(self, primary_error: BaseException | None) -> None:
        """Release owned resources and publish terminal lifecycle evidence."""
        cleanup_failures: list[BaseException] = []

        def attempt(action: str, callback: Callable[[], object]) -> object | None:
            try:
                return callback()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
                note = f"Phase D0 lifecycle action {action!r} failed: {cleanup_error!r}"
                if primary_error is not None:
                    try:
                        primary_error.add_note(note)
                    except BaseException:
                        pass
                logging.getLogger(__name__).error(note, exc_info=cleanup_error)
                return None

        teardown_start = time.perf_counter()
        feature_store = self.feature_row_store or getattr(
            self.phase2_resource_owner, "feature_row_store", None
        )
        nonfeature_store = self.nonfeature_row_store or getattr(
            self.phase2_resource_owner, "nonfeature_row_store", None
        )
        if feature_store is not None:
            attempt("feature row-store cleanup", feature_store.cleanup)
        if nonfeature_store is not None:
            attempt("nonfeature row-store cleanup", nonfeature_store.cleanup)
        if self.ctx is not None:
            attempt("context cleanup", self._cleanup_context)

        teardown_elapsed_ms = (time.perf_counter() - teardown_start) * 1000.0
        attempt(
            "teardown terminal-event emission",
            lambda: self.telemetry_observer.phase(
                name="teardown.cleanup",
                phase="teardown",
                elapsed_ms=teardown_elapsed_ms,
                attrs={
                    "ctx_present": self.ctx is not None,
                    "feature_row_store": feature_store is not None,
                },
                wall_clock=True,
            ),
        )
        self._close_run_event(primary_error, attempt)
        telemetry_export = attempt(
            "sink close/export",
            lambda: self.telemetry_observer.close_export(include_events=True),
        )
        if not isinstance(telemetry_export, dict):
            telemetry_export = {"summary": {}, "events": []}
        self._attach_terminal_evidence(primary_error, telemetry_export, attempt)
        if primary_error is None and cleanup_failures:
            raise_cleanup_failures(cleanup_failures)

    def _cleanup_context(self) -> None:
        self.log_memory_boundary(self.logger, "Teardown start", self.model.device)
        cleanup = getattr(self.ctx, "cleanup", None)
        if callable(cleanup):
            cleanup()
        else:
            clear_decoder_cache = getattr(self.ctx, "clear_decoder_cache", None)
            if callable(clear_decoder_cache):
                clear_decoder_cache()
        self.log_memory_boundary(self.logger, "Teardown done", self.model.device)

    def _close_run_event(
        self,
        primary_error: BaseException | None,
        attempt: Callable[[str, Callable[[], object]], object | None],
    ) -> None:
        elapsed_ms = (time.perf_counter() - self.run_start) * 1000.0
        if primary_error is None:
            attrs = {"compact_output": self.compact_output}
            name = "attribute.done"
        else:
            attrs = {
                "compact_output": self.compact_output,
                "error_type": type(primary_error).__name__,
                "error_message": str(primary_error),
            }
            name = "attribute.failed"
        attempt(
            "run terminal-event emission",
            lambda: self.telemetry_observer.run(
                name=name,
                elapsed_ms=elapsed_ms,
                attrs=attrs,
                wall_clock=True,
            ),
        )

    def _attach_terminal_evidence(
        self,
        primary_error: BaseException | None,
        telemetry_export: dict[str, object],
        attempt: Callable[[str, Callable[[], object]], object | None],
    ) -> None:
        result = self.compact_output_result
        if result is None:
            if primary_error is not None:
                attempt(
                    "exception attachment",
                    lambda: self.telemetry_observer.attach_exception(
                        primary_error, telemetry_export
                    ),
                )
            if self.profile:
                attempt(
                    "human telemetry rendering",
                    lambda: self.telemetry_observer.render_human_summary(
                        self.logger, telemetry_export
                    ),
                )
            return

        attempt(
            "result attachment",
            lambda: self.telemetry_observer.attach_compact_result(result, telemetry_export),
        )
        optional_evidence = {
            "prefix_view_metadata": self.prefix_view_metadata,
            "phase4_anomaly_debug": self.anomaly_debug_result,
            "cross_cluster_debug_summary": self.cross_cluster_debug_summary,
            "cross_cluster_debug_checkpoints": self.cross_cluster_debug_checkpoints,
            "cross_cluster_debug_batches": self.cross_cluster_debug_batches,
        }
        for key, value in optional_evidence.items():
            if value is not None and key not in result:
                result[key] = dict(value) if key == "prefix_view_metadata" else value
