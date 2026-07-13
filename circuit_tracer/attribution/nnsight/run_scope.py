"""NNSight-specific resource ownership for one attribution execution."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


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
    """Own only NNSight resources; terminal evidence belongs to the canonical runner."""

    offload_handles: list[Callable[[], object]]
    ctx: Any | None = None
    phase2_resource_owner: Any | None = None
    feature_row_store: Any | None = None
    nonfeature_row_store: Any | None = None

    def close(self, primary_error: BaseException | None) -> None:
        cleanup_failures: list[BaseException] = []

        def attempt(action: str, callback: Callable[[], object]) -> None:
            try:
                callback()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
                note = f"NNSight resource cleanup {action!r} failed: {cleanup_error!r}"
                if primary_error is not None:
                    try:
                        primary_error.add_note(note)
                    except BaseException:
                        pass
                logging.getLogger(__name__).error(note, exc_info=cleanup_error)

        feature_store = self.feature_row_store or getattr(
            self.phase2_resource_owner, "feature_row_store", None
        )
        nonfeature_store = self.nonfeature_row_store or getattr(
            self.phase2_resource_owner, "nonfeature_row_store", None
        )
        if feature_store is not None:
            attempt("feature row store", feature_store.cleanup)
        if nonfeature_store is not None:
            attempt("nonfeature row store", nonfeature_store.cleanup)
        if self.ctx is not None:
            attempt("attribution context", self._cleanup_context)
        for index, reload_handle in enumerate(self.offload_handles):
            attempt(f"module offload handle {index}", reload_handle)
        if primary_error is None:
            raise_cleanup_failures(cleanup_failures)

    def _cleanup_context(self) -> None:
        cleanup = getattr(self.ctx, "cleanup", None)
        if callable(cleanup):
            cleanup()
            return
        clear_decoder_cache = getattr(self.ctx, "clear_decoder_cache", None)
        if callable(clear_decoder_cache):
            clear_decoder_cache()
