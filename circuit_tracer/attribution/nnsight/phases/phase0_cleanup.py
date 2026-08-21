"""Cleanup ownership transfer for NNSight Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Phase0CleanupOwner:
    """Expose a successfully-created Phase 0 context to the orchestrator on failure."""

    ctx: Any | None = None


class Phase0ExecutionError(RuntimeError):
    """Phase 0 failed after creating a context that the orchestrator must clean up."""

    def __init__(self, ctx: Any, cause: BaseException) -> None:
        super().__init__("Phase 0 failed after attribution context creation")
        self.ctx = ctx
        self.cause = cause


def transfer_phase0_cleanup_ownership(*, owner: Phase0CleanupOwner, ctx: Any) -> None:
    """Make the newly created attribution context available for failure cleanup."""
    owner.ctx = ctx


def phase0_cleanup_error(
    *, owner: Phase0CleanupOwner, cause: BaseException
) -> Phase0ExecutionError | None:
    """Build the ownership-bearing error only after context transfer."""
    if owner.ctx is None:
        return None
    return Phase0ExecutionError(owner.ctx, cause)
