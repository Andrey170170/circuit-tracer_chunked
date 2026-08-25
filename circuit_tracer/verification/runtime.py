from typing import Protocol, runtime_checkable

from .contracts import InterventionExecutionRequest, InterventionExecutionResult


@runtime_checkable
class InterventionRuntimePort(Protocol):
    """Deep adapter seam hiding model-specific tracing and teardown mechanics."""

    def evaluate(self, request: InterventionExecutionRequest) -> InterventionExecutionResult: ...
