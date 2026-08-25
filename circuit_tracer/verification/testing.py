from __future__ import annotations

import math
from dataclasses import dataclass, field

from .contracts import (
    BaselineCapture,
    InterventionExecutionRequest,
    InterventionExecutionResult,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    FeatureValue,
    VariantObservation,
)


@dataclass(frozen=True)
class ScriptedVariant:
    target_value: float
    raw_target_logits: tuple[float, ...] = (0.0,)
    downstream_feature_values: tuple[FeatureValue, ...] = ()
    actual_seconds: float = 0.0
    predicted_seconds: float | None = None

    def __post_init__(self) -> None:
        numbers = (self.target_value, *self.raw_target_logits, self.actual_seconds)
        if self.predicted_seconds is not None:
            numbers += (self.predicted_seconds,)
        if not self.raw_target_logits or any(not math.isfinite(value) for value in numbers) or (
            self.actual_seconds < 0
            or (self.predicted_seconds is not None and self.predicted_seconds < 0)
        ):
            raise ValueError("scripted durations must be nonnegative")


@dataclass
class DeterministicInterventionRuntime:
    """CPU fake exercising the same adapter seam without sleeping."""

    baseline: BaselineCapture
    variants: dict[str, ScriptedVariant]
    baseline_actual_seconds: float = 0.0
    baseline_predicted_seconds: float | None = None
    cleanup_completed: bool = True
    requests: list[InterventionExecutionRequest] = field(default_factory=list, init=False)

    def evaluate(self, request: InterventionExecutionRequest) -> InterventionExecutionResult:
        self.requests.append(request)
        elapsed = 0.0
        observations: list[VariantObservation] = []
        baseline_prediction = (
            request.predicted_baseline_seconds
            if self.baseline_predicted_seconds is None
            else self.baseline_predicted_seconds
        )
        if baseline_prediction + request.cleanup_reserve_seconds > request.deadline_seconds:
            return self._result(
                request,
                RuntimeExecutionStatus.REFUSED,
                None,
                (),
                RuntimeRefusal("deadline_admission", "baseline predicted over budget"),
                elapsed,
            )
        elapsed += self.baseline_actual_seconds
        if elapsed + request.cleanup_reserve_seconds > request.deadline_seconds:
            return self._result(
                request,
                RuntimeExecutionStatus.REFUSED,
                self.baseline,
                (),
                RuntimeRefusal(
                    "deadline_in_flight_overrun",
                    "admitted baseline crossed the cooperative deadline",
                ),
                elapsed,
            )
        for variant in request.variants:
            script = self.variants.get(variant.variant_id)
            if script is None:
                return self._result(
                    request,
                    RuntimeExecutionStatus.PARTIAL,
                    self.baseline,
                    tuple(observations),
                    RuntimeRefusal("missing_script", "fake lacks observation", variant.variant_id),
                    elapsed,
                )
            predicted = (
                request.predicted_variant_seconds
                if script.predicted_seconds is None
                else script.predicted_seconds
            )
            if elapsed + predicted + request.cleanup_reserve_seconds > request.deadline_seconds:
                status = RuntimeExecutionStatus.PARTIAL if observations else RuntimeExecutionStatus.REFUSED
                return self._result(
                    request,
                    status,
                    self.baseline,
                    tuple(observations),
                    RuntimeRefusal(
                        "deadline_admission",
                        "variant predicted over remaining budget",
                        variant.variant_id,
                    ),
                    elapsed,
                )
            elapsed += script.actual_seconds
            observations.append(
                VariantObservation(
                    variant.variant_id,
                    script.target_value,
                    script.raw_target_logits,
                    script.downstream_feature_values,
                    elapsed,
                )
            )
            if elapsed + request.cleanup_reserve_seconds > request.deadline_seconds:
                return self._result(
                    request,
                    RuntimeExecutionStatus.PARTIAL,
                    self.baseline,
                    tuple(observations),
                    RuntimeRefusal(
                        "deadline_in_flight_overrun",
                        "admitted variant crossed the cooperative deadline",
                        variant.variant_id,
                    ),
                    elapsed,
                )
        return self._result(
            request,
            RuntimeExecutionStatus.COMPLETE,
            self.baseline,
            tuple(observations),
            None,
            elapsed,
        )

    def _result(
        self,
        request: InterventionExecutionRequest,
        status: RuntimeExecutionStatus,
        baseline: BaselineCapture | None,
        observations: tuple[VariantObservation, ...],
        refusal: RuntimeRefusal | None,
        elapsed: float,
    ) -> InterventionExecutionResult:
        return InterventionExecutionResult(
            status,
            baseline,
            observations,
            refusal,
            elapsed,
            self.cleanup_completed,
            max(0.0, elapsed - request.deadline_seconds),
        )
