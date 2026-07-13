"""Scientific inputs and semantic choices for canonical circuit tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import torch

from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.attribution.targets import TargetSpec


Prompt = str | torch.Tensor | list[int]
AttributionTargets = Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None


def _positive(name: str, value: int | float | None) -> None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be positive when provided")


def _nonnegative(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be nonnegative when provided")


@dataclass(frozen=True)
class AttributionProblem:
    """Model, input, target selection, and graph objective for one trace."""

    model: Any = field(repr=False, compare=False)
    prompt: Prompt
    targets: AttributionTargets = field(default=None, repr=False)
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    output_position: int | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("model is required")
        if isinstance(self.prompt, list) and not self.prompt:
            raise ValueError("prompt token ids cannot be empty")
        _positive("max_n_logits", self.max_n_logits)
        if not 0 < self.desired_logit_prob <= 1:
            raise ValueError("desired_logit_prob must be in (0, 1]")
        _nonnegative("output_position", self.output_position)


@dataclass(frozen=True)
class TraceSemantics:
    """Complete choices that are allowed to change the mathematical result."""

    source_batch_size: int = 512
    feature_batch_size: int | None = None
    logit_batch_size: int | None = None
    max_feature_nodes: int | None = None
    diagnostic_feature_cap: int | None = None
    update_interval: int = 4
    exact_trace_internal_dtype: Literal["fp32", "fp64"] = "fp32"
    phase0_activation_threshold_compare_mode: Literal[
        "baseline", "bf16", "fp32", "fp64"
    ] = "baseline"
    sparsification: SparsificationConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "source_batch_size",
            "feature_batch_size",
            "logit_batch_size",
            "max_feature_nodes",
            "diagnostic_feature_cap",
            "update_interval",
        ):
            _positive(name, getattr(self, name))

