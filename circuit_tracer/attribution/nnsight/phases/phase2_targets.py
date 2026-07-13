"""Target/logit selection for NNSight attribution Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)


@dataclass(frozen=True)
class TargetSelectionPolicy:
    output_position: int | None
    n_input_pos: int
    max_n_logits: int
    desired_logit_prob: float

    def __post_init__(self) -> None:
        if self.n_input_pos <= 0 or self.max_n_logits <= 0:
            raise ValueError("token and logit limits must be positive")
        if not 0.0 < self.desired_logit_prob <= 1.0:
            raise ValueError("desired logit probability must be in (0, 1]")


@dataclass(frozen=True)
class TargetSelection:
    targets: AttributionTargets
    output_logits: torch.Tensor
    target_token_ids: torch.Tensor


def select_attribution_targets(
    *,
    logger: Any,
    model: Any,
    ctx: Any,
    policy: TargetSelectionPolicy,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None,
    target_logits_override: torch.Tensor | None,
) -> TargetSelection:
    """Resolve the effective logits and construct the attribution targets."""
    output_logits = (
        ctx.get_logits_at_position(policy.output_position)[0]
        if policy.output_position is not None
        and policy.output_position != policy.n_input_pos - 1
        else ctx.get_last_token_logits()[0]
    )
    if target_logits_override is not None:
        output_logits = target_logits_override.to(device=output_logits.device)
    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=output_logits,
        unembed_proj=cast(torch.Tensor, model.unembed_weight),
        tokenizer=model.tokenizer,
        max_n_logits=policy.max_n_logits,
        desired_logit_prob=policy.desired_logit_prob,
    )
    log_attribution_target_info(targets, attribution_targets, logger)
    target_token_ids = torch.tensor(
        [int(target.vocab_idx) for target in targets.logit_targets],
        dtype=torch.int64,
        device=output_logits.device,
    )
    return TargetSelection(targets, output_logits, target_token_ids)
