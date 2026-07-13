"""Replay and activation-state handling for NNSight Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata,
    _apply_prefix_view_activation_mask,
)
from circuit_tracer.observability.events import SparsificationProfile, TraceObserver


@dataclass(frozen=True)
class Phase0ActivationState:
    """Activation matrix and prefix-mask evidence passed to later phases."""

    activation_matrix: torch.Tensor
    prefix_view_activation_mask_metadata: dict[str, int] | None


def prepare_phase0_activation_state(
    *,
    ctx: Any,
    prefix_view_metadata: PrefixViewMetadata | None,
    diagnostic_feature_cap: int | None,
    profile: bool,
    logger: Any,
    observer: TraceObserver,
) -> Phase0ActivationState:
    """Apply prefix replay state, diagnostic caps, and sparsification evidence."""
    prefix_mask_metadata: dict[str, int] | None = None
    if (
        prefix_view_metadata is not None
        and prefix_view_metadata.get("mode") == "full_sequence_target_position"
    ):
        replace_state = getattr(ctx, "replace_phase0_activation_state", None)
        if not callable(replace_state):
            raise RuntimeError(
                "Attribution context does not support Phase-0 activation-state replacement"
            )
        prefix_mask_metadata = _apply_prefix_view_activation_mask(
            ctx, int(prefix_view_metadata["target_position"])
        )
        if isinstance(getattr(ctx, "setup_diagnostic_stats", None), dict):
            ctx.setup_diagnostic_stats["prefix_view_activation_mask"] = dict(prefix_mask_metadata)

    if diagnostic_feature_cap is not None and diagnostic_feature_cap > 0:
        before_cap, after_cap = ctx.apply_diagnostic_feature_cap(diagnostic_feature_cap)
        logger.info(
            "Diagnostic feature cap applied before attribution rows: "
            f"{before_cap} -> {after_cap} active features"
        )
    if profile and getattr(ctx, "sparsification_stats", None):
        observer.observe(SparsificationProfile(ctx.sparsification_stats))

    return Phase0ActivationState(
        activation_matrix=ctx.activation_matrix,
        prefix_view_activation_mask_metadata=prefix_mask_metadata,
    )
