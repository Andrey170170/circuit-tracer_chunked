"""Replay evidence packaging for completed NNSight Phase 3 batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.attribution.nnsight.replay import (
    _build_phase3_gradient_bundle_payload,
    _build_phase3_row_bundle_payload,
)

from .phase3_rows import Phase3ReplayRows


@dataclass(frozen=True)
class Phase3ReplayEvidence:
    """Optional immutable replay payloads produced from effective Phase 3 state."""

    gradient_bundle: dict[str, object] | None
    row_bundle: dict[str, object] | None


def package_phase3_replay_evidence(
    *, inputs: Any, config: Any, rows: Phase3ReplayRows
) -> Phase3ReplayEvidence:
    """Package requested gradient and row evidence without changing capture order."""
    target_token_ids = torch.tensor(
        [int(target.vocab_idx) for target in inputs.targets.logit_targets],
        dtype=torch.int64,
    )
    gradient_bundle = inputs.phase3_gradient_bundle_payload
    if config.capture_phase3_gradient_bundle_enabled:
        captures = getattr(inputs.ctx, "phase3_gradient_captures", [])
        gradient_bundle = _build_phase3_gradient_bundle_payload(
            gradient_captures=captures if isinstance(captures, list) else [],
            active_features=inputs.activation_matrix.indices().T,
            activation_values=inputs.activation_matrix.values(),
            target_token_ids=target_token_ids,
            target_probabilities=inputs.targets.logit_probabilities,
            status=(
                "captured_replayed_effective_state"
                if config.phase3_gradient_replay_mode_resolved != "disabled"
                else "captured"
            ),
        )
    row_bundle = inputs.phase3_row_bundle_payload
    if config.capture_phase3_row_bundle_enabled:
        row_bundle = _build_phase3_row_bundle_payload(
            feature_rows=rows.feature_rows,
            row_abs_sums=rows.row_abs_sums,
            feature_abs_sums=rows.feature_abs_sums,
            error_abs_sums=rows.error_abs_sums,
            token_abs_sums=rows.token_abs_sums,
            error_rows_by_layer=rows.error_rows_by_layer,
            token_rows=rows.token_rows,
            active_features=inputs.activation_matrix.indices().T,
            activation_values=inputs.activation_matrix.values(),
            target_token_ids=target_token_ids,
            target_probabilities=inputs.targets.logit_probabilities,
            total_active_features=int(config.total_active_feats),
            error_column_count=int(config.n_layers * config.n_pos),
            token_column_count=int(config.n_pos),
            status=(
                "captured_replayed_effective_state"
                if (
                    config.phase3_gradient_replay_mode_resolved != "disabled"
                    or config.phase3_row_replay_mode_resolved != "disabled"
                )
                else "captured"
            ),
        )
    return Phase3ReplayEvidence(gradient_bundle, row_bundle)
