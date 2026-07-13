"""Token and prefix-view preparation for NNSight Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from circuit_tracer.attribution.nnsight.prefix_view import (
    PrefixViewMetadata,
    _resolve_prefix_view_trace_input_ids,
)


@dataclass(frozen=True)
class Phase0TokenPreparation:
    """Validated token tensors and positions consumed by Phase 0 setup."""

    input_ids: torch.Tensor
    n_input_pos: int
    output_position: int | None
    trace_input_ids: torch.Tensor
    prefix_view_length: int | None


def prepare_phase0_tokens(
    *,
    model: Any,
    prompt: str | torch.Tensor | list[int],
    output_position: int | None,
    prefix_view_metadata: PrefixViewMetadata | None,
) -> Phase0TokenPreparation:
    """Tokenize the prompt and resolve its validated prefix-view projection."""
    input_ids = model.ensure_tokenized(prompt)
    n_input_pos = int(input_ids.shape[-1])
    if output_position is not None:
        output_position = int(output_position)
        if output_position < 0 or output_position >= n_input_pos:
            raise ValueError(
                f"output_position must be in [0, {n_input_pos}) (got {output_position})"
            )
    trace_input_ids, prefix_view_length = _resolve_prefix_view_trace_input_ids(
        input_ids, prefix_view_metadata
    )
    return Phase0TokenPreparation(
        input_ids=input_ids,
        n_input_pos=n_input_pos,
        output_position=output_position,
        trace_input_ids=trace_input_ids,
        prefix_view_length=prefix_view_length,
    )
