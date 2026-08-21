"""Owned full-sequence NNSight forward capability."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from circuit_tracer.tracing.problem import AttributionProblem
from circuit_tracer.tracing.request import TraceRequest

@dataclass(frozen=True)
class ForwardOverrides:
    """Reusable physical forward capabilities supplied by an owning session."""

    phase0_context: object | None = None
    target_logits: torch.Tensor | None = None
    target_logit_source: str | None = None
    decoder_chunk_cache: object | None = None
    decoder_cache_fingerprint: object | None = None


class ForwardTraceSession:
    """Lazily owns one saved full-prefix context and closes it deterministically."""

    def __init__(
        self,
        *,
        model: Any,
        full_token_ids: torch.Tensor | list[int],
        window_max_prefix_len: int | None,
        reuse_phase0_window_state: bool,
        reuse_target_logits: bool,
        decoder_cache_owner: Any | None = None,
    ) -> None:
        if isinstance(full_token_ids, str):
            raise ValueError("full-sequence sessions require token ids, not text")
        self.model = model
        self.full_token_ids = (
            full_token_ids.detach().clone().to(dtype=torch.long).reshape(-1)
            if isinstance(full_token_ids, torch.Tensor)
            else torch.tensor([int(value) for value in full_token_ids], dtype=torch.long)
        )
        available = int(self.full_token_ids.numel())
        self.window_max_prefix_len = (
            available if window_max_prefix_len is None else int(window_max_prefix_len)
        )
        if not 0 < self.window_max_prefix_len <= available:
            raise ValueError("window max prefix length must fit the full token sequence")
        self.reuse_phase0_window_state = reuse_phase0_window_state
        self.reuse_target_logits = reuse_target_logits
        self._decoder_cache_owner = decoder_cache_owner
        self._window_context: Any = None

    def _get_window_context(self) -> Any:
        if self._window_context is None:
            self._window_context = self.model.setup_attribution(
                self.full_token_ids[: self.window_max_prefix_len],
                retain_full_logits=True,
            )
        return self._window_context

    def prepare_target_position(
        self,
        target_position: int,
        request: TraceRequest,
    ) -> tuple[AttributionProblem, ForwardOverrides]:
        target_position = int(target_position)
        if not 0 < target_position < int(self.full_token_ids.numel()):
            raise ValueError("target_position must select a non-initial token")
        if target_position > self.window_max_prefix_len:
            raise ValueError("target_position exceeds the session prefix capacity")
        prefix_view = request.problem.prefix_view
        if prefix_view is None or prefix_view.target_position != target_position:
            raise ValueError("window request must own its prefix-view target")
        if (
            self.reuse_phase0_window_state or self.reuse_target_logits
        ) and prefix_view.mode != "full_sequence_target_position":
            raise ValueError("window reuse requires full_sequence_target_position mode")

        if not self.reuse_phase0_window_state and not self.reuse_target_logits:
            problem = replace(
                request.problem,
                prompt=self.full_token_ids[:target_position],
                output_position=target_position - 1,
            )
            cache, fingerprint = self._acquire_decoder_cache()
            return problem, ForwardOverrides(
                decoder_chunk_cache=cache,
                decoder_cache_fingerprint=fingerprint,
            )

        context = self._get_window_context()
        phase0_context = None
        if self.reuse_phase0_window_state:
            derive = getattr(context, "derive_prefix_view_context", None)
            if not callable(derive):
                raise RuntimeError("attribution context does not support prefix views")
            phase0_context = derive(target_position)
        target_logits = None
        target_logit_source = None
        if self.reuse_target_logits:
            target_logits = context.get_logits_at_position(target_position - 1)[0].detach()
            target_logit_source = "full_sequence_window_logits"
        cache, fingerprint = self._acquire_decoder_cache()
        return request.problem, ForwardOverrides(
            phase0_context=phase0_context,
            target_logits=target_logits,
            target_logit_source=target_logit_source,
            decoder_chunk_cache=cache,
            decoder_cache_fingerprint=fingerprint,
        )

    def _acquire_decoder_cache(self) -> tuple[object | None, object | None]:
        if self._decoder_cache_owner is None:
            return None, None
        return self._decoder_cache_owner.acquire()

    def close(self) -> None:
        if self._window_context is not None:
            cleanup = getattr(self._window_context, "cleanup", None)
            if callable(cleanup):
                cleanup()
            self._window_context = None
