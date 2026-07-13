"""Owned full-sequence NNSight forward capability."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from circuit_tracer.tracing.plan import ResolvedTracePlan
from circuit_tracer.tracing.request import TraceRequest

from .backend import run_nnsight_trace

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

    def trace_target_position(
        self,
        target_position: int,
        request: TraceRequest,
        plan: ResolvedTracePlan,
    ) -> Any:
        target_position = int(target_position)
        if not 0 < target_position < int(self.full_token_ids.numel()):
            raise ValueError("target_position must select a non-initial token")
        if target_position > self.window_max_prefix_len:
            raise ValueError("target_position exceeds the session prefix capacity")
        metadata = plan.evidence_metadata.get("prefix_view_metadata")
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("prefix_view_metadata evidence must be a mapping")
            if int(metadata.get("target_position", -1)) != target_position:
                raise ValueError("prefix metadata target_position must match the requested window")
            if (
                self.reuse_phase0_window_state or self.reuse_target_logits
            ) and metadata.get("mode") != "full_sequence_target_position":
                raise ValueError("window reuse requires full_sequence_target_position metadata")

        if not self.reuse_phase0_window_state and not self.reuse_target_logits:
            prompt = self.full_token_ids if metadata is not None else self.full_token_ids[:target_position]
            problem = replace(request.problem, prompt=prompt, output_position=target_position - 1)
            if self._decoder_cache_owner is None:
                return run_nnsight_trace(problem, plan)
            cache = fingerprint = None
            cache, fingerprint = self._decoder_cache_owner.acquire()
            try:
                return run_nnsight_trace(
                    problem,
                    plan,
                    forward_overrides=ForwardOverrides(
                        decoder_chunk_cache=cache,
                        decoder_cache_fingerprint=fingerprint,
                    ),
                )
            except BaseException:
                if self._decoder_cache_owner is not None:
                    self._decoder_cache_owner.reset()
                raise

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
        cache = fingerprint = None
        if self._decoder_cache_owner is not None:
            cache, fingerprint = self._decoder_cache_owner.acquire()
        try:
            return run_nnsight_trace(
                request.problem,
                plan,
                forward_overrides=ForwardOverrides(
                    phase0_context=phase0_context,
                    target_logits=target_logits,
                    target_logit_source=target_logit_source,
                    decoder_chunk_cache=cache,
                    decoder_cache_fingerprint=fingerprint,
                ),
            )
        except BaseException:
            if self._decoder_cache_owner is not None:
                self._decoder_cache_owner.reset()
            raise

    def close(self) -> None:
        if self._window_context is not None:
            cleanup = getattr(self._window_context, "cleanup", None)
            if callable(cleanup):
                cleanup()
            self._window_context = None
