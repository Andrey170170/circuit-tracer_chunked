"""Invariant-bearing state used to construct an NNSight attribution context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

from circuit_tracer.tracing.plan import (
    BackwardEngineMode,
    BackwardExecutionTopology,
)

from circuit_tracer.transcoder.attribution_result import DecoderRowSeed

from circuit_tracer.transcoder.provider import get_transcoder_capabilities, provider_fingerprint


EncoderResidency = Literal["lazy", "active_cpu"]


@dataclass(frozen=True)
class AttributionTensorState:
    activation_matrix: torch.Tensor
    error_vectors: torch.Tensor
    token_vectors: torch.Tensor
    decoder_vectors: torch.Tensor
    encoder_vectors: torch.Tensor
    encoder_to_decoder_map: torch.Tensor
    decoder_locations: torch.Tensor
    logits: torch.Tensor
    full_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.activation_matrix.is_sparse or self.activation_matrix.ndim != 3:
            raise ValueError("activation_matrix must be a sparse rank-3 tensor")
        layers, positions, _ = self.activation_matrix.shape
        if tuple(self.error_vectors.shape[:2]) != (layers, positions):
            raise ValueError("error vectors must align with activation layers and positions")
        if self.token_vectors.ndim != 2 or int(self.token_vectors.shape[0]) != positions:
            raise ValueError("token vectors must contain one row per token position")
        if self.decoder_vectors.ndim != 2 or self.encoder_vectors.ndim != 2:
            raise ValueError("encoder and decoder vectors must be rank 2")
        if self.encoder_to_decoder_map.ndim != 1:
            raise ValueError("encoder_to_decoder_map must be rank 1")
        if self.decoder_locations.ndim != 2 or int(self.decoder_locations.shape[0]) != 2:
            raise ValueError("decoder_locations must have shape (2, decoder rows)")


@dataclass(frozen=True)
class ContextExecutionPolicy:
    exact_chunked_mode: bool
    encoder_residency_requested: EncoderResidency
    encoder_residency_effective: EncoderResidency
    encoder_residency_fallback_reason: str | None
    stage_encoder_vectors_on_cpu: bool
    stage_error_vectors_on_cpu: bool
    error_vector_prefetch_lookahead: int
    chunked_feature_replay_window: int
    row_subchunk_size: int | None
    backward_engine_mode: BackwardEngineMode
    backward_batch_capacity: int

    @classmethod
    def resolve(
        cls,
        *,
        chunked_decoder_state: dict[str, torch.Tensor] | None,
        encoder_vectors: torch.Tensor,
        error_vectors: torch.Tensor,
        exact_encoder_residency: str,
        stage_encoder_vectors_on_cpu: bool | None,
        stage_error_vectors_on_cpu: bool | None,
        error_vector_prefetch_lookahead: int,
        chunked_feature_replay_window: int,
        row_subchunk_size: int | None,
        backward_engine_mode: str = "duplicated_lanes",
        backward_batch_capacity: int = 1,
    ) -> "ContextExecutionPolicy":
        normalized = str(exact_encoder_residency).strip().lower()
        allowed = {"lazy", "active_cpu"}
        if normalized not in allowed:
            raise ValueError(
                "exact_encoder_residency must be one of: "
                f"{', '.join(sorted(allowed))} (got {exact_encoder_residency!r})"
            )
        requested = cast(EncoderResidency, normalized)
        exact_chunked_mode = chunked_decoder_state is not None
        effective = requested
        fallback_reason = None
        if requested != "lazy" and not exact_chunked_mode:
            effective = "lazy"
            fallback_reason = (
                "active encoder residency requires exact chunked decoder state; "
                "falling back to lazy execution"
            )
        if stage_encoder_vectors_on_cpu is None:
            stage_encoder_vectors_on_cpu = exact_chunked_mode and encoder_vectors.numel() > 0
        if effective != "lazy":
            stage_encoder_vectors_on_cpu = True
        if stage_error_vectors_on_cpu is None:
            stage_error_vectors_on_cpu = exact_chunked_mode and error_vectors.numel() > 0
        topology = BackwardExecutionTopology.resolve(
            mode=cast(
                BackwardEngineMode,
                str(backward_engine_mode).strip().lower(),
            ),
            batch_capacity=int(backward_batch_capacity),
        )
        return cls(
            exact_chunked_mode=exact_chunked_mode,
            encoder_residency_requested=requested,
            encoder_residency_effective=effective,
            encoder_residency_fallback_reason=fallback_reason,
            stage_encoder_vectors_on_cpu=bool(stage_encoder_vectors_on_cpu),
            stage_error_vectors_on_cpu=bool(stage_error_vectors_on_cpu),
            error_vector_prefetch_lookahead=max(1, int(error_vector_prefetch_lookahead)),
            chunked_feature_replay_window=max(1, int(chunked_feature_replay_window)),
            row_subchunk_size=None if row_subchunk_size is None else max(1, int(row_subchunk_size)),
            backward_engine_mode=topology.mode,
            backward_batch_capacity=topology.batch_capacity,
        )


@dataclass(frozen=True)
class DecoderRuntime:
    provider: object | None
    chunked_state: dict[str, torch.Tensor] | None
    chunk_cache: object | None
    cache_fingerprint: object | None
    owns_cache: bool
    decoder_row_seed: DecoderRowSeed | None
    decoder_row_seed_refusal_reason: str | None
    decoder_row_seed_estimated_bytes: int | None

    @classmethod
    def resolve(
        cls,
        *,
        provider: object | None,
        chunked_state: dict[str, torch.Tensor] | None,
        chunk_cache: object | None = None,
        cache_fingerprint: object | None = None,
        decoder_row_seed: DecoderRowSeed | None = None,
        decoder_row_seed_refusal_reason: str | None = None,
        decoder_row_seed_estimated_bytes: int | None = None,
    ) -> "DecoderRuntime":
        if chunked_state is None and provider is not None:
            raise ValueError("decoder provider requires chunked decoder state")
        if cache_fingerprint is None and provider is not None:
            capabilities = get_transcoder_capabilities(provider)
            if capabilities.supports_exact_chunked_provider:
                cache_fingerprint = provider_fingerprint(provider)
        if chunk_cache is not None:
            if cache_fingerprint is None:
                raise ValueError("shared decoder cache requires fingerprint metadata")
            if not hasattr(chunk_cache, "fingerprint"):
                raise ValueError("shared decoder cache is missing fingerprint metadata")
            actual = getattr(chunk_cache, "fingerprint")
            if actual != cache_fingerprint:
                raise ValueError(
                    "shared decoder cache fingerprint mismatch "
                    f"({actual!r} != {cache_fingerprint!r})"
                )
        return cls(
            provider=provider,
            chunked_state=chunked_state,
            chunk_cache=chunk_cache,
            cache_fingerprint=cache_fingerprint,
            owns_cache=chunk_cache is None,
            decoder_row_seed=decoder_row_seed,
            decoder_row_seed_refusal_reason=decoder_row_seed_refusal_reason,
            decoder_row_seed_estimated_bytes=decoder_row_seed_estimated_bytes,
        )


@dataclass(frozen=True)
class ContextNumericPolicy:
    materialized_encoder_vectors_during_phase0: bool = False
    internal_precision_requested: str | None = None
    resolved_dtype_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.resolved_dtype_map is not None:
            object.__setattr__(self, "resolved_dtype_map", dict(self.resolved_dtype_map))
