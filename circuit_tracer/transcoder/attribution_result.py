"""Domain result produced by transcoders during attribution setup."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from circuit_tracer.transcoder.phase0_decoder_ranges import (
    Phase0DecoderRangeTelemetry,
)


@dataclass(frozen=True)
class DecoderRowSeedLayer:
    """CPU-staged unique raw decoder rows captured during Phase-0 reconstruction."""

    source_layer: int
    output_layers: tuple[int, ...]
    feature_ids: torch.Tensor
    rows: torch.Tensor

    def __post_init__(self) -> None:
        if self.feature_ids.device.type != "cpu" or self.rows.device.type != "cpu":
            raise ValueError("decoder row seed tensors must be CPU staged")
        if self.feature_ids.ndim != 1 or self.feature_ids.dtype != torch.long:
            raise ValueError("decoder row seed feature ids must be a rank-1 long tensor")
        if self.rows.ndim != 3:
            raise ValueError("decoder row seed rows must be rank 3")
        if tuple(self.rows.shape[:2]) != (int(self.feature_ids.numel()), len(self.output_layers)):
            raise ValueError("decoder row seed rows must align with feature ids and output layers")
        if self.feature_ids.numel() > 1 and not bool(
            torch.all(self.feature_ids[1:] > self.feature_ids[:-1])
        ):
            raise ValueError("decoder row seed feature ids must be strictly increasing")


@dataclass(frozen=True)
class DecoderRowSeed:
    """Bounded CPU seed used to materialize final active decoder rows after admission."""

    layers: tuple[DecoderRowSeedLayer | None, ...]
    source_fingerprint: dict[str, object]
    occurrence_estimated_bytes: int
    capture_seconds: float
    shared_traversal_bytes: int
    shared_decoder_load_count: int
    shared_decoder_load_bytes: int
    phase0_decoder_range_telemetry: Phase0DecoderRangeTelemetry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_fingerprint", dict(self.source_fingerprint))

    @property
    def unique_row_count(self) -> int:
        return sum(int(layer.feature_ids.numel()) for layer in self.layers if layer is not None)

    @property
    def seed_bytes(self) -> int:
        return sum(
            int(layer.rows.numel() * layer.rows.element_size())
            for layer in self.layers
            if layer is not None
        )


@dataclass(frozen=True)
class AttributionComponents:
    """Validated feature decomposition consumed by attribution runtimes.

    Runtime setup consumes named fields so additions cannot silently change its
    contract or mix unrelated values.
    """

    activation_matrix: torch.Tensor
    reconstruction: torch.Tensor
    encoder_vectors: torch.Tensor
    decoder_vectors: torch.Tensor
    encoder_to_decoder_map: torch.Tensor
    decoder_locations: torch.Tensor
    chunked_decoder_state: dict[str, torch.Tensor] | None = None
    sparsification_stats: dict[str, object] | None = None
    decoder_row_seed: DecoderRowSeed | None = None
    decoder_row_seed_refusal_reason: str | None = None
    decoder_row_seed_estimated_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.activation_matrix.is_sparse:
            raise ValueError("activation_matrix must be a sparse tensor")
        if self.reconstruction.ndim != 3:
            raise ValueError("reconstruction must have shape (layers, positions, d_model)")
        if self.encoder_vectors.ndim != 2 or self.decoder_vectors.ndim != 2:
            raise ValueError("encoder and decoder vectors must be rank-2 tensors")
        if self.encoder_to_decoder_map.ndim != 1:
            raise ValueError("encoder_to_decoder_map must be rank 1")
        if self.decoder_locations.ndim != 2 or self.decoder_locations.shape[0] != 2:
            raise ValueError("decoder_locations must have shape (2, decoder_rows)")

        active_features = int(self.activation_matrix._nnz())
        if self.encoder_vectors.shape[0] not in (0, active_features):
            raise ValueError("encoder vectors must be empty or contain one row per active feature")

        decoder_rows = int(self.decoder_vectors.shape[0])
        if int(self.decoder_locations.shape[1]) != decoder_rows:
            raise ValueError("decoder locations and decoder vectors must have equal row counts")
        if int(self.encoder_to_decoder_map.numel()) not in (0, decoder_rows):
            raise ValueError("encoder-to-decoder map must be empty or match decoder rows")

        if self.chunked_decoder_state is not None:
            if (
                decoder_rows
                or self.encoder_to_decoder_map.numel()
                or self.decoder_locations.numel()
            ):
                raise ValueError("chunked components must not materialize decoder rows")
            required = {"source_layers", "positions", "feature_ids", "activation_values"}
            missing = required.difference(self.chunked_decoder_state)
            if missing:
                raise ValueError(f"chunked decoder state is missing: {', '.join(sorted(missing))}")
            lengths = {int(self.chunked_decoder_state[name].numel()) for name in required}
            if lengths != {active_features}:
                raise ValueError("chunked decoder state must contain one entry per active feature")
        elif self.decoder_row_seed is not None:
            raise ValueError("decoder row seed requires chunked decoder state")

    @property
    def active_feature_count(self) -> int:
        return int(self.activation_matrix._nnz())
