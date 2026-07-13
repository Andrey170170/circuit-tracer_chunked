"""Domain result produced by transcoders during attribution setup."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
            if decoder_rows or self.encoder_to_decoder_map.numel() or self.decoder_locations.numel():
                raise ValueError("chunked components must not materialize decoder rows")
            required = {"source_layers", "positions", "feature_ids", "activation_values"}
            missing = required.difference(self.chunked_decoder_state)
            if missing:
                raise ValueError(f"chunked decoder state is missing: {', '.join(sorted(missing))}")
            lengths = {int(self.chunked_decoder_state[name].numel()) for name in required}
            if lengths != {active_features}:
                raise ValueError("chunked decoder state must contain one entry per active feature")

    @property
    def active_feature_count(self) -> int:
        return int(self.activation_matrix._nnz())
