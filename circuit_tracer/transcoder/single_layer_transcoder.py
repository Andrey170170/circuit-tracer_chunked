import os
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from circuit_tracer.attribution.sparsification import (
    SparsificationConfig,
    filter_sparse_activations,
    select_candidate_feature_indices,
)
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.transcoder.attribution_result import (
    AttributionComponents,
    DecoderRowSeed,
    DecoderRowSeedLayer,
)
from circuit_tracer.transcoder.checkpoint_working_set import ProviderCheckpointLifecycle
from circuit_tracer.transcoder.checkpoint_assets import CheckpointAssetScope
from circuit_tracer.transcoder.checkpoint_manifest import build_checkpoint_manifest
from circuit_tracer.transcoder.provider import TranscoderCapabilities, provider_fingerprint
from circuit_tracer.transcoder.phase0_decoder_ranges import (
    Phase0DecoderRangeTelemetry,
    combine_phase0_decoder_range_telemetry,
    load_decoder_row_ranges,
    plan_decoder_row_ranges,
)
from circuit_tracer.utils import get_default_device


DEFAULT_CROSS_BATCH_DECODER_CACHE_BYTES = 0


def _validate_decoder_chunk_size(decoder_chunk_size: int) -> int:
    decoder_chunk_size = int(decoder_chunk_size)
    if decoder_chunk_size <= 0:
        raise ValueError(f"decoder_chunk_size must be positive, got {decoder_chunk_size}")
    return decoder_chunk_size


def _slice_rows(safe_slice, row_ids, *, device: torch.device) -> torch.Tensor:
    if isinstance(row_ids, torch.Tensor):
        row_ids = row_ids.detach().cpu().reshape(-1).tolist()
    rows = [safe_slice[int(row_id) : int(row_id) + 1] for row_id in row_ids]
    if not rows:
        shape = safe_slice.get_shape()
        return torch.empty((0, shape[1]), device=device)
    return torch.cat(rows, dim=0)


def _slice_columns_transposed(safe_slice, column_ids, *, device: torch.device) -> torch.Tensor:
    if isinstance(column_ids, torch.Tensor):
        column_ids = column_ids.detach().cpu().reshape(-1).tolist()
    rows = [safe_slice[:, int(column_id) : int(column_id) + 1].T for column_id in column_ids]
    if not rows:
        shape = safe_slice.get_shape()
        return torch.empty((0, shape[0]), device=device)
    return torch.cat(rows, dim=0).contiguous()


def safetensors_has_gemmascope2_plt_keys(path: str) -> bool:
    if Path(path).suffix != ".safetensors":
        return False
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
    return {"w_enc", "w_dec", "threshold"}.issubset(keys)


def select_single_layer_transcoder_load_fn(
    path: str,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None] = None,
):
    npz_format = Path(path).suffix == ".npz"
    if special_load_fn == "gemma-scope" and npz_format:
        return load_gemma_scope_transcoder
    if special_load_fn == "gemma-scope-2" or safetensors_has_gemmascope2_plt_keys(path):
        return load_gemma_scope_2_transcoder
    return load_relu_transcoder


class SingleLayerTranscoder(nn.Module):
    """
    A per-layer transcoder (PLT) that replaces MLP computation with interpretable features.

    Per-layer transcoders decompose the output of a single MLP layer into sparsely active
    features that often correspond to interpretable concepts. Unlike cross-layer transcoders,
    each PLT operates independently on its assigned layer, which can result in longer paths
    through attribution graphs when features amplify across multiple layers.

    Attributes:
        d_model: Dimension of the transformer's residual stream
        d_transcoder: Number of learned features (typically >> d_model for superposition)
        layer_idx: Which transformer layer this transcoder replaces
        W_enc: Encoder weights mapping residual stream to feature space
        W_dec: Decoder weights mapping features back to residual stream
        b_enc: Encoder bias terms
        b_dec: Decoder bias terms (reconstruction baseline)
        W_skip: Optional skip connection weights (https://arxiv.org/abs/2501.18823)
        activation_function: Sparsity-inducing nonlinearity (e.g., ReLU, JumpReLU)
    """

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function,
        layer_idx: int,
        skip_connection: bool = False,
        transcoder_path: str | None = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        weight_format: Literal["standard", "gemmascope2"] = "standard",
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layer_idx = layer_idx
        self.transcoder_path = transcoder_path
        self.lazy_encoder = lazy_encoder
        self.lazy_decoder = lazy_decoder
        self.weight_format = weight_format

        if lazy_encoder or lazy_decoder:
            assert self.transcoder_path is not None, "Transcoder path must be set for lazy loading"

        if not lazy_encoder:
            self.W_enc = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        if not lazy_decoder:
            self.W_dec = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        self.b_enc = nn.Parameter(torch.zeros(d_transcoder, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    def __getattr__(self, name):
        """Dynamically load weights when accessed if lazy loading is enabled."""

        if name == "W_enc" and self.lazy_encoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                if self.weight_format == "gemmascope2":
                    return f.get_tensor("w_enc").T.contiguous().to(self.dtype)
                return f.get_tensor("W_enc").to(self.dtype)
        elif name == "W_dec" and self.lazy_decoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                key = "w_dec" if self.weight_format == "gemmascope2" else "W_dec"
                return f.get_tensor(key).to(self.dtype)

        return super().__getattr__(name)

    def _get_decoder_vectors(self, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]
        if not self.lazy_decoder:
            return self.W_dec[to_read].to(self.dtype)

        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
            key = "w_dec" if self.weight_format == "gemmascope2" else "W_dec"
            if isinstance(to_read, slice):
                return f.get_slice(key)[to_read].to(self.dtype)
            return _slice_rows(f.get_slice(key), to_read, device=self.device).to(self.dtype)

    def materialize_encoder_rows(self, feature_ids) -> torch.Tensor:
        if isinstance(feature_ids, torch.Tensor):
            feature_ids_read = feature_ids.detach().cpu()
        else:
            feature_ids_read = feature_ids
        if not self.lazy_encoder:
            return self.W_enc[feature_ids].to(dtype=self.dtype)
        assert self.transcoder_path is not None
        with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
            if self.weight_format == "gemmascope2":
                return _slice_columns_transposed(
                    f.get_slice("w_enc"), feature_ids_read, device=self.device
                ).to(self.dtype)
            return _slice_rows(f.get_slice("W_enc"), feature_ids_read, device=self.device).to(
                self.dtype
            )

    def get_decoder_chunk(self, chunk_id: int, decoder_chunk_size: int) -> torch.Tensor:
        start = chunk_id * decoder_chunk_size
        stop = min(start + decoder_chunk_size, self.d_transcoder)
        if start >= self.d_transcoder or stop <= start:
            raise IndexError(f"Decoder chunk {chunk_id} out of range for layer {self.layer_idx}")
        if not self.lazy_decoder:
            block = self.W_dec[start:stop]
        else:
            assert self.transcoder_path is not None
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                key = "w_dec" if self.weight_format == "gemmascope2" else "W_dec"
                block = f.get_slice(key)[start:stop]
        return block.to(dtype=self.dtype).unsqueeze(1)

    def encode(self, input_acts, apply_activation_function: bool = True):
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        if not apply_activation_function:
            return pre_acts
        return self.activation_function(pre_acts)

    def decode(self, acts, input_acts: torch.Tensor | None = None):
        W_dec = self.W_dec
        reconstruction = acts @ W_dec + self.b_dec
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        return reconstruction

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            return input_acts @ self.W_skip.T
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts):
        transcoder_acts = self.encode(input_acts)
        decoded = self.decode(transcoder_acts, input_acts)
        # decoded = decoded.detach()
        # decoded.requires_grad = True

        return decoded

    def encode_sparse(
        self,
        input_acts,
        zero_positions: slice = slice(0, 1),
        *,
        return_encoder_vectors: bool = True,
    ):
        """Encode and return sparse activations with active encoder vectors.

        Args:
            input_acts: Input activations
            zero_positions: slice representing the positions to zero out

        Returns:
            sparse_acts: Sparse tensor of activations
            active_encoders: Encoder vectors for active features only
        """
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        acts = self.activation_function(pre_acts)

        acts[zero_positions] = 0

        sparse_acts = acts.to_sparse()
        _, feat_idx = sparse_acts.indices()
        active_encoders = W_enc[feat_idx] if return_encoder_vectors else None

        return sparse_acts, active_encoders

    def decode_sparse(self, sparse_acts, input_acts: torch.Tensor | None = None):
        """Decode sparse activations and return reconstruction with scaled decoder vectors.

        Returns:
            reconstruction: Decoded output
            scaled_decoders: Decoder vectors scaled by activation values
        """
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()

        # Get decoder vectors for active features only
        W_dec = self._get_decoder_vectors(feat_idx.cpu())
        scaled_decoders = W_dec * values[:, None]

        # Reconstruct using index_add
        n_pos = sparse_acts.shape[0]
        reconstruction = torch.zeros(
            n_pos, self.d_model, device=sparse_acts.device, dtype=sparse_acts.dtype
        )
        reconstruction = reconstruction.index_add_(0, pos_idx, scaled_decoders)
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        reconstruction = reconstruction + self.b_dec

        return reconstruction, scaled_decoders

    def to_safetensors(self, save_path: str):
        """Save transcoder to safetensors format compatible with lazy loading.

        Saves the transcoder state dict to a single safetensors file with keys:
        W_enc, W_dec, b_enc, b_dec, and optionally activation_function.threshold and W_skip.

        Args:
            save_path: Path to the safetensors file to save
        """
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_enc": self.b_enc.cpu(),
            "b_dec": self.b_dec.cpu(),
        }

        if isinstance(self.activation_function, JumpReLU):
            state_dict["activation_function.threshold"] = self.activation_function.threshold.cpu()

        if self.W_skip is not None:
            state_dict["W_skip"] = self.W_skip.cpu()

        save_file(state_dict, save_path)


class TranscoderSet(nn.Module):
    """
    A collection of per-layer transcoders that enable construction of a replacement model.

    TranscoderSet manages the collection of SingleLayerTranscoders needed for this substitution,
    where each transcoder replaces the MLP computation at its corresponding layer.

    Attributes:
        transcoders: ModuleList of SingleLayerTranscoder instances, one per layer
        n_layers: Total number of layers covered
        d_transcoder: Common feature dimension across all transcoders
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier to identify corresponding feature visualization
        skip_connection: Whether transcoders include learned skip connections
    """

    def __init__(
        self,
        transcoders: dict[int, SingleLayerTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan: str | list[str] | None = None,
        exact_chunked_provider: bool = False,
        decoder_chunk_size: int = 1024,
        cross_batch_decoder_cache_bytes: int | None = DEFAULT_CROSS_BATCH_DECODER_CACHE_BYTES,
        checkpoint_lifecycle: ProviderCheckpointLifecycle | None = None,
    ):
        super().__init__()
        if exact_chunked_provider:
            decoder_chunk_size = _validate_decoder_chunk_size(decoder_chunk_size)
        # Validate that we have continuous layers from 0 to max
        assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
            f"Each layer should have a transcoder, but got transcoders for layers "
            f"{set(transcoders.keys())}"
        )

        self.transcoders = nn.ModuleList([transcoders[i] for i in range(len(transcoders))])
        self.n_layers = len(self.transcoders)
        self.d_transcoder = self.transcoders[0].d_transcoder

        # Verify all transcoders have the same d_transcoder
        for transcoder in self.transcoders:
            assert transcoder.d_transcoder == self.d_transcoder, (
                f"All transcoders must have the same d_transcoder, but got "
                f"{transcoder.d_transcoder} != {self.d_transcoder}"
            )

        # Store hook configuration
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.scan = scan
        self.skip_connection = self.transcoders[0].W_skip is not None
        self.exact_chunked_provider = exact_chunked_provider
        self.decoder_chunk_size = int(decoder_chunk_size)
        self.cross_batch_decoder_cache_bytes = (
            DEFAULT_CROSS_BATCH_DECODER_CACHE_BYTES
            if cross_batch_decoder_cache_bytes is None
            else int(cross_batch_decoder_cache_bytes)
        )
        self.checkpoint_lifecycle = checkpoint_lifecycle
        self._decoder_diagnostic_stats = {
            "decoder_chunk_request_count": 0,
            "decoder_chunk_request_bytes": 0,
            "decoder_load_count": 0,
            "decoder_load_bytes": 0,
            "decoder_cache_hit_count": 0,
            "decoder_cache_miss_count": 0,
            "decoder_prefetch_request_count": 0,
            "decoder_prefetch_load_count": 0,
            "decoder_prefetch_load_bytes": 0,
            "decoder_prefetch_cache_hit_count": 0,
            "decoder_prefetch_consume_hit_count": 0,
            "decoder_prefetch_host_wait_count": 0,
            "decoder_prefetch_host_wait_seconds": 0.0,
            "decoder_prefetch_in_flight_count": 0,
            "decoder_prefetch_in_flight_high_watermark": 0,
            "decoder_prefetch_in_flight_bytes": 0,
            "decoder_prefetch_in_flight_bytes_high_watermark": 0,
            "decoder_prefetch_consumer_active_count": 0,
            "decoder_prefetch_consumer_active_bytes": 0,
            "decoder_prefetch_consumer_retained_count": 0,
            "decoder_prefetch_consumer_retained_bytes": 0,
            "decoder_prefetch_consumer_retained_bytes_high_watermark": 0,
            "decoder_prefetch_consumer_retirement_count": 0,
            "decoder_prefetch_consumer_backpressure_count": 0,
            "decoder_prefetch_consumer_backpressure_seconds": 0.0,
            "decoder_prefetch_pipeline_owned_final_page_count": 0,
            "decoder_prefetch_pipeline_owned_final_page_high_watermark": 0,
            "decoder_prefetch_pipeline_owned_final_page_bytes": 0,
            "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": 0,
            "decoder_prefetch_owner_count": 0,
            "decoder_prefetch_owner_high_watermark": 0,
            "decoder_prefetch_owner_open_count": 0,
            "decoder_prefetch_owner_close_count": 0,
        }
        self._decoder_diagnostic_lock = Lock()

    @property
    def architecture(self):
        return "plt"

    @property
    def d_model(self):
        return self.transcoders[0].d_model

    @property
    def dtype(self):
        return self.transcoders[0].dtype

    @property
    def capabilities(self) -> TranscoderCapabilities:
        exact = bool(self.exact_chunked_provider)
        return TranscoderCapabilities(
            architecture="plt",
            checkpoint_format=str(getattr(self.transcoders[0], "weight_format", "standard")),
            supports_exact_chunked_provider=exact,
            supports_compact_row_store=exact,
            supports_decoder_chunk_cache=exact,
            supports_exact_encoder_residency=exact,
            supports_encoder_row_materialization=exact,
            supports_lazy_decoder=any(t.lazy_decoder for t in self.transcoders),
            supports_lazy_encoder=any(t.lazy_encoder for t in self.transcoders),
            supports_lazy_decoder_chunks=exact,
            supports_lazy_encoder_rows=exact,
            supports_exact_row_replay=exact,
            supports_decoder_page_prefetch=bool(
                exact and any(t.lazy_decoder for t in self.transcoders)
            ),
            supports_active_decoder_row_residency=exact,
            supports_phase0_decoder_row_ranges=bool(
                exact
                and all(t.lazy_decoder and t.transcoder_path for t in self.transcoders)
            ),
            decoder_output_topology="same_layer",
            default_decoder_chunk_size=int(self.decoder_chunk_size),
            default_cross_batch_decoder_cache_bytes=int(self.cross_batch_decoder_cache_bytes),
            legacy_exact_chunked_decoder=False,
        )

    def close_decoder_checkpoint_handles(self) -> None:
        """Close provider-owned decoder mappings before page advice.

        Single-layer safetensors loads currently use short-lived context
        managers and therefore own no persistent mappings.
        """

    def decoder_output_layers_for_source(
        self, source_layer: int, active_output_layers=None
    ) -> list[int]:
        if source_layer < 0 or source_layer >= self.n_layers:
            raise ValueError(f"source_layer out of range: {source_layer}")
        if active_output_layers is not None and source_layer not in [
            int(x) for x in active_output_layers
        ]:
            return []
        return [source_layer]

    def decoder_output_slot(self, source_layer: int, output_layer: int) -> int:
        if source_layer < 0 or source_layer >= self.n_layers:
            raise ValueError(f"source_layer out of range: {source_layer}")
        if output_layer != source_layer:
            raise ValueError(
                f"PLT decoder for source_layer {source_layer} only writes to same layer"
            )
        return 0

    def materialize_encoder_rows(self, source_layers, feature_ids):
        source_layers = torch.as_tensor(source_layers, dtype=torch.long).reshape(-1).cpu()
        feature_ids = torch.as_tensor(feature_ids, dtype=torch.long).reshape(-1).cpu()
        if source_layers.numel() != feature_ids.numel():
            raise ValueError("source_layers and feature_ids must have matching lengths")

        first = cast(SingleLayerTranscoder, self.transcoders[0])
        if source_layers.numel() == 0:
            return torch.empty((0, self.d_model), device=first.device, dtype=first.dtype)

        active_encoders = torch.empty(
            (source_layers.numel(), self.d_model),
            device=first.device,
            dtype=first.dtype,
        )
        for layer_id in torch.unique(source_layers, sorted=True).tolist():
            layer_mask = source_layers == int(layer_id)
            layer_rows = torch.nonzero(layer_mask, as_tuple=False).squeeze(-1)
            transcoder = cast(SingleLayerTranscoder, self.transcoders[int(layer_id)])
            active_encoders[layer_rows.to(device=active_encoders.device)] = (
                transcoder.materialize_encoder_rows(feature_ids[layer_rows]).to(
                    device=active_encoders.device, dtype=active_encoders.dtype
                )
            )
        return active_encoders

    @property
    def decoder_device(self) -> torch.device:
        return cast(SingleLayerTranscoder, self.transcoders[0]).device

    def decoder_chunk_nbytes(self, source_layer: int, chunk_id: int) -> int:
        transcoder = cast(SingleLayerTranscoder, self.transcoders[int(source_layer)])
        start = int(chunk_id) * self.decoder_chunk_size
        stop = min(start + self.decoder_chunk_size, transcoder.d_transcoder)
        if start >= transcoder.d_transcoder or stop <= start:
            raise IndexError(f"Decoder chunk {chunk_id} out of range for layer {source_layer}")
        return int((stop - start) * self.d_model * self.dtype.itemsize)

    def get_decoder_chunk(
        self,
        source_layer: int,
        chunk_id: int,
        decoder_cache=None,
        *,
        request_kind: Literal["demand", "prefetch"] = "demand",
    ) -> torch.Tensor:
        cache_key = (int(source_layer), int(chunk_id))
        if decoder_cache is not None:
            cached = decoder_cache.get(cache_key)
            if cached is not None:
                cached_nbytes = int(cached.numel() * cached.element_size())
                with self._decoder_diagnostic_lock:
                    self._decoder_diagnostic_stats["decoder_chunk_request_count"] += 1
                    self._decoder_diagnostic_stats["decoder_chunk_request_bytes"] += cached_nbytes
                    self._decoder_diagnostic_stats["decoder_cache_hit_count"] += 1
                    if request_kind == "prefetch":
                        self._decoder_diagnostic_stats["decoder_prefetch_request_count"] += 1
                        self._decoder_diagnostic_stats["decoder_prefetch_cache_hit_count"] += 1
                return cached
        transcoder = cast(SingleLayerTranscoder, self.transcoders[int(source_layer)])
        result = transcoder.get_decoder_chunk(chunk_id, self.decoder_chunk_size)
        result_nbytes = int(result.numel() * result.element_size())
        with self._decoder_diagnostic_lock:
            self._decoder_diagnostic_stats["decoder_chunk_request_count"] += 1
            self._decoder_diagnostic_stats["decoder_chunk_request_bytes"] += result_nbytes
            self._decoder_diagnostic_stats["decoder_load_count"] += 1
            self._decoder_diagnostic_stats["decoder_load_bytes"] += result_nbytes
            self._decoder_diagnostic_stats["decoder_cache_miss_count"] += 1
            if request_kind == "prefetch":
                self._decoder_diagnostic_stats["decoder_prefetch_request_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_load_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_load_bytes"] += result_nbytes
        if decoder_cache is not None:
            if hasattr(decoder_cache, "put"):
                decoder_cache.put(cache_key, result)
            else:
                decoder_cache[cache_key] = result
        return result

    def record_decoder_prefetch_event(self, event: str, **attrs: object) -> None:
        nbytes = int(attrs.get("nbytes", 0))
        with self._decoder_diagnostic_lock:
            if event == "owner_open":
                for key, value in self._decoder_diagnostic_stats.items():
                    if key.startswith("decoder_prefetch_"):
                        self._decoder_diagnostic_stats[key] = 0.0 if isinstance(value, float) else 0
            if event == "schedule":
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_bytes"] += nbytes
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_high_watermark"] = max(
                    self._decoder_diagnostic_stats["decoder_prefetch_in_flight_high_watermark"],
                    self._decoder_diagnostic_stats["decoder_prefetch_in_flight_count"],
                )
                self._decoder_diagnostic_stats[
                    "decoder_prefetch_in_flight_bytes_high_watermark"
                ] = max(
                    self._decoder_diagnostic_stats[
                        "decoder_prefetch_in_flight_bytes_high_watermark"
                    ],
                    self._decoder_diagnostic_stats["decoder_prefetch_in_flight_bytes"],
                )
            elif event == "consume":
                self._decoder_diagnostic_stats["decoder_prefetch_consume_hit_count"] += 1
                if bool(attrs.get("host_waited", False)):
                    self._decoder_diagnostic_stats["decoder_prefetch_host_wait_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_host_wait_seconds"] += float(
                    attrs.get("host_wait_seconds", 0.0)
                )
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_count"] -= 1
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_bytes"] -= nbytes
            elif event == "release":
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_count"] -= 1
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_bytes"] -= nbytes
            elif event == "handoff":
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_bytes"] += nbytes
            elif event == "consumer_finish":
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_count"] -= 1
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_bytes"] -= nbytes
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_bytes"] += nbytes
                self._decoder_diagnostic_stats[
                    "decoder_prefetch_consumer_retained_bytes_high_watermark"
                ] = max(
                    self._decoder_diagnostic_stats[
                        "decoder_prefetch_consumer_retained_bytes_high_watermark"
                    ],
                    self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_bytes"],
                )
            elif event == "consumer_retire":
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_count"] -= 1
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_bytes"] -= nbytes
                self._decoder_diagnostic_stats["decoder_prefetch_consumer_retirement_count"] += 1
                if bool(attrs.get("backpressure_waited", False)):
                    self._decoder_diagnostic_stats[
                        "decoder_prefetch_consumer_backpressure_count"
                    ] += 1
                self._decoder_diagnostic_stats[
                    "decoder_prefetch_consumer_backpressure_seconds"
                ] += float(attrs.get("backpressure_wait_seconds", 0.0))
            elif event == "owner_open":
                self._decoder_diagnostic_stats["decoder_prefetch_owner_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_owner_open_count"] += 1
                self._decoder_diagnostic_stats["decoder_prefetch_owner_high_watermark"] = max(
                    self._decoder_diagnostic_stats["decoder_prefetch_owner_high_watermark"],
                    self._decoder_diagnostic_stats["decoder_prefetch_owner_count"],
                )
            elif event == "owner_close":
                self._decoder_diagnostic_stats["decoder_prefetch_owner_count"] -= 1
                self._decoder_diagnostic_stats["decoder_prefetch_owner_close_count"] += 1

            owned_count = (
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_count"]
                + self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_count"]
                + self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_count"]
            )
            owned_bytes = (
                self._decoder_diagnostic_stats["decoder_prefetch_in_flight_bytes"]
                + self._decoder_diagnostic_stats["decoder_prefetch_consumer_active_bytes"]
                + self._decoder_diagnostic_stats["decoder_prefetch_consumer_retained_bytes"]
            )
            self._decoder_diagnostic_stats["decoder_prefetch_pipeline_owned_final_page_count"] = (
                owned_count
            )
            self._decoder_diagnostic_stats["decoder_prefetch_pipeline_owned_final_page_bytes"] = (
                owned_bytes
            )
            self._decoder_diagnostic_stats[
                "decoder_prefetch_pipeline_owned_final_page_high_watermark"
            ] = max(
                self._decoder_diagnostic_stats[
                    "decoder_prefetch_pipeline_owned_final_page_high_watermark"
                ],
                owned_count,
            )
            self._decoder_diagnostic_stats[
                "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark"
            ] = max(
                self._decoder_diagnostic_stats[
                    "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark"
                ],
                owned_bytes,
            )

    def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
        from circuit_tracer.transcoder.cross_layer_transcoder import DecoderChunkCache

        cache_bytes = self.cross_batch_decoder_cache_bytes if max_bytes is None else int(max_bytes)
        if cache_bytes <= 0:
            return None
        return DecoderChunkCache(cache_bytes, fingerprint=fingerprint)

    def clear_decoder_block_cache(self, cache) -> None:
        if cache is not None:
            cache.clear()

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "capabilities": self.capabilities,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_transcoder": self.d_transcoder,
            **self._decoder_diagnostic_stats,
        }

    def _decode_sparse_with_decoder_row_ranges(
        self,
        layer: int,
        sparse_acts: torch.Tensor,
        input_acts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecoderRowSeedLayer | None, int, Phase0DecoderRangeTelemetry]:
        """Selectively load unique rows, then replay the canonical chunk groups."""

        sparse_acts = sparse_acts.coalesce()
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()
        unique_feature_ids = torch.unique(feat_idx, sorted=True)
        transcoder = cast(SingleLayerTranscoder, self.transcoders[layer])
        assert transcoder.transcoder_path is not None
        plan = plan_decoder_row_ranges(
            unique_feature_ids,
            d_model=self.d_model,
            d_transcoder=transcoder.d_transcoder,
            itemsize=int(self.dtype.itemsize),
            decoder_chunk_size=self.decoder_chunk_size,
            max_gap_rows=8,
            max_overfetch_fraction=0.25,
            max_range_count=4096,
            max_singleton_range_fraction=0.5,
            max_ranges_per_baseline_page=4,
        )
        if not plan.admitted:
            reconstruct_started = time.perf_counter()
            reconstruction, seed_layer, traversal_bytes = (
                self._decode_sparse_with_decoder_chunks(
                    layer,
                    sparse_acts,
                    input_acts,
                    capture_decoder_row_seed=True,
                )
            )
            reconstruction_seconds = time.perf_counter() - reconstruct_started
            unique_bytes = int(unique_feature_ids.numel()) * self.d_model * int(
                self.dtype.itemsize
            )
            return (
                reconstruction,
                seed_layer,
                traversal_bytes,
                Phase0DecoderRangeTelemetry(
                    requested=True,
                    effective=False,
                    fallback_reason=plan.fallback_reason,
                    planning_seconds=plan.planning_seconds,
                    read_seconds=0.0,
                    gather_seconds=0.0,
                    reconstruction_seconds=reconstruction_seconds,
                    seed_capture_seconds=0.0,
                    unique_row_count=int(unique_feature_ids.numel()),
                    unique_row_bytes=unique_bytes,
                    range_request_count=0,
                    range_rows=(),
                    merged_gap_rows=plan.merged_gap_rows,
                    overfetch_bytes=0,
                    logical_requested_bytes=unique_bytes,
                    logical_materialized_bytes=traversal_bytes,
                    baseline_full_page_count=plan.baseline_full_page_count,
                    baseline_full_page_bytes=plan.baseline_full_page_bytes,
                ),
            )

        key = "w_dec" if transcoder.weight_format == "gemmascope2" else "W_dec"
        compact_rows: torch.Tensor | None = None
        decoder_vectors: torch.Tensor | None = None
        scaled_decoders: torch.Tensor | None = None
        try:
            compact_rows, read_seconds, gather_seconds = load_decoder_row_ranges(
                path=transcoder.transcoder_path,
                key=key,
                plan=plan,
                dtype=self.dtype,
            )
            reconstruction = torch.zeros(
                sparse_acts.shape[0],
                self.d_model,
                device=sparse_acts.device,
                dtype=sparse_acts.dtype,
            )
            reconstruct_started = time.perf_counter()
            if feat_idx.numel() > 0:
                chunk_ids = torch.div(
                    feat_idx, self.decoder_chunk_size, rounding_mode="floor"
                )
                unique_chunk_ids = torch.div(
                    unique_feature_ids,
                    self.decoder_chunk_size,
                    rounding_mode="floor",
                )
                for chunk_id_tensor in torch.unique(chunk_ids, sorted=True):
                    chunk_mask = chunk_ids == chunk_id_tensor
                    unique_chunk_mask = unique_chunk_ids == chunk_id_tensor
                    unique_destinations = unique_chunk_mask.nonzero(
                        as_tuple=False
                    ).flatten()
                    chunk_feature_ids = unique_feature_ids[unique_chunk_mask]
                    occurrence_rows = torch.searchsorted(
                        chunk_feature_ids, feat_idx[chunk_mask]
                    )
                    decoder_vectors = compact_rows.index_select(
                        0, unique_destinations.to(device="cpu")
                    ).to(
                        device=sparse_acts.device,
                        dtype=sparse_acts.dtype,
                    )
                    scaled_decoders = (
                        decoder_vectors[occurrence_rows] * values[chunk_mask, None]
                    )
                    reconstruction.index_add_(
                        0, pos_idx[chunk_mask], scaled_decoders
                    )
            if transcoder.W_skip is not None:
                assert input_acts is not None, (
                    "Transcoder has skip connection but no input_acts were provided"
                )
                reconstruction = reconstruction + transcoder.compute_skip(input_acts)
            reconstruction = reconstruction + transcoder.b_dec.to(
                device=reconstruction.device,
                dtype=reconstruction.dtype,
            )
            reconstruction_seconds = time.perf_counter() - reconstruct_started
            seed_layer = None
            if compact_rows.numel():
                seed_layer = DecoderRowSeedLayer(
                    source_layer=layer,
                    output_layers=(layer,),
                    feature_ids=plan.unique_feature_ids,
                    rows=compact_rows.unsqueeze(1),
                )
            materialized_rows = sum(row_range.materialized_rows for row_range in plan.ranges)
            materialized_bytes = materialized_rows * self.d_model * int(self.dtype.itemsize)
            requested_bytes = int(unique_feature_ids.numel()) * self.d_model * int(
                self.dtype.itemsize
            )
            telemetry = Phase0DecoderRangeTelemetry(
                requested=True,
                effective=True,
                fallback_reason=None,
                planning_seconds=plan.planning_seconds,
                read_seconds=read_seconds,
                gather_seconds=gather_seconds,
                reconstruction_seconds=reconstruction_seconds,
                seed_capture_seconds=0.0,
                unique_row_count=int(unique_feature_ids.numel()),
                unique_row_bytes=requested_bytes,
                range_request_count=len(plan.ranges),
                range_rows=tuple(row_range.materialized_rows for row_range in plan.ranges),
                merged_gap_rows=plan.merged_gap_rows,
                overfetch_bytes=materialized_bytes - requested_bytes,
                logical_requested_bytes=requested_bytes,
                logical_materialized_bytes=materialized_bytes,
                baseline_full_page_count=plan.baseline_full_page_count,
                baseline_full_page_bytes=plan.baseline_full_page_bytes,
            )
            return reconstruction, seed_layer, materialized_bytes, telemetry
        except BaseException:
            decoder_vectors = None
            scaled_decoders = None
            compact_rows = None
            raise

    def _decode_sparse_with_decoder_chunks(
        self,
        layer: int,
        sparse_acts: torch.Tensor,
        input_acts: torch.Tensor | None = None,
        *,
        capture_decoder_row_seed: bool = False,
    ) -> tuple[torch.Tensor, DecoderRowSeedLayer | None, int]:
        sparse_acts = sparse_acts.coalesce()
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()
        reconstruction = torch.zeros(
            sparse_acts.shape[0],
            self.d_model,
            device=sparse_acts.device,
            dtype=sparse_acts.dtype,
        )
        unique_feature_ids = torch.unique(feat_idx, sorted=True)
        seed_rows: torch.Tensor | None = None
        traversal_bytes = 0

        try:
            if feat_idx.numel() > 0:
                chunk_ids = torch.div(feat_idx, self.decoder_chunk_size, rounding_mode="floor")
                unique_chunk_ids = torch.div(
                    unique_feature_ids, self.decoder_chunk_size, rounding_mode="floor"
                )
                for chunk_id_tensor in torch.unique(chunk_ids, sorted=True):
                    chunk_id = int(chunk_id_tensor.item())
                    chunk_mask = chunk_ids == chunk_id_tensor
                    local_feat_idx = (
                        feat_idx[chunk_mask] - chunk_id * self.decoder_chunk_size
                    ).long()
                    decoder_chunk = self.get_decoder_chunk(layer, chunk_id)
                    traversal_bytes += int(decoder_chunk.numel() * decoder_chunk.element_size())
                    if capture_decoder_row_seed:
                        if seed_rows is None:
                            seed_rows = torch.empty(
                                (int(unique_feature_ids.numel()), 1, self.d_model),
                                device=decoder_chunk.device,
                                dtype=decoder_chunk.dtype,
                            )
                        unique_chunk_mask = unique_chunk_ids == chunk_id_tensor
                        unique_destinations = unique_chunk_mask.nonzero(as_tuple=False).flatten()
                        unique_local_ids = (
                            unique_feature_ids[unique_chunk_mask]
                            - chunk_id * self.decoder_chunk_size
                        ).to(device=decoder_chunk.device, dtype=torch.long)
                        seed_rows[unique_destinations.to(device=decoder_chunk.device)] = (
                            decoder_chunk[unique_local_ids, :1]
                        )
                    decoder_vectors = decoder_chunk[:, 0].to(
                        device=sparse_acts.device,
                        dtype=sparse_acts.dtype,
                        non_blocking=decoder_chunk.device.type == "cuda",
                    )
                    scaled_decoders = decoder_vectors[local_feat_idx] * values[chunk_mask, None]
                    reconstruction.index_add_(0, pos_idx[chunk_mask], scaled_decoders)

        except BaseException:
            decoder_vectors = None
            decoder_chunk = None
            seed_rows = None
            raise

        transcoder = cast(SingleLayerTranscoder, self.transcoders[layer])
        if transcoder.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + transcoder.compute_skip(input_acts)
        reconstruction = reconstruction + transcoder.b_dec.to(
            device=reconstruction.device, dtype=reconstruction.dtype
        )
        seed_layer = None
        if capture_decoder_row_seed and seed_rows is not None:
            seed_layer = DecoderRowSeedLayer(
                source_layer=layer,
                output_layers=(layer,),
                feature_ids=unique_feature_ids.detach()
                .to(device="cpu", dtype=torch.long)
                .contiguous(),
                rows=seed_rows.detach().to(device="cpu").contiguous(),
            )
        return reconstruction, seed_layer, traversal_bytes

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx: int) -> SingleLayerTranscoder:
        return self.transcoders[idx]  # type: ignore

    def __iter__(self) -> Iterator[SingleLayerTranscoder]:
        return iter(self.transcoders)  # type: ignore

    def apply_activation_function(self, layer_id, features):
        return self.transcoders[layer_id].activation_function(features)  # type: ignore

    def compute_skip(self, layer_id: int, inputs):
        return self.transcoders[layer_id].compute_skip(inputs)  # type: ignore

    def encode(self, input_acts):
        return torch.stack(
            [transcoder.encode(input_acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def _get_decoder_vectors(self, layer_id, features):
        return self.transcoders[layer_id]._get_decoder_vectors(features)  # type: ignore

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()

        all_layer_idx, all_pos_idx, all_feat_idx = features.indices()
        all_activations = features.values()
        all_scaled_decoder_vectors = []
        for unique_layer in all_layer_idx.unique():
            layer_mask = all_layer_idx == unique_layer
            feat_idx = all_feat_idx[layer_mask]
            activations = all_activations[layer_mask]

            decoder_vectors = self._get_decoder_vectors(unique_layer.item(), feat_idx)

            # Multiply each activation by its corresponding decoder vector
            scaled_decoder_vectors = activations.unsqueeze(-1) * decoder_vectors
            all_scaled_decoder_vectors.append(scaled_decoder_vectors)

        all_scaled_decoder_vectors = torch.cat(all_scaled_decoder_vectors)
        encoder_mapping = torch.arange(features._nnz(), device=features.device)

        return (
            all_pos_idx,
            all_layer_idx,
            all_feat_idx,
            all_scaled_decoder_vectors,
            encoder_mapping,
        )

    def decode(self, acts, input_acts: torch.Tensor | None):
        return torch.stack(
            [
                transcoder.decode(acts[i], None if input_acts is None else input_acts[i])
                for i, transcoder in enumerate[SingleLayerTranscoder](self.transcoders)  # type: ignore
            ],
            dim=0,
        )

    def compute_attribution_components(
        self,
        mlp_inputs: torch.Tensor,
        zero_positions: slice = slice(0, 1),
        sparsification: SparsificationConfig | None = None,
        *,
        materialize_encoder_vecs: bool = True,
        decoder_active_row_residency: bool = False,
        phase0_decoder_row_ranges: bool = False,
        decoder_active_row_max_bytes: int = 0,
    ) -> AttributionComponents:
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            mlp_inputs: (n_layers, n_pos, d_model) tensor of MLP inputs
            zero_positions: (slice) slice indicating which positions to zero out

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse (n_layers, n_pos, d_transcoder) activations
                - reconstruction: (n_layers, n_pos, d_model) reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
        """
        device = mlp_inputs.device

        sparse_acts_list = []

        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            sparse_acts, _ = transcoder.encode_sparse(
                mlp_inputs[layer], zero_positions=zero_positions, return_encoder_vectors=False
            )
            sparse_acts_list.append(sparse_acts)

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        sparsification_stats = None
        if sparsification is not None:
            selected_indices, sparsification_stats = select_candidate_feature_indices(
                activation_matrix, sparsification
            )
            activation_matrix = filter_sparse_activations(activation_matrix, selected_indices)

        reconstruction = torch.zeros_like(mlp_inputs)
        encoder_vectors = []
        decoder_vectors = []
        layer_ids, pos_ids, feat_ids = activation_matrix.indices()

        if self.exact_chunked_provider:
            seed_estimated_bytes = (
                int(activation_matrix._nnz()) * self.d_model * int(self.dtype.itemsize)
            )
            seed_refusal_reason = None
            if not decoder_active_row_residency:
                seed_refusal_reason = "not_requested"
            elif int(decoder_active_row_max_bytes) <= 0:
                seed_refusal_reason = "max_bytes_nonpositive"
            elif seed_estimated_bytes > int(decoder_active_row_max_bytes):
                seed_refusal_reason = "phase0_occurrence_bytes_exceed_max"
            capture_seed = seed_refusal_reason is None
            seed_started = time.perf_counter()
            load_count_before = int(self._decoder_diagnostic_stats["decoder_load_count"])
            load_bytes_before = int(self._decoder_diagnostic_stats["decoder_load_bytes"])
            seed_layers: list[DecoderRowSeedLayer | None] = []
            seed_traversal_bytes = 0
            range_telemetry_layers: list[Phase0DecoderRangeTelemetry] = []
            for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
                layer_mask = layer_ids == layer
                layer_sparse = torch.sparse_coo_tensor(
                    torch.stack((pos_ids[layer_mask], feat_ids[layer_mask])),
                    activation_matrix.values()[layer_mask],
                    size=(mlp_inputs.shape[1], transcoder.d_transcoder),
                    device=device,
                    dtype=activation_matrix.dtype,
                ).coalesce()
                if capture_seed and phase0_decoder_row_ranges:
                    layer_reconstruction, seed_layer, traversal_bytes, range_telemetry = (
                        self._decode_sparse_with_decoder_row_ranges(
                            layer,
                            layer_sparse,
                            mlp_inputs[layer],
                        )
                    )
                    range_telemetry_layers.append(range_telemetry)
                else:
                    layer_reconstruction, seed_layer, traversal_bytes = (
                        self._decode_sparse_with_decoder_chunks(
                            layer,
                            layer_sparse,
                            mlp_inputs[layer],
                            capture_decoder_row_seed=capture_seed,
                        )
                    )
                reconstruction[layer] = layer_reconstruction
                seed_layers.append(seed_layer)
                seed_traversal_bytes += traversal_bytes

            decoder_row_seed = None
            if capture_seed:
                seed_capture_seconds = time.perf_counter() - seed_started
                decoder_row_seed = DecoderRowSeed(
                    layers=tuple(seed_layers),
                    source_fingerprint=provider_fingerprint(self),
                    occurrence_estimated_bytes=seed_estimated_bytes,
                    capture_seconds=seed_capture_seconds,
                    shared_traversal_bytes=seed_traversal_bytes,
                    shared_decoder_load_count=(
                        int(self._decoder_diagnostic_stats["decoder_load_count"])
                        - load_count_before
                    ),
                    shared_decoder_load_bytes=(
                        int(self._decoder_diagnostic_stats["decoder_load_bytes"])
                        - load_bytes_before
                    ),
                    phase0_decoder_range_telemetry=(
                        combine_phase0_decoder_range_telemetry(
                            range_telemetry_layers, seed_capture_seconds=seed_capture_seconds
                        )
                    ),
                )

            encoder_vecs = (
                self.materialize_encoder_rows(layer_ids.tolist(), feat_ids.tolist())
                if materialize_encoder_vecs
                else torch.empty((0, self.d_model), device=device, dtype=self.dtype)
            )
            empty_decoder_vecs = torch.empty(
                (0, self.d_model), device=device, dtype=activation_matrix.dtype
            )
            empty_locations = torch.empty((2, 0), device=device, dtype=torch.long)
            return AttributionComponents(
                activation_matrix=activation_matrix,
                reconstruction=reconstruction,
                encoder_vectors=encoder_vecs,
                decoder_vectors=empty_decoder_vecs,
                encoder_to_decoder_map=torch.empty((0,), device=device, dtype=torch.long),
                decoder_locations=empty_locations,
                chunked_decoder_state={
                    "source_layers": layer_ids,
                    "positions": pos_ids,
                    "feature_ids": feat_ids,
                    "activation_values": activation_matrix.values(),
                },
                sparsification_stats=sparsification_stats,
                decoder_row_seed=decoder_row_seed,
                decoder_row_seed_refusal_reason=seed_refusal_reason,
                decoder_row_seed_estimated_bytes=seed_estimated_bytes,
            )

        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            layer_mask = layer_ids == layer
            layer_sparse = torch.sparse_coo_tensor(
                torch.stack((pos_ids[layer_mask], feat_ids[layer_mask])),
                activation_matrix.values()[layer_mask],
                size=(mlp_inputs.shape[1], transcoder.d_transcoder),
                device=device,
                dtype=activation_matrix.dtype,
            ).coalesce()
            _, layer_feat_ids = layer_sparse.indices()
            encoder_vectors.append(transcoder.W_enc[layer_feat_ids])
            reconstruction[layer], active_decoders = transcoder.decode_sparse(
                layer_sparse, mlp_inputs[layer]
            )
            decoder_vectors.append(active_decoders)

        encoder_to_decoder_map = torch.arange(activation_matrix._nnz(), device=device)

        return AttributionComponents(
            activation_matrix=activation_matrix,
            reconstruction=reconstruction,
            encoder_vectors=torch.cat(encoder_vectors, dim=0)
            if encoder_vectors
            else torch.empty((0, self.d_model), device=device, dtype=self.dtype),
            decoder_vectors=torch.cat(decoder_vectors, dim=0)
            if decoder_vectors
            else torch.empty((0, self.d_model), device=device, dtype=activation_matrix.dtype),
            encoder_to_decoder_map=encoder_to_decoder_map,
            decoder_locations=activation_matrix.indices()[:2],
            sparsification_stats=sparsification_stats,
        )

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        return self.transcoders[layer_id].encode(
            x, apply_activation_function=apply_activation_function
        )  # type: ignore

    def to_safetensors(self, save_dir: str):
        """Save all transcoders in the set to safetensors files.

        Saves each transcoder as layer_{i}.safetensors in the specified directory.

        Args:
            save_dir: Directory path where the safetensors files will be saved
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, transcoder in enumerate(self.transcoders):
            save_path = os.path.join(save_dir, f"layer_{i}.safetensors")
            transcoder.to_safetensors(save_path)  # type: ignore


def load_gemma_scope_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    **kwargs,
) -> SingleLayerTranscoder:
    if device is None:
        device = get_default_device()
    if os.path.isfile(path):
        path_to_params = path
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-transcoders",
            filename=path,
            revision=revision,
            force_download=False,
        )

    # load the parameters, have to rename the threshold key,
    # as ours is nested inside the activation_function module
    param_dict = np.load(path_to_params)
    param_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in param_dict.items()}
    param_dict["activation_function.threshold"] = param_dict["threshold"]
    param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()
    del param_dict["threshold"]

    # create the transcoders
    d_transcoder, d_model = param_dict["W_enc"].shape

    # JumpReLU; will get loaded via load_state_dict
    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(d_model, d_transcoder, activation_function, layer)
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_relu_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
):
    if device is None:
        device = get_default_device()

    param_dict = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            if lazy_encoder and k == "W_enc":
                continue
            if lazy_decoder and k == "W_dec":
                continue
            param_dict[k] = f.get_tensor(k)

    d_sae = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    assert param_dict.get("log_thresholds") is None
    activation_function = (
        JumpReLU(param_dict["activation_function.threshold"], 0.1)
        if "activation_function.threshold" in param_dict
        else F.relu
    )
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_sae,
            activation_function,
            layer,
            skip_connection=param_dict.get("W_skip") is not None,
            transcoder_path=path,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype)


def load_gemma_scope_2_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = False,
) -> SingleLayerTranscoder:
    """Load a SingleLayerTranscoder from a GemmaScope2 JumpReLUSAE checkpoint.

    Args:
        path: Path to the checkpoint file
        layer: Layer index for the transcoder
        device: Device to load to
        dtype: Data type to use
        lazy_encoder: Whether to use lazy loading for encoder weights
        lazy_decoder: Whether to use lazy loading for decoder weights

    Returns:
        SingleLayerTranscoder: The loaded transcoder
    """
    if device is None:
        device = get_default_device()

    with safe_open(path, framework="pt", device=device.type) as f:
        b_enc = f.get_tensor("b_enc").to(device=device, dtype=dtype)
        b_dec = f.get_tensor("b_dec").to(device=device, dtype=dtype)
        param_dict = {
            "b_enc": b_enc,
            "b_dec": b_dec,
            "activation_function.threshold": f.get_tensor("threshold").to(
                device=device, dtype=dtype
            ),
        }
        if not lazy_encoder:
            param_dict["W_enc"] = (
                f.get_tensor("w_enc").T.contiguous().to(device=device, dtype=dtype)
            )
        if not lazy_decoder:
            param_dict["W_dec"] = f.get_tensor("w_dec").to(device=device, dtype=dtype)
        if "affine_skip_connection" in f.keys():
            param_dict["W_skip"] = (
                f.get_tensor("affine_skip_connection").T.contiguous().to(device=device, dtype=dtype)
            )

    d_transcoder = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)

    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_transcoder,
            activation_function,
            layer,
            skip_connection="W_skip" in param_dict,
            transcoder_path=path,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
            weight_format="gemmascope2",
        )

    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_transcoder_set(
    transcoder_paths: dict,
    scan: str,
    feature_input_hook: str,
    feature_output_hook: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None] = None,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
    exact_chunked_provider: bool = False,
    decoder_chunk_size: int = 1024,
    cross_batch_decoder_cache_bytes: int | None = DEFAULT_CROSS_BATCH_DECODER_CACHE_BYTES,
    checkpoint_asset_scope: CheckpointAssetScope = CheckpointAssetScope.SHARED,
    checkpoint_prefault_budget_bytes: int = 0,
) -> TranscoderSet:
    if device is None:
        device = get_default_device()
    """Loads either a preset set of transcoders, or a set specified by a file.

    Args:
        transcoder_paths: Dictionary mapping layer indices to transcoder paths
        scan: Scan identifier
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        device (torch.device | None, optional): Device to load to
        dtype (torch.dtype, optional): Data type to use
        special_load_fn: Which special loading function to use
        lazy_encoder: Whether to use lazy loading for encoder weights
        lazy_decoder: Whether to use lazy loading for decoder weights

    Returns:
        TranscoderSet: The loaded transcoder set with all configuration
    """
    if exact_chunked_provider:
        decoder_chunk_size = _validate_decoder_chunk_size(decoder_chunk_size)

    transcoders = {}
    for layer in range(len(transcoder_paths)):
        load_fn = select_single_layer_transcoder_load_fn(transcoder_paths[layer], special_load_fn)

        transcoders[layer] = load_fn(
            transcoder_paths[layer],
            layer,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )

    manifest_discovery = build_checkpoint_manifest(
        "plt",
        transcoder_paths,
        scope=checkpoint_asset_scope,
    )
    checkpoint_lifecycle = (
        ProviderCheckpointLifecycle(
            manifest_discovery.manifest,
            prefault_budget_bytes=checkpoint_prefault_budget_bytes,
        )
        if manifest_discovery.manifest is not None
        else None
    )
    provider = TranscoderSet(
        transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
        exact_chunked_provider=exact_chunked_provider,
        decoder_chunk_size=decoder_chunk_size,
        cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
        checkpoint_lifecycle=checkpoint_lifecycle,
    )
    provider.checkpoint_manifest_diagnostics = manifest_discovery.diagnostics
    return provider
