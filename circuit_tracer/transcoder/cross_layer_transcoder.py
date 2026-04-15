import glob
import os
import time
from collections import OrderedDict
from typing import cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from circuit_tracer.attribution.sparsification import (
    SparsificationConfig,
    filter_sparse_activations,
    select_candidate_feature_indices,
)
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils import get_default_device


DEFAULT_EXACT_DECODER_CHUNK_SIZE = 1024


class DecoderChunkCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_resident = 0
        self._entries: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def get(self, key: tuple[int, int]) -> torch.Tensor | None:
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[int, int], value: torch.Tensor) -> list[tuple[tuple[int, int], int]]:
        value_nbytes = self._tensor_nbytes(value)
        evicted: list[tuple[tuple[int, int], int]] = []
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.bytes_resident -= self._tensor_nbytes(existing)

        if self.max_bytes <= 0 or value_nbytes > self.max_bytes:
            return evicted

        while self._entries and self.bytes_resident + value_nbytes > self.max_bytes:
            evicted_key, evicted_value = self._entries.popitem(last=False)
            evicted_nbytes = self._tensor_nbytes(evicted_value)
            self.bytes_resident -= evicted_nbytes
            evicted.append((evicted_key, evicted_nbytes))

        self._entries[key] = value
        self.bytes_resident += value_nbytes
        return evicted

    def clear(self) -> None:
        self._entries.clear()
        self.bytes_resident = 0


class CrossLayerTranscoder(torch.nn.Module):
    """
    A cross-layer transcoder (CLT) where features read from one layer and write to all
    subsequent layers.

    Cross-layer transcoders are the core architecture enabling the circuit tracing methodology.
    Unlike per-layer transcoders, CLT features can "bridge over" multiple MLP layers, allowing
    a single feature to represent computation that spans the entire forward pass. This dramatically
    shortens paths in attribution graphs by collapsing amplification chains into single features.

    Each CLT feature has:
    - One encoder that reads from the residual stream at a specific layer
    - Multiple decoders that can write to all subsequent MLP outputs
    - The ability to represent cross-layer superposition where related computation
    is distributed across multiple transformer layers

    A single CLT provides an alternative to using multiple per-layer transcoders (managed by
    TranscoderSet) for feature-based model interpretation and replacement.

    Attributes:
        n_layers: Number of transformer layers the CLT spans
        d_transcoder: Number of features per layer
        d_model: Dimension of transformer residual stream
        W_enc: Encoder weights for each layer [n_layers, d_transcoder, d_model]
        W_dec: Decoder weights (lazily loaded) for cross-layer outputs
        b_enc: Encoder biases [n_layers, d_transcoder]
        b_dec: Decoder biases [n_layers, d_model]
        W_skip: Optional skip connection weights (https://arxiv.org/abs/2501.18823)
        activation_function: Sparsity-inducing nonlinearity (default: ReLU)
        lazy_decoder: Whether to load decoder weights on-demand to save memory
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier for feature visualization
    """

    def __init__(
        self,
        n_layers: int,
        d_transcoder: int,
        d_model: int,
        activation_function: str = "relu",
        skip_connection: bool = False,
        lazy_decoder=True,
        lazy_encoder=False,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
        scan: str | list[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        clt_path: str | None = None,
        layer_paths: dict[int, str] | None = None,
        weight_format: str = "standard",
        exact_chunked_decoder: bool = False,
        decoder_chunk_size: int = DEFAULT_EXACT_DECODER_CHUNK_SIZE,
        cross_batch_decoder_cache_bytes: int | None = None,
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.n_layers = n_layers
        self.d_transcoder = d_transcoder
        self.d_model = d_model
        self.lazy_decoder = lazy_decoder
        self.lazy_encoder = lazy_encoder
        self.clt_path = clt_path
        self.layer_paths = layer_paths
        self.weight_format = weight_format
        self.exact_chunked_decoder = exact_chunked_decoder
        self.decoder_chunk_size = decoder_chunk_size
        if cross_batch_decoder_cache_bytes is None:
            cross_batch_decoder_cache_bytes = 0
        self.cross_batch_decoder_cache_bytes = max(0, int(cross_batch_decoder_cache_bytes))
        self._diagnostic_stats = self._make_empty_diagnostic_stats()
        self._trace_logger = None
        self._trace_chunk_interval = 16
        self._trace_decoder_load_interval = 32

        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.skip_connection = skip_connection
        self.scan = scan

        if activation_function == "jump_relu":
            self.activation_function = JumpReLU(
                torch.zeros(n_layers, 1, d_transcoder, device=device, dtype=dtype)
            )
        elif activation_function == "relu":
            self.activation_function = F.relu
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")

        if not lazy_encoder:
            self.W_enc = torch.nn.Parameter(
                torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
            )

        self.b_dec = torch.nn.Parameter(torch.zeros(n_layers, d_model, device=device, dtype=dtype))
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)
        )

        if not lazy_decoder:
            self.W_dec = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(
                            d_transcoder,
                            n_layers - i,
                            d_model,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.W_dec = None

        if skip_connection:
            self.W_skip = torch.nn.Parameter(
                torch.zeros(n_layers, d_model, d_model, device=device, dtype=dtype)
            )
        else:
            self.W_skip = None

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return self.b_enc.device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    @staticmethod
    def _make_empty_diagnostic_stats() -> dict[str, object]:
        return {
            "encoder_load_count": 0,
            "encoder_load_seconds": 0.0,
            "encoder_load_by_layer": {},
            "decoder_load_count": 0,
            "decoder_load_seconds": 0.0,
            "decoder_load_by_layer": {},
            "decoder_cache_hit_count": 0,
            "decoder_cache_miss_count": 0,
            "decoder_cache_eviction_count": 0,
            "decoder_cache_skip_count": 0,
            "decoder_cache_auto_disable_count": 0,
            "decoder_cache_bytes_resident": 0,
            "decoder_cache_max_bytes": 0,
            "encode_sparse_seconds": 0.0,
            "encode_sparse_by_layer": {},
            "encode_sparse_active_features_by_layer": {},
            "reconstruction_chunk_count": 0,
            "reconstruction_seconds": 0.0,
            "reconstruction_by_layer": {},
            "reconstruction_chunks_by_layer": {},
        }

    def reset_diagnostic_stats(self) -> None:
        self._diagnostic_stats = self._make_empty_diagnostic_stats()

    def configure_trace_logging(
        self,
        logger=None,
        *,
        chunk_interval: int = 16,
        decoder_load_interval: int = 32,
    ) -> None:
        self._trace_logger = logger
        self._trace_chunk_interval = max(1, chunk_interval)
        self._trace_decoder_load_interval = max(1, decoder_load_interval)

    def emit_trace_event(self, event: str, **fields: object) -> None:
        if self._trace_logger is None:
            return
        payload = ", ".join(f"{key}={value}" for key, value in fields.items())
        message = f"TRACE {event}"
        if payload:
            message = f"{message} | {payload}"
        self._trace_logger(message)

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for key, value in self._diagnostic_stats.items():
            snapshot[key] = dict(value) if isinstance(value, dict) else value
        return snapshot

    def _add_diagnostic_value(self, key: str, value: float) -> None:
        current = cast(float, self._diagnostic_stats.get(key, 0.0))
        self._diagnostic_stats[key] = current + value

    def _add_diagnostic_layer_value(self, key: str, layer_id: int, value: float) -> None:
        layer_stats = cast(dict[int, float], self._diagnostic_stats.setdefault(key, {}))
        layer_stats[layer_id] = layer_stats.get(layer_id, 0.0) + value

    def _move_lazy_slice_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == self.device and tensor.dtype == self.dtype:
            return tensor

        if tensor.device.type == "cpu" and self.device.type == "cuda":
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
            return tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

        return tensor.to(device=self.device, dtype=self.dtype)

    def create_decoder_block_cache(self, max_bytes: int | None = None) -> DecoderChunkCache | None:
        if not self.lazy_decoder:
            return None

        cache_budget = self.cross_batch_decoder_cache_bytes if max_bytes is None else max_bytes
        cache_budget = max(0, int(cache_budget))
        self._diagnostic_stats["decoder_cache_max_bytes"] = cache_budget
        self._diagnostic_stats["decoder_cache_bytes_resident"] = 0
        if cache_budget <= 0:
            return None

        self.emit_trace_event("decoder.cache.init", max_bytes=cache_budget)
        return DecoderChunkCache(cache_budget)

    def clear_decoder_block_cache(self, cache: DecoderChunkCache | None) -> None:
        if cache is None:
            self._diagnostic_stats["decoder_cache_bytes_resident"] = 0
            return

        cache.clear()
        self._diagnostic_stats["decoder_cache_bytes_resident"] = 0
        self.emit_trace_event("decoder.cache.clear", max_bytes=cache.max_bytes)

    def note_decoder_cache_auto_disabled(self, reason: str) -> None:
        self._add_diagnostic_value("decoder_cache_auto_disable_count", 1)
        self._diagnostic_stats["decoder_cache_bytes_resident"] = 0
        self.emit_trace_event("decoder.cache.auto_disabled", reason=reason)

    def _record_decoder_cache_resident_bytes(self, cache: DecoderChunkCache | None) -> None:
        self._diagnostic_stats["decoder_cache_bytes_resident"] = (
            0 if cache is None else cache.bytes_resident
        )

    def _record_decoder_cache_hit(
        self, cache: DecoderChunkCache | None, *, layer_id: int, chunk_id: int
    ) -> None:
        self._add_diagnostic_value("decoder_cache_hit_count", 1)
        self._record_decoder_cache_resident_bytes(cache)
        hit_count = int(cast(float, self._diagnostic_stats["decoder_cache_hit_count"]))
        if hit_count <= 3 or hit_count % self._trace_decoder_load_interval == 0:
            self.emit_trace_event(
                "decoder.cache.hit",
                source_layer=layer_id,
                chunk_id=chunk_id,
                hit_count=hit_count,
                resident_bytes=self._diagnostic_stats["decoder_cache_bytes_resident"],
            )

    def _record_decoder_cache_miss(
        self,
        cache: DecoderChunkCache | None,
        *,
        layer_id: int,
        chunk_id: int,
    ) -> None:
        self._add_diagnostic_value("decoder_cache_miss_count", 1)
        self._record_decoder_cache_resident_bytes(cache)
        miss_count = int(cast(float, self._diagnostic_stats["decoder_cache_miss_count"]))
        if miss_count <= 3 or miss_count % self._trace_decoder_load_interval == 0:
            self.emit_trace_event(
                "decoder.cache.miss",
                source_layer=layer_id,
                chunk_id=chunk_id,
                miss_count=miss_count,
            )

    def _record_decoder_cache_skip(
        self,
        cache: DecoderChunkCache | None,
        *,
        layer_id: int,
        chunk_id: int,
        chunk_bytes: int,
    ) -> None:
        self._add_diagnostic_value("decoder_cache_skip_count", 1)
        self._record_decoder_cache_resident_bytes(cache)
        self.emit_trace_event(
            "decoder.cache.skip",
            source_layer=layer_id,
            chunk_id=chunk_id,
            chunk_bytes=chunk_bytes,
            max_bytes=0 if cache is None else cache.max_bytes,
        )

    def _record_decoder_cache_put(
        self,
        cache: DecoderChunkCache | None,
        *,
        layer_id: int,
        chunk_id: int,
        evicted: list[tuple[tuple[int, int], int]],
    ) -> None:
        if evicted:
            self._add_diagnostic_value("decoder_cache_eviction_count", len(evicted))
            for (evicted_layer, evicted_chunk), evicted_nbytes in evicted:
                self.emit_trace_event(
                    "decoder.cache.evict",
                    source_layer=evicted_layer,
                    chunk_id=evicted_chunk,
                    evicted_bytes=evicted_nbytes,
                )
        self._record_decoder_cache_resident_bytes(cache)
        self.emit_trace_event(
            "decoder.cache.store",
            source_layer=layer_id,
            chunk_id=chunk_id,
            resident_bytes=self._diagnostic_stats["decoder_cache_bytes_resident"],
        )

    def _get_encoder_weights(self, layer_id=None):
        """Get encoder weights, loading from disk if lazy."""
        if not self.lazy_encoder:
            return self.W_enc if layer_id is None else self.W_enc[layer_id]

        start = time.perf_counter()

        if self.layer_paths is not None:
            if layer_id is not None:
                with safe_open(
                    self.layer_paths[layer_id], framework="pt", device=str(self.device)
                ) as f:
                    result = (
                        f.get_tensor("w_enc").transpose(-1, -2).to(dtype=self.dtype).contiguous()
                    )
                self._add_diagnostic_value("encoder_load_count", 1)
                self._add_diagnostic_value("encoder_load_seconds", time.perf_counter() - start)
                self._add_diagnostic_layer_value(
                    "encoder_load_by_layer", layer_id, time.perf_counter() - start
                )
                return result

            W_enc = torch.zeros(
                self.n_layers,
                self.d_transcoder,
                self.d_model,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(self.n_layers):
                with safe_open(self.layer_paths[i], framework="pt", device=str(self.device)) as f:
                    W_enc[i] = f.get_tensor("w_enc").transpose(-1, -2).to(dtype=self.dtype)
            self._add_diagnostic_value("encoder_load_count", self.n_layers)
            self._add_diagnostic_value("encoder_load_seconds", time.perf_counter() - start)
            return W_enc

        assert self.clt_path is not None, "CLT path is not set"
        if layer_id is not None:
            # Load single layer encoder
            enc_file = os.path.join(self.clt_path, f"W_enc_{layer_id}.safetensors")
            with safe_open(enc_file, framework="pt", device=str(self.device)) as f:
                result = f.get_tensor(f"W_enc_{layer_id}").to(dtype=self.dtype)
            self._add_diagnostic_value("encoder_load_count", 1)
            self._add_diagnostic_value("encoder_load_seconds", time.perf_counter() - start)
            self._add_diagnostic_layer_value(
                "encoder_load_by_layer", layer_id, time.perf_counter() - start
            )
            return result

        # Load all encoder weights
        W_enc = torch.zeros(
            self.n_layers,
            self.d_transcoder,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        )
        for i in range(self.n_layers):
            enc_file = os.path.join(self.clt_path, f"W_enc_{i}.safetensors")
            with safe_open(enc_file, framework="pt", device=str(self.device)) as f:
                W_enc[i] = f.get_tensor(f"W_enc_{i}").to(dtype=self.dtype)
        self._add_diagnostic_value("encoder_load_count", self.n_layers)
        self._add_diagnostic_value("encoder_load_seconds", time.perf_counter() - start)
        return W_enc

    def encode(self, x):
        W_enc = self._get_encoder_weights()
        features = torch.einsum("lbd,lfd->lbf", x, W_enc) + self.b_enc[:, None]
        return self.activation_function(features)

    def apply_activation_function(self, layer_id, features):
        if isinstance(self.activation_function, JumpReLU):
            thresholds = self.activation_function.threshold
            mask = features > thresholds[layer_id]
            features = features * mask
        else:
            features = self.activation_function(features)
        return features

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        W_enc_layer = self._get_encoder_weights(layer_id)
        features = torch.einsum("...d,fd->...f", x, W_enc_layer) + self.b_enc[layer_id]
        if not apply_activation_function:
            return features

        return self.apply_activation_function(layer_id, features)

    def encode_sparse(
        self,
        x,
        zero_positions: slice = slice(0, 1),
        *,
        return_encoder_vectors: bool = True,
    ):
        """Encode input to sparse activations, processing one layer at a time for memory efficiency.

        This method processes layers sequentially and converts to sparse format immediately
        to minimize peak memory usage, especially beneficial for large cross-layer transcoders.

        Args:
            x: Input tensor of shape (n_layers, n_pos, d_model)
            zero_first_pos: Whether to zero out position 0

        Returns:
            sparse_features: Sparse tensor of shape (n_layers, n_pos, d_transcoder)
            active_encoders: Encoder vectors for active features only
        """
        sparse_layers = []
        feature_ids_by_layer = [] if return_encoder_vectors else None
        encode_start = time.perf_counter()
        self.emit_trace_event(
            "phase0.encode_sparse.start",
            n_layers=self.n_layers,
            n_pos=x.shape[1],
            d_model=self.d_model,
            lazy_encoder=self.lazy_encoder,
        )

        for layer_id in range(self.n_layers):
            layer_start = time.perf_counter()
            W_enc_layer = self._get_encoder_weights(layer_id)
            layer_features = (
                torch.einsum("bd,fd->bf", x[layer_id], W_enc_layer) + self.b_enc[layer_id]
            )

            layer_features = self.apply_activation_function(layer_id, layer_features)

            layer_features[zero_positions] = 0

            sparse_layer = layer_features.to_sparse().coalesce()
            sparse_layers.append(sparse_layer)

            _, feat_idx = sparse_layer.indices()
            if feature_ids_by_layer is not None:
                feature_ids_by_layer.append(feat_idx)
            self._add_diagnostic_layer_value(
                "encode_sparse_by_layer", layer_id, time.perf_counter() - layer_start
            )
            self._add_diagnostic_layer_value(
                "encode_sparse_active_features_by_layer", layer_id, float(len(feat_idx))
            )
            self.emit_trace_event(
                "phase0.encode_sparse.layer_done",
                layer=layer_id,
                active_features=len(feat_idx),
                elapsed_s=f"{time.perf_counter() - layer_start:.2f}",
            )

        sparse_features = torch.stack(sparse_layers).coalesce()
        active_encoders = (
            self._gather_encoder_vectors_by_layer(feature_ids_by_layer, device=x.device)
            if feature_ids_by_layer is not None
            else None
        )
        encode_elapsed = time.perf_counter() - encode_start
        self._add_diagnostic_value("encode_sparse_seconds", encode_elapsed)
        self.emit_trace_event(
            "phase0.encode_sparse.done",
            total_active_features=sparse_features._nnz(),
            elapsed_s=f"{encode_elapsed:.2f}",
        )
        return sparse_features, active_encoders

    def _gather_encoder_vectors_by_layer(
        self,
        feature_ids_by_layer: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        total_active_features = sum(int(feat_ids.numel()) for feat_ids in feature_ids_by_layer)
        if total_active_features == 0:
            return torch.empty((0, self.d_model), device=device, dtype=self.dtype)

        active_encoders = torch.empty(
            (total_active_features, self.d_model),
            device=device,
            dtype=self.dtype,
        )
        offset = 0
        for layer_id, feat_ids in enumerate(feature_ids_by_layer):
            count = int(feat_ids.numel())
            if count == 0:
                continue
            active_encoders[offset : offset + count] = self._get_encoder_weights(layer_id)[feat_ids]
            offset += count
        return active_encoders

    def gather_encoder_vectors(self, features: torch.Tensor) -> torch.Tensor:
        if not features.is_sparse:
            features = features.to_sparse()
        features = features.coalesce()

        source_layers, _, feat_ids = features.indices()
        layer_counts = torch.bincount(source_layers, minlength=self.n_layers).tolist()
        feature_ids_by_layer: list[torch.Tensor] = []
        offset = 0
        for count in layer_counts:
            feature_ids_by_layer.append(feat_ids[offset : offset + count])
            offset += count

        return self._gather_encoder_vectors_by_layer(feature_ids_by_layer, device=features.device)

    def _get_encoder_rows_for_layer(
        self,
        layer_id: int,
        feat_ids: torch.Tensor,
    ) -> torch.Tensor:
        feat_ids = feat_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        if feat_ids.numel() == 0:
            return torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)

        if not self.lazy_encoder:
            assert self.W_enc is not None, "Encoder weights are not set"
            return self.W_enc[layer_id][feat_ids.to(device=self.W_enc.device)].to(dtype=self.dtype)

        start = time.perf_counter()
        if self.layer_paths is not None:
            path = self.layer_paths[layer_id]
            with safe_open(path, framework="pt", device="cpu") as f:
                encoder_rows = f.get_slice("w_enc")[:, feat_ids].transpose(0, 1).contiguous()
                result = self._move_lazy_slice_to_device(encoder_rows)
        else:
            assert self.clt_path is not None, "CLT path is not set"
            path = os.path.join(self.clt_path, f"W_enc_{layer_id}.safetensors")
            with safe_open(path, framework="pt", device="cpu") as f:
                result = self._move_lazy_slice_to_device(f.get_slice(f"W_enc_{layer_id}")[feat_ids])

        elapsed = time.perf_counter() - start
        self._add_diagnostic_value("encoder_load_count", 1)
        self._add_diagnostic_value("encoder_load_seconds", elapsed)
        self._add_diagnostic_layer_value("encoder_load_by_layer", layer_id, elapsed)
        return result

    def materialize_encoder_rows(
        self,
        source_layers: torch.Tensor,
        feature_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize encoder rows for specific (layer, feature) pairs.

        Args:
            source_layers: 1-D tensor of source layer indices in requested row order.
            feature_ids: 1-D tensor of feature indices aligned with ``source_layers``.

        Returns:
            Tensor of shape ``(len(source_layers), d_model)`` with rows in exactly
            the input order.
        """

        source_layers = source_layers.reshape(-1).to(device="cpu", dtype=torch.long)
        feature_ids = feature_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        if source_layers.numel() != feature_ids.numel():
            raise ValueError("source_layers and feature_ids must have matching lengths")

        if source_layers.numel() == 0:
            return torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)

        active_encoders = torch.empty(
            (source_layers.numel(), self.d_model),
            device=self.device,
            dtype=self.dtype,
        )
        for layer_id in torch.unique(source_layers, sorted=True).tolist():
            layer_mask = source_layers == layer_id
            layer_rows = torch.nonzero(layer_mask, as_tuple=False).squeeze(-1)
            if layer_rows.numel() == 0:
                continue

            layer_feat_ids = feature_ids[layer_rows]
            active_encoders[layer_rows.to(device=self.device)] = self._get_encoder_rows_for_layer(
                layer_id,
                layer_feat_ids,
            )

        return active_encoders

    def _get_decoder_vectors(self, layer_id, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]

        if not self.lazy_decoder:
            assert self.W_dec is not None, "Decoder weights are not set"
            return self.W_dec[layer_id][to_read].to(dtype=self.dtype)

        start = time.perf_counter()

        if self.layer_paths is not None:
            path = self.layer_paths[layer_id]
            if isinstance(to_read, torch.Tensor):
                to_read = to_read.cpu()
            with safe_open(path, framework="pt", device="cpu") as f:
                decoder_block = f.get_slice("w_dec")[to_read]
                result = self._move_lazy_slice_to_device(decoder_block[:, layer_id:, :])
            elapsed = time.perf_counter() - start
            self._add_diagnostic_value("decoder_load_count", 1)
            self._add_diagnostic_value("decoder_load_seconds", elapsed)
            self._add_diagnostic_layer_value("decoder_load_by_layer", layer_id, elapsed)
            load_count = int(cast(float, self._diagnostic_stats["decoder_load_count"]))
            if load_count <= 3 or load_count % self._trace_decoder_load_interval == 0:
                self.emit_trace_event(
                    "decoder.load",
                    source_layer=layer_id,
                    load_count=load_count,
                    elapsed_s=f"{elapsed:.2f}",
                    lazy_decoder=self.lazy_decoder,
                )
            return result

        assert self.clt_path is not None, "CLT path is not set"
        path = os.path.join(self.clt_path, f"W_dec_{layer_id}.safetensors")
        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        with safe_open(path, framework="pt", device="cpu") as f:
            result = self._move_lazy_slice_to_device(f.get_slice(f"W_dec_{layer_id}")[to_read])
        elapsed = time.perf_counter() - start
        self._add_diagnostic_value("decoder_load_count", 1)
        self._add_diagnostic_value("decoder_load_seconds", elapsed)
        self._add_diagnostic_layer_value("decoder_load_by_layer", layer_id, elapsed)
        load_count = int(cast(float, self._diagnostic_stats["decoder_load_count"]))
        if load_count <= 3 or load_count % self._trace_decoder_load_interval == 0:
            self.emit_trace_event(
                "decoder.load",
                source_layer=layer_id,
                load_count=load_count,
                elapsed_s=f"{elapsed:.2f}",
                lazy_decoder=self.lazy_decoder,
            )
        return result

    def _get_decoder_chunk_uncached(self, layer_id: int, chunk_id: int) -> torch.Tensor:
        start_idx = chunk_id * self.decoder_chunk_size
        stop_idx = min(start_idx + self.decoder_chunk_size, self.d_transcoder)
        if start_idx >= self.d_transcoder or stop_idx <= start_idx:
            raise IndexError(f"Decoder chunk {chunk_id} out of range for source layer {layer_id}")

        if not self.lazy_decoder:
            assert self.W_dec is not None, "Decoder weights are not set"
            return self.W_dec[layer_id][start_idx:stop_idx].to(dtype=self.dtype)

        start = time.perf_counter()
        if self.layer_paths is not None:
            path = self.layer_paths[layer_id]
            with safe_open(path, framework="pt", device="cpu") as f:
                decoder_block = f.get_slice("w_dec")[start_idx:stop_idx]
                result = self._move_lazy_slice_to_device(decoder_block[:, layer_id:, :])
        else:
            assert self.clt_path is not None, "CLT path is not set"
            path = os.path.join(self.clt_path, f"W_dec_{layer_id}.safetensors")
            with safe_open(path, framework="pt", device="cpu") as f:
                result = self._move_lazy_slice_to_device(
                    f.get_slice(f"W_dec_{layer_id}")[start_idx:stop_idx]
                )

        elapsed = time.perf_counter() - start
        self._add_diagnostic_value("decoder_load_count", 1)
        self._add_diagnostic_value("decoder_load_seconds", elapsed)
        self._add_diagnostic_layer_value("decoder_load_by_layer", layer_id, elapsed)
        load_count = int(cast(float, self._diagnostic_stats["decoder_load_count"]))
        if load_count <= 3 or load_count % self._trace_decoder_load_interval == 0:
            self.emit_trace_event(
                "decoder.load",
                source_layer=layer_id,
                chunk_id=chunk_id,
                load_count=load_count,
                elapsed_s=f"{elapsed:.2f}",
                lazy_decoder=self.lazy_decoder,
            )
        return result

    def get_decoder_chunk(
        self,
        layer_id: int,
        chunk_id: int,
        decoder_cache: DecoderChunkCache | None = None,
    ) -> torch.Tensor:
        cache_key = (layer_id, chunk_id)
        if decoder_cache is not None:
            cached = decoder_cache.get(cache_key)
            if cached is not None:
                self._record_decoder_cache_hit(decoder_cache, layer_id=layer_id, chunk_id=chunk_id)
                return cached
            self._record_decoder_cache_miss(decoder_cache, layer_id=layer_id, chunk_id=chunk_id)

        result = self._get_decoder_chunk_uncached(layer_id, chunk_id)
        if decoder_cache is None:
            return result

        chunk_bytes = DecoderChunkCache._tensor_nbytes(result)
        if chunk_bytes > decoder_cache.max_bytes:
            self._record_decoder_cache_skip(
                decoder_cache,
                layer_id=layer_id,
                chunk_id=chunk_id,
                chunk_bytes=chunk_bytes,
            )
            return result

        evicted = decoder_cache.put(cache_key, result)
        self._record_decoder_cache_put(
            decoder_cache,
            layer_id=layer_id,
            chunk_id=chunk_id,
            evicted=evicted,
        )
        return result

    def get_decoder_block(self, layer_id, feat_ids=None):
        return self._get_decoder_vectors(layer_id, feat_ids)

    def get_decoder_vectors_for_output_layer(self, layer_id, output_layer, feat_ids=None):
        if output_layer < layer_id:
            raise ValueError(
                f"Output layer {output_layer} must be >= source layer {layer_id} for CLT decoders"
            )

        relative_output_layer = output_layer - layer_id
        decoder_block = self.get_decoder_block(layer_id, feat_ids)
        return decoder_block[:, relative_output_layer].to(dtype=self.dtype)

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()
        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        n_layers = features.shape[0]
        device = features.device

        pos_ids = []
        layer_ids = []
        feat_ids = []

        decoder_vectors = []
        encoder_mapping = []
        st = 0

        for layer_id in range(n_layers):
            current_layer = layer_idx == layer_id
            if not current_layer.any():
                continue

            current_layer_features = feat_idx[current_layer]
            unique_feats, inv = current_layer_features.unique(return_inverse=True)

            unique_decoders = self._get_decoder_vectors(layer_id, unique_feats.cpu())
            scaled_decoders = unique_decoders[inv] * activations[current_layer, None, None]
            decoder_vectors.append(scaled_decoders.reshape(-1, self.d_model))

            n_output_layers = self.n_layers - layer_id
            pos_ids.append(pos_idx[current_layer].repeat_interleave(n_output_layers))
            feat_ids.append(current_layer_features.repeat_interleave(n_output_layers))
            layer_ids.append(
                torch.arange(layer_id, self.n_layers, device=device).repeat(
                    len(current_layer_features)
                )
            )

            source_ids = torch.arange(len(current_layer_features), device=device) + st
            st += len(current_layer_features)
            encoder_mapping.append(torch.repeat_interleave(source_ids, n_output_layers))

        pos_ids = torch.cat(pos_ids, dim=0)
        layer_ids = torch.cat(layer_ids, dim=0)
        feat_ids = torch.cat(feat_ids, dim=0)
        decoder_vectors = torch.cat(decoder_vectors, dim=0)
        encoder_mapping = torch.cat(encoder_mapping, dim=0)

        return pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping

    def compute_reconstruction(
        self, pos_ids, layer_ids, decoder_vectors, input_acts: torch.Tensor | None = None
    ):
        n_pos = pos_ids.max() + 1
        flat_idx = layer_ids * n_pos + pos_ids
        accumulation_dtype = (
            torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
        )
        recon = torch.zeros(
            n_pos * self.n_layers,
            self.d_model,
            device=decoder_vectors.device,
            dtype=accumulation_dtype,
        ).index_add_(0, flat_idx, decoder_vectors.to(dtype=accumulation_dtype))
        recon = recon.reshape(self.n_layers, n_pos, self.d_model) + self.b_dec[:, None].to(
            dtype=accumulation_dtype
        )
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            recon = recon + (input_acts @ self.W_skip).to(dtype=accumulation_dtype)
        return recon.to(dtype=self.dtype)

    def compute_reconstruction_chunked(
        self,
        features: torch.Tensor,
        input_acts: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ):
        if not features.is_sparse:
            features = features.to_sparse()

        chunk_size = chunk_size or self.decoder_chunk_size
        source_layers, positions, feat_ids = features.indices()
        activations = features.values()
        _, n_pos, _ = features.shape
        accumulation_dtype = (
            torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
        )
        recon = torch.zeros(
            self.n_layers * n_pos,
            self.d_model,
            device=features.device,
            dtype=accumulation_dtype,
        )

        reconstruction_start = time.perf_counter()
        self.emit_trace_event(
            "phase0.reconstruction.start",
            n_layers=self.n_layers,
            nnz=features._nnz(),
            chunk_size=chunk_size,
        )

        for layer_id in range(self.n_layers):
            layer_start = time.perf_counter()
            layer_chunk_count = 0
            layer_mask = source_layers == layer_id
            if not layer_mask.any():
                continue

            layer_indices = torch.where(layer_mask)[0]
            self.emit_trace_event(
                "phase0.reconstruction.layer_start",
                layer=layer_id,
                active_features=len(layer_indices),
            )
            output_layers = torch.arange(layer_id, self.n_layers, device=positions.device)
            n_output_layers = len(output_layers)
            for start in range(0, len(layer_indices), chunk_size):
                chunk_indices = layer_indices[start : start + chunk_size]
                layer_chunk_count += 1
                chunk_feat_ids = feat_ids[chunk_indices]
                unique_feats, inv = chunk_feat_ids.unique(return_inverse=True)
                decoder_block = self.get_decoder_block(layer_id, unique_feats.cpu())
                scaled_decoders = decoder_block[inv] * activations[chunk_indices, None, None].to(
                    device=decoder_block.device,
                    dtype=decoder_block.dtype,
                    non_blocking=decoder_block.device.type == "cuda",
                )

                chunk_positions = positions[chunk_indices]
                flat_idx = output_layers.repeat(
                    len(chunk_indices)
                ) * n_pos + chunk_positions.repeat_interleave(n_output_layers)
                recon.index_add_(
                    0,
                    flat_idx,
                    scaled_decoders.reshape(-1, self.d_model).to(dtype=accumulation_dtype),
                )

                if layer_chunk_count <= 2 or layer_chunk_count % self._trace_chunk_interval == 0:
                    self.emit_trace_event(
                        "phase0.reconstruction.chunk",
                        layer=layer_id,
                        chunk=layer_chunk_count,
                        processed=min(start + len(chunk_indices), len(layer_indices)),
                        total=len(layer_indices),
                    )

            layer_elapsed = time.perf_counter() - layer_start
            self._add_diagnostic_layer_value("reconstruction_by_layer", layer_id, layer_elapsed)
            self._add_diagnostic_layer_value(
                "reconstruction_chunks_by_layer", layer_id, float(layer_chunk_count)
            )
            self._add_diagnostic_value("reconstruction_chunk_count", layer_chunk_count)
            self.emit_trace_event(
                "phase0.reconstruction.layer_done",
                layer=layer_id,
                chunks=layer_chunk_count,
                elapsed_s=f"{layer_elapsed:.2f}",
            )

        recon = recon.reshape(self.n_layers, n_pos, self.d_model) + self.b_dec[:, None].to(
            dtype=accumulation_dtype
        )
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            recon = recon + (input_acts @ self.W_skip).to(dtype=accumulation_dtype)
        reconstruction_elapsed = time.perf_counter() - reconstruction_start
        self._add_diagnostic_value("reconstruction_seconds", reconstruction_elapsed)
        self.emit_trace_event(
            "phase0.reconstruction.done",
            total_chunks=int(cast(float, self._diagnostic_stats["reconstruction_chunk_count"])),
            elapsed_s=f"{reconstruction_elapsed:.2f}",
        )
        return recon.to(dtype=self.dtype)

    def decode(self, features, input_acts: torch.Tensor | None = None):
        if self.exact_chunked_decoder:
            return self.compute_reconstruction_chunked(features, input_acts)

        pos_ids, layer_ids, feat_ids, decoder_vectors, _ = self.select_decoder_vectors(features)
        return self.compute_reconstruction(pos_ids, layer_ids, decoder_vectors, input_acts)

    def compute_skip(self, layer_id: int, inputs):
        if self.W_skip is not None:
            return inputs @ self.W_skip[layer_id]
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, x):
        features = self.encode(x).to_sparse()
        decoded = self.decode(features)

        if self.W_skip is not None:
            skip = x @ self.W_skip
            decoded = decoded + skip

        return decoded

    def compute_attribution_components(
        self,
        inputs,
        zero_positions: slice = slice(0, 1),
        sparsification: SparsificationConfig | None = None,
        *,
        materialize_encoder_vecs: bool = True,
    ):
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            inputs: Input tensor to encode

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse activation matrix
                - reconstruction: Reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
        """
        self.emit_trace_event(
            "phase0.components.start",
            input_shape=tuple(inputs.shape),
            exact_chunked_decoder=self.exact_chunked_decoder,
            materialize_encoder_vecs=materialize_encoder_vecs,
        )
        component_start = time.perf_counter()

        if not materialize_encoder_vecs and not self.exact_chunked_decoder:
            raise ValueError(
                "materialize_encoder_vecs=False is only supported with exact_chunked_decoder"
            )

        features, encoder_vectors = self.encode_sparse(
            inputs,
            zero_positions=zero_positions,
            return_encoder_vectors=materialize_encoder_vecs and sparsification is None,
        )
        sparsification_stats = None

        if sparsification is not None:
            selected_indices, sparsification_stats = select_candidate_feature_indices(
                features, sparsification
            )
            features = filter_sparse_activations(features, selected_indices)
            if materialize_encoder_vecs:
                encoder_vectors = self.gather_encoder_vectors(features)
        elif encoder_vectors is None and materialize_encoder_vecs:
            encoder_vectors = self.gather_encoder_vectors(features)

        if not materialize_encoder_vecs:
            encoder_vectors = torch.empty((0, self.d_model), dtype=self.dtype, device=inputs.device)

        if self.exact_chunked_decoder:
            reconstruction = self.compute_reconstruction_chunked(features, inputs)
            empty_long = torch.empty(0, dtype=torch.long, device=inputs.device)
            decoder_vectors = torch.empty((0, self.d_model), dtype=self.dtype, device=inputs.device)
            encoder_to_decoder_map = empty_long
            decoder_locations = torch.empty((2, 0), dtype=torch.long, device=inputs.device)
            chunked_decoder_state = {
                "source_layers": features.indices()[0],
                "positions": features.indices()[1],
                "feature_ids": features.indices()[2],
                "activation_values": features.values(),
            }
        else:
            pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_to_decoder_map = (
                self.select_decoder_vectors(features)
            )
            reconstruction = self.compute_reconstruction(
                pos_ids, layer_ids, decoder_vectors, inputs
            )
            decoder_locations = torch.stack((layer_ids, pos_ids))
            chunked_decoder_state = None

        attribution_data = {
            "activation_matrix": features,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vectors,
            "decoder_vecs": decoder_vectors,
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": decoder_locations,
        }

        if chunked_decoder_state is not None:
            attribution_data["chunked_decoder_state"] = chunked_decoder_state
        if sparsification_stats is not None:
            attribution_data["sparsification_stats"] = sparsification_stats

        self.emit_trace_event(
            "phase0.components.done",
            active_features=features._nnz(),
            elapsed_s=f"{time.perf_counter() - component_start:.2f}",
        )

        return attribution_data

    def to_safetensors(self, save_path: str):
        """Save CLT to safetensors format compatible with lazy loading.

        Saves the CLT state dict split across multiple safetensors files:
        - W_enc_{i}.safetensors: Contains W_enc_{i}, b_enc_{i}, b_dec_{i}, and optionally threshold_{i}
        - W_dec_{i}.safetensors: Contains W_dec_{i}

        Args:
            save_path: Directory path where the safetensors files will be saved
        """
        os.makedirs(save_path, exist_ok=True)

        has_threshold = isinstance(self.activation_function, JumpReLU)
        thresholds = None
        if has_threshold:
            thresholds = cast(JumpReLU, self.activation_function).threshold

        for i in range(self.n_layers):
            # Save encoder weights and biases
            enc_dict = {
                f"W_enc_{i}": self._get_encoder_weights(i).cpu(),
                f"b_enc_{i}": self.b_enc[i].cpu(),
                f"b_dec_{i}": self.b_dec[i].cpu(),
            }

            if has_threshold:
                assert thresholds is not None
                enc_dict[f"threshold_{i}"] = thresholds[i].squeeze(0).cpu()

            enc_path = os.path.join(save_path, f"W_enc_{i}.safetensors")
            save_file(enc_dict, enc_path)

            # Save decoder weights
            if self.W_dec is not None:
                dec_dict = {f"W_dec_{i}": self.W_dec[i].cpu()}
            else:
                dec_dict = {f"W_dec_{i}": self._get_decoder_vectors(i).cpu()}

            dec_path = os.path.join(save_path, f"W_dec_{i}.safetensors")
            save_file(dec_dict, dec_path)


def load_clt(
    clt_path: str,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
    exact_chunked_decoder: bool = False,
    decoder_chunk_size: int = DEFAULT_EXACT_DECODER_CHUNK_SIZE,
    cross_batch_decoder_cache_bytes: int | None = None,
) -> CrossLayerTranscoder:
    """Load a cross-layer transcoder from safetensors files.

    Args:
        clt_path: Path to directory containing W_enc_*.safetensors and W_dec_*.safetensors files
        dtype: Data type for loaded tensors
        lazy_decoder: Whether to load decoder weights on-demand
        lazy_encoder: Whether to load encoder weights on-demand
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        scan: Optional identifier for feature visualization
        device: Device to load tensors to (defaults to auto-detected)

    Returns:
        CrossLayerTranscoder: Loaded transcoder instance
    """
    if device is None:
        device = get_default_device()

    state_dict = _load_state_dict(clt_path, lazy_decoder, lazy_encoder, device, dtype)

    # Infer dimensions from loaded tensors
    n_layers = state_dict["b_dec"].shape[0]
    d_transcoder = state_dict["b_enc"].shape[1]
    d_model = state_dict["b_dec"].shape[1]

    act_fn = "jump_relu" if "activation_function.threshold" in state_dict else "relu"

    # Create instance and load state dict
    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function=act_fn,
            skip_connection=state_dict.get("W_skip") is not None,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=torch.device("meta"),
            dtype=dtype,
            clt_path=clt_path,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
            cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
        )

    instance.load_state_dict(state_dict, assign=True)

    return instance


def load_gemma_scope_2_clt(
    paths: dict[int, str],
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
    decoder_chunk_size: int = DEFAULT_EXACT_DECODER_CHUNK_SIZE,
    cross_batch_decoder_cache_bytes: int | None = None,
) -> CrossLayerTranscoder:
    """Load a CrossLayerTranscoder from a GemmaScope2 JumpReLUMultiLayerSAE checkpoint.

    Args:
        path: Path to the checkpoint file
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        scan: Optional identifier for feature visualization
        device: Device to load to
        dtype: Data type to use
        lazy_decoder: Whether to lazily load decoder weights from per-layer safetensors files
        lazy_encoder: Whether to lazily load encoder weights from per-layer safetensors files

    Returns:
        CrossLayerTranscoder: The loaded transcoder
    """
    if device is None:
        device = get_default_device()

    ordered_layers = sorted(paths)
    if ordered_layers != list(range(len(ordered_layers))):
        raise ValueError("GemmaScope-2 CLT paths must be indexed contiguously from 0")

    normalized_path_list: list[str] = []
    for layer_idx in ordered_layers:
        path = paths[layer_idx]
        if not normalized_path_list or normalized_path_list[-1] != path:
            normalized_path_list.append(path)

    paths = {layer_idx: path for layer_idx, path in enumerate(normalized_path_list)}

    with safe_open(paths[0], framework="pt", device="cpu") as f:
        d_model, d_transcoder = f.get_slice("w_enc").get_shape()
        has_skip = "affine_skip_connection" in f.keys()

    n_layers = len(paths)

    state_dict = {
        "b_enc": torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype),
        "b_dec": torch.zeros(n_layers, d_model, device=device, dtype=dtype),
        "activation_function.threshold": torch.zeros(
            n_layers, 1, d_transcoder, device=device, dtype=dtype
        ),
    }

    if not lazy_encoder:
        state_dict["W_enc"] = torch.zeros(
            n_layers, d_transcoder, d_model, device=device, dtype=dtype
        )

    if has_skip:
        state_dict["W_skip"] = torch.zeros(n_layers, d_model, d_model, device=device, dtype=dtype)

    for i in range(n_layers):
        with safe_open(paths[i], framework="pt", device=str(device)) as f:
            state_dict["b_enc"][i] = f.get_tensor("b_enc").to(dtype=dtype)
            state_dict["b_dec"][i] = f.get_tensor("b_dec").to(dtype=dtype)
            state_dict["activation_function.threshold"][i] = (
                f.get_tensor("threshold").to(dtype=dtype).unsqueeze(0)
            )

            if not lazy_encoder:
                state_dict["W_enc"][i] = f.get_tensor("w_enc").transpose(-1, -2).to(dtype=dtype)

            if not lazy_decoder:
                state_dict[f"W_dec.{i}"] = f.get_tensor("w_dec")[:, i:, :].to(dtype=dtype)

            if has_skip:
                state_dict["W_skip"][i] = f.get_tensor("affine_skip_connection").to(dtype=dtype)

    # Create instance
    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function="jump_relu",
            skip_connection=("W_skip" in state_dict),
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=torch.device("meta"),
            dtype=dtype,
            layer_paths=paths if (lazy_encoder or lazy_decoder) else None,
            weight_format="gemmascope2",
            exact_chunked_decoder=True,
            decoder_chunk_size=decoder_chunk_size,
            cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
        )

    instance.load_state_dict(state_dict, assign=True)

    return instance


def _load_state_dict(
    clt_path, lazy_decoder=True, lazy_encoder=False, device=None, dtype=torch.bfloat16
):
    if device is None:
        device = get_default_device()

    enc_files = glob.glob(os.path.join(clt_path, "W_enc_*.safetensors"))
    n_layers = len(enc_files)

    # Get dimensions from first file
    dec_file = "W_enc_0.safetensors"
    with safe_open(os.path.join(clt_path, dec_file), framework="pt", device=str(device)) as f:
        d_transcoder, d_model = f.get_slice("W_enc_0").get_shape()
        has_threshold = "threshold_0" in f.keys()

    # Preallocate tensors
    b_dec = torch.zeros(n_layers, d_model, device=device, dtype=dtype)
    b_enc = torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)

    state_dict = {"b_dec": b_dec, "b_enc": b_enc}

    if has_threshold:
        state_dict["activation_function.threshold"] = torch.zeros(
            n_layers, 1, d_transcoder, device=device, dtype=dtype
        )

    # Only create W_enc if not lazy
    if not lazy_encoder:
        W_enc = torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
        state_dict["W_enc"] = W_enc

    # Load all layers
    for i in range(n_layers):
        enc_file = f"W_enc_{i}.safetensors"
        with safe_open(os.path.join(clt_path, enc_file), framework="pt", device=str(device)) as f:
            b_dec[i] = f.get_tensor(f"b_dec_{i}").to(dtype)
            b_enc[i] = f.get_tensor(f"b_enc_{i}").to(dtype)

            # Only load W_enc if not lazy
            if not lazy_encoder:
                W_enc[i] = f.get_tensor(f"W_enc_{i}").to(dtype)

            if has_threshold:
                threshold = f.get_tensor(f"threshold_{i}").to(dtype)
                state_dict["activation_function.threshold"][i] = threshold.unsqueeze(0)

        # Load W_dec for this layer if not lazy
        if not lazy_decoder:
            dec_file = os.path.join(clt_path, f"W_dec_{i}.safetensors")
            with safe_open(dec_file, framework="pt", device=str(device)) as f:
                state_dict[f"W_dec.{i}"] = f.get_tensor(f"W_dec_{i}").to(dtype)

    return state_dict
