"""
Attribution context for managing hooks during attribution computation.
"""

import time
import weakref
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from einops import einsum


if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,  # type: ignore
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        encoder_vecs: torch.Tensor,
        encoder_to_decoder_map: torch.Tensor,
        decoder_locations: torch.Tensor,
        logits: torch.Tensor,
        decoder_provider=None,
        chunked_decoder_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: list[torch.Tensor] = []
        self._feature_output_activations: list[torch.Tensor] = []
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        self.logits = logits
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs

        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations
        self.decoder_provider = decoder_provider
        self.chunked_decoder_state = chunked_decoder_state
        self.setup_diagnostic_stats: dict[str, object] | None = None
        self.diagnostic_mode = False
        self._diagnostic_stats: dict[str, object] = {
            "compute_batch_calls": 0.0,
            "compute_batch_seconds": 0.0,
            "compute_batch_seconds_by_phase": {},
            "feature_attr_seconds_by_layer": {},
            "error_attr_seconds_by_layer": {},
            "token_attr_seconds": 0.0,
            "chunked_attr_chunks_by_output_layer": {},
            "chunked_attr_seconds_by_output_layer": {},
            "chunked_attr_seconds_by_source_layer": {},
        }

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def set_diagnostic_mode(self, enabled: bool) -> None:
        self.diagnostic_mode = enabled

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for key, value in self._diagnostic_stats.items():
            snapshot[key] = dict(value) if isinstance(value, dict) else value
        return snapshot

    def _add_stat(self, key: str, value: float) -> None:
        current = cast(float, self._diagnostic_stats.get(key, 0.0))
        self._diagnostic_stats[key] = current + value

    def _add_layer_stat(self, key: str, layer: int, value: float) -> None:
        bucket = cast(dict[int, float], self._diagnostic_stats.setdefault(key, {}))
        bucket[layer] = bucket.get(layer, 0.0) + value

    def apply_diagnostic_feature_cap(self, max_features: int) -> tuple[int, int]:
        total_active_feats = self.activation_matrix._nnz()
        if max_features >= total_active_feats:
            return total_active_feats, total_active_feats

        selected = (
            torch.topk(self.activation_matrix.values().abs(), k=max_features, sorted=False)
            .indices.sort()
            .values
        )

        self.activation_matrix = torch.sparse_coo_tensor(
            self.activation_matrix.indices()[:, selected],
            self.activation_matrix.values()[selected],
            size=self.activation_matrix.shape,
            device=self.activation_matrix.device,
            dtype=self.activation_matrix.dtype,
        ).coalesce()
        self.encoder_vecs = self.encoder_vecs[selected]

        if self.chunked_decoder_state is not None:
            for key in self.chunked_decoder_state:
                self.chunked_decoder_state[key] = self.chunked_decoder_state[key][selected]
        elif self.encoder_to_decoder_map.numel():
            old_to_new = torch.full(
                (total_active_feats,),
                -1,
                device=self.encoder_to_decoder_map.device,
                dtype=torch.long,
            )
            selected_on_map_device = selected.to(device=old_to_new.device)
            old_to_new[selected_on_map_device] = torch.arange(
                max_features, device=old_to_new.device
            )
            keep_decoder = old_to_new[self.encoder_to_decoder_map.long()] >= 0
            self.decoder_vecs = self.decoder_vecs[keep_decoder]
            self.decoder_locations = self.decoder_locations[:, keep_decoder]
            self.encoder_to_decoder_map = old_to_new[
                self.encoder_to_decoder_map[keep_decoder].long()
            ]

        n_layers, n_pos, _ = self.activation_matrix.shape
        self._row_size = self.activation_matrix._nnz() + (n_layers + 1) * n_pos
        return total_active_feats, self.activation_matrix._nnz()

    def _compute_chunked_feature_attributions(self, layer: int, grads: torch.Tensor):
        assert self.chunked_decoder_state is not None
        assert self.decoder_provider is not None
        assert self._batch_buffer is not None

        source_layers = self.chunked_decoder_state["source_layers"]
        positions = self.chunked_decoder_state["positions"]
        feature_ids = self.chunked_decoder_state["feature_ids"]
        activation_values = self.chunked_decoder_state["activation_values"]
        chunk_size = getattr(self.decoder_provider, "decoder_chunk_size", 256)
        output_layer_start = time.perf_counter()
        chunk_count = 0

        for source_layer in range(layer + 1):
            source_layer_start = time.perf_counter()
            layer_mask = source_layers == source_layer
            if not layer_mask.any():
                continue

            layer_indices = torch.where(layer_mask)[0]
            for start in range(0, len(layer_indices), chunk_size):
                chunk_indices = layer_indices[start : start + chunk_size]
                chunk_count += 1
                chunk_feature_ids = feature_ids[chunk_indices]
                unique_feats, inv = chunk_feature_ids.unique(return_inverse=True)
                decoder_vecs = self.decoder_provider.get_decoder_vectors_for_output_layer(
                    source_layer, layer, unique_feats.cpu()
                )
                scaled_decoders = decoder_vecs[inv].to(device=grads.device) * activation_values[
                    chunk_indices, None
                ].to(device=grads.device, dtype=decoder_vecs.dtype)

                selected_grads = grads.to(scaled_decoders.dtype)[:, positions[chunk_indices]]
                self._batch_buffer[chunk_indices] += einsum(
                    selected_grads,
                    scaled_decoders,
                    "batch position d_model, position d_model -> position batch",
                )

            if self.diagnostic_mode:
                self._add_layer_stat(
                    "chunked_attr_seconds_by_source_layer",
                    source_layer,
                    time.perf_counter() - source_layer_start,
                )

        if self.diagnostic_mode:
            self._add_layer_stat("chunked_attr_chunks_by_output_layer", layer, float(chunk_count))
            self._add_layer_stat(
                "chunked_attr_seconds_by_output_layer",
                layer,
                time.perf_counter() - output_layer_start,
            )

    def cache_residual(self, model: "NNSightReplacementModel", tracer, barrier=None):
        """Cache the model's residual for use in the attribution context."""
        with tracer.invoke():
            for feature_input_loc in model.feature_input_locs:
                self._resid_activations.append(feature_input_loc.output)  # type: ignore

            self._resid_activations.append(model.pre_logit_location.output.last_hidden_state)  # type: ignore

        with tracer.invoke():
            self._feature_output_activations.append(model.embed_location.output)  # type: ignore
            for feature_output_loc_ in model.feature_output_locs:
                if barrier:
                    barrier()

                self._feature_output_activations.append(feature_output_loc_.output)  # type: ignore

    def compute_score(
        self,
        grads: torch.Tensor,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> None:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and accumulates the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)
        proxy._batch_buffer[write_index] += einsum(
            grads.to(output_vecs.dtype)[read_index],
            output_vecs,
            "batch position d_model, position d_model -> position batch",
        )

    def compute_feature_attributions(self, layer, grads):
        if self.chunked_decoder_state is not None:
            self._compute_chunked_feature_attributions(layer, grads)
            return

        nnz_layers, nnz_positions = self.decoder_locations

        # Feature nodes - use decoder_locations to find decoders that write to this layer
        layer_mask = nnz_layers == layer
        if layer_mask.any():
            self.compute_score(
                grads,
                self.decoder_vecs[layer_mask],
                write_index=self.encoder_to_decoder_map[layer_mask],  # type: ignore
                read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore
            )

    def compute_error_attributions(self, layer, grads):
        _, n_pos, _ = self.activation_matrix.shape

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        self.compute_score(
            grads,
            self.error_vectors[layer],
            write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
        )

    def compute_token_attributions(self, grads):
        n_layers, n_pos, _ = self.activation_matrix.shape

        # Token-embedding nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        tok_start = error_offset(n_layers)
        self.compute_score(
            grads,
            self.token_vectors,
            write_index=np.s_[tok_start : tok_start + n_pos],
        )

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
        phase_label: str = "unknown",
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]
        batch_start = time.perf_counter()
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grad_point, *, batch_indices, pos_indices, values):
            grads_out = grad_point.grad.clone()
            grads_out.index_put_((batch_indices, pos_indices), values.to(grads_out.dtype))
            grad_point.grad = grads_out

        layers_in_batch = sorted(layers.unique().tolist(), reverse=True)

        last_layer = max(layers_in_batch)
        with self._resid_activations[last_layer].backward(
            gradient=torch.zeros_like(self._resid_activations[last_layer]),
            retain_graph=retain_graph,
        ):
            for layer in reversed(range(last_layer + 1)):
                if layer != last_layer:
                    grad = self._feature_output_activations[layer + 1].grad.clone()  # type:ignore
                    feature_start = time.perf_counter()
                    self.compute_feature_attributions(layer, grad)
                    if self.diagnostic_mode:
                        self._add_layer_stat(
                            "feature_attr_seconds_by_layer",
                            layer,
                            time.perf_counter() - feature_start,
                        )
                    error_start = time.perf_counter()
                    self.compute_error_attributions(layer, grad)
                    if self.diagnostic_mode:
                        self._add_layer_stat(
                            "error_attr_seconds_by_layer", layer, time.perf_counter() - error_start
                        )

                mask = layers == layer
                if mask.any():
                    _inject(
                        grad_point=self._resid_activations[layer],
                        batch_indices=batch_idx[mask],
                        pos_indices=positions[mask],
                        values=inject_values[mask],
                    )

            token_start = time.perf_counter()
            self.compute_token_attributions(self._feature_output_activations[0].grad)
            if self.diagnostic_mode:
                self._add_stat("token_attr_seconds", time.perf_counter() - token_start)

        buf, self._batch_buffer = self._batch_buffer, None
        if self.diagnostic_mode:
            self._add_stat("compute_batch_calls", 1)
            elapsed = time.perf_counter() - batch_start
            self._add_stat("compute_batch_seconds", elapsed)
            phase_bucket = cast(
                dict[str, float],
                self._diagnostic_stats.setdefault("compute_batch_seconds_by_phase", {}),
            )
            phase_bucket[phase_label] = phase_bucket.get(phase_label, 0.0) + elapsed
        return buf.T[: len(layers)]
