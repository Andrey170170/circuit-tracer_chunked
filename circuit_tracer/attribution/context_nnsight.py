"""
Attribution context for managing hooks during attribution computation.
"""

import inspect
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from einops import einsum

from circuit_tracer.attribution.nnsight.active_decoder_contraction import (
    contract_active_decoder_rows,
)
from circuit_tracer.attribution.nnsight.active_decoder_rows import (
    ActiveDecoderRows,
    build_active_decoder_rows,
    decoder_state_signature,
    estimate_active_decoder_row_bytes,
    materialize_active_decoder_rows_from_seed,
)
from circuit_tracer.attribution.nnsight.batch_execution import (
    BatchAttributionRequest,
    execute_observed_batch,
    slice_phase3_gradient_replay_batch as _slice_phase3_gradient_replay_batch,  # noqa: F401
)
from circuit_tracer.attribution.nnsight.feature_vjp_tape import (
    FeatureVjpTapeByteEstimate,
    FeatureVjpTapeEntry,
)
from circuit_tracer.attribution.nnsight.decoder_page_prefetch import DecoderPagePrefetch
from circuit_tracer.attribution.nnsight.context_state import (
    AttributionTensorState,
    ContextNumericPolicy,
    ContextExecutionPolicy,
    DecoderRuntime,
)
from circuit_tracer.observability.events import TraceEvent, TraceObserver


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
        tensor_state: AttributionTensorState,
        execution_policy: ContextExecutionPolicy,
        decoder_runtime: DecoderRuntime,
        numeric_policy: ContextNumericPolicy,
    ) -> None:
        tensors = tensor_state
        policy = execution_policy
        n_layers, n_pos, _ = tensors.activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: list[torch.Tensor] = []
        self._feature_output_activations: list[torch.Tensor] = []
        self._batch_buffer: torch.Tensor | None = None
        self._produced_feature_range: tuple[int, int] | None = None
        self._produce_nonfeature = True
        self.n_layers: int = n_layers

        exact_chunked_mode = policy.exact_chunked_mode
        self._execution_device = tensors.token_vectors.device
        self._stage_encoder_vecs_on_cpu = policy.stage_encoder_vectors_on_cpu
        self._stage_error_vectors_on_cpu = policy.stage_error_vectors_on_cpu
        self._error_vector_prefetch_lookahead = policy.error_vector_prefetch_lookahead
        self._chunked_feature_replay_window = policy.chunked_feature_replay_window
        self._row_subchunk_size = policy.row_subchunk_size
        self._materialized_error_vector_layers: dict[int, torch.Tensor] = {}
        self._cleanup_complete = False
        self.exact_encoder_residency_requested = policy.encoder_residency_requested
        self.exact_encoder_residency_effective = policy.encoder_residency_effective
        self.exact_encoder_residency_applicable = policy.exact_chunked_mode
        self.exact_encoder_residency_fallback_reason = policy.encoder_residency_fallback_reason
        self.exact_encoder_materialized_during_phase0 = bool(
            numeric_policy.materialized_encoder_vectors_during_phase0
        )
        self.exact_encoder_pinned_requested = bool(
            policy.encoder_residency_effective == "active_pinned_cpu"
        )
        self.exact_encoder_pinned_effective = False
        self.exact_encoder_pinning_success: bool | None = None
        self.exact_encoder_pinning_failure_reason: str | None = None
        self.exact_encoder_staging_destination = "none"

        self.logits = tensors.logits
        self.full_logits = tensors.full_logits
        self.logit_retention = (
            "full"
            if tensors.full_logits is not None
            or (tensors.logits.ndim >= 2 and tensors.logits.shape[1] != 1)
            else "last_token"
        )
        logit_source = tensors.full_logits if tensors.full_logits is not None else tensors.logits
        self.logit_source_shape = tuple(logit_source.shape)
        self.activation_matrix = tensors.activation_matrix
        if self._stage_error_vectors_on_cpu:
            self.error_vectors = self._stage_tensor_on_cpu(tensors.error_vectors)
        else:
            self.error_vectors = tensors.error_vectors
        self.token_vectors = tensors.token_vectors
        self.decoder_vecs = tensors.decoder_vectors
        if self._stage_encoder_vecs_on_cpu:
            self.encoder_vecs, pinning_success, pinning_failure_reason = self._stage_encoder_tensor(
                tensors.encoder_vectors,
                pin_memory=self.exact_encoder_pinned_requested,
            )
            self.exact_encoder_pinning_success = pinning_success
            self.exact_encoder_pinning_failure_reason = pinning_failure_reason
            self.exact_encoder_pinned_effective = bool(
                self.exact_encoder_pinned_requested and self.encoder_vecs.is_pinned()
            )
        else:
            self.encoder_vecs = tensors.encoder_vectors
        self.exact_encoder_staging_destination = self._resolve_encoder_staging_destination(
            self.encoder_vecs,
            exact_chunked_mode=exact_chunked_mode,
            encoder_residency=self.exact_encoder_residency_effective,
        )

        self.encoder_to_decoder_map = tensors.encoder_to_decoder_map
        self.decoder_locations = tensors.decoder_locations
        self.decoder_provider = decoder_runtime.provider
        self.chunked_decoder_state = decoder_runtime.chunked_state
        self.decoder_chunk_cache = None
        self._chunked_layer_spans: list[tuple[int, int] | None] | None = None
        self.setup_diagnostic_stats: dict[str, object] | None = None
        self.sparsification_stats: dict[str, object] | None = None
        self.internal_precision_requested = numeric_policy.internal_precision_requested
        self.resolved_dtype_map = numeric_policy.resolved_dtype_map
        self.diagnostic_mode = False
        self._trace_logger = None
        self._trace_observer: TraceObserver | None = None
        self._trace_chunk_interval = 16
        self.capture_phase3_gradients = False
        self.phase3_gradient_captures: list[dict[str, torch.Tensor | int]] = []
        self.phase3_gradient_replay_tensor: torch.Tensor | None = None
        self.phase3_gradient_replay_status = "disabled"
        self.phase3_gradient_replay_column_offset = 0
        self._compute_batch_call_index = 0
        self.decoder_page_prefetch_depth = 0
        self._decoder_page_prefetch: DecoderPagePrefetch | None = None
        self._active_decoder_rows: ActiveDecoderRows | None = None
        self._decoder_row_seed = decoder_runtime.decoder_row_seed
        self._active_decoder_row_diagnostics: dict[str, object] = {
            "decoder_active_row_residency_requested": False,
            "decoder_active_row_residency_effective": False,
            "decoder_active_row_max_bytes": 0,
            "decoder_active_row_fallback_reason": "disabled",
            "decoder_active_row_count": 0,
            "decoder_active_row_bytes": 0,
            "decoder_active_row_owner_count": 0,
            "decoder_active_row_sealed": False,
            "decoder_active_row_seed_available": self._decoder_row_seed is not None,
            "decoder_active_row_seed_missing_keys": 0,
            "decoder_active_row_seed_source_mismatch": False,
            "decoder_active_row_seed_fallback_reason": None,
            "decoder_active_row_seed_capture_refusal_reason": (
                decoder_runtime.decoder_row_seed_refusal_reason
            ),
            "decoder_active_row_seed_phase0_estimated_bytes": (
                decoder_runtime.decoder_row_seed_estimated_bytes
            ),
        }
        if self._decoder_row_seed is not None:
            self._active_decoder_row_diagnostics.update(
                {
                    "decoder_active_row_seed_capture_seconds": (
                        self._decoder_row_seed.capture_seconds
                    ),
                    "decoder_active_row_seed_shared_traversal_bytes": (
                        self._decoder_row_seed.shared_traversal_bytes
                    ),
                    "decoder_active_row_seed_shared_decoder_load_count": (
                        self._decoder_row_seed.shared_decoder_load_count
                    ),
                    "decoder_active_row_seed_shared_decoder_load_bytes": (
                        self._decoder_row_seed.shared_decoder_load_bytes
                    ),
                    "decoder_active_row_seed_unique_row_count": (
                        self._decoder_row_seed.unique_row_count
                    ),
                    "decoder_active_row_seed_bytes": self._decoder_row_seed.seed_bytes,
                }
            )
            range_telemetry = self._decoder_row_seed.phase0_decoder_range_telemetry
            if range_telemetry is not None:
                self._active_decoder_row_diagnostics.update(
                    {
                        f"phase0_decoder_row_ranges_{key}": value
                        for key, value in range_telemetry.as_dict().items()
                    }
                )
        self._replay_model = None
        self._replay_trace_input_ids: torch.Tensor | None = None
        self._replay_trace_batch_size: int | None = None
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
            "chunked_attr_replay_seconds": 0.0,
            "chunked_attr_grad_window_peak": 0.0,
            "error_vector_layers_resident_peak": 0.0,
        }

        total_active_feats = tensors.activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later
        self._refresh_chunked_layer_spans()
        self._owns_decoder_chunk_cache = decoder_runtime.owns_cache
        self.decoder_cache_fingerprint = decoder_runtime.cache_fingerprint
        if decoder_runtime.chunk_cache is not None:
            self.decoder_chunk_cache = decoder_runtime.chunk_cache
        else:
            self.decoder_chunk_cache = self._create_decoder_cache()

    @staticmethod
    def _stage_tensor_on_cpu(tensor: torch.Tensor) -> torch.Tensor:
        staged = tensor.detach()
        if staged.device.type != "cpu":
            staged = staged.to(device="cpu", non_blocking=staged.device.type == "cuda")
        else:
            staged = staged.clone()
        return staged

    @staticmethod
    def _stage_encoder_tensor(
        tensor: torch.Tensor,
        *,
        pin_memory: bool,
    ) -> tuple[torch.Tensor, bool | None, str | None]:
        staged = AttributionContext._stage_tensor_on_cpu(tensor)
        if not pin_memory:
            return staged, None, None
        try:
            pinned = staged.pin_memory()
        except Exception as exc:  # pragma: no cover - platform dependent
            return staged, False, f"{type(exc).__name__}: {exc}"
        if not pinned.is_pinned():
            return pinned, False, "pin_memory returned a non-pinned tensor"
        return pinned, True, None

    @staticmethod
    def _normalize_exact_encoder_residency(
        exact_encoder_residency: str,
    ) -> Literal["lazy", "active_cpu", "active_pinned_cpu"]:
        normalized = str(exact_encoder_residency).strip().lower()
        allowed_values = {"lazy", "active_cpu", "active_pinned_cpu"}
        if normalized not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ValueError(
                "exact_encoder_residency must be one of: "
                f"{allowed} (got {exact_encoder_residency!r})"
            )
        return cast(Literal["lazy", "active_cpu", "active_pinned_cpu"], normalized)

    @staticmethod
    def _resolve_encoder_staging_destination(
        encoder_vecs: torch.Tensor,
        *,
        exact_chunked_mode: bool,
        encoder_residency: Literal["lazy", "active_cpu", "active_pinned_cpu"],
    ) -> str:
        if encoder_residency == "lazy":
            if exact_chunked_mode and encoder_vecs.numel() == 0:
                return "lazy_chunk_materialization"
            return "none"
        if encoder_vecs.device.type != "cpu":
            return str(encoder_vecs.device)
        return "pinned_cpu" if encoder_vecs.is_pinned() else "cpu"

    def _materialize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device = self._execution_device if device is None else device
        target_dtype = tensor.dtype if dtype is None else dtype
        if tensor.device == target_device and tensor.dtype == target_dtype:
            return tensor
        return tensor.to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=tensor.device.type == "cpu" and target_device.type == "cuda",
        )

    def get_last_token_logits(self) -> torch.Tensor:
        if self.logits.ndim >= 2 and self.logits.shape[1] == 1:
            return self.logits[:, 0]
        if self.logits.ndim >= 2:
            return self.logits[:, -1]
        return self.logits

    def get_logits_at_position(self, position: int) -> torch.Tensor:
        logit_source = self.full_logits if self.full_logits is not None else self.logits
        if self.logits.ndim < 2:
            if int(position) != 0:
                raise IndexError("logits are unpositioned; only position 0 is available")
            return logit_source
        return logit_source[:, int(position)]

    def materialize_encoder_vectors(
        self,
        indices: torch.Tensor | slice,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if self.encoder_vecs.numel() > 0:
            if isinstance(indices, slice):
                encoder_slice = self.encoder_vecs[indices]
            else:
                encoder_slice = self.encoder_vecs[
                    indices.to(device=self.encoder_vecs.device, dtype=torch.long)
                ]
            return self._materialize_tensor(
                encoder_slice, device=device, dtype=self.encoder_vecs.dtype
            )

        if self.chunked_decoder_state is None:
            raise RuntimeError("encoder vectors are unavailable and no chunked state was provided")

        materialize_rows = getattr(self.decoder_provider, "materialize_encoder_rows", None)
        if not callable(materialize_rows):
            raise RuntimeError(
                "encoder vectors are unavailable and decoder_provider does not support "
                "materialize_encoder_rows"
            )

        total_active_feats = self.activation_matrix._nnz()
        if isinstance(indices, slice):
            row_indices = torch.arange(
                total_active_feats,
                device=self.chunked_decoder_state["source_layers"].device,
                dtype=torch.long,
            )[indices]
        else:
            row_indices = indices.to(
                device=self.chunked_decoder_state["source_layers"].device,
                dtype=torch.long,
            )

        source_layers = self.chunked_decoder_state["source_layers"][row_indices]
        feature_ids = self.chunked_decoder_state["feature_ids"][row_indices]
        encoder_slice = materialize_rows(source_layers=source_layers, feature_ids=feature_ids)
        return self._materialize_tensor(encoder_slice, device=device, dtype=encoder_slice.dtype)

    def _prepare_error_vector_window(
        self, layer: int, *, device: torch.device | None = None
    ) -> None:
        if not self._stage_error_vectors_on_cpu:
            return

        start_layer = max(0, layer - self._error_vector_prefetch_lookahead + 1)
        keep_layers = set(range(start_layer, layer + 1))
        for cached_layer in list(self._materialized_error_vector_layers):
            if cached_layer not in keep_layers:
                del self._materialized_error_vector_layers[cached_layer]

        for layer_id in range(layer, start_layer - 1, -1):
            if layer_id in self._materialized_error_vector_layers:
                continue
            self._materialized_error_vector_layers[layer_id] = self._materialize_tensor(
                self.error_vectors[layer_id],
                device=device,
                dtype=self.error_vectors.dtype,
            )

        if self.diagnostic_mode:
            peak = cast(float, self._diagnostic_stats["error_vector_layers_resident_peak"])
            self._diagnostic_stats["error_vector_layers_resident_peak"] = max(
                peak,
                float(len(self._materialized_error_vector_layers)),
            )

    def get_error_vectors_for_layer(
        self, layer: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        if not self._stage_error_vectors_on_cpu:
            return self.error_vectors[layer]
        self._prepare_error_vector_window(layer, device=device)
        return self._materialized_error_vector_layers[layer]

    def _flush_chunked_feature_grad_window(
        self,
        grad_window: dict[int, torch.Tensor],
        window_layers: list[int],
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ) -> None:
        if not window_layers:
            return
        flush_start = time.perf_counter()
        flushed_layers = len(window_layers)
        grads_to_replay: list[torch.Tensor | None] = [None] * self.n_layers
        for layer in window_layers:
            grads_to_replay[layer] = grad_window.pop(layer)
        self._compute_chunked_feature_attributions_from_grads(
            grads_to_replay,
            phase_label=phase_label,
            batch_index=batch_index,
        )
        window_layers.clear()
        self._record_telemetry_event(
            scope="op",
            name="context.chunked_grad_window_flush",
            phase=phase_label,
            batch_index=batch_index,
            elapsed_ms=(time.perf_counter() - flush_start) * 1000.0,
            attrs={"flushed_layers": flushed_layers},
        )

    @staticmethod
    def _empty_like_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_sparse:
            return torch.sparse_coo_tensor(
                torch.empty((tensor.sparse_dim(), 0), dtype=torch.long),
                torch.empty((0,), dtype=tensor.dtype),
                size=tensor.shape,
            ).coalesce()
        empty_shape = (0, *tensor.shape[1:]) if tensor.ndim > 0 else (0,)
        return torch.empty(empty_shape, dtype=tensor.dtype)

    def cleanup(self) -> None:
        if self._cleanup_complete:
            return

        self.release_active_decoder_rows(reason="cleanup")
        self._decoder_row_seed = None
        self._active_decoder_row_diagnostics["decoder_active_row_seed_available"] = False
        self.clear_decoder_cache()
        self._clear_saved_grads()
        self._materialized_error_vector_layers.clear()
        self._resid_activations.clear()
        self._feature_output_activations.clear()
        self._batch_buffer = None
        self.activation_matrix = cast(torch.Tensor, self._empty_like_tensor(self.activation_matrix))
        self.error_vectors = self._empty_like_tensor(self.error_vectors)
        self.token_vectors = self._empty_like_tensor(self.token_vectors)
        self.decoder_vecs = self._empty_like_tensor(self.decoder_vecs)
        self.encoder_vecs = self._empty_like_tensor(self.encoder_vecs)
        self.encoder_to_decoder_map = self._empty_like_tensor(self.encoder_to_decoder_map)
        self.decoder_locations = self._empty_like_tensor(self.decoder_locations)
        self.logits = self._empty_like_tensor(self.logits)
        if self.full_logits is not None:
            self.full_logits = self._empty_like_tensor(self.full_logits)
        if self.chunked_decoder_state is not None:
            for key, value in list(self.chunked_decoder_state.items()):
                self.chunked_decoder_state[key] = self._empty_like_tensor(value)
        self.chunked_decoder_state = None
        self._chunked_layer_spans = None
        self._cleanup_complete = True

    def set_diagnostic_mode(self, enabled: bool) -> None:
        self.diagnostic_mode = enabled

    def configure_trace_logging(
        self,
        logger=None,
        *,
        chunk_interval: int = 16,
        trace_observer: TraceObserver | None = None,
    ) -> None:
        self._trace_logger = logger
        self._trace_observer = trace_observer
        self._trace_chunk_interval = max(1, chunk_interval)

    def _emit_trace(self, event: str, **fields: object) -> None:
        if self._trace_logger is not None:
            payload = ", ".join(f"{key}={value}" for key, value in fields.items())
            message = f"TRACE {event}"
            if payload:
                message = f"{message} | {payload}"
            self._trace_logger(message)

        if self._trace_observer is not None:
            phase_value = fields.get("phase")
            phase = str(phase_value) if isinstance(phase_value, str) else None
            if phase is None:
                phase_name = event.split(".", 1)[0]
                if phase_name.startswith("phase") and phase_name[len("phase") :].isdigit():
                    phase = phase_name
            elapsed_ms = fields.get("elapsed_ms")
            elapsed_ms_value = float(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else None
            self._trace_observer.observe(
                TraceEvent(
                    scope="op",
                    name=f"context.{event}",
                    phase=phase,
                    elapsed_ms=elapsed_ms_value,
                    attrs=fields,
                )
            )

    def _record_telemetry_event(
        self,
        *,
        scope: str,
        name: str,
        phase: str | None = None,
        step_index: int | None = None,
        batch_index: int | None = None,
        elapsed_ms: float | None = None,
        attrs: dict[str, object] | None = None,
    ) -> None:
        if self._trace_observer is None:
            return
        self._trace_observer.observe(
            TraceEvent(
                scope=cast(Literal["run", "phase", "batch", "op"], scope),
                name=name,
                phase=phase,
                step_index=step_index,
                batch_index=batch_index,
                elapsed_ms=elapsed_ms,
                attrs=attrs or {},
            )
        )

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for key, value in self._diagnostic_stats.items():
            snapshot[key] = dict(value) if isinstance(value, dict) else value
        snapshot["encoder_vecs_staged_on_cpu"] = float(self._stage_encoder_vecs_on_cpu)
        snapshot["error_vectors_staged_on_cpu"] = float(self._stage_error_vectors_on_cpu)
        snapshot["error_vector_prefetch_lookahead"] = float(self._error_vector_prefetch_lookahead)
        snapshot["error_vector_layers_resident"] = float(
            len(self._materialized_error_vector_layers)
        )
        snapshot["chunked_feature_replay_window"] = float(self._chunked_feature_replay_window)
        if self.chunked_decoder_state is not None:
            snapshot["row_subchunk_size"] = float(self._effective_row_subchunk_size())
        snapshot["logit_retention"] = self.logit_retention
        snapshot["exact_encoder_residency_requested"] = self.exact_encoder_residency_requested
        snapshot["exact_encoder_residency_effective"] = self.exact_encoder_residency_effective
        snapshot["exact_encoder_residency_applicable"] = bool(
            self.exact_encoder_residency_applicable
        )
        snapshot["exact_encoder_residency_fallback_reason"] = (
            self.exact_encoder_residency_fallback_reason
        )
        snapshot["exact_encoder_staging_destination"] = self.exact_encoder_staging_destination
        snapshot["exact_encoder_materialized_during_phase0"] = bool(
            self.exact_encoder_materialized_during_phase0
        )
        snapshot["active_encoder_shape"] = tuple(self.encoder_vecs.shape)
        snapshot["active_encoder_bytes"] = float(
            int(self.encoder_vecs.numel() * self.encoder_vecs.element_size())
        )
        snapshot["exact_encoder_pinned_requested"] = bool(self.exact_encoder_pinned_requested)
        snapshot["exact_encoder_pinned_effective"] = bool(self.exact_encoder_pinned_effective)
        snapshot["exact_encoder_pinning_success"] = self.exact_encoder_pinning_success
        snapshot["exact_encoder_pinning_failure_reason"] = self.exact_encoder_pinning_failure_reason
        snapshot["internal_precision_requested"] = self.internal_precision_requested
        snapshot["resolved_dtype_map"] = self.resolved_dtype_map
        snapshot.update(self._active_decoder_row_diagnostics)
        return snapshot

    def _add_stat(self, key: str, value: float) -> None:
        current = cast(float, self._diagnostic_stats.get(key, 0.0))
        self._diagnostic_stats[key] = current + value

    def _add_layer_stat(self, key: str, layer: int, value: float) -> None:
        bucket = cast(dict[int, float], self._diagnostic_stats.setdefault(key, {}))
        bucket[layer] = bucket.get(layer, 0.0) + value

    def _build_chunked_layer_spans(self) -> list[tuple[int, int] | None]:
        spans: list[tuple[int, int] | None] = [None] * self.n_layers
        if self.chunked_decoder_state is None:
            return spans

        source_layers = self.chunked_decoder_state["source_layers"]
        if source_layers.numel() > 1 and not bool(
            torch.all(source_layers[1:] >= source_layers[:-1])
        ):
            raise ValueError("chunked_decoder_state source layers must be sorted by layer")

        counts = torch.bincount(source_layers, minlength=self.n_layers).cpu().tolist()
        offset = 0
        for layer, count in enumerate(counts):
            if count:
                spans[layer] = (offset, offset + count)
                offset += count
        return spans

    def _refresh_chunked_layer_spans(self) -> None:
        if self.chunked_decoder_state is None:
            self._chunked_layer_spans = None
            return
        self._chunked_layer_spans = self._build_chunked_layer_spans()

    @staticmethod
    def _clear_grad_list(grad_points: list[object]) -> None:
        for grad_point in grad_points:
            proxy = cast(object, grad_point)
            try:
                setattr(proxy, "grad", None)
            except Exception:
                continue

    def _clear_saved_grads(self) -> None:
        self._clear_grad_list(cast(list[object], self._resid_activations))
        self._clear_grad_list(cast(list[object], self._feature_output_activations))

    def _create_decoder_cache(self):
        init_decoder_cache = getattr(self.decoder_provider, "create_decoder_block_cache", None)
        if self.chunked_decoder_state is None or not callable(init_decoder_cache):
            return None
        if "fingerprint" in inspect.signature(init_decoder_cache).parameters:
            return init_decoder_cache(fingerprint=self.decoder_cache_fingerprint)
        return init_decoder_cache()

    def _effective_row_subchunk_size(self) -> int:
        if self._row_subchunk_size is not None:
            return self._row_subchunk_size
        if self.decoder_provider is None:
            return 1
        chunk_size = getattr(self.decoder_provider, "decoder_chunk_size", 256)
        return max(1, int(chunk_size))

    def clear_decoder_cache(self) -> None:
        if not self._owns_decoder_chunk_cache:
            self.decoder_chunk_cache = None
            return
        clear_decoder_cache = getattr(self.decoder_provider, "clear_decoder_block_cache", None)
        if callable(clear_decoder_cache):
            clear_decoder_cache(self.decoder_chunk_cache)
        self.decoder_chunk_cache = None

    def reset_decoder_cache(self) -> None:
        if self._owns_decoder_chunk_cache:
            self.clear_decoder_cache()
            self.decoder_chunk_cache = self._create_decoder_cache()

    def release_active_decoder_rows(self, *, reason: str) -> None:
        owner = self._active_decoder_rows
        if owner is not None:
            diagnostics = owner.get_diagnostic_snapshot()
            owner.release()
            diagnostics.update(owner.get_diagnostic_snapshot())
            self._active_decoder_row_diagnostics.update(diagnostics)
        self._active_decoder_rows = None
        self._active_decoder_row_diagnostics["decoder_active_row_residency_effective"] = False
        self._active_decoder_row_diagnostics["decoder_active_row_fallback_reason"] = reason

    def prepare_active_decoder_rows(
        self,
        *,
        requested: bool,
        enabled: bool,
        max_bytes: int,
        fallback_reason: str | None = None,
        admitted_estimated_bytes: int | None = None,
    ) -> bool:
        self.release_active_decoder_rows(reason="reprepare")
        self._active_decoder_row_diagnostics.update(
            {
                "decoder_active_row_residency_requested": bool(requested),
                "decoder_active_row_residency_effective": False,
                "decoder_active_row_max_bytes": int(max_bytes),
                "decoder_active_row_fallback_reason": fallback_reason,
            }
        )
        if not enabled:
            self._decoder_row_seed = None
            self._active_decoder_row_diagnostics["decoder_active_row_seed_available"] = False
            return False
        if self.chunked_decoder_state is None or self.decoder_provider is None:
            raise RuntimeError(
                "active decoder row admission requires chunked decoder state and provider"
            )
        if self._chunked_layer_spans is None:
            raise RuntimeError("active decoder row admission requires layer spans")
        estimated_bytes = self.estimate_active_decoder_row_bytes()
        if admitted_estimated_bytes is not None and estimated_bytes != admitted_estimated_bytes:
            raise RuntimeError("active decoder row byte estimate changed after Phase-3 admission")
        self._active_decoder_row_diagnostics["decoder_active_row_estimated_bytes"] = estimated_bytes
        if estimated_bytes > int(max_bytes):
            self._decoder_row_seed = None
            self._active_decoder_row_diagnostics["decoder_active_row_seed_available"] = False
            self._active_decoder_row_diagnostics["decoder_active_row_fallback_reason"] = (
                "estimated_bytes_exceed_max"
            )
            return False
        owner = None
        seed = self._decoder_row_seed
        seed_fallback_reason = None
        if seed is not None:
            owner, missing_keys, source_mismatch = materialize_active_decoder_rows_from_seed(
                seed=seed,
                state=self.chunked_decoder_state,
                layer_spans=self._chunked_layer_spans,
                provider=self.decoder_provider,
                estimated_bytes=estimated_bytes,
                device=self._execution_device,
            )
            self._active_decoder_row_diagnostics["decoder_active_row_seed_missing_keys"] = (
                missing_keys
            )
            self._active_decoder_row_diagnostics["decoder_active_row_seed_source_mismatch"] = (
                source_mismatch
            )
            if source_mismatch:
                seed_fallback_reason = "seed_source_mismatch"
            elif missing_keys:
                seed_fallback_reason = "seed_missing_keys"
            self._active_decoder_row_diagnostics["decoder_active_row_seed_fallback_reason"] = (
                seed_fallback_reason
            )
        self._decoder_row_seed = None
        self._active_decoder_row_diagnostics["decoder_active_row_seed_available"] = False
        if owner is None:
            owner = build_active_decoder_rows(
                state=self.chunked_decoder_state,
                layer_spans=self._chunked_layer_spans,
                provider=self.decoder_provider,
                estimated_bytes=estimated_bytes,
            )
            if seed is not None:
                owner.build_source = (
                    "page_scan_after_seed_source_mismatch"
                    if seed_fallback_reason == "seed_source_mismatch"
                    else "page_scan_after_seed_miss"
                )
                owner.seed_capture_seconds = seed.capture_seconds
                owner.seed_shared_traversal_bytes = seed.shared_traversal_bytes
                owner.seed_shared_decoder_load_count = seed.shared_decoder_load_count
                owner.seed_shared_decoder_load_bytes = seed.shared_decoder_load_bytes
                owner.seed_unique_row_count = seed.unique_row_count
                owner.seed_bytes = seed.seed_bytes
                owner.seed_fallback_reason = seed_fallback_reason
        if owner.active_row_bytes != estimated_bytes:
            owner.release()
            raise RuntimeError(
                "active decoder row build size did not match the admitted byte estimate"
            )
        self._active_decoder_rows = owner
        self._active_decoder_row_diagnostics.update(owner.get_diagnostic_snapshot())
        self._active_decoder_row_diagnostics["decoder_active_row_residency_effective"] = True
        self._active_decoder_row_diagnostics["decoder_active_row_fallback_reason"] = None
        return True

    def estimate_active_decoder_row_bytes(self) -> int:
        """Return the exact compact residency size for the current active state."""

        if self.chunked_decoder_state is None or self.decoder_provider is None:
            raise RuntimeError(
                "active decoder row admission requires chunked decoder state and provider"
            )
        if self._chunked_layer_spans is None:
            raise RuntimeError("active decoder row admission requires layer spans")
        return estimate_active_decoder_row_bytes(
            layer_spans=self._chunked_layer_spans,
            provider=self.decoder_provider,
        )

    def seal_active_decoder_rows_for_checkpoint_transition(self) -> int:
        """Seal complete active-row coverage before releasing decoder assets."""

        owner = self._validated_active_decoder_rows()
        if owner is None:
            raise RuntimeError(
                "checkpoint transition requires admitted active decoder row residency"
            )
        if self.chunked_decoder_state is None or self._chunked_layer_spans is None:
            raise RuntimeError("checkpoint transition requires fixed Phase-0 decoder state")
        owner.seal(
            state=self.chunked_decoder_state,
            layer_spans=self._chunked_layer_spans,
            provider=self.decoder_provider,
        )
        self._active_decoder_row_diagnostics.update(owner.get_diagnostic_snapshot())
        self._record_telemetry_event(
            scope="op",
            name="checkpoint.active_decoder_rows.sealed",
            phase="phase3",
            attrs={
                "active_row_bytes": owner.active_row_bytes,
                "active_row_count": owner.active_row_count,
            },
        )
        return owner.active_row_bytes

    def close_owned_decoder_resources_for_checkpoint_transition(self) -> None:
        """Close in-flight decoder ownership and clear only context-owned caches."""

        owner = self._validated_active_decoder_rows()
        if owner is None or not owner.sealed:
            raise RuntimeError("decoder resources cannot close before active rows are sealed")
        prefetch = self._decoder_page_prefetch
        if prefetch is not None:
            self.close_decoder_page_prefetch(prefetch)
        self.clear_decoder_cache()

    def _validated_active_decoder_rows(self) -> ActiveDecoderRows | None:
        owner = getattr(self, "_active_decoder_rows", None)
        if owner is None or self.chunked_decoder_state is None:
            return None
        if owner.state_signature != decoder_state_signature(self.chunked_decoder_state):
            self.release_active_decoder_rows(reason="state_signature_mismatch")
            return None
        return owner

    def derive_prefix_view_context(self, target_position: int) -> "AttributionContext":
        """Return a non-mutating causal prefix view of this Phase-0 state.

        The source context may cover a longer window.  The derived context keeps
        only active features with ``position < target_position`` and slices
        position-indexed token/error state to the same prefix length.  Full logits
        are intentionally preserved so callers can still read target-position
        logits from the cached window pass.  Any decoder cache is shared but not
        owned by the returned context.
        """

        activation_matrix = self.activation_matrix.coalesce()
        n_layers, n_pos, n_features = activation_matrix.shape
        target_position = int(target_position)
        if target_position <= 0 or target_position > int(n_pos):
            raise ValueError(
                "target_position must be in [1, n_positions] for prefix view "
                f"({target_position} not in [1, {int(n_pos)}])"
            )

        if self.error_vectors.ndim < 2 or int(self.error_vectors.shape[1]) < target_position:
            raise ValueError("error_vectors do not contain the requested prefix")
        if self.token_vectors.ndim < 1 or int(self.token_vectors.shape[0]) < target_position:
            raise ValueError("token_vectors do not contain the requested prefix")

        indices = activation_matrix.indices()
        values = activation_matrix.values()
        keep = indices[1] < target_position
        kept_rows = keep.nonzero(as_tuple=False).flatten()
        filtered_activation_matrix = torch.sparse_coo_tensor(
            indices[:, keep],
            values[keep],
            size=(int(n_layers), target_position, int(n_features)),
            device=activation_matrix.device,
            dtype=activation_matrix.dtype,
        ).coalesce()

        chunked_decoder_state = None
        if self.chunked_decoder_state is not None:
            kept_indices = filtered_activation_matrix.indices()
            chunked_decoder_state = {
                "source_layers": kept_indices[0].contiguous(),
                "positions": kept_indices[1].contiguous(),
                "feature_ids": kept_indices[2].contiguous(),
                "activation_values": filtered_activation_matrix.values().contiguous(),
            }

        encoder_vecs = self.encoder_vecs
        if encoder_vecs.numel() > 0:
            encoder_vecs = encoder_vecs[kept_rows.to(device=encoder_vecs.device, dtype=torch.long)]

        decoder_vecs = self.decoder_vecs
        encoder_to_decoder_map = self.encoder_to_decoder_map
        decoder_locations = self.decoder_locations
        if chunked_decoder_state is None and encoder_to_decoder_map.numel():
            old_to_new = torch.full(
                (int(activation_matrix._nnz()),),
                -1,
                device=encoder_to_decoder_map.device,
                dtype=torch.long,
            )
            kept_rows_on_map_device = kept_rows.to(device=old_to_new.device, dtype=torch.long)
            old_to_new[kept_rows_on_map_device] = torch.arange(
                int(kept_rows.numel()), device=old_to_new.device, dtype=torch.long
            )
            keep_decoder = old_to_new[encoder_to_decoder_map.long()] >= 0
            decoder_vecs = decoder_vecs[keep_decoder.to(device=decoder_vecs.device)]
            decoder_locations = decoder_locations[
                :, keep_decoder.to(device=decoder_locations.device)
            ]
            encoder_to_decoder_map = old_to_new[encoder_to_decoder_map[keep_decoder].long()]

        shared_cache = self.decoder_chunk_cache
        if shared_cache is not None and self.decoder_cache_fingerprint is None:
            raise ValueError("cannot share decoder cache without fingerprint metadata")

        return AttributionContext(
            tensor_state=AttributionTensorState(
                activation_matrix=filtered_activation_matrix,
                error_vectors=self.error_vectors[:, :target_position].contiguous(),
                token_vectors=self.token_vectors[:target_position].contiguous(),
                decoder_vectors=decoder_vecs,
                encoder_vectors=encoder_vecs,
                encoder_to_decoder_map=encoder_to_decoder_map,
                decoder_locations=decoder_locations,
                logits=self.logits,
                full_logits=self.full_logits,
            ),
            execution_policy=ContextExecutionPolicy.resolve(
                chunked_decoder_state=chunked_decoder_state,
                encoder_vectors=encoder_vecs,
                error_vectors=self.error_vectors[:, :target_position],
                exact_encoder_residency=self.exact_encoder_residency_requested,
                stage_encoder_vectors_on_cpu=self._stage_encoder_vecs_on_cpu,
                stage_error_vectors_on_cpu=self._stage_error_vectors_on_cpu,
                error_vector_prefetch_lookahead=self._error_vector_prefetch_lookahead,
                chunked_feature_replay_window=self._chunked_feature_replay_window,
                row_subchunk_size=self._row_subchunk_size,
            ),
            decoder_runtime=DecoderRuntime.resolve(
                provider=self.decoder_provider,
                chunked_state=chunked_decoder_state,
                chunk_cache=shared_cache,
                cache_fingerprint=self.decoder_cache_fingerprint,
                decoder_row_seed=self._decoder_row_seed,
                decoder_row_seed_refusal_reason=(
                    self._active_decoder_row_diagnostics.get(
                        "decoder_active_row_seed_capture_refusal_reason"
                    )
                ),
                decoder_row_seed_estimated_bytes=(
                    self._active_decoder_row_diagnostics.get(
                        "decoder_active_row_seed_phase0_estimated_bytes"
                    )
                ),
            ),
            numeric_policy=ContextNumericPolicy(
                materialized_encoder_vectors_during_phase0=self.exact_encoder_materialized_during_phase0,
                internal_precision_requested=self.internal_precision_requested,
                resolved_dtype_map=self.resolved_dtype_map,
            ),
        )

    def apply_diagnostic_feature_cap(self, max_features: int) -> tuple[int, int]:
        total_active_feats = self.activation_matrix._nnz()
        if max_features >= total_active_feats:
            return total_active_feats, total_active_feats

        self.release_active_decoder_rows(reason="diagnostic_feature_cap")

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
        if self.encoder_vecs.numel() > 0:
            self.encoder_vecs = self.encoder_vecs[selected.to(device=self.encoder_vecs.device)]
            self.exact_encoder_pinned_effective = bool(
                self.exact_encoder_pinned_requested and self.encoder_vecs.is_pinned()
            )
            self.exact_encoder_staging_destination = self._resolve_encoder_staging_destination(
                self.encoder_vecs,
                exact_chunked_mode=self.chunked_decoder_state is not None,
                encoder_residency=self.exact_encoder_residency_effective,
            )

        if self.chunked_decoder_state is not None:
            for key in self.chunked_decoder_state:
                self.chunked_decoder_state[key] = self.chunked_decoder_state[key][selected]
            self._refresh_chunked_layer_spans()
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

    def replace_phase0_activation_state(
        self,
        activation_matrix: torch.Tensor,
    ) -> dict[str, int]:
        self.release_active_decoder_rows(reason="phase0_state_replacement")
        activation_matrix = activation_matrix.coalesce()
        n_layers, n_pos, _ = activation_matrix.shape

        if self.error_vectors.ndim >= 2 and (
            int(self.error_vectors.shape[0]) != int(n_layers)
            or int(self.error_vectors.shape[1]) != int(n_pos)
        ):
            raise ValueError(
                "replacement activation_matrix shape must match error_vectors on "
                "(layers, positions)"
            )
        if self.token_vectors.ndim >= 1 and int(self.token_vectors.shape[0]) != int(n_pos):
            raise ValueError(
                "replacement activation_matrix positions must match token_vectors length"
            )

        old_active_feature_count = int(self.activation_matrix._nnz())
        self.activation_matrix = activation_matrix
        self.n_layers = int(n_layers)

        if self.chunked_decoder_state is not None:
            activation_indices = activation_matrix.indices()
            self.chunked_decoder_state = {
                "source_layers": activation_indices[0].contiguous(),
                "positions": activation_indices[1].contiguous(),
                "feature_ids": activation_indices[2].contiguous(),
                "activation_values": activation_matrix.values().contiguous(),
            }
            self._refresh_chunked_layer_spans()
            self.reset_decoder_cache()

        self._materialized_error_vector_layers.clear()
        self._row_size = int(self.activation_matrix._nnz()) + (self.n_layers + 1) * int(n_pos)
        return {
            "old_active_feature_count": old_active_feature_count,
            "new_active_feature_count": int(self.activation_matrix._nnz()),
            "n_layers": int(n_layers),
            "n_positions": int(n_pos),
        }

    def apply_prefix_view_state(self, target_position: int) -> dict[str, int]:
        """Truncate position-indexed Phase-0 state to a causal prefix view.

        Stage-A full-sequence experiments keep full logits available for explicit
        target-position selection, but per-target attribution rows should be laid
        out exactly like an independent-prefix trace.  This method removes active
        features at target/future positions and slices error/token vectors to
        positions ``0..target_position-1`` before rebuilding derived feature-row
        state.
        """
        activation_matrix = self.activation_matrix.coalesce()
        n_layers, n_pos, n_features = activation_matrix.shape
        target_position = int(target_position)
        if target_position <= 0 or target_position > int(n_pos):
            raise ValueError(
                "target_position must be in [1, n_positions] for prefix view "
                f"({target_position} not in [1, {int(n_pos)}])"
            )

        old_active_feature_count = int(activation_matrix._nnz())
        old_position_count = int(n_pos)
        indices = activation_matrix.indices()
        values = activation_matrix.values()
        keep = indices[1] < target_position
        kept_rows = keep.nonzero(as_tuple=False).flatten()
        filtered_activation_matrix = torch.sparse_coo_tensor(
            indices[:, keep],
            values[keep],
            size=(int(n_layers), target_position, int(n_features)),
            device=activation_matrix.device,
            dtype=activation_matrix.dtype,
        ).coalesce()

        if self.error_vectors.ndim >= 2 and int(self.error_vectors.shape[1]) == int(n_pos):
            self.error_vectors = self.error_vectors[:, :target_position].contiguous()
        if self.token_vectors.ndim >= 1 and int(self.token_vectors.shape[0]) == int(n_pos):
            self.token_vectors = self.token_vectors[:target_position].contiguous()
        self._materialized_error_vector_layers.clear()

        if self.encoder_vecs.numel() > 0:
            self.encoder_vecs = self.encoder_vecs[
                kept_rows.to(device=self.encoder_vecs.device, dtype=torch.long)
            ]
            self.exact_encoder_pinned_effective = bool(
                self.exact_encoder_pinned_requested and self.encoder_vecs.is_pinned()
            )
            self.exact_encoder_staging_destination = self._resolve_encoder_staging_destination(
                self.encoder_vecs,
                exact_chunked_mode=self.chunked_decoder_state is not None,
                encoder_residency=self.exact_encoder_residency_effective,
            )

        if self.chunked_decoder_state is None and self.encoder_to_decoder_map.numel():
            old_to_new = torch.full(
                (old_active_feature_count,),
                -1,
                device=self.encoder_to_decoder_map.device,
                dtype=torch.long,
            )
            kept_rows_on_map_device = kept_rows.to(device=old_to_new.device, dtype=torch.long)
            old_to_new[kept_rows_on_map_device] = torch.arange(
                int(kept_rows.numel()), device=old_to_new.device, dtype=torch.long
            )
            keep_decoder = old_to_new[self.encoder_to_decoder_map.long()] >= 0
            self.decoder_vecs = self.decoder_vecs[keep_decoder.to(device=self.decoder_vecs.device)]
            self.decoder_locations = self.decoder_locations[
                :, keep_decoder.to(device=self.decoder_locations.device)
            ]
            self.encoder_to_decoder_map = old_to_new[
                self.encoder_to_decoder_map[keep_decoder].long()
            ]

        stats = self.replace_phase0_activation_state(filtered_activation_matrix)
        stats.update(
            {
                "old_position_count": old_position_count,
                "new_position_count": target_position,
                "old_active_feature_count": old_active_feature_count,
                "masked_active_feature_count": int((~keep).sum().item()),
                "prefix_view_target_position": target_position,
            }
        )
        return stats

    def _compute_chunked_feature_attributions_from_grad_batches(
        self,
        grad_batches: list[
            tuple[tuple[torch.Tensor | None, ...] | list[torch.Tensor | None], torch.Tensor, int]
        ],
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ) -> None:
        active_decoder_rows = self._validated_active_decoder_rows()
        prefetch = self._decoder_page_prefetch
        owns_prefetch = prefetch is None and active_decoder_rows is None
        if prefetch is None and active_decoder_rows is None:
            prefetch = self.open_decoder_page_prefetch(depth=int(self.decoder_page_prefetch_depth))
        primary_error: BaseException | None = None
        try:
            self._compute_chunked_feature_attributions_from_grad_batches_impl(
                grad_batches,
                phase_label=phase_label,
                batch_index=batch_index,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if owns_prefetch:
                assert prefetch is not None
                try:
                    self.close_decoder_page_prefetch(prefetch)
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        "decoder page prefetch cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )

    def open_decoder_page_prefetch(self, *, depth: int) -> DecoderPagePrefetch:
        if self._decoder_page_prefetch is not None:
            raise RuntimeError("decoder page prefetch lifecycle is already open")
        owner = DecoderPagePrefetch(
            provider=self.decoder_provider,
            decoder_cache=self.decoder_chunk_cache,
            depth=int(depth),
        )
        self._decoder_page_prefetch = owner
        return owner

    def close_decoder_page_prefetch(self, owner: DecoderPagePrefetch) -> None:
        if self._decoder_page_prefetch is not owner:
            raise RuntimeError("decoder page prefetch lifecycle owner mismatch")
        try:
            owner.close()
        finally:
            self._decoder_page_prefetch = None

    def _compute_chunked_feature_attributions_from_grad_batches_impl(
        self,
        grad_batches: list[
            tuple[tuple[torch.Tensor | None, ...] | list[torch.Tensor | None], torch.Tensor, int]
        ],
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ) -> None:
        assert self.chunked_decoder_state is not None
        assert self.decoder_provider is not None
        assert self._chunked_layer_spans is not None
        if not grad_batches:
            return

        active_decoder_rows = self._validated_active_decoder_rows()
        if active_decoder_rows is not None:
            contract_active_decoder_rows(
                self,
                active_decoder_rows,
                grad_batches,
                phase_label=phase_label,
                batch_index=batch_index,
            )
            return

        positions = self.chunked_decoder_state["positions"]
        feature_ids = self.chunked_decoder_state["feature_ids"]
        activation_values = self.chunked_decoder_state["activation_values"]
        chunk_size = getattr(self.decoder_provider, "decoder_chunk_size", 256)
        row_subchunk_size = min(self._effective_row_subchunk_size(), chunk_size)
        active_output_layers = sorted(
            {
                layer
                for output_layer_grads, _, _ in grad_batches
                for layer, grads in enumerate(output_layer_grads)
                if grads is not None
            }
        )
        if not active_output_layers:
            return

        output_layer_seconds = {layer: 0.0 for layer in active_output_layers}
        chunk_counts = {layer: 0 for layer in active_output_layers}
        grad_cache: dict[tuple[int, int], torch.Tensor] = {}
        replay_start = time.perf_counter()

        for layer in active_output_layers:
            self._emit_trace(
                "phase3.chunked_attr.output_layer_start",
                output_layer=layer,
                total_sources=layer + 1,
            )

        for source_layer in range(max(active_output_layers) + 1):
            source_layer_start = time.perf_counter()
            span = self._chunked_layer_spans[source_layer]
            if span is None:
                continue

            requires_provider_topology = bool(
                getattr(
                    getattr(self.decoder_provider, "capabilities", None),
                    "supports_exact_chunked_provider",
                    False,
                )
            )
            if hasattr(self.decoder_provider, "decoder_output_layers_for_source"):
                relevant_output_layers = self.decoder_provider.decoder_output_layers_for_source(
                    source_layer, active_output_layers
                )
            elif requires_provider_topology:
                raise TypeError(
                    "exact chunked provider is missing decoder_output_layers_for_source"
                )
            else:
                relevant_output_layers = [
                    layer for layer in active_output_layers if layer >= source_layer
                ]
            if not relevant_output_layers:
                continue

            layer_start, layer_end = span
            if self._produced_feature_range is not None:
                requested_start, requested_end = self._produced_feature_range
                layer_start = max(layer_start, requested_start)
                layer_end = min(layer_end, requested_end)
                if layer_start >= layer_end:
                    continue
            layer_rows = torch.arange(layer_start, layer_end, device=feature_ids.device)
            layer_feature_ids = feature_ids[layer_start:layer_end]
            layer_chunk_ids = torch.div(layer_feature_ids, chunk_size, rounding_mode="floor")
            monotonic_chunk_order = bool(
                layer_chunk_ids.numel() <= 1
                or torch.all(layer_chunk_ids[1:] >= layer_chunk_ids[:-1])
            )
            if monotonic_chunk_order:
                ordered_chunk_ids, ordered_chunk_counts = torch.unique_consecutive(
                    layer_chunk_ids,
                    return_counts=True,
                )
            else:
                ordered_chunk_ids = torch.unique(layer_chunk_ids, sorted=True)
                ordered_chunk_counts = None

            total_chunks = int(ordered_chunk_ids.numel())
            chunk_offset = 0
            for chunk_position, chunk_id_tensor in enumerate(ordered_chunk_ids, start=1):
                chunk_id = int(chunk_id_tensor.item())
                if ordered_chunk_counts is not None:
                    chunk_count = int(ordered_chunk_counts[chunk_position - 1].item())
                    chunk_end = chunk_offset + chunk_count
                    chunk_rows = layer_rows[chunk_offset:chunk_end]
                    chunk_positions = positions[chunk_rows]
                    chunk_local_feat_ids = (
                        layer_feature_ids[chunk_offset:chunk_end] - (chunk_id * chunk_size)
                    ).long()
                    chunk_offset = chunk_end
                else:
                    chunk_mask = layer_chunk_ids == chunk_id_tensor
                    chunk_rows = layer_rows[chunk_mask]
                    chunk_positions = positions[chunk_rows]
                    chunk_local_feat_ids = (
                        layer_feature_ids[chunk_mask] - (chunk_id * chunk_size)
                    ).long()
                assert self._decoder_page_prefetch is not None
                decoder_chunk = self._decoder_page_prefetch.get(source_layer, chunk_id)
                if chunk_position < total_chunks:
                    next_chunk_id = int(ordered_chunk_ids[chunk_position].item())
                    self._decoder_page_prefetch.schedule(source_layer, next_chunk_id)
                chunk_activations = activation_values[chunk_rows].to(
                    device=decoder_chunk.device,
                    dtype=grad_batches[0][1].dtype,
                    non_blocking=decoder_chunk.device.type == "cuda",
                )[:, None]
                total_row_subchunks = max(
                    (len(chunk_rows) + row_subchunk_size - 1) // row_subchunk_size,
                    1,
                )

                for output_layer in relevant_output_layers:
                    output_layer_start = time.perf_counter()
                    if hasattr(self.decoder_provider, "decoder_output_slot"):
                        decoder_slot = self.decoder_provider.decoder_output_slot(
                            source_layer, output_layer
                        )
                    elif requires_provider_topology:
                        raise TypeError("exact chunked provider is missing decoder_output_slot")
                    else:
                        decoder_slot = output_layer - source_layer
                    decoder_vectors = decoder_chunk[:, decoder_slot]
                    for output_layer_grads, batch_buffer, grad_batch_index in grad_batches:
                        grads = output_layer_grads[output_layer]
                        if grads is None:
                            continue
                        cache_key = (grad_batch_index, output_layer)
                        typed_grads = grad_cache.get(cache_key)
                        if typed_grads is None:
                            typed_grads = grads.to(
                                device=decoder_chunk.device,
                                dtype=batch_buffer.dtype,
                                non_blocking=decoder_chunk.device.type == "cuda",
                            )
                            grad_cache[cache_key] = typed_grads
                        for row_subchunk_idx, row_start in enumerate(
                            range(0, len(chunk_rows), row_subchunk_size),
                            start=1,
                        ):
                            row_stop = row_start + row_subchunk_size
                            row_slice = slice(row_start, row_stop)
                            row_chunk_rows = chunk_rows[row_slice]
                            row_chunk_positions = chunk_positions[row_slice]
                            row_chunk_local_feat_ids = chunk_local_feat_ids[row_slice]
                            row_chunk_activations = chunk_activations[row_slice]
                            selected_decoder_vectors = decoder_vectors[row_chunk_local_feat_ids].to(
                                dtype=batch_buffer.dtype
                            )
                            scaled_decoders = selected_decoder_vectors * row_chunk_activations
                            selected_grads = typed_grads[:, row_chunk_positions]
                            write_rows = row_chunk_rows
                            if self._produced_feature_range is not None:
                                write_rows = write_rows - self._produced_feature_range[0]
                            batch_buffer[write_rows] += einsum(
                                selected_grads,
                                scaled_decoders,
                                "batch position d_model, position d_model -> position batch",
                            )
                            chunk_counts[output_layer] += 1

                            if (
                                chunk_counts[output_layer] <= 2
                                or chunk_counts[output_layer] % self._trace_chunk_interval == 0
                            ):
                                self._emit_trace(
                                    "phase3.chunked_attr.chunk",
                                    output_layer=output_layer,
                                    source_layer=source_layer,
                                    chunk=chunk_counts[output_layer],
                                    decoder_chunk_id=chunk_id,
                                    processed_chunks=chunk_position,
                                    total_chunks=total_chunks,
                                    row_subchunk=row_subchunk_idx,
                                    total_row_subchunks=total_row_subchunks,
                                )

                    output_layer_seconds[output_layer] += time.perf_counter() - output_layer_start
                self._decoder_page_prefetch.finish(decoder_chunk)

            if self.diagnostic_mode:
                self._add_layer_stat(
                    "chunked_attr_seconds_by_source_layer",
                    source_layer,
                    time.perf_counter() - source_layer_start,
                )
            self._record_telemetry_event(
                scope="op",
                name="context.chunked_replay.source_layer",
                phase=phase_label,
                batch_index=batch_index,
                elapsed_ms=(time.perf_counter() - source_layer_start) * 1000.0,
                attrs={
                    "source_layer": source_layer,
                    "active_decoder_chunks": total_chunks,
                    "relevant_output_layers": int(len(relevant_output_layers)),
                },
            )

        for output_layer in active_output_layers:
            elapsed = output_layer_seconds[output_layer]
            if self.diagnostic_mode:
                self._add_layer_stat(
                    "chunked_attr_chunks_by_output_layer",
                    output_layer,
                    float(chunk_counts[output_layer]),
                )
                self._add_layer_stat("chunked_attr_seconds_by_output_layer", output_layer, elapsed)
                self._add_layer_stat("feature_attr_seconds_by_layer", output_layer, elapsed)
            self._emit_trace(
                "phase3.chunked_attr.output_layer_done",
                output_layer=output_layer,
                chunks=chunk_counts[output_layer],
                elapsed_s=f"{elapsed:.2f}",
                elapsed_ms=elapsed * 1000.0,
            )
        if self.diagnostic_mode:
            self._add_stat("chunked_attr_replay_seconds", time.perf_counter() - replay_start)
        self._record_telemetry_event(
            scope="op",
            name="context.chunked_replay",
            phase=phase_label,
            batch_index=batch_index,
            elapsed_ms=(time.perf_counter() - replay_start) * 1000.0,
            attrs={"active_output_layers": int(len(active_output_layers))},
        )

    def _compute_chunked_feature_attributions_from_grads(
        self,
        output_layer_grads: list[torch.Tensor | None],
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ) -> None:
        assert self._batch_buffer is not None
        self._compute_chunked_feature_attributions_from_grad_batches(
            [(output_layer_grads, self._batch_buffer, int(batch_index or 0))],
            phase_label=phase_label,
            batch_index=batch_index,
        )

    def replay_feature_vjp_tape(
        self,
        entries: tuple[FeatureVjpTapeEntry, ...],
        *,
        phase_label: str = "phase4_features",
    ) -> list[torch.Tensor]:
        """Contract captured batches decoder-page-major without coalescing arithmetic."""
        if self.chunked_decoder_state is None:
            raise RuntimeError("FeatureVjpTape requires exact chunked decoder state")
        if self._produced_feature_range is not None:
            raise RuntimeError("FeatureVjpTape does not support tiled feature ranges")
        self._compute_chunked_feature_attributions_from_grad_batches(
            [(entry.gradients, entry.row_buffer, entry.batch_call_index) for entry in entries],
            phase_label=phase_label,
            batch_index=None,
        )
        return [entry.row_buffer.T[: entry.batch_size] for entry in entries]

    def _compute_chunked_feature_attributions(
        self,
        layer: int,
        grads: torch.Tensor,
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ):
        self._compute_chunked_feature_attributions_from_grads(
            [grads if output_layer == layer else None for output_layer in range(self.n_layers)],
            phase_label=phase_label,
            batch_index=batch_index,
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

    def run_forward_pass(
        self,
        model: "NNSightReplacementModel",
        trace_input_ids: torch.Tensor,
        *,
        trace_batch_size: int,
    ) -> None:
        """Run and cache the Phase-1 forward pass."""
        self._replay_model = model
        self._replay_trace_input_ids = trace_input_ids.detach()
        self._replay_trace_batch_size = int(trace_batch_size)
        with model.trace() as tracer:
            with tracer.invoke(trace_input_ids.expand(trace_batch_size, -1)):
                pass

            detach_barrier = tracer.barrier(2)

            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            self.cache_residual(model, tracer, barrier=detach_barrier)

    def reset_saved_graph_handles(self) -> None:
        """Clear only Phase-1 graph handles while preserving immutable Phase-0 state."""
        self._clear_saved_grads()
        self._resid_activations.clear()
        self._feature_output_activations.clear()
        self._batch_buffer = None
        self._produced_feature_range = None
        self._produce_nonfeature = True

    def rebuild_saved_graph_handles(self) -> None:
        """Re-run the identical Phase-1 input and capacity captured by the context."""
        if (
            self._replay_model is None
            or self._replay_trace_input_ids is None
            or self._replay_trace_batch_size is None
        ):
            raise RuntimeError("Phase-1 replay state has not been configured")
        self.run_forward_pass(
            self._replay_model,
            self._replay_trace_input_ids,
            trace_batch_size=self._replay_trace_batch_size,
        )

    def release_saved_graph_handles(self) -> None:
        """Deterministically release a replay graph without clearing Phase-0 state."""
        self.reset_saved_graph_handles()

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
        acc_dtype = proxy._batch_buffer.dtype
        proxy._batch_buffer[write_index] += einsum(
            grads.to(dtype=acc_dtype)[read_index],
            output_vecs.to(dtype=acc_dtype),
            "batch position d_model, position d_model -> position batch",
        )

    def compute_feature_attributions(
        self,
        layer,
        grads,
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ):
        if self.chunked_decoder_state is not None:
            self._compute_chunked_feature_attributions(
                layer,
                grads,
                phase_label=phase_label,
                batch_index=batch_index,
            )
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
        if not self._produce_nonfeature:
            return
        _, n_pos, _ = self.activation_matrix.shape

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            feature_width = (
                self._produced_feature_range[1] - self._produced_feature_range[0]
                if self._produced_feature_range is not None
                else self.activation_matrix._nnz()
            )
            return feature_width + layer * n_pos

        self.compute_score(
            grads,
            self.get_error_vectors_for_layer(layer, device=grads.device),
            write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            read_index=np.s_[:, :n_pos],
        )

    def compute_token_attributions(self, grads):
        if not self._produce_nonfeature:
            return
        n_layers, n_pos, _ = self.activation_matrix.shape

        # Token-embedding nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            feature_width = (
                self._produced_feature_range[1] - self._produced_feature_range[0]
                if self._produced_feature_range is not None
                else self.activation_matrix._nnz()
            )
            return feature_width + layer * n_pos

        tok_start = error_offset(n_layers)
        self.compute_score(
            grads,
            self.token_vectors,
            write_index=np.s_[tok_start : tok_start + n_pos],
            read_index=np.s_[:, :n_pos],
        )

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
        phase_label: str = "unknown",
        feature_column_range: tuple[int, int] | None = None,
        include_nonfeature: bool = True,
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

        self._compute_batch_call_index += 1
        result = execute_observed_batch(
            self,
            BatchAttributionRequest(
                layers=layers,
                positions=positions,
                inject_values=inject_values,
                retain_graph=retain_graph,
                phase_label=phase_label,
                feature_column_range=feature_column_range,
                include_nonfeature=include_nonfeature,
            ),
            batch_call_index=self._compute_batch_call_index,
        )
        return result.rows

    def estimate_feature_vjp_tape_entry_nbytes(
        self,
        *,
        layers: torch.Tensor,
        batch_size: int,
    ) -> FeatureVjpTapeByteEstimate:
        """Return the exact owned bytes for a full-range deferred batch."""
        if batch_size <= 0:
            raise ValueError("FeatureVjpTape batch_size must be > 0")
        last_layer = int(layers.max().item())
        gradient_numel = sum(
            int(batch_size * self._feature_output_activations[layer + 1][0].numel())
            for layer in range(last_layer)
        )
        host_nbytes = sum(
            int(
                batch_size
                * self._feature_output_activations[layer + 1][0].numel()
                * self._feature_output_activations[layer + 1].element_size()
            )
            for layer in range(last_layer)
        )
        device_nbytes = int(gradient_numel * torch.float32.itemsize)
        row_nbytes = int(self._row_size * batch_size * torch.float32.itemsize)
        # One conservative cap bounds simultaneous host, replay-device, and row ownership.
        return FeatureVjpTapeByteEstimate(
            host_nbytes=host_nbytes,
            device_nbytes=device_nbytes,
            row_nbytes=row_nbytes,
            total_nbytes=host_nbytes + device_nbytes + row_nbytes,
        )

    def capture_feature_vjp_batch(
        self,
        *,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool,
        phase_label: str = "phase4_features",
    ) -> FeatureVjpTapeEntry:
        """Run backward/error/token attribution and defer only decoder contraction."""
        if self.chunked_decoder_state is None:
            raise RuntimeError("FeatureVjpTape requires exact chunked decoder state")
        self._compute_batch_call_index += 1
        result = execute_observed_batch(
            self,
            BatchAttributionRequest(
                layers=layers,
                positions=positions,
                inject_values=inject_values,
                retain_graph=retain_graph,
                phase_label=phase_label,
                feature_column_range=None,
                include_nonfeature=True,
            ),
            batch_call_index=self._compute_batch_call_index,
            defer_feature_vjps=True,
        )
        assert result.feature_vjp_tape_entry is not None
        return result.feature_vjp_tape_entry

    def produce_row_tiles(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        *,
        feature_column_tile_size: int,
        consume_feature_tile: Callable[[int, int, torch.Tensor], None],
        phase_label: str = "unknown",
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Produce canonical active-feature tiles and return only nonfeature columns."""
        if self.chunked_decoder_state is None:
            raise ValueError("column-tiled row production requires an exact chunked provider")
        if feature_column_tile_size <= 0:
            raise ValueError("feature_column_tile_size must be > 0")
        n_features = int(self.activation_matrix._nnz())
        for start in range(0, n_features, feature_column_tile_size):
            end = min(start + feature_column_tile_size, n_features)
            tile = self.compute_batch(
                layers,
                positions,
                inject_values,
                retain_graph=True,
                phase_label=phase_label,
                feature_column_range=(start, end),
                include_nonfeature=False,
            )
            consume_feature_tile(start, end, tile)
        return self.compute_batch(
            layers,
            positions,
            inject_values,
            retain_graph=retain_graph,
            phase_label=phase_label,
            feature_column_range=(n_features, n_features),
            include_nonfeature=True,
        )
