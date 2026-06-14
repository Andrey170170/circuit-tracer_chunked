"""
Attribution context for managing hooks during attribution computation.
"""

import time
import weakref
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from einops import einsum

from circuit_tracer.utils.telemetry import (
    TelemetryRecorder,
    build_memory_before_after_attrs,
    build_memory_snapshot_attrs,
    get_memory_snapshot,
)


if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )


def _slice_phase3_gradient_replay_batch(
    replay_gradients: torch.Tensor,
    *,
    layer: int,
    column_offset: int,
    batch_size: int,
) -> torch.Tensor:
    """Return one Phase-3 target/logit batch from a donor gradient tensor."""

    replay_grad = replay_gradients[layer, column_offset : column_offset + batch_size]
    if int(replay_grad.shape[0]) != int(batch_size):
        raise ValueError(
            "Phase-3 gradient replay batch slice shape mismatch "
            f"(offset={int(column_offset)}, expected={int(batch_size)}, "
            f"got={int(replay_grad.shape[0])})"
        )
    return replay_grad


_COMPUTE_BATCH_MEMORY_ATTR_KEYS: tuple[str, ...] = (
    "rss_current_gib",
    "proc_rss_anon_gib",
    "proc_rss_file_gib",
    "cgroup_memory_current_gib",
    "cgroup_memory_anon_gib",
    "cgroup_memory_file_gib",
    "cuda_allocated_gib",
    "cuda_reserved_gib",
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
        full_logits: torch.Tensor | None = None,
        decoder_provider=None,
        chunked_decoder_state: dict[str, torch.Tensor] | None = None,
        stage_encoder_vecs_on_cpu: bool | None = None,
        stage_error_vectors_on_cpu: bool | None = None,
        error_vector_prefetch_lookahead: int = 2,
        chunked_feature_replay_window: int = 4,
        row_subchunk_size: int | None = None,
        exact_encoder_residency: Literal["lazy", "active_cpu", "active_pinned_cpu"] = "lazy",
        materialized_encoder_vecs_during_phase0: bool = False,
        internal_precision_requested: str | None = None,
        resolved_dtype_map: dict[str, str] | None = None,
        decoder_chunk_cache=None,
        decoder_cache_fingerprint: object | None = None,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: list[torch.Tensor] = []
        self._feature_output_activations: list[torch.Tensor] = []
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        exact_chunked_mode = chunked_decoder_state is not None
        requested_encoder_residency = self._normalize_exact_encoder_residency(
            exact_encoder_residency
        )
        encoder_residency_applicable = bool(exact_chunked_mode)
        encoder_residency_fallback_reason: str | None = None
        effective_encoder_residency = requested_encoder_residency
        if requested_encoder_residency != "lazy" and not encoder_residency_applicable:
            effective_encoder_residency = "lazy"
            encoder_residency_fallback_reason = (
                "active encoder residency requires exact chunked decoder state; "
                "falling back to lazy execution"
            )

        if stage_encoder_vecs_on_cpu is None:
            stage_encoder_vecs_on_cpu = exact_chunked_mode and encoder_vecs.numel() > 0
        if effective_encoder_residency != "lazy":
            stage_encoder_vecs_on_cpu = True
        if stage_error_vectors_on_cpu is None:
            stage_error_vectors_on_cpu = exact_chunked_mode and error_vectors.numel() > 0

        self._execution_device = token_vectors.device
        self._stage_encoder_vecs_on_cpu = bool(stage_encoder_vecs_on_cpu)
        self._stage_error_vectors_on_cpu = bool(stage_error_vectors_on_cpu)
        self._error_vector_prefetch_lookahead = max(1, int(error_vector_prefetch_lookahead))
        self._chunked_feature_replay_window = max(1, int(chunked_feature_replay_window))
        self._row_subchunk_size = (
            None if row_subchunk_size is None else max(1, int(row_subchunk_size))
        )
        self._materialized_error_vector_layers: dict[int, torch.Tensor] = {}
        self._cleanup_complete = False
        self.exact_encoder_residency_requested = requested_encoder_residency
        self.exact_encoder_residency_effective = effective_encoder_residency
        self.exact_encoder_residency_applicable = encoder_residency_applicable
        self.exact_encoder_residency_fallback_reason = encoder_residency_fallback_reason
        self.exact_encoder_materialized_during_phase0 = bool(
            materialized_encoder_vecs_during_phase0
        )
        self.exact_encoder_pinned_requested = bool(
            effective_encoder_residency == "active_pinned_cpu"
        )
        self.exact_encoder_pinned_effective = False
        self.exact_encoder_pinning_success: bool | None = None
        self.exact_encoder_pinning_failure_reason: str | None = None
        self.exact_encoder_staging_destination = "none"

        self.logits = logits
        self.full_logits = full_logits
        self.logit_retention = (
            "full"
            if full_logits is not None or (logits.ndim >= 2 and logits.shape[1] != 1)
            else "last_token"
        )
        self.logit_source_shape = tuple((full_logits if full_logits is not None else logits).shape)
        self.activation_matrix = activation_matrix
        if self._stage_error_vectors_on_cpu:
            self.error_vectors = self._stage_tensor_on_cpu(error_vectors)
        else:
            self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        if self._stage_encoder_vecs_on_cpu:
            self.encoder_vecs, pinning_success, pinning_failure_reason = self._stage_encoder_tensor(
                encoder_vecs,
                pin_memory=self.exact_encoder_pinned_requested,
            )
            self.exact_encoder_pinning_success = pinning_success
            self.exact_encoder_pinning_failure_reason = pinning_failure_reason
            self.exact_encoder_pinned_effective = bool(
                self.exact_encoder_pinned_requested and self.encoder_vecs.is_pinned()
            )
        else:
            self.encoder_vecs = encoder_vecs
        self.exact_encoder_staging_destination = self._resolve_encoder_staging_destination(
            self.encoder_vecs,
            exact_chunked_mode=exact_chunked_mode,
            encoder_residency=self.exact_encoder_residency_effective,
        )

        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations
        self.decoder_provider = decoder_provider
        self.chunked_decoder_state = chunked_decoder_state
        self.decoder_chunk_cache = None
        self._chunked_layer_spans: list[tuple[int, int] | None] | None = None
        self.setup_diagnostic_stats: dict[str, object] | None = None
        self.sparsification_stats: dict[str, object] | None = None
        self.internal_precision_requested = internal_precision_requested
        self.resolved_dtype_map = dict(resolved_dtype_map) if resolved_dtype_map else None
        self.diagnostic_mode = False
        self._trace_logger = None
        self._telemetry_recorder: TelemetryRecorder | None = None
        self._trace_chunk_interval = 16
        self.capture_phase3_gradients = False
        self.phase3_gradient_captures: list[dict[str, torch.Tensor | int]] = []
        self.phase3_gradient_replay_tensor: torch.Tensor | None = None
        self.phase3_gradient_replay_status = "disabled"
        self.phase3_gradient_replay_column_offset = 0
        self._compute_batch_call_index = 0
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

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later
        self._refresh_chunked_layer_spans()
        self._owns_decoder_chunk_cache = decoder_chunk_cache is None
        self.decoder_cache_fingerprint = decoder_cache_fingerprint
        if decoder_chunk_cache is not None:
            expected = decoder_cache_fingerprint
            if expected is None:
                raise ValueError("shared decoder cache requires fingerprint metadata")
            actual = getattr(decoder_chunk_cache, "fingerprint", None)
            if not hasattr(decoder_chunk_cache, "fingerprint"):
                raise ValueError("shared decoder cache is missing fingerprint metadata")
            if actual != expected:
                raise ValueError(
                    f"shared decoder cache fingerprint mismatch ({actual!r} != {expected!r})"
                )
            self.decoder_chunk_cache = decoder_chunk_cache
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
        telemetry_recorder: TelemetryRecorder | None = None,
    ) -> None:
        self._trace_logger = logger
        self._telemetry_recorder = telemetry_recorder
        self._trace_chunk_interval = max(1, chunk_interval)

    def _emit_trace(self, event: str, **fields: object) -> None:
        if self._trace_logger is not None:
            payload = ", ".join(f"{key}={value}" for key, value in fields.items())
            message = f"TRACE {event}"
            if payload:
                message = f"{message} | {payload}"
            self._trace_logger(message)

        if self._telemetry_recorder is not None:
            phase_value = fields.get("phase")
            phase = str(phase_value) if isinstance(phase_value, str) else None
            if phase is None:
                phase_name = event.split(".", 1)[0]
                if phase_name.startswith("phase") and phase_name[len("phase") :].isdigit():
                    phase = phase_name
            elapsed_ms = fields.get("elapsed_ms")
            elapsed_ms_value = float(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else None
            self._telemetry_recorder.record_event(
                scope="op",
                name=f"context.{event}",
                phase=phase,
                elapsed_ms=elapsed_ms_value,
                attrs=fields,
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
        if self._telemetry_recorder is None:
            return
        self._telemetry_recorder.record_event(
            scope=scope,
            name=name,
            phase=phase,
            step_index=step_index,
            batch_index=batch_index,
            elapsed_ms=elapsed_ms,
            attrs=attrs,
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

    def _compute_chunked_feature_attributions_from_grads(
        self,
        output_layer_grads: list[torch.Tensor | None],
        *,
        phase_label: str | None = None,
        batch_index: int | None = None,
    ) -> None:
        assert self.chunked_decoder_state is not None
        assert self.decoder_provider is not None
        assert self._batch_buffer is not None
        assert self._chunked_layer_spans is not None

        positions = self.chunked_decoder_state["positions"]
        feature_ids = self.chunked_decoder_state["feature_ids"]
        activation_values = self.chunked_decoder_state["activation_values"]
        chunk_size = getattr(self.decoder_provider, "decoder_chunk_size", 256)
        row_subchunk_size = self._effective_row_subchunk_size()
        active_output_layers = [
            layer for layer, grads in enumerate(output_layer_grads) if grads is not None
        ]
        if not active_output_layers:
            return

        output_layer_seconds = {layer: 0.0 for layer in active_output_layers}
        chunk_counts = {layer: 0 for layer in active_output_layers}
        grad_cache: dict[int, torch.Tensor] = {}
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

            relevant_output_layers = [
                layer for layer in active_output_layers if layer >= source_layer
            ]
            if not relevant_output_layers:
                continue

            layer_start, layer_end = span
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
                decoder_chunk = self.decoder_provider.get_decoder_chunk(
                    source_layer,
                    chunk_id,
                    decoder_cache=self.decoder_chunk_cache,
                )
                chunk_activations = activation_values[chunk_rows].to(
                    device=decoder_chunk.device,
                    dtype=self._batch_buffer.dtype,
                    non_blocking=decoder_chunk.device.type == "cuda",
                )[:, None]
                total_row_subchunks = max(
                    (len(chunk_rows) + row_subchunk_size - 1) // row_subchunk_size,
                    1,
                )

                for output_layer in relevant_output_layers:
                    output_layer_start = time.perf_counter()
                    typed_grads = grad_cache.get(output_layer)
                    if typed_grads is None:
                        grads = output_layer_grads[output_layer]
                        assert grads is not None
                        typed_grads = grads.to(
                            device=decoder_chunk.device,
                            dtype=self._batch_buffer.dtype,
                            non_blocking=decoder_chunk.device.type == "cuda",
                        )
                        grad_cache[output_layer] = typed_grads

                    decoder_vectors = decoder_chunk[:, output_layer - source_layer].to(
                        dtype=self._batch_buffer.dtype
                    )
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
                        scaled_decoders = (
                            decoder_vectors[row_chunk_local_feat_ids] * row_chunk_activations
                        )
                        selected_grads = typed_grads[:, row_chunk_positions]
                        self._batch_buffer[row_chunk_rows] += einsum(
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
        _, n_pos, _ = self.activation_matrix.shape

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        self.compute_score(
            grads,
            self.get_error_vectors_for_layer(layer, device=grads.device),
            write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            read_index=np.s_[:, :n_pos],
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
            read_index=np.s_[:, :n_pos],
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
        self._compute_batch_call_index += 1
        batch_call_index = self._compute_batch_call_index
        batch_nodes = int(len(layers))
        unique_layers_count = int(layers.unique().numel())
        execution_device = self._resid_activations[0].device
        memory_before = get_memory_snapshot(execution_device)
        inject_values_input_nbytes = int(inject_values.numel() * inject_values.element_size())
        planned_batch_buffer_nbytes = int(
            self._row_size * batch_size * torch.tensor([], dtype=torch.float32).element_size()
        )
        self._emit_trace(
            "compute_batch.start",
            phase=phase_label,
            batch_nodes=batch_nodes,
            unique_layers=unique_layers_count,
            retain_graph=retain_graph,
            inject_values_input_nbytes=inject_values_input_nbytes,
            planned_batch_buffer_nbytes=planned_batch_buffer_nbytes,
            chunked_feature_replay_window=int(self._chunked_feature_replay_window),
            **build_memory_snapshot_attrs(
                memory_before,
                keys=_COMPUTE_BATCH_MEMORY_ATTR_KEYS,
                prefix="memory_before",
            ),
        )
        self._clear_saved_grads()
        layers = layers.to(
            device=execution_device,
            dtype=torch.long,
            non_blocking=layers.device.type == "cpu" and execution_device.type == "cuda",
        )
        positions = positions.to(
            device=execution_device,
            dtype=torch.long,
            non_blocking=positions.device.type == "cpu" and execution_device.type == "cuda",
        )
        inject_values = self._materialize_tensor(
            inject_values,
            device=execution_device,
            dtype=inject_values.dtype,
        )
        inject_values_nbytes = int(inject_values.numel() * inject_values.element_size())
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=torch.float32,
            device=inject_values.device,
        )
        batch_buffer_nbytes = int(self._batch_buffer.numel() * self._batch_buffer.element_size())

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grad_point, *, batch_indices, pos_indices, values):
            grads_out = grad_point.grad.clone()
            target_device = grads_out.device
            grads_out.index_put_(
                (
                    batch_indices.to(
                        device=target_device,
                        non_blocking=batch_indices.device.type == "cpu"
                        and target_device.type == "cuda",
                    ),
                    pos_indices.to(
                        device=target_device,
                        non_blocking=pos_indices.device.type == "cpu"
                        and target_device.type == "cuda",
                    ),
                ),
                values.to(
                    device=target_device,
                    dtype=grads_out.dtype,
                    non_blocking=values.device.type == "cpu" and target_device.type == "cuda",
                ),
            )
            grad_point.grad = grads_out

        layers_in_batch = sorted(layers.unique().tolist(), reverse=True)
        chunked_feature_grads = {} if self.chunked_decoder_state is not None else None
        chunked_feature_grad_layers: list[int] = []
        capture_phase3_gradients = bool(
            self.capture_phase3_gradients and phase_label == "phase3_logits"
        )
        replay_phase3_gradients = bool(
            self.phase3_gradient_replay_tensor is not None and phase_label == "phase3_logits"
        )
        captured_phase3_grads: list[torch.Tensor | None] | None = (
            [None] * self.n_layers if capture_phase3_gradients else None
        )
        replay_gradients = self.phase3_gradient_replay_tensor
        replay_gradient_offset = int(self.phase3_gradient_replay_column_offset)
        chunked_feature_grad_window_peak = 0

        last_layer = max(layers_in_batch)
        try:
            with self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            ):
                for layer in reversed(range(last_layer + 1)):
                    if layer != last_layer:
                        grad = self._feature_output_activations[layer + 1].grad.detach()  # type:ignore
                        if replay_phase3_gradients:
                            assert replay_gradients is not None
                            replay_grad = _slice_phase3_gradient_replay_batch(
                                replay_gradients,
                                layer=layer,
                                column_offset=replay_gradient_offset,
                                batch_size=batch_size,
                            )
                            grad = replay_grad.to(
                                device=grad.device,
                                dtype=grad.dtype,
                                non_blocking=replay_gradients.device.type == "cpu"
                                and grad.device.type == "cuda",
                            )
                        if captured_phase3_grads is not None and 0 <= layer < self.n_layers:
                            captured_phase3_grads[layer] = (
                                grad.detach().to(device="cpu", dtype=torch.float32).contiguous()
                            )
                        feature_start = time.perf_counter()
                        if chunked_feature_grads is None:
                            self.compute_feature_attributions(
                                layer,
                                grad,
                                phase_label=phase_label,
                                batch_index=batch_call_index,
                            )
                            if self.diagnostic_mode:
                                self._add_layer_stat(
                                    "feature_attr_seconds_by_layer",
                                    layer,
                                    time.perf_counter() - feature_start,
                                )
                        else:
                            chunked_feature_grads[layer] = grad
                            chunked_feature_grad_layers.append(layer)
                            chunked_feature_grad_window_peak = max(
                                chunked_feature_grad_window_peak,
                                len(chunked_feature_grad_layers),
                            )
                            if self.diagnostic_mode:
                                peak = cast(
                                    float, self._diagnostic_stats["chunked_attr_grad_window_peak"]
                                )
                                self._diagnostic_stats["chunked_attr_grad_window_peak"] = max(
                                    peak,
                                    float(len(chunked_feature_grad_layers)),
                                )
                            if (
                                len(chunked_feature_grad_layers)
                                >= self._chunked_feature_replay_window
                            ):
                                self._flush_chunked_feature_grad_window(
                                    chunked_feature_grads,
                                    chunked_feature_grad_layers,
                                    phase_label=phase_label,
                                    batch_index=batch_call_index,
                                )
                        error_start = time.perf_counter()
                        self.compute_error_attributions(layer, grad)
                        if self.diagnostic_mode:
                            self._add_layer_stat(
                                "error_attr_seconds_by_layer",
                                layer,
                                time.perf_counter() - error_start,
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
                token_grad = self._feature_output_activations[0].grad
                self.compute_token_attributions(token_grad)
                if self.diagnostic_mode:
                    self._add_stat("token_attr_seconds", time.perf_counter() - token_start)

                if chunked_feature_grads is not None:
                    self._flush_chunked_feature_grad_window(
                        chunked_feature_grads,
                        chunked_feature_grad_layers,
                        phase_label=phase_label,
                        batch_index=batch_call_index,
                    )
        finally:
            self._clear_saved_grads()

        if captured_phase3_grads is not None:
            present = [grad is not None for grad in captured_phase3_grads]
            if any(present):
                sample_grad = next(grad for grad in captured_phase3_grads if grad is not None)
                assert sample_grad is not None
                stacked_grads = []
                for grad in captured_phase3_grads:
                    if grad is None:
                        stacked_grads.append(torch.zeros_like(sample_grad))
                    else:
                        stacked_grads.append(grad)
                self.phase3_gradient_captures.append(
                    {
                        "batch_call_index": int(batch_call_index),
                        "layer_mask": torch.tensor(present, dtype=torch.bool),
                        "gradients": torch.stack(stacked_grads, dim=0),
                    }
                )

        buf, self._batch_buffer = self._batch_buffer, None
        elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        memory_after = get_memory_snapshot(execution_device)
        if self.diagnostic_mode:
            self._add_stat("compute_batch_calls", 1)
            elapsed = elapsed_ms / 1000.0
            self._add_stat("compute_batch_seconds", elapsed)
            phase_bucket = cast(
                dict[str, float],
                self._diagnostic_stats.setdefault("compute_batch_seconds_by_phase", {}),
            )
            phase_bucket[phase_label] = phase_bucket.get(phase_label, 0.0) + elapsed
        self._record_telemetry_event(
            scope="batch",
            name="context.compute_batch",
            phase=phase_label,
            batch_index=batch_call_index,
            elapsed_ms=elapsed_ms,
            attrs={
                "batch_nodes": batch_nodes,
                "batch_size": int(batch_size),
                "row_size": int(self._row_size),
                "unique_layers": len(layers_in_batch),
                "retain_graph": retain_graph,
                "chunked_decoder": self.chunked_decoder_state is not None,
                "inject_values_input_nbytes": inject_values_input_nbytes,
                "inject_values_nbytes": inject_values_nbytes,
                "batch_buffer_nbytes": batch_buffer_nbytes,
                "chunked_feature_replay_window": int(self._chunked_feature_replay_window),
                "chunked_feature_grad_window_peak": int(chunked_feature_grad_window_peak),
                **build_memory_before_after_attrs(
                    before=memory_before,
                    after=memory_after,
                    keys=_COMPUTE_BATCH_MEMORY_ATTR_KEYS,
                ),
            },
        )
        self._emit_trace(
            "compute_batch.done",
            phase=phase_label,
            batch_nodes=batch_nodes,
            unique_layers=unique_layers_count,
            retain_graph=retain_graph,
            inject_values_nbytes=inject_values_nbytes,
            batch_buffer_nbytes=batch_buffer_nbytes,
            chunked_feature_replay_window=int(self._chunked_feature_replay_window),
            chunked_feature_grad_window_peak=int(chunked_feature_grad_window_peak),
            elapsed_s=f"{elapsed_ms / 1000.0:.2f}",
            elapsed_ms=elapsed_ms,
            **build_memory_before_after_attrs(
                before=memory_before,
                after=memory_after,
                keys=_COMPUTE_BATCH_MEMORY_ATTR_KEYS,
            ),
        )
        return buf.T[: len(layers)]
