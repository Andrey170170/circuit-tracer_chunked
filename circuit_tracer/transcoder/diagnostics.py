from typing import cast

import torch

from circuit_tracer.transcoder.decoder_cache import DecoderChunkCache
from circuit_tracer.observability.events import TraceEvent, TraceObserver


class DiagnosticsMixin:
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
            "phase0_activation_threshold_compare_mode": "baseline",
            "phase0_activation_threshold_compare_dtype": None,
            "phase0_threshold_membership_debug_enabled": False,
            "phase0_threshold_membership_sample_limit_per_layer": 0,
            "phase0_threshold_membership": None,
            "phase0_boundary_fingerprints": None,
        }

    def reset_diagnostic_stats(self) -> None:
        self._diagnostic_stats = self._make_empty_diagnostic_stats()

    def configure_trace_logging(
        self,
        logger=None,
        *,
        chunk_interval: int = 16,
        decoder_load_interval: int = 32,
        trace_observer: TraceObserver | None = None,
    ) -> None:
        self._trace_logger = logger
        self._trace_observer = trace_observer
        self._trace_chunk_interval = max(1, chunk_interval)
        self._trace_decoder_load_interval = max(1, decoder_load_interval)

    @staticmethod
    def _infer_phase_from_trace_event(event: str) -> str | None:
        phase_name = event.split(".", 1)[0]
        if phase_name.startswith("phase") and len(phase_name) > len("phase"):
            suffix = phase_name[len("phase") :]
            if suffix.isdigit():
                return phase_name
        return None

    def emit_trace_event(self, event: str, **fields: object) -> None:
        if self._trace_logger is not None:
            payload = ", ".join(f"{key}={value}" for key, value in fields.items())
            message = f"TRACE {event}"
            if payload:
                message = f"{message} | {payload}"
            self._trace_logger(message)

        if self._trace_observer is not None:
            elapsed_ms = fields.get("elapsed_ms")
            elapsed_ms_value: float | None
            if isinstance(elapsed_ms, (int, float)):
                elapsed_ms_value = float(elapsed_ms)
            else:
                elapsed_ms_value = None
            self._trace_observer.observe(
                TraceEvent(
                    scope="op",
                    name=f"transcoder.{event}",
                    phase=self._infer_phase_from_trace_event(event),
                    elapsed_ms=elapsed_ms_value,
                    attrs=fields,
                )
            )

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for key, value in self._diagnostic_stats.items():
            snapshot[key] = dict(value) if isinstance(value, dict) else value
        caps = self.capabilities
        snapshot.update(
            {
                "architecture": caps.architecture,
                "checkpoint_format": caps.checkpoint_format,
                "decoder_output_topology": caps.decoder_output_topology,
                "supports_exact_chunked_provider": caps.supports_exact_chunked_provider,
                "supports_decoder_chunk_cache": caps.supports_decoder_chunk_cache,
                "supports_compact_row_store": caps.supports_compact_row_store,
                "supports_exact_encoder_residency": caps.supports_exact_encoder_residency,
                "supports_encoder_row_materialization": caps.supports_encoder_row_materialization,
                "supports_lazy_decoder_chunks": caps.supports_lazy_decoder_chunks,
                "supports_lazy_encoder_rows": caps.supports_lazy_encoder_rows,
                "decoder_chunk_size": caps.default_decoder_chunk_size,
                "cross_batch_decoder_cache_bytes": caps.default_cross_batch_decoder_cache_bytes,
                "legacy_exact_chunked_decoder": caps.legacy_exact_chunked_decoder,
            }
        )
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

    def create_decoder_block_cache(
        self, max_bytes: int | None = None, *, fingerprint: object | None = None
    ) -> DecoderChunkCache | None:
        if not self.lazy_decoder:
            return None

        cache_budget = self.cross_batch_decoder_cache_bytes if max_bytes is None else max_bytes
        cache_budget = max(0, int(cache_budget))
        self._diagnostic_stats["decoder_cache_max_bytes"] = cache_budget
        self._diagnostic_stats["decoder_cache_bytes_resident"] = 0
        if cache_budget <= 0:
            return None

        self.emit_trace_event("decoder.cache.init", max_bytes=cache_budget)
        return DecoderChunkCache(cache_budget, fingerprint=fingerprint)

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
