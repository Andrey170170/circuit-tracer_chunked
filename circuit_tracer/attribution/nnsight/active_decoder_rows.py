"""Compact, context-owned decoder rows for the fixed active feature set."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

from circuit_tracer.transcoder.attribution_result import DecoderRowSeed
from circuit_tracer.transcoder.provider import provider_fingerprint

import torch


DecoderStateSignature = tuple[tuple[str, tuple[int, ...], str, str, int, int], ...]


def decoder_state_signature(
    state: Mapping[str, torch.Tensor],
) -> DecoderStateSignature:
    """Return a cheap mutation/replacement guard for decoder-row identity."""

    signature = []
    for key in ("source_layers", "feature_ids"):
        tensor = state[key]
        signature.append(
            (
                key,
                tuple(int(dim) for dim in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                int(tensor.data_ptr()),
                int(tensor._version),
            )
        )
    return tuple(signature)


@dataclass(frozen=True)
class ActiveDecoderLayerRows:
    """Provider-dtype active decoder rows for one source layer."""

    source_layer: int
    global_row_start: int
    global_row_end: int
    output_layers: tuple[int, ...]
    rows: torch.Tensor

    def __post_init__(self) -> None:
        expected_rows = self.global_row_end - self.global_row_start
        if self.rows.ndim != 3:
            raise ValueError("active decoder layer rows must be rank 3")
        if tuple(self.rows.shape[:2]) != (expected_rows, len(self.output_layers)):
            raise ValueError(
                "active decoder layer row shape must match its global span and output layers"
            )


@dataclass
class ActiveDecoderRows:
    """Ragged-ready active decoder rows indexed by source layer."""

    layers: tuple[ActiveDecoderLayerRows | None, ...]
    state_signature: DecoderStateSignature
    estimated_bytes: int
    build_seconds: float
    build_traversal_bytes: int
    build_decoder_load_count: int
    build_decoder_load_bytes: int
    build_source: str = "page_scan"
    seed_capture_seconds: float = 0.0
    seed_shared_traversal_bytes: int = 0
    seed_shared_decoder_load_count: int = 0
    seed_shared_decoder_load_bytes: int = 0
    seed_unique_row_count: int = 0
    seed_bytes: int = 0
    seed_materialization_h2d_bytes: int = 0
    seed_fallback_reason: str | None = None
    released: bool = False

    @property
    def active_row_count(self) -> int:
        return sum(
            layer.global_row_end - layer.global_row_start
            for layer in self.layers
            if layer is not None
        )

    @property
    def active_row_bytes(self) -> int:
        return sum(
            int(layer.rows.numel() * layer.rows.element_size())
            for layer in self.layers
            if layer is not None
        )

    @property
    def residency_device(self) -> str | None:
        return next(
            (str(layer.rows.device) for layer in self.layers if layer is not None),
            None,
        )

    def release(self) -> None:
        if self.released:
            return
        self.layers = tuple(None for _ in self.layers)
        self.released = True

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "decoder_active_row_count": self.active_row_count if not self.released else 0,
            "decoder_active_row_bytes": self.active_row_bytes if not self.released else 0,
            "decoder_active_row_estimated_bytes": self.estimated_bytes,
            "decoder_active_row_build_seconds": self.build_seconds,
            "decoder_active_row_build_traversal_bytes": self.build_traversal_bytes,
            "decoder_active_row_build_decoder_load_count": self.build_decoder_load_count,
            "decoder_active_row_build_decoder_load_bytes": self.build_decoder_load_bytes,
            "decoder_active_row_build_count": 1,
            "decoder_active_row_build_source": self.build_source,
            "decoder_active_row_seed_capture_seconds": self.seed_capture_seconds,
            "decoder_active_row_seed_shared_traversal_bytes": self.seed_shared_traversal_bytes,
            "decoder_active_row_seed_shared_decoder_load_count": self.seed_shared_decoder_load_count,
            "decoder_active_row_seed_shared_decoder_load_bytes": self.seed_shared_decoder_load_bytes,
            "decoder_active_row_seed_unique_row_count": self.seed_unique_row_count,
            "decoder_active_row_seed_bytes": self.seed_bytes,
            "decoder_active_row_seed_materialization_seconds": (
                self.build_seconds if self.build_source == "phase0_fused_seed" else 0.0
            ),
            "decoder_active_row_seed_materialization_h2d_bytes": (
                self.seed_materialization_h2d_bytes
            ),
            "decoder_active_row_seed_fallback_reason": self.seed_fallback_reason,
            "decoder_active_row_seed_source_mismatch": (
                self.seed_fallback_reason == "seed_source_mismatch"
            ),
            "decoder_active_row_residency_device": (
                self.residency_device if not self.released else None
            ),
            "decoder_active_row_owner_count": 0 if self.released else 1,
        }


def estimate_active_decoder_row_bytes(
    *,
    layer_spans: list[tuple[int, int] | None],
    provider: object,
) -> int:
    """Compute the exact compact tensor size without allocating it."""

    d_model = int(getattr(provider, "d_model"))
    dtype = getattr(provider, "dtype")
    itemsize = int(dtype.itemsize)
    total = 0
    for source_layer, span in enumerate(layer_spans):
        if span is None:
            continue
        output_layers = provider.decoder_output_layers_for_source(source_layer, None)
        total += (span[1] - span[0]) * len(output_layers) * d_model * itemsize
    return total


def _diagnostic_counter(provider: object, key: str) -> int:
    snapshot = getattr(provider, "get_diagnostic_snapshot", None)
    if not callable(snapshot):
        return 0
    payload = snapshot()
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def build_active_decoder_rows(
    *,
    state: Mapping[str, torch.Tensor],
    layer_spans: list[tuple[int, int] | None],
    provider: object,
    estimated_bytes: int,
) -> ActiveDecoderRows:
    """Gather active rows during one traversal of each required decoder page."""

    started = time.perf_counter()
    load_count_before = _diagnostic_counter(provider, "decoder_load_count")
    load_bytes_before = _diagnostic_counter(provider, "decoder_load_bytes")
    feature_ids = state["feature_ids"]
    chunk_size = int(getattr(provider, "decoder_chunk_size"))
    layers: list[ActiveDecoderLayerRows | None] = []
    traversal_bytes = 0
    decoder_page: torch.Tensor | None = None
    layer_rows: torch.Tensor | None = None
    selected: torch.Tensor | None = None

    try:
        for source_layer, span in enumerate(layer_spans):
            if span is None:
                layers.append(None)
                continue
            layer_start, layer_end = span
            layer_feature_ids = feature_ids[layer_start:layer_end]
            output_layers = tuple(
                int(layer)
                for layer in provider.decoder_output_layers_for_source(source_layer, None)
            )
            layer_rows = None
            chunk_ids = torch.unique(
                torch.div(layer_feature_ids, chunk_size, rounding_mode="floor"),
                sorted=True,
            )
            for chunk_id_tensor in chunk_ids:
                chunk_id = int(chunk_id_tensor.item())
                decoder_page = provider.get_decoder_chunk(
                    source_layer,
                    chunk_id,
                    decoder_cache=None,
                )
                traversal_bytes += int(decoder_page.numel() * decoder_page.element_size())
                if layer_rows is None:
                    layer_rows = torch.empty(
                        (
                            layer_end - layer_start,
                            len(output_layers),
                            int(decoder_page.shape[-1]),
                        ),
                        device=decoder_page.device,
                        dtype=decoder_page.dtype,
                    )
                chunk_mask = (
                    torch.div(layer_feature_ids, chunk_size, rounding_mode="floor")
                    == chunk_id_tensor
                )
                destination_rows = chunk_mask.nonzero(as_tuple=False).flatten()
                local_feature_ids = (layer_feature_ids[chunk_mask] - chunk_id * chunk_size).to(
                    device=decoder_page.device, dtype=torch.long
                )
                slots = torch.tensor(
                    [
                        int(provider.decoder_output_slot(source_layer, output_layer))
                        for output_layer in output_layers
                    ],
                    device=decoder_page.device,
                    dtype=torch.long,
                )
                selected = decoder_page[local_feature_ids][:, slots]
                layer_rows[destination_rows.to(device=layer_rows.device, dtype=torch.long)] = (
                    selected
                )
                selected = None
                decoder_page = None
            if layer_rows is None:
                raise RuntimeError("active decoder layer span unexpectedly contained no rows")
            layers.append(
                ActiveDecoderLayerRows(
                    source_layer=source_layer,
                    global_row_start=layer_start,
                    global_row_end=layer_end,
                    output_layers=output_layers,
                    rows=layer_rows,
                )
            )
            layer_rows = None
    except BaseException:
        # A re-raised exception retains this frame. Clear every tensor that may own
        # or view builder allocations so a caught traceback cannot retain HBM.
        selected = None
        decoder_page = None
        layer_rows = None
        layers.clear()
        raise

    return ActiveDecoderRows(
        layers=tuple(layers),
        state_signature=decoder_state_signature(state),
        estimated_bytes=estimated_bytes,
        build_seconds=time.perf_counter() - started,
        build_traversal_bytes=traversal_bytes,
        build_decoder_load_count=(
            _diagnostic_counter(provider, "decoder_load_count") - load_count_before
        ),
        build_decoder_load_bytes=(
            _diagnostic_counter(provider, "decoder_load_bytes") - load_bytes_before
        ),
    )


def materialize_active_decoder_rows_from_seed(
    *,
    seed: DecoderRowSeed,
    state: Mapping[str, torch.Tensor],
    layer_spans: list[tuple[int, int] | None],
    provider: object,
    estimated_bytes: int,
    device: torch.device,
) -> tuple[ActiveDecoderRows | None, int, bool]:
    """Map a source-matched CPU seed without reading decoder pages."""

    if provider_fingerprint(provider) != seed.source_fingerprint:
        return None, 0, True

    started = time.perf_counter()
    feature_ids = state["feature_ids"]
    layers: list[ActiveDecoderLayerRows | None] = []
    missing_keys = 0
    h2d_bytes = 0
    rows: torch.Tensor | None = None
    try:
        for source_layer, span in enumerate(layer_spans):
            if span is None:
                layers.append(None)
                continue
            seed_layer = seed.layers[source_layer] if source_layer < len(seed.layers) else None
            if seed_layer is None or seed_layer.source_layer != source_layer:
                missing_keys += span[1] - span[0]
                layers.append(None)
                continue
            output_layers = tuple(
                int(layer)
                for layer in provider.decoder_output_layers_for_source(source_layer, None)
            )
            if seed_layer.output_layers != output_layers:
                missing_keys += span[1] - span[0]
                layers.append(None)
                continue
            requested = feature_ids[span[0] : span[1]].detach().to(device="cpu", dtype=torch.long)
            locations = torch.searchsorted(seed_layer.feature_ids, requested)
            bounded = locations.clamp(max=max(0, int(seed_layer.feature_ids.numel()) - 1))
            covered = locations < int(seed_layer.feature_ids.numel())
            if seed_layer.feature_ids.numel():
                covered &= seed_layer.feature_ids[bounded] == requested
            if not bool(torch.all(covered)):
                missing_keys += int((~covered).sum().item())
                layers.append(None)
                continue
            rows = seed_layer.rows[locations].to(device=device)
            h2d_bytes += int(rows.numel() * rows.element_size())
            layers.append(
                ActiveDecoderLayerRows(
                    source_layer=source_layer,
                    global_row_start=span[0],
                    global_row_end=span[1],
                    output_layers=output_layers,
                    rows=rows,
                )
            )
    except BaseException:
        rows = None
        layers.clear()
        raise

    if missing_keys:
        rows = None
        layers.clear()
        return None, missing_keys, False
    return (
        ActiveDecoderRows(
            layers=tuple(layers),
            state_signature=decoder_state_signature(state),
            estimated_bytes=estimated_bytes,
            build_seconds=time.perf_counter() - started,
            build_traversal_bytes=0,
            build_decoder_load_count=0,
            build_decoder_load_bytes=0,
            build_source="phase0_fused_seed",
            seed_capture_seconds=seed.capture_seconds,
            seed_shared_traversal_bytes=seed.shared_traversal_bytes,
            seed_shared_decoder_load_count=seed.shared_decoder_load_count,
            seed_shared_decoder_load_bytes=seed.shared_decoder_load_bytes,
            seed_unique_row_count=seed.unique_row_count,
            seed_bytes=seed.seed_bytes,
            seed_materialization_h2d_bytes=h2d_bytes,
        ),
        0,
        False,
    )
