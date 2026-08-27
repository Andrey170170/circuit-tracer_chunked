from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

from circuit_tracer.transcoder.attribution_result import AttributionComponents
from circuit_tracer.transcoder.checkpoint_working_set import ProviderCheckpointLifecycle


TranscoderArchitecture = Literal["clt", "plt"]
DecoderOutputTopology = Literal["cross_layer", "same_layer"]


@dataclass(frozen=True)
class TranscoderCapabilities:
    architecture: TranscoderArchitecture
    checkpoint_format: str
    supports_exact_chunked_provider: bool = False
    supports_compact_row_store: bool = False
    supports_decoder_chunk_cache: bool = False
    supports_exact_encoder_residency: bool = False
    supports_encoder_row_materialization: bool = False
    supports_lazy_decoder: bool = False
    supports_lazy_encoder: bool = False
    supports_lazy_decoder_chunks: bool = False
    supports_lazy_encoder_rows: bool = False
    supports_exact_row_replay: bool = False
    supports_decoder_page_prefetch: bool = False
    supports_active_decoder_row_residency: bool = False
    supports_phase0_decoder_row_ranges: bool = False
    supports_decoder_row_source: bool = False
    decoder_output_topology: DecoderOutputTopology = "cross_layer"
    default_decoder_chunk_size: int | None = None
    default_cross_batch_decoder_cache_bytes: int | None = None
    legacy_exact_chunked_decoder: bool | None = None

    @property
    def supports_encoder_residency(self) -> bool:
        return self.supports_exact_encoder_residency


@runtime_checkable
class ExactChunkedProvider(Protocol):
    architecture: TranscoderArchitecture
    capabilities: TranscoderCapabilities

    def decoder_output_layers_for_source(
        self, source_layer: int, active_output_layers: list[int] | None = None
    ) -> list[int]: ...

    def decoder_output_slot(self, source_layer: int, output_layer: int) -> int: ...

    def get_decoder_chunk(self, layer_id: int, chunk_id: int, **kwargs): ...

    def compute_attribution_components(self, *args, **kwargs) -> AttributionComponents: ...

    def materialize_encoder_rows(
        self,
        source_layers,
        feature_ids,
        *,
        device: torch.device | None = None,
    ): ...

    def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None): ...

    def clear_decoder_block_cache(self, cache) -> None: ...

    def create_decoder_row_source(
        self, source_layer: int, *, max_staging_bytes: int
    ): ...


@runtime_checkable
class CheckpointLifecycleProvider(Protocol):
    """Optional provider capability for safe checkpoint-page lifecycle control."""

    checkpoint_lifecycle: ProviderCheckpointLifecycle

    def close_decoder_checkpoint_handles(self) -> None: ...


def get_checkpoint_lifecycle_provider(
    obj: object | None,
) -> CheckpointLifecycleProvider | None:
    """Return only an explicit, fully typed lifecycle provider capability."""

    if obj is None:
        return None
    lifecycle = getattr(obj, "checkpoint_lifecycle", None)
    close_handles = getattr(obj, "close_decoder_checkpoint_handles", None)
    if not isinstance(lifecycle, ProviderCheckpointLifecycle) or not callable(close_handles):
        return None
    return obj  # type: ignore[return-value]


def get_transcoder_capabilities(obj: object) -> TranscoderCapabilities:
    capabilities = getattr(obj, "capabilities", None)
    if isinstance(capabilities, TranscoderCapabilities):
        return capabilities
    legacy_exact = bool(getattr(obj, "exact_chunked_decoder", False))
    architecture = getattr(obj, "architecture", "unknown")
    return TranscoderCapabilities(
        architecture=architecture,  # type: ignore[arg-type]
        checkpoint_format=str(getattr(obj, "weight_format", "unknown")),
        supports_exact_chunked_provider=legacy_exact,
        supports_compact_row_store=legacy_exact,
        supports_decoder_chunk_cache=legacy_exact,
        supports_exact_encoder_residency=legacy_exact,
        supports_encoder_row_materialization=bool(
            hasattr(obj, "materialize_encoder_rows") or hasattr(obj, "materialize_encoder_vectors")
        ),
        supports_lazy_decoder=bool(getattr(obj, "lazy_decoder", False)),
        supports_lazy_encoder=bool(getattr(obj, "lazy_encoder", False)),
        supports_lazy_decoder_chunks=legacy_exact,
        supports_lazy_encoder_rows=legacy_exact,
        supports_exact_row_replay=False,
        decoder_output_topology="cross_layer",
        default_decoder_chunk_size=getattr(obj, "decoder_chunk_size", None),
        default_cross_batch_decoder_cache_bytes=getattr(
            obj, "cross_batch_decoder_cache_bytes", None
        ),
        legacy_exact_chunked_decoder=legacy_exact,
    )


def supports_exact_chunked_provider(obj: object) -> bool:
    return bool(get_transcoder_capabilities(obj).supports_exact_chunked_provider)


def provider_contract_missing_methods(obj: object | None) -> tuple[str, ...]:
    if obj is None:
        return (
            "compute_attribution_components",
            "get_decoder_chunk",
            "decoder_output_layers_for_source",
            "decoder_output_slot",
        )
    caps = get_transcoder_capabilities(obj)
    required = [
        "compute_attribution_components",
        "get_decoder_chunk",
        "decoder_output_layers_for_source",
        "decoder_output_slot",
    ]
    if caps.supports_decoder_chunk_cache:
        required.extend(("create_decoder_block_cache", "clear_decoder_block_cache"))
    if caps.supports_decoder_row_source:
        required.append("create_decoder_row_source")
    required.append("materialize_encoder_rows")
    return tuple(name for name in required if not callable(getattr(obj, name, None)))


def exact_chunked_provider_usable(obj: object | None) -> bool:
    if obj is None:
        return False
    caps = get_transcoder_capabilities(obj)
    return bool(caps.supports_exact_chunked_provider and not provider_contract_missing_methods(obj))


def require_exact_chunked_provider(obj: object | None) -> bool:
    if exact_chunked_provider_usable(obj):
        return True
    if obj is not None and supports_exact_chunked_provider(obj):
        missing = ", ".join(provider_contract_missing_methods(obj))
        raise TypeError(f"exact chunked provider is missing required methods: {missing}")
    return False


def require_exact_row_replay_provider(obj: object | None) -> None:
    """Reject explicit no-retention replay unless the provider guarantees it."""

    explicit = getattr(obj, "capabilities", None)
    if not isinstance(explicit, TranscoderCapabilities):
        raise ValueError(
            "none_recompute requires explicit TranscoderCapabilities with row replay support"
        )
    if explicit.architecture not in ("clt", "plt"):
        raise ValueError("none_recompute requires explicit provider architecture clt or plt")
    if not exact_chunked_provider_usable(obj):
        missing = ", ".join(provider_contract_missing_methods(obj))
        suffix = f"; missing methods: {missing}" if missing else ""
        raise ValueError(f"none_recompute requires an exact chunked CLT/PLT provider{suffix}")
    if not explicit.supports_exact_row_replay:
        raise ValueError(
            "none_recompute requires explicit provider capability supports_exact_row_replay=True"
        )


def normalize_provider_fingerprints_for_comparison(
    expected: dict[str, object],
    current: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Project a legacy pair onto schema v1 without weakening other fields.

    Physical decoder-row optimizations were added as optional capabilities
    without changing fingerprint schema version 1. If either side predates one
    of these keys, both sides compare at the legacy default of ``False``.
    Fingerprints that both record a key continue to compare its explicit value.
    """

    normalized_expected = dict(expected)
    normalized_current = dict(current)
    for capability_key in (
        "supports_active_decoder_row_residency",
        "supports_phase0_decoder_row_ranges",
        "supports_decoder_row_source",
    ):
        if capability_key not in expected or capability_key not in current:
            normalized_expected[capability_key] = False
            normalized_current[capability_key] = False
    if (
        "decoder_row_source_backend" not in expected
        or "decoder_row_source_backend" not in current
    ):
        normalized_expected["decoder_row_source_backend"] = None
        normalized_current["decoder_row_source_backend"] = None
    return normalized_expected, normalized_current


def provider_fingerprint(
    obj: object,
    *,
    checkpoint_format: str | None = None,
    checkpoint_identity: object | None = None,
    dtype: object | None = None,
) -> dict[str, object]:
    # NNSight exposes traced modules through an Envoy whose own ``scan`` is a
    # method. Provider identity belongs to the wrapped module, not the tracing
    # facade, so normalize that boundary before reading any fingerprint fields.
    wrapped_module = getattr(obj, "_module", None)
    if wrapped_module is not None:
        obj = wrapped_module
    caps = get_transcoder_capabilities(obj)
    if checkpoint_identity is None:
        checkpoint_identity = getattr(obj, "scan", None)
    if dtype is None:
        dtype = getattr(obj, "dtype", None)
    return {
        "schema_version": 1,
        "architecture": caps.architecture,
        "checkpoint_format": checkpoint_format or caps.checkpoint_format,
        "checkpoint_identity": checkpoint_identity,
        "dtype": None if dtype is None else str(dtype),
        "n_layers": getattr(obj, "n_layers", None),
        "d_model": getattr(obj, "d_model", None),
        "d_transcoder": getattr(obj, "d_transcoder", None),
        "decoder_output_topology": caps.decoder_output_topology,
        "supports_exact_chunked_provider": caps.supports_exact_chunked_provider,
        "supports_compact_row_store": caps.supports_compact_row_store,
        "supports_decoder_chunk_cache": caps.supports_decoder_chunk_cache,
        "supports_exact_encoder_residency": caps.supports_exact_encoder_residency,
        "supports_encoder_row_materialization": caps.supports_encoder_row_materialization,
        "supports_lazy_decoder_chunks": caps.supports_lazy_decoder_chunks,
        "supports_lazy_encoder_rows": caps.supports_lazy_encoder_rows,
        "supports_exact_row_replay": caps.supports_exact_row_replay,
        "supports_active_decoder_row_residency": caps.supports_active_decoder_row_residency,
        "supports_phase0_decoder_row_ranges": caps.supports_phase0_decoder_row_ranges,
        "supports_decoder_row_source": caps.supports_decoder_row_source,
        "decoder_row_source_backend": (
            "mapped_safetensors_v1" if caps.supports_decoder_row_source else None
        ),
        "decoder_chunk_size": caps.default_decoder_chunk_size,
        "cross_batch_decoder_cache_bytes": caps.default_cross_batch_decoder_cache_bytes,
        "legacy_exact_chunked_decoder": caps.legacy_exact_chunked_decoder,
    }
