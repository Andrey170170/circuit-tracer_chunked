"""Stable Phase-D runtime API over the existing attribution implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from circuit_tracer.governor.contracts import canonical_json, fingerprint


RUNTIME_SCHEMA_VERSION = 1
LEGACY_TRANSLATOR_VERSION = "attribute_kwargs_v1"


@dataclass(frozen=True)
class TraceLogicalSemantics:
    """Choices which may change the mathematical result of a trace."""

    decoder_reduction_tile: int | None = None
    decoder_reduction_order: str = "canonical"
    source_group_size: int | None = None
    feature_group_size: int | None = None
    logit_group_size: int | None = None
    phase4_reference_frontier_batch: int | None = None
    phase4_refresh_stride: int = 4
    phase4_refresh_checkpoints: tuple[int, ...] = ()
    phase4_refresh_policy: str = "standard"
    phase4_refresh_interval_multiplier: int = 1
    dtype: str = "fp32"
    hooks: tuple[str, ...] = ()
    max_feature_nodes: int | None = None
    diagnostic_feature_cap: int | None = None
    provider_semantics: tuple[tuple[str, str], ...] = ()
    sequence_semantics: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TracePhysicalControls:
    """Execution-only controls; changing these must not change semantics."""

    session_capacity: int | None = None
    phase3_microbatch_max_rows: int | None = None
    phase4_microbatch_max_rows: int | None = None
    decoder_contraction_tile: int | None = None
    replay_window: int = 4
    row_production_tile: int = 2048
    influence_row_tile: int = 4096
    influence_column_tile: int = 2048
    retention: str = "full_file"
    full_retention_backend: str = "full_file"
    row_store_cache_control: str = "off"
    row_store_preallocate: bool = True
    replay_tile_cache_bytes: int | None = None
    encoder_residency: str = "lazy"
    error_vector_prefetch_lookahead: int = 2
    stage_encoder_vecs_on_cpu: bool | None = None
    stage_error_vectors_on_cpu: bool | None = None


@dataclass(frozen=True)
class TraceRequest:
    prompt: Any
    model: Any
    attribution_targets: Any = None
    logical: TraceLogicalSemantics = field(default_factory=TraceLogicalSemantics)
    physical: TracePhysicalControls = field(default_factory=TracePhysicalControls)
    legacy_kwargs: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def semantic_fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": RUNTIME_SCHEMA_VERSION,
                "logical": self.logical,
                "prompt": _stable_value(self.prompt),
                "targets": _stable_value(self.attribution_targets),
                "provider": _provider_identity(self.model),
            }
        )

    @property
    def execution_fingerprint(self) -> str:
        return fingerprint({"schema_version": RUNTIME_SCHEMA_VERSION, "physical": self.physical})


class TraceStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TraceResult:
    output: Any
    semantic_fingerprint: str
    execution_fingerprint: str
    status: TraceStatus
    telemetry_summary: Mapping[str, Any] = field(default_factory=dict)
    compatibility_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def graph(self) -> Any:
        return self.output


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _provider_identity(model: Any) -> dict[str, Any]:
    identity = {
        name: _stable_value(getattr(model, name))
        for name in ("backend", "scan", "model_name", "provider_id", "dtype")
        if hasattr(model, name)
    }
    config = getattr(model, "config", None)
    if config is not None:
        identity["model_checkpoint"] = getattr(config, "_name_or_path", None)
        identity["architectures"] = _stable_value(getattr(config, "architectures", None))
    provider = getattr(model, "transcoders", None)
    if provider is not None:
        provider = getattr(provider, "_module", provider)
        from circuit_tracer.transcoder.provider import provider_fingerprint

        identity["transcoder_provider"] = provider_fingerprint(provider)
        identity["hooks"] = {
            name: getattr(provider, name, None)
            for name in ("feature_input_hook", "feature_output_hook")
        }
    return identity


_LOGICAL_LEGACY = {
    "batch_size": "source_group_size",
    "logit_batch_size": "logit_group_size",
    "update_interval": "phase4_refresh_stride",
    "phase4_refresh_policy": "phase4_refresh_policy",
    "phase4_refresh_interval_multiplier": "phase4_refresh_interval_multiplier",
    "exact_trace_internal_dtype": "dtype",
    "max_feature_nodes": "max_feature_nodes",
    "diagnostic_feature_cap": "diagnostic_feature_cap",
}
_PHYSICAL_LEGACY = {
    "nnsight_session_capacity": "session_capacity",
    "phase3_compute_microbatch_max_rows": "phase3_microbatch_max_rows",
    "phase4_compute_microbatch_max_rows": "phase4_microbatch_max_rows",
    "row_subchunk_size": "decoder_contraction_tile",
    "chunked_feature_replay_window": "replay_window",
    "feature_row_column_tile_size": "row_production_tile",
    "influence_row_tile_size": "influence_row_tile",
    "influence_column_tile_size": "influence_column_tile",
    "feature_row_retention": "retention",
    "full_retention_backend": "full_retention_backend",
    "row_store_cache_control": "row_store_cache_control",
    "row_store_preallocate": "row_store_preallocate",
    "replay_tile_cache_bytes": "replay_tile_cache_bytes",
    "exact_encoder_residency": "encoder_residency",
    "error_vector_prefetch_lookahead": "error_vector_prefetch_lookahead",
    "stage_encoder_vecs_on_cpu": "stage_encoder_vecs_on_cpu",
    "stage_error_vectors_on_cpu": "stage_error_vectors_on_cpu",
}


def _different(explicit: Any, legacy: Any) -> bool:
    return explicit is not None and canonical_json(explicit) != canonical_json(legacy)


def _translate(request: TraceRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Purely validate and translate a request before attribution creates observers."""
    kwargs = dict(request.legacy_kwargs)
    translated: dict[str, dict[str, Any]] = {"logical": {}, "physical": {}}
    logical_defaults = TraceLogicalSemantics()
    physical_defaults = TracePhysicalControls()
    provider_identity = _provider_identity(request.model)
    provider_details = provider_identity.get("transcoder_provider", {})
    provider_decoder_tile = (
        provider_details.get("decoder_chunk_size")
        if isinstance(provider_details, Mapping)
        else None
    )
    if request.logical.decoder_reduction_order != "canonical":
        raise ValueError("provider supports only canonical decoder reduction order")
    if (
        request.logical.decoder_reduction_tile is not None
        and provider_decoder_tile is not None
        and request.logical.decoder_reduction_tile != provider_decoder_tile
    ):
        raise ValueError(
            "decoder_reduction_tile conflicts with the provider decoder chunk source"
        )
    translated["logical"]["provider_decoder_reduction"] = {
        "tile": provider_decoder_tile,
        "order": "canonical",
        "source": "transcoder_provider.decoder_chunk_size",
    }
    for legacy, field_name in _LOGICAL_LEGACY.items():
        explicit = getattr(request.logical, field_name)
        default = getattr(logical_defaults, field_name)
        if legacy in kwargs and explicit != default and _different(explicit, kwargs[legacy]):
            raise ValueError(f"conflicting logical value for {field_name}: explicit and {legacy}")
        value = kwargs.get(legacy, explicit)
        if value is not None:
            kwargs[legacy] = value
        translated["logical"][legacy] = value
    if "feature_batch_size" in kwargs:
        legacy_feature_batch = kwargs["feature_batch_size"]
        coupled = {
            "feature_group_size": request.logical.feature_group_size,
            "phase4_reference_frontier_batch": request.logical.phase4_reference_frontier_batch,
            "phase4_microbatch_max_rows": request.physical.phase4_microbatch_max_rows,
        }
        coupled_defaults = {
            "feature_group_size": logical_defaults.feature_group_size,
            "phase4_reference_frontier_batch": logical_defaults.phase4_reference_frontier_batch,
            "phase4_microbatch_max_rows": physical_defaults.phase4_microbatch_max_rows,
        }
        for field_name, explicit in coupled.items():
            if (
                request.metadata.get("source") != "legacy_facade"
                and explicit != coupled_defaults[field_name]
                and _different(
                explicit, legacy_feature_batch
                )
            ):
                raise ValueError(
                    f"conflicting coupled value for {field_name}: explicit and feature_batch_size"
                )
        kwargs.setdefault("phase4_compute_microbatch_max_rows", legacy_feature_batch)
        translated["logical"]["feature_batch_size"] = {
            "feature_group_size": legacy_feature_batch,
            "phase4_reference_frontier_batch": legacy_feature_batch,
        }
        translated["physical"]["feature_batch_size"] = {
            "phase4_microbatch_max_rows": legacy_feature_batch,
            "derivation": "legacy_matching_default",
        }
    for legacy, field_name in _PHYSICAL_LEGACY.items():
        explicit = getattr(request.physical, field_name)
        default = getattr(physical_defaults, field_name)
        if legacy in kwargs and explicit != default and _different(explicit, kwargs[legacy]):
            raise ValueError(f"conflicting physical value for {field_name}: explicit and {legacy}")
        value = kwargs.get(legacy, explicit)
        if value is not None:
            kwargs[legacy] = value
        translated["physical"][legacy] = value
    kwargs["attribution_targets"] = request.attribution_targets
    compatibility = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "translator": LEGACY_TRANSLATOR_VERSION,
        "translated": translated,
        "semantic_fingerprint": request.semantic_fingerprint,
        "execution_fingerprint": request.execution_fingerprint,
    }
    context = dict(kwargs.get("telemetry_context") or {})
    context["runtime_compatibility"] = compatibility
    kwargs["telemetry_context"] = context
    return kwargs, compatibility


def trace_one(request: TraceRequest) -> TraceResult:
    kwargs, compatibility = _translate(request)
    from circuit_tracer.attribution.attribute_nnsight import _attribute_impl as execute

    output = execute(request.prompt, request.model, **kwargs)
    if isinstance(output, Mapping):
        summary = output.get("telemetry_summary", {})
    else:
        summary = getattr(output, "telemetry_summary", {})
    return TraceResult(
        output=output,
        semantic_fingerprint=request.semantic_fingerprint,
        execution_fingerprint=request.execution_fingerprint,
        status=TraceStatus.SUCCEEDED,
        telemetry_summary=summary,
        compatibility_metadata=compatibility,
    )


def trace_batch(
    requests: Sequence[TraceRequest], *, failure: str = "raise", cancellation: Any = None
) -> list[TraceResult]:
    """Trace independently in input order; optionally retain explicit failed results."""
    if failure not in {"raise", "return"}:
        raise ValueError("failure must be 'raise' or 'return'")
    results: list[TraceResult] = []
    for request in requests:
        if cancellation is not None and cancellation.is_set():
            if failure == "raise":
                raise RuntimeError("trace batch cancelled")
            results.append(_terminal_result(request, TraceStatus.CANCELLED, "cancelled"))
            continue
        try:
            results.append(trace_one(request))
        except Exception as exc:
            if failure == "raise":
                raise
            results.append(_terminal_result(request, TraceStatus.FAILED, exc))
    return results


def _terminal_result(request: TraceRequest, status: TraceStatus, error: Any) -> TraceResult:
    return TraceResult(
        output=None,
        semantic_fingerprint=request.semantic_fingerprint,
        execution_fingerprint=request.execution_fingerprint,
        status=status,
        telemetry_summary={"error_type": type(error).__name__, "error_message": str(error)},
    )


class TraceSession:
    def __init__(self, request: TraceRequest, **session_kwargs: Any) -> None:
        self.request = request
        self._session_kwargs = dict(session_kwargs)
        self._delegate: Any = None
        self._reuse: bool | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trace session is closed")

    def trace(self, request: TraceRequest | None = None) -> TraceResult:
        self._ensure_open()
        return trace_one(request or self.request)

    def trace_window(self, target_position: int, *, reuse: bool, **kwargs: Any) -> TraceResult:
        self._ensure_open()
        if self._delegate is None:
            from circuit_tracer.attribution.attribute_nnsight import FullSequenceWindowAttributionSession

            self._delegate = FullSequenceWindowAttributionSession(
                model=self.request.model,
                full_token_ids=self.request.prompt,
                reuse_phase0_window_state=reuse,
                reuse_target_logits=reuse,
                **self._session_kwargs,
            )
            self._reuse = bool(reuse)
        elif self._reuse != bool(reuse):
            raise ValueError("reuse cannot change without reset()")
        graph = self._delegate.attribute_target_position(target_position, **kwargs)
        return TraceResult(graph, self.request.semantic_fingerprint, self.request.execution_fingerprint, TraceStatus.SUCCEEDED)

    def reset(self) -> None:
        self._ensure_open()
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            self._reuse = None
            delegate.cleanup()

    def close(self) -> None:
        try:
            if self._delegate is not None:
                delegate, self._delegate = self._delegate, None
                delegate.cleanup()
        finally:
            self._reuse = None
            self._closed = True

    def __enter__(self) -> "TraceSession":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_session(request: TraceRequest, **session_kwargs: Any) -> TraceSession:
    return TraceSession(request, **session_kwargs)


def request_from_legacy(prompt: Any, model: Any, **kwargs: Any) -> TraceRequest:
    """Deterministically translate the compatibility facade's kwargs."""
    targets = kwargs.pop("attribution_targets", None)
    logical_values = {}
    physical_values = {}
    for legacy, field_name in _LOGICAL_LEGACY.items():
        if legacy in kwargs:
            logical_values[field_name] = kwargs[legacy]
    if "feature_batch_size" in kwargs:
        logical_values["feature_group_size"] = kwargs["feature_batch_size"]
        logical_values["phase4_reference_frontier_batch"] = kwargs["feature_batch_size"]
        physical_values["phase4_microbatch_max_rows"] = kwargs["feature_batch_size"]
    for legacy, field_name in _PHYSICAL_LEGACY.items():
        if legacy in kwargs:
            physical_values[field_name] = kwargs[legacy]
    return TraceRequest(
        prompt=prompt,
        model=model,
        attribution_targets=targets,
        logical=TraceLogicalSemantics(**logical_values),
        physical=TracePhysicalControls(**physical_values),
        legacy_kwargs=kwargs,
        metadata={"source": "legacy_facade", "translator": LEGACY_TRANSLATOR_VERSION},
    )
