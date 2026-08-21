"""Pre-execution validation, capability discovery, and plan resolution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from os import PathLike
from typing import Any, Mapping

from circuit_tracer.governor.contracts import ProviderProfile, ResourceEnvelope, fingerprint

from .plan import ExecutionConstraints, ResolvedTracePlan
from .problem import AllActiveSources
from .request import TraceRequest


RUNTIME_SCHEMA_VERSION = 4


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _stable(getattr(value, item.name)) for item in fields(value)}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _provider_identity(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = {
        name: _stable(getattr(model, name))
        for name in (
            "backend",
            "model_name",
            "repo_id",
            "provider_id",
            "revision",
            "dtype",
        )
        if hasattr(model, name)
    }
    scan_name = getattr(model, "scan_name", None)
    if scan_name is not None:
        semantic["scan_name"] = _stable(scan_name)
    else:
        scan = getattr(model, "scan", None)
        if scan is not None and not callable(scan):
            semantic["scan"] = _stable(scan)
    config = getattr(model, "config", None)
    if config is not None:
        semantic["model_checkpoint"] = getattr(config, "_name_or_path", None)
        semantic["architectures"] = _stable(getattr(config, "architectures", None))
    physical: dict[str, Any] = {}
    provider = getattr(model, "transcoders", None)
    if provider is not None:
        provider = getattr(provider, "_module", provider)
        from circuit_tracer.transcoder.provider import provider_fingerprint

        identity = provider_fingerprint(provider)
        semantic_keys = {
            "architecture",
            "checkpoint_format",
            "checkpoint_identity",
            "dtype",
            "n_layers",
            "d_model",
            "d_transcoder",
            "activation_kind",
            "activation_k",
            "decoder_output_topology",
            "decoder_chunk_size",
        }
        semantic["transcoder_provider"] = {
            key: value for key, value in identity.items() if key in semantic_keys
        }
        physical["transcoder_provider"] = {
            key: value for key, value in identity.items() if key not in semantic_keys
        }
    return semantic, physical


def _validate_backend_constraints(backend: str, execution: ExecutionConstraints) -> None:
    if backend == "nnsight":
        return
    if backend != "transformerlens":
        raise ValueError(f"unsupported tracing backend: {backend!r}")
    defaults = ExecutionConstraints()
    if execution != defaults:
        raise ValueError("TransformerLens tracing accepts only default execution constraints")


def _resolve_explicit_trace_request(request: TraceRequest) -> ResolvedTracePlan:
    """Validate every cross-domain choice before execution resources are created."""

    backend = getattr(request.problem.model, "backend", None)
    if not isinstance(backend, str):
        raise ValueError("problem model must declare a tracing backend")
    _validate_backend_constraints(backend, request.execution)
    if backend != "nnsight" and not isinstance(
        request.problem.source_selection, AllActiveSources
    ):
        raise ValueError("restricted source selection requires the NNSight backend")
    semantic_provider, physical_provider = _provider_identity(request.problem.model)
    semantics = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "problem": {
            "prompt": _stable(request.problem.prompt),
            "targets": _stable(request.problem.targets),
            "max_n_logits": request.problem.max_n_logits,
            "desired_logit_prob": request.problem.desired_logit_prob,
            "output_position": request.problem.output_position,
            "prefix_view": _stable(request.problem.prefix_view),
            "source_selection": _stable(request.problem.source_selection),
        },
        "semantics": _stable(request.semantics),
        "provider": semantic_provider,
    }
    execution_constraints = _stable(request.execution)
    observability = execution_constraints.get("observability")
    if isinstance(observability, dict):
        observability.pop("telemetry_jsonl_path", None)
        observability.pop("telemetry_context", None)
    execution = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "constraints": execution_constraints,
        "provider": physical_provider,
    }
    return ResolvedTracePlan(
        semantics=request.semantics,
        execution=request.execution,
        semantic_fingerprint=fingerprint(semantics),
        requested_execution_fingerprint=fingerprint(execution),
        backend=backend,
        governor_admission_mode=request.governor_admission_mode,
        evidence_metadata=request.evidence.metadata,
    )


def resolve_trace_request(
    request: TraceRequest,
    *,
    resources: ResourceEnvelope | None = None,
    provider_profile: ProviderProfile | None = None,
) -> ResolvedTracePlan:
    """Resolve an explicit request or compile one governed pre-execution plan."""

    if (resources is None) != (provider_profile is None):
        raise ValueError("resources and provider_profile must be supplied together")
    if resources is None and request.physical_requirements is not None:
        raise ValueError("physical_requirements require governed resources and provider_profile")
    explicit = _resolve_explicit_trace_request(request)
    if resources is None:
        return explicit
    from .governor_bridge import resolve_governed_trace_request

    return resolve_governed_trace_request(
        request,
        resources,
        provider_profile,
        explicit_plan=explicit,
        resolve_explicit=_resolve_explicit_trace_request,
    )
