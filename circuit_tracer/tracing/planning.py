"""Pre-execution validation, capability discovery, and plan resolution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from os import PathLike
from typing import Any, Mapping

from circuit_tracer.governor.contracts import fingerprint

from .plan import ExecutionConstraints, ResolvedTracePlan
from .request import TraceRequest


RUNTIME_SCHEMA_VERSION = 3


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _stable(getattr(value, item.name)) for item in fields(value)}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _provider_identity(model: Any) -> dict[str, Any]:
    identity = {
        name: _stable(getattr(model, name))
        for name in ("backend", "scan", "model_name", "provider_id", "dtype")
        if hasattr(model, name)
    }
    config = getattr(model, "config", None)
    if config is not None:
        identity["model_checkpoint"] = getattr(config, "_name_or_path", None)
        identity["architectures"] = _stable(getattr(config, "architectures", None))
    provider = getattr(model, "transcoders", None)
    if provider is not None:
        provider = getattr(provider, "_module", provider)
        from circuit_tracer.transcoder.provider import provider_fingerprint

        identity["transcoder_provider"] = provider_fingerprint(provider)
    return identity


def _validate_backend_constraints(backend: str, execution: ExecutionConstraints) -> None:
    if backend == "nnsight":
        return
    if backend != "transformerlens":
        raise ValueError(f"unsupported tracing backend: {backend!r}")
    defaults = ExecutionConstraints()
    if execution != defaults:
        raise ValueError("TransformerLens tracing accepts only default execution constraints")


def resolve_trace_request(request: TraceRequest) -> ResolvedTracePlan:
    """Validate every cross-domain choice before execution resources are created."""

    backend = getattr(request.problem.model, "backend", None)
    if not isinstance(backend, str):
        raise ValueError("problem model must declare a tracing backend")
    _validate_backend_constraints(backend, request.execution)
    provider = _provider_identity(request.problem.model)
    semantics = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "problem": {
            "prompt": _stable(request.problem.prompt),
            "targets": _stable(request.problem.targets),
            "max_n_logits": request.problem.max_n_logits,
            "desired_logit_prob": request.problem.desired_logit_prob,
            "output_position": request.problem.output_position,
        },
        "semantics": _stable(request.semantics),
        "provider": provider,
        "evidence": {"name": request.evidence.name, "version": request.evidence.version},
    }
    execution = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "constraints": _stable(request.execution),
    }
    return ResolvedTracePlan(
        semantics=request.semantics,
        execution=request.execution,
        semantic_fingerprint=fingerprint(semantics),
        execution_fingerprint=fingerprint(execution),
        backend=backend,
        evidence_metadata=request.evidence.metadata,
        admission_report=request.evidence.advisory_governor_plan,
    )
