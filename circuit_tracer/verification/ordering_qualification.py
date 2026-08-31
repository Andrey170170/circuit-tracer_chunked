"""Model-backed qualification for selective NNSight intervention ordering.

The production verifier deliberately retains only a few scalar values from an
NNSight trace.  This module compares that path with a second execution engine:
plain Hugging Face eager forwards instrumented with ordinary PyTorch hooks.
The eager oracle is intentionally limited to the first qualified scope --
Gemma 3, same-layer PLTs, and BF16 -- rather than pretending to be a generic
intervention implementation.  It shares production plan translation and
activation-delta math, so the receipt qualifies capture ordering only; it is
not independent evidence for intervention planning or decoder arithmetic.

Running this gate does not itself enable required-mode behavioral probes.  The
model capability declaration remains a separate reviewed promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Protocol
from unittest.mock import patch

import torch
from torch import nn

from circuit_tracer.transcoder.provider import get_transcoder_capabilities

from .contracts import (
    BaselineCapture,
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionExecutionResult,
    InterventionSemantics,
    OrderingAdmissionMode,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    VariantObservation,
)
from .nnsight_runtime import NNSightInterventionRuntime, _provider_activation_delta
from .runtime import InterventionRuntimePort


RECEIPT_SCHEMA = "nnsight_intervened_forward_ordering_qualification"
RECEIPT_SCHEMA_VERSION = 1
QUALIFICATION_CLAIM = "intervened_forward_capture_ordering_only"
MAX_REFUSAL_MESSAGE_CHARS = 2_048


class OrderingQualificationVerdict(str, Enum):
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    REFUSED = "refused"


@dataclass(frozen=True)
class OrderingQualificationScope:
    model_family: str = "gemma3"
    transcoder_architecture: str = "plt"
    decoder_output_topology: str = "same_layer"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if (
            self.model_family,
            self.transcoder_architecture,
            self.decoder_output_topology,
            self.dtype,
        ) != ("gemma3", "plt", "same_layer", "bfloat16"):
            raise ValueError(
                "ordering qualification v1 is limited to Gemma3 PLT same-layer BF16"
            )


@dataclass(frozen=True)
class OrderingQualificationTolerance:
    # Candidate and oracle both report values materialized from BF16 execution.
    # Allow at most two representable BF16 steps at the larger local spacing.
    objective_max_bf16_ulps: float = 2.0
    feature_max_bf16_ulps: float = 2.0
    downstream_effect_min_abs: float = 1.0 / 128.0
    downstream_effect_min_bf16_ulps: float = 2.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class OrderingQualificationRequest:
    execution: InterventionExecutionRequest
    scope: OrderingQualificationScope = OrderingQualificationScope()
    tolerance: OrderingQualificationTolerance = OrderingQualificationTolerance()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    scope_bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        variants = self.execution.variants
        if len(variants) != 3 or variants[0].interventions:
            raise ValueError("qualification requires exactly no-op, direct, propagated")
        direct, propagated = variants[1:]
        if (
            direct.semantics is not InterventionSemantics.DIRECT_FROZEN
            or propagated.semantics
            is not InterventionSemantics.PROPAGATED_FROZEN_ATTENTION
        ):
            raise ValueError("qualification variants must be no-op, direct, propagated")
        if len(direct.interventions) != 1 or len(propagated.interventions) != 1:
            raise ValueError("qualification direct and propagated cases require one source")
        source = direct.interventions[0].node
        if propagated.interventions[0].node != source:
            raise ValueError("qualification direct and propagated sources must match")
        observed = self.execution.observed_downstream_nodes
        if not observed or any(node.layer <= source.layer for node in observed):
            raise ValueError("qualification observations must be strictly downstream")
        direct_predictions = tuple(
            value.node for value in direct.predicted_downstream_feature_deltas
        )
        if direct_predictions != observed:
            raise ValueError("qualification observations must equal direct predictions")

    @classmethod
    def from_execution_requests(
        cls,
        *,
        execution_requests: Sequence[InterventionExecutionRequest],
        scope: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> OrderingQualificationRequest:
        """Construct the single-case v1 gate from the project-owned inputs."""

        if len(execution_requests) != 1:
            raise ValueError("ordering qualification v1 requires exactly one execution request")
        execution = execution_requests[0]
        no_op = execution.variants[0]
        paired_candidates: list[
            tuple[int, int, Any, Any, tuple[FeatureValue, ...]]
        ] = []
        for index, direct in enumerate(execution.variants[1:], start=1):
            if (
                direct.semantics is not InterventionSemantics.DIRECT_FROZEN
                or len(direct.interventions) != 1
            ):
                continue
            source = direct.interventions[0].node
            later_predictions = tuple(
                sorted(
                    (
                        value
                        for value in direct.predicted_downstream_feature_deltas
                        if value.node.layer > source.layer
                    ),
                    key=lambda value: value.node,
                )
            )
            if not later_predictions:
                continue
            propagated = next(
                (
                    variant
                    for variant in execution.variants[1:]
                    if variant.semantics
                    is InterventionSemantics.PROPAGATED_FROZEN_ATTENTION
                    and len(variant.interventions) == 1
                    and variant.interventions[0].node == source
                ),
                None,
            )
            if propagated is not None:
                paired_candidates.append(
                    (source.layer, index, direct, propagated, later_predictions)
                )
        if not paired_candidates:
            raise ValueError(
                "ordering qualification requires a matched direct/propagated downstream case"
            )
        _, _, direct, propagated, later_predictions = min(
            paired_candidates, key=lambda item: (item[0], item[1])
        )
        observed_nodes = tuple(value.node for value in later_predictions)
        observed_set = set(observed_nodes)
        reduced_variants = (
            replace(no_op, predicted_downstream_feature_deltas=()),
            replace(direct, predicted_downstream_feature_deltas=later_predictions),
            replace(
                propagated,
                predicted_downstream_feature_deltas=tuple(
                    value
                    for value in propagated.predicted_downstream_feature_deltas
                    if value.node in observed_set
                ),
            ),
        )
        reduced_execution = replace(
            execution,
            variants=reduced_variants,
            observed_downstream_nodes=observed_nodes,
        )
        values = dict(scope)
        scope_bindings = dict(scope)
        model_name = str(values.pop("model_name", ""))
        provider_family = str(values.pop("provider_family", ""))
        inferred_architecture = "plt"
        if provider_family:
            if "plt" in provider_family.lower():
                inferred_architecture = "plt"
            elif "clt" in provider_family.lower():
                inferred_architecture = "clt"
            else:
                inferred_architecture = provider_family
        inferred_model_family = "gemma3"
        if model_name and "gemma-3" not in model_name.lower():
            inferred_model_family = model_name
        architecture = values.pop(
            "transcoder_architecture",
            values.pop(
                "provider_architecture",
                inferred_architecture,
            ),
        )
        resolved_scope = OrderingQualificationScope(
            model_family=str(
                values.pop(
                    "model_family",
                    inferred_model_family,
                )
            ),
            transcoder_architecture=str(architecture),
            decoder_output_topology=str(
                values.pop("decoder_output_topology", "same_layer")
            ),
            dtype=str(values.pop("dtype", values.pop("model_dtype", "bfloat16"))),
        )
        allowed_bindings = {
            "feature_input_hook",
            "feature_output_hook",
            "skip_connection",
            "ordering_mechanism",
        }
        unknown = set(values) - allowed_bindings
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown ordering qualification scope fields: {names}")
        return cls(
            execution=reduced_execution,
            scope=resolved_scope,
            provenance=dict(provenance),
            scope_bindings=scope_bindings,
        )


@dataclass(frozen=True)
class OrderingComparison:
    path: str
    oracle_value: float
    selective_value: float
    absolute_error: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class OrderingRefusalDiagnostic:
    runtime: str
    code: str
    exception_type: str | None
    message: str
    before_variant_id: str | None


@dataclass(frozen=True)
class OrderingQualificationResult:
    verdict: OrderingQualificationVerdict
    comparisons: tuple[OrderingComparison, ...]
    reasons: tuple[str, ...]
    selective_status: str
    oracle_status: str
    selective_cleanup_completed: bool
    oracle_cleanup_completed: bool
    refusal_diagnostics: tuple[OrderingRefusalDiagnostic, ...] = ()


@dataclass(frozen=True)
class OrderingQualificationReceipt:
    schema: str
    schema_version: int
    request_fingerprint: str
    scope: OrderingQualificationScope
    result: OrderingQualificationResult
    evidence_fingerprint: str
    qualification_fingerprint: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    scope_bindings: Mapping[str, Any] = field(default_factory=dict)
    request_evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _json_value(asdict(self))
        payload["status"] = self.result.verdict.value
        payload["qualification_claim"] = QUALIFICATION_CLAIM
        return payload


class _EvaluationRuntime(Protocol):
    def evaluate(self, request: InterventionExecutionRequest) -> InterventionExecutionResult: ...


@dataclass
class _EagerBaselineState:
    attention_probabilities: list[torch.Tensor]
    norm_denominators: dict[int, torch.Tensor]
    feature_inputs: dict[int, torch.Tensor]
    feature_outputs: dict[int, torch.Tensor]

    def clear(self) -> None:
        self.attention_probabilities.clear()
        self.norm_denominators.clear()
        self.feature_inputs.clear()
        self.feature_outputs.clear()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_payload(request: OrderingQualificationRequest) -> dict[str, Any]:
    execution = request.execution
    return {
        "scope": asdict(request.scope),
        "scope_bindings": request.scope_bindings,
        "tolerance": asdict(request.tolerance),
        "identity": asdict(execution.identity),
        "target": asdict(execution.target),
        "variants": [asdict(item) for item in execution.variants],
        "observed_downstream_nodes": [asdict(item) for item in execution.observed_downstream_nodes],
        "execution_policy": {
            "deadline_seconds": execution.deadline_seconds,
            "cleanup_reserve_seconds": execution.cleanup_reserve_seconds,
            "predicted_baseline_seconds": execution.predicted_baseline_seconds,
            "predicted_variant_seconds": execution.predicted_variant_seconds,
        },
    }


def _science_request_payload(request_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return only execution semantics that may affect qualification evidence."""

    return {
        key: request_evidence.get(key)
        for key in (
            "scope",
            "tolerance",
            "identity",
            "target",
            "variants",
            "observed_downstream_nodes",
        )
    }


def _qualification_evidence(
    *,
    scope: Any,
    result: Any,
    request_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "qualification_claim": QUALIFICATION_CLAIM,
        "science_request_fingerprint": _fingerprint(
            _science_request_payload(request_evidence)
        ),
        "scope": scope,
        "result": result,
    }


def validate_serialized_ordering_qualification_receipt(
    payload: Mapping[str, Any],
) -> None:
    """Fail closed when an offline receipt is incomplete or internally inconsistent."""

    serialized = _json_value(dict(payload))
    if serialized.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("ordering qualification receipt schema mismatch")
    if serialized.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("ordering qualification receipt schema version mismatch")
    if serialized.get("qualification_claim") != QUALIFICATION_CLAIM:
        raise ValueError("ordering qualification receipt claim mismatch")
    result = serialized.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("ordering qualification receipt result is missing")
    if serialized.get("status") != result.get("verdict"):
        raise ValueError("ordering qualification receipt status disagrees with verdict")
    request_evidence = serialized.get("request_evidence")
    if not isinstance(request_evidence, Mapping):
        raise ValueError("ordering qualification receipt request evidence is missing")
    if request_evidence.get("scope") != serialized.get("scope"):
        raise ValueError("ordering qualification receipt request scope mismatch")
    if request_evidence.get("scope_bindings") != serialized.get("scope_bindings"):
        raise ValueError("ordering qualification receipt scope bindings mismatch")
    request_fingerprint = serialized.get("request_fingerprint")
    if request_fingerprint != _fingerprint(request_evidence):
        raise ValueError("ordering qualification receipt request fingerprint mismatch")
    qualification_evidence = _qualification_evidence(
        scope=serialized.get("scope"),
        result=result,
        request_evidence=request_evidence,
    )
    qualification_fingerprint = serialized.get("qualification_fingerprint")
    if qualification_fingerprint != _fingerprint(qualification_evidence):
        raise ValueError("ordering qualification receipt qualification fingerprint mismatch")
    bound_evidence = {
        **qualification_evidence,
        "request_fingerprint": request_fingerprint,
        "qualification_fingerprint": qualification_fingerprint,
        "provenance": serialized.get("provenance", {}),
    }
    if serialized.get("evidence_fingerprint") != _fingerprint(bound_evidence):
        raise ValueError("ordering qualification receipt evidence fingerprint mismatch")


def validate_ordering_qualification_receipt(
    request: OrderingQualificationRequest,
    receipt: OrderingQualificationReceipt,
) -> None:
    """Validate a live typed receipt against its exact qualification request."""

    expected_request = _request_payload(request)
    if _json_value(receipt.request_evidence) != _json_value(expected_request):
        raise ValueError("ordering qualification receipt request evidence mismatch")
    validate_serialized_ordering_qualification_receipt(receipt.to_dict())


def _provider(model: Any) -> Any:
    provider = model.transcoders
    return getattr(provider, "_module", provider)


def _raw_model(model: Any) -> nn.Module:
    raw = getattr(model, "_model", None)
    if isinstance(raw, nn.Module):
        return raw
    wrapper = getattr(model, "model", None)
    raw = getattr(wrapper, "_model", None)
    if isinstance(raw, nn.Module):
        return raw
    raise ValueError("NNSight model does not expose its eager Hugging Face module")


def _language_model(raw: nn.Module) -> nn.Module:
    body = getattr(raw, "model", None)
    if body is None:
        raise ValueError("Gemma3 eager model lacks model body")
    language_model = getattr(body, "language_model", None)
    return language_model if isinstance(language_model, nn.Module) else body


def _model_family(model: Any, raw: nn.Module) -> str:
    adapter = getattr(model, "model_adapter", None)
    architecture = str(getattr(adapter, "architecture", ""))
    if not architecture:
        architectures = getattr(getattr(raw, "config", None), "architectures", None)
        if isinstance(architectures, Sequence) and architectures:
            architecture = str(architectures[0])
        else:
            architecture = type(raw).__name__
    return "gemma3" if architecture.startswith("Gemma3") else architecture.lower()


def _zero_positions(model: Any, prompt_length: int) -> set[int]:
    raw = getattr(model, "zero_positions", ())
    if isinstance(raw, slice):
        return set(range(*raw.indices(prompt_length)))
    return {int(item) for item in raw}


def _clone_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError("qualification hook expected one tensor output")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError("qualification hook expected tensor output")
    return value.detach().clone()


def _replace_tensor_output(original: Any, replacement: torch.Tensor) -> Any:
    if isinstance(original, tuple):
        if not original:
            raise ValueError("cannot replace an empty module output")
        return (replacement, *original[1:])
    return replacement


def _bounded_refusal_diagnostic(
    runtime: str, refusal: RuntimeRefusal
) -> OrderingRefusalDiagnostic:
    prefix, separator, remainder = refusal.detail.partition(": ")
    exception_type = prefix if separator and prefix.isidentifier() else None
    message = remainder if exception_type is not None else refusal.detail
    truncation = "...[truncated]"
    if len(message) > MAX_REFUSAL_MESSAGE_CHARS:
        message = message[: MAX_REFUSAL_MESSAGE_CHARS - len(truncation)] + truncation
    return OrderingRefusalDiagnostic(
        runtime,
        refusal.code,
        exception_type,
        message,
        refusal.before_variant_id,
    )


def _forward_logits(raw: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    output = raw(input_ids=tokens, use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Gemma3 eager forward did not return logits")
    return logits


class Gemma3PLTEagerHookOracle(InterventionRuntimePort):
    """Independent eager-HF oracle for the initial ordering qualification scope."""

    def __init__(self, model: Any, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._model = model
        self._raw = _raw_model(model)
        self._language = _language_model(self._raw)
        self._provider = _provider(model)
        self._clock = clock
        layers = getattr(self._language, "layers", None)
        if layers is None:
            raise ValueError("Gemma3 eager language model lacks layers")
        self._layers = tuple(layers)
        self._feature_inputs = tuple(layer.pre_feedforward_layernorm for layer in self._layers)
        self._feature_outputs = tuple(layer.post_feedforward_layernorm for layer in self._layers)
        self._norms = tuple(
            module
            for module in self._language.modules()
            if type(module).__name__ == "Gemma3RMSNorm"
        )
        self._validate_scope()

    def _validate_scope(self) -> None:
        capabilities = get_transcoder_capabilities(self._provider)
        if _model_family(self._model, self._raw) != "gemma3":
            raise ValueError("eager ordering oracle supports Gemma3 only")
        if capabilities.architecture != "plt":
            raise ValueError("eager ordering oracle supports PLT only")
        if capabilities.decoder_output_topology != "same_layer":
            raise ValueError("eager ordering oracle requires same-layer decoder output")
        if getattr(self._model, "dtype", None) is not torch.bfloat16:
            raise ValueError("eager ordering oracle requires BF16 model execution")

    def _run_forward(
        self,
        request: InterventionExecutionRequest,
        *,
        plan: Any | None,
        baseline_state: _EagerBaselineState | None,
    ) -> tuple[float, float, tuple[FeatureValue, ...], _EagerBaselineState | None]:
        prompt_length = len(request.identity.prompt_token_ids)
        device = torch.device(self._model.device)
        tokens = torch.tensor(
            (request.identity.prompt_token_ids,), dtype=torch.long, device=device
        )
        observed_nodes = tuple(
            sorted(
                set(request.observed_downstream_nodes)
                | (
                    {item.node for item in plan.interventions}
                    if plan is not None
                    else {
                        item.node
                        for variant in request.variants
                        for item in variant.interventions
                    }
                )
            )
        )
        current_inputs: dict[int, torch.Tensor] = {}
        encoded: dict[FeatureNode, torch.Tensor] = {}
        collected_attention: list[torch.Tensor] = []
        collected_denominators: dict[int, torch.Tensor] = {}
        collected_outputs: dict[int, torch.Tensor] = {}
        attention_index = 0
        original_dropout = torch.nn.functional.dropout

        direct = bool(
            plan is not None
            and plan.interventions
            and plan.semantics is InterventionSemantics.DIRECT_FROZEN
        )
        propagated = (
            plan is not None
            and bool(plan.interventions)
            and plan.semantics is InterventionSemantics.PROPAGATED_FROZEN_ATTENTION
        )
        interventions_by_layer: dict[int, list[Any]] = defaultdict(list)
        if plan is not None:
            for intervention in plan.interventions:
                interventions_by_layer[intervention.node.layer].append(intervention)
        nodes_by_layer: dict[int, tuple[FeatureNode, ...]] = {}
        for node in observed_nodes:
            nodes_by_layer.setdefault(node.layer, ())
            nodes_by_layer[node.layer] = (*nodes_by_layer[node.layer], node)
        retain_full_inputs = bool(getattr(self._provider, "skip_connection", False))

        def dropout_oracle(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal attention_index
            output = original_dropout(input, *args, **kwargs)
            if baseline_state is None:
                collected_attention.append(output.detach().clone())
                return output
            if direct or propagated:
                if attention_index >= len(baseline_state.attention_probabilities):
                    raise RuntimeError("attention ordering exceeded baseline capture")
                frozen = baseline_state.attention_probabilities[attention_index]
                attention_index += 1
                if frozen.shape != output.shape:
                    raise RuntimeError("attention ordering changed tensor shape")
                return frozen.to(device=output.device, dtype=output.dtype)
            return output

        def norm_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Gemma3 RMSNorm hook lacks tensor input")
            values = inputs[0]
            key = id(module)
            if baseline_state is None:
                eps = float(getattr(module, "eps"))
                collected_denominators[key] = torch.rsqrt(
                    values.float().pow(2).mean(-1, keepdim=True) + eps
                ).detach()
                return output
            if direct:
                denominator = baseline_state.norm_denominators.get(key)
                weight = getattr(module, "weight", None)
                if denominator is None or not isinstance(weight, torch.Tensor):
                    raise RuntimeError("direct ordering lacks baseline RMSNorm state")
                replacement = values.float() * denominator * (1.0 + weight.float())
                return _replace_tensor_output(output, replacement.to(dtype=values.dtype))
            return output

        def input_hook(layer: int):
            def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                values = _clone_tensor(output)
                if retain_full_inputs:
                    current_inputs[layer] = values
                layer_nodes = nodes_by_layer.get(layer, ())
                if not layer_nodes:
                    return
                positions = tuple(sorted({node.position for node in layer_nodes}))
                position_index = {position: index for index, position in enumerate(positions)}
                selected = values[:, positions, :]
                layer_encoded = self._provider.encode_layer(
                    selected,
                    layer,
                    apply_activation_function=False,
                ).detach().squeeze(0)
                for node in layer_nodes:
                    encoded[node] = layer_encoded[
                        position_index[node.position], node.feature
                    ]

            return capture

        def output_hook(layer: int):
            def intervene(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                values = _clone_tensor(output)
                if baseline_state is None:
                    collected_outputs[layer] = values
                    return output
                if direct:
                    values = baseline_state.feature_outputs[layer].to(
                        device=values.device, dtype=values.dtype
                    )
                    if bool(getattr(self._provider, "skip_connection", False)):
                        current = current_inputs[layer]
                        frozen = baseline_state.feature_inputs[layer].to(
                            device=current.device, dtype=current.dtype
                        )
                        values = values + (
                            self._provider.compute_skip(layer, current)
                            - self._provider.compute_skip(layer, frozen)
                        )
                else:
                    values = values.clone()
                for intervention in interventions_by_layer[layer]:
                    if direct:
                        if intervention.graph_baseline_value is None:
                            raise RuntimeError("direct oracle intervention lacks graph baseline")
                        baseline_value = torch.tensor(
                            intervention.graph_baseline_value,
                            dtype=getattr(self._model, "dtype"),
                            device=device,
                        )
                    else:
                        baseline_value = encoded[intervention.node]
                    activation_delta = _provider_activation_delta(
                        self._provider,
                        layer,
                        intervention.node.feature,
                        baseline_value,
                        intervention.absolute_value,
                    )
                    feature_id = torch.tensor(
                        [intervention.node.feature], dtype=torch.long, device=device
                    )
                    decoder = self._provider._get_decoder_vectors(layer, feature_id)
                    if decoder.ndim != 2:
                        raise RuntimeError("PLT eager oracle received non-same-layer decoder")
                    contribution = decoder[0] * activation_delta
                    values[:, intervention.node.position, :] = (
                        values[:, intervention.node.position, :] + contribution
                    )
                return _replace_tensor_output(output, values)

            return intervene

        zero_positions = _zero_positions(self._model, prompt_length)
        softcap = getattr(self._model, "zero_softcap", None)
        with ExitStack() as stack, torch.inference_mode():
            if callable(softcap):
                stack.enter_context(softcap())
            stack.enter_context(patch("torch.nn.functional.dropout", side_effect=dropout_oracle))
            for norm in self._norms:
                stack.callback(norm.register_forward_hook(norm_hook).remove)
            for layer, module in enumerate(self._feature_inputs):
                if retain_full_inputs or layer in nodes_by_layer:
                    stack.callback(module.register_forward_hook(input_hook(layer)).remove)
            for layer, module in enumerate(self._feature_outputs):
                stack.callback(module.register_forward_hook(output_hook(layer)).remove)
            logits = _forward_logits(self._raw, tokens)

        if baseline_state is not None and (direct or propagated):
            if attention_index != len(baseline_state.attention_probabilities):
                raise RuntimeError("attention ordering consumed a different capture count")
        target_logits = logits[0, request.identity.target_position - 1]
        target_logit = float(target_logits[request.identity.target_token_id].float().item())
        mean_logit = float(target_logits.float().mean().item())
        features = []
        for node in observed_nodes:
            value = 0.0 if node.position in zero_positions else float(
                encoded[node].float().item()
            )
            features.append(FeatureValue(node, value))
        state = None
        if baseline_state is None:
            if not collected_attention:
                raise RuntimeError("eager attention oracle captured no attention probabilities")
            state = _EagerBaselineState(
                collected_attention,
                collected_denominators,
                current_inputs,
                collected_outputs,
            )
        return target_logit, mean_logit, tuple(features), state

    def evaluate(self, request: InterventionExecutionRequest) -> InterventionExecutionResult:
        started = self._clock()
        state: _EagerBaselineState | None = None
        observations: list[VariantObservation] = []
        baseline: BaselineCapture | None = None
        refusal: RuntimeRefusal | None = None
        status = RuntimeExecutionStatus.REFUSED
        cleanup_completed = False
        try:
            target, mean, features, state = self._run_forward(
                request, plan=None, baseline_state=None
            )
            baseline = BaselineCapture(target - mean, (target, mean), features)
            capabilities = get_transcoder_capabilities(self._provider)
            from .nnsight_runtime import _translate_variant

            plans = tuple(
                _translate_variant(
                    variant,
                    architecture=capabilities.architecture,
                    n_layers=len(self._layers),
                    observed_nodes=request.observed_downstream_nodes,
                )
                for variant in request.variants
            )
            for plan in plans:
                target, mean, features, _ = self._run_forward(
                    request, plan=plan, baseline_state=state
                )
                observed = set(request.observed_downstream_nodes)
                intervention = set(plan.retain_intervention_nodes)
                observations.append(
                    VariantObservation(
                        plan.variant_id,
                        target - mean,
                        (target, mean),
                        tuple(item for item in features if item.node in observed),
                        self._clock() - started,
                        tuple(item for item in features if item.node in intervention),
                    )
                )
            status = RuntimeExecutionStatus.COMPLETE
        except Exception as error:
            refusal = RuntimeRefusal("oracle_failure", f"{type(error).__name__}: {error}")
            status = RuntimeExecutionStatus.PARTIAL if observations else RuntimeExecutionStatus.REFUSED
        finally:
            if state is not None:
                state.clear()
                cleanup_completed = not (
                    state.attention_probabilities
                    or state.norm_denominators
                    or state.feature_inputs
                    or state.feature_outputs
                )
            else:
                cleanup_completed = True
        return InterventionExecutionResult(
            status,
            baseline,
            tuple(observations),
            refusal,
            self._clock() - started,
            cleanup_completed,
            ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
        )


def _compare_value(
    comparisons: list[OrderingComparison],
    *,
    path: str,
    oracle: float,
    selective: float,
    max_bf16_ulps: float,
) -> None:
    absolute = abs(selective - oracle)
    relative = absolute / max(abs(oracle), 1e-12)
    allowed_error = max_bf16_ulps * max(
        _bf16_ulp(oracle),
        _bf16_ulp(selective),
    )
    comparisons.append(
        OrderingComparison(
            path,
            oracle,
            selective,
            absolute,
            relative,
            absolute <= allowed_error,
        )
    )


def _feature_map(values: Sequence[FeatureValue]) -> dict[FeatureNode, float]:
    return {item.node: item.preactivation for item in values}


def _bf16_ulp(value: float) -> float:
    """Return the spacing of finite BF16 values at ``value`` without tensor casts."""

    magnitude = abs(value)
    if magnitude < 2.0**-126:
        return 2.0**-133
    _, exponent = math.frexp(magnitude)
    return 2.0 ** (exponent - 8)


def _has_meaningful_downstream_effect(
    baseline_features: Mapping[FeatureNode, float],
    observation: VariantObservation,
    *,
    source: FeatureNode,
    tolerance: OrderingQualificationTolerance,
) -> bool:
    for item in observation.downstream_feature_values:
        if item.node.layer <= source.layer:
            continue
        baseline = baseline_features.get(item.node)
        if baseline is None:
            continue
        floor = max(
            tolerance.downstream_effect_min_abs,
            tolerance.downstream_effect_min_bf16_ulps
            * max(_bf16_ulp(baseline), _bf16_ulp(item.preactivation)),
        )
        if abs(item.preactivation - baseline) > floor:
            return True
    return False


def _compare_execution_results(
    request: OrderingQualificationRequest,
    oracle: InterventionExecutionResult,
    selective: InterventionExecutionResult,
) -> OrderingQualificationResult:
    reasons: list[str] = []
    comparisons: list[OrderingComparison] = []
    refusal_diagnostics: list[OrderingRefusalDiagnostic] = []
    for label, result in (("oracle", oracle), ("selective", selective)):
        if result.status is not RuntimeExecutionStatus.COMPLETE:
            reasons.append(f"{label}_status_{result.status.value}")
        if result.refusal is not None:
            reasons.append(f"{label}_refusal_{result.refusal.code}")
            refusal_diagnostics.append(
                _bounded_refusal_diagnostic(label, result.refusal)
            )
        if not result.cleanup_completed:
            reasons.append(f"{label}_cleanup_incomplete")
    if reasons or oracle.baseline is None or selective.baseline is None:
        if oracle.baseline is None:
            reasons.append("oracle_baseline_missing")
        if selective.baseline is None:
            reasons.append("selective_baseline_missing")
        return OrderingQualificationResult(
            OrderingQualificationVerdict.REFUSED,
            (),
            tuple(dict.fromkeys(reasons)),
            selective.status.value,
            oracle.status.value,
            selective.cleanup_completed,
            oracle.cleanup_completed,
            tuple(refusal_diagnostics),
        )

    tolerance = request.tolerance
    for index, name in enumerate(("target_logit", "mean_logit")):
        _compare_value(
            comparisons,
            path=f"baseline.{name}",
            oracle=oracle.baseline.raw_target_logits[index],
            selective=selective.baseline.raw_target_logits[index],
            max_bf16_ulps=tolerance.objective_max_bf16_ulps,
        )
    oracle_baseline_features = _feature_map(oracle.baseline.feature_values)
    selective_baseline_features = _feature_map(selective.baseline.feature_values)
    if oracle_baseline_features.keys() != selective_baseline_features.keys():
        reasons.append("baseline_feature_key_mismatch")
    else:
        for node in sorted(oracle_baseline_features):
            _compare_value(
                comparisons,
                path=f"baseline.feature.{node.layer}.{node.position}.{node.feature}",
                oracle=oracle_baseline_features[node],
                selective=selective_baseline_features[node],
                max_bf16_ulps=tolerance.feature_max_bf16_ulps,
            )

    oracle_by_id = {item.variant_id: item for item in oracle.observations}
    selective_by_id = {item.variant_id: item for item in selective.observations}
    expected_ids = tuple(item.variant_id for item in request.execution.variants)
    if tuple(oracle_by_id) != expected_ids or tuple(selective_by_id) != expected_ids:
        reasons.append("variant_order_mismatch")
    for variant_id in expected_ids:
        oracle_observation = oracle_by_id.get(variant_id)
        selective_observation = selective_by_id.get(variant_id)
        if oracle_observation is None or selective_observation is None:
            continue
        for index, name in enumerate(("target_logit", "mean_logit")):
            _compare_value(
                comparisons,
                path=f"variant.{variant_id}.{name}",
                oracle=oracle_observation.raw_target_logits[index],
                selective=selective_observation.raw_target_logits[index],
                max_bf16_ulps=tolerance.objective_max_bf16_ulps,
            )
        for family, oracle_values, selective_values in (
            (
                "downstream",
                oracle_observation.downstream_feature_values,
                selective_observation.downstream_feature_values,
            ),
            (
                "intervention",
                oracle_observation.intervention_feature_values,
                selective_observation.intervention_feature_values,
            ),
        ):
            oracle_features = _feature_map(oracle_values)
            selective_features = _feature_map(selective_values)
            if oracle_features.keys() != selective_features.keys():
                reasons.append(f"variant_{variant_id}_{family}_feature_key_mismatch")
                continue
            for node in sorted(oracle_features):
                _compare_value(
                    comparisons,
                    path=(
                        f"variant.{variant_id}.{family}.feature."
                        f"{node.layer}.{node.position}.{node.feature}"
                    ),
                    oracle=oracle_features[node],
                    selective=selective_features[node],
                    max_bf16_ulps=tolerance.feature_max_bf16_ulps,
                )

    observed_nodes = set(request.execution.observed_downstream_nodes)
    for label, result, observations in (
        ("oracle", oracle, oracle_by_id),
        ("selective", selective, selective_by_id),
    ):
        no_op = observations.get(expected_ids[0])
        assert result.baseline is not None
        if no_op is None:
            reasons.append(f"{label}_no_op_baseline_missing")
            continue
        _compare_value(
            comparisons,
            path=f"{label}.no_op_baseline.target_value",
            oracle=result.baseline.target_value,
            selective=no_op.target_value,
            max_bf16_ulps=tolerance.objective_max_bf16_ulps,
        )
        for index, name in enumerate(("target_logit", "mean_logit")):
            _compare_value(
                comparisons,
                path=f"{label}.no_op_baseline.{name}",
                oracle=result.baseline.raw_target_logits[index],
                selective=no_op.raw_target_logits[index],
                max_bf16_ulps=tolerance.objective_max_bf16_ulps,
            )
        baseline_downstream = {
            node: value
            for node, value in _feature_map(result.baseline.feature_values).items()
            if node in observed_nodes
        }
        no_op_downstream = _feature_map(no_op.downstream_feature_values)
        if baseline_downstream.keys() != no_op_downstream.keys():
            reasons.append(f"{label}_no_op_baseline_feature_key_mismatch")
        else:
            for node in sorted(baseline_downstream):
                _compare_value(
                    comparisons,
                    path=(
                        f"{label}.no_op_baseline.downstream.feature."
                        f"{node.layer}.{node.position}.{node.feature}"
                    ),
                    oracle=baseline_downstream[node],
                    selective=no_op_downstream[node],
                    max_bf16_ulps=tolerance.feature_max_bf16_ulps,
                )

    baseline_features = _feature_map(oracle.baseline.feature_values)
    for semantics in (
        InterventionSemantics.DIRECT_FROZEN,
        InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
    ):
        candidates = tuple(
            variant
            for variant in request.execution.variants
            if variant.interventions and variant.semantics is semantics
        )
        effect_observed = False
        for variant in candidates:
            observation = oracle_by_id.get(variant.variant_id)
            if observation is None:
                continue
            source = variant.interventions[0].node
            if _has_meaningful_downstream_effect(
                baseline_features,
                observation,
                source=source,
                tolerance=tolerance,
            ):
                effect_observed = True
                break
        if not effect_observed:
            reasons.append(f"non_vacuous_effect_missing:{semantics.value}")
    failed = tuple(item.path for item in comparisons if not item.passed)
    reasons.extend(f"comparison_failed:{path}" for path in failed)
    verdict = (
        OrderingQualificationVerdict.QUALIFIED
        if not reasons
        else OrderingQualificationVerdict.REJECTED
    )
    return OrderingQualificationResult(
        verdict,
        tuple(comparisons),
        tuple(dict.fromkeys(reasons)),
        selective.status.value,
        oracle.status.value,
        selective.cleanup_completed,
        oracle.cleanup_completed,
        tuple(refusal_diagnostics),
    )


def _with_repeat_evidence(
    primary: OrderingQualificationResult,
    *,
    oracle_repeat: OrderingQualificationResult,
    selective_repeat: OrderingQualificationResult,
) -> OrderingQualificationResult:
    """Merge the A-B-B-A repeat comparisons without duplicating non-vacuity policy."""

    comparisons = list(primary.comparisons)
    reasons = list(primary.reasons)
    refusal_diagnostics = list(primary.refusal_diagnostics)
    verdict = primary.verdict
    for label, repeated in (
        ("oracle_repeat", oracle_repeat),
        ("selective_repeat", selective_repeat),
    ):
        comparisons.extend(
            OrderingComparison(
                f"{label}.{comparison.path}",
                comparison.oracle_value,
                comparison.selective_value,
                comparison.absolute_error,
                comparison.relative_error,
                comparison.passed,
            )
            for comparison in repeated.comparisons
        )
        repeat_reasons = tuple(
            reason
            for reason in repeated.reasons
            if not reason.startswith("non_vacuous_effect_missing:")
        )
        reasons.extend(f"{label}:{reason}" for reason in repeat_reasons)
        refusal_diagnostics.extend(
            replace(item, runtime=f"{label}.{item.runtime}")
            for item in repeated.refusal_diagnostics
        )
        if repeat_reasons:
            if repeated.verdict is OrderingQualificationVerdict.REFUSED:
                verdict = OrderingQualificationVerdict.REFUSED
            elif verdict is OrderingQualificationVerdict.QUALIFIED:
                verdict = OrderingQualificationVerdict.REJECTED
    return OrderingQualificationResult(
        verdict,
        tuple(comparisons),
        tuple(dict.fromkeys(reasons)),
        primary.selective_status,
        primary.oracle_status,
        primary.selective_cleanup_completed,
        primary.oracle_cleanup_completed,
        tuple(refusal_diagnostics),
    )


def qualify_intervened_forward_ordering(
    model: Any,
    request: OrderingQualificationRequest,
    *,
    selective_runtime: _EvaluationRuntime | None = None,
    oracle_runtime: _EvaluationRuntime | None = None,
) -> OrderingQualificationReceipt:
    """Compare the production selective runtime with the eager hook oracle.

    ``selective_runtime`` and ``oracle_runtime`` are injectable for mutation
    testing, while normal callers always use the production candidate-smoke
    adapter and independent eager implementation.
    """

    capabilities = get_transcoder_capabilities(_provider(model))
    raw = _raw_model(model)
    observed_scope = OrderingQualificationScope(
        model_family=_model_family(model, raw),
        transcoder_architecture=capabilities.architecture,
        decoder_output_topology=capabilities.decoder_output_topology,
        dtype=str(getattr(model, "dtype", "")).removeprefix("torch."),
    )
    if observed_scope != request.scope:
        raise ValueError("loaded model/provider scope does not match qualification request")
    oracle_adapter = oracle_runtime or Gemma3PLTEagerHookOracle(model)
    selective_adapter = selective_runtime or NNSightInterventionRuntime(
        model,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    )
    # A-B-B-A ordering makes within-process recovery and repeatability evidence
    # part of the qualification instead of relying only on fresh-process repeats.
    oracle_result = oracle_adapter.evaluate(request.execution)
    selective_result = selective_adapter.evaluate(request.execution)
    selective_repeat_result = selective_adapter.evaluate(request.execution)
    oracle_repeat_result = oracle_adapter.evaluate(request.execution)
    result = _with_repeat_evidence(
        _compare_execution_results(request, oracle_result, selective_result),
        oracle_repeat=_compare_execution_results(
            request, oracle_result, oracle_repeat_result
        ),
        selective_repeat=_compare_execution_results(
            request, selective_result, selective_repeat_result
        ),
    )
    request_evidence = _request_payload(request)
    request_fingerprint = _fingerprint(request_evidence)
    qualification_evidence = _qualification_evidence(
        scope=asdict(request.scope),
        result=asdict(result),
        request_evidence=request_evidence,
    )
    qualification_fingerprint = _fingerprint(qualification_evidence)
    bound_evidence = {
        **qualification_evidence,
        "request_fingerprint": request_fingerprint,
        "qualification_fingerprint": qualification_fingerprint,
        "provenance": request.provenance,
    }
    return OrderingQualificationReceipt(
        RECEIPT_SCHEMA,
        RECEIPT_SCHEMA_VERSION,
        request_fingerprint,
        request.scope,
        result,
        _fingerprint(bound_evidence),
        qualification_fingerprint,
        request.provenance,
        request.scope_bindings,
        request_evidence,
    )


qualify_nnsight_ordering = qualify_intervened_forward_ordering


__all__ = [
    "Gemma3PLTEagerHookOracle",
    "OrderingComparison",
    "OrderingQualificationReceipt",
    "OrderingRefusalDiagnostic",
    "OrderingQualificationRequest",
    "OrderingQualificationResult",
    "OrderingQualificationScope",
    "OrderingQualificationTolerance",
    "OrderingQualificationVerdict",
    "qualify_intervened_forward_ordering",
    "qualify_nnsight_ordering",
    "validate_ordering_qualification_receipt",
    "validate_serialized_ordering_qualification_receipt",
]
