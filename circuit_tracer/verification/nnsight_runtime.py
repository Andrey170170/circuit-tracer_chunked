from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

import torch

from circuit_tracer.transcoder.provider import get_transcoder_capabilities

from .contracts import (
    BaselineCapture,
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionExecutionResult,
    InterventionSemantics,
    InterventionVariant,
    OrderingAdmissionMode,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    VariantObservation,
)


@dataclass(frozen=True)
class NNSightInterventionPlan:
    node: FeatureNode
    absolute_value: float
    graph_baseline_value: float | None
    exact_graph_delta: float | None
    output_layers: tuple[int, ...]


@dataclass(frozen=True)
class NNSightVariantPlan:
    variant_id: str
    semantics: InterventionSemantics
    interventions: tuple[NNSightInterventionPlan, ...]
    observed_nodes: tuple[FeatureNode, ...]
    retain_intervention_nodes: tuple[FeatureNode, ...]
    freeze_attention: bool
    freeze_feature_outputs: bool
    freeze_layernorm_denominators: bool


@dataclass
class SelectiveProbeCapture:
    """Only scalar objective terms and explicitly requested feature values escape a trace."""

    target_logit: Any
    mean_logit: Any
    feature_values: tuple[tuple[FeatureNode, Any], ...]
    origin: "CaptureOrigin"
    retained_state: object | None = None


class CaptureOrigin(str, Enum):
    BASELINE_FORWARD = "baseline_forward"
    INTERVENED_FORWARD = "intervened_forward"


def _invoke_ordered_hook_families(
    tracer: Any,
    *,
    attention: Callable[[], None] | None,
    layernorm_groups: Sequence[Callable[[], None]],
    feature_input: Callable[[], None] | None,
    feature_output: Callable[[], None] | None,
) -> None:
    """Give each backward hook family/group its own ordered NNSight mediator."""

    if attention is not None:
        with tracer.invoke():  # type: ignore[attr-defined]
            attention()
    for layernorm_group in layernorm_groups:
        with tracer.invoke():  # type: ignore[attr-defined]
            layernorm_group()
    if feature_input is not None:
        with tracer.invoke():  # type: ignore[attr-defined]
            feature_input()
    if feature_output is not None:
        with tracer.invoke():  # type: ignore[attr-defined]
            feature_output()


class SelectiveNNSightProbeModel(Protocol):
    backend: str
    transcoders: object
    device: object
    cfg: object
    config: object
    verification_intervened_capture_ordering_qualified: bool

    def _verification_capture_baseline(
        self,
        prompt_token_ids: tuple[int, ...],
        retained_nodes: tuple[FeatureNode, ...],
        *,
        target_position: int,
        target_token_id: int,
        retain_attention_state: bool,
        retain_direct_freeze_state: bool,
    ) -> SelectiveProbeCapture: ...

    def _verification_run_variant(
        self,
        prompt_token_ids: tuple[int, ...],
        plan: NNSightVariantPlan,
        baseline_state: object | None,
        *,
        target_position: int,
        target_token_id: int,
    ) -> SelectiveProbeCapture: ...

    def _verification_release(self, baseline_state: object | None) -> None: ...

    def _verification_health_check(self, baseline_state: object | None) -> bool: ...


def _provider(model: SelectiveNNSightProbeModel) -> object:
    provider = model.transcoders
    return getattr(provider, "_module", provider)


def _translate_variant(
    variant: InterventionVariant,
    *,
    architecture: str,
    n_layers: int,
    observed_nodes: tuple[FeatureNode, ...],
) -> NNSightVariantPlan:
    if architecture not in ("clt", "plt"):
        raise ValueError(f"unsupported transcoder architecture: {architecture}")
    interventions: list[NNSightInterventionPlan] = []
    for intervention in variant.interventions:
        if intervention.node.layer >= n_layers:
            raise ValueError("intervention layer exceeds the loaded model")
        if variant.semantics is InterventionSemantics.DIRECT_FROZEN:
            if intervention.graph_delta is None or intervention.graph_baseline_value is None:
                raise ValueError("DIRECT_FROZEN requires graph baseline and exact delta")
            exact_delta = intervention.graph_delta
        else:
            exact_delta = None
        output_layers = (
            tuple(range(intervention.node.layer, n_layers))
            if architecture == "clt"
            else (intervention.node.layer,)
        )
        interventions.append(
            NNSightInterventionPlan(
                intervention.node,
                intervention.absolute_value,
                intervention.graph_baseline_value,
                exact_delta,
                output_layers,
            )
        )
    direct = variant.semantics is InterventionSemantics.DIRECT_FROZEN and bool(interventions)
    return NNSightVariantPlan(
        variant_id=variant.variant_id,
        semantics=variant.semantics,
        interventions=tuple(interventions),
        observed_nodes=observed_nodes,
        retain_intervention_nodes=tuple(sorted({item.node for item in interventions})),
        freeze_attention=bool(interventions),
        freeze_feature_outputs=direct,
        freeze_layernorm_denominators=direct,
    )


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(device="cpu", dtype=torch.float64).item())
    return float(value)  # type: ignore[arg-type]


def _feature_values(capture: SelectiveProbeCapture) -> tuple[FeatureValue, ...]:
    return tuple(FeatureValue(node, _float(value)) for node, value in capture.feature_values)


def _provider_activation_delta(
    provider: Any,
    layer: int,
    baseline_preactivation: Any,
    absolute_preactivation: float,
) -> Any:
    absolute = baseline_preactivation * 0 + absolute_preactivation
    activated = provider.apply_activation_function(
        layer, torch.stack((baseline_preactivation, absolute))
    )
    return activated[1] - activated[0]


def _skip_transcoder_correction(
    provider: Any,
    layer: int,
    frozen_feature_input: Any,
    current_feature_input: Any,
) -> Any:
    return provider.compute_skip(layer, current_feature_input) - provider.compute_skip(
        layer, frozen_feature_input
    )


def _default_synchronize(model: Any) -> None:
    device = torch.device(model.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class NNSightInterventionRuntime:
    """Production adapter with selective retention and cooperative deadline admission."""

    def __init__(
        self,
        model: Any,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[Any], None] = _default_synchronize,
        ordering_admission_mode: OrderingAdmissionMode | None = None,
    ) -> None:
        if ordering_admission_mode is not None and not isinstance(
            ordering_admission_mode, OrderingAdmissionMode
        ):
            raise ValueError("ordering_admission_mode must be candidate_smoke or qualified")
        self._model = model
        self._clock = clock
        self._synchronize = synchronize
        self._ordering_admission_mode = ordering_admission_mode

    def _unsupported(self, request: InterventionExecutionRequest) -> RuntimeRefusal | None:
        if getattr(self._model, "backend", None) != "nnsight":
            return RuntimeRefusal("unsupported_backend", "behavioral probes require NNSight")
        provider = _provider(self._model)
        capabilities = get_transcoder_capabilities(provider)
        expected_topology = "cross_layer" if capabilities.architecture == "clt" else "same_layer"
        if capabilities.architecture not in ("clt", "plt"):
            return RuntimeRefusal("unsupported_provider", "provider must declare CLT or PLT")
        if capabilities.decoder_output_topology != expected_topology:
            return RuntimeRefusal(
                "unsupported_provider",
                "provider architecture and decoder output topology disagree",
            )
        if not 1 <= request.identity.target_position <= len(request.identity.prompt_token_ids):
            return RuntimeRefusal(
                "unsupported_target_position",
                "target_position must be a 1-based prefix length; objective uses position - 1",
            )
        required = (
            "_verification_capture_baseline",
            "_verification_run_variant",
            "_verification_release",
            "_verification_health_check",
        )
        missing = [name for name in required if not callable(getattr(self._model, name, None))]
        if missing:
            return RuntimeRefusal(
                "unsupported_backend",
                f"NNSight model lacks selective verification hooks: {', '.join(missing)}",
            )
        requires_intervened_capture = bool(
            request.observed_downstream_nodes
            or any(variant.interventions for variant in request.variants)
        )
        if requires_intervened_capture and self._ordering_admission_mode is None:
            return RuntimeRefusal(
                "ordering_unqualified",
                "an explicit candidate_smoke or qualified ordering admission is required",
            )
        if (
            requires_intervened_capture
            and self._ordering_admission_mode is OrderingAdmissionMode.QUALIFIED
            and not bool(
                getattr(
                    self._model,
                    "verification_intervened_capture_ordering_qualified",
                    False,
                )
            )
        ):
            return RuntimeRefusal(
                "ordering_unqualified",
                "the NNSight model does not advertise qualified intervened-forward ordering",
            )
        return None

    def evaluate(self, request: InterventionExecutionRequest) -> InterventionExecutionResult:
        started = self._clock()
        refusal = self._unsupported(request)
        if refusal is not None:
            return InterventionExecutionResult(
                RuntimeExecutionStatus.REFUSED,
                None,
                (),
                refusal,
                0.0,
                True,
                ordering_admission_mode=self._ordering_admission_mode,
            )

        provider = _provider(self._model)
        capabilities = get_transcoder_capabilities(provider)
        try:
            n_layers = int(getattr(self._model.cfg, "n_layers"))
            d_transcoder = int(getattr(provider, "d_transcoder"))
            vocab_size = int(getattr(self._model.config, "vocab_size"))
        except (AttributeError, TypeError, ValueError) as error:
            return InterventionExecutionResult(
                RuntimeExecutionStatus.REFUSED,
                None,
                (),
                RuntimeRefusal("unsupported_backend", f"model dimensions unavailable: {error}"),
                self._clock() - started,
                True,
                ordering_admission_mode=self._ordering_admission_mode,
            )
        prompt_length = len(request.identity.prompt_token_ids)
        raw_zero_positions = getattr(self._model, "zero_positions", ())
        if isinstance(raw_zero_positions, slice):
            zero_positions = set(range(*raw_zero_positions.indices(prompt_length)))
        else:
            zero_positions = {int(position) for position in raw_zero_positions}
        all_nodes = {
            *request.observed_downstream_nodes,
            *(
                intervention.node
                for variant in request.variants
                for intervention in variant.interventions
            ),
        }
        for node in all_nodes:
            if node.layer >= n_layers or node.position >= prompt_length or node.feature >= d_transcoder:
                return InterventionExecutionResult(
                    RuntimeExecutionStatus.REFUSED,
                    None,
                    (),
                    RuntimeRefusal("invalid_bounds", f"feature node is outside loaded dimensions: {node}"),
                    self._clock() - started,
                    True,
                    ordering_admission_mode=self._ordering_admission_mode,
                )
            if node.position in zero_positions:
                return InterventionExecutionResult(
                    RuntimeExecutionStatus.REFUSED,
                    None,
                    (),
                    RuntimeRefusal(
                        "canonical_zero_position",
                        f"behavioral evidence is unavailable at canonical zero position: {node}",
                    ),
                    self._clock() - started,
                    True,
                    ordering_admission_mode=self._ordering_admission_mode,
                )
        if request.identity.target_token_id >= vocab_size:
            return InterventionExecutionResult(
                RuntimeExecutionStatus.REFUSED,
                None,
                (),
                RuntimeRefusal("invalid_bounds", "target token exceeds loaded vocabulary"),
                self._clock() - started,
                True,
                ordering_admission_mode=self._ordering_admission_mode,
            )
        try:
            plans = tuple(
                _translate_variant(
                    variant,
                    architecture=capabilities.architecture,
                    n_layers=n_layers,
                    observed_nodes=request.observed_downstream_nodes,
                )
                for variant in request.variants
            )
        except (TypeError, ValueError) as error:
            return InterventionExecutionResult(
                RuntimeExecutionStatus.REFUSED,
                None,
                (),
                RuntimeRefusal("unsupported_plan", str(error)),
                self._clock() - started,
                True,
                ordering_admission_mode=self._ordering_admission_mode,
            )

        retained_nodes = tuple(
            sorted(
                set(request.observed_downstream_nodes)
                | {
                    intervention.node
                    for plan in plans
                    for intervention in plan.interventions
                }
            )
        )
        need_direct_state = any(plan.freeze_feature_outputs for plan in plans)
        observations: list[VariantObservation] = []
        baseline: BaselineCapture | None = None
        baseline_state: object | None = None
        runtime_refusal: RuntimeRefusal | None = None
        cleanup_completed = False
        status = RuntimeExecutionStatus.REFUSED
        predicted_variant_seconds = request.predicted_variant_seconds

        if (
            request.predicted_baseline_seconds + request.cleanup_reserve_seconds
            > request.deadline_seconds
        ):
            return self._finish(
                request,
                started,
                status,
                baseline,
                observations,
                RuntimeRefusal(
                    "deadline_admission", "baseline predicted over cooperative budget"
                ),
                True,
            )

        try:
            self._synchronize(self._model)
            baseline_started = self._clock()
            captured = self._model._verification_capture_baseline(
                request.identity.prompt_token_ids,
                retained_nodes,
                target_position=request.identity.target_position,
                target_token_id=request.identity.target_token_id,
                retain_attention_state=any(plan.freeze_attention for plan in plans),
                retain_direct_freeze_state=need_direct_state,
            )
            baseline_state = captured.retained_state
            if captured.origin is not CaptureOrigin.BASELINE_FORWARD:
                raise RuntimeError("baseline helper returned non-baseline evidence")
            self._synchronize(self._model)
            baseline_seconds = self._clock() - baseline_started
            target_logit = _float(captured.target_logit)
            mean_logit = _float(captured.mean_logit)
            baseline = BaselineCapture(
                target_logit - mean_logit,
                (target_logit, mean_logit),
                _feature_values(captured),
            )
            if self._clock() - started + request.cleanup_reserve_seconds > request.deadline_seconds:
                runtime_refusal = RuntimeRefusal(
                    "deadline_in_flight_overrun",
                    "admitted baseline crossed the cooperative deadline",
                )
            else:
                predicted_variant_seconds = max(predicted_variant_seconds, baseline_seconds)
                for plan in plans:
                    elapsed = self._clock() - started
                    if (
                        elapsed + predicted_variant_seconds + request.cleanup_reserve_seconds
                        > request.deadline_seconds
                    ):
                        status = (
                            RuntimeExecutionStatus.PARTIAL
                            if observations
                            else RuntimeExecutionStatus.REFUSED
                        )
                        runtime_refusal = RuntimeRefusal(
                            "deadline_admission",
                            "variant predicted over remaining cooperative budget",
                            plan.variant_id,
                        )
                        break
                    self._synchronize(self._model)
                    variant_started = self._clock()
                    capture = self._model._verification_run_variant(
                        request.identity.prompt_token_ids,
                        plan,
                        baseline_state,
                        target_position=request.identity.target_position,
                        target_token_id=request.identity.target_token_id,
                    )
                    if capture.origin is not CaptureOrigin.INTERVENED_FORWARD:
                        raise RuntimeError(
                            "variant helper did not return intervened-forward evidence"
                        )
                    self._synchronize(self._model)
                    variant_seconds = self._clock() - variant_started
                    predicted_variant_seconds = max(predicted_variant_seconds, variant_seconds)
                    target_logit = _float(capture.target_logit)
                    mean_logit = _float(capture.mean_logit)
                    all_features = _feature_values(capture)
                    observed_set = set(request.observed_downstream_nodes)
                    intervention_set = set(plan.retain_intervention_nodes)
                    observations.append(
                        VariantObservation(
                            plan.variant_id,
                            target_logit - mean_logit,
                            (target_logit, mean_logit),
                            tuple(item for item in all_features if item.node in observed_set),
                            self._clock() - started,
                            tuple(item for item in all_features if item.node in intervention_set),
                        )
                    )
                    if (
                        self._clock() - started + request.cleanup_reserve_seconds
                        > request.deadline_seconds
                    ):
                        status = RuntimeExecutionStatus.PARTIAL
                        runtime_refusal = RuntimeRefusal(
                            "deadline_in_flight_overrun",
                            "admitted variant crossed the cooperative deadline",
                            plan.variant_id,
                        )
                        break
                else:
                    status = RuntimeExecutionStatus.COMPLETE
        except Exception as error:
            status = RuntimeExecutionStatus.PARTIAL if observations else RuntimeExecutionStatus.REFUSED
            runtime_refusal = RuntimeRefusal(
                "runtime_failure", f"{type(error).__name__}: {error}"
            )
        finally:
            try:
                self._model._verification_release(baseline_state)
                self._synchronize(self._model)
                cleanup_completed = bool(
                    self._model._verification_health_check(baseline_state)
                )
                if not cleanup_completed:
                    raise RuntimeError("selective verification health check failed")
            except Exception as error:
                cleanup_completed = False
                status = RuntimeExecutionStatus.PARTIAL if baseline is not None else RuntimeExecutionStatus.REFUSED
                runtime_refusal = RuntimeRefusal(
                    "cleanup_failure", f"{type(error).__name__}: {error}"
                )

        if status is not RuntimeExecutionStatus.COMPLETE and runtime_refusal is None:
            runtime_refusal = RuntimeRefusal("runtime_failure", "verification ended without completion")
        return self._finish(
            request,
            started,
            status,
            baseline,
            observations,
            runtime_refusal,
            cleanup_completed,
        )

    def _finish(
        self,
        request: InterventionExecutionRequest,
        started: float,
        status: RuntimeExecutionStatus,
        baseline: BaselineCapture | None,
        observations: list[VariantObservation],
        refusal: RuntimeRefusal | None,
        cleanup_completed: bool,
    ) -> InterventionExecutionResult:
        elapsed = self._clock() - started
        return InterventionExecutionResult(
            status,
            baseline,
            tuple(observations),
            refusal,
            elapsed,
            cleanup_completed,
            max(0.0, elapsed - request.deadline_seconds),
            self._ordering_admission_mode,
        )
