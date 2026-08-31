"""Production-faithful eager and NNSight adapters for ordering diagnosis."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from circuit_tracer.transcoder.provider import get_transcoder_capabilities

from .nnsight_runtime import (
    NNSightVariantPlan,
    PropagationProbeCapture,
    PropagationProbeSpec,
    _translate_variant,
)
from .ordering_diagnostics import (
    PropagationDiagnosticEngineCapture,
    PropagationDiagnosticRunCapture,
)
from .ordering_qualification import Gemma3PLTEagerHookOracle, OrderingQualificationRequest


def _tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("propagation diagnostic expected a materialized tensor")
    return value.detach().to(device="cpu").clone()


def _run_capture(probe: PropagationProbeCapture | None) -> PropagationDiagnosticRunCapture:
    if probe is None:
        raise ValueError("propagation diagnostic engine returned no probe capture")
    return PropagationDiagnosticRunCapture(
        tuple((layer, _tensor(value)) for layer, value in probe.feature_inputs),
        (
            float(_tensor(probe.source_preactivation).float().item())
            if probe.source_preactivation is not None
            else None
        ),
        _tensor(probe.source_output_pre) if probe.source_output_pre is not None else None,
        (
            _tensor(probe.decoder_contribution)
            if probe.decoder_contribution is not None
            else None
        ),
        _tensor(probe.source_output_post) if probe.source_output_post is not None else None,
    )


def _plans(
    model: Any, request: OrderingQualificationRequest
) -> tuple[PropagationProbeSpec, NNSightVariantPlan, NNSightVariantPlan]:
    propagated = request.execution.variants[2]
    intervention = propagated.interventions[0]
    if intervention.graph_baseline_value is None or intervention.graph_delta is None:
        raise ValueError(
            "propagation diagnostic common control requires graph baseline and delta"
        )
    capabilities = get_transcoder_capabilities(
        getattr(model.transcoders, "_module", model.transcoders)
    )
    n_layers = int(model.cfg.n_layers)
    native = _translate_variant(
        propagated,
        architecture=capabilities.architecture,
        n_layers=n_layers,
        observed_nodes=request.execution.observed_downstream_nodes,
    )
    common = replace(
        native,
        variant_id=f"{native.variant_id}__common_graph_delta",
        interventions=tuple(
            replace(item, exact_graph_delta=intervention.graph_delta)
            for item in native.interventions
        ),
        retain_live_source_for_schedule=True,
    )
    maximum = max(node.layer for node in request.execution.observed_downstream_nodes)
    probe = PropagationProbeSpec(
        intervention.node.position,
        tuple(range(intervention.node.layer, maximum + 1)),
        intervention.node.layer,
    )
    return probe, native, common


class EagerPropagationDiagnosticEngine:
    def __init__(self, model: Any) -> None:
        self._oracle = Gemma3PLTEagerHookOracle(model)

    def capture(
        self, request: OrderingQualificationRequest
    ) -> PropagationDiagnosticEngineCapture:
        probe, native, common = _plans(self._oracle._model, request)
        state = None
        try:
            _, _, _, state, baseline_probe = self._oracle._run_forward(
                request.execution,
                plan=None,
                baseline_state=None,
                propagation_probe=probe,
            )
            if state is None:
                raise ValueError("eager propagation diagnostic did not retain baseline state")
            _, _, _, _, native_probe = self._oracle._run_forward(
                request.execution,
                plan=native,
                baseline_state=state,
                propagation_probe=probe,
            )
            _, _, _, _, common_probe = self._oracle._run_forward(
                request.execution,
                plan=common,
                baseline_state=state,
                propagation_probe=probe,
            )
            return PropagationDiagnosticEngineCapture(
                _run_capture(baseline_probe),
                _run_capture(native_probe),
                _run_capture(common_probe),
            )
        finally:
            if state is not None:
                state.clear()


class NNSightPropagationDiagnosticEngine:
    def __init__(self, model: Any) -> None:
        self._model = model

    def capture(
        self, request: OrderingQualificationRequest
    ) -> PropagationDiagnosticEngineCapture:
        probe, native, common = _plans(self._model, request)
        identity = request.execution.identity
        retained_nodes = tuple(
            sorted(
                set(request.execution.observed_downstream_nodes)
                | {item.node for item in native.interventions}
            )
        )
        baseline_state = None
        cleanup_completed = False
        try:
            baseline = self._model._verification_capture_baseline(
                identity.prompt_token_ids,
                retained_nodes,
                target_position=identity.target_position,
                target_token_id=identity.target_token_id,
                retain_attention_state=True,
                retain_direct_freeze_state=False,
                propagation_probe=probe,
            )
            baseline_state = baseline.retained_state
            native_capture = self._model._verification_run_variant(
                identity.prompt_token_ids,
                native,
                baseline_state,
                target_position=identity.target_position,
                target_token_id=identity.target_token_id,
                propagation_probe=probe,
            )
            common_capture = self._model._verification_run_variant(
                identity.prompt_token_ids,
                common,
                baseline_state,
                target_position=identity.target_position,
                target_token_id=identity.target_token_id,
                propagation_probe=probe,
            )
            result = PropagationDiagnosticEngineCapture(
                _run_capture(baseline.propagation_probe),
                _run_capture(native_capture.propagation_probe),
                _run_capture(common_capture.propagation_probe),
            )
        finally:
            self._model._verification_release(baseline_state)
            cleanup_completed = self._model._verification_health_check(baseline_state)
        return replace(result, cleanup_completed=cleanup_completed)


__all__ = [
    "EagerPropagationDiagnosticEngine",
    "NNSightPropagationDiagnosticEngine",
]
