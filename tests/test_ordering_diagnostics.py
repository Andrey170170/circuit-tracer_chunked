from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circuit_tracer.verification import (
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionSemantics,
    InterventionVariant,
    OrderingQualificationRequest,
    PreactivationIntervention,
    TargetState,
    TraceIdentity,
    VariantKind,
    diagnose_propagated_ordering,
    validate_propagated_ordering_diagnostic_receipt,
    validate_serialized_propagated_ordering_diagnostic_receipt,
)
from circuit_tracer.verification.ordering_diagnostics import (
    PropagationDiagnosticEngineCapture,
    PropagationDiagnosticRunCapture,
)
from circuit_tracer.verification.nnsight_runtime import (
    NNSightInterventionPlan,
    NNSightVariantPlan,
    _activation_schedule_for_plan,
)


def _request() -> OrderingQualificationRequest:
    source = FeatureNode(0, 1, 0)
    downstream = FeatureNode(1, 1, 1)
    predicted = (FeatureValue(downstream, 0.0),)
    execution = InterventionExecutionRequest(
        TraceIdentity(
            "propagation-diagnostic",
            "graph",
            "provider",
            "semantic",
            "execution",
            (1, 2, 3),
            3,
            4,
        ),
        TargetState(),
        (
            InterventionVariant(
                "no_op",
                VariantKind.NO_OP,
                InterventionSemantics.DIRECT_FROZEN,
                (),
                0.0,
            ),
            InterventionVariant(
                "direct",
                VariantKind.DIRECT_DOUBLE,
                InterventionSemantics.DIRECT_FROZEN,
                (PreactivationIntervention(source, 4.0, 2.0, 2.0),),
                None,
                predicted,
            ),
            InterventionVariant(
                "propagated",
                VariantKind.NECESSITY_HIGH,
                InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
                (PreactivationIntervention(source, 0.0, 2.0, -2.0),),
                None,
                predicted,
            ),
        ),
        (downstream,),
        120.0,
        1.0,
        1.0,
        1.0,
    )
    return OrderingQualificationRequest.from_execution_requests(
        execution_requests=(execution,),
        scope={"model_family": "gemma3", "provider_architecture": "plt"},
        provenance={"campaign": "test"},
    )


def _run(
    layer_1: tuple[float, float],
    *,
    source_preactivation: float | None = None,
) -> PropagationDiagnosticRunCapture:
    write = torch.tensor([1.0, 2.0]) if source_preactivation is not None else None
    return PropagationDiagnosticRunCapture(
        feature_inputs=(
            (0, torch.tensor([1.0, 2.0])),
            (1, torch.tensor(layer_1)),
        ),
        source_preactivation=source_preactivation,
        source_output_pre=torch.tensor([10.0, 20.0]) if write is not None else None,
        decoder_contribution=write,
        source_output_post=torch.tensor([11.0, 22.0]) if write is not None else None,
    )


class _StaticEngine:
    def __init__(self, capture: PropagationDiagnosticEngineCapture) -> None:
        self._capture = capture

    def capture(self, request: OrderingQualificationRequest) -> PropagationDiagnosticEngineCapture:
        assert request == _request()
        return self._capture


def test_common_graph_delta_preserves_native_activation_barrier_schedule() -> None:
    source = FeatureNode(0, 1, 0)
    native_intervention = NNSightInterventionPlan(source, 0.0, 2.0, None, (0,))
    common_intervention = replace(native_intervention, exact_graph_delta=-2.0)
    native = NNSightVariantPlan(
        "native",
        InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
        (native_intervention,),
        (),
        (source,),
        True,
        False,
        False,
    )
    common = replace(
        native,
        variant_id="common",
        interventions=(common_intervention,),
        retain_live_source_for_schedule=True,
    )

    native_schedule = _activation_schedule_for_plan(native)
    common_schedule = _activation_schedule_for_plan(common)
    assert native_schedule == common_schedule
    assert native_schedule.activation_nodes == (source,)
    assert native_schedule.prepass_invokes == 1
    assert native_schedule.intervention_invokes == 1
    assert native_schedule.barrier_participants == 2


def test_diagnostic_separates_native_source_form_drift_from_common_delta_control() -> None:
    eager = PropagationDiagnosticEngineCapture(
        baseline=_run((3.0, 4.0)),
        native=_run((2.0, 2.0), source_preactivation=2.0),
        common=_run((2.0, 2.0), source_preactivation=2.0),
    )
    selective = PropagationDiagnosticEngineCapture(
        baseline=_run((3.0, 4.0)),
        native=_run((2.5, 2.0), source_preactivation=2.5),
        common=_run((2.0, 2.0), source_preactivation=2.0),
    )

    receipt = diagnose_propagated_ordering(
        object(),
        _request(),
        oracle_engine=_StaticEngine(eager),
        selective_engine=_StaticEngine(selective),
    )

    assert receipt.result.status == "complete"
    assert receipt.result.first_native_delta_difference_layer == 1
    assert receipt.result.first_native_delta_material_divergence_layer == 1
    assert receipt.result.first_common_delta_difference_layer is None
    assert receipt.result.first_common_delta_material_divergence_layer is None
    assert receipt.result.common_delta_aligned is True
    assert receipt.result.source_absolute_target == 0.0
    assert receipt.result.common_graph_baseline == 2.0
    assert receipt.result.common_graph_delta == -2.0
    assert receipt.result.common_decoder_contribution_cross_engine.within_tolerance
    assert receipt.result.source_writes[0].native_preactivation_delta_to_target == -2.0
    assert receipt.result.source_writes[1].native_preactivation_delta_to_target == -2.5
    assert receipt.result.source_writes[0].common_decoder_contribution.max_abs == 2.0
    assert all(
        item.native_injection_identity.max_abs_error == 0.0
        for item in receipt.result.source_writes
    )
    assert all(
        item.common_injection_identity.max_abs_error == 0.0
        for item in receipt.result.source_writes
    )
    validate_propagated_ordering_diagnostic_receipt(_request(), receipt)
    validate_serialized_propagated_ordering_diagnostic_receipt(receipt.to_dict())


def test_diagnostic_receipt_rejects_source_write_error_and_serialized_tampering() -> None:
    baseline = _run((3.0, 4.0))
    good = _run((2.0, 2.0), source_preactivation=2.0)
    bad = replace(good, source_output_post=torch.tensor([11.0, 21.0]))

    receipt = diagnose_propagated_ordering(
        object(),
        _request(),
        oracle_engine=_StaticEngine(
            PropagationDiagnosticEngineCapture(baseline, good, good)
        ),
        selective_engine=_StaticEngine(
            PropagationDiagnosticEngineCapture(baseline, bad, good)
        ),
    )

    selective = next(
        item for item in receipt.result.source_writes if item.runtime == "selective"
    )
    assert selective.native_injection_identity.max_abs_error == 1.0
    assert "selective_native_injection_identity_mismatch" in receipt.result.findings

    payload = receipt.to_dict()
    payload["result"]["findings"].append("tampered")
    with pytest.raises(ValueError, match="diagnostic fingerprint"):
        validate_serialized_propagated_ordering_diagnostic_receipt(payload)
