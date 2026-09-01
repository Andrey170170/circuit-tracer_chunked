"""Focused, non-promoting diagnosis of propagated NNSight ordering drift.

The qualification gate deliberately answers only qualified/rejected.  This
module explains a rejection by comparing each engine's intervention against
its own baseline and by running one common graph-delta control.  Raw checkpoint
vectors are transient; durable receipts contain only bounded scalar evidence.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import torch

from .ordering_qualification import (
    OrderingQualificationRequest,
    _bf16_ulp,
    _execution_request_payload,
    _fingerprint,
    _json_value,
)


DIAGNOSTIC_SCHEMA = "nnsight_propagated_ordering_diagnostic"
DIAGNOSTIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PropagationDiagnosticRunCapture:
    """Transient tensors retained from one baseline or propagated forward."""

    feature_inputs: tuple[tuple[int, torch.Tensor], ...]
    source_preactivation: float | None = None
    source_output_pre: torch.Tensor | None = None
    decoder_contribution: torch.Tensor | None = None
    source_output_post: torch.Tensor | None = None


@dataclass(frozen=True)
class PropagationDiagnosticEngineCapture:
    """The three forwards required from one execution engine."""

    baseline: PropagationDiagnosticRunCapture
    native: PropagationDiagnosticRunCapture
    common: PropagationDiagnosticRunCapture
    cleanup_completed: bool = True


class _PropagationDiagnosticEngine(Protocol):
    def capture(
        self, request: OrderingQualificationRequest
    ) -> PropagationDiagnosticEngineCapture: ...


@dataclass(frozen=True)
class VectorDifference:
    max_abs_error: float
    rms_error: float
    max_bf16_ulp_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class VectorSummary:
    max_abs: float
    l2: float
    sha256_float32: str


@dataclass(frozen=True)
class PropagationLayerComparison:
    layer: int
    baseline_cross_engine: VectorDifference
    native_delta_cross_engine: VectorDifference
    common_delta_cross_engine: VectorDifference


@dataclass(frozen=True)
class PropagationSourceWriteDiagnostic:
    runtime: str
    native_source_preactivation: float
    common_source_preactivation: float
    native_preactivation_delta_to_target: float
    common_preactivation_delta_to_target: float
    native_decoder_contribution: VectorSummary
    common_decoder_contribution: VectorSummary
    native_injection_identity: VectorDifference
    common_injection_identity: VectorDifference


@dataclass(frozen=True)
class PropagationOrderingDiagnosticResult:
    status: str
    findings: tuple[str, ...]
    layer_comparisons: tuple[PropagationLayerComparison, ...]
    source_writes: tuple[PropagationSourceWriteDiagnostic, ...]
    first_baseline_difference_layer: int | None
    first_baseline_material_divergence_layer: int | None
    first_native_delta_difference_layer: int | None
    first_native_delta_material_divergence_layer: int | None
    first_common_delta_difference_layer: int | None
    first_common_delta_material_divergence_layer: int | None
    common_delta_aligned: bool
    source_absolute_target: float
    common_graph_baseline: float
    common_graph_delta: float
    common_decoder_contribution_cross_engine: VectorDifference
    oracle_cleanup_completed: bool
    selective_cleanup_completed: bool


@dataclass(frozen=True)
class PropagationOrderingDiagnosticReceipt:
    schema: str
    schema_version: int
    request_fingerprint: str
    scope: Mapping[str, Any]
    result: PropagationOrderingDiagnosticResult
    diagnostic_fingerprint: str
    evidence_fingerprint: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    request_evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _json_value(asdict(self))
        payload["status"] = self.result.status
        payload["diagnostic_only_no_runtime_promotion"] = True
        return payload


def _difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    max_bf16_ulps: float,
) -> VectorDifference:
    if left.shape != right.shape:
        raise ValueError(
            f"diagnostic checkpoint shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    delta = left.detach().float().cpu() - right.detach().float().cpu()
    if not delta.numel():
        return VectorDifference(0.0, 0.0, 0.0, True)
    left_flat = left.detach().float().cpu().flatten()
    right_flat = right.detach().float().cpu().flatten()
    max_ulp_error = max(
        (
            abs(float(left_value) - float(right_value))
            / max(_bf16_ulp(float(left_value)), _bf16_ulp(float(right_value)))
            for left_value, right_value in zip(left_flat, right_flat, strict=True)
        ),
        default=0.0,
    )
    return VectorDifference(
        float(delta.abs().max().item()),
        float(delta.square().mean().sqrt().item()),
        float(max_ulp_error),
        max_ulp_error <= max_bf16_ulps,
    )


def _feature_inputs(run: PropagationDiagnosticRunCapture) -> dict[int, torch.Tensor]:
    values = dict(run.feature_inputs)
    if len(values) != len(run.feature_inputs):
        raise ValueError("diagnostic feature-input layers must be unique")
    return values


def _summary(value: torch.Tensor) -> VectorSummary:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return VectorSummary(
        float(tensor.abs().max().item()) if tensor.numel() else 0.0,
        float(tensor.square().sum().sqrt().item()),
        "sha256:" + hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    )


def _source_identity(
    run: PropagationDiagnosticRunCapture, *, max_bf16_ulps: float
) -> VectorDifference:
    if (
        run.source_output_pre is None
        or run.decoder_contribution is None
        or run.source_output_post is None
    ):
        raise ValueError("propagated diagnostic run lacks source-write tensors")
    return _difference(
        run.source_output_post,
        run.source_output_pre + run.decoder_contribution,
        max_bf16_ulps=max_bf16_ulps,
    )


def _source_diagnostic(
    runtime: str,
    capture: PropagationDiagnosticEngineCapture,
    *,
    source_absolute_target: float,
    max_bf16_ulps: float,
) -> PropagationSourceWriteDiagnostic:
    native = capture.native.source_preactivation
    common = capture.common.source_preactivation
    if native is None or common is None or not math.isfinite(native) or not math.isfinite(common):
        raise ValueError("propagated diagnostic run lacks finite source preactivation")
    native_contribution = capture.native.decoder_contribution
    common_contribution = capture.common.decoder_contribution
    if native_contribution is None or common_contribution is None:
        raise ValueError("propagated diagnostic run lacks decoder contribution")
    return PropagationSourceWriteDiagnostic(
        runtime,
        float(native),
        float(common),
        source_absolute_target - float(native),
        source_absolute_target - float(common),
        _summary(native_contribution),
        _summary(common_contribution),
        _source_identity(capture.native, max_bf16_ulps=max_bf16_ulps),
        _source_identity(capture.common, max_bf16_ulps=max_bf16_ulps),
    )


def _compare_captures(
    request: OrderingQualificationRequest,
    oracle: PropagationDiagnosticEngineCapture,
    selective: PropagationDiagnosticEngineCapture,
) -> PropagationOrderingDiagnosticResult:
    propagated = request.execution.variants[2]
    source_intervention = propagated.interventions[0]
    graph_baseline = source_intervention.graph_baseline_value
    graph_delta = source_intervention.graph_delta
    if graph_baseline is None or graph_delta is None:
        raise ValueError("propagation diagnostic request lacks common graph delta")
    max_bf16_ulps = request.tolerance.feature_max_bf16_ulps
    baseline_oracle = _feature_inputs(oracle.baseline)
    native_oracle = _feature_inputs(oracle.native)
    common_oracle = _feature_inputs(oracle.common)
    baseline_selective = _feature_inputs(selective.baseline)
    native_selective = _feature_inputs(selective.native)
    common_selective = _feature_inputs(selective.common)
    layer_sets = {
        tuple(sorted(values))
        for values in (
            baseline_oracle,
            native_oracle,
            common_oracle,
            baseline_selective,
            native_selective,
            common_selective,
        )
    }
    if len(layer_sets) != 1:
        raise ValueError("diagnostic engines retained different feature-input layers")
    layers = next(iter(layer_sets))
    comparisons: list[PropagationLayerComparison] = []
    for layer in layers:
        oracle_native_delta = native_oracle[layer].float() - baseline_oracle[layer].float()
        selective_native_delta = (
            native_selective[layer].float() - baseline_selective[layer].float()
        )
        oracle_common_delta = common_oracle[layer].float() - baseline_oracle[layer].float()
        selective_common_delta = (
            common_selective[layer].float() - baseline_selective[layer].float()
        )
        comparisons.append(
            PropagationLayerComparison(
                layer,
                _difference(
                    baseline_oracle[layer],
                    baseline_selective[layer],
                    max_bf16_ulps=max_bf16_ulps,
                ),
                _difference(
                    oracle_native_delta,
                    selective_native_delta,
                    max_bf16_ulps=max_bf16_ulps,
                ),
                _difference(
                    oracle_common_delta,
                    selective_common_delta,
                    max_bf16_ulps=max_bf16_ulps,
                ),
            )
        )
    source_writes = (
        _source_diagnostic(
            "oracle",
            oracle,
            source_absolute_target=source_intervention.absolute_value,
            max_bf16_ulps=max_bf16_ulps,
        ),
        _source_diagnostic(
            "selective",
            selective,
            source_absolute_target=source_intervention.absolute_value,
            max_bf16_ulps=max_bf16_ulps,
        ),
    )
    assert oracle.common.decoder_contribution is not None
    assert selective.common.decoder_contribution is not None
    common_contribution_difference = _difference(
        oracle.common.decoder_contribution,
        selective.common.decoder_contribution,
        max_bf16_ulps=max_bf16_ulps,
    )

    def first(field: str, *, material: bool) -> int | None:
        return next(
            (
                comparison.layer
                for comparison in comparisons
                if (
                    not getattr(comparison, field).within_tolerance
                    if material
                    else getattr(comparison, field).max_abs_error > 0.0
                )
            ),
            None,
        )

    first_baseline = first("baseline_cross_engine", material=False)
    first_baseline_material = first("baseline_cross_engine", material=True)
    first_native = first("native_delta_cross_engine", material=False)
    first_native_material = first("native_delta_cross_engine", material=True)
    first_common = first("common_delta_cross_engine", material=False)
    first_common_material = first("common_delta_cross_engine", material=True)
    findings: list[str] = []
    for source in source_writes:
        if not source.native_injection_identity.within_tolerance:
            findings.append(f"{source.runtime}_native_injection_identity_mismatch")
        if not source.common_injection_identity.within_tolerance:
            findings.append(f"{source.runtime}_common_injection_identity_mismatch")
    if not common_contribution_difference.within_tolerance:
        findings.append("common_decoder_contribution_mismatch")
    if first_native_material is not None and first_common_material is None:
        findings.append("native_drift_removed_by_common_graph_delta")
    if not oracle.cleanup_completed:
        findings.append("oracle_cleanup_incomplete")
    if not selective.cleanup_completed:
        findings.append("selective_cleanup_incomplete")
    return PropagationOrderingDiagnosticResult(
        "complete",
        tuple(findings),
        tuple(comparisons),
        source_writes,
        first_baseline,
        first_baseline_material,
        first_native,
        first_native_material,
        first_common,
        first_common_material,
        first_common_material is None and common_contribution_difference.within_tolerance,
        source_intervention.absolute_value,
        graph_baseline,
        graph_delta,
        common_contribution_difference,
        oracle.cleanup_completed,
        selective.cleanup_completed,
    )


def _diagnostic_evidence(
    *,
    scope: Mapping[str, Any],
    result: Mapping[str, Any],
    request_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only_no_runtime_promotion": True,
        "scope": scope,
        "result": result,
        "request_fingerprint": _fingerprint(request_evidence),
    }


def diagnose_propagated_ordering(
    model: Any,
    request: OrderingQualificationRequest,
    *,
    oracle_engine: _PropagationDiagnosticEngine | None = None,
    selective_engine: _PropagationDiagnosticEngine | None = None,
) -> PropagationOrderingDiagnosticReceipt:
    """Explain propagated-ordering drift without making an admission decision."""

    if oracle_engine is None or selective_engine is None:
        from .ordering_diagnostic_engines import (
            EagerPropagationDiagnosticEngine,
            NNSightPropagationDiagnosticEngine,
        )

        oracle_engine = oracle_engine or EagerPropagationDiagnosticEngine(model)
        selective_engine = selective_engine or NNSightPropagationDiagnosticEngine(model)
    oracle_capture = oracle_engine.capture(request)
    selective_capture = selective_engine.capture(request)
    result = _compare_captures(request, oracle_capture, selective_capture)
    request_evidence = _execution_request_payload(request)
    request_fingerprint = _fingerprint(request_evidence)
    scope = asdict(request.scope)
    diagnostic_evidence = _diagnostic_evidence(
        scope=scope,
        result=asdict(result),
        request_evidence=request_evidence,
    )
    diagnostic_fingerprint = _fingerprint(diagnostic_evidence)
    bound = {
        **diagnostic_evidence,
        "diagnostic_fingerprint": diagnostic_fingerprint,
        "provenance": request.provenance,
    }
    return PropagationOrderingDiagnosticReceipt(
        DIAGNOSTIC_SCHEMA,
        DIAGNOSTIC_SCHEMA_VERSION,
        request_fingerprint,
        scope,
        result,
        diagnostic_fingerprint,
        _fingerprint(bound),
        request.provenance,
        request_evidence,
    )


def validate_serialized_propagated_ordering_diagnostic_receipt(
    payload: Mapping[str, Any],
) -> None:
    serialized = _json_value(dict(payload))
    if serialized.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError("propagated ordering diagnostic receipt schema mismatch")
    if serialized.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("propagated ordering diagnostic receipt schema version mismatch")
    if serialized.get("diagnostic_only_no_runtime_promotion") is not True:
        raise ValueError("propagated ordering diagnostic receipt promotion marker mismatch")
    result = serialized.get("result")
    request_evidence = serialized.get("request_evidence")
    scope = serialized.get("scope")
    if not isinstance(result, Mapping) or not isinstance(request_evidence, Mapping):
        raise ValueError("propagated ordering diagnostic receipt evidence is missing")
    if serialized.get("status") != result.get("status"):
        raise ValueError("propagated ordering diagnostic status mismatch")
    if serialized.get("request_fingerprint") != _fingerprint(request_evidence):
        raise ValueError("propagated ordering diagnostic request fingerprint mismatch")
    diagnostic_evidence = _diagnostic_evidence(
        scope=scope,
        result=result,
        request_evidence=request_evidence,
    )
    diagnostic_fingerprint = _fingerprint(diagnostic_evidence)
    if serialized.get("diagnostic_fingerprint") != diagnostic_fingerprint:
        raise ValueError("propagated ordering diagnostic fingerprint mismatch")
    bound = {
        **diagnostic_evidence,
        "diagnostic_fingerprint": diagnostic_fingerprint,
        "provenance": serialized.get("provenance", {}),
    }
    if serialized.get("evidence_fingerprint") != _fingerprint(bound):
        raise ValueError("propagated ordering diagnostic evidence fingerprint mismatch")


def validate_propagated_ordering_diagnostic_receipt(
    request: OrderingQualificationRequest,
    receipt: PropagationOrderingDiagnosticReceipt,
) -> None:
    if _json_value(receipt.request_evidence) != _json_value(
        _execution_request_payload(request)
    ):
        raise ValueError("propagated ordering diagnostic request evidence mismatch")
    validate_serialized_propagated_ordering_diagnostic_receipt(receipt.to_dict())


__all__ = [
    "PropagationDiagnosticEngineCapture",
    "PropagationDiagnosticRunCapture",
    "PropagationOrderingDiagnosticReceipt",
    "PropagationOrderingDiagnosticResult",
    "diagnose_propagated_ordering",
    "validate_propagated_ordering_diagnostic_receipt",
    "validate_serialized_propagated_ordering_diagnostic_receipt",
]
