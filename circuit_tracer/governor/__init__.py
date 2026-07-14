from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .contracts import AdmissionReport
    from .contracts import CachePolicy
    from .contracts import DecoderTopology
    from .contracts import DemandClass
    from .contracts import DemandEstimate
    from .contracts import DemandLifetime
    from .contracts import DemandTier
    from .contracts import EncoderResidency
    from .contracts import FidelityMode
    from .contracts import PhysicalExecutionConfig
    from .contracts import PhysicalExecutionRequirements
    from .contracts import PlanStatus
    from .contracts import ProviderCapabilities
    from .contracts import ProviderCostMetadata
    from .contracts import ProviderDimensions
    from .contracts import ProviderIdentity
    from .contracts import ProviderProfile
    from .contracts import ResourceEnvelope
    from .contracts import RowStorePolicy
    from .contracts import StorageTier
    from .contracts import TracePlan
    from .contracts import TraceSemantics
    from .contracts import TRUSTED_VALIDATION_EVIDENCE_REGISTRY
    from .contracts import ValidationEvidence
    from .contracts import canonical_json
    from .contracts import dtype_byte_width
    from .contracts import execution_fingerprint
    from .contracts import fingerprint
    from .contracts import semantic_fingerprint
    from .host_budget import HostBudgetCandidate
    from .host_budget import HostBudgetDiscovery
    from .host_budget import discover_host_budget
    from .profiles import GRANITE_H200_CALIBRATIONS
    from .profiles import HISTORICAL_STRESS_FIXTURES
    from .profiles import HISTORICAL_STRESS_RECOMMENDATIONS
    from .profiles import RECORDED_PROVIDER_PROFILES
    from .profiles import ResourceCalibrationObservation
    from .profiles import StressArithmeticFixture
    from .profiles import StressRecommendation
    from .resolver import compute_work_units
    from .resolver import resolve_trace_plan


_CONTRACT_EXPORTS = {
    "AdmissionReport",
    "CachePolicy",
    "DecoderTopology",
    "DemandClass",
    "DemandEstimate",
    "DemandLifetime",
    "DemandTier",
    "EncoderResidency",
    "FidelityMode",
    "PhysicalExecutionConfig",
    "PhysicalExecutionRequirements",
    "PlanStatus",
    "ProviderCapabilities",
    "ProviderCostMetadata",
    "ProviderDimensions",
    "ProviderIdentity",
    "ProviderProfile",
    "ResourceEnvelope",
    "RowStorePolicy",
    "StorageTier",
    "TracePlan",
    "TraceSemantics",
    "TRUSTED_VALIDATION_EVIDENCE_REGISTRY",
    "ValidationEvidence",
    "canonical_json",
    "dtype_byte_width",
    "execution_fingerprint",
    "fingerprint",
    "semantic_fingerprint",
}
_HOST_EXPORTS = {"HostBudgetCandidate", "HostBudgetDiscovery", "discover_host_budget"}
_PROFILE_EXPORTS = {
    "GRANITE_H200_CALIBRATIONS",
    "HISTORICAL_STRESS_FIXTURES",
    "HISTORICAL_STRESS_RECOMMENDATIONS",
    "RECORDED_PROVIDER_PROFILES",
    "ResourceCalibrationObservation",
    "StressArithmeticFixture",
    "StressRecommendation",
}
_RESOLVER_EXPORTS = {"compute_work_units", "resolve_trace_plan"}

__all__ = [
    "AdmissionReport",
    "CachePolicy",
    "DecoderTopology",
    "DemandClass",
    "DemandEstimate",
    "DemandLifetime",
    "DemandTier",
    "EncoderResidency",
    "FidelityMode",
    "GRANITE_H200_CALIBRATIONS",
    "HISTORICAL_STRESS_FIXTURES",
    "HISTORICAL_STRESS_RECOMMENDATIONS",
    "HostBudgetCandidate",
    "HostBudgetDiscovery",
    "PhysicalExecutionConfig",
    "PhysicalExecutionRequirements",
    "PlanStatus",
    "ProviderCapabilities",
    "ProviderCostMetadata",
    "ProviderDimensions",
    "ProviderIdentity",
    "ProviderProfile",
    "RECORDED_PROVIDER_PROFILES",
    "ResourceCalibrationObservation",
    "ResourceEnvelope",
    "RowStorePolicy",
    "StressArithmeticFixture",
    "StressRecommendation",
    "StorageTier",
    "TracePlan",
    "TraceSemantics",
    "TRUSTED_VALIDATION_EVIDENCE_REGISTRY",
    "ValidationEvidence",
    "canonical_json",
    "compute_work_units",
    "discover_host_budget",
    "dtype_byte_width",
    "execution_fingerprint",
    "fingerprint",
    "resolve_trace_plan",
    "semantic_fingerprint",
]


def __getattr__(name: str):
    if name in _CONTRACT_EXPORTS:
        module_name = "circuit_tracer.governor.contracts"
    elif name in _HOST_EXPORTS:
        module_name = "circuit_tracer.governor.host_budget"
    elif name in _PROFILE_EXPORTS:
        module_name = "circuit_tracer.governor.profiles"
    elif name in _RESOLVER_EXPORTS:
        module_name = "circuit_tracer.governor.resolver"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)
