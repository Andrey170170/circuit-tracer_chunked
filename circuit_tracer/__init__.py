from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.attribution.attribute import attribute_phase0_stats
    from circuit_tracer.attribution.sparsification import SparsificationConfig
    from circuit_tracer.governor import AdmissionReport
    from circuit_tracer.governor import CachePolicy
    from circuit_tracer.governor import DecoderTopology
    from circuit_tracer.governor import DemandClass
    from circuit_tracer.governor import DemandEstimate
    from circuit_tracer.governor import DemandLifetime
    from circuit_tracer.governor import DemandTier
    from circuit_tracer.governor import FidelityMode
    from circuit_tracer.governor import GRANITE_H200_CALIBRATIONS
    from circuit_tracer.governor import HISTORICAL_STRESS_FIXTURES
    from circuit_tracer.governor import HISTORICAL_STRESS_RECOMMENDATIONS
    from circuit_tracer.governor import HostBudgetCandidate
    from circuit_tracer.governor import HostBudgetDiscovery
    from circuit_tracer.governor import PhysicalExecutionConfig
    from circuit_tracer.governor import PlanStatus
    from circuit_tracer.governor import ProviderCapabilities
    from circuit_tracer.governor import ProviderCostMetadata
    from circuit_tracer.governor import ProviderDimensions
    from circuit_tracer.governor import ProviderIdentity
    from circuit_tracer.governor import ProviderProfile
    from circuit_tracer.governor import RECORDED_PROVIDER_PROFILES
    from circuit_tracer.governor import ResourceCalibrationObservation
    from circuit_tracer.governor import ResourceEnvelope
    from circuit_tracer.governor import StressArithmeticFixture
    from circuit_tracer.governor import StressRecommendation
    from circuit_tracer.governor import TracePlan
    from circuit_tracer.governor import TraceSemantics
    from circuit_tracer.governor import TRUSTED_VALIDATION_EVIDENCE_REGISTRY
    from circuit_tracer.governor import ValidationEvidence
    from circuit_tracer.governor import canonical_json
    from circuit_tracer.governor import compute_work_units
    from circuit_tracer.governor import discover_host_budget
    from circuit_tracer.governor import dtype_byte_width
    from circuit_tracer.governor import execution_fingerprint
    from circuit_tracer.governor import fingerprint
    from circuit_tracer.governor import resolve_trace_plan
    from circuit_tracer.governor import semantic_fingerprint
    from circuit_tracer.graph import Graph
    from circuit_tracer.replacement_model import ReplacementModel


__all__ = [
    "ReplacementModel",
    "Graph",
    "attribute",
    "attribute_phase0_stats",
    "SparsificationConfig",
    "AdmissionReport",
    "CachePolicy",
    "DecoderTopology",
    "DemandClass",
    "DemandEstimate",
    "DemandLifetime",
    "DemandTier",
    "FidelityMode",
    "GRANITE_H200_CALIBRATIONS",
    "HISTORICAL_STRESS_FIXTURES",
    "HISTORICAL_STRESS_RECOMMENDATIONS",
    "HostBudgetCandidate",
    "HostBudgetDiscovery",
    "PhysicalExecutionConfig",
    "PlanStatus",
    "ProviderCapabilities",
    "ProviderCostMetadata",
    "ProviderDimensions",
    "ProviderIdentity",
    "ProviderProfile",
    "RECORDED_PROVIDER_PROFILES",
    "ResourceCalibrationObservation",
    "ResourceEnvelope",
    "StressArithmeticFixture",
    "StressRecommendation",
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


_GOVERNOR_EXPORTS = set(__all__) - {
    "ReplacementModel",
    "Graph",
    "attribute",
    "attribute_phase0_stats",
    "SparsificationConfig",
}


def __getattr__(name):
    lazy_imports = {
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "attribute_phase0_stats": (
            "circuit_tracer.attribution.attribute",
            "attribute_phase0_stats",
        ),
        "SparsificationConfig": (
            "circuit_tracer.attribution.sparsification",
            "SparsificationConfig",
        ),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
    }
    if name in _GOVERNOR_EXPORTS:
        lazy_imports[name] = ("circuit_tracer.governor", name)
    if name not in lazy_imports:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = lazy_imports[name]
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
