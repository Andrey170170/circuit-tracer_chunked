from __future__ import annotations

import subprocess
import sys
from types import MappingProxyType

import pytest

from circuit_tracer.governor import GRANITE_H200_CALIBRATIONS
from circuit_tracer.governor import HISTORICAL_STRESS_FIXTURES
from circuit_tracer.governor import RECORDED_PROVIDER_PROFILES
from circuit_tracer.governor import ResourceEnvelope
from circuit_tracer.governor import TRUSTED_VALIDATION_EVIDENCE_REGISTRY
from circuit_tracer.governor import resolve_trace_plan
from circuit_tracer.governor.contracts import ValidationEvidence


GIB = 1024**3
KIB = 1024


def _amounts(plan) -> dict[str, float]:
    return {estimate.name: estimate.amount for estimate in plan.admission.estimates}


def test_recorded_granite_calibration_metadata_is_exact_and_resource_only():
    assert [item.batch_size for item in GRANITE_H200_CALIBRATIONS] == [1000, 128, 128, 64]
    assert [item.fetch_chunk_size for item in GRANITE_H200_CALIBRATIONS] == [4096] * 4
    assert [item.decoder_cache_bytes for item in GRANITE_H200_CALIBRATIONS] == [8 * GIB, 0, 0, 0]
    assert [item.requested_host_memory_bytes for item in GRANITE_H200_CALIBRATIONS] == [
        200 * GIB,
        200 * GIB,
        400 * GIB,
        600 * GIB,
    ]
    assert [item.walltime_seconds_range for item in GRANITE_H200_CALIBRATIONS] == [
        (83.25, 156.50),
        (2717.81, 2928.02),
        (4972.08, 5471.70),
        (20878.41, 23051.72),
    ]
    assert GRANITE_H200_CALIBRATIONS[0].max_rss_bytes_range == (
        4_736_848 * KIB,
        38_675_944 * KIB,
    )
    assert GRANITE_H200_CALIBRATIONS[-1].max_rss_bytes_range == (
        38_719_956 * KIB,
        244_429_132 * KIB,
    )
    assert {item.evidence_class for item in GRANITE_H200_CALIBRATIONS} == {
        "resource_calibration_only"
    }


def test_each_recorded_profile_admits_and_reproduces_calibrated_plan():
    fixed_vram: list[float] = []
    fixed_host: list[float] = []
    checkpoint_files: list[float] = []
    for observation in GRANITE_H200_CALIBRATIONS:
        profile = RECORDED_PROVIDER_PROFILES[observation.profile_name]
        semantics = observation.reference_semantics()
        envelope = ResourceEnvelope(
            total_vram_bytes=141 * GIB,
            host_budget_bytes=observation.requested_host_memory_bytes,
            file_cache_allowance_bytes=64 * GIB,
            local_disk_bytes=100 * GIB,
            scratch_disk_bytes=100 * GIB,
            walltime_seconds=observation.walltime_seconds_range[1],
        )
        plan = resolve_trace_plan(semantics, profile, envelope)
        amounts = _amounts(plan)
        assert plan.admission.admitted, plan.format()
        assert plan.physical.decoder_fetch_chunk_size == observation.fetch_chunk_size
        assert plan.physical.source_microbatch_size == observation.batch_size
        assert plan.physical.feature_microbatch_size == observation.batch_size
        assert plan.physical.logit_microbatch_size == observation.batch_size
        assert plan.physical.decoder_cache_bytes == observation.decoder_cache_bytes
        assert profile.dimensions.d_features == 262_144
        assert amounts["predicted_walltime_low"] == pytest.approx(
            observation.walltime_seconds_range[0]
        )
        assert amounts["predicted_walltime_high"] == pytest.approx(
            observation.walltime_seconds_range[1]
        )
        assert amounts["baseline_total_host"] == observation.max_rss_bytes_range[1]
        assert amounts["known_rigid_host"] == 0
        assert amounts["active_host"] == 0
        assert amounts["prompt_host"] == 0
        assert amounts["encoder_residency_host"] == 0
        assert amounts["replay_host"] == 0
        assert profile.capabilities.supports_full_row_store
        assert not profile.capabilities.supports_tiled_row_store
        assert not profile.capabilities.supports_recompute_row_store
        fixed_vram.append(amounts["model_vram"])
        fixed_host.append(amounts["baseline_total_host"])
        checkpoint_files.append(amounts["checkpoint_file_working_set"])

        positive_disk = [
            item
            for item in plan.admission.estimates
            if item.tier.value in {"local_disk", "scratch_disk"} and item.amount > 0
        ]
        assert len(positive_disk) == 1
        assert positive_disk[0].name == "row_store_disk"
        assert amounts["checkpoint_file_working_set"] != amounts["baseline_total_host"]
        assert plan.admission.effective_file_cache_allowance_bytes == 64 * GIB
        assert any("unknown recorded file-cache component" in warning for warning in plan.admission.warnings)

    assert len(set(fixed_vram)) == 4
    assert len(set(fixed_host)) == 4
    assert len(set(checkpoint_files)) == 4


def test_historical_stress_fixtures_resolve_to_recommendations():
    assert len(HISTORICAL_STRESS_FIXTURES) == 3
    for fixture in HISTORICAL_STRESS_FIXTURES:
        plan = resolve_trace_plan(fixture.semantics, fixture.profile, fixture.envelope)
        recommendation = fixture.recommendation
        assert plan.admission.admitted, plan.format()
        assert plan.admission.trace_capacity == recommendation.batch_size
        assert plan.physical.source_microbatch_size == recommendation.batch_size
        assert plan.physical.feature_microbatch_size == recommendation.batch_size
        assert plan.physical.logit_microbatch_size == recommendation.batch_size
        assert plan.physical.decoder_fetch_chunk_size == recommendation.fetch_chunk_size
        assert plan.physical.decoder_cache_bytes == recommendation.decoder_cache_bytes


def test_calibration_profiles_are_not_validation_evidence_registry():
    assert isinstance(TRUSTED_VALIDATION_EVIDENCE_REGISTRY, MappingProxyType)
    assert not TRUSTED_VALIDATION_EVIDENCE_REGISTRY
    assert not any(
        isinstance(value, ValidationEvidence)
        for value in RECORDED_PROVIDER_PROFILES.values()
    )


def test_recorded_host_baseline_does_not_double_reserve_included_file_cache():
    observation = GRANITE_H200_CALIBRATIONS[0]
    profile = RECORDED_PROVIDER_PROFILES[observation.profile_name]
    baseline = profile.costs.baseline_total_host_bytes
    assert baseline is not None
    envelope = ResourceEnvelope(
        total_vram_bytes=141 * GIB,
        host_budget_bytes=baseline + GIB,
        file_cache_allowance_bytes=64 * GIB,
        local_disk_bytes=100 * GIB,
        scratch_disk_bytes=100 * GIB,
        walltime_seconds=observation.walltime_seconds_range[1],
    )
    plan = resolve_trace_plan(observation.reference_semantics(), profile, envelope)
    assert plan.admission.admitted, plan.format()
    assert plan.admission.effective_file_cache_allowance_bytes == min(
        64 * GIB, envelope.host_budget_bytes
    )
    assert plan.admission.effective_file_cache_allowance_bytes > GIB


def test_root_and_governor_imports_are_lazy_and_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import circuit_tracer; import circuit_tracer.governor; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
