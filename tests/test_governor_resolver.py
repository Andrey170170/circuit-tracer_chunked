from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

import circuit_tracer.governor.resolver as resolver_module

from circuit_tracer.governor import DecoderTopology
from circuit_tracer.governor import FidelityMode
from circuit_tracer.governor import PhysicalExecutionRequirements
from circuit_tracer.governor import ProviderCapabilities
from circuit_tracer.governor import ProviderCostMetadata
from circuit_tracer.governor import ProviderDimensions
from circuit_tracer.governor import ProviderIdentity
from circuit_tracer.governor import ProviderProfile
from circuit_tracer.governor import ResourceEnvelope
from circuit_tracer.governor import RowStorePolicy
from circuit_tracer.governor import TraceSemantics
from circuit_tracer.governor import ValidationEvidence
from circuit_tracer.governor import dtype_byte_width
from circuit_tracer.governor import resolve_trace_plan


GIB = 1024**3


def _semantics(**changes) -> TraceSemantics:
    base = TraceSemantics(
        prompt_token_count=16,
        estimated_active_features=100,
        max_feature_nodes=9,
        target_count=2,
        scenario_id="synthetic-scenario",
        window_id="0:16",
        environment_label="synthetic-environment",
        source_batch_size=64,
        feature_batch_size=32,
        logit_batch_size=16,
        decoder_reduction_tile=512,
        frontier_refresh_stride=4,
        frontier_checkpoints=(4, 8),
        hooks=("hook-a",),
    )
    return replace(base, **changes)


def _profile(
    topology: DecoderTopology = DecoderTopology.CROSS_LAYER,
    **changes,
) -> ProviderProfile:
    approximation = "top_k" if topology is DecoderTopology.TOP_K else "exact"
    span = 4 if topology is DecoderTopology.CROSS_LAYER else 1
    semantic_parameters = (("top_k", "32"),) if topology is DecoderTopology.TOP_K else ()
    base = ProviderProfile(
        profile_name=f"synthetic-{topology.value}",
        profile_version="synthetic-v2",
        planner_version="governor-v0.2",
        identity=ProviderIdentity(
            provider_type="synthetic",
            provider_version="2",
            checkpoint_format="fixture",
            checkpoint_identity="fixture-weights",
            hook_identity="fixture-hook",
            architecture="fixture",
            decoder_topology=topology,
            approximation=approximation,
            semantic_parameters=semantic_parameters,
        ),
        dimensions=ProviderDimensions(4, 128, 4096, span),
        capabilities=ProviderCapabilities(
            supports_decoder_chunk_cache=True,
            supports_streaming_decoder=True,
            supports_encoder_row_materialization=True,
            supports_lazy_encoder_rows=True,
            supports_prefetch=True,
            supports_replay=True,
            supports_full_row_store=True,
            supports_tiled_row_store=True,
            supports_recompute_row_store=True,
        ),
        costs=ProviderCostMetadata(
            cost_model_version="synthetic-cost-v2",
            fixed_vram_bytes=GIB,
            trace_vram_coefficient=1.0,
            target_vram_coefficient=1.0,
            source_microbatch_vram_coefficient=1.0,
            feature_microbatch_vram_coefficient=1.0,
            logit_microbatch_vram_coefficient=1.0,
            replay_vram_coefficient=1.0,
            known_rigid_host_bytes=GIB,
            baseline_total_host_bytes=None,
            file_cache_included_in_host_baseline=False,
            reference_replay_window=1,
            reference_encoder_residency="eager",
            active_host_coefficient=2.0,
            prompt_host_coefficient=0.25,
            checkpoint_file_bytes=2 * GIB,
            calibrated_walltime_low_seconds=1.0,
            calibrated_walltime_high_seconds=2.0,
            walltime_reference_work_units=1.0,
        ),
        default_fetch_chunk_size=1024,
        max_fetch_chunk_size=4096,
        max_physical_microbatch=32,
        default_decoder_cache_bytes=GIB,
        max_decoder_cache_bytes=128 * GIB,
        default_replay_window=2,
        max_replay_window=8,
        default_prefetch_depth=1,
        max_prefetch_depth=4,
        row_store_tile_column_bound=2,
    )
    return replace(base, **changes)


def _envelope(**changes) -> ResourceEnvelope:
    base = ResourceEnvelope(
        total_vram_bytes=80 * GIB,
        host_budget_bytes=64 * GIB,
        file_cache_allowance_bytes=8 * GIB,
        local_disk_bytes=1024**2,
        scratch_disk_bytes=1024**2,
        walltime_seconds=10**30,
    )
    return replace(base, **changes)


def _estimates(plan) -> dict[str, float]:
    return {estimate.name: estimate.amount for estimate in plan.admission.estimates}


def _evidence(semantics: TraceSemantics, profile: ProviderProfile) -> ValidationEvidence:
    identity = profile.identity
    return ValidationEvidence(
        evidence_id=semantics.evidence_name or "missing",
        evidence_version=semantics.evidence_version or "missing",
        provider_type=identity.provider_type,
        provider_version=identity.provider_version,
        checkpoint_identity=identity.checkpoint_identity,
        hook_identity=identity.hook_identity,
        architecture=identity.architecture,
        decoder_topology=identity.decoder_topology,
        provider_approximation=identity.approximation,
        provider_semantic_parameters=identity.semantic_parameters,
        semantic_parameters=semantics.evidence_scope_parameters(),
        dtype=semantics.dtype,
        scenario_id=semantics.scenario_id,
        window_id=semantics.window_id,
        environment_label=semantics.environment_label,
        allowed_semantic_overrides=semantics.semantic_overrides,
        source_artifact_fingerprints=("artifact-sha256-a",),
        report_fingerprint="report-sha256-a",
        compared_configurations=(("baseline", "strict-a"), ("candidate", "relaxed-a")),
        metrics=(("weighted_edge_similarity", 0.99),),
        acceptance_thresholds=(("weighted_edge_similarity", 0.98),),
    )


def test_dtype_mapping_is_fail_closed():
    assert dtype_byte_width("bf16") == 2
    assert dtype_byte_width("fp32") == 4
    with pytest.raises(ValueError, match="unsupported dtype"):
        _semantics(dtype="float32")


def test_complete_semantic_inputs_and_physical_values_drive_demand_arithmetic():
    base_semantics = _semantics()
    profile = _profile()
    base = resolve_trace_plan(base_semantics, profile, _envelope())
    amounts = _estimates(base)

    assert _estimates(resolve_trace_plan(replace(base_semantics, prompt_token_count=32), profile, _envelope()))["trace_vram"] > amounts["trace_vram"]
    assert _estimates(resolve_trace_plan(replace(base_semantics, estimated_active_features=200), profile, _envelope()))["row_store_disk"] > amounts["row_store_disk"]
    assert _estimates(resolve_trace_plan(replace(base_semantics, max_feature_nodes=19), profile, _envelope()))["row_store_disk"] > amounts["row_store_disk"]
    assert _estimates(resolve_trace_plan(replace(base_semantics, target_count=4), profile, _envelope()))["target_vram"] > amounts["target_vram"]
    assert _estimates(resolve_trace_plan(replace(base_semantics, dtype="bf16"), profile, _envelope()))["trace_vram"] < amounts["trace_vram"]
    assert _estimates(resolve_trace_plan(replace(base_semantics, source_batch_size=128), profile, _envelope()))["trace_vram"] > amounts["trace_vram"]

    wider = replace(profile, dimensions=replace(profile.dimensions, d_model=256))
    wider_amounts = _estimates(resolve_trace_plan(base_semantics, wider, _envelope()))
    assert wider_amounts["target_vram"] > amounts["target_vram"]
    assert wider_amounts["trace_vram"] > amounts["trace_vram"]
    same_layer = replace(
        profile,
        identity=replace(profile.identity, decoder_topology=DecoderTopology.SAME_LAYER),
        dimensions=replace(profile.dimensions, decoder_output_span=1),
    )
    assert _estimates(resolve_trace_plan(base_semantics, same_layer, _envelope()))["trace_vram"] < amounts["trace_vram"]

    physical = resolve_trace_plan(
        base_semantics,
        profile,
        _envelope(),
        PhysicalExecutionRequirements(
            decoder_cache_bytes=2 * GIB,
            decoder_fetch_chunk_size=2048,
            prefetch_depth=2,
        ),
    )
    changed = _estimates(physical)
    assert changed["decoder_cache_vram"] == 2 * GIB
    assert changed["decoder_fetch_vram"] > amounts["decoder_fetch_vram"]
    assert changed["prefetch_vram"] > amounts["prefetch_vram"]
    assert changed["predicted_walltime_high"] != amounts["predicted_walltime_high"]


def test_source_cap_uses_effective_source_and_reports_all_bindings():
    plan = resolve_trace_plan(
        _semantics(source_batch_size=64, feature_batch_size=16, logit_batch_size=16, phase1_source_cap=16),
        _profile(),
        _envelope(),
    )
    assert plan.admission.trace_capacity == 16
    assert plan.admission.binding_reasons == ("source", "feature", "logit")
    assert plan.physical.source_microbatch_size <= 16

    still_bound = resolve_trace_plan(
        _semantics(source_batch_size=64, feature_batch_size=64, logit_batch_size=8, phase1_source_cap=16),
        _profile(),
        _envelope(),
    )
    assert still_bound.admission.trace_capacity == 64
    assert still_bound.admission.binding_reasons == ("feature",)
    assert any("does not lower trace capacity" in item for item in still_bound.admission.warnings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("source_microbatch_size", 65),
        ("feature_microbatch_size", 33),
        ("logit_microbatch_size", 17),
    ],
)
def test_each_physical_microbatch_is_independently_bounded(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_trace_plan(
            _semantics(),
            _profile(),
            _envelope(),
            PhysicalExecutionRequirements(**{name: value}),
        )


def test_decoder_cache_is_vram_and_huge_override_refuses():
    plan = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(total_vram_bytes=10 * GIB),
        PhysicalExecutionRequirements(decoder_cache_bytes=9 * GIB),
    )
    assert not plan.admission.admitted
    assert _estimates(plan)["decoder_cache_vram"] == 9 * GIB
    assert any("VRAM allocations require" in reason for reason in plan.admission.refusals)


def test_microbatch_and_replay_workspaces_change_vram_and_can_refuse():
    base = resolve_trace_plan(_semantics(), _profile(), _envelope())
    smaller_source = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(),
        PhysicalExecutionRequirements(source_microbatch_size=16),
    )
    larger_replay = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(),
        PhysicalExecutionRequirements(replay_window=4),
    )
    assert _estimates(smaller_source)["source_microbatch_vram"] < _estimates(base)[
        "source_microbatch_vram"
    ]
    assert _estimates(larger_replay)["replay_vram"] > _estimates(base)["replay_vram"]

    expensive = _profile(
        costs=replace(_profile().costs, replay_vram_coefficient=1_000_000.0)
    )
    refused = resolve_trace_plan(
        _semantics(),
        expensive,
        _envelope(total_vram_bytes=2 * GIB),
        PhysicalExecutionRequirements(replay_window=8),
    )
    assert not refused.admission.admitted
    assert any("VRAM allocations require" in reason for reason in refused.admission.refusals)


def test_unsupported_or_invalid_physical_requirements_fail_clearly():
    profile = _profile(
        capabilities=replace(_profile().capabilities, supports_replay=False, supports_prefetch=False),
        default_replay_window=1,
        max_replay_window=1,
        default_prefetch_depth=0,
        max_prefetch_depth=0,
    )
    with pytest.raises(ValueError, match="replay_window"):
        resolve_trace_plan(
            _semantics(),
            profile,
            _envelope(),
            PhysicalExecutionRequirements(replay_window=2),
        )
    with pytest.raises(ValueError, match="prefetch_depth"):
        resolve_trace_plan(
            _semantics(),
            profile,
            _envelope(),
            PhysicalExecutionRequirements(prefetch_depth=1),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        PhysicalExecutionRequirements(mystery=1)
    with pytest.raises(ValueError, match="encoder_residency"):
        resolve_trace_plan(
            _semantics(),
            _profile(),
            _envelope(),
            PhysicalExecutionRequirements(encoder_residency="unavailable"),
        )


def test_file_cache_allowance_is_clamped_under_total_host_budget():
    plan = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(host_budget_bytes=2 * GIB, file_cache_allowance_bytes=2 * GIB),
    )
    host_total = sum(
        item.amount for item in plan.admission.estimates if item.tier.value == "host"
    )
    assert plan.admission.effective_file_cache_allowance_bytes == max(
        0, 2 * GIB - int(host_total)
    )
    assert any("clamped file-cache allowance" in item for item in plan.admission.warnings)


def test_row_store_selects_full_then_tiled_then_recompute_by_capacity():
    semantics = _semantics()
    profile = _profile()
    full_bytes = (semantics.max_feature_nodes + 1) * semantics.estimated_nnz * 4
    tiled_bytes = (semantics.max_feature_nodes + 1) * 2 * 4

    full = resolve_trace_plan(semantics, profile, _envelope(local_disk_bytes=full_bytes))
    tiled = resolve_trace_plan(
        semantics,
        profile,
        _envelope(local_disk_bytes=tiled_bytes, scratch_disk_bytes=tiled_bytes),
    )
    recompute = resolve_trace_plan(semantics, profile, _envelope(local_disk_bytes=1, scratch_disk_bytes=1))
    assert (full.physical.row_store_policy, full.physical.row_store_bytes) == (
        "file_backed_full",
        full_bytes,
    )
    assert (tiled.physical.row_store_policy, tiled.physical.row_store_bytes) == (
        "tiled",
        tiled_bytes,
    )
    assert (recompute.physical.row_store_policy, recompute.physical.row_store_bytes) == (
        "recompute",
        0,
    )
    assert recompute.physical.spill_target is None


def test_row_store_overrides_validate_capability_and_capacity():
    full_only = _profile(
        capabilities=replace(
            _profile().capabilities,
            supports_tiled_row_store=False,
            supports_recompute_row_store=False,
        ),
        row_store_tile_column_bound=None,
    )
    with pytest.raises(ValueError, match="does not support"):
        resolve_trace_plan(
            _semantics(),
            full_only,
            _envelope(),
            PhysicalExecutionRequirements(row_store_policy=RowStorePolicy.TILED),
        )
    refused = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(local_disk_bytes=1, scratch_disk_bytes=1),
        PhysicalExecutionRequirements(row_store_policy=RowStorePolicy.FULL),
    )
    assert not refused.admission.admitted
    assert any("no configured spill tier fits" in reason for reason in refused.admission.refusals)


def test_validated_relaxed_uses_only_package_owned_trusted_registry(monkeypatch):
    semantics = _semantics(
        fidelity=FidelityMode.VALIDATED_RELAXED,
        decoder_reduction_tile=1024,
        evidence_name="evidence-a",
        evidence_version="1",
        semantic_overrides=(("decoder_reduction_tile", "1024"),),
    )
    profile = _profile()
    evidence = _evidence(semantics, profile)
    assert not resolver_module.TRUSTED_VALIDATION_EVIDENCE_REGISTRY
    with pytest.raises(ValueError, match="no trusted package evidence"):
        resolve_trace_plan(semantics, profile, _envelope())
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        getattr(resolver_module, "resolve_trace_plan")(
            semantics, profile, _envelope(), evidence=evidence
        )

    monkeypatch.setattr(
        resolver_module,
        "TRUSTED_VALIDATION_EVIDENCE_REGISTRY",
        MappingProxyType({("evidence-a", "1"): evidence}),
    )
    plan = resolve_trace_plan(semantics, profile, _envelope())
    assert plan.evidence_fingerprint == evidence.evidence_fingerprint
    monkeypatch.setattr(
        resolver_module,
        "TRUSTED_VALIDATION_EVIDENCE_REGISTRY",
        MappingProxyType(
            {("evidence-a", "1"): replace(evidence, environment_label="other")}
        ),
    )
    with pytest.raises(ValueError, match="scope mismatch.*environment_label"):
        resolve_trace_plan(semantics, profile, _envelope())
    monkeypatch.setattr(
        resolver_module,
        "TRUSTED_VALIDATION_EVIDENCE_REGISTRY",
        MappingProxyType(
            {
                ("evidence-a", "1"): replace(
                    evidence,
                    allowed_semantic_overrides=(("frontier_refresh_stride", "8"),),
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="scope mismatch.*allowed_semantic_overrides"):
        resolve_trace_plan(semantics, profile, _envelope())


def test_research_overrides_are_explicit_and_fingerprinted():
    research = _semantics(
        fidelity=FidelityMode.RESEARCH,
        frontier_refresh_stride=8,
        research_overrides=(("frontier_refresh_stride", "8"),),
    )
    assert (
        resolve_trace_plan(research, _profile(), _envelope()).semantic_fingerprint
        != resolve_trace_plan(_semantics(), _profile(), _envelope()).semantic_fingerprint
    )
    with pytest.raises(ValueError, match="canonical requested value"):
        _semantics(
            fidelity=FidelityMode.RESEARCH,
            frontier_refresh_stride=8,
            research_overrides=(("frontier_refresh_stride", "7"),),
        )


def test_provider_semantic_parameters_prevent_top_k_fingerprint_collision():
    semantics = _semantics(provider_approximation="top_k")
    top32 = _profile(DecoderTopology.TOP_K)
    top999 = replace(
        top32,
        identity=replace(top32.identity, semantic_parameters=(("top_k", "999"),)),
    )
    assert (
        resolve_trace_plan(semantics, top32, _envelope()).semantic_fingerprint
        != resolve_trace_plan(semantics, top999, _envelope()).semantic_fingerprint
    )


@pytest.mark.parametrize(
    "semantic_parameters",
    [(), (("top_k", "0"),), (("top_k", "-1"),), (("top_k", "abc"),)],
)
def test_top_k_identity_requires_positive_integer_parameter(semantic_parameters):
    with pytest.raises(ValueError, match="positive integer top_k"):
        replace(
            _profile(DecoderTopology.TOP_K).identity,
            semantic_parameters=semantic_parameters,
        )


def test_streaming_policy_requires_streaming_decoder_capability():
    profile = _profile(
        capabilities=replace(
            _profile().capabilities,
            supports_streaming_decoder=False,
        )
    )
    plan = resolve_trace_plan(
        _semantics(), profile, _envelope(file_cache_allowance_bytes=0)
    )
    assert plan.physical.cache_policy.value == "streaming"
    assert not plan.admission.admitted
    assert any("supports_streaming_decoder" in reason for reason in plan.admission.refusals)


def test_profile_fingerprint_is_part_of_execution_fingerprint():
    profile = _profile()
    changed = replace(profile, costs=replace(profile.costs, fixed_vram_bytes=2 * GIB))
    first = resolve_trace_plan(_semantics(), profile, _envelope())
    second = resolve_trace_plan(_semantics(), changed, _envelope())
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert profile.profile_fingerprint != changed.profile_fingerprint
    assert first.execution_fingerprint != second.execution_fingerprint


@pytest.mark.parametrize(
    ("topology", "approximation"),
    [
        (DecoderTopology.CROSS_LAYER, "exact"),
        (DecoderTopology.SAME_LAYER, "exact"),
        (DecoderTopology.TOP_K, "top_k"),
    ],
)
def test_generic_topologies_admit_without_name_branches(topology, approximation):
    plan = resolve_trace_plan(
        _semantics(provider_approximation=approximation),
        _profile(topology),
        _envelope(),
    )
    assert plan.admission.admitted
    assert plan.profile.identity.decoder_topology is topology


def test_incomplete_cost_metadata_fails_at_construction():
    with pytest.raises(ValueError, match="cost_model_version"):
        replace(_profile().costs, cost_model_version="")


def test_report_is_deterministic_and_actionable():
    plan = resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(total_vram_bytes=GIB, host_budget_bytes=1, walltime_seconds=1),
    )
    report = plan.format()
    assert not plan.admission.admitted
    assert report == resolve_trace_plan(
        _semantics(),
        _profile(),
        _envelope(total_vram_bytes=GIB, host_budget_bytes=1, walltime_seconds=1),
    ).format()
    assert "VRAM allocations require" in report
    assert "rigid host allocations require" in report
    assert "predicted walltime upper bound" in report
    assert "trace_capacity:" in report and "decisions:" in report
