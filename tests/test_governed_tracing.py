from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_tracer.governor import (
    AdmissionMode,
    RECORDED_PROVIDER_PROFILES,
    ActiveUniverseObservation,
    FidelityMode,
    PlanStatus,
    ResourceEnvelope,
    PhysicalExecutionRequirements,
    RowStorePolicy,
)
from circuit_tracer.governor.host_budget import HostBudgetCandidate, HostBudgetDiscovery
from circuit_tracer.governor.runtime import LoadedStateObservation, ProviderUnitProbe
from circuit_tracer.tracing import (
    AttributionProblem,
    GovernorFidelityPolicy,
    TraceRequest,
    TraceSemantics,
    open_session,
    resolve_trace_request,
    trace_batch,
    trace_one,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities


GIB = 1024**3


class Provider:
    def __init__(self, profile) -> None:
        identity = profile.identity
        dimensions = profile.dimensions
        self.n_layers = dimensions.n_layers
        self.d_model = dimensions.d_model
        self.d_transcoder = dimensions.d_features
        if identity.architecture == "clt":
            self.d_transcoder = 10_080
        self.capabilities = TranscoderCapabilities(
            architecture=identity.architecture,
            checkpoint_format="fixture",
            supports_decoder_chunk_cache=True,
            decoder_output_topology=identity.decoder_topology.value,
        )

    def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
        return (max_bytes, fingerprint)

    def clear_decoder_block_cache(self, cache) -> None:
        del cache


class Model:
    backend = "nnsight"

    def __init__(self, profile) -> None:
        self.transcoders = Provider(profile)


def envelope(*, disk: int = 100 * GIB, vram: int = 141 * GIB) -> ResourceEnvelope:
    return ResourceEnvelope(
        total_vram_bytes=vram,
        host_budget_bytes=800 * GIB,
        file_cache_allowance_bytes=64 * GIB,
        local_disk_bytes=disk,
        scratch_disk_bytes=disk,
        walltime_seconds=10**30,
    )


def request(profile, *, batch: int, prompt_tokens: int = 16) -> TraceRequest:
    return TraceRequest(
        problem=AttributionProblem(model=Model(profile), prompt=list(range(prompt_tokens))),
        semantics=TraceSemantics(
            source_batch_size=batch,
            feature_batch_size=batch,
            logit_batch_size=batch,
            max_feature_nodes=100,
        ),
    )


def test_governor_fidelity_policy_validates_mode_matrix() -> None:
    assert GovernorFidelityPolicy().mode is FidelityMode.STRICT
    with pytest.raises(ValueError, match="strict.*no overrides"):
        GovernorFidelityPolicy(override_fields=("source_batch_size",))
    with pytest.raises(ValueError, match="research.*requires override fields"):
        GovernorFidelityPolicy(mode=FidelityMode.RESEARCH)
    with pytest.raises(ValueError, match="research.*no evidence"):
        GovernorFidelityPolicy(
            mode=FidelityMode.RESEARCH,
            override_fields=("source_batch_size",),
            evidence_name="gate",
            evidence_version="1",
        )
    with pytest.raises(ValueError, match="validated_relaxed.*requires"):
        GovernorFidelityPolicy(
            mode=FidelityMode.VALIDATED_RELAXED,
            override_fields=("source_batch_size",),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        GovernorFidelityPolicy(
            mode=FidelityMode.RESEARCH,
            override_fields=("source_batch_size", "decoder_reduction_tile"),
        )


def test_governor_fidelity_types_are_exported_from_public_roots() -> None:
    import circuit_tracer
    import circuit_tracer.tracing as tracing

    assert circuit_tracer.GovernorFidelityPolicy is GovernorFidelityPolicy
    assert circuit_tracer.FidelityMode is FidelityMode
    assert tracing.GovernorFidelityPolicy is GovernorFidelityPolicy
    assert tracing.FidelityMode is FidelityMode


def test_admission_mode_is_public_enforce_default_and_not_fingerprinted() -> None:
    import circuit_tracer
    import circuit_tracer.tracing as tracing

    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    enforced = request(profile, batch=1000)
    advisory = replace(enforced, governor_admission_mode=AdmissionMode.ADVISORY)

    enforced_plan = resolve_trace_request(
        enforced, resources=envelope(), provider_profile=profile
    )
    advisory_plan = resolve_trace_request(
        advisory, resources=envelope(), provider_profile=profile
    )

    assert enforced.governor_admission_mode is AdmissionMode.ENFORCE
    assert circuit_tracer.AdmissionMode is AdmissionMode
    assert tracing.AdmissionMode is AdmissionMode
    assert enforced_plan.requested_execution_fingerprint == (
        advisory_plan.requested_execution_fingerprint
    )
    assert advisory_plan.governor_admission_mode is AdmissionMode.ADVISORY


def test_default_governor_fidelity_remains_strict() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    governed = resolve_trace_request(
        request(profile, batch=1000),
        resources=envelope(),
        provider_profile=profile,
    )

    workload = governed.planning_workload
    assert workload.fidelity is FidelityMode.STRICT
    assert workload.evidence_name is None
    assert workload.evidence_version is None
    assert workload.semantic_overrides == ()
    assert workload.research_overrides == ()


def test_research_fidelity_forwards_exact_canonical_workload_values() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = replace(
        request(profile, batch=1000),
        governor_fidelity=GovernorFidelityPolicy(
            mode=FidelityMode.RESEARCH,
            override_fields=(
                "decoder_reduction_tile",
                "frontier_refresh_stride",
                "phase1_source_cap",
            ),
        ),
    )

    governed = resolve_trace_request(
        selected,
        resources=envelope(),
        provider_profile=profile,
    )

    workload = governed.planning_workload
    assert workload.fidelity is FidelityMode.RESEARCH
    assert workload.research_overrides == (
        ("decoder_reduction_tile", "4096"),
        ("frontier_refresh_stride", "4"),
        ("phase1_source_cap", "null"),
    )
    assert workload.semantic_overrides == ()


def test_governor_fidelity_rejects_unknown_or_bookkeeping_fields_before_resolution(
    monkeypatch,
) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = replace(
        request(profile, batch=1000),
        governor_fidelity=GovernorFidelityPolicy(
            mode=FidelityMode.RESEARCH,
            override_fields=("fidelity", "not_a_workload_field"),
        ),
    )
    monkeypatch.setattr(
        "circuit_tracer.tracing.governor_bridge.resolve_trace_plan",
        lambda *_args, **_kwargs: pytest.fail("governor resolver reached"),
    )

    with pytest.raises(ValueError, match="unknown or non-semantic.*fidelity"):
        resolve_trace_request(
            selected,
            resources=envelope(),
            provider_profile=profile,
        )


def test_validated_relaxed_fidelity_forwards_without_registry_mutation(monkeypatch) -> None:
    from circuit_tracer.governor import TRUSTED_VALIDATION_EVIDENCE_REGISTRY

    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = replace(
        request(profile, batch=1000),
        governor_fidelity=GovernorFidelityPolicy(
            mode=FidelityMode.VALIDATED_RELAXED,
            override_fields=("decoder_reduction_tile",),
            evidence_name="wave-a-gate",
            evidence_version="1",
        ),
    )
    captured = {}

    class WorkloadCaptured(RuntimeError):
        pass

    def capture(workload, *_args):
        captured["workload"] = workload
        raise WorkloadCaptured

    assert not TRUSTED_VALIDATION_EVIDENCE_REGISTRY
    monkeypatch.setattr(
        "circuit_tracer.tracing.governor_bridge.resolve_trace_plan",
        capture,
    )
    with pytest.raises(WorkloadCaptured):
        resolve_trace_request(
            selected,
            resources=envelope(),
            provider_profile=profile,
        )

    workload = captured["workload"]
    assert workload.fidelity is FidelityMode.VALIDATED_RELAXED
    assert workload.evidence_name == "wave-a-gate"
    assert workload.evidence_version == "1"
    assert workload.semantic_overrides == (("decoder_reduction_tile", "4096"),)
    assert workload.research_overrides == ()
    assert not TRUSTED_VALIDATION_EVIDENCE_REGISTRY


def test_governed_clt_compiles_roomy_optimized_plan() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)
    governed = resolve_trace_request(selected, resources=envelope(), provider_profile=profile)

    assert governed.semantics.source_batch_size == 1000
    assert governed.execution.session.capacity == governed.planning_trace_plan.physical.session_capacity
    assert governed.execution.session.source_microbatch_max_rows == governed.planning_trace_plan.physical.source_microbatch_size
    assert governed.execution.session.phase3_microbatch_max_rows == governed.planning_trace_plan.physical.logit_microbatch_size
    assert (
        governed.execution.session.phase4_execution_batch_max_rows
        == governed.planning_trace_plan.physical.feature_microbatch_size
    )
    assert governed.execution.session.phase1_trace_batch_policy == "legacy"
    assert governed.execution.session.decoder_cache.max_bytes == 8 * GIB
    assert governed.execution.decoder.fetch_chunk_size == 4096
    assert governed.execution.replay.feature_window == 4
    assert governed.execution.replay.error_vector_prefetch_lookahead == 2
    assert governed.execution.storage.exact_encoder_residency == "lazy"
    assert governed.execution.storage.placement.value == "local"
    assert governed.execution.storage.temp_root_policy == "env_node_local"
    assert governed.planning_requirements == PhysicalExecutionRequirements()
    assert governed.admission_report.candidate_count > 1
    assert governed.admission_report.admissible_candidate_count > 1
    assert "row_store_policy" in governed.admission_report.free_fields


@pytest.mark.parametrize(
    ("disk", "expected"),
    [(2_000_000, "full_file"), (900_000, "column_tiled_v1"), (1, "none_recompute")],
)
def test_constrained_envelopes_select_validated_storage_rungs(disk, expected) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    plan = resolve_trace_request(
        request(profile, batch=1000),
        resources=envelope(disk=disk),
        provider_profile=profile,
    )
    actual = (
        plan.execution.storage.retention
        if plan.execution.storage.retention == "none_recompute"
        else plan.execution.storage.full_retention_backend
    )
    assert actual == expected
    if expected == "column_tiled_v1":
        assert plan.execution.storage.feature_column_tile_size == 2048
    if expected == "none_recompute":
        assert plan.execution.storage.replay_tile_cache_bytes == 1 * GIB
    else:
        assert plan.execution.storage.replay_tile_cache_bytes == 0


def test_provider_mismatch_and_planning_refusal_fail_closed() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)
    selected.problem.model.transcoders.capabilities = replace(
        selected.problem.model.transcoders.capabilities, architecture="plt"
    )
    with pytest.raises(ValueError, match="provider profile mismatch"):
        resolve_trace_request(selected, resources=envelope(), provider_profile=profile)

    admitted = request(profile, batch=1000)
    refused = trace_batch([admitted], resources=envelope(vram=GIB), provider_profile=profile)[0]
    assert refused.status.value == "refused"
    assert refused.output is None
    assert refused.admission_report is not None
    names = [event["name"] for event in refused.telemetry_events]
    assert names[:2] == ["attribute.start", "planning.pre_execution_admission"]
    assert "planning.refusal" in names
    assert names[-1] == "attribute.refused"


def test_clt_profile_validates_aggregate_width_against_per_layer_provider() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)

    plan = resolve_trace_request(selected, resources=envelope(), provider_profile=profile)

    assert selected.problem.model.transcoders.d_transcoder == 10_080
    assert profile.dimensions.d_features == 262_144
    assert plan.planning_trace_plan is not None


def test_governed_storage_rejects_unmanaged_temp_root() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)
    selected = replace(
        selected,
        execution=replace(
            selected.execution,
            storage=replace(selected.execution.storage, temp_root="/tmp/unmanaged"),
        ),
    )

    with pytest.raises(ValueError, match="unmanaged temp_root"):
        resolve_trace_request(
            selected,
            resources=envelope(),
            provider_profile=profile,
        )


def test_batch_and_session_pin_and_propagate_governed_inputs(monkeypatch) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    resources = envelope()
    selected = request(profile, batch=128)
    unavailable = ProviderUnitProbe("cpu_test", False, reason="injected")
    monkeypatch.setattr(
        "circuit_tracer.governor.runtime.discover_host_budget",
        lambda requested: HostBudgetDiscovery(
            requested,
            "test_allocation",
            (HostBudgetCandidate("test_allocation", requested),),
        ),
    )
    monkeypatch.setattr(
        "circuit_tracer.tracing.runner.TorchLoadedStateSampler.sample",
        lambda self, provider: LoadedStateObservation(
            cuda_available=False,
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_total_bytes=None,
            host_rss_bytes=None,
            host_available_bytes=None,
            decoder_probe=unavailable,
            encoder_probe=unavailable,
        ),
    )

    def execute(
        problem, plan, *, observer, forward_overrides, execution_identity, governor_runtime
    ):
        del observer, forward_overrides, governor_runtime
        execution_identity.mark_requested_as_effective()
        return (problem.prompt[0], plan.execution.session.capacity)

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)
    results = trace_batch([selected], resources=resources, provider_profile=profile)
    assert results[0].output == (0, 128)

    session = open_session(selected, resources=resources, provider_profile=profile)
    assert session.resources is resources
    assert session.provider_profile is profile
    assert session.trace().output == (0, 128)
    session.close()


def test_active_universe_refusal_returns_terminal_refused_result(monkeypatch) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    resources = envelope()
    selected = request(profile, batch=128)
    unavailable = ProviderUnitProbe("cpu_test", False, reason="injected")
    monkeypatch.setattr(
        "circuit_tracer.governor.runtime.discover_host_budget",
        lambda requested: HostBudgetDiscovery(
            requested,
            "test_allocation",
            (HostBudgetCandidate("test_allocation", requested),),
        ),
    )
    monkeypatch.setattr(
        "circuit_tracer.tracing.runner.TorchLoadedStateSampler.sample",
        lambda self, provider: LoadedStateObservation(
            cuda_available=False,
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_total_bytes=None,
            host_rss_bytes=None,
            host_available_bytes=None,
            decoder_probe=unavailable,
            encoder_probe=unavailable,
        ),
    )
    from circuit_tracer.governor.resolver import resolve_trace_plan as pure_resolve

    calls = 0

    def resolve_with_late_refusal(*args, **kwargs):
        nonlocal calls
        calls += 1
        plan = pure_resolve(*args, **kwargs)
        if calls == 2:
            plan = replace(
                plan,
                admission=replace(
                    plan.admission,
                    admitted=False,
                    refusals=("late capacity refusal",),
                ),
                status=PlanStatus.ADVISORY_REFUSED,
            )
        return plan

    monkeypatch.setattr(
        "circuit_tracer.governor.runtime.resolve_trace_plan", resolve_with_late_refusal
    )

    def execute(
        problem, plan, *, observer, forward_overrides, execution_identity, governor_runtime
    ):
        del problem, plan, observer, forward_overrides, execution_identity
        nnz = governor_runtime.workload.estimated_active_features
        governor_runtime.active_universe_replan(
            ActiveUniverseObservation(
                total_nnz=nnz,
                shape=(1, 1, nnz),
                per_layer_counts=(nnz,),
                per_position_counts=(nnz,),
                membership_fingerprint="late-refusal",
                membership_sample=((0, 0, 0),),
            )
        )
        raise AssertionError("refused active plan must not execute")

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)

    result = trace_one(selected, resources=resources, provider_profile=profile)

    assert result.status.value == "refused"
    assert [event["name"] for event in result.telemetry_events].count(
        "planning.refusal"
    ) == 1
    assert result.output is None
    assert result.admission_report.refusals == ("late capacity refusal",)
    names = [event["name"] for event in result.telemetry_events]
    assert "planning.active_universe_replan" in names
    assert "planning.refusal" in names
    assert "planning.terminal_cleanup" in names
    assert names[-1] == "attribute.refused"
    active_revision = next(
        event
        for event in result.telemetry_events
        if event["name"] == "planning.active_universe_replan"
    )
    assert active_revision["attrs"]["execution_fingerprint"]
    assert result.telemetry_summary["requested_execution_fingerprint"] == (
        result.requested_execution_fingerprint
    )


def test_advisory_late_refusal_succeeds_with_latest_report(monkeypatch) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    resources = envelope()
    selected = replace(
        request(profile, batch=128),
        governor_admission_mode=AdmissionMode.ADVISORY,
    )
    unavailable = ProviderUnitProbe("cpu_test", False, reason="injected")
    monkeypatch.setattr(
        "circuit_tracer.governor.runtime.discover_host_budget",
        lambda requested: HostBudgetDiscovery(
            requested,
            "test_allocation",
            (HostBudgetCandidate("test_allocation", requested),),
        ),
    )
    monkeypatch.setattr(
        "circuit_tracer.tracing.runner.TorchLoadedStateSampler.sample",
        lambda self, provider: LoadedStateObservation(
            cuda_available=False,
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_total_bytes=None,
            host_rss_bytes=None,
            host_available_bytes=None,
            decoder_probe=unavailable,
            encoder_probe=unavailable,
        ),
    )
    from circuit_tracer.governor.resolver import resolve_trace_plan as pure_resolve

    calls = 0

    def refuse_phase4(*args, **kwargs):
        nonlocal calls
        calls += 1
        plan = pure_resolve(*args, **kwargs)
        if calls == 4:
            plan = replace(
                plan,
                admission=replace(
                    plan.admission,
                    admitted=False,
                    refusals=("phase4 advisory refusal",),
                ),
                status=PlanStatus.ADVISORY_REFUSED,
            )
        return plan

    monkeypatch.setattr(
        "circuit_tracer.governor.runtime.resolve_trace_plan", refuse_phase4
    )

    def execute(
        problem, plan, *, observer, forward_overrides, execution_identity, governor_runtime
    ):
        del problem, plan, observer, forward_overrides
        nnz = governor_runtime.workload.estimated_active_features
        governor_runtime.active_universe_replan(
            ActiveUniverseObservation(
                total_nnz=nnz,
                shape=(1, 1, nnz),
                per_layer_counts=(nnz,),
                per_position_counts=(nnz,),
                membership_fingerprint="advisory-late-refusal",
                membership_sample=((0, 0, 0),),
            )
        )
        governor_runtime.phase3_entry_replan()
        governor_runtime.phase4_entry_replan()
        execution_identity.mark_requested_as_effective()
        return "completed"

    monkeypatch.setattr("circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute)

    result = trace_one(selected, resources=resources, provider_profile=profile)

    assert result.status.value == "succeeded"
    assert result.output == "completed"
    assert result.admission_report.admitted is False
    assert result.admission_report.refusals == ("phase4 advisory refusal",)
    names = [event["name"] for event in result.telemetry_events]
    assert "planning.admission_bypassed" in names
    assert "attribute.refused" not in names
    assert names[-1] == "attribute.done"
    assert result.telemetry_summary["governor_admission_mode"] == "advisory"


@pytest.mark.parametrize("api", [resolve_trace_request, trace_batch, open_session])
def test_public_governed_inputs_must_be_paired(api) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    selected = request(profile, batch=128)
    value = [selected] if api is trace_batch else selected
    with pytest.raises(ValueError, match="must be supplied together"):
        api(value, resources=envelope())


def test_explicit_zero_and_full_requirements_override_profile_defaults() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = replace(
        request(profile, batch=1000),
        physical_requirements=PhysicalExecutionRequirements(
            decoder_cache_bytes=0,
            row_store_policy=RowStorePolicy.FULL,
        ),
    )

    plan = resolve_trace_request(selected, resources=envelope(), provider_profile=profile)

    assert plan.execution.session.decoder_cache.enabled is False
    assert plan.execution.storage.full_retention_backend == "full_file"
    assert plan.planning_requirements.decoder_cache_bytes == 0
    assert plan.planning_requirements.row_store_policy is RowStorePolicy.FULL
