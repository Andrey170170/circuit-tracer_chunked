from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circuit_tracer.governor.contracts import (
    PhysicalExecutionRequirements,
    ResourceEnvelope,
    TraceSemantics,
)
from circuit_tracer.governor.ledger import PhaseId
from circuit_tracer.governor.host_budget import HostBudgetCandidate, HostBudgetDiscovery
from circuit_tracer.governor.profiles import RECORDED_PROVIDER_PROFILES
from circuit_tracer.governor.resolver import resolve_trace_plan
from circuit_tracer.governor.runtime import (
    ActiveUniverseObservation,
    FrozenMechanismRevisionError,
    LoadedStateObservation,
    PlanningEpoch,
    PlanningEpochOrderError,
    ProviderUnitProbe,
    ResourceUsageExceededError,
    ResourceUsageObservation,
    TorchLoadedStateSampler,
    TraceGovernorRuntime,
    _EXCLUDED_ESTIMATES,
    _claims_for_phase,
)


GIB = 1024**3


class Observer:
    def __init__(self) -> None:
        self.events = []

    def observe(self, event):
        self.events.append(event)


def workload(*, nnz: int = 1000) -> TraceSemantics:
    return TraceSemantics(
        prompt_token_count=16,
        estimated_active_features=nnz,
        max_feature_nodes=100,
        target_count=10,
        scenario_id="runtime-test",
        environment_label="cpu-test",
        source_batch_size=64,
        feature_batch_size=64,
        logit_batch_size=64,
    )


def envelope(*, disk: int = 100 * GIB) -> ResourceEnvelope:
    return ResourceEnvelope(
        total_vram_bytes=141 * GIB,
        host_budget_bytes=800 * GIB,
        file_cache_allowance_bytes=64 * GIB,
        local_disk_bytes=disk,
        scratch_disk_bytes=disk,
        walltime_seconds=10**30,
    )


def runtime(*, nnz: int = 1000, disk: int = 100 * GIB):
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    inputs = workload(nnz=nnz)
    resources = envelope(disk=disk)
    requirements = PhysicalExecutionRequirements()
    plan = resolve_trace_plan(inputs, profile, resources, requirements)
    observer = Observer()
    return TraceGovernorRuntime(
        plan=plan,
        workload=inputs,
        profile=profile,
        envelope=resources,
        requirements=requirements,
        observer=observer,
        host_budget_discoverer=lambda requested: HostBudgetDiscovery(
            requested,
            "test_allocation",
            (HostBudgetCandidate("test_allocation", requested),),
        ),
    ), observer


def unavailable_loaded_state(*, allocated: int | None = None) -> LoadedStateObservation:
    unavailable = ProviderUnitProbe("unavailable", False, reason="cpu test")
    return LoadedStateObservation(
        cuda_available=False,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=allocated,
        cuda_total_bytes=None,
        host_rss_bytes=None,
        host_available_bytes=None,
        decoder_probe=unavailable,
        encoder_probe=unavailable,
    )


def active_observation(nnz: int) -> ActiveUniverseObservation:
    members = tuple((0, index, index) for index in range(nnz))
    return ActiveUniverseObservation(
        total_nnz=nnz,
        shape=(1, nnz, nnz),
        per_layer_counts=(nnz,),
        per_position_counts=tuple(1 for _ in range(nnz)),
        membership_fingerprint="membership",
        membership_sample=members[:16],
    )


def advance_to_active(governed: TraceGovernorRuntime) -> None:
    governed.pre_execution_admission()
    governed.loaded_state_calibration(unavailable_loaded_state())


def test_all_planning_epochs_are_ordered_and_semantics_stay_stable() -> None:
    governed, _ = runtime(nnz=1000)
    semantic = governed.current_plan.semantic_fingerprint

    governed.pre_execution_admission()
    governed.loaded_state_calibration(unavailable_loaded_state(allocated=1234))
    governed.active_universe_replan(active_observation(100))

    assert [revision.epoch for revision in governed.revisions] == [
        PlanningEpoch.PRE_EXECUTION_ADMISSION,
        PlanningEpoch.LOADED_STATE_CALIBRATION,
        PlanningEpoch.ACTIVE_UNIVERSE_REPLAN,
    ]
    assert all(revision.semantic_fingerprint == semantic for revision in governed.revisions)
    model_vram = next(
        estimate.amount
        for estimate in governed.revisions[1].plan.admission.estimates
        if estimate.name == "model_vram"
    )
    assert model_vram == 1234
    epoch_events = [
        event
        for event in governed.observer.events
        if event.name.startswith("planning.")
        and event.name != "planning.observation"
        and event.name != "planning.host_budget_discovery"
        and event.name != "planning.estimate_excluded"
        and event.name != "planning.ledger_grant"
        and event.name != "planning.ledger_warning"
    ]
    admission_event = next(
        event for event in epoch_events if event.name == "planning.pre_execution_admission"
    )
    for key in (
        "row_store_policy",
        "row_store_bytes",
        "spill_target",
        "source_microbatch_size",
        "feature_microbatch_size",
        "logit_microbatch_size",
        "decoder_cache_bytes",
        "decoder_fetch_chunk_size",
        "replay_window",
        "prefetch_depth",
        "encoder_residency",
        "cache_policy",
        "admission_decisions",
    ):
        assert key in admission_event.attrs
    observation_events = [
        event for event in governed.observer.events if event.name == "planning.observation"
    ]
    assert observation_events[-1].attrs["total_nnz"] == 100
    assert all(
        not isinstance(value, (tuple, list, dict))
        for event in observation_events
        for value in event.attrs.values()
    )
    assert any(
        event.name == "planning.active_universe_layer"
        and event.attrs == {"layer_index": 0, "active_count": 100}
        for event in governed.observer.events
    )


def test_actual_nnz_revises_storage_rung_before_claim() -> None:
    governed, _ = runtime(nnz=1000, disk=100_000)
    assert governed.current_plan.physical.row_store_policy == "recompute"
    advance_to_active(governed)

    revision = governed.active_universe_replan(active_observation(100))

    assert revision.plan.physical.row_store_policy == "file_backed_full"
    assert revision.plan.physical.row_store_bytes == 40_400
    grant = governed.grant(PhaseId.PHASE2)
    assert grant is not None
    assert next(claim.amount for claim in grant.claims if claim.name == "row_store_disk") == 40_400
    governed.release(grant)


def test_late_revision_rejects_frozen_mechanism(monkeypatch) -> None:
    governed, observer = runtime()
    advance_to_active(governed)
    original = governed.current_plan

    def changed(*args, **kwargs):
        del args, kwargs
        return replace(
            original,
            physical=replace(
                original.physical,
                decoder_fetch_chunk_size=original.physical.decoder_fetch_chunk_size // 2,
            ),
        )

    monkeypatch.setattr("circuit_tracer.governor.runtime.resolve_trace_plan", changed)
    with pytest.raises(FrozenMechanismRevisionError, match="decoder_fetch_chunk_size"):
        governed.active_universe_replan(active_observation(100))
    assert observer.events[-1].name == "planning.refusal"


@pytest.mark.parametrize("raises", [False, True])
def test_grants_release_on_success_and_failure(raises: bool) -> None:
    governed, observer = runtime()
    governed.pre_execution_admission()
    governed.loaded_state_calibration(unavailable_loaded_state())
    grant = governed.grant(PhaseId.PHASE3)
    try:
        if raises:
            raise RuntimeError("phase failed")
    except RuntimeError:
        pass
    finally:
        governed.release(grant)
        governed.close()

    assert governed.ledger.actual.active_grant_ids == ()
    names = [event.name for event in observer.events]
    assert "planning.ledger_grant" in names
    assert "planning.ledger_released" in names
    assert names[-1] == "planning.terminal_cleanup"


def test_active_universe_observation_is_deterministic() -> None:
    indices = torch.tensor([[0, 0, 1], [0, 1, 1], [2, 3, 4]])
    values = torch.ones(3)
    matrix = torch.sparse_coo_tensor(indices, values, (2, 2, 5))

    first = ActiveUniverseObservation.from_sparse_tensor(matrix)
    second = ActiveUniverseObservation.from_sparse_tensor(matrix)

    assert first == second
    assert first.total_nnz == 3
    assert first.per_layer_counts == (2, 1)
    assert first.per_position_counts == (1, 2)


def test_active_universe_observation_bounds_python_membership_sample() -> None:
    nnz = 50_000
    indices = torch.stack(
        (
            torch.zeros(nnz, dtype=torch.long),
            torch.arange(nnz, dtype=torch.long) % 128,
            torch.arange(nnz, dtype=torch.long),
        )
    )
    matrix = torch.sparse_coo_tensor(indices, torch.ones(nnz), (1, 128, nnz))

    observation = ActiveUniverseObservation.from_sparse_tensor(matrix, sample_size=8)

    assert observation.total_nnz == nnz
    assert len(observation.membership_sample) == 8
    assert sum(observation.per_position_counts) == nnz


def test_loaded_state_clamps_host_budget_and_emits_discovery() -> None:
    governed, observer = runtime()
    governed.pre_execution_admission()
    discovered = 12 * GIB
    governed._host_budget_discoverer = lambda requested: HostBudgetDiscovery(
        discovered,
        "cgroup_v2:test",
        (
            HostBudgetCandidate("explicit_override", requested),
            HostBudgetCandidate("cgroup_v2:test", discovered),
        ),
        ("test warning",),
    )
    observation = replace(
        unavailable_loaded_state(),
        host_rss_bytes=2 * GIB,
        host_available_bytes=20 * GIB,
    )

    governed.loaded_state_calibration(observation)

    assert governed.envelope.host_budget_bytes == discovered
    assert governed.ledger.envelope.host_budget_bytes == discovered
    event = next(
        event for event in observer.events if event.name == "planning.host_budget_discovery"
    )
    assert event.attrs["source"] == "cgroup_v2:test"
    assert event.attrs["warnings"] == ("test warning",)


def test_every_demand_estimate_has_one_phase_owner_or_explicit_exclusion() -> None:
    governed, _ = runtime()
    estimates = governed.current_plan.admission.estimates
    owners: dict[str, list[PhaseId]] = {estimate.name: [] for estimate in estimates}
    for phase in PhaseId:
        for claim in _claims_for_phase(estimates, phase):
            owners[claim.name].append(phase)

    for estimate in estimates:
        if estimate.name in _EXCLUDED_ESTIMATES:
            assert owners[estimate.name] == []
        else:
            assert len(owners[estimate.name]) == 1, (estimate.name, owners[estimate.name])


def test_encoder_unit_probe_uses_tensor_indices() -> None:
    class Capabilities:
        supports_lazy_decoder_chunks = False
        supports_encoder_row_materialization = True

    class Provider:
        capabilities = Capabilities()

        def materialize_encoder_rows(self, source_layers, feature_ids):
            assert isinstance(source_layers, torch.Tensor)
            assert isinstance(feature_ids, torch.Tensor)
            return torch.ones((1, 4))

    observation = TorchLoadedStateSampler().sample(Provider())

    assert observation.encoder_probe.available
    assert observation.encoder_probe.materialized_bytes == 16


def test_ledger_phase_sequence_and_persistent_row_store_claim() -> None:
    governed, _ = runtime()
    governed.pre_execution_admission()
    governed.loaded_state_calibration(unavailable_loaded_state())
    session = governed.grant(PhaseId.SESSION)
    phase0 = governed.grant(PhaseId.PHASE0)
    governed.release(phase0)
    phase1 = governed.grant(PhaseId.PHASE1)
    governed.release(phase1)
    row_store = governed.grant(PhaseId.PHASE2)
    for phase in (PhaseId.PHASE3, PhaseId.PHASE4, PhaseId.PHASE5):
        grant = governed.grant(phase)
        assert row_store is not None
        assert row_store.id in governed.ledger.actual.active_grant_ids
        governed.release(grant)
    governed.release(row_store)
    governed.release(session)
    governed.close()

    admitted = [event.phase for event in governed.ledger.history if event.kind == "admitted"]
    assert admitted == [
        PhaseId.LOADED,
        PhaseId.SESSION,
        PhaseId.PHASE0,
        PhaseId.PHASE1,
        PhaseId.PHASE2,
        PhaseId.PHASE3,
        PhaseId.PHASE4,
        PhaseId.PHASE5,
    ]
    assert governed.ledger.actual.active_grant_ids == ()


def test_phase_boundaries_record_measured_and_planned_usage() -> None:
    governed, observer = runtime()
    governed._resource_usage_sampler = type(
        "Sampler",
        (),
        {
            "sample": lambda self, *, started_at: ResourceUsageObservation(
                cuda_allocated_bytes=10,
                cuda_reserved_bytes=20,
                cuda_total_bytes=141 * GIB,
                host_rss_bytes=30,
                host_available_bytes=40,
                elapsed_seconds=1.5,
            )
        },
    )()
    advance_to_active(governed)
    governed.active_universe_replan(active_observation(100))

    grant = governed.grant(PhaseId.PHASE3)
    governed.release(grant)

    events = [
        event for event in observer.events if event.name == "planning.resource_actual"
    ]
    assert [event.attrs["boundary"] for event in events] == ["grant", "release"]
    assert events[0].attrs["cuda_reserved_bytes"] == 20
    assert events[0].attrs["planned_vram_bytes"] > 0


def test_measured_usage_over_budget_fails_and_releases_grant() -> None:
    governed, _ = runtime()
    governed._resource_usage_sampler = type(
        "Sampler",
        (),
        {
            "sample": lambda self, *, started_at: ResourceUsageObservation(
                cuda_allocated_bytes=None,
                cuda_reserved_bytes=governed.envelope.effective_vram_budget_bytes + 1,
                cuda_total_bytes=governed.envelope.total_vram_bytes,
                host_rss_bytes=1,
                host_available_bytes=1,
                elapsed_seconds=0.0,
            )
        },
    )()
    advance_to_active(governed)
    governed.active_universe_replan(active_observation(100))
    active_before_grant = governed.ledger.actual.active_grant_ids

    with pytest.raises(ResourceUsageExceededError, match="VRAM budget"):
        governed.grant(PhaseId.PHASE3)

    assert governed.ledger.actual.active_grant_ids == active_before_grant


def test_planning_epoch_state_machine_rejects_skips_and_duplicates() -> None:
    governed, _ = runtime()
    with pytest.raises(PlanningEpochOrderError, match="expected pre_execution_admission"):
        governed.loaded_state_calibration(unavailable_loaded_state())
    with pytest.raises(PlanningEpochOrderError, match="expected pre_execution_admission"):
        governed.active_universe_replan(active_observation(10))

    governed.pre_execution_admission()
    with pytest.raises(PlanningEpochOrderError, match="expected loaded_state_calibration"):
        governed.pre_execution_admission()
    governed.loaded_state_calibration(unavailable_loaded_state())
    with pytest.raises(PlanningEpochOrderError, match="expected active_universe_replan"):
        governed.loaded_state_calibration(unavailable_loaded_state())
    governed.active_universe_replan(active_observation(100))
    with pytest.raises(PlanningEpochOrderError, match="expected complete"):
        governed.active_universe_replan(active_observation(100))
