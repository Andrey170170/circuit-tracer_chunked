from __future__ import annotations

import pytest

from circuit_tracer.governor.contracts import (
    DemandClass,
    DemandLifetime,
    DemandTier,
    ResourceEnvelope,
)
from circuit_tracer.governor.ledger import (
    LedgerViolation,
    PhaseId,
    ResourceAdmissionError,
    ResourceClaim,
    ResourceLedger,
)


def _envelope() -> ResourceEnvelope:
    return ResourceEnvelope(
        total_vram_bytes=100,
        host_budget_bytes=100,
        file_cache_allowance_bytes=10,
        local_disk_bytes=100,
        scratch_disk_bytes=100,
        walltime_seconds=100,
        vram_fraction=1.0,
    )


def _claim(
    name: str,
    tier: DemandTier,
    lifetime: DemandLifetime,
    amount: int = 10,
    demand_class: DemandClass = DemandClass.RIGID,
) -> ResourceClaim:
    return ResourceClaim(name, tier, demand_class, lifetime, amount, "seconds" if tier is DemandTier.WALLTIME else "bytes")


def test_admitted_grant_release_updates_accounting() -> None:
    ledger = ResourceLedger(_envelope())
    grant = ledger.grant(PhaseId.PHASE1, [_claim("trace", DemandTier.VRAM, DemandLifetime.PHASE)])

    assert ledger.actual.amount_for(DemandTier.VRAM) == 10
    ledger.release(grant)
    assert ledger.actual.amount_for(DemandTier.VRAM) == 0
    assert [event.kind for event in ledger.history] == ["admitted", "released"]


def test_permanent_reservation_persists_after_phase_release() -> None:
    ledger = ResourceLedger(_envelope())
    ledger.reserve_permanent([_claim("model", DemandTier.HOST, DemandLifetime.PERMANENT, 30)])
    phase = ledger.grant(PhaseId.PHASE2, [_claim("active", DemandTier.HOST, DemandLifetime.PHASE, 20)])

    ledger.release(phase)
    assert ledger.actual.amount_for(DemandTier.HOST) == 30


def test_overcommit_fails_before_registration() -> None:
    ledger = ResourceLedger(_envelope())
    with pytest.raises(ResourceAdmissionError, match="vram admission exceeds budget"):
        ledger.grant(PhaseId.PHASE3, [_claim("too-big", DemandTier.VRAM, DemandLifetime.PHASE, 101)])

    assert ledger.actual.amount_for(DemandTier.VRAM) == 0
    assert ledger.history == ()


def test_context_release_runs_after_exception() -> None:
    ledger = ResourceLedger(_envelope())
    with pytest.raises(RuntimeError, match="boom"):
        with ledger.grant_context(PhaseId.PHASE4, [_claim("scratch", DemandTier.SCRATCH_DISK, DemandLifetime.TRANSIENT)]):
            raise RuntimeError("boom")

    assert ledger.actual.amount_for(DemandTier.SCRATCH_DISK) == 0
    assert [event.kind for event in ledger.history] == ["admitted", "released"]


def test_double_release_is_idempotent_without_accounting_corruption() -> None:
    ledger = ResourceLedger(_envelope())
    grant = ledger.grant(PhaseId.PHASE5, [_claim("wall", DemandTier.WALLTIME, DemandLifetime.PHASE)])

    ledger.release(grant)
    ledger.release(grant)
    assert ledger.actual.amount_for(DemandTier.WALLTIME) == 0
    assert [event.kind for event in ledger.history] == ["admitted", "released"]


def test_file_backed_elastic_claim_warns_without_rigid_reservation() -> None:
    ledger = ResourceLedger(_envelope())
    ledger.reserve_permanent([
        _claim("checkpoint", DemandTier.FILE_BACKED, DemandLifetime.PERMANENT, 1_000, DemandClass.ELASTIC)
    ])

    assert ledger.actual.amount_for(DemandTier.HOST) == 0
    assert ledger.history[-1].violation is LedgerViolation.ELASTIC_FILE_BACKED


def test_tiers_are_isolated() -> None:
    ledger = ResourceLedger(_envelope())
    ledger.grant(PhaseId.PHASE0, [_claim("host", DemandTier.HOST, DemandLifetime.PHASE, 100)])
    ledger.grant(PhaseId.PHASE1, [_claim("vram", DemandTier.VRAM, DemandLifetime.PHASE, 100)])

    assert ledger.actual.amount_for(DemandTier.HOST) == 100
    assert ledger.actual.amount_for(DemandTier.VRAM) == 100


def test_history_is_deterministic_and_callback_receives_events() -> None:
    events = []
    ledger = ResourceLedger(_envelope(), event_sink=events.append)
    grant = ledger.grant(PhaseId.PHASE1, [_claim("trace", DemandTier.VRAM, DemandLifetime.PHASE)])
    ledger.release(grant)

    assert [(event.sequence, event.kind, event.grant_id) for event in ledger.history] == [
        (1, "admitted", "grant-0001"),
        (2, "released", "grant-0001"),
    ]
    assert events == list(ledger.history)
