"""Runtime resource accounting for Phase E execution.

The ledger owns only admissions and releases.  It deliberately does not know
which tracing backend produced a claim, nor does it render or persist telemetry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from .contracts import DemandClass, DemandLifetime, DemandTier, ResourceEnvelope


class PhaseId(str, Enum):
    """Execution boundary associated with a runtime resource grant."""

    LOADED = "loaded"
    SESSION = "session"
    PHASE0 = "phase0"
    PHASE1 = "phase1"
    PHASE2 = "phase2"
    PHASE3 = "phase3"
    PHASE4 = "phase4"
    PHASE5 = "phase5"


class LedgerViolation(str, Enum):
    """Non-fatal conditions retained for telemetry and policy decisions."""

    ELASTIC_FILE_BACKED = "elastic_file_backed"


class ResourceAdmissionError(RuntimeError):
    """A rigid claim cannot fit within the configured resource envelope."""


@dataclass(frozen=True)
class ResourceClaim:
    name: str
    tier: DemandTier
    demand_class: DemandClass
    lifetime: DemandLifetime
    amount: int | float
    unit: str = "bytes"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.tier, DemandTier):
            raise TypeError("tier must be a DemandTier")
        if not isinstance(self.demand_class, DemandClass):
            raise TypeError("demand_class must be a DemandClass")
        if not isinstance(self.lifetime, DemandLifetime):
            raise TypeError("lifetime must be a DemandLifetime")
        if (
            not isinstance(self.amount, (int, float))
            or isinstance(self.amount, bool)
            or self.amount < 0
        ):
            raise ValueError("amount must be a nonnegative number")
        if self.unit not in {"bytes", "seconds"}:
            raise ValueError("unit must be 'bytes' or 'seconds'")
        expected_unit = "seconds" if self.tier is DemandTier.WALLTIME else "bytes"
        if self.unit != expected_unit:
            raise ValueError(f"{self.tier.value} claims must use {expected_unit!r}")
        if self.tier is DemandTier.FILE_BACKED and self.demand_class is not DemandClass.ELASTIC:
            raise ValueError("file-backed claims must be elastic")


@dataclass
class ResourceGrant:
    id: str
    phase: PhaseId
    claims: tuple[ResourceClaim, ...]
    released: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must be nonempty")
        if not isinstance(self.phase, PhaseId):
            raise TypeError("phase must be a PhaseId")
        if not self.claims:
            raise ValueError("claims must not be empty")


@dataclass(frozen=True)
class ResourceActual:
    """A deterministic snapshot of live rigid reservations."""

    reserved: tuple[tuple[DemandTier, int | float], ...]
    active_grant_ids: tuple[str, ...]

    def amount_for(self, tier: DemandTier) -> int | float:
        return dict(self.reserved).get(tier, 0)


@dataclass(frozen=True)
class LedgerEvent:
    sequence: int
    kind: str
    grant_id: str
    phase: PhaseId
    actual: ResourceActual
    violation: LedgerViolation | None = None


EventSink = Callable[[LedgerEvent], None]


class ResourceLedger:
    """Admission ledger with no global state and append-only event history."""

    _RIGID_TIERS = (
        DemandTier.VRAM,
        DemandTier.HOST,
        DemandTier.LOCAL_DISK,
        DemandTier.SCRATCH_DISK,
        DemandTier.WALLTIME,
    )

    def __init__(self, envelope: ResourceEnvelope, *, event_sink: EventSink | None = None) -> None:
        self.envelope = envelope
        self._event_sink = event_sink
        self._grants: dict[str, ResourceGrant] = {}
        self._history: list[LedgerEvent] = []
        self._next_grant_number = 1

    @property
    def history(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._history)

    @property
    def actual(self) -> ResourceActual:
        totals = {tier: 0 for tier in self._RIGID_TIERS}
        active_ids: list[str] = []
        for grant in self._grants.values():
            if grant.released:
                continue
            active_ids.append(grant.id)
            for claim in grant.claims:
                if claim.demand_class is DemandClass.RIGID:
                    totals[claim.tier] += claim.amount
        return ResourceActual(
            reserved=tuple((tier, totals[tier]) for tier in self._RIGID_TIERS),
            active_grant_ids=tuple(active_ids),
        )

    def reserve_permanent(
        self, claims: tuple[ResourceClaim, ...] | list[ResourceClaim]
    ) -> ResourceGrant:
        return self._admit(PhaseId.LOADED, tuple(claims), permanent=True)

    def grant(
        self, phase: PhaseId, claims: tuple[ResourceClaim, ...] | list[ResourceClaim]
    ) -> ResourceGrant:
        return self._admit(phase, tuple(claims), permanent=False)

    @contextmanager
    def grant_context(
        self, phase: PhaseId, claims: tuple[ResourceClaim, ...] | list[ResourceClaim]
    ) -> Iterator[ResourceGrant]:
        grant = self.grant(phase, claims)
        try:
            yield grant
        finally:
            self.release(grant)

    def release(self, grant: ResourceGrant) -> None:
        registered = self._grants.get(grant.id)
        if registered is not grant:
            raise ValueError("grant was not issued by this ledger")
        if grant.released:
            return
        grant.released = True
        self._record("released", grant)

    def _admit(self, phase: PhaseId, claims: tuple[ResourceClaim, ...], *, permanent: bool) -> ResourceGrant:
        if not claims:
            raise ValueError("claims must not be empty")
        expected_lifetimes = {DemandLifetime.PERMANENT} if permanent else {
            DemandLifetime.PHASE,
            DemandLifetime.TRANSIENT,
        }
        if any(claim.lifetime not in expected_lifetimes for claim in claims):
            kind = "permanent" if permanent else "phase/transient"
            raise ValueError(f"{kind} grants require matching claim lifetimes")
        self._check_admission(claims)
        grant = ResourceGrant(
            id=f"grant-{self._next_grant_number:04d}", phase=phase, claims=claims
        )
        self._next_grant_number += 1
        self._grants[grant.id] = grant
        self._record("admitted", grant)
        for claim in claims:
            if claim.tier is DemandTier.FILE_BACKED:
                self._record("warning", grant, LedgerViolation.ELASTIC_FILE_BACKED)
        return grant

    def _check_admission(self, claims: tuple[ResourceClaim, ...]) -> None:
        totals = dict(self.actual.reserved)
        for claim in claims:
            if claim.demand_class is DemandClass.RIGID:
                totals[claim.tier] = totals.get(claim.tier, 0) + claim.amount
        for tier, limit in self._limits().items():
            if totals.get(tier, 0) > limit:
                raise ResourceAdmissionError(
                    f"{tier.value} admission exceeds budget: {totals[tier]} > {limit}"
                )

    def _limits(self) -> dict[DemandTier, int | float]:
        return {
            DemandTier.VRAM: self.envelope.effective_vram_budget_bytes,
            DemandTier.HOST: self.envelope.host_budget_bytes,
            DemandTier.LOCAL_DISK: self.envelope.local_disk_bytes,
            DemandTier.SCRATCH_DISK: self.envelope.scratch_disk_bytes,
            DemandTier.WALLTIME: self.envelope.walltime_seconds,
        }

    def _record(
        self, kind: str, grant: ResourceGrant, violation: LedgerViolation | None = None) -> None:
        event = LedgerEvent(
            sequence=len(self._history) + 1,
            kind=kind,
            grant_id=grant.id,
            phase=grant.phase,
            actual=self.actual,
            violation=violation,
        )
        self._history.append(event)
        if self._event_sink is not None:
            self._event_sink(event)
