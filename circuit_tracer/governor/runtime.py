"""Live Phase E planning epochs and resource accounting.

This module owns runtime planning state.  It emits typed domain events through
the observer boundary and deliberately knows nothing about telemetry sinks.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Callable
from typing import Protocol

import torch

from circuit_tracer.observability.events import TraceEvent, TraceObserver

from .calibration import CalibrationCatalog

from .contracts import (
    AdmissionMode,
    DemandEstimate,
    DemandLifetime,
    DemandTier,
    EncoderResidency,
    PhysicalExecutionConfig,
    PhysicalExecutionRequirements,
    PlanningProgress,
    ProviderProfile,
    ResourceEnvelope,
    RowStorePolicy,
    StorageTier,
    TracePlan,
    TraceSemantics,
)
from .ledger import LedgerEvent, PhaseId, ResourceClaim, ResourceGrant, ResourceLedger
from .host_budget import HostBudgetDiscovery, discover_host_budget
from .resolver import resolve_trace_plan


class PlanningEpoch(str, Enum):
    PRE_EXECUTION_ADMISSION = "pre_execution_admission"
    LOADED_STATE_CALIBRATION = "loaded_state_calibration"
    ACTIVE_UNIVERSE_REPLAN = "active_universe_replan"
    PHASE3_ENTRY_REPLAN = "phase3_entry_replan"
    PHASE4_ENTRY_REPLAN = "phase4_entry_replan"


class PlanningEpochOrderError(RuntimeError):
    pass


class RuntimePlanningRefusedError(RuntimeError):
    def __init__(self, revision: "PlanRevision") -> None:
        self.revision = revision
        super().__init__(
            f"{revision.epoch.value} refused execution: "
            + "; ".join(revision.plan.admission.refusals)
        )


@dataclass(frozen=True)
class ProviderUnitProbe:
    name: str
    available: bool
    elapsed_ms: float | None = None
    materialized_bytes: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LoadedStateObservation:
    cuda_available: bool
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_total_bytes: int | None
    host_rss_bytes: int | None
    host_available_bytes: int | None
    decoder_probe: ProviderUnitProbe
    encoder_probe: ProviderUnitProbe


@dataclass(frozen=True)
class ActiveUniverseObservation:
    total_nnz: int
    shape: tuple[int, ...]
    per_layer_counts: tuple[int, ...]
    per_position_counts: tuple[int, ...]
    membership_fingerprint: str
    membership_sample: tuple[tuple[int, ...], ...]

    @classmethod
    def from_sparse_tensor(
        cls, value: torch.Tensor, *, sample_size: int = 16
    ) -> "ActiveUniverseObservation":
        sparse = value.coalesce()
        indices = sparse.indices().detach()
        shape = tuple(int(v) for v in sparse.shape)
        layer_count = shape[0] if shape else 0
        position_count = shape[1] if len(shape) > 1 else 0
        per_layer = torch.bincount(indices[0], minlength=layer_count).cpu().tolist()
        per_position = (
            torch.bincount(indices[1], minlength=position_count).cpu().tolist()
            if len(shape) > 1
            else []
        )
        digest = hashlib.sha256(repr(shape).encode("ascii"))
        for start in range(0, indices.shape[1], 4096):
            chunk = indices[:, start : start + 4096].to(device="cpu").contiguous()
            digest.update(chunk.numpy().tobytes())
        sample = indices[:, :sample_size].T.to(device="cpu").tolist()
        return cls(
            total_nnz=int(sparse._nnz()),
            shape=shape,
            per_layer_counts=tuple(int(value) for value in per_layer),
            per_position_counts=tuple(int(value) for value in per_position),
            membership_fingerprint=digest.hexdigest(),
            membership_sample=tuple(tuple(int(value) for value in row) for row in sample),
        )


@dataclass(frozen=True)
class PlanRevision:
    epoch: PlanningEpoch
    parent_execution_fingerprint: str
    execution_fingerprint: str
    semantic_fingerprint: str
    changed_mechanisms: tuple[str, ...]
    plan: TracePlan


class LoadedStateSampler(Protocol):
    def sample(self, provider: object) -> LoadedStateObservation: ...


@dataclass(frozen=True)
class ResourceUsageObservation:
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_total_bytes: int | None
    host_rss_bytes: int | None
    host_available_bytes: int | None
    elapsed_seconds: float
    cuda_peak_allocated_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None


@dataclass(frozen=True)
class PhaseEntryObservation:
    target_phase: PhaseId
    prior_phase: PhaseId | None
    prior_boundary: str | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_total_bytes: int | None
    host_rss_bytes: int | None
    host_available_bytes: int | None
    elapsed_seconds: float | None
    predicted_persistent_vram_bytes: int
    vram_prediction_error_bytes: int | None
    effective_phase_vram_budget_bytes: int


class ResourceUsageSampler(Protocol):
    def sample(self, *, started_at: float) -> ResourceUsageObservation: ...

    def reset_cuda_peak(self) -> None: ...


class TorchResourceUsageSampler:
    """Low-overhead process/CUDA measurements at governor phase boundaries."""

    def sample(self, *, started_at: float) -> ResourceUsageObservation:
        allocated = reserved = total = peak_allocated = peak_reserved = None
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            total = int(
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            )
        rss = available = None
        try:
            import psutil

            rss = int(psutil.Process(os.getpid()).memory_info().rss)
            available = int(psutil.virtual_memory().available)
        except ImportError:
            pass
        return ResourceUsageObservation(
            cuda_allocated_bytes=allocated,
            cuda_reserved_bytes=reserved,
            cuda_total_bytes=total,
            host_rss_bytes=rss,
            host_available_bytes=available,
            elapsed_seconds=time.perf_counter() - started_at,
            cuda_peak_allocated_bytes=peak_allocated,
            cuda_peak_reserved_bytes=peak_reserved,
        )

    def reset_cuda_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _tensor_bytes(value: object) -> int | None:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return None


def _probe(name: str, callback) -> ProviderUnitProbe:
    started = time.perf_counter()
    try:
        value = callback()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return ProviderUnitProbe(
            name=name,
            available=True,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            materialized_bytes=_tensor_bytes(value),
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError, RuntimeError) as exc:
        return ProviderUnitProbe(name=name, available=False, reason=f"{type(exc).__name__}: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TorchLoadedStateSampler:
    """Best-effort loaded-model sampler with deterministic one-unit probes."""

    def sample(self, provider: object) -> LoadedStateObservation:
        capabilities = getattr(provider, "capabilities", None)
        decoder = ProviderUnitProbe("decoder_chunk", False, reason="capability unavailable")
        if bool(getattr(capabilities, "supports_lazy_decoder_chunks", False)) and callable(
            getattr(provider, "get_decoder_chunk", None)
        ):
            decoder = _probe("decoder_chunk", lambda: provider.get_decoder_chunk(0, 0))
        encoder = ProviderUnitProbe("encoder_row", False, reason="capability unavailable")
        if bool(getattr(capabilities, "supports_encoder_row_materialization", False)) and callable(
            getattr(provider, "materialize_encoder_rows", None)
        ):
            encoder = _probe(
                "encoder_row",
                lambda: provider.materialize_encoder_rows(
                    torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)
                ),
            )

        cuda = torch.cuda.is_available()
        allocated = reserved = total = None
        if cuda:
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            total = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
        rss = available = None
        try:
            import psutil

            rss = int(psutil.Process(os.getpid()).memory_info().rss)
            available = int(psutil.virtual_memory().available)
        except ImportError:
            pass
        return LoadedStateObservation(
            cuda_available=cuda,
            cuda_allocated_bytes=allocated,
            cuda_reserved_bytes=reserved,
            cuda_total_bytes=total,
            host_rss_bytes=rss,
            host_available_bytes=available,
            decoder_probe=decoder,
            encoder_probe=encoder,
        )


_FROZEN_FIELDS = (
    "decoder_fetch_chunk_size",
    "decoder_cache_bytes",
    "session_capacity",
    "phase1_source_batch_size",
    "source_microbatch_size",
    "replay_window",
    "prefetch_depth",
)

_PHASE_ENTRY_PHYSICAL_FIELDS = tuple(PhysicalExecutionConfig.__dataclass_fields__)


class FrozenMechanismRevisionError(RuntimeError):
    pass


class ResourceUsageExceededError(RuntimeError):
    pass


class TraceGovernorRuntime:
    """One run's cohesive planning epochs, revisions, and resource ledger."""

    def __init__(
        self,
        *,
        plan: TracePlan,
        workload: TraceSemantics,
        profile: ProviderProfile,
        envelope: ResourceEnvelope,
        requirements: PhysicalExecutionRequirements,
        observer: TraceObserver,
        admission_mode: AdmissionMode = AdmissionMode.ENFORCE,
        host_budget_discoverer: Callable[[int | None], HostBudgetDiscovery] | None = None,
        resource_usage_sampler: ResourceUsageSampler | None = None,
        calibration_catalog: CalibrationCatalog | None = None,
    ) -> None:
        self.workload = workload
        self.profile = profile
        self.envelope = envelope
        self.requirements = requirements
        self.current_plan = plan
        self.observer = observer
        self.admission_mode = admission_mode
        self.calibration_catalog = calibration_catalog
        self._host_budget_discoverer = host_budget_discoverer or discover_host_budget
        self._resource_usage_sampler = resource_usage_sampler or TorchResourceUsageSampler()
        self._started_at = time.perf_counter()
        self._last_resource_usage: tuple[PhaseId, str, ResourceUsageObservation] | None = None
        self.revisions: list[PlanRevision] = []
        self._next_epoch: PlanningEpoch | None = PlanningEpoch.PRE_EXECUTION_ADMISSION
        self.ledger = ResourceLedger(
            envelope,
            admission_mode=admission_mode,
            event_sink=self._ledger_event,
        )
        self._grants: list[ResourceGrant] = []

    def pre_execution_admission(self) -> PlanRevision:
        epoch = PlanningEpoch.PRE_EXECUTION_ADMISSION
        self._require_epoch(epoch)
        revision = self._record_revision(epoch, self.current_plan)
        self._next_epoch = PlanningEpoch.LOADED_STATE_CALIBRATION
        return revision

    def loaded_state_calibration(self, observation: LoadedStateObservation) -> PlanRevision:
        epoch = PlanningEpoch.LOADED_STATE_CALIBRATION
        self._require_epoch(epoch)
        costs = self.profile.costs
        measured_values = tuple(
            value
            for value in (
                observation.cuda_allocated_bytes,
                observation.cuda_reserved_bytes,
            )
            if value is not None
        )
        measured = max(measured_values) if measured_values else None
        if measured is not None:
            self.profile = replace(self.profile, costs=replace(costs, fixed_vram_bytes=measured))
        measured_models = tuple(
            replace(
                model,
                fixed_vram_bytes=(
                    observation.decoder_probe.materialized_bytes
                    if model.phase == "phase0"
                    and observation.decoder_probe.materialized_bytes is not None
                    else model.fixed_vram_bytes
                ),
                host_bytes_per_item=(
                    observation.encoder_probe.materialized_bytes
                    if model.phase == "phase4"
                    and observation.encoder_probe.materialized_bytes is not None
                    else model.host_bytes_per_item
                ),
            )
            for model in self.profile.phase_memory_models
        )
        if measured_models:
            self.profile = replace(self.profile, phase_memory_models=measured_models)
        if observation.cuda_total_bytes is not None:
            self.envelope = replace(self.envelope, total_vram_bytes=observation.cuda_total_bytes)
        discovery = self._host_budget_discoverer(self.envelope.host_budget_bytes)
        host_candidates = [
            value
            for value in (
                discovery.budget_bytes,
                (
                    observation.host_rss_bytes + observation.host_available_bytes
                    if observation.host_rss_bytes is not None
                    and observation.host_available_bytes is not None
                    else None
                ),
            )
            if value is not None and value > 0
        ]
        if host_candidates:
            self.envelope = replace(self.envelope, host_budget_bytes=min(host_candidates))
        # Loaded-state calibration is the last epoch before any grant exists.
        # Bind accounting to the measured envelope used by the revised plan.
        self.ledger = ResourceLedger(
            self.envelope,
            admission_mode=self.admission_mode,
            event_sink=self._ledger_event,
        )
        self.observer.observe(
            TraceEvent(
                scope="run",
                name="planning.host_budget_discovery",
                attrs={
                    "source": discovery.source,
                    "discovered_budget_bytes": discovery.budget_bytes,
                    "rss_plus_available_bytes": (
                        observation.host_rss_bytes + observation.host_available_bytes
                        if observation.host_rss_bytes is not None
                        and observation.host_available_bytes is not None
                        else None
                    ),
                    "effective_host_budget_bytes": self.envelope.host_budget_bytes,
                    "warnings": discovery.warnings,
                },
            )
        )
        candidate = resolve_trace_plan(
            self.workload,
            self.profile,
            self.envelope,
            self.requirements,
            catalog=self.calibration_catalog,
        )
        self._emit_observation(epoch, observation)
        revision = self._record_revision(epoch, candidate)
        self._next_epoch = PlanningEpoch.ACTIVE_UNIVERSE_REPLAN
        if candidate.admission.admitted or self.admission_mode is AdmissionMode.ADVISORY:
            for name, reason in _EXCLUDED_ESTIMATES.items():
                self._emit_estimate_excluded(name, reason)
            self._reserve_permanent(candidate.admission.estimates)
        return revision

    def active_universe_replan(self, observation: ActiveUniverseObservation) -> PlanRevision:
        epoch = PlanningEpoch.ACTIVE_UNIVERSE_REPLAN
        self._require_epoch(epoch)
        if observation.total_nnz <= 0:
            raise ValueError("active universe must contain at least one member")
        self.workload = replace(self.workload, estimated_active_features=observation.total_nnz)
        current = self.current_plan.physical
        late_requirements = replace(
            self.requirements,
            decoder_fetch_chunk_size=current.decoder_fetch_chunk_size,
            decoder_cache_bytes=current.decoder_cache_bytes,
            session_capacity=current.session_capacity,
            phase1_source_batch_size=current.phase1_source_batch_size,
            source_microbatch_size=current.source_microbatch_size,
            replay_window=current.replay_window,
            prefetch_depth=current.prefetch_depth,
            replay_tile_cache_bytes=self.requirements.replay_tile_cache_bytes,
        )
        candidate = resolve_trace_plan(
            self.workload,
            self.profile,
            self.envelope,
            late_requirements,
            frozen_fields=_FROZEN_FIELDS,
            reported_hard_constraints=self.requirements,
            progress=self._planning_progress(("phase0",)),
            catalog=self.calibration_catalog,
        )
        candidate = replace(candidate, semantic_fingerprint=self.current_plan.semantic_fingerprint)
        changed = _changed_physical(self.current_plan.physical, candidate.physical)
        forbidden = tuple(name for name in changed if name in _FROZEN_FIELDS)
        if forbidden:
            self.observer.observe(
                TraceEvent(
                    scope="run",
                    name="planning.refusal",
                    attrs={
                        "epoch": epoch.value,
                        "reason": "frozen mechanism revision",
                        "mechanisms": forbidden,
                    },
                )
            )
            raise FrozenMechanismRevisionError(
                "late plan attempted to revise frozen mechanisms: " + ", ".join(forbidden)
            )
        self._emit_observation(epoch, observation)
        revision = self._record_revision(epoch, candidate)
        self._next_epoch = PlanningEpoch.PHASE3_ENTRY_REPLAN
        if (
            not candidate.admission.admitted
            and self.admission_mode is AdmissionMode.ENFORCE
        ):
            raise RuntimePlanningRefusedError(revision)
        return revision

    def phase3_entry_replan(self) -> PlanRevision:
        revision = self._phase_entry_replan(
            epoch=PlanningEpoch.PHASE3_ENTRY_REPLAN,
            target_phase=PhaseId.PHASE3,
            target_field="logit_microbatch_size",
        )
        self._next_epoch = PlanningEpoch.PHASE4_ENTRY_REPLAN
        return revision

    def phase4_entry_replan(self) -> PlanRevision:
        revision = self._phase_entry_replan(
            epoch=PlanningEpoch.PHASE4_ENTRY_REPLAN,
            target_phase=PhaseId.PHASE4,
            target_field="feature_microbatch_size",
        )
        self._next_epoch = None
        return revision

    def _phase_entry_replan(
        self,
        *,
        epoch: PlanningEpoch,
        target_phase: PhaseId,
        target_field: str,
    ) -> PlanRevision:
        self._require_epoch(epoch)
        prior_phase = PhaseId.PHASE2 if target_phase is PhaseId.PHASE3 else PhaseId.PHASE3
        self._observe_resource_usage(phase=prior_phase, boundary=f"{target_phase.value}_entry")
        observation = self._phase_entry_observation(target_phase)
        phase_envelope = replace(
            self.envelope,
            vram_budget_bytes=observation.effective_phase_vram_budget_bytes,
        )
        current = self.current_plan.physical
        requirements = replace(
            self.requirements,
            decoder_fetch_chunk_size=current.decoder_fetch_chunk_size,
            decoder_cache_bytes=current.decoder_cache_bytes,
            session_capacity=current.session_capacity,
            phase1_source_batch_size=current.phase1_source_batch_size,
            source_microbatch_size=current.source_microbatch_size,
            feature_microbatch_size=current.feature_microbatch_size,
            logit_microbatch_size=current.logit_microbatch_size,
            replay_window=current.replay_window,
            prefetch_depth=current.prefetch_depth,
            replay_tile_cache_bytes=current.replay_tile_cache_bytes,
            encoder_residency=EncoderResidency(current.encoder_residency),
            row_store_policy=RowStorePolicy(current.row_store_policy),
            spill_target=(
                StorageTier(current.spill_target) if current.spill_target is not None else None
            ),
            cache_policy=current.cache_policy,
        )
        original_target = getattr(self.requirements, target_field)
        requirements = replace(requirements, **{target_field: original_target})
        frozen_fields = tuple(
            name for name in _PHASE_ENTRY_PHYSICAL_FIELDS if name != target_field
        )
        candidate = resolve_trace_plan(
            self.workload,
            self.profile,
            phase_envelope,
            requirements,
            frozen_fields=frozen_fields,
            reported_hard_constraints=self.requirements,
            progress=self._planning_progress(
                ("phase0", "phase1", "phase2")
                if target_phase is PhaseId.PHASE3
                else ("phase0", "phase1", "phase2", "phase3")
            ),
            catalog=self.calibration_catalog,
        )
        candidate = replace(candidate, semantic_fingerprint=self.current_plan.semantic_fingerprint)
        changed = _changed_physical(current, candidate.physical)
        forbidden = tuple(name for name in changed if name != target_field)
        if forbidden:
            self.observer.observe(
                TraceEvent(
                    scope="run",
                    name="planning.refusal",
                    attrs={
                        "epoch": epoch.value,
                        "reason": "phase-entry frozen mechanism revision",
                        "mechanisms": forbidden,
                    },
                )
            )
            raise FrozenMechanismRevisionError(
                f"{epoch.value} attempted to revise frozen mechanisms: "
                + ", ".join(forbidden)
            )
        self._emit_observation(epoch, observation)
        revision = self._record_revision(epoch, candidate)
        if (
            not candidate.admission.admitted
            and self.admission_mode is AdmissionMode.ENFORCE
        ):
            raise RuntimePlanningRefusedError(revision)
        return revision

    def _planning_progress(
        self, completed_phases: tuple[str, ...]
    ) -> PlanningProgress:
        if self._last_resource_usage is not None:
            elapsed = self._last_resource_usage[2].elapsed_seconds
        else:
            elapsed = sum(
                high
                for phase, _low, high in self.current_plan.admission.phase_predictions
                if phase in completed_phases
            )
        return PlanningProgress(
            completed_phases=completed_phases,
            observed_elapsed_seconds=elapsed,
        )

    def _phase_entry_observation(self, target_phase: PhaseId) -> PhaseEntryObservation:
        persistent_names = {
            "model_vram",
            "trace_vram",
            "decoder_cache_vram",
            "source_microbatch_vram",
        }
        predicted_persistent = int(
            sum(
                estimate.amount
                for estimate in self.current_plan.admission.estimates
                if estimate.name in persistent_names
            )
        )
        prior = self._last_resource_usage
        if prior is None:
            return PhaseEntryObservation(
                target_phase=target_phase,
                prior_phase=None,
                prior_boundary=None,
                cuda_allocated_bytes=None,
                cuda_reserved_bytes=None,
                cuda_total_bytes=None,
                host_rss_bytes=None,
                host_available_bytes=None,
                elapsed_seconds=None,
                predicted_persistent_vram_bytes=predicted_persistent,
                vram_prediction_error_bytes=None,
                effective_phase_vram_budget_bytes=(
                    self.envelope.effective_vram_budget_bytes
                ),
            )
        prior_phase, prior_boundary, usage = prior
        observed_residencies = tuple(
            value
            for value in (
                usage.cuda_allocated_bytes,
                usage.cuda_reserved_bytes,
            )
            if value is not None
        )
        observed_residency = max(observed_residencies) if observed_residencies else None
        prediction_error = (
            None if observed_residency is None else observed_residency - predicted_persistent
        )
        effective_budget = max(
            0,
            self.envelope.effective_vram_budget_bytes
            - max(0, prediction_error or 0),
        )
        return PhaseEntryObservation(
            target_phase=target_phase,
            prior_phase=prior_phase,
            prior_boundary=prior_boundary,
            cuda_allocated_bytes=usage.cuda_allocated_bytes,
            cuda_reserved_bytes=usage.cuda_reserved_bytes,
            cuda_total_bytes=usage.cuda_total_bytes,
            host_rss_bytes=usage.host_rss_bytes,
            host_available_bytes=usage.host_available_bytes,
            elapsed_seconds=usage.elapsed_seconds,
            predicted_persistent_vram_bytes=predicted_persistent,
            vram_prediction_error_bytes=prediction_error,
            effective_phase_vram_budget_bytes=effective_budget,
        )

    def grant(self, phase: PhaseId) -> ResourceGrant | None:
        claims = _claims_for_phase(self.current_plan.admission.estimates, phase)
        if not claims:
            return None
        reset = getattr(self._resource_usage_sampler, "reset_cuda_peak", None)
        if callable(reset):
            reset()
        grant = self.ledger.grant(phase, claims)
        self._grants.append(grant)
        try:
            self._observe_resource_usage(phase=phase, boundary="grant")
        except BaseException:
            self.ledger.release(grant)
            raise
        return grant

    def release(self, grant: ResourceGrant | None) -> None:
        if grant is not None:
            try:
                self._observe_resource_usage(phase=grant.phase, boundary="release")
            finally:
                self.ledger.release(grant)

    def close(self) -> None:
        for grant in reversed(self._grants):
            self.ledger.release(grant)
        self.observer.observe(
            TraceEvent(
                scope="run",
                name="planning.terminal_cleanup",
                attrs={
                    "active_grants": len(self.ledger.actual.active_grant_ids),
                    "revision_count": len(self.revisions),
                },
            )
        )

    def _reserve_permanent(self, estimates: tuple[DemandEstimate, ...]) -> None:
        claims = list(_claims_for_phase(estimates, PhaseId.LOADED))
        if any(claim.name == "baseline_total_host" and claim.amount > 0 for claim in claims):
            self._emit_estimate_excluded("known_rigid_host", "included in baseline_total_host")
            claims = [claim for claim in claims if claim.name != "known_rigid_host"]
        if claims:
            self._grants.append(self.ledger.reserve_permanent(claims))

    def _emit_estimate_excluded(self, name: str, reason: str) -> None:
        self.observer.observe(
            TraceEvent(
                scope="run",
                name="planning.estimate_excluded",
                attrs={
                    "estimate": name,
                    "reason": reason,
                },
            )
        )

    def _observe_resource_usage(self, *, phase: PhaseId, boundary: str) -> None:
        usage = self._resource_usage_sampler.sample(started_at=self._started_at)
        self._last_resource_usage = (phase, boundary, usage)
        reserved = self.ledger.actual
        self.observer.observe(
            TraceEvent(
                scope="phase",
                phase=phase.value,
                name="planning.resource_actual",
                attrs={
                    "boundary": boundary,
                    "cuda_allocated_bytes": usage.cuda_allocated_bytes,
                    "cuda_reserved_bytes": usage.cuda_reserved_bytes,
                    "cuda_total_bytes": usage.cuda_total_bytes,
                    "cuda_peak_allocated_bytes": usage.cuda_peak_allocated_bytes,
                    "cuda_peak_reserved_bytes": usage.cuda_peak_reserved_bytes,
                    "host_rss_bytes": usage.host_rss_bytes,
                    "host_available_bytes": usage.host_available_bytes,
                    "elapsed_seconds": usage.elapsed_seconds,
                    "planned_vram_bytes": reserved.amount_for(DemandTier.VRAM),
                    "planned_host_bytes": reserved.amount_for(DemandTier.HOST),
                    "planned_local_disk_bytes": reserved.amount_for(
                        DemandTier.LOCAL_DISK
                    ),
                    "planned_scratch_disk_bytes": reserved.amount_for(
                        DemandTier.SCRATCH_DISK
                    ),
                },
            )
        )
        violations = []
        if (
            usage.cuda_reserved_bytes is not None
            and usage.cuda_reserved_bytes > self.envelope.effective_vram_budget_bytes
        ):
            violations.append(
                "CUDA reserved memory exceeds governed VRAM budget: "
                f"{usage.cuda_reserved_bytes} > {self.envelope.effective_vram_budget_bytes}"
            )
        if (
            usage.host_rss_bytes is not None
            and usage.host_rss_bytes > self.envelope.host_budget_bytes
        ):
            violations.append(
                "process RSS exceeds governed host budget: "
                f"{usage.host_rss_bytes} > {self.envelope.host_budget_bytes}"
            )
        if violations:
            if self.admission_mode is AdmissionMode.ENFORCE:
                raise ResourceUsageExceededError("; ".join(violations))
            self.observer.observe(
                TraceEvent(
                    scope="phase",
                    phase=phase.value,
                    name="planning.resource_limit_bypassed",
                    attrs={
                        "boundary": boundary,
                        "reason": "observed_governor_budget_exceeded",
                        "violations": tuple(violations),
                    },
                )
            )

    def _record_revision(self, epoch: PlanningEpoch, candidate: TracePlan) -> PlanRevision:
        parent = self.current_plan.execution_fingerprint
        changed = _changed_physical(self.current_plan.physical, candidate.physical)
        revision = PlanRevision(
            epoch=epoch,
            parent_execution_fingerprint=parent,
            execution_fingerprint=candidate.execution_fingerprint,
            semantic_fingerprint=candidate.semantic_fingerprint,
            changed_mechanisms=changed,
            plan=candidate,
        )
        self.current_plan = candidate
        self.revisions.append(revision)
        self.observer.observe(
            TraceEvent(
                scope="run",
                name=f"planning.{epoch.value}",
                attrs={
                    "admitted": candidate.admission.admitted,
                    "governor_admission_mode": self.admission_mode.value,
                    "parent_execution_fingerprint": parent,
                    "execution_fingerprint": candidate.execution_fingerprint,
                    "semantic_fingerprint": candidate.semantic_fingerprint,
                    "changed_mechanisms": changed,
                    "refusals": candidate.admission.refusals,
                    "admission_decisions": candidate.admission.decisions,
                    "admission_warnings": candidate.admission.warnings,
                    "candidate_count": candidate.admission.candidate_count,
                    "admissible_candidate_count": (
                        candidate.admission.admissible_candidate_count
                    ),
                    "hard_constraints": candidate.admission.hard_constraints,
                    "frozen_fields": candidate.admission.frozen_fields,
                    "free_fields": candidate.admission.free_fields,
                    "binding_constraints": candidate.admission.binding_constraints,
                    "selected_objective": candidate.admission.selected_objective,
                    "rejected_candidates": candidate.admission.rejected_candidates,
                    "row_store_policy": candidate.physical.row_store_policy,
                    "row_store_bytes": candidate.physical.row_store_bytes,
                    "spill_target": candidate.physical.spill_target,
                    "session_capacity": candidate.physical.session_capacity,
                    "phase1_source_batch_size": (
                        candidate.physical.phase1_source_batch_size
                    ),
                    "source_microbatch_size": candidate.physical.source_microbatch_size,
                    "feature_microbatch_size": candidate.physical.feature_microbatch_size,
                    "logit_microbatch_size": candidate.physical.logit_microbatch_size,
                    "decoder_cache_bytes": candidate.physical.decoder_cache_bytes,
                    "decoder_fetch_chunk_size": candidate.physical.decoder_fetch_chunk_size,
                    "replay_window": candidate.physical.replay_window,
                    "prefetch_depth": candidate.physical.prefetch_depth,
                    "replay_tile_cache_bytes": (
                        candidate.physical.replay_tile_cache_bytes
                    ),
                    "encoder_residency": candidate.physical.encoder_residency,
                    "cache_policy": candidate.physical.cache_policy.value,
                    "confidence": candidate.admission.confidence,
                    "extrapolated_dimensions": (
                        candidate.admission.extrapolated_dimensions
                    ),
                    "calibration_evidence": candidate.admission.calibration_evidence,
                    "phase_predictions": candidate.admission.phase_predictions,
                    "remaining_projection": candidate.admission.remaining_projection,
                    "domain_summary": candidate.admission.domain_summary,
                },
            )
        )
        if (
            not candidate.admission.admitted
            and self.admission_mode is AdmissionMode.ADVISORY
        ):
            self.observer.observe(
                TraceEvent(
                    scope="run",
                    name="planning.admission_bypassed",
                    attrs={
                        "epoch": epoch.value,
                        "reason": "admission_report_refused",
                        "refusals": candidate.admission.refusals,
                    },
                )
            )
        return revision

    def _require_epoch(self, requested: PlanningEpoch) -> None:
        if self._next_epoch is not requested:
            expected = "complete" if self._next_epoch is None else self._next_epoch.value
            raise PlanningEpochOrderError(
                f"planning epoch {requested.value} is out of order; expected {expected}"
            )

    def _emit_observation(self, epoch: PlanningEpoch, observation: object) -> None:
        if isinstance(observation, LoadedStateObservation):
            attrs = {
                "epoch": epoch.value,
                "cuda_available": observation.cuda_available,
                "cuda_allocated_bytes": observation.cuda_allocated_bytes,
                "cuda_reserved_bytes": observation.cuda_reserved_bytes,
                "cuda_total_bytes": observation.cuda_total_bytes,
                "host_rss_bytes": observation.host_rss_bytes,
                "host_available_bytes": observation.host_available_bytes,
            }
            probes = (observation.decoder_probe, observation.encoder_probe)
        elif isinstance(observation, ActiveUniverseObservation):
            attrs = {
                "epoch": epoch.value,
                "total_nnz": observation.total_nnz,
                "rank": len(observation.shape),
                "shape": "x".join(str(value) for value in observation.shape),
                "layer_count": len(observation.per_layer_counts),
                "position_count": len(observation.per_position_counts),
                "membership_fingerprint": observation.membership_fingerprint,
                "membership_sample_count": len(observation.membership_sample),
            }
            probes = ()
        elif isinstance(observation, PhaseEntryObservation):
            attrs = {
                "epoch": epoch.value,
                "target_phase": observation.target_phase.value,
                "prior_phase": (
                    observation.prior_phase.value
                    if observation.prior_phase is not None
                    else None
                ),
                "prior_boundary": observation.prior_boundary,
                "cuda_allocated_bytes": observation.cuda_allocated_bytes,
                "cuda_reserved_bytes": observation.cuda_reserved_bytes,
                "cuda_total_bytes": observation.cuda_total_bytes,
                "host_rss_bytes": observation.host_rss_bytes,
                "host_available_bytes": observation.host_available_bytes,
                "elapsed_seconds": observation.elapsed_seconds,
                "predicted_persistent_vram_bytes": (
                    observation.predicted_persistent_vram_bytes
                ),
                "vram_prediction_error_bytes": observation.vram_prediction_error_bytes,
                "effective_phase_vram_budget_bytes": (
                    observation.effective_phase_vram_budget_bytes
                ),
            }
            probes = ()
        else:
            raise TypeError(f"unsupported planning observation: {type(observation).__name__}")
        self.observer.observe(
            TraceEvent(
                scope="run",
                name="planning.observation",
                attrs=attrs,
            )
        )
        for probe in probes:
            self.observer.observe(
                TraceEvent(
                    scope="run",
                    name="planning.provider_probe",
                    attrs={
                        "epoch": epoch.value,
                        "probe": probe.name,
                        "available": probe.available,
                        "elapsed_ms": probe.elapsed_ms,
                        "materialized_bytes": probe.materialized_bytes,
                        "reason": probe.reason,
                    },
                )
            )
        if isinstance(observation, ActiveUniverseObservation):
            for index, count in enumerate(observation.per_layer_counts):
                self.observer.observe(
                    TraceEvent(
                        scope="run",
                        name="planning.active_universe_layer",
                        attrs={"layer_index": index, "active_count": count},
                    )
                )
            for index, count in enumerate(observation.per_position_counts):
                self.observer.observe(
                    TraceEvent(
                        scope="run",
                        name="planning.active_universe_position",
                        attrs={"position_index": index, "active_count": count},
                    )
                )

    def _ledger_event(self, event: LedgerEvent) -> None:
        planned = tuple(
            (estimate.name, estimate.amount, estimate.unit)
            for estimate in self.current_plan.admission.estimates
            if estimate.name in _PHASE_CLAIMS.get(event.phase, set())
            or (event.phase is PhaseId.LOADED and estimate.lifetime is DemandLifetime.PERMANENT)
        )
        event_name = (
            "planning.resource_limit_bypassed"
            if event.kind == "resource_limit_bypassed"
            else f"planning.ledger_{'grant' if event.kind == 'admitted' else event.kind}"
        )
        attrs = {
            "grant_id": event.grant_id,
            "active_grants": event.actual.active_grant_ids,
            "reserved": tuple((tier.value, amount) for tier, amount in event.actual.reserved),
            "planned": planned,
            "violation": None if event.violation is None else event.violation.value,
        }
        if event.kind == "resource_limit_bypassed":
            attrs.update(
                reason="predicted_resource_ledger_budget_exceeded",
                violations=event.reasons,
            )
        self.observer.observe(
            TraceEvent(
                scope="phase",
                phase=event.phase.value,
                name=event_name,
                attrs=attrs,
            )
        )


def _changed_physical(
    before: PhysicalExecutionConfig, after: PhysicalExecutionConfig
) -> tuple[str, ...]:
    return tuple(
        name
        for name in before.__dataclass_fields__
        if getattr(before, name) != getattr(after, name)
    )


_PHASE_CLAIMS = {
    PhaseId.SESSION: {"trace_vram", "decoder_cache_vram", "source_microbatch_vram"},
    PhaseId.PHASE0: {"decoder_fetch_vram", "prefetch_vram", "replay_vram", "replay_host"},
    PhaseId.PHASE1: {"prompt_host"},
    PhaseId.PHASE2: {"active_host", "row_store_disk"},
    PhaseId.PHASE3: {"target_vram", "logit_microbatch_vram"},
    PhaseId.PHASE4: {
        "feature_microbatch_vram", "encoder_residency_host",
        "replay_tile_cache_host",
    },
    PhaseId.PHASE5: {"predicted_walltime_high"},
}

_EXCLUDED_ESTIMATES = {
    "predicted_walltime_low": "upper-bound walltime claim supersedes lower-bound estimate",
    **{
        f"predicted_walltime_phase{phase}_{bound}": (
            "additive phase projection is metadata; total upper-bound claim owns admission"
        )
        for phase in range(6)
        for bound in ("low", "high")
    },
}


def _claims_for_phase(
    estimates: tuple[DemandEstimate, ...], phase: PhaseId
) -> tuple[ResourceClaim, ...]:
    if phase is PhaseId.LOADED:
        selected = (
            estimate for estimate in estimates if estimate.lifetime is DemandLifetime.PERMANENT
        )
    else:
        names = _PHASE_CLAIMS.get(phase, set())
        selected = (estimate for estimate in estimates if estimate.name in names)
    return tuple(
        ResourceClaim(
            name=estimate.name,
            tier=estimate.tier,
            demand_class=estimate.demand_class,
            lifetime=estimate.lifetime,
            amount=estimate.amount,
            unit=estimate.unit,
        )
        for estimate in selected
    )
