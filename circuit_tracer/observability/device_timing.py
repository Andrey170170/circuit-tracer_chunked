"""Deferred, sampled device timing for Phase-4 lifecycle evidence.

This module owns the Phase-4 substage vocabulary, sampling policies, CUDA event
construction, and the emitted timing contract. Callers open typed spans without
choosing telemetry names or sample strides.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from types import MappingProxyType
import time
from typing import Any, Final, Mapping

import torch


PHASE4_DEVICE_TIMING_CONTRACT_V1: Final = "phase4_device_timing_v1"
PHASE4_DEVICE_TIMING_CONTRACT_V2: Final = "phase4_device_timing_v2"
PHASE4_DEVICE_TIMING_BACKEND_V2: Final = "cuda_events_stratified_jitter_deferred_v2"
PHASE4_CUDA_SAMPLING_SCHEME_V2: Final = (
    "deterministic_blake2b_primary_plus_block_start_fallback_per_stride_block_v2"
)
PHASE4_CUDA_SAMPLING_HASH: Final = "blake2b_64"
PHASE4_CUDA_ESTIMATOR_V2: Final = "stratified_actual_block_size_weight_v2"
PHASE4_CUDA_INTERVAL_SEMANTICS: Final = (
    "captured_current_stream_interval_latency_includes_host_enqueue_gaps_and_stream_waits_v1"
)
PHASE4_CUDA_ACCOUNTING_SCOPE_V2: Final = "non_overlapping_ranges_complete_strata_cuda_estimate_v2"
PHASE4_CUDA_SAMPLING_SEED: Final = 0x0C17C017


class Phase4TimingSubstage(StrEnum):
    """Observability-owned names for non-overlapping Phase-4 timing spans."""

    REFRESH_ROW_STORE_READ = "refresh_row_store_read"
    REFRESH_INFLUENCE_NORMALIZATION = "refresh_influence_normalization"
    REFRESH_INFLUENCE_MATMUL = "refresh_influence_matmul"
    REFRESH_DIRECT_ACCUMULATION = "refresh_direct_accumulation"
    REFRESH_TRANSFER_CAST_ABS = "refresh_transfer_cast_abs"
    EXECUTOR_ENCODER_MATERIALIZE = "executor_encoder_materialize"
    EXECUTOR_COMPUTE_BATCH = "executor_compute_batch"
    EXECUTOR_CPU_STAGING = "executor_cpu_staging"
    EXECUTOR_DENOMINATOR = "executor_denominator"
    EXECUTOR_ROW_STORE_WRITE = "executor_row_store_write"


@dataclass(frozen=True)
class Phase4TimingSamplingPolicy:
    cuda_sample_stride: int
    qualification_required: bool = True


PHASE4_TIMING_SAMPLING_POLICIES: Mapping[Phase4TimingSubstage, Phase4TimingSamplingPolicy] = (
    MappingProxyType(
        {
            Phase4TimingSubstage.REFRESH_ROW_STORE_READ: Phase4TimingSamplingPolicy(16),
            Phase4TimingSubstage.REFRESH_INFLUENCE_NORMALIZATION: Phase4TimingSamplingPolicy(16),
            Phase4TimingSubstage.REFRESH_INFLUENCE_MATMUL: Phase4TimingSamplingPolicy(
                16, qualification_required=False
            ),
            Phase4TimingSubstage.REFRESH_DIRECT_ACCUMULATION: Phase4TimingSamplingPolicy(16),
            Phase4TimingSubstage.REFRESH_TRANSFER_CAST_ABS: Phase4TimingSamplingPolicy(
                16, qualification_required=False
            ),
            Phase4TimingSubstage.EXECUTOR_ENCODER_MATERIALIZE: Phase4TimingSamplingPolicy(1),
            Phase4TimingSubstage.EXECUTOR_COMPUTE_BATCH: Phase4TimingSamplingPolicy(1),
            Phase4TimingSubstage.EXECUTOR_CPU_STAGING: Phase4TimingSamplingPolicy(1),
            Phase4TimingSubstage.EXECUTOR_DENOMINATOR: Phase4TimingSamplingPolicy(1),
            Phase4TimingSubstage.EXECUTOR_ROW_STORE_WRITE: Phase4TimingSamplingPolicy(1),
        }
    )
)
PHASE4_REQUIRED_TIMING_SUBSTAGES: Final = tuple(
    substage
    for substage, policy in PHASE4_TIMING_SAMPLING_POLICIES.items()
    if policy.qualification_required
)
PHASE4_OPTIONAL_TIMING_SUBSTAGES: Final = tuple(
    substage
    for substage, policy in PHASE4_TIMING_SAMPLING_POLICIES.items()
    if not policy.qualification_required
)


@dataclass(frozen=True)
class DeviceTimingSummary:
    """Resolved wall evidence and complete-strata CUDA estimates."""

    contract_version: str
    backend: str
    device: str
    cuda_visible_device_ordinal: int | None
    cuda_device_name: str | None
    cuda_device_uuid: str | None
    cuda_device_identity_unavailable_reason: str | None
    synchronization_scope: str
    stream_scope: str
    cuda_interval_semantics: str
    accounting_scope: str
    cuda_sampling_scheme: str
    cuda_sampling_seed: int
    cuda_sampling_hash: str
    cuda_estimator: str
    unavailable_reason: str | None
    population_count_by_substage: dict[str, int]
    cuda_sample_count_by_substage: dict[str, int]
    cuda_recorded_sample_count_by_substage: dict[str, int]
    cuda_sample_stride_by_substage: dict[str, int]
    cuda_tail_population_count_by_substage: dict[str, int]
    cuda_tail_status_by_substage: dict[str, str]
    cuda_tail_sample_source_by_substage: dict[str, str]
    cuda_estimate_status_by_substage: dict[str, str]
    wall_elapsed_ms_by_substage: dict[str, float]
    cuda_sampled_elapsed_ms_by_substage: dict[str, float] | None
    cuda_complete_block_sampled_elapsed_ms_by_substage: dict[str, float] | None
    cuda_incomplete_tail_sampled_elapsed_ms_by_substage: dict[str, float] | None
    cuda_estimated_total_elapsed_ms_by_substage: dict[str, float | None] | None
    lifecycle_wall_elapsed_ms: float
    lifecycle_cuda_event_elapsed_ms: float | None
    wall_accounted_elapsed_ms: float
    wall_residual_elapsed_ms: float
    cuda_estimated_accounted_elapsed_ms: float | None
    cuda_estimated_residual_elapsed_ms: float | None
    cuda_event_object_count: int
    cuda_event_record_count: int
    instrumentation_overhead_scope: str
    instrumentation_recording_host_overhead_ms: float
    instrumentation_total_host_overhead_ms: float
    resolution_host_elapsed_ms: float

    def as_attrs(self, *, prefix: str) -> dict[str, object]:
        """Flatten the v2 timing contract into telemetry-safe attributes."""
        attrs: dict[str, object] = {
            f"{prefix}_timing_contract_version": self.contract_version,
            f"{prefix}_timing_backend": self.backend,
            f"{prefix}_timing_device": self.device,
            f"{prefix}_timing_cuda_visible_device_ordinal": self.cuda_visible_device_ordinal,
            f"{prefix}_timing_cuda_device_name": self.cuda_device_name,
            f"{prefix}_timing_cuda_device_uuid": self.cuda_device_uuid,
            f"{prefix}_timing_cuda_device_identity_unavailable_reason": self.cuda_device_identity_unavailable_reason,
            f"{prefix}_timing_synchronization_scope": self.synchronization_scope,
            f"{prefix}_timing_stream_scope": self.stream_scope,
            f"{prefix}_timing_cuda_interval_semantics": self.cuda_interval_semantics,
            f"{prefix}_timing_accounting_scope": self.accounting_scope,
            f"{prefix}_timing_cuda_sampling_scheme": self.cuda_sampling_scheme,
            f"{prefix}_timing_cuda_sampling_seed": self.cuda_sampling_seed,
            f"{prefix}_timing_cuda_sampling_hash": self.cuda_sampling_hash,
            f"{prefix}_timing_cuda_estimator": self.cuda_estimator,
            f"{prefix}_timing_unavailable_reason": self.unavailable_reason,
            f"{prefix}_timing_population_count_by_substage": dict(
                self.population_count_by_substage
            ),
            f"{prefix}_timing_cuda_sample_count_by_substage": dict(
                self.cuda_sample_count_by_substage
            ),
            f"{prefix}_timing_cuda_recorded_sample_count_by_substage": dict(
                self.cuda_recorded_sample_count_by_substage
            ),
            f"{prefix}_timing_cuda_sample_stride_by_substage": dict(
                self.cuda_sample_stride_by_substage
            ),
            f"{prefix}_timing_cuda_tail_population_count_by_substage": dict(
                self.cuda_tail_population_count_by_substage
            ),
            f"{prefix}_timing_cuda_tail_status_by_substage": dict(
                self.cuda_tail_status_by_substage
            ),
            f"{prefix}_timing_cuda_tail_sample_source_by_substage": dict(
                self.cuda_tail_sample_source_by_substage
            ),
            f"{prefix}_timing_cuda_estimate_status_by_substage": dict(
                self.cuda_estimate_status_by_substage
            ),
            f"{prefix}_timing_wall_elapsed_ms_by_substage": dict(self.wall_elapsed_ms_by_substage),
            f"{prefix}_timing_cuda_sampled_elapsed_ms_by_substage": None
            if self.cuda_sampled_elapsed_ms_by_substage is None
            else dict(self.cuda_sampled_elapsed_ms_by_substage),
            f"{prefix}_timing_cuda_complete_block_sampled_elapsed_ms_by_substage": None
            if self.cuda_complete_block_sampled_elapsed_ms_by_substage is None
            else dict(self.cuda_complete_block_sampled_elapsed_ms_by_substage),
            f"{prefix}_timing_cuda_incomplete_tail_sampled_elapsed_ms_by_substage": None
            if self.cuda_incomplete_tail_sampled_elapsed_ms_by_substage is None
            else dict(self.cuda_incomplete_tail_sampled_elapsed_ms_by_substage),
            f"{prefix}_timing_cuda_estimated_total_elapsed_ms_by_substage": None
            if self.cuda_estimated_total_elapsed_ms_by_substage is None
            else dict(self.cuda_estimated_total_elapsed_ms_by_substage),
            f"{prefix}_timing_lifecycle_wall_elapsed_ms": self.lifecycle_wall_elapsed_ms,
            f"{prefix}_timing_lifecycle_cuda_event_elapsed_ms": self.lifecycle_cuda_event_elapsed_ms,
            f"{prefix}_timing_wall_accounted_elapsed_ms": self.wall_accounted_elapsed_ms,
            f"{prefix}_timing_wall_residual_elapsed_ms": self.wall_residual_elapsed_ms,
            f"{prefix}_timing_cuda_estimated_accounted_elapsed_ms": self.cuda_estimated_accounted_elapsed_ms,
            f"{prefix}_timing_cuda_estimated_residual_elapsed_ms": self.cuda_estimated_residual_elapsed_ms,
            f"{prefix}_timing_cuda_event_object_count": self.cuda_event_object_count,
            f"{prefix}_timing_cuda_event_record_count": self.cuda_event_record_count,
            f"{prefix}_timing_instrumentation_overhead_scope": self.instrumentation_overhead_scope,
            f"{prefix}_timing_instrumentation_recording_host_overhead_ms": self.instrumentation_recording_host_overhead_ms,
            f"{prefix}_timing_instrumentation_total_host_overhead_ms": self.instrumentation_total_host_overhead_ms,
            f"{prefix}_timing_resolution_host_elapsed_ms": self.resolution_host_elapsed_ms,
        }
        sampled = self.cuda_sampled_elapsed_ms_by_substage or {}
        complete = self.cuda_complete_block_sampled_elapsed_ms_by_substage or {}
        tail = self.cuda_incomplete_tail_sampled_elapsed_ms_by_substage or {}
        estimated = self.cuda_estimated_total_elapsed_ms_by_substage or {}
        for substage, population in self.population_count_by_substage.items():
            field = f"{prefix}_timing_{substage}"
            attrs[f"{field}_population_count"] = population
            attrs[f"{field}_cuda_sample_count"] = self.cuda_sample_count_by_substage.get(
                substage, 0
            )
            attrs[f"{field}_cuda_recorded_sample_count"] = (
                self.cuda_recorded_sample_count_by_substage.get(substage, 0)
            )
            attrs[f"{field}_cuda_sample_stride"] = self.cuda_sample_stride_by_substage[substage]
            attrs[f"{field}_cuda_tail_population_count"] = (
                self.cuda_tail_population_count_by_substage[substage]
            )
            attrs[f"{field}_cuda_tail_status"] = self.cuda_tail_status_by_substage[substage]
            attrs[f"{field}_cuda_tail_sample_source"] = self.cuda_tail_sample_source_by_substage[
                substage
            ]
            attrs[f"{field}_cuda_estimate_status"] = self.cuda_estimate_status_by_substage[substage]
            attrs[f"{field}_wall_elapsed_ms"] = self.wall_elapsed_ms_by_substage[substage]
            attrs[f"{field}_cuda_sampled_elapsed_ms"] = sampled.get(substage)
            attrs[f"{field}_cuda_complete_block_sampled_elapsed_ms"] = complete.get(substage)
            attrs[f"{field}_cuda_incomplete_tail_sampled_elapsed_ms"] = tail.get(substage)
            attrs[f"{field}_cuda_estimated_total_elapsed_ms"] = estimated.get(substage)
        return attrs


@dataclass
class _PendingCudaSample:
    substage: str
    occurrence_index: int
    start_event: Any
    end_event: Any


class DeviceTimingRange(AbstractContextManager["DeviceTimingRange"]):
    def __init__(self, owner: "DeferredDeviceTimer", substage: Phase4TimingSubstage) -> None:
        self._owner = owner
        self.substage = substage
        self.cuda_sample_stride = PHASE4_TIMING_SAMPLING_POLICIES[substage].cuda_sample_stride
        self.wall_elapsed_ms = 0.0
        self._wall_started = 0.0
        self._sample_cuda = False
        self._occurrence_index = 0
        self._start_event: Any | None = None
        self._end_event: Any | None = None

    def __enter__(self) -> "DeviceTimingRange":
        self._occurrence_index, self._sample_cuda = self._owner._register_occurrence(
            self.substage.value, self.cuda_sample_stride
        )
        if self._sample_cuda:
            self._start_event, self._end_event = self._owner._start_cuda_range()
        self._wall_started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.wall_elapsed_ms = (time.perf_counter() - self._wall_started) * 1000.0
        if self._sample_cuda:
            self._owner._finish_cuda_range(self._end_event)
        self._owner._finish_occurrence(
            self.substage.value,
            self.wall_elapsed_ms,
            self._occurrence_index,
            self._start_event,
            self._end_event,
        )
        return False


class DeferredDeviceTimer:
    """Collect wall ranges and one deterministic CUDA sample per stride block."""

    def __init__(
        self,
        device: torch.device | str | None,
        *,
        cuda_api: Any = None,
        synchronization_scope: str = "lifecycle_completion_single_boundary",
        cuda_sampling_seed: int = PHASE4_CUDA_SAMPLING_SEED,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self._cuda = torch.cuda if cuda_api is None else cuda_api
        self.synchronization_scope = synchronization_scope
        self.cuda_sampling_seed = int(cuda_sampling_seed)
        self.stream_scope = "torch_cuda_current_stream_for_device"
        self._samples: list[_PendingCudaSample] = []
        self._population_counts: defaultdict[str, int] = defaultdict(int)
        self._recorded_sample_counts: defaultdict[str, int] = defaultdict(int)
        self._sample_strides: dict[str, int] = {}
        self._wall_totals: defaultdict[str, float] = defaultdict(float)
        self._wall_started = time.perf_counter()
        self._resolved: DeviceTimingSummary | None = None
        self._unavailable_reason: str | None = None
        self._lifecycle_start_event: Any | None = None
        self._lifecycle_end_event: Any | None = None
        self._stream: Any | None = None
        self._visible_device_ordinal: int | None = None
        self._device_name: str | None = None
        self._device_uuid: str | None = None
        self._device_identity_unavailable_reason: str | None = None
        self._event_object_count = 0
        self._event_record_count = 0
        self._recording_host_overhead_ms = 0.0
        self._resolution_host_overhead_ms = 0.0
        self._cuda_enabled = self._initialize_cuda_events()

    @property
    def backend(self) -> str:
        return PHASE4_DEVICE_TIMING_BACKEND_V2 if self._cuda_enabled else "wall_clock_v1"

    def measure(self, substage: Phase4TimingSubstage) -> DeviceTimingRange:
        if self._resolved is not None:
            raise RuntimeError("cannot add a timing range after resolution")
        if not isinstance(substage, Phase4TimingSubstage):
            raise TypeError("timing substage must be a Phase4TimingSubstage")
        return DeviceTimingRange(self, substage)

    def resolve(self) -> DeviceTimingSummary:
        if self._resolved is not None:
            return self._resolved
        lifecycle_wall = (time.perf_counter() - self._wall_started) * 1000.0
        sampled: dict[str, float] | None = None
        complete_sampled: dict[str, float] | None = None
        tail_sampled: dict[str, float] | None = None
        estimated: dict[str, float | None] | None = None
        tail_population = {
            name: population % self._sample_strides[name]
            for name, population in self._population_counts.items()
        }
        tail_status: dict[str, str] = {}
        tail_sample_source: dict[str, str] = {}
        estimate_status: dict[str, str] = {}
        used_sample_counts: dict[str, int] = {}
        lifecycle_cuda: float | None = None
        if self._cuda_enabled:
            resolution_started = time.perf_counter()
            try:
                self._lifecycle_end_event = self._new_event()
                self._record_event(self._lifecycle_end_event)
                self._lifecycle_end_event.synchronize()
                recorded_by_name_and_occurrence: defaultdict[
                    str, dict[int, tuple[_PendingCudaSample, float]]
                ] = defaultdict(dict)
                for item in self._samples:
                    elapsed = float(item.start_event.elapsed_time(item.end_event))
                    recorded_by_name_and_occurrence[item.substage][item.occurrence_index] = (
                        item,
                        elapsed,
                    )
                sampled_totals: defaultdict[str, float] = defaultdict(float)
                complete_totals: defaultdict[str, float] = defaultdict(float)
                tail_totals: defaultdict[str, float] = defaultdict(float)
                estimated = {}
                for name, population in self._population_counts.items():
                    stride = self._sample_strides[name]
                    complete_blocks, tail_size = divmod(population, stride)
                    block_count = complete_blocks + int(tail_size > 0)
                    used_sample_counts[name] = 0
                    estimate_status[name] = "complete"
                    tail_status[name] = "sampled" if tail_size else "not_applicable"
                    tail_sample_source[name] = "not_applicable"
                    for block in range(block_count):
                        block_start = block * stride
                        hashed_offset = self._sample_block_offset(name, block, stride)
                        primary_occurrence = block_start + hashed_offset
                        selected = recorded_by_name_and_occurrence[name].get(primary_occurrence)
                        is_tail = block == complete_blocks and tail_size > 0
                        if selected is None and is_tail:
                            selected = recorded_by_name_and_occurrence[name].get(block_start)
                            if selected is not None:
                                tail_sample_source[name] = "block_start_fallback"
                        elif selected is not None and is_tail:
                            tail_sample_source[name] = "hashed_primary"
                        if selected is None:
                            estimate_status[name] = (
                                "refused_incomplete_tail"
                                if is_tail
                                else "refused_incomplete_complete_blocks"
                            )
                            if is_tail:
                                tail_status[name] = "unsampled"
                                tail_sample_source[name] = "missing"
                            break
                        _, elapsed = selected
                        used_sample_counts[name] += 1
                        sampled_totals[name] += elapsed
                        if is_tail:
                            tail_totals[name] += elapsed
                        else:
                            complete_totals[name] += elapsed
                    if estimate_status[name] != "complete":
                        estimated[name] = None
                    else:
                        estimated[name] = (
                            complete_totals[name] * stride + tail_totals[name] * tail_size
                        )
                sampled = {name: float(sampled_totals[name]) for name in self._population_counts}
                complete_sampled = {
                    name: float(complete_totals[name]) for name in self._population_counts
                }
                tail_sampled = {name: float(tail_totals[name]) for name in self._population_counts}
                assert self._lifecycle_start_event is not None
                lifecycle_cuda = float(
                    self._lifecycle_start_event.elapsed_time(self._lifecycle_end_event)
                )
            except Exception as error:
                self._disable_cuda(f"cuda_event_resolution_failed:{type(error).__name__}")
                sampled = complete_sampled = tail_sampled = estimated = None
                lifecycle_cuda = None
            finally:
                self._resolution_host_overhead_ms += (
                    time.perf_counter() - resolution_started
                ) * 1000.0
        if not self._cuda_enabled:
            for name, tail_size in tail_population.items():
                tail_status[name] = "not_applicable" if tail_size == 0 else "unavailable"
                tail_sample_source[name] = "unavailable"
                estimate_status[name] = "unavailable"
                used_sample_counts[name] = 0
        wall_accounted = float(sum(self._wall_totals.values()))
        estimates_complete = estimated is not None and all(
            status == "complete" for status in estimate_status.values()
        )
        cuda_accounted = (
            float(sum(value for value in estimated.values() if value is not None))
            if estimates_complete and estimated is not None
            else None
        )
        self._resolved = DeviceTimingSummary(
            contract_version=PHASE4_DEVICE_TIMING_CONTRACT_V2,
            backend=self.backend,
            device=str(self.device),
            cuda_visible_device_ordinal=self._visible_device_ordinal,
            cuda_device_name=self._device_name,
            cuda_device_uuid=self._device_uuid,
            cuda_device_identity_unavailable_reason=self._device_identity_unavailable_reason,
            synchronization_scope=self.synchronization_scope,
            stream_scope=self.stream_scope,
            cuda_interval_semantics=PHASE4_CUDA_INTERVAL_SEMANTICS,
            accounting_scope=PHASE4_CUDA_ACCOUNTING_SCOPE_V2,
            cuda_sampling_scheme=PHASE4_CUDA_SAMPLING_SCHEME_V2,
            cuda_sampling_seed=self.cuda_sampling_seed,
            cuda_sampling_hash=PHASE4_CUDA_SAMPLING_HASH,
            cuda_estimator=PHASE4_CUDA_ESTIMATOR_V2,
            unavailable_reason=self._unavailable_reason,
            population_count_by_substage=dict(self._population_counts),
            cuda_sample_count_by_substage={
                name: used_sample_counts.get(name, 0) for name in self._population_counts
            },
            cuda_recorded_sample_count_by_substage={
                name: self._recorded_sample_counts.get(name, 0) for name in self._population_counts
            },
            cuda_sample_stride_by_substage=dict(self._sample_strides),
            cuda_tail_population_count_by_substage=tail_population,
            cuda_tail_status_by_substage=tail_status,
            cuda_tail_sample_source_by_substage=tail_sample_source,
            cuda_estimate_status_by_substage=estimate_status,
            wall_elapsed_ms_by_substage=dict(self._wall_totals),
            cuda_sampled_elapsed_ms_by_substage=sampled,
            cuda_complete_block_sampled_elapsed_ms_by_substage=complete_sampled,
            cuda_incomplete_tail_sampled_elapsed_ms_by_substage=tail_sampled,
            cuda_estimated_total_elapsed_ms_by_substage=estimated,
            lifecycle_wall_elapsed_ms=float(lifecycle_wall),
            lifecycle_cuda_event_elapsed_ms=lifecycle_cuda,
            wall_accounted_elapsed_ms=wall_accounted,
            wall_residual_elapsed_ms=float(lifecycle_wall - wall_accounted),
            cuda_estimated_accounted_elapsed_ms=cuda_accounted,
            cuda_estimated_residual_elapsed_ms=None
            if lifecycle_cuda is None or cuda_accounted is None
            else float(lifecycle_cuda - cuda_accounted),
            cuda_event_object_count=self._event_object_count,
            cuda_event_record_count=self._event_record_count,
            instrumentation_overhead_scope="sampling_selection_event_record_and_bookkeeping_excludes_resolution_sync_v2",
            instrumentation_recording_host_overhead_ms=self._recording_host_overhead_ms,
            instrumentation_total_host_overhead_ms=self._recording_host_overhead_ms,
            resolution_host_elapsed_ms=self._resolution_host_overhead_ms,
        )
        return self._resolved

    def _register_occurrence(self, substage: str, stride: int) -> tuple[int, bool]:
        started = time.perf_counter()
        prior = self._sample_strides.setdefault(substage, stride)
        if prior != stride:
            raise ValueError(
                f"timing substage {substage!r} used inconsistent CUDA sample strides ({prior} and {stride})"
            )
        occurrence = self._population_counts[substage]
        self._population_counts[substage] += 1
        block = occurrence // stride
        offset = self._sample_block_offset(substage, block, stride)
        offset_in_block = occurrence % stride
        self._recording_host_overhead_ms += (time.perf_counter() - started) * 1000.0
        return occurrence, self._cuda_enabled and offset_in_block in {0, offset}

    def _finish_occurrence(
        self,
        substage: str,
        wall_elapsed: float,
        occurrence: int,
        start: Any | None,
        end: Any | None,
    ) -> None:
        started = time.perf_counter()
        self._wall_totals[substage] += float(wall_elapsed)
        if start is not None and end is not None:
            self._samples.append(_PendingCudaSample(substage, occurrence, start, end))
            self._recorded_sample_counts[substage] += 1
        self._recording_host_overhead_ms += (time.perf_counter() - started) * 1000.0

    def _sample_block_offset(self, substage: str, block: int, stride: int) -> int:
        payload = f"{self.cuda_sampling_seed}:{substage}:{block}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big") % stride

    def _initialize_cuda_events(self) -> bool:
        if self.device.type != "cuda":
            self._unavailable_reason = "runtime_device_is_not_cuda"
            self._device_identity_unavailable_reason = "runtime_device_is_not_cuda"
            return False
        started = time.perf_counter()
        try:
            if not bool(self._cuda.is_available()):
                self._unavailable_reason = "torch_cuda_is_unavailable"
                self._device_identity_unavailable_reason = "torch_cuda_is_unavailable"
                return False
            self._visible_device_ordinal = (
                int(self.device.index)
                if self.device.index is not None
                else int(self._cuda.current_device())
            )
            properties = self._cuda.get_device_properties(self._visible_device_ordinal)
            self._device_name = (
                None if getattr(properties, "name", None) is None else str(properties.name)
            )
            self._device_uuid = (
                None if getattr(properties, "uuid", None) is None else str(properties.uuid)
            )
            if self._device_uuid is None:
                self._device_identity_unavailable_reason = "cuda_device_uuid_unavailable"
            self._stream = self._cuda.current_stream(device=self.device)
            self._lifecycle_start_event = self._new_event()
            self._record_event(self._lifecycle_start_event)
            return True
        except Exception as error:
            self._unavailable_reason = f"cuda_event_initialization_failed:{type(error).__name__}"
            self._device_identity_unavailable_reason = (
                f"cuda_device_identity_failed:{type(error).__name__}"
            )
            return False
        finally:
            self._recording_host_overhead_ms += (time.perf_counter() - started) * 1000.0

    def _start_cuda_range(self) -> tuple[Any | None, Any | None]:
        if not self._cuda_enabled:
            return None, None
        started = time.perf_counter()
        try:
            start, end = self._new_event(), self._new_event()
            self._record_event(start)
            return start, end
        except Exception as error:
            self._disable_cuda(f"cuda_event_record_failed:{type(error).__name__}")
            return None, None
        finally:
            self._recording_host_overhead_ms += (time.perf_counter() - started) * 1000.0

    def _finish_cuda_range(self, end: Any | None) -> None:
        if not self._cuda_enabled or end is None:
            return
        started = time.perf_counter()
        try:
            self._record_event(end)
        except Exception as error:
            self._disable_cuda(f"cuda_event_record_failed:{type(error).__name__}")
        finally:
            self._recording_host_overhead_ms += (time.perf_counter() - started) * 1000.0

    def _new_event(self) -> Any:
        event = self._cuda.Event(enable_timing=True)
        self._event_object_count += 1
        return event

    def _record_event(self, event: Any) -> None:
        event.record(self._stream)
        self._event_record_count += 1

    def _disable_cuda(self, reason: str) -> None:
        self._cuda_enabled = False
        if self._unavailable_reason is None:
            self._unavailable_reason = reason


__all__ = [
    "DeferredDeviceTimer",
    "DeviceTimingRange",
    "DeviceTimingSummary",
    "PHASE4_CUDA_ACCOUNTING_SCOPE_V2",
    "PHASE4_CUDA_ESTIMATOR_V2",
    "PHASE4_CUDA_INTERVAL_SEMANTICS",
    "PHASE4_CUDA_SAMPLING_HASH",
    "PHASE4_CUDA_SAMPLING_SCHEME_V2",
    "PHASE4_CUDA_SAMPLING_SEED",
    "PHASE4_DEVICE_TIMING_BACKEND_V2",
    "PHASE4_DEVICE_TIMING_CONTRACT_V1",
    "PHASE4_DEVICE_TIMING_CONTRACT_V2",
    "PHASE4_OPTIONAL_TIMING_SUBSTAGES",
    "PHASE4_REQUIRED_TIMING_SUBSTAGES",
    "PHASE4_TIMING_SAMPLING_POLICIES",
    "Phase4TimingSamplingPolicy",
    "Phase4TimingSubstage",
]
