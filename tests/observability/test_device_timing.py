from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.observability.device_timing import (
    DeferredDeviceTimer,
    PHASE4_CUDA_ESTIMATOR_V2,
    PHASE4_DEVICE_TIMING_CONTRACT_V2,
    PHASE4_TIMING_SAMPLING_POLICIES,
    Phase4TimingSubstage,
)


class _FakeEvent:
    def __init__(self, owner: "_FakeCuda") -> None:
        self._owner = owner
        self._tick: float | None = None

    def record(self, stream: object) -> None:
        assert stream is self._owner.stream
        self._tick = self._owner.next_tick
        self._owner.next_tick += 2.5
        self._owner.record_count += 1

    def synchronize(self) -> None:
        self._owner.synchronize_count += 1

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert self._tick is not None and other._tick is not None
        return other._tick - self._tick


class _FakeCuda:
    def __init__(self) -> None:
        self.stream = object()
        self.next_tick = 0.0
        self.record_count = 0
        self.synchronize_count = 0

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 3

    def current_stream(self, *, device: torch.device) -> object:
        assert device == torch.device("cuda:3")
        return self.stream

    def get_device_properties(self, ordinal: int) -> SimpleNamespace:
        assert ordinal == 3
        return SimpleNamespace(name="Fake H200", uuid="GPU-fake-uuid")

    def Event(self, *, enable_timing: bool) -> _FakeEvent:
        assert enable_timing
        return _FakeEvent(self)


def test_cuda_ranges_are_block_jitter_sampled_and_defer_one_synchronization() -> None:
    cuda = _FakeCuda()
    timer = DeferredDeviceTimer(torch.device("cuda:3"), cuda_api=cuda)

    for _ in range(34):
        with timer.measure(Phase4TimingSubstage.REFRESH_INFLUENCE_MATMUL):
            pass
    for _ in range(2):
        with timer.measure(Phase4TimingSubstage.EXECUTOR_COMPUTE_BATCH):
            pass

    assert cuda.synchronize_count == 0
    refresh_sample_indices = [
        item.occurrence_index
        for item in timer._samples
        if item.substage == "refresh_influence_matmul"
    ]
    assert refresh_sample_indices == [0, 10, 16, 25, 32, 33]

    summary = timer.resolve()

    assert cuda.synchronize_count == 1
    assert summary.contract_version == PHASE4_DEVICE_TIMING_CONTRACT_V2
    assert summary.backend == "cuda_events_stratified_jitter_deferred_v2"
    assert summary.cuda_visible_device_ordinal == 3
    assert summary.cuda_device_name == "Fake H200"
    assert summary.cuda_device_uuid == "GPU-fake-uuid"
    assert summary.cuda_device_identity_unavailable_reason is None
    assert summary.synchronization_scope == "lifecycle_completion_single_boundary"
    assert summary.cuda_interval_semantics == (
        "captured_current_stream_interval_latency_includes_host_enqueue_gaps_and_stream_waits_v1"
    )
    assert summary.accounting_scope == ("non_overlapping_ranges_complete_strata_cuda_estimate_v2")
    assert (
        summary.cuda_sampling_scheme
        == "deterministic_blake2b_primary_plus_block_start_fallback_per_stride_block_v2"
    )
    assert summary.cuda_sampling_seed == 0x0C17C017
    assert summary.cuda_sampling_hash == "blake2b_64"
    assert summary.cuda_estimator == PHASE4_CUDA_ESTIMATOR_V2
    assert summary.population_count_by_substage == {
        "refresh_influence_matmul": 34,
        "executor_compute_batch": 2,
    }
    assert summary.cuda_sample_count_by_substage == {
        "refresh_influence_matmul": 3,
        "executor_compute_batch": 2,
    }
    assert summary.cuda_recorded_sample_count_by_substage == {
        "refresh_influence_matmul": 6,
        "executor_compute_batch": 2,
    }
    assert summary.cuda_sample_stride_by_substage == {
        "refresh_influence_matmul": 16,
        "executor_compute_batch": 1,
    }
    assert summary.cuda_sampled_elapsed_ms_by_substage == {
        "refresh_influence_matmul": 7.5,
        "executor_compute_batch": 5.0,
    }
    assert summary.cuda_estimated_total_elapsed_ms_by_substage == {
        "refresh_influence_matmul": 85.0,
        "executor_compute_batch": 5.0,
    }
    assert summary.cuda_tail_population_count_by_substage == {
        "refresh_influence_matmul": 2,
        "executor_compute_batch": 0,
    }
    assert summary.cuda_tail_status_by_substage == {
        "refresh_influence_matmul": "sampled",
        "executor_compute_batch": "not_applicable",
    }
    assert summary.cuda_tail_sample_source_by_substage == {
        "refresh_influence_matmul": "hashed_primary",
        "executor_compute_batch": "not_applicable",
    }
    assert summary.cuda_incomplete_tail_sampled_elapsed_ms_by_substage == {
        "refresh_influence_matmul": 2.5,
        "executor_compute_batch": 0.0,
    }
    assert summary.cuda_estimated_accounted_elapsed_ms == 90.0
    assert summary.cuda_event_object_count == 18
    assert summary.cuda_event_record_count == 18
    assert summary.instrumentation_recording_host_overhead_ms >= 0.0
    assert summary.instrumentation_total_host_overhead_ms == (
        summary.instrumentation_recording_host_overhead_ms
    )
    assert summary.resolution_host_elapsed_ms >= 0.0
    attrs = summary.as_attrs(prefix="phase4")
    assert attrs["phase4_timing_refresh_influence_matmul_population_count"] == 34
    assert attrs["phase4_timing_refresh_influence_matmul_cuda_sample_count"] == 3
    assert attrs["phase4_timing_refresh_influence_matmul_cuda_recorded_sample_count"] == 6
    assert attrs["phase4_timing_refresh_influence_matmul_cuda_sample_stride"] == 16
    assert attrs["phase4_timing_refresh_influence_matmul_cuda_estimated_total_elapsed_ms"] == 85.0
    assert attrs["phase4_timing_refresh_influence_matmul_cuda_tail_status"] == ("sampled")
    assert "phase4_timing_refresh_influence_matmul_cuda_event_elapsed_ms" not in attrs
    assert timer.resolve() is summary
    assert cuda.synchronize_count == 1


def test_cpu_timer_preserves_wall_evidence_and_explicit_unavailability() -> None:
    timer = DeferredDeviceTimer(torch.device("cpu"))

    with timer.measure(Phase4TimingSubstage.EXECUTOR_ROW_STORE_WRITE):
        pass

    summary = timer.resolve()
    attrs = summary.as_attrs(prefix="phase4")

    assert summary.backend == "wall_clock_v1"
    assert summary.cuda_sampled_elapsed_ms_by_substage is None
    assert summary.unavailable_reason == "runtime_device_is_not_cuda"
    assert summary.cuda_device_identity_unavailable_reason == "runtime_device_is_not_cuda"
    assert summary.population_count_by_substage == {"executor_row_store_write": 1}
    assert summary.cuda_sample_count_by_substage == {"executor_row_store_write": 0}
    assert summary.cuda_recorded_sample_count_by_substage == {"executor_row_store_write": 0}
    assert summary.cuda_sample_stride_by_substage == {"executor_row_store_write": 1}
    assert attrs["phase4_timing_backend"] == "wall_clock_v1"
    assert attrs["phase4_timing_cuda_sampled_elapsed_ms_by_substage"] is None


def test_incomplete_final_stratum_uses_block_start_fallback() -> None:
    cuda = _FakeCuda()
    timer = DeferredDeviceTimer(torch.device("cuda:3"), cuda_api=cuda)

    for _ in range(33):
        with timer.measure(Phase4TimingSubstage.REFRESH_INFLUENCE_MATMUL):
            pass

    summary = timer.resolve()

    assert summary.cuda_tail_population_count_by_substage == {"refresh_influence_matmul": 1}
    assert summary.cuda_sample_count_by_substage == {"refresh_influence_matmul": 3}
    assert summary.cuda_recorded_sample_count_by_substage == {"refresh_influence_matmul": 5}
    assert summary.cuda_tail_status_by_substage == {"refresh_influence_matmul": "sampled"}
    assert summary.cuda_tail_sample_source_by_substage == {
        "refresh_influence_matmul": "block_start_fallback"
    }
    assert summary.cuda_estimate_status_by_substage == {"refresh_influence_matmul": "complete"}
    assert summary.cuda_estimated_total_elapsed_ms_by_substage == {"refresh_influence_matmul": 82.5}
    assert summary.cuda_estimated_accounted_elapsed_ms == 82.5


@pytest.mark.parametrize(
    "substage",
    [
        Phase4TimingSubstage.REFRESH_INFLUENCE_NORMALIZATION,
        Phase4TimingSubstage.REFRESH_DIRECT_ACCUMULATION,
    ],
)
def test_24101_occurrences_guarantee_tail_sample_with_fallback(
    substage: Phase4TimingSubstage,
) -> None:
    cuda = _FakeCuda()
    timer = DeferredDeviceTimer(torch.device("cuda:3"), cuda_api=cuda)

    for _ in range(24_101):
        with timer.measure(substage):
            pass

    summary = timer.resolve()
    name = substage.value

    assert summary.cuda_sample_count_by_substage[name] == 1_507
    assert 1_507 < summary.cuda_recorded_sample_count_by_substage[name] <= 3_014
    assert summary.cuda_tail_population_count_by_substage[name] == 5
    assert summary.cuda_tail_status_by_substage[name] == "sampled"
    assert summary.cuda_tail_sample_source_by_substage[name] == ("block_start_fallback")
    assert summary.cuda_estimate_status_by_substage[name] == "complete"
    assert summary.cuda_sampled_elapsed_ms_by_substage[name] == 3_767.5
    assert summary.cuda_incomplete_tail_sampled_elapsed_ms_by_substage[name] == 2.5
    assert summary.cuda_estimated_total_elapsed_ms_by_substage[name] == 60_252.5


def test_callers_cannot_override_observability_owned_sampling_policy() -> None:
    timer = DeferredDeviceTimer(torch.device("cpu"))

    assert (
        PHASE4_TIMING_SAMPLING_POLICIES[
            Phase4TimingSubstage.REFRESH_ROW_STORE_READ
        ].cuda_sample_stride
        == 16
    )
    try:
        timer.measure("refresh_row_store_read")  # type: ignore[arg-type]
    except TypeError as error:
        assert "Phase4TimingSubstage" in str(error)
    else:
        raise AssertionError("expected an untyped substage to fail")
