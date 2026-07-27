"""Exact Phase-0 decoder row-range planning and safetensor loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class DecoderRowRange:
    """One half-open safetensor row range and its requested feature IDs."""

    start: int
    stop: int
    feature_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("decoder row range must be a non-empty half-open interval")
        if self.feature_ids.device.type != "cpu":
            raise ValueError("decoder row range feature ids must be CPU staged")
        if self.feature_ids.ndim != 1 or self.feature_ids.dtype != torch.long:
            raise ValueError("decoder row range feature ids must be rank-1 long")
        if self.feature_ids.numel() == 0:
            raise ValueError("decoder row range must contain a requested row")
        if int(self.feature_ids[0]) < self.start or int(self.feature_ids[-1]) >= self.stop:
            raise ValueError("requested decoder rows must lie inside their range")

    @property
    def materialized_rows(self) -> int:
        return self.stop - self.start

    @property
    def requested_rows(self) -> int:
        return int(self.feature_ids.numel())


@dataclass(frozen=True)
class Phase0DecoderRangeTelemetry:
    """Typed evidence for exact selective Phase-0 decoder loading."""

    requested: bool
    effective: bool
    fallback_reason: str | None
    planning_seconds: float
    read_seconds: float
    gather_seconds: float
    reconstruction_seconds: float
    seed_capture_seconds: float
    unique_row_count: int
    unique_row_bytes: int
    range_request_count: int
    range_rows: tuple[int, ...]
    merged_gap_rows: int
    overfetch_bytes: int
    logical_requested_bytes: int
    logical_materialized_bytes: int
    baseline_full_page_count: int
    baseline_full_page_bytes: int
    backend: str = "coalesced_ranges"
    occurrence_row_count: int = 0
    mapping_count: int = 0
    block_count: int = 0
    read_count: int = 0
    backend_request_count: int = 0
    page_span_bytes: int = 0
    backend_requested_bytes: int = 0
    backend_materialized_bytes: int = 0
    planned_overfetch_ratio: float = 0.0
    physical_read_estimate_bytes: int | None = None
    fault_read_seconds: float = 0.0
    reorder_seconds: float = 0.0
    h2d_seconds: float = 0.0
    total_seconds: float = 0.0
    occurrence_row_bytes: int = 0
    mapping_open_count: int = 0
    range_count: int = 0
    output_bytes: int = 0
    temporary_staging_high_water_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "fallback_reason": self.fallback_reason,
            "planning_seconds": self.planning_seconds,
            "read_seconds": self.read_seconds,
            "gather_seconds": self.gather_seconds,
            "reconstruction_seconds": self.reconstruction_seconds,
            "seed_capture_seconds": self.seed_capture_seconds,
            "unique_row_count": self.unique_row_count,
            "unique_row_bytes": self.unique_row_bytes,
            "range_request_count": self.range_request_count,
            "range_rows": self.range_rows,
            "merged_gap_rows": self.merged_gap_rows,
            "overfetch_bytes": self.overfetch_bytes,
            "logical_requested_bytes": self.logical_requested_bytes,
            "logical_materialized_bytes": self.logical_materialized_bytes,
            "baseline_full_page_count": self.baseline_full_page_count,
            "baseline_full_page_bytes": self.baseline_full_page_bytes,
            "backend": self.backend,
            "occurrence_row_count": self.occurrence_row_count,
            "mapping_count": self.mapping_count,
            "block_count": self.block_count,
            "read_count": self.read_count,
            "backend_request_count": self.backend_request_count,
            "page_span_bytes": self.page_span_bytes,
            "backend_requested_bytes": self.backend_requested_bytes,
            "backend_materialized_bytes": self.backend_materialized_bytes,
            "planned_overfetch_ratio": self.planned_overfetch_ratio,
            "physical_read_estimate_bytes": self.physical_read_estimate_bytes,
            "fault_read_seconds": self.fault_read_seconds,
            "reorder_seconds": self.reorder_seconds,
            "h2d_seconds": self.h2d_seconds,
            "total_seconds": self.total_seconds,
            "occurrence_row_bytes": self.occurrence_row_bytes,
            "mapping_open_count": self.mapping_open_count,
            "range_count": self.range_count,
            "output_bytes": self.output_bytes,
            "temporary_staging_high_water_bytes": (
                self.temporary_staging_high_water_bytes
            ),
        }


@dataclass(frozen=True)
class DecoderRowRangePlan:
    """Admitted coalesced ranges or an explicit full-page fallback."""

    ranges: tuple[DecoderRowRange, ...]
    unique_feature_ids: torch.Tensor
    planning_seconds: float
    fallback_reason: str | None
    merged_gap_rows: int
    baseline_full_page_count: int
    baseline_full_page_bytes: int

    @property
    def admitted(self) -> bool:
        return self.fallback_reason is None


def plan_decoder_row_ranges(
    feature_ids: torch.Tensor,
    *,
    d_model: int,
    d_transcoder: int,
    itemsize: int,
    decoder_chunk_size: int,
    max_gap_rows: int,
    max_overfetch_fraction: float,
    max_range_count: int,
    max_singleton_range_fraction: float,
    max_ranges_per_baseline_page: int,
) -> DecoderRowRangePlan:
    """Coalesce sorted unique IDs without issuing any checkpoint reads."""

    started = time.perf_counter()
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if d_transcoder <= 0:
        raise ValueError("d_transcoder must be positive")
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    if decoder_chunk_size <= 0:
        raise ValueError("decoder_chunk_size must be positive")
    if max_gap_rows < 0:
        raise ValueError("max_gap_rows must be nonnegative")
    if max_overfetch_fraction < 0:
        raise ValueError("max_overfetch_fraction must be nonnegative")
    if max_range_count <= 0:
        raise ValueError("max_range_count must be positive")
    if not 0 <= max_singleton_range_fraction <= 1:
        raise ValueError("max_singleton_range_fraction must lie in [0, 1]")
    if max_ranges_per_baseline_page <= 0:
        raise ValueError("max_ranges_per_baseline_page must be positive")
    unique_ids = torch.unique(
        feature_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1),
        sorted=True,
    ).contiguous()
    if unique_ids.numel() and (
        int(unique_ids[0]) < 0 or int(unique_ids[-1]) >= d_transcoder
    ):
        raise IndexError("decoder feature ids must lie inside the transcoder width")

    chunk_ids = torch.div(unique_ids, decoder_chunk_size, rounding_mode="floor")
    baseline_pages = int(torch.unique(chunk_ids).numel())
    baseline_bytes = baseline_pages * decoder_chunk_size * d_model * itemsize
    if unique_ids.numel():
        last_chunk = int(chunk_ids[-1])
        last_page_rows = min(
            decoder_chunk_size,
            d_transcoder - last_chunk * decoder_chunk_size,
        )
        baseline_bytes -= (decoder_chunk_size - last_page_rows) * d_model * itemsize

    if unique_ids.numel() == 0:
        return DecoderRowRangePlan(
            ranges=(),
            unique_feature_ids=unique_ids,
            planning_seconds=time.perf_counter() - started,
            fallback_reason=None,
            merged_gap_rows=0,
            baseline_full_page_count=0,
            baseline_full_page_bytes=0,
        )

    ids = unique_ids.tolist()
    groups: list[tuple[int, int, list[int]]] = []
    start = previous = int(ids[0])
    group_ids = [start]
    merged_gap_rows = 0
    for raw_id in ids[1:]:
        feature_id = int(raw_id)
        gap = feature_id - previous - 1
        if gap <= max_gap_rows:
            merged_gap_rows += gap
            group_ids.append(feature_id)
        else:
            groups.append((start, previous + 1, group_ids))
            start = feature_id
            group_ids = [feature_id]
        previous = feature_id
    groups.append((start, previous + 1, group_ids))

    requested_bytes = int(unique_ids.numel()) * d_model * itemsize
    overfetch_bytes = merged_gap_rows * d_model * itemsize
    fallback_reason = None
    singleton_range_count = sum(
        group_stop - group_start == 1 for group_start, group_stop, _ in groups
    )
    if len(groups) > max_range_count:
        fallback_reason = "range_count_exceeds_max"
    elif singleton_range_count > int(len(groups) * max_singleton_range_fraction):
        fallback_reason = "singleton_range_fraction_exceeds_max"
    elif len(groups) > baseline_pages * max_ranges_per_baseline_page:
        fallback_reason = "range_fragmentation_exceeds_baseline_ratio"
    elif overfetch_bytes > int(requested_bytes * max_overfetch_fraction):
        fallback_reason = "overfetch_fraction_exceeds_max"

    ranges = tuple(
        DecoderRowRange(
            start=group_start,
            stop=group_stop,
            feature_ids=torch.tensor(group_ids, dtype=torch.long),
        )
        for group_start, group_stop, group_ids in groups
    )
    return DecoderRowRangePlan(
        ranges=ranges,
        unique_feature_ids=unique_ids,
        planning_seconds=time.perf_counter() - started,
        fallback_reason=fallback_reason,
        merged_gap_rows=merged_gap_rows,
        baseline_full_page_count=baseline_pages,
        baseline_full_page_bytes=baseline_bytes,
    )


def combine_phase0_decoder_range_telemetry(
    layers: list[Phase0DecoderRangeTelemetry],
    *,
    seed_capture_seconds: float,
) -> Phase0DecoderRangeTelemetry | None:
    """Combine per-layer evidence without losing range-size distribution."""

    if not layers:
        return None
    fallback_reasons = tuple(
        telemetry.fallback_reason
        for telemetry in layers
        if telemetry.fallback_reason is not None
    )
    return Phase0DecoderRangeTelemetry(
        requested=any(telemetry.requested for telemetry in layers),
        effective=all(telemetry.effective for telemetry in layers),
        fallback_reason=(
            None
            if not fallback_reasons
            else ",".join(dict.fromkeys(fallback_reasons))
        ),
        planning_seconds=sum(telemetry.planning_seconds for telemetry in layers),
        read_seconds=sum(telemetry.read_seconds for telemetry in layers),
        gather_seconds=sum(telemetry.gather_seconds for telemetry in layers),
        reconstruction_seconds=sum(
            telemetry.reconstruction_seconds for telemetry in layers
        ),
        seed_capture_seconds=seed_capture_seconds,
        unique_row_count=sum(telemetry.unique_row_count for telemetry in layers),
        unique_row_bytes=sum(telemetry.unique_row_bytes for telemetry in layers),
        range_request_count=sum(telemetry.range_request_count for telemetry in layers),
        range_rows=tuple(
            row_count
            for telemetry in layers
            for row_count in telemetry.range_rows
        ),
        merged_gap_rows=sum(telemetry.merged_gap_rows for telemetry in layers),
        overfetch_bytes=sum(telemetry.overfetch_bytes for telemetry in layers),
        logical_requested_bytes=sum(
            telemetry.logical_requested_bytes for telemetry in layers
        ),
        logical_materialized_bytes=sum(
            telemetry.logical_materialized_bytes for telemetry in layers
        ),
        baseline_full_page_count=sum(
            telemetry.baseline_full_page_count for telemetry in layers
        ),
        baseline_full_page_bytes=sum(
            telemetry.baseline_full_page_bytes for telemetry in layers
        ),
        backend=",".join(dict.fromkeys(telemetry.backend for telemetry in layers)),
        occurrence_row_count=sum(
            telemetry.occurrence_row_count for telemetry in layers
        ),
        mapping_count=sum(telemetry.mapping_count for telemetry in layers),
        block_count=sum(telemetry.block_count for telemetry in layers),
        read_count=sum(telemetry.read_count for telemetry in layers),
        backend_request_count=sum(
            telemetry.backend_request_count for telemetry in layers
        ),
        page_span_bytes=sum(telemetry.page_span_bytes for telemetry in layers),
        backend_requested_bytes=sum(
            telemetry.backend_requested_bytes for telemetry in layers
        ),
        backend_materialized_bytes=sum(
            telemetry.backend_materialized_bytes for telemetry in layers
        ),
        planned_overfetch_ratio=(
            max(
                0.0,
                sum(telemetry.backend_materialized_bytes for telemetry in layers)
                / max(
                    1,
                    sum(telemetry.backend_requested_bytes for telemetry in layers),
                )
                - 1.0,
            )
        ),
        physical_read_estimate_bytes=(
            None
            if any(
                telemetry.physical_read_estimate_bytes is None for telemetry in layers
            )
            else sum(
                int(telemetry.physical_read_estimate_bytes or 0)
                for telemetry in layers
            )
        ),
        fault_read_seconds=sum(telemetry.fault_read_seconds for telemetry in layers),
        reorder_seconds=sum(telemetry.reorder_seconds for telemetry in layers),
        h2d_seconds=sum(telemetry.h2d_seconds for telemetry in layers),
        total_seconds=sum(telemetry.total_seconds for telemetry in layers),
        occurrence_row_bytes=sum(
            telemetry.occurrence_row_bytes for telemetry in layers
        ),
        mapping_open_count=sum(telemetry.mapping_open_count for telemetry in layers),
        range_count=sum(telemetry.range_count for telemetry in layers),
        output_bytes=sum(telemetry.output_bytes for telemetry in layers),
        temporary_staging_high_water_bytes=max(
            telemetry.temporary_staging_high_water_bytes for telemetry in layers
        ),
    )


def load_decoder_row_ranges(
    *,
    path: str,
    key: str,
    plan: DecoderRowRangePlan,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """Load admitted ranges on CPU and gather one compact sorted row table."""

    if not plan.admitted:
        raise ValueError("cannot load a refused decoder row range plan")
    if Path(path).suffix != ".safetensors":
        raise ValueError("decoder row ranges require a safetensors checkpoint")

    read_seconds = 0.0
    gather_seconds = 0.0
    compact_rows: torch.Tensor | None = None
    range_block: torch.Tensor | None = None
    gathered: torch.Tensor | None = None
    destination = 0
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            safe_slice = checkpoint.get_slice(key)
            shape = safe_slice.get_shape()
            compact_rows = torch.empty(
                (int(plan.unique_feature_ids.numel()), int(shape[1])),
                dtype=dtype,
                device="cpu",
            )
            for row_range in plan.ranges:
                read_started = time.perf_counter()
                range_block = safe_slice[row_range.start : row_range.stop]
                if range_block.dtype != dtype:
                    range_block = range_block.to(dtype=dtype)
                read_seconds += time.perf_counter() - read_started

                gather_started = time.perf_counter()
                local_ids = row_range.feature_ids - row_range.start
                gathered = range_block.index_select(0, local_ids)
                count = row_range.requested_rows
                compact_rows[destination : destination + count].copy_(gathered)
                destination += count
                gather_seconds += time.perf_counter() - gather_started
                gathered = None
                range_block = None
        if destination != int(plan.unique_feature_ids.numel()):
            raise RuntimeError("decoder row range gather did not fill the compact table")
        return compact_rows, read_seconds, gather_seconds
    except BaseException:
        gathered = None
        range_block = None
        compact_rows = None
        raise
