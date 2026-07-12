"""True column-tiled attribution row production."""

from __future__ import annotations

import time
from typing import Any

import torch

from circuit_tracer.attribution.nnsight.replay import _compute_row_denominator_scaled_l1


_CANONICAL_DENOMINATOR_CHUNK_SIZE = 4096


def _canonical_denominator_from_reader(
    *,
    row_count: int,
    feature_columns: int,
    nonfeature: torch.Tensor,
    read_feature: Any,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match scaled-L1 over the full concatenated row in global 4096-column chunks."""
    total_columns = feature_columns + int(nonfeature.shape[1])

    def read_global(start: int, end: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        feature_end = min(end, feature_columns)
        if start < feature_end:
            pieces.append(read_feature(start, feature_end).to(device="cpu", dtype=dtype))
        nonfeature_start = max(start, feature_columns) - feature_columns
        nonfeature_end = end - feature_columns
        if nonfeature_end > nonfeature_start:
            pieces.append(nonfeature[:, nonfeature_start:nonfeature_end])
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    global_max = torch.zeros(row_count, dtype=dtype)
    for start in range(0, total_columns, _CANONICAL_DENOMINATOR_CHUNK_SIZE):
        tile = read_global(start, min(start + _CANONICAL_DENOMINATOR_CHUNK_SIZE, total_columns))
        global_max = torch.maximum(global_max, tile.abs().amax(dim=1))
    scaled_sum = torch.zeros_like(global_max)
    finite = (global_max > 0) & torch.isfinite(global_max)
    for start in range(0, total_columns, _CANONICAL_DENOMINATOR_CHUNK_SIZE):
        tile = read_global(start, min(start + _CANONICAL_DENOMINATOR_CHUNK_SIZE, total_columns))
        if bool(finite.any()):
            scaled_sum[finite] += (
                tile[finite].abs() / global_max[finite, None]
            ).sum(dim=1)
    scaled_sum[torch.isinf(global_max)] = 1
    return global_max, scaled_sum


def _merge_scaled_l1(
    current: tuple[torch.Tensor, torch.Tensor] | None,
    values: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate stable L1 state without concatenating column tiles."""
    new_max, new_scaled = _compute_row_denominator_scaled_l1(values, dtype=dtype)
    if current is None:
        return new_max, new_scaled
    old_max, old_scaled = current
    merged_max = torch.maximum(old_max, new_max)
    merged_scaled = torch.zeros_like(merged_max)
    finite = (merged_max > 0) & torch.isfinite(merged_max)
    if bool(finite.any()):
        merged_scaled[finite] = old_scaled[finite] * (
            old_max[finite] / merged_max[finite]
        ) + new_scaled[finite] * (new_max[finite] / merged_max[finite])
    infinite = torch.isinf(merged_max)
    if bool(infinite.any()):
        merged_scaled[infinite] = 1
    return merged_max, merged_scaled


def produce_and_store_tiled_rows(
    *,
    ctx: Any,
    layers: torch.Tensor,
    positions: torch.Tensor,
    inject_values: torch.Tensor,
    row_start: int,
    feature_row_store: Any,
    nonfeature_row_store: Any,
    feature_column_tile_size: int,
    dtype: torch.dtype,
    phase_label: str,
    retain_graph: bool = True,
    telemetry: dict[str, int | float] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Stream feature tiles to storage, then produce/store nonfeature columns."""
    denominator: tuple[torch.Tensor, torch.Tensor] | None = None

    def consume(start: int, _end: int, tile: torch.Tensor) -> None:
        nonlocal denominator
        copy_start = time.perf_counter()
        tile_cpu = tile.detach().to(device="cpu", dtype=dtype)
        if telemetry is not None:
            telemetry["feature_transfer_bytes"] = int(telemetry.get("feature_transfer_bytes", 0)) + (
                int(tile_cpu.numel() * tile_cpu.element_size()) if tile.device.type != "cpu" else 0
            )
            telemetry["feature_copy_count"] = int(telemetry.get("feature_copy_count", 0)) + int(
                tile.device.type != "cpu" or tile.dtype != dtype
            )
            telemetry["feature_cpu_copy_elapsed_ms"] = float(
                telemetry.get("feature_cpu_copy_elapsed_ms", 0.0)
            ) + (time.perf_counter() - copy_start) * 1000.0
        store_start = time.perf_counter()
        feature_row_store.append_tile(
            row_start=row_start, column_start=start, values=tile_cpu, phase=phase_label
        )
        if telemetry is not None:
            telemetry["feature_store_write_elapsed_ms"] = float(
                telemetry.get("feature_store_write_elapsed_ms", 0.0)
            ) + (time.perf_counter() - store_start) * 1000.0
        denominator = _merge_scaled_l1(denominator, tile_cpu, dtype=dtype)

    nonfeature = (
        ctx.produce_row_tiles(
            layers,
            positions,
            inject_values,
            feature_column_tile_size=feature_column_tile_size,
            consume_feature_tile=consume,
            phase_label=phase_label,
            retain_graph=retain_graph,
        )
        .detach()
        .to(device="cpu", dtype=dtype)
    )
    # Re-read persisted feature tiles so full-retention tiled execution follows
    # the canonical global-max then 4096-column scaled-sum reduction order.
    # The streaming merge above remains useful for no-retention execution, but
    # its floating-point association is not the full-file reference contract.
    row_count = int(nonfeature.shape[0])
    denominator_start = time.perf_counter()
    denominator = _canonical_denominator_from_reader(
        row_count=row_count,
        feature_columns=int(feature_row_store.n_feature_columns),
        nonfeature=nonfeature,
        read_feature=lambda start, end: feature_row_store.read_tile(
            row_start, row_start + row_count, start, end
        ),
        dtype=dtype,
    )
    if telemetry is not None:
        telemetry["feature_backward_count"] = int(telemetry.get("feature_backward_count", 0)) + int(
            (int(feature_row_store.n_feature_columns) + feature_column_tile_size - 1)
            // feature_column_tile_size
            + 1
        )
        telemetry["feature_denominator_elapsed_ms"] = float(
            telemetry.get("feature_denominator_elapsed_ms", 0.0)
        ) + (time.perf_counter() - denominator_start) * 1000.0
    feature_row_store.set_row_denominator(row_start=row_start, value=denominator)
    nonfeature_row_store.append_tile(
        row_start=row_start, column_start=0, values=nonfeature, nonfeature=True, phase=phase_label
    )
    nonfeature_row_store.set_row_denominator(row_start=row_start, value=denominator)
    return nonfeature, denominator


def produce_tiled_rows_no_retention(
    *,
    ctx: Any,
    layers: torch.Tensor,
    positions: torch.Tensor,
    inject_values: torch.Tensor,
    feature_column_tile_size: int,
    dtype: torch.dtype,
    phase_label: str,
    telemetry: dict[str, int | float] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Reuse current-microbatch tiles for the canonical denominator reduction."""
    feature_columns = 0
    global_max: torch.Tensor | None = None
    feature_tiles: list[tuple[int, int, torch.Tensor]] = []
    transient_bytes = 0
    global_max_elapsed_ms = 0.0

    def record_transient_peak(extra_bytes: int = 0) -> None:
        if telemetry is not None:
            telemetry["feature_transient_peak_bytes"] = max(
                int(telemetry.get("feature_transient_peak_bytes", 0)),
                transient_bytes + extra_bytes,
            )

    def consume(start: int, end: int, tile: torch.Tensor) -> None:
        nonlocal feature_columns, global_max, transient_bytes, global_max_elapsed_ms
        copy_start = time.perf_counter()
        tile_cpu = tile.detach().to(device="cpu", dtype=dtype)
        if tile_cpu.ndim != 2 or int(tile_cpu.shape[0]) != int(layers.numel()):
            raise RuntimeError("none_recompute feature tile had an unexpected row shape")
        if end <= start or int(tile_cpu.shape[1]) != end - start:
            raise RuntimeError("none_recompute feature tile width did not match its column range")
        feature_tiles.append((start, end, tile_cpu))
        feature_columns = max(feature_columns, end)
        transient_bytes += int(tile_cpu.numel() * tile_cpu.element_size())
        global_max_start = time.perf_counter()
        tile_max = tile_cpu.abs().amax(dim=1)
        global_max_elapsed_ms += (time.perf_counter() - global_max_start) * 1000.0
        if telemetry is not None:
            transferred = (
                int(tile.numel() * tile.element_size())
                if tile.device.type != "cpu"
                else 0
            )
            telemetry["feature_transfer_bytes"] = int(telemetry.get("feature_transfer_bytes", 0)) + transferred
            telemetry["feature_copy_count"] = int(
                telemetry.get("feature_copy_count", 0)
            ) + int(tile.device.type != "cpu" or tile.dtype != dtype)
            telemetry["feature_cpu_copy_elapsed_ms"] = float(
                telemetry.get("feature_cpu_copy_elapsed_ms", 0.0)
            ) + (time.perf_counter() - copy_start) * 1000.0
            record_transient_peak()
        global_max = tile_max if global_max is None else torch.maximum(global_max, tile_max)

    nonfeature = ctx.produce_row_tiles(
        layers,
        positions,
        inject_values,
        feature_column_tile_size=feature_column_tile_size,
        consume_feature_tile=consume,
        phase_label=phase_label,
        retain_graph=True,
    ).detach().to(device="cpu", dtype=dtype)
    global_max_start = time.perf_counter()
    nonfeature_max = nonfeature.abs().amax(dim=1)
    global_max = nonfeature_max if global_max is None else torch.maximum(global_max, nonfeature_max)
    global_max_elapsed_ms += (time.perf_counter() - global_max_start) * 1000.0
    scaled_sum = torch.zeros_like(global_max)
    finite = (global_max > 0) & torch.isfinite(global_max)
    pending: torch.Tensor | None = None
    def consume_values(values: torch.Tensor) -> None:
        nonlocal pending, scaled_sum
        values = values.detach().to(device="cpu", dtype=dtype)
        offset = 0
        while offset < values.shape[1]:
            pending_width = 0 if pending is None else int(pending.shape[1])
            take = min(
                _CANONICAL_DENOMINATOR_CHUNK_SIZE - pending_width,
                int(values.shape[1]) - offset,
            )
            piece = values[:, offset : offset + take]
            pending = piece if pending is None else torch.cat((pending, piece), dim=1)
            record_transient_peak(int(pending.numel() * pending.element_size()))
            offset += take
            if pending.shape[1] == _CANONICAL_DENOMINATOR_CHUNK_SIZE:
                if bool(finite.any()):
                    scaled_sum[finite] += (
                        pending[finite].abs() / global_max[finite, None]
                    ).sum(dim=1)
                pending = None

    scaled_sum_start = time.perf_counter()
    expected_start = 0
    for start, end, tile in feature_tiles:
        if start != expected_start or end <= start:
            raise RuntimeError("none_recompute feature tiles were not contiguous and ordered")
        consume_values(tile)
        expected_start = end
    if expected_start != feature_columns:
        raise RuntimeError("none_recompute feature tile coverage was incomplete")
    consume_values(nonfeature)
    if pending is not None and bool(finite.any()):
        scaled_sum[finite] += (pending[finite].abs() / global_max[finite, None]).sum(dim=1)
    scaled_sum[torch.isinf(global_max)] = 1
    if telemetry is not None:
        tile_count = len(feature_tiles)
        backward_count = tile_count + 1
        telemetry["feature_produced_tile_count"] = int(
            telemetry.get("feature_produced_tile_count", 0)
        ) + tile_count
        telemetry["feature_backward_tile_count"] = int(
            telemetry.get("feature_backward_tile_count", 0)
        ) + backward_count
        telemetry["feature_backward_count"] = int(
            telemetry.get("feature_backward_count", 0)
        ) + backward_count
        telemetry["feature_denominator_elapsed_ms"] = float(
            telemetry.get("feature_denominator_elapsed_ms", 0.0)
        ) + global_max_elapsed_ms + (time.perf_counter() - scaled_sum_start) * 1000.0
        telemetry["feature_denominator_global_max_elapsed_ms"] = float(
            telemetry.get("feature_denominator_global_max_elapsed_ms", 0.0)
        ) + global_max_elapsed_ms
        telemetry["feature_denominator_scaled_sum_elapsed_ms"] = float(
            telemetry.get("feature_denominator_scaled_sum_elapsed_ms", 0.0)
        ) + (time.perf_counter() - scaled_sum_start) * 1000.0
    return nonfeature, (global_max, scaled_sum)
