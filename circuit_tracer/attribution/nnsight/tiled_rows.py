"""True column-tiled attribution row production."""

from __future__ import annotations

from typing import Any

import torch

from circuit_tracer.attribution.nnsight.replay import _compute_row_denominator_scaled_l1


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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Stream feature tiles to storage, then produce/store nonfeature columns."""
    denominator: tuple[torch.Tensor, torch.Tensor] | None = None

    def consume(start: int, _end: int, tile: torch.Tensor) -> None:
        nonlocal denominator
        tile_cpu = tile.detach().to(device="cpu", dtype=dtype)
        feature_row_store.append_tile(
            row_start=row_start, column_start=start, values=tile_cpu, phase=phase_label
        )
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
    global_max = nonfeature.abs().amax(dim=1).to(dtype=dtype)
    for start in range(0, int(feature_row_store.n_feature_columns), 4096):
        end = min(start + 4096, int(feature_row_store.n_feature_columns))
        tile = feature_row_store.read_tile(row_start, row_start + row_count, start, end)
        global_max = torch.maximum(global_max, tile.abs().amax(dim=1).to(dtype=dtype))
    scaled_sum = torch.zeros_like(global_max)
    finite = (global_max > 0) & torch.isfinite(global_max)
    for start in range(0, int(feature_row_store.n_feature_columns), 4096):
        end = min(start + 4096, int(feature_row_store.n_feature_columns))
        tile = feature_row_store.read_tile(row_start, row_start + row_count, start, end)
        if bool(finite.any()):
            scaled_sum[finite] += (
                tile[finite].abs().to(dtype=dtype) / global_max[finite, None]
            ).sum(dim=1)
    for start in range(0, int(nonfeature.shape[1]), 4096):
        end = min(start + 4096, int(nonfeature.shape[1]))
        if bool(finite.any()):
            scaled_sum[finite] += (
                nonfeature[finite, start:end].abs().to(dtype=dtype)
                / global_max[finite, None]
            ).sum(dim=1)
    scaled_sum[torch.isinf(global_max)] = 1
    denominator = (global_max, scaled_sum)
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Produce each tile once, retain only stable denominator state and nonfeatures."""
    denominator: tuple[torch.Tensor, torch.Tensor] | None = None

    def consume(_start: int, _end: int, tile: torch.Tensor) -> None:
        nonlocal denominator
        denominator = _merge_scaled_l1(
            denominator, tile.detach().to(device="cpu", dtype=dtype), dtype=dtype
        )

    nonfeature = ctx.produce_row_tiles(
        layers,
        positions,
        inject_values,
        feature_column_tile_size=feature_column_tile_size,
        consume_feature_tile=consume,
        phase_label=phase_label,
        retain_graph=True,
    ).detach().to(device="cpu", dtype=dtype)
    denominator = _merge_scaled_l1(denominator, nonfeature, dtype=dtype)
    assert denominator is not None
    return nonfeature, denominator
