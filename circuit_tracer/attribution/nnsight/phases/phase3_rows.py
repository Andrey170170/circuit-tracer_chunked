"""Row production, normalization, capture, and commit for NNSight Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, cast

import torch

from circuit_tracer.attribution.nnsight.phase_support import (
    _copy_rows_to_cpu_staging,
    _resolve_phase3_effective_row_state,
)
from circuit_tracer.attribution.nnsight.replay import _compute_row_abs_sums
from circuit_tracer.attribution.nnsight.row_replay import RowRecipe, RowRecipeLedger
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.attribution.nnsight.telemetry import (
    _build_row_transfer_telemetry,
    _build_tensor_transfer_estimate,
)
from circuit_tracer.attribution.nnsight.tiled_rows import (
    produce_and_store_tiled_rows,
    produce_tiled_rows_no_retention,
)


@dataclass
class Phase3TransferMetrics:
    """Ordered transfer and numerical-kernel measurements for Phase 3."""

    compute_batch_elapsed_ms: float = 0.0
    cpu_staging_elapsed_ms: float = 0.0
    denominator_elapsed_ms: float = 0.0
    denominator_global_max_elapsed_ms: float = 0.0
    denominator_scaled_sum_elapsed_ms: float = 0.0
    row_store_write_elapsed_ms: float = 0.0
    gpu_to_cpu_bytes: int = 0
    cpu_to_gpu_bytes: int = 0
    copy_count: int = 0
    feature_backward_count: int = 0
    feature_produced_tile_count: int = 0
    feature_backward_tile_count: int = 0
    feature_transient_peak_bytes: int = 0

    def absorb(self, other: Phase3TransferMetrics) -> None:
        for name in (
            "compute_batch_elapsed_ms",
            "cpu_staging_elapsed_ms",
            "denominator_elapsed_ms",
            "denominator_global_max_elapsed_ms",
            "denominator_scaled_sum_elapsed_ms",
            "row_store_write_elapsed_ms",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "gpu_to_cpu_bytes",
            "cpu_to_gpu_bytes",
            "copy_count",
            "feature_backward_count",
            "feature_produced_tile_count",
            "feature_backward_tile_count",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.feature_transient_peak_bytes = max(
            self.feature_transient_peak_bytes, other.feature_transient_peak_bytes
        )


@dataclass
class Phase3ReplayRows:
    """Ordered row material retained only for a requested replay bundle."""

    feature_rows: list[torch.Tensor] = field(default_factory=list)
    row_abs_sums: list[torch.Tensor] = field(default_factory=list)
    feature_abs_sums: list[torch.Tensor] = field(default_factory=list)
    error_abs_sums: list[torch.Tensor] = field(default_factory=list)
    token_abs_sums: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class ProducedRows:
    rows: torch.Tensor
    rows_cpu: torch.Tensor
    rows_cpu_staging: torch.Tensor | None
    tiled_denominator: tuple[torch.Tensor, torch.Tensor] | None
    tiled_production: bool
    no_retention: bool
    inject_transfer: dict[str, object]
    metrics: Phase3TransferMetrics


@dataclass(frozen=True)
class EffectiveRows:
    rows_cpu: torch.Tensor
    row_input: torch.Tensor
    feature_rows: torch.Tensor
    denominator: tuple[torch.Tensor, torch.Tensor]
    row_abs_sums: torch.Tensor
    row_transfer: dict[str, object]
    donor_feature_abs_sums: torch.Tensor | None
    donor_error_abs_sums: torch.Tensor | None
    donor_token_abs_sums: torch.Tensor | None
    denominator_elapsed_ms: float


def produce_logit_rows(
    *,
    ctx: Any,
    model_device: torch.device,
    batch: torch.Tensor,
    row_start: int,
    n_layers: int,
    n_pos: int,
    output_position: int | None,
    dtype: torch.dtype,
    full_retention_backend: str,
    feature_row_retention: str,
    feature_column_tile_size: int,
    feature_row_store: _FileBackedFeatureRowStore | None,
    nonfeature_row_store: _FileBackedFeatureRowStore | None,
    rows_cpu_staging: torch.Tensor | None,
) -> ProducedRows:
    """Produce one logit batch without changing its compute/copy order."""
    metrics = Phase3TransferMetrics()
    inject_transfer = _build_tensor_transfer_estimate(
        prefix="inject_values", source=batch, destination_device=model_device
    )
    if (
        inject_transfer["inject_values_source"] == "cpu"
        and inject_transfer["inject_values_destination"] == "cuda"
    ):
        metrics.cpu_to_gpu_bytes += int(inject_transfer["inject_values_transfer_bytes"])

    no_retention = feature_row_retention == "none_recompute"
    tiled_production = full_retention_backend == "column_tiled_v1" or no_retention
    tiled_telemetry: dict[str, int | float] = {}
    layers = torch.full((batch.shape[0],), n_layers)
    positions = torch.full(
        (batch.shape[0],), output_position if output_position is not None else n_pos - 1
    )
    compute_start = time.perf_counter()
    tiled_denominator: tuple[torch.Tensor, torch.Tensor] | None = None
    if no_retention:
        assert feature_row_store is not None and nonfeature_row_store is not None
        rows, tiled_denominator = produce_tiled_rows_no_retention(
            ctx=ctx,
            layers=layers,
            positions=positions,
            inject_values=batch,
            feature_column_tile_size=feature_column_tile_size,
            dtype=dtype,
            phase_label="phase3_logits",
            telemetry=tiled_telemetry,
        )
    elif tiled_production:
        assert feature_row_store is not None and nonfeature_row_store is not None
        rows, tiled_denominator = produce_and_store_tiled_rows(
            ctx=ctx,
            layers=layers,
            positions=positions,
            inject_values=batch,
            row_start=row_start,
            feature_row_store=feature_row_store,
            nonfeature_row_store=nonfeature_row_store,
            feature_column_tile_size=feature_column_tile_size,
            dtype=dtype,
            phase_label="phase3_logits",
            telemetry=tiled_telemetry,
        )
    else:
        rows = ctx.compute_batch(
            layers=layers,
            positions=positions,
            inject_values=batch,
            phase_label="phase3_logits",
        )
    metrics.compute_batch_elapsed_ms = (time.perf_counter() - compute_start) * 1000.0
    if tiled_production:
        _absorb_tiled_telemetry(metrics, tiled_telemetry)

    staging_start = time.perf_counter()
    if tiled_production:
        rows_cpu = rows
    else:
        rows_cpu, rows_cpu_staging = _copy_rows_to_cpu_staging(
            rows, staging_buffer=rows_cpu_staging
        )
    metrics.cpu_staging_elapsed_ms += (time.perf_counter() - staging_start) * 1000.0
    return ProducedRows(
        rows=rows,
        rows_cpu=rows_cpu,
        rows_cpu_staging=rows_cpu_staging,
        tiled_denominator=tiled_denominator,
        tiled_production=tiled_production,
        no_retention=no_retention,
        inject_transfer=cast(dict[str, object], inject_transfer),
        metrics=metrics,
    )


def resolve_effective_rows(
    *, produced: ProducedRows, donor_bundle: dict[str, object] | None, row_start: int,
    row_count: int, logit_offset: int, total_active_features: int, dtype: torch.dtype,
) -> EffectiveRows:
    """Apply replay replacement and compute the stable row-L1 denominator."""
    donor_feature_rows = donor_row_abs_sums = None
    donor_feature_abs_sums = donor_error_abs_sums = donor_token_abs_sums = None
    if donor_bundle is not None:
        end = row_start + row_count
        donor_feature_rows = cast(torch.Tensor, donor_bundle["phase3_feature_rows"])[row_start:end]
        donor_row_abs_sums = cast(torch.Tensor, donor_bundle["row_abs_sums"])[row_start:end]
        donor_feature_abs_sums = cast(torch.Tensor, donor_bundle["feature_abs_sums"])[row_start:end]
        donor_error_abs_sums = cast(torch.Tensor, donor_bundle["error_abs_sums"])[row_start:end]
        donor_token_abs_sums = cast(torch.Tensor, donor_bundle["token_abs_sums"])[row_start:end]

    start = time.perf_counter()
    if produced.tiled_production:
        assert produced.tiled_denominator is not None
        rows_cpu = row_input = produced.rows_cpu
        feature_rows = torch.empty((rows_cpu.shape[0], 0), dtype=rows_cpu.dtype)
        denominator = produced.tiled_denominator
        row_abs_sums = denominator[0] * denominator[1]
    else:
        rows_cpu, row_input, feature_rows, denominator, row_abs_sums = (
            _resolve_phase3_effective_row_state(
                rows_cpu=produced.rows_cpu,
                row_input_column_count=logit_offset,
                total_active_features=total_active_features,
                dtype=dtype,
                donor_feature_rows=donor_feature_rows,
                donor_row_abs_sums=donor_row_abs_sums,
            )
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    row_transfer = _build_row_transfer_telemetry(
        rows=produced.rows,
        rows_cpu=rows_cpu,
        row_input_slice=row_input,
        feature_row_slice=feature_rows,
    )
    return EffectiveRows(
        rows_cpu, row_input, feature_rows, denominator, row_abs_sums,
        cast(dict[str, object], row_transfer), donor_feature_abs_sums,
        donor_error_abs_sums, donor_token_abs_sums, elapsed_ms,
    )


def capture_replay_rows(
    *, captures: Phase3ReplayRows, rows: EffectiveRows, total_active_features: int,
    n_layers: int, n_pos: int, logit_offset: int,
) -> None:
    """Append replay evidence in the same batch order as attribution."""
    feature_rows = rows.feature_rows.contiguous()
    error_start = total_active_features
    error_end = total_active_features + n_layers * n_pos
    captures.feature_rows.append(feature_rows)
    captures.row_abs_sums.append(rows.row_abs_sums.contiguous())
    if all(
        item is not None
        for item in (
            rows.donor_feature_abs_sums,
            rows.donor_error_abs_sums,
            rows.donor_token_abs_sums,
        )
    ):
        captures.feature_abs_sums.append(cast(torch.Tensor, rows.donor_feature_abs_sums).contiguous())
        captures.error_abs_sums.append(cast(torch.Tensor, rows.donor_error_abs_sums).contiguous())
        captures.token_abs_sums.append(cast(torch.Tensor, rows.donor_token_abs_sums).contiguous())
        return
    captures.feature_abs_sums.append(_compute_row_abs_sums(feature_rows, dtype=torch.float64).contiguous())
    captures.error_abs_sums.append(
        _compute_row_abs_sums(rows.rows_cpu[:, error_start:error_end], dtype=torch.float64).contiguous()
    )
    captures.token_abs_sums.append(
        _compute_row_abs_sums(rows.rows_cpu[:, error_end:logit_offset], dtype=torch.float64).contiguous()
    )


def commit_effective_rows(
    *, produced: ProducedRows, rows: EffectiveRows, batch: torch.Tensor, row_start: int,
    n_layers: int, n_pos: int, output_position: int | None, logit_offset: int,
    total_active_features: int, use_compact_store: bool,
    feature_row_store: _FileBackedFeatureRowStore | None,
    nonfeature_row_store: _FileBackedFeatureRowStore | None,
    edge_matrix: torch.Tensor | None, row_to_node_index: torch.Tensor,
) -> float:
    """Commit rows to exactly one retention backend and assign node indices."""
    start = time.perf_counter()
    if produced.no_retention:
        assert isinstance(feature_row_store, RowRecipeLedger)
        assert isinstance(nonfeature_row_store, RowRecipeLedger)
        for local_index in range(batch.shape[0]):
            ordinal = row_start + local_index
            recipe = RowRecipe(
                ordinal=ordinal,
                source_kind="logit",
                layer=n_layers,
                position=output_position if output_position is not None else n_pos - 1,
                injection=batch[local_index],
            )
            denominator = tuple(
                value[local_index : local_index + 1] for value in rows.denominator
            )
            node_index = logit_offset + ordinal
            feature_row_store.append_recipe(recipe, node_index=node_index, denominator=denominator)
            nonfeature_row_store.append_recipe(recipe, node_index=node_index, denominator=denominator)
        elapsed_ms = 0.0
    elif produced.tiled_production:
        elapsed_ms = 0.0
    elif use_compact_store:
        assert feature_row_store is not None and nonfeature_row_store is not None
        feature_row_store.append_rows(
            row_start=row_start,
            feature_rows=rows.feature_rows,
            row_denominator_scaled_l1=rows.denominator,
            phase="phase3",
        )
        nonfeature_row_store.append_rows(
            row_start=row_start,
            feature_rows=rows.rows_cpu[:, total_active_features:logit_offset],
            row_denominator_scaled_l1=rows.denominator,
            phase="phase3",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    else:
        assert edge_matrix is not None
        edge_matrix[row_start : row_start + batch.shape[0], :logit_offset] = rows.rows_cpu
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    row_to_node_index[row_start : row_start + batch.shape[0]] = (
        torch.arange(row_start, row_start + batch.shape[0]) + logit_offset
    )
    return elapsed_ms


def account_row_transfer(metrics: Phase3TransferMetrics, rows: EffectiveRows) -> None:
    if rows.row_transfer["row_transfer_source"] == "cuda":
        metrics.gpu_to_cpu_bytes += int(rows.row_transfer["row_transfer_bytes"])
    if rows.row_transfer["row_transfer_destination"] == "cuda":
        metrics.cpu_to_gpu_bytes += int(rows.row_transfer["row_transfer_bytes"])
    if int(rows.row_transfer["row_transfer_bytes"]) > 0:
        metrics.copy_count += 1


def _absorb_tiled_telemetry(
    metrics: Phase3TransferMetrics, telemetry: dict[str, int | float]
) -> None:
    metrics.gpu_to_cpu_bytes += int(telemetry.get("feature_transfer_bytes", 0))
    metrics.copy_count += int(telemetry.get("feature_copy_count", 0))
    metrics.feature_backward_count += int(telemetry.get("feature_backward_count", 0))
    metrics.feature_produced_tile_count += int(telemetry.get("feature_produced_tile_count", 0))
    metrics.feature_backward_tile_count += int(telemetry.get("feature_backward_tile_count", 0))
    metrics.feature_transient_peak_bytes = int(telemetry.get("feature_transient_peak_bytes", 0))
    metrics.cpu_staging_elapsed_ms += float(telemetry.get("feature_cpu_copy_elapsed_ms", 0.0))
    metrics.denominator_elapsed_ms += float(telemetry.get("feature_denominator_elapsed_ms", 0.0))
    metrics.denominator_global_max_elapsed_ms += float(
        telemetry.get("feature_denominator_global_max_elapsed_ms", 0.0)
    )
    metrics.denominator_scaled_sum_elapsed_ms += float(
        telemetry.get("feature_denominator_scaled_sum_elapsed_ms", 0.0)
    )
    metrics.row_store_write_elapsed_ms += float(
        telemetry.get("feature_store_write_elapsed_ms", 0.0)
    )
