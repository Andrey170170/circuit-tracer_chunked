"""Resident active-decoder-row contraction for NNSight attribution."""

from __future__ import annotations

import time
from typing import Any

import torch
from einops import einsum

from .active_decoder_rows import ActiveDecoderRows


def contract_active_decoder_rows(
    ctx: Any,
    owner: ActiveDecoderRows,
    grad_batches: list[
        tuple[tuple[torch.Tensor | None, ...] | list[torch.Tensor | None], torch.Tensor, int]
    ],
    *,
    phase_label: str | None,
    batch_index: int | None,
) -> None:
    """Contract resident rows without changing source/output/batch ordering."""

    state = ctx.chunked_decoder_state
    assert state is not None
    positions = state["positions"]
    activation_values = state["activation_values"]
    active_output_layers = sorted(
        {
            layer
            for output_layer_grads, _, _ in grad_batches
            for layer, grads in enumerate(output_layer_grads)
            if grads is not None
        }
    )
    if not active_output_layers:
        return

    row_subchunk_size = ctx._effective_row_subchunk_size()
    output_layer_seconds = {layer: 0.0 for layer in active_output_layers}
    contraction_counts = {layer: 0 for layer in active_output_layers}
    grad_cache: dict[tuple[int, int], torch.Tensor] = {}
    replay_start = time.perf_counter()

    for output_layer in active_output_layers:
        ctx._emit_trace(
            "phase3.chunked_attr.output_layer_start",
            output_layer=output_layer,
            total_sources=output_layer + 1,
        )

    for source_layer, block in enumerate(owner.layers):
        source_layer_start = time.perf_counter()
        if block is None:
            continue
        relevant_output_layers = [
            layer for layer in block.output_layers if layer in active_output_layers
        ]
        if not relevant_output_layers:
            continue

        layer_start = block.global_row_start
        layer_end = block.global_row_end
        if ctx._produced_feature_range is not None:
            requested_start, requested_end = ctx._produced_feature_range
            layer_start = max(layer_start, requested_start)
            layer_end = min(layer_end, requested_end)
            if layer_start >= layer_end:
                continue

        local_start = layer_start - block.global_row_start
        local_end = layer_end - block.global_row_start
        layer_rows = torch.arange(layer_start, layer_end, device=positions.device)
        layer_positions = positions[layer_start:layer_end]
        resident_rows = block.rows[local_start:local_end]
        layer_activations = activation_values[layer_start:layer_end].to(
            device=resident_rows.device,
            dtype=grad_batches[0][1].dtype,
            non_blocking=resident_rows.device.type == "cuda",
        )[:, None]

        for output_layer in relevant_output_layers:
            output_layer_start = time.perf_counter()
            output_slot = block.output_layers.index(output_layer)
            decoder_vectors = resident_rows[:, output_slot]
            for output_layer_grads, batch_buffer, grad_batch_index in grad_batches:
                grads = output_layer_grads[output_layer]
                if grads is None:
                    continue
                cache_key = (grad_batch_index, output_layer)
                typed_grads = grad_cache.get(cache_key)
                if typed_grads is None:
                    typed_grads = grads.to(
                        device=resident_rows.device,
                        dtype=batch_buffer.dtype,
                        non_blocking=resident_rows.device.type == "cuda",
                    )
                    grad_cache[cache_key] = typed_grads
                total_subchunks = max(
                    (len(layer_rows) + row_subchunk_size - 1) // row_subchunk_size,
                    1,
                )
                for row_subchunk_idx, row_start in enumerate(
                    range(0, len(layer_rows), row_subchunk_size),
                    start=1,
                ):
                    row_slice = slice(row_start, row_start + row_subchunk_size)
                    row_chunk_rows = layer_rows[row_slice]
                    row_chunk_positions = layer_positions[row_slice]
                    scaled_decoders = (
                        decoder_vectors[row_slice].to(dtype=batch_buffer.dtype)
                        * layer_activations[row_slice]
                    )
                    write_rows = row_chunk_rows
                    if ctx._produced_feature_range is not None:
                        write_rows = write_rows - ctx._produced_feature_range[0]
                    batch_buffer[write_rows] += einsum(
                        typed_grads[:, row_chunk_positions],
                        scaled_decoders,
                        "batch position d_model, position d_model -> position batch",
                    )
                    contraction_counts[output_layer] += 1
                    if (
                        contraction_counts[output_layer] <= 2
                        or contraction_counts[output_layer] % ctx._trace_chunk_interval == 0
                    ):
                        ctx._emit_trace(
                            "phase3.chunked_attr.chunk",
                            output_layer=output_layer,
                            source_layer=source_layer,
                            chunk=contraction_counts[output_layer],
                            decoder_chunk_id="resident",
                            processed_chunks=1,
                            total_chunks=1,
                            row_subchunk=row_subchunk_idx,
                            total_row_subchunks=total_subchunks,
                        )
            output_layer_seconds[output_layer] += time.perf_counter() - output_layer_start

        if ctx.diagnostic_mode:
            ctx._add_layer_stat(
                "chunked_attr_seconds_by_source_layer",
                source_layer,
                time.perf_counter() - source_layer_start,
            )
        ctx._record_telemetry_event(
            scope="op",
            name="context.chunked_replay.source_layer",
            phase=phase_label,
            batch_index=batch_index,
            elapsed_ms=(time.perf_counter() - source_layer_start) * 1000.0,
            attrs={
                "source_layer": source_layer,
                "active_decoder_chunks": 0,
                "resident_active_decoder_rows": layer_end - layer_start,
                "relevant_output_layers": len(relevant_output_layers),
            },
        )

    for output_layer in active_output_layers:
        elapsed = output_layer_seconds[output_layer]
        if ctx.diagnostic_mode:
            ctx._add_layer_stat(
                "chunked_attr_chunks_by_output_layer",
                output_layer,
                float(contraction_counts[output_layer]),
            )
            ctx._add_layer_stat("chunked_attr_seconds_by_output_layer", output_layer, elapsed)
            ctx._add_layer_stat("feature_attr_seconds_by_layer", output_layer, elapsed)
        ctx._emit_trace(
            "phase3.chunked_attr.output_layer_done",
            output_layer=output_layer,
            chunks=contraction_counts[output_layer],
            elapsed_s=f"{elapsed:.2f}",
            elapsed_ms=elapsed * 1000.0,
        )
    if ctx.diagnostic_mode:
        ctx._add_stat("chunked_attr_replay_seconds", time.perf_counter() - replay_start)
    ctx._record_telemetry_event(
        scope="op",
        name="context.chunked_replay",
        phase=phase_label,
        batch_index=batch_index,
        elapsed_ms=(time.perf_counter() - replay_start) * 1000.0,
        attrs={
            "active_output_layers": len(active_output_layers),
            "decoder_active_row_residency_effective": True,
        },
    )
