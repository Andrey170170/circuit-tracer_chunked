"""Bounded semantic signatures from Phase-5 resident decoder rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from circuit_tracer.attribution.nnsight.active_decoder_rows import ActiveDecoderRows
from circuit_tracer.transcoder.provider import provider_fingerprint


DESCRIPTOR_KIND = "active_decoder_countsketch_v1"
DESCRIPTOR_SCOPE = "active_occurrence_downstream_decoder_rows_v1"
PROJECTION_ID = "fixed_width_countsketch_l2_v1"
PROJECTION_MAX_BYTES = 64 * 1024 * 1024
# `_projection_coordinates` retains one int64 coordinate vector while its
# arithmetic can simultaneously materialize three additional int64 operands
# and results.  Eight int64 values per coordinate is a conservative peak; it
# also covers the final float32 sign and boolean predicate temporaries.
PROJECTION_COORDINATE_PEAK_BYTES_PER_COLUMN = 8 * torch.int64.itemsize


@dataclass(frozen=True)
class DecoderSignatureProjection:
    sketch: torch.Tensor
    descriptor_kind: str
    descriptor_scope: str
    decoder_source_fingerprint: str
    decoder_evidence_fingerprint: str
    projection_fingerprint: str
    max_bytes: int
    required_bytes: int
    workspace_peak_bytes: int


def attach_projected_decoder_signatures(
    payload: dict[str, object],
    *,
    active_decoder_rows: ActiveDecoderRows,
    decoder_provider: object,
) -> None:
    """Replace staging metadata with qualification-grade decoder evidence."""

    candidate_rows = payload.get("candidate_row_indices")
    candidate_features = payload.get("candidate_features")
    raw_dim = payload.get("descriptor_dim")
    if not isinstance(candidate_rows, torch.Tensor) or not isinstance(
        candidate_features, torch.Tensor
    ):
        raise TypeError("finalized semantic descriptor candidates are required")
    if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        raise TypeError("semantic descriptor dimension must be an integer")
    projection = project_active_decoder_signatures(
        active_decoder_rows=active_decoder_rows,
        decoder_provider=decoder_provider,
        candidate_row_indices=candidate_rows,
        candidate_features=candidate_features,
        descriptor_dim=raw_dim,
    )
    payload.update(
        {
            "descriptor_version": "v2",
            "descriptor_kind": projection.descriptor_kind,
            "descriptor_scope": projection.descriptor_scope,
            "descriptor_is_decoder_evidence": True,
            "semantic_sketch": projection.sketch,
            "decoder_source_fingerprint": projection.decoder_source_fingerprint,
            "decoder_evidence_fingerprint": projection.decoder_evidence_fingerprint,
            "projection_id": PROJECTION_ID,
            "projection_fingerprint": projection.projection_fingerprint,
            "semantic_descriptor_projection_max_bytes": projection.max_bytes,
            "semantic_descriptor_projection_required_bytes": projection.required_bytes,
            "semantic_descriptor_projection_workspace_peak_bytes": (
                projection.workspace_peak_bytes
            ),
            "semantic_descriptor_projection_admitted": True,
            "semantic_descriptor_projection_released": True,
        }
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _projection_coordinates(width: int, descriptor_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate = torch.arange(width, dtype=torch.int64)
    # Arithmetic stays below signed-int64 overflow for practical decoder widths.
    mixed = ((coordinate % 2_147_483_647) * 1_103_515_245 + 12_345) % 2_147_483_647
    buckets = torch.remainder(mixed, descriptor_dim)
    signs = torch.where(
        torch.remainder(mixed // 97 + coordinate * 17, 2) == 0,
        1.0,
        -1.0,
    ).to(dtype=torch.float32)
    return buckets, signs


def project_active_decoder_signatures(
    *,
    active_decoder_rows: ActiveDecoderRows,
    decoder_provider: object,
    candidate_row_indices: torch.Tensor,
    candidate_features: torch.Tensor,
    descriptor_dim: int,
) -> DecoderSignatureProjection:
    """Project selected active-occurrence decoder rows without provider I/O."""

    if active_decoder_rows.released or not active_decoder_rows.sealed:
        raise RuntimeError("decoder descriptor capture requires sealed resident rows")
    descriptor_dim = int(descriptor_dim)
    if descriptor_dim <= 0:
        raise ValueError("descriptor_dim must be positive")
    indices = candidate_row_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    features = candidate_features.detach().to(device="cpu", dtype=torch.int64)
    if features.ndim != 2 or tuple(features.shape[1:]) != (3,):
        raise ValueError("candidate_features must have shape (N, 3)")
    if int(features.shape[0]) != int(indices.numel()):
        raise ValueError("candidate features and row indices must align")
    if len(set(indices.tolist())) != int(indices.numel()):
        raise ValueError("candidate row indices must be unique")

    candidate_count = int(indices.numel())
    output_bytes = candidate_count * descriptor_dim * torch.float32.itemsize
    static_bytes = int(
        indices.numel() * indices.element_size()
        + features.numel() * features.element_size()
        + indices.numel() * torch.bool.itemsize
        + indices.numel() * torch.float32.itemsize  # final row norms
    )
    if output_bytes + static_bytes > PROJECTION_MAX_BYTES:
        raise MemoryError("decoder descriptor output exceeds the 64 MiB projection bound")
    output = torch.zeros((candidate_count, descriptor_dim), dtype=torch.float32)
    covered = torch.zeros(indices.numel(), dtype=torch.bool)
    # `isfinite(output)` is the largest final-validation temporary.  The norm
    # vector is retained in static bytes above.
    workspace_peak = candidate_count * descriptor_dim * torch.bool.itemsize

    for layer in active_decoder_rows.layers:
        if layer is None:
            continue
        selected_positions = torch.where(
            (indices >= layer.global_row_start) & (indices < layer.global_row_end)
        )[0]
        if selected_positions.numel() == 0:
            continue
        source_layers = features[selected_positions, 0]
        if not bool(torch.all(source_layers == layer.source_layer)):
            raise ValueError("candidate feature layer disagrees with resident decoder span")
        local_indices = indices[selected_positions] - layer.global_row_start
        row_width = int(layer.rows.shape[1] * layer.rows.shape[2])
        native_row_bytes = row_width * int(layer.rows.element_size())
        fp32_row_bytes = row_width * torch.float32.itemsize
        per_row_workspace = native_row_bytes + fp32_row_bytes + descriptor_dim * 4
        coordinate_bytes = row_width * (torch.int64.itemsize + torch.float32.itemsize)
        coordinate_generation_peak_bytes = (
            row_width * PROJECTION_COORDINATE_PEAK_BYTES_PER_COLUMN
        )
        # Boolean membership plus selected/source/local/device index vectors
        # coexist with both coordinate generation and row projection.
        layer_index_bytes = int(
            indices.numel() * 3 * torch.bool.itemsize
            + selected_positions.numel() * 4 * torch.int64.itemsize
        )
        coordinate_phase_workspace = (
            coordinate_generation_peak_bytes + layer_index_bytes
        )
        if (
            output_bytes
            + static_bytes
            + coordinate_phase_workspace
            > PROJECTION_MAX_BYTES
        ):
            raise MemoryError(
                "decoder descriptor coordinate generation exceeds the projection workspace bound"
            )
        available = (
            PROJECTION_MAX_BYTES
            - output_bytes
            - static_bytes
            - coordinate_bytes
            - layer_index_bytes
        )
        chunk_rows = available // max(1, per_row_workspace)
        if chunk_rows < 1:
            raise MemoryError("one decoder descriptor row exceeds the projection workspace bound")
        buckets, signs = _projection_coordinates(row_width, descriptor_dim)
        actual_coordinate_bytes = int(
            buckets.numel() * buckets.element_size() + signs.numel() * signs.element_size()
        )
        if actual_coordinate_bytes != coordinate_bytes:
            raise RuntimeError("decoder projection coordinate byte estimate changed")
        workspace_peak = max(workspace_peak, coordinate_phase_workspace)
        for start in range(0, int(local_indices.numel()), int(chunk_rows)):
            stop = min(int(local_indices.numel()), start + int(chunk_rows))
            local = local_indices[start:stop].to(device=layer.rows.device)
            gathered = layer.rows.index_select(0, local)
            flattened = gathered.detach().to(device="cpu", dtype=torch.float32).reshape(
                stop - start, row_width
            )
            flattened.mul_(signs)
            projected = torch.zeros((stop - start, descriptor_dim), dtype=torch.float32)
            projected.index_add_(1, buckets, flattened)
            destinations = selected_positions[start:stop]
            output[destinations] = projected
            covered[destinations] = True
            workspace_peak = max(
                workspace_peak,
                int(gathered.numel() * gathered.element_size())
                + int(flattened.numel() * flattened.element_size())
                + int(projected.numel() * projected.element_size())
                + coordinate_bytes
                + layer_index_bytes,
            )

    if not bool(torch.all(covered)):
        missing = indices[~covered].tolist()
        raise RuntimeError(f"resident decoder rows do not cover candidate indices: {missing}")
    norms = output.norm(dim=1)
    if bool(torch.any(~torch.isfinite(output))) or bool(torch.any(norms <= 0)):
        raise ValueError("projected decoder signatures must be finite and non-zero")
    output.div_(norms[:, None])
    required_bytes = (
        output_bytes
        + static_bytes
        + workspace_peak
    )
    if required_bytes > PROJECTION_MAX_BYTES:
        raise MemoryError("decoder descriptor projection exceeded its admitted workspace bound")

    source_fingerprint = _sha256_json(provider_fingerprint(decoder_provider))
    projection_fingerprint = _sha256_json(
        {
            "projection_id": PROJECTION_ID,
            "descriptor_dim": descriptor_dim,
            "accumulator_dtype": "float32",
            "normalization": "row_l2",
        }
    )
    evidence_digest = hashlib.sha256()
    evidence_digest.update(source_fingerprint.encode("ascii"))
    evidence_digest.update(projection_fingerprint.encode("ascii"))
    evidence_digest.update(features.contiguous().numpy().tobytes())
    evidence_digest.update(output.contiguous().numpy().tobytes())
    return DecoderSignatureProjection(
        sketch=output,
        descriptor_kind=DESCRIPTOR_KIND,
        descriptor_scope=DESCRIPTOR_SCOPE,
        decoder_source_fingerprint=source_fingerprint,
        decoder_evidence_fingerprint=f"sha256:{evidence_digest.hexdigest()}",
        projection_fingerprint=projection_fingerprint,
        max_bytes=PROJECTION_MAX_BYTES,
        required_bytes=required_bytes,
        workspace_peak_bytes=workspace_peak,
    )
