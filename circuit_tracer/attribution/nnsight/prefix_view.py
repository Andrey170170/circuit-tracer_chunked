"""Prefix-view metadata, trace-input, masking, and compact-output helpers."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Mapping, TypedDict

import numpy as np
import torch

from circuit_tracer.attribution.targets import TargetSpec


class PrefixViewMetadata(TypedDict, total=False):
    schema_version: int
    mode: str
    target_position: int
    prefix_token_count: int
    output_position: int
    target_token_ids: list[int]
    prefix_token_ids_sha256: str
    input_token_count: int
    full_sequence_token_count: int
    input_token_ids_sha256: str
    trace_id: str
    trajectory_id: str


def _hash_token_ids(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(token_id) for token_id in token_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens_from_prompt_for_prefix_view(prompt: str | torch.Tensor | list[int]) -> list[int]:
    if isinstance(prompt, str):
        raise ValueError("prefix_view_metadata validation requires token-id prompt, not str")
    if isinstance(prompt, torch.Tensor):
        tensor = prompt.detach().cpu().reshape(-1)
        return [int(token_id) for token_id in tensor.tolist()]
    return [int(token_id) for token_id in prompt]


def _token_ids_from_attribution_targets(
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None,
) -> list[int]:
    if attribution_targets is None:
        return []
    if isinstance(attribution_targets, torch.Tensor):
        return [
            int(token_id) for token_id in attribution_targets.detach().cpu().reshape(-1).tolist()
        ]
    token_ids: list[int] = []
    for target in attribution_targets:
        if isinstance(target, int):
            token_ids.append(int(target))
        elif isinstance(target, np.integer):
            token_ids.append(int(target))
        else:
            raise ValueError(
                "prefix_view_metadata target validation requires integer token-id "
                "attribution_targets"
            )
    return token_ids


def validate_prefix_view_metadata(
    *,
    prompt: str | torch.Tensor | list[int],
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None,
    prefix_view_metadata: Mapping[str, Any] | None,
) -> PrefixViewMetadata | None:
    """Validate independent-prefix metadata without changing attribution semantics."""
    if prefix_view_metadata is None:
        return None
    prefix_tokens = _tokens_from_prompt_for_prefix_view(prompt)
    target_ids = _token_ids_from_attribution_targets(attribution_targets)
    normalized: PrefixViewMetadata = {
        "schema_version": int(prefix_view_metadata.get("schema_version", 1)),
        "mode": str(prefix_view_metadata.get("mode", "independent_prefix")),
        "target_position": int(prefix_view_metadata["target_position"]),
        "prefix_token_count": int(prefix_view_metadata["prefix_token_count"]),
        "target_token_ids": [
            int(token_id) for token_id in prefix_view_metadata["target_token_ids"]
        ],
        "prefix_token_ids_sha256": str(prefix_view_metadata["prefix_token_ids_sha256"]),
    }
    for key in ("trace_id", "trajectory_id"):
        if key in prefix_view_metadata and prefix_view_metadata[key] is not None:
            normalized[key] = str(prefix_view_metadata[key])
    for key in ("output_position", "input_token_count", "full_sequence_token_count"):
        if key in prefix_view_metadata and prefix_view_metadata[key] is not None:
            normalized[key] = int(prefix_view_metadata[key])
    if (
        "input_token_ids_sha256" in prefix_view_metadata
        and prefix_view_metadata["input_token_ids_sha256"] is not None
    ):
        normalized["input_token_ids_sha256"] = str(prefix_view_metadata["input_token_ids_sha256"])
    if normalized["mode"] not in ("independent_prefix", "full_sequence_target_position"):
        raise ValueError(
            "prefix_view_metadata mode must be 'independent_prefix' or "
            f"'full_sequence_target_position' (got {normalized['mode']!r})"
        )
    if normalized["mode"] == "independent_prefix" and normalized["prefix_token_count"] != len(
        prefix_tokens
    ):
        raise ValueError(
            "prefix_view_metadata prefix_token_count does not match prompt token count "
            f"({normalized['prefix_token_count']} != {len(prefix_tokens)})"
        )
    if normalized["prefix_token_count"] > len(prefix_tokens):
        raise ValueError(
            "prefix_view_metadata prefix_token_count exceeds prompt token count "
            f"({normalized['prefix_token_count']} > {len(prefix_tokens)})"
        )
    if normalized["target_position"] != normalized["prefix_token_count"]:
        raise ValueError(
            "prefix_view_metadata target_position must equal prefix_token_count "
            f"({normalized['target_position']} != {normalized['prefix_token_count']})"
        )
    expected_output_position = normalized["target_position"] - 1
    if (
        "output_position" in normalized
        and normalized["output_position"] != expected_output_position
    ):
        raise ValueError(
            "prefix_view_metadata output_position must equal target_position - 1 "
            f"({normalized['output_position']} != {expected_output_position})"
        )
    if normalized["target_token_ids"] != target_ids:
        raise ValueError(
            "prefix_view_metadata target_token_ids do not match attribution_targets "
            f"({normalized['target_token_ids']} != {target_ids})"
        )
    prefix_for_hash = prefix_tokens[: normalized["prefix_token_count"]]
    actual_hash = _hash_token_ids(prefix_for_hash)
    if normalized["prefix_token_ids_sha256"] != actual_hash:
        raise ValueError(
            "prefix_view_metadata prefix_token_ids_sha256 does not match prompt tokens"
        )
    for key in ("input_token_count", "full_sequence_token_count"):
        if key in normalized and normalized[key] != len(prefix_tokens):
            raise ValueError(
                f"prefix_view_metadata {key} does not match prompt token count "
                f"({normalized[key]} != {len(prefix_tokens)})"
            )
    if "input_token_ids_sha256" in normalized:
        input_hash = _hash_token_ids(prefix_tokens)
        if normalized["input_token_ids_sha256"] != input_hash:
            raise ValueError(
                "prefix_view_metadata input_token_ids_sha256 does not match prompt tokens"
            )
    return normalized


def _resolve_prefix_view_output_position(
    normalized_prefix_view_metadata: PrefixViewMetadata | None,
    output_position: int | None,
) -> int | None:
    if normalized_prefix_view_metadata is None:
        return output_position

    effective_output_position = None if output_position is None else int(output_position)
    metadata_output_position = normalized_prefix_view_metadata.get("output_position")
    if metadata_output_position is not None:
        metadata_output_position = int(metadata_output_position)
        if (
            effective_output_position is not None
            and effective_output_position != metadata_output_position
        ):
            raise ValueError(
                "output_position does not match prefix_view_metadata output_position "
                f"({effective_output_position} != {metadata_output_position})"
            )

    if normalized_prefix_view_metadata.get("mode") == "full_sequence_target_position":
        expected_output_position = int(normalized_prefix_view_metadata["target_position"]) - 1
        if effective_output_position is None:
            return expected_output_position
        if effective_output_position != expected_output_position:
            raise ValueError(
                "output_position must equal target_position - 1 for full_sequence_target_position "
                f"({effective_output_position} != {expected_output_position})"
            )

    return effective_output_position


def _resolve_prefix_view_trace_input_ids(
    input_ids: torch.Tensor,
    normalized_prefix_view_metadata: PrefixViewMetadata | None,
) -> tuple[torch.Tensor, int | None]:
    """Return the effective token IDs to feed to NNsight traces.

    Full-sequence prefix views carry the complete audited input for provenance,
    but causal trace execution only needs the prefix visible to the target.
    """

    if (
        normalized_prefix_view_metadata is None
        or normalized_prefix_view_metadata.get("mode") != "full_sequence_target_position"
    ):
        return input_ids, None

    prefix_view_length = int(normalized_prefix_view_metadata["target_position"])
    return input_ids[:prefix_view_length].contiguous(), prefix_view_length


def _apply_prefix_view_activation_mask(ctx: Any, target_position: int) -> dict[str, int]:
    apply_prefix_view_state = getattr(ctx, "apply_prefix_view_state", None)
    if not callable(apply_prefix_view_state):
        raise RuntimeError("Attribution context does not support prefix-view state truncation")
    return apply_prefix_view_state(int(target_position))


def _compact_selected_feature_columns(
    selected_features: torch.Tensor,
    *,
    n_feature_columns: int,
) -> torch.Tensor:
    """Return selected compact feature columns visible to the row store.

    Full-sequence target-position traces may retain full-sequence bookkeeping for
    audit metadata while the compact feature row store is intentionally laid out
    like the causal prefix view.  Drop any stale/full-sequence feature ordinals
    before materializing typed compact graph columns; keeping them would either
    address columns beyond the prefix-visible row-store width or admit future
    position nodes into the compact graph.
    """
    selected_features = selected_features.detach().to(device="cpu", dtype=torch.long)
    if selected_features.numel() == 0:
        return selected_features
    return selected_features[
        (selected_features >= 0) & (selected_features < int(n_feature_columns))
    ].contiguous()


def _compact_nonfeature_column_counts(
    *, n_layers: int, compact_token_count: int
) -> tuple[int, int, int]:
    """Return error/token/total nonfeature column counts for compact output."""
    compact_token_count = int(compact_token_count)
    if compact_token_count < 0:
        raise ValueError("compact_token_count must be non-negative")
    n_error_columns = int(n_layers) * compact_token_count
    n_token_columns = compact_token_count
    return n_error_columns, n_token_columns, n_error_columns + n_token_columns


def validate_compact_prefix_view_output(compact: Mapping[str, Any], *, n_layers: int) -> None:
    """Validate compact-output shape/position invariants for prefix-view traces."""
    metadata = compact.get("prefix_view_metadata")
    if not isinstance(metadata, Mapping):
        return
    target_position = int(metadata["target_position"])
    prefix_len = int(metadata["prefix_token_count"])
    if int(compact["n_token_nodes"]) != prefix_len:
        raise ValueError("compact prefix-view n_token_nodes does not match prefix length")
    if int(compact["n_error_nodes"]) != int(n_layers) * prefix_len:
        raise ValueError("compact prefix-view n_error_nodes does not match prefix length")
    for key in ("feature_token_edges", "logit_token_edges"):
        if int(compact[key].shape[1]) != prefix_len:
            raise ValueError(f"compact prefix-view {key} width does not match prefix length")
    for key in ("feature_error_edges", "logit_error_edges"):
        if int(compact[key].shape[1]) != int(n_layers) * prefix_len:
            raise ValueError(f"compact prefix-view {key} width does not match prefix length")
    for key in (
        "active_features",
        "selected_features",
        "feature_row_node_indices",
        "logit_row_node_indices",
    ):
        tensor = compact.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if key == "active_features":
            positions = tensor[:, 1]
        elif key == "selected_features":
            active = compact.get("active_features")
            if not isinstance(active, torch.Tensor) or active.numel() == 0:
                continue
            positions = active[tensor.to(dtype=torch.long), 1]
        else:
            positions = (tensor.to(dtype=torch.long) % prefix_len) if prefix_len else tensor
        if bool(torch.any(positions >= target_position)):
            raise ValueError(f"compact prefix-view {key} contains future positions")
