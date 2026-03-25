from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SparsificationConfig:
    """Optional feature-screening config for attribution.

    The current implementation keeps the highest-activation candidates per
    ``(layer, position)`` bucket and can optionally apply a final global cap.
    """

    per_layer_position_topk: int | None = None
    global_cap: int | None = None

    def __post_init__(self) -> None:
        if self.per_layer_position_topk is None and self.global_cap is None:
            raise ValueError(
                "SparsificationConfig requires at least one budget: "
                "per_layer_position_topk or global_cap"
            )
        if self.per_layer_position_topk is not None and self.per_layer_position_topk < 1:
            raise ValueError("per_layer_position_topk must be >= 1")
        if self.global_cap is not None and self.global_cap < 1:
            raise ValueError("global_cap must be >= 1")


def _layer_count_dict(layer_ids: torch.Tensor, n_layers: int) -> dict[int, int]:
    if layer_ids.numel() == 0:
        return {layer: 0 for layer in range(n_layers)}
    counts = torch.bincount(layer_ids.to(dtype=torch.long).cpu(), minlength=n_layers)
    return {layer: int(counts[layer].item()) for layer in range(n_layers)}


def select_candidate_feature_indices(
    activation_matrix: torch.Tensor,
    config: SparsificationConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    activation_matrix = activation_matrix.coalesce()
    indices = activation_matrix.indices()
    values = activation_matrix.values()
    nnz = values.numel()
    n_layers, n_pos, _ = activation_matrix.shape
    screen_start = time.perf_counter()

    if nnz == 0:
        return torch.empty(0, dtype=torch.long, device=values.device), {
            "strategy": "per_layer_position_topk",
            "per_layer_position_topk": config.per_layer_position_topk,
            "global_cap": config.global_cap,
            "candidate_count_before": 0,
            "candidate_count_after": 0,
            "retained_activation_mass": 1.0,
            "screen_seconds": time.perf_counter() - screen_start,
            "per_layer_candidate_counts": {layer: 0 for layer in range(n_layers)},
            "per_layer_retained_counts": {layer: 0 for layer in range(n_layers)},
        }

    selected = torch.arange(nnz, device=values.device)

    if config.per_layer_position_topk is not None:
        bucket_ids = indices[0] * n_pos + indices[1]
        bucket_order = torch.argsort(bucket_ids)
        sorted_bucket_ids = bucket_ids[bucket_order]
        sorted_scores = values[bucket_order].abs()
        _, bucket_counts = torch.unique_consecutive(sorted_bucket_ids, return_counts=True)
        keep_in_sorted = torch.zeros(nnz, dtype=torch.bool, device=values.device)

        start = 0
        for count in bucket_counts.tolist():
            end = start + count
            keep_k = min(config.per_layer_position_topk, count)
            if keep_k == count:
                keep_in_sorted[start:end] = True
            else:
                top_local = (
                    torch.topk(sorted_scores[start:end], k=keep_k, sorted=False).indices + start
                )
                keep_in_sorted[top_local] = True
            start = end

        selected = bucket_order[keep_in_sorted]

    if config.global_cap is not None and selected.numel() > config.global_cap:
        top_global = torch.topk(values[selected].abs(), k=config.global_cap, sorted=False).indices
        selected = selected[top_global]

    selected = selected.sort().values
    total_activation_mass = float(values.abs().sum().item())
    retained_activation_mass = (
        float(values[selected].abs().sum().item()) / total_activation_mass
        if total_activation_mass > 0.0
        else 1.0
    )

    stats = {
        "strategy": "per_layer_position_topk",
        "per_layer_position_topk": config.per_layer_position_topk,
        "global_cap": config.global_cap,
        "candidate_count_before": int(nnz),
        "candidate_count_after": int(selected.numel()),
        "retained_activation_mass": retained_activation_mass,
        "screen_seconds": time.perf_counter() - screen_start,
        "per_layer_candidate_counts": _layer_count_dict(indices[0], n_layers),
        "per_layer_retained_counts": _layer_count_dict(indices[0, selected], n_layers),
    }
    return selected, stats


def filter_sparse_activations(
    activation_matrix: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    activation_matrix = activation_matrix.coalesce()
    return torch.sparse_coo_tensor(
        activation_matrix.indices()[:, selected_indices],
        activation_matrix.values()[selected_indices],
        size=activation_matrix.shape,
        device=activation_matrix.device,
        dtype=activation_matrix.dtype,
    ).coalesce()


def filter_chunked_decoder_state(
    chunked_decoder_state: dict[str, torch.Tensor],
    selected_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {key: value[selected_indices] for key, value in chunked_decoder_state.items()}
