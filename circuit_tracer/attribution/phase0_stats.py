"""Compact Phase-0-only attribution statistics."""

from typing import TYPE_CHECKING, cast

import torch

from circuit_tracer.attribution.sparsification import SparsificationConfig

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )


def _count_active_features_by_axis(
    activation_matrix: torch.Tensor, axis: int
) -> list[int]:
    activation_matrix = activation_matrix.coalesce()
    axis_indices = activation_matrix.indices()[axis].detach().cpu()
    axis_size = int(activation_matrix.shape[axis])
    if axis_indices.numel() == 0:
        return [0] * axis_size
    return torch.bincount(axis_indices, minlength=axis_size).tolist()


def _infer_prompt_token_count(
    prompt: str | torch.Tensor | list[int],
    model: "NNSightReplacementModel | TransformerLensReplacementModel",
) -> int | None:
    if isinstance(prompt, list):
        return len(prompt)
    if isinstance(prompt, torch.Tensor):
        return int(prompt.numel()) if prompt.ndim > 0 else 1

    ensure_tokenized = getattr(model, "ensure_tokenized", None)
    if callable(ensure_tokenized):
        tokens = ensure_tokenized(prompt)
        if isinstance(tokens, torch.Tensor):
            return int(tokens.numel())
    return None


def _cleanup_attribution_context(ctx: object) -> None:
    cleanup = getattr(ctx, "cleanup", None)
    if callable(cleanup):
        cleanup()
        return

    clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
    if callable(clear_decoder_cache):
        clear_decoder_cache()


def attribute_phase0_stats(
    prompt: str | torch.Tensor | list[int],
    model: "NNSightReplacementModel | TransformerLensReplacementModel",
    *,
    sparsification: SparsificationConfig | None = None,
) -> dict[str, object]:
    """Run setup only and return compact active-feature counts and timings."""

    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()

    ctx = None
    setup_prompt: str | torch.Tensor = (
        torch.tensor(prompt, dtype=torch.long) if isinstance(prompt, list) else prompt
    )

    try:
        if getattr(model, "backend", None) == "nnsight":
            ctx = cast("NNSightReplacementModel", model).setup_attribution(
                setup_prompt,
                sparsification=sparsification,
                retain_full_logits=False,
            )
        else:
            ctx = cast("TransformerLensReplacementModel", model).setup_attribution(
                setup_prompt,
                sparsification=sparsification,
            )
        activation_matrix = ctx.activation_matrix.coalesce()
        setup_stats = getattr(ctx, "setup_diagnostic_stats", None) or {}

        transcoder_stats: dict[str, object] = {}
        get_snapshot = getattr(model.transcoders, "get_diagnostic_snapshot", None)
        if callable(get_snapshot):
            snapshot = get_snapshot()
            if isinstance(snapshot, dict):
                transcoder_stats = snapshot

        inferred_token_count = _infer_prompt_token_count(prompt, model)
        token_count = int(
            setup_stats.get("token_count", inferred_token_count or activation_matrix.shape[1])
        )

        return {
            "token_count": token_count,
            "prompt_token_count": token_count,
            "total_active_features": int(activation_matrix._nnz()),
            "active_features_by_layer": _count_active_features_by_axis(
                activation_matrix, axis=0
            ),
            "active_features_by_token": _count_active_features_by_axis(
                activation_matrix, axis=1
            ),
            "phase0_encode_seconds": transcoder_stats.get("encode_sparse_seconds"),
            "phase0_reconstruction_seconds": transcoder_stats.get(
                "reconstruction_seconds"
            ),
        }
    finally:
        if ctx is not None:
            _cleanup_attribution_context(ctx)
