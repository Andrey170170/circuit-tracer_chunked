"""
Unified attribution interface that routes to the correct implementation based on the ReplacementModel backend.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import torch

from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.graph import Graph

if TYPE_CHECKING:
    from circuit_tracer.attribution.targets import TargetSpec
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )


def _count_active_features_by_axis(activation_matrix: torch.Tensor, axis: int) -> list[int]:
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


def _cleanup_attribution_context(ctx) -> None:
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
    """Run only Phase 0 setup and return compact feature-count statistics.

    The returned dict is intentionally small and count-based so downstream analysis can
    derive scaling metrics without materializing the full attribution graph.
    """

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
            "active_features_by_layer": _count_active_features_by_axis(activation_matrix, axis=0),
            "active_features_by_token": _count_active_features_by_axis(activation_matrix, axis=1),
            "phase0_encode_seconds": transcoder_stats.get("encode_sparse_seconds"),
            "phase0_reconstruction_seconds": transcoder_stats.get("reconstruction_seconds"),
        }
    finally:
        if ctx is not None:
            _cleanup_attribution_context(ctx)


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: "NNSightReplacementModel | TransformerLensReplacementModel",
    *,
    attribution_targets: "Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None" = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: SparsificationConfig | None = None,
    chunked_feature_replay_window: int = 4,
    error_vector_prefetch_lookahead: int = 2,
    stage_encoder_vecs_on_cpu: bool | None = None,
    stage_error_vectors_on_cpu: bool | None = None,
    row_subchunk_size: int | None = None,
    plan_feature_batch_size: bool = False,
    auto_scale_feature_batch_size: bool = False,
    feature_batch_size_max: int | None = None,
    feature_batch_target_reserved_fraction: float = 0.9,
    feature_batch_min_free_fraction: float = 0.05,
    feature_batch_probe_batches: int = 1,
    exact_trace_internal_dtype: Literal["fp32", "fp64"] = "fp32",
) -> Graph:
    """Compute an attribution graph for *prompt*.

    This function automatically routes to the correct attribution implementation
    based on the type of ReplacementModel provided.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel`` (either nnsight or transformerlens backend)
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or
                            tuple[str, float, torch.Tensor]) with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        feature_batch_size: Optional override for NNSight feature-attribution batches.
        logit_batch_size: Optional override for NNSight logit-attribution batches.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        profile: Whether to emit batch-level diagnostic profiling logs.
        profile_log_interval: Log every N attribution batches when profiling.
        diagnostic_feature_cap: Optional debug-only early cap on active features before
            attribution rows are computed. This changes attribution semantics and should
            only be used for profiling/scaling experiments.
        sparsification: Optional candidate-screening config. When provided, phase 0
            keeps only retained feature candidates before reconstruction, and later
            attribution phases reuse the same candidate set.
        exact_trace_internal_dtype: Internal dtype used by compact exact-trace
            normalization/ranking internals ("fp32" or "fp64"). Defaults to
            ``"fp32"`` on the post-fix stable path.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    planner_enabled = bool(plan_feature_batch_size or auto_scale_feature_batch_size)
    if planner_enabled:
        raise ValueError(
            "Phase-4 feature batch planner is unsupported via circuit_tracer.attribution.attribute(). "
            "Use the NNSight entrypoint with compact_output=True on exact_chunked_decoder paths."
        )

    if model.backend == "nnsight":
        from .attribute_nnsight import attribute as attribute_nnsight

        return attribute_nnsight(
            prompt=prompt,
            model=model,  # type: ignore[arg-type]
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            feature_batch_size=feature_batch_size,
            logit_batch_size=logit_batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            update_interval=update_interval,
            profile=profile,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            sparsification=sparsification,
            chunked_feature_replay_window=chunked_feature_replay_window,
            error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
            stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
            stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
            row_subchunk_size=row_subchunk_size,
            plan_feature_batch_size=plan_feature_batch_size,
            auto_scale_feature_batch_size=auto_scale_feature_batch_size,
            feature_batch_size_max=feature_batch_size_max,
            feature_batch_target_reserved_fraction=feature_batch_target_reserved_fraction,
            feature_batch_min_free_fraction=feature_batch_min_free_fraction,
            feature_batch_probe_batches=feature_batch_probe_batches,
            exact_trace_internal_dtype=exact_trace_internal_dtype,
        )
    else:
        from .attribute_transformerlens import attribute as attribute_transformerlens

        return attribute_transformerlens(
            prompt=prompt,
            model=model,  # type: ignore[arg-type]
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            update_interval=update_interval,
            profile=profile,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            sparsification=sparsification,
        )
