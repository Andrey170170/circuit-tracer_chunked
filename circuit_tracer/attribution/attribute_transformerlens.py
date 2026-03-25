"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import logging
import time
from collections.abc import Sequence
from typing import Literal

import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.graph import Graph, compute_partial_influences
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.utils.disk_offload import offload_modules
from circuit_tracer.utils.telemetry import (
    diff_numeric_metrics,
    format_memory_snapshot,
    format_numeric_metrics,
)


def _log_phase_metrics(logger, label: str, phase_start: float, device, **extra):
    logger.info(
        f"{label} completed in {time.time() - phase_start:.2f}s | "
        f"{format_memory_snapshot(device=device, extra=extra)}"
    )


def _snapshot_diagnostics(obj) -> dict[str, object] | None:
    if obj is None or not hasattr(obj, "get_diagnostic_snapshot"):
        return None
    return obj.get_diagnostic_snapshot()


def _log_batch_profile(
    logger,
    label: str,
    batch_idx: int,
    total_batches: int,
    elapsed: float,
    ctx_before: dict[str, object] | None,
    ctx_after: dict[str, object] | None,
    transcoder_before: dict[str, object] | None,
    transcoder_after: dict[str, object] | None,
):
    parts = [f"{label} batch {batch_idx}/{total_batches} in {elapsed:.2f}s"]
    ctx_delta = diff_numeric_metrics(ctx_before, ctx_after) if ctx_after is not None else {}
    transcoder_delta = (
        diff_numeric_metrics(transcoder_before, transcoder_after)
        if transcoder_after is not None
        else {}
    )
    if ctx_delta:
        parts.append(f"ctx[{format_numeric_metrics(ctx_delta, limit=12)}]")
    if transcoder_delta:
        parts.append(f"transcoder[{format_numeric_metrics(transcoder_delta, limit=12)}]")
    logger.info(" | ".join(parts))


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: TransformerLensReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
) -> Graph:
    """Compute an attribution graph for *prompt* using TransformerLens backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``TransformerLensReplacementModel``
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or tuple)
                            with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        profile: Whether to emit batch-level diagnostic profiling logs.
        profile_log_interval: Log every N batches when profiling.
        diagnostic_feature_cap: Optional debug-only early cap on active features.
            This changes attribution semantics and should only be used for profiling.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if (verbose or profile) and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            profile=profile,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _run_attribution(
    model,
    prompt,
    attribution_targets,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    logger,
    update_interval=4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)

    if profile:
        reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
        if callable(reset_diagnostics):
            reset_diagnostics()
        configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
        if callable(configure_trace_logging):
            configure_trace_logging(logger.info)
        logger.info(
            "Profiling enabled | "
            f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
            f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
            f"exact_chunked_decoder={getattr(model.transcoders, 'exact_chunked_decoder', False)} | "
            f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
            f"prompt_tokens={input_ids.shape[-1]} | attribution_batch_size={batch_size}"
        )

    ctx = model.setup_attribution(input_ids)
    if hasattr(ctx, "set_diagnostic_mode"):
        ctx.set_diagnostic_mode(profile)
    configure_ctx_trace_logging = getattr(ctx, "configure_trace_logging", None)
    if callable(configure_ctx_trace_logging):
        configure_ctx_trace_logging(logger.info if profile else None)

    if diagnostic_feature_cap is not None and diagnostic_feature_cap > 0:
        before_cap, after_cap = ctx.apply_diagnostic_feature_cap(diagnostic_feature_cap)
        logger.info(
            f"Diagnostic feature cap applied before attribution rows: {before_cap} -> {after_cap} active features"
        )
    activation_matrix = ctx.activation_matrix

    _log_phase_metrics(
        logger,
        "Precomputation",
        phase_start,
        model.cfg.device,
        active_features=ctx.activation_matrix._nnz(),
    )
    if profile:
        if getattr(ctx, "setup_diagnostic_stats", None):
            logger.info(
                f"Phase 0 setup diagnostics | {format_numeric_metrics(ctx.setup_diagnostic_stats, limit=20)}"
            )
        transcoder_snapshot = _snapshot_diagnostics(model.transcoders)
        if transcoder_snapshot:
            logger.info(
                f"Precompute diagnostics | {format_numeric_metrics(transcoder_snapshot, limit=20)}"
            )
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")

    if offload and not getattr(model.transcoders, "exact_chunked_decoder", False):
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    _log_phase_metrics(logger, "Forward pass", phase_start, model.cfg.device)

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.logits[0, -1],
        unembed_proj=model.unembed.W_U,
        tokenizer=model.tokenizer,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    log_attribution_target_info(targets, attribution_targets, logger)

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(targets)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    _log_phase_metrics(
        logger,
        "Input vector build",
        phase_start,
        model.cfg.device,
        edge_matrix_shape=f"{tuple(edge_matrix.shape)}",
        edge_matrix_dtype=edge_matrix.dtype,
    )

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    total_logit_batches = max((len(targets) + batch_size - 1) // batch_size, 1)
    for i in range(0, len(targets), batch_size):
        batch = targets.logit_vectors[i : i + batch_size]
        ctx_before = _snapshot_diagnostics(ctx) if profile else None
        transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
        batch_start = time.perf_counter()
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
            phase_label="phase3_logits",
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
        if profile and ((i // batch_size) + 1) % profile_log_interval == 0:
            _log_batch_profile(
                logger,
                "Phase 3",
                (i // batch_size) + 1,
                total_logit_batches,
                time.perf_counter() - batch_start,
                ctx_before,
                _snapshot_diagnostics(ctx),
                transcoder_before,
                _snapshot_diagnostics(model.transcoders),
            )
    _log_phase_metrics(logger, "Logit attributions", phase_start, model.cfg.device)

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st],
                targets.logit_probabilities,
                row_to_node_index[:st],
                device=edge_matrix.device,
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            ctx_before = _snapshot_diagnostics(ctx) if profile else None
            transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
            batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.encoder_vecs[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
                phase_label="phase4_features",
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))
            if profile:
                batch_number = max(1, (n_visited + batch_size - 1) // batch_size)
                if batch_number % profile_log_interval == 0:
                    _log_batch_profile(
                        logger,
                        "Phase 4",
                        batch_number,
                        max((max_feature_nodes + batch_size - 1) // batch_size, 1),
                        time.perf_counter() - batch_start,
                        ctx_before,
                        _snapshot_diagnostics(ctx),
                        transcoder_before,
                        _snapshot_diagnostics(model.transcoders),
                    )

    pbar.close()
    _log_phase_metrics(
        logger,
        "Feature attributions",
        phase_start,
        model.cfg.device,
        selected_features=int(visited.sum().item()),
    )

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    non_feature_nodes = torch.arange(total_active_feats, total_nodes)
    if max_feature_nodes < total_active_feats:
        col_read = torch.cat([selected_features, non_feature_nodes])
    else:
        col_read = torch.arange(total_nodes)

    final_node_count = len(col_read)
    full_edge_matrix = torch.zeros(final_node_count, final_node_count, dtype=edge_matrix.dtype)
    feature_row_order = row_to_node_index[n_logits:st].argsort()
    full_edge_matrix[:max_feature_nodes] = edge_matrix[n_logits:st][feature_row_order][:, col_read]
    full_edge_matrix[-n_logits:] = edge_matrix[:n_logits, :][:, col_read]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
        vocab_size=targets.vocab_size,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    logger.info(
        f"Attribution completed in {time.time() - start_time:.2f}s | "
        f"{format_memory_snapshot(device=model.cfg.device, extra={'adjacency_shape': tuple(full_edge_matrix.shape)})}"
    )

    return graph
