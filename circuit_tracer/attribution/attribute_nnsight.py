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
from typing import Literal, cast

import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.graph import (
    Graph,
    compute_partial_feature_influences,
    compute_partial_influences,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
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


def _log_memory_boundary(logger, label: str, device, **extra) -> None:
    logger.info(f"{label} | {format_memory_snapshot(device=device, extra=extra)}")


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


def _log_sparsification_profile(logger, stats: dict[str, object]) -> None:
    retained = stats.get("per_layer_retained_counts", {})
    logger.info(
        "Sparsification screening | "
        f"candidates={stats.get('candidate_count_before')}->{stats.get('candidate_count_after')} | "
        f"per_layer_position_topk={stats.get('per_layer_position_topk')} | "
        f"global_cap={stats.get('global_cap')} | "
        f"retained_activation_mass={stats.get('retained_activation_mass', 1.0):.4f} | "
        f"screen_seconds={stats.get('screen_seconds', 0.0):.4f} | "
        f"per_layer_retained={retained}"
    )


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: NNSightReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
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
    compact_output: bool = False,
) -> Graph:
    """Compute an attribution graph for *prompt* using NNSight backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``NNSightReplacementModel``
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
        feature_batch_size: Optional override for feature-attribution batches.
            Defaults to ``batch_size`` when omitted.
        logit_batch_size: Optional override for logit-attribution batches.
            Defaults to ``batch_size`` when omitted.
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
        sparsification: Optional candidate-screening config applied before
            reconstruction and reused by later attribution phases.
        chunked_feature_replay_window: Exact-mode knob controlling how many
            layer grads are buffered before chunked feature replay flush.
        error_vector_prefetch_lookahead: Exact-mode knob controlling staged
            error-vector lookahead window size.
        stage_encoder_vecs_on_cpu: Exact-mode knob to force/disable CPU staging
            of encoder vectors. ``None`` preserves backend default behavior.
        stage_error_vectors_on_cpu: Exact-mode knob to force/disable CPU
            staging of error vectors. ``None`` preserves backend defaults.
        row_subchunk_size: Optional exact-mode knob controlling inner replay
            row subchunk size. ``None`` preserves current behavior (equal to
            decoder chunk size).

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
            feature_batch_size=feature_batch_size,
            logit_batch_size=logit_batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
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
            compact_output=compact_output,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _run_attribution(
    model: NNSightReplacementModel,
    prompt,
    attribution_targets,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
    feature_batch_size: int | None,
    logit_batch_size: int | None,
    max_feature_nodes: int | None,
    offload: Literal["cpu", "disk", None],
    verbose: bool,
    offload_handles,
    logger,
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
    compact_output: bool = False,
):
    start_time = time.time()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if feature_batch_size is not None and feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be > 0 when provided")
    if logit_batch_size is not None and logit_batch_size <= 0:
        raise ValueError("logit_batch_size must be > 0 when provided")
    if chunked_feature_replay_window <= 0:
        raise ValueError("chunked_feature_replay_window must be > 0")
    if error_vector_prefetch_lookahead <= 0:
        raise ValueError("error_vector_prefetch_lookahead must be > 0")
    if row_subchunk_size is not None and row_subchunk_size <= 0:
        raise ValueError("row_subchunk_size must be > 0 when provided")

    effective_feature_batch_size = batch_size if feature_batch_size is None else feature_batch_size
    effective_logit_batch_size = batch_size if logit_batch_size is None else logit_batch_size
    trace_batch_size = max(batch_size, effective_feature_batch_size, effective_logit_batch_size)
    ctx = None

    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)
    _log_memory_boundary(logger, "Phase 0 start", model.device)

    configure_trace_logging = getattr(model.transcoders, "configure_trace_logging", None)
    if callable(configure_trace_logging):
        configure_trace_logging(logger.info if profile else None)

    reset_diagnostics = getattr(model.transcoders, "reset_diagnostic_stats", None)
    if callable(reset_diagnostics):
        reset_diagnostics()

    if profile:
        logger.info(
            "Profiling enabled | "
            f"lazy_encoder={getattr(model.transcoders, 'lazy_encoder', 'n/a')} | "
            f"lazy_decoder={getattr(model.transcoders, 'lazy_decoder', 'n/a')} | "
            f"exact_chunked_decoder={getattr(model.transcoders, 'exact_chunked_decoder', False)} | "
            f"decoder_chunk_size={getattr(model.transcoders, 'decoder_chunk_size', 'n/a')} | "
            f"decoder_cache_bytes={getattr(model.transcoders, 'cross_batch_decoder_cache_bytes', 0)} | "
            f"chunked_feature_replay_window={chunked_feature_replay_window} | "
            f"error_vector_prefetch_lookahead={error_vector_prefetch_lookahead} | "
            f"stage_encoder_vecs_on_cpu={stage_encoder_vecs_on_cpu} | "
            f"stage_error_vectors_on_cpu={stage_error_vectors_on_cpu} | "
            f"row_subchunk_size={row_subchunk_size} | "
            f"prompt_tokens={input_ids.shape[-1]} | feature_batch_size={effective_feature_batch_size} | "
            f"logit_batch_size={effective_logit_batch_size}"
        )

    ctx = model.setup_attribution(
        input_ids,
        sparsification=sparsification,
        retain_full_logits=False,
        chunked_feature_replay_window=chunked_feature_replay_window,
        error_vector_prefetch_lookahead=error_vector_prefetch_lookahead,
        stage_encoder_vecs_on_cpu=stage_encoder_vecs_on_cpu,
        stage_error_vectors_on_cpu=stage_error_vectors_on_cpu,
        row_subchunk_size=row_subchunk_size,
    )
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
    if profile and getattr(ctx, "sparsification_stats", None):
        _log_sparsification_profile(logger, ctx.sparsification_stats)

    try:
        activation_matrix = ctx.activation_matrix

        _log_phase_metrics(
            logger,
            "Precomputation",
            phase_start,
            model.device,
            active_features=ctx.activation_matrix._nnz(),
            logit_retention=getattr(ctx, "logit_retention", "full"),
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

        if (
            offload
            and not model.skip_transcoder
            and not getattr(model.transcoders, "exact_chunked_decoder", False)
        ):
            offload_handles += offload_modules(model.transcoders, offload)

        # Phase 1: forward pass
        logger.info("Phase 1: Running forward pass")
        phase_start = time.time()
        _log_memory_boundary(logger, "Phase 1 start", model.device)
        with model.trace() as tracer:
            with tracer.invoke(input_ids.expand(trace_batch_size, -1)):
                pass

            detach_barrier = tracer.barrier(2)

            model.configure_gradient_flow(tracer)
            model.configure_skip_connection(tracer, barrier=detach_barrier)
            ctx.cache_residual(model, tracer, barrier=detach_barrier)

        _log_phase_metrics(logger, "Forward pass", phase_start, model.device)

        if offload:
            offload_handles += offload_modules(
                [layer.mlp for layer in getattr(model.pre_logit_location, "layers")], offload
            )
            if model.skip_transcoder and not getattr(
                model.transcoders, "exact_chunked_decoder", False
            ):
                offload_handles += offload_modules(model.transcoders, offload)

        # Phase 2: build input vector list
        logger.info("Phase 2: Building input vectors")
        phase2_start = time.time()
        _log_memory_boundary(logger, "Phase 2 start", model.device)
        feat_layers, feat_pos, _ = activation_matrix.indices()
        n_layers, n_pos, _ = activation_matrix.shape
        total_active_feats = activation_matrix._nnz()

        # Create AttributionTargets using NNSight's unembed_weight accessor
        targets = AttributionTargets(
            attribution_targets=attribution_targets,
            logits=ctx.get_last_token_logits()[0],
            unembed_proj=cast(torch.Tensor, model.unembed_weight),  # NNSight uses unembed_weight
            tokenizer=model.tokenizer,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )

        log_attribution_target_info(targets, attribution_targets, logger)

        if offload:
            offload_handles += offload_modules([model.embed_location], offload)
            tied_embeds = (
                model.embed_weight.untyped_storage().data_ptr()  # type:ignore
                == model.unembed_weight.untyped_storage().data_ptr()  # type:ignore
            )
            if not tied_embeds:
                offload_handles += offload_modules([model.lm_head], offload)

        logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
        n_logits = len(targets)
        total_nodes = logit_offset + n_logits

        actual_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
        logger.info(
            f"Will include {actual_max_feature_nodes} of {total_active_feats} feature nodes"
        )

        use_compact_feature_row_store = compact_output and bool(
            getattr(model.transcoders, "exact_chunked_decoder", False)
        )
        if use_compact_feature_row_store:
            feature_edge_matrix = torch.zeros(
                actual_max_feature_nodes + n_logits,
                total_active_feats,
            )
            row_abs_sums = torch.zeros(actual_max_feature_nodes + n_logits)
        else:
            edge_matrix = torch.zeros(actual_max_feature_nodes + n_logits, total_nodes)

        # Maps row indices in edge_matrix to original feature/node indices
        # First populated with logit node IDs, then feature IDs in attribution order
        row_to_node_index = torch.zeros(actual_max_feature_nodes + n_logits, dtype=torch.int32)

        phase2_extra: dict[str, object] = {
            "row_store_mode": "compact_feature" if use_compact_feature_row_store else "dense_full"
        }
        if use_compact_feature_row_store:
            phase2_extra.update(
                feature_edge_matrix_shape=f"{tuple(feature_edge_matrix.shape)}",
                row_abs_sums_shape=f"{tuple(row_abs_sums.shape)}",
                feature_edge_matrix_dtype=feature_edge_matrix.dtype,
            )
        else:
            phase2_extra.update(
                edge_matrix_shape=f"{tuple(edge_matrix.shape)}",
                edge_matrix_dtype=edge_matrix.dtype,
            )

        _log_phase_metrics(
            logger,
            "Input vector build",
            phase2_start,
            model.device,
            **phase2_extra,
        )

        # Phase 3: logit attribution
        logger.info("Phase 3: Computing logit attributions")
        phase3_start = time.time()
        _log_memory_boundary(logger, "Phase 3 start", model.device)
        i = -1
        total_logit_batches = max(
            (len(targets) + effective_logit_batch_size - 1) // effective_logit_batch_size,
            1,
        )
        for i in range(0, len(targets), effective_logit_batch_size):
            batch = targets.logit_vectors[i : i + effective_logit_batch_size]
            ctx_before = _snapshot_diagnostics(ctx) if profile else None
            transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
            batch_start = time.perf_counter()
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), n_layers),
                positions=torch.full((batch.shape[0],), n_pos - 1),
                inject_values=batch,
                phase_label="phase3_logits",
            )
            rows_cpu = rows.cpu()
            if use_compact_feature_row_store:
                end = i + batch.shape[0]
                feature_edge_matrix[i:end] = rows_cpu[:, :total_active_feats]
                row_abs_sums[i:end] = rows_cpu[:, :logit_offset].abs().sum(dim=1)
            else:
                edge_matrix[i : i + batch.shape[0], :logit_offset] = rows_cpu
            row_to_node_index[i : i + batch.shape[0]] = (
                torch.arange(i, i + batch.shape[0]) + logit_offset
            )
            if profile and ((i // effective_logit_batch_size) + 1) % profile_log_interval == 0:
                _log_batch_profile(
                    logger,
                    "Phase 3",
                    (i // effective_logit_batch_size) + 1,
                    total_logit_batches,
                    time.perf_counter() - batch_start,
                    ctx_before,
                    _snapshot_diagnostics(ctx),
                    transcoder_before,
                    _snapshot_diagnostics(model.transcoders),
                )

        _log_phase_metrics(
            logger,
            f"{i + 1} logit attribution(s)",
            phase3_start,
            model.device,
        )
        reset_decoder_cache = getattr(ctx, "reset_decoder_cache", None)
        if callable(reset_decoder_cache):
            reset_decoder_cache()

        # Phase 4: feature attribution
        logger.info("Phase 4: Computing feature attributions")
        phase4_start = time.time()
        _log_memory_boundary(logger, "Phase 4 start", model.device)
        st = n_logits
        visited = torch.zeros(total_active_feats, dtype=torch.bool)
        n_visited = 0

        pbar = tqdm(
            total=actual_max_feature_nodes,
            desc="Feature influence computation",
            disable=not verbose,
        )

        while n_visited < actual_max_feature_nodes:
            if actual_max_feature_nodes == total_active_feats:
                pending = torch.arange(total_active_feats)
            else:
                if use_compact_feature_row_store:
                    feature_influences = compute_partial_feature_influences(
                        feature_edge_matrix[:st],
                        row_abs_sums[:st],
                        targets.logit_probabilities,
                        row_to_node_index[:st],
                        n_feature_nodes=total_active_feats,
                        n_logits=n_logits,
                        device=feature_edge_matrix.device,
                    )
                else:
                    influences = compute_partial_influences(
                        edge_matrix[:st],
                        targets.logit_probabilities,
                        row_to_node_index[:st],
                        device=edge_matrix.device,
                    )
                    feature_influences = influences[:total_active_feats]

                feature_rank = torch.argsort(feature_influences, descending=True).cpu()
                queue_size = min(
                    update_interval * effective_feature_batch_size,
                    actual_max_feature_nodes - n_visited,
                )
                pending = feature_rank[~visited[feature_rank]][:queue_size]

            queue = [
                pending[i : i + effective_feature_batch_size]
                for i in range(0, len(pending), effective_feature_batch_size)
            ]

            for idx_batch in queue:
                n_visited += len(idx_batch)

                ctx_before = _snapshot_diagnostics(ctx) if profile else None
                transcoder_before = _snapshot_diagnostics(model.transcoders) if profile else None
                batch_start = time.perf_counter()
                rows = ctx.compute_batch(
                    layers=feat_layers[idx_batch],
                    positions=feat_pos[idx_batch],
                    inject_values=ctx.materialize_encoder_vectors(idx_batch),
                    retain_graph=n_visited < actual_max_feature_nodes,
                    phase_label="phase4_features",
                )

                end = min(st + effective_feature_batch_size, st + rows.shape[0])
                rows_cpu = rows.cpu()
                if use_compact_feature_row_store:
                    row_count = end - st
                    feature_edge_matrix[st:end] = rows_cpu[:row_count, :total_active_feats]
                    row_abs_sums[st:end] = rows_cpu[:row_count, :logit_offset].abs().sum(dim=1)
                else:
                    edge_matrix[st:end, :logit_offset] = rows_cpu
                row_to_node_index[st:end] = idx_batch
                visited[idx_batch] = True
                st = end
                pbar.update(len(idx_batch))
                if profile:
                    batch_number = max(
                        1,
                        (n_visited + effective_feature_batch_size - 1)
                        // effective_feature_batch_size,
                    )
                    if batch_number % profile_log_interval == 0:
                        _log_batch_profile(
                            logger,
                            "Phase 4",
                            batch_number,
                            max(
                                (actual_max_feature_nodes + effective_feature_batch_size - 1)
                                // effective_feature_batch_size,
                                1,
                            ),
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
            phase4_start,
            model.device,
            selected_features=int(visited.sum().item()),
        )

        # Phase 5: packaging graph / compact output
        selected_features = torch.where(visited)[0]
        if compact_output:
            if use_compact_feature_row_store:
                feature_feature_edges = (
                    feature_edge_matrix[n_logits:st, selected_features].detach().cpu()
                )
                logit_feature_edges = (
                    feature_edge_matrix[:n_logits, selected_features].detach().cpu()
                )
            else:
                feature_feature_edges = edge_matrix[n_logits:st, selected_features].detach().cpu()
                logit_feature_edges = edge_matrix[:n_logits, selected_features].detach().cpu()

            compact_output_result = {
                "input_string": model.tokenizer.decode(input_ids),
                "input_tokens": input_ids.detach().cpu(),
                "logit_targets": targets.logit_targets,
                "logit_probabilities": targets.logit_probabilities.detach().cpu(),
                "vocab_size": targets.vocab_size,
                "active_features": activation_matrix.indices().T.detach().cpu(),
                "activation_values": activation_matrix.values().detach().cpu(),
                "selected_features": selected_features.detach().cpu(),
                "feature_row_node_indices": row_to_node_index[n_logits:st].detach().cpu(),
                "logit_row_node_indices": row_to_node_index[:n_logits].detach().cpu(),
                "feature_feature_edges": feature_feature_edges,
                "logit_feature_edges": logit_feature_edges,
                "cfg": model.config,
                "scan": model.scan,
            }
            if use_compact_feature_row_store:
                del feature_edge_matrix
                del row_abs_sums
            else:
                del edge_matrix
            logger.info(
                "Attribution completed in "
                f"{time.time() - start_time:.2f}s | "
                f"compact_feature_edge_shape={tuple(compact_output_result['feature_feature_edges'].shape)} | "
                f"compact_logit_edge_shape={tuple(compact_output_result['logit_feature_edges'].shape)}"
            )
            return compact_output_result

        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        if actual_max_feature_nodes < total_active_feats:
            col_read = torch.cat([selected_features, non_feature_nodes])
        else:
            col_read = torch.arange(total_nodes)

        final_node_count = len(col_read)
        full_edge_matrix = torch.zeros(final_node_count, final_node_count, dtype=edge_matrix.dtype)
        feature_row_order = row_to_node_index[n_logits:st].argsort()
        full_edge_matrix[:actual_max_feature_nodes] = edge_matrix[n_logits:st][feature_row_order][
            :, col_read
        ]
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
            adjacency_matrix=full_edge_matrix.detach(),
            cfg=model.config,
            scan=model.scan,
        )

        logger.info(
            f"Attribution completed in {time.time() - start_time:.2f}s | "
            f"{format_memory_snapshot(device=model.device, extra={'adjacency_shape': tuple(full_edge_matrix.shape)})}"
        )

        return graph
    finally:
        if ctx is not None:
            _log_memory_boundary(logger, "Teardown start", model.device)
            cleanup = getattr(ctx, "cleanup", None)
            if callable(cleanup):
                cleanup()
            else:
                clear_decoder_cache = getattr(ctx, "clear_decoder_cache", None)
                if callable(clear_decoder_cache):
                    clear_decoder_cache()
            _log_memory_boundary(logger, "Teardown done", model.device)
