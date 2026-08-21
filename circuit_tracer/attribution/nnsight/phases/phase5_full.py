"""Full dense graph assembly for attribution Phase 5."""

from __future__ import annotations

import torch

from circuit_tracer.attribution.nnsight.phases.phase5_types import Phase5Config, Phase5Inputs
from circuit_tracer.graph import Graph


def assemble_full_graph(
    *, inputs: Phase5Inputs, config: Phase5Config, selected_features: torch.Tensor
) -> tuple[Graph, torch.Tensor]:
    """Reorder retained rows and assemble the public dense Graph."""
    graph = inputs.graph
    limits = config.graph_limits
    assert graph.edge_matrix is not None
    non_feature_nodes = torch.arange(limits.total_active_feats, limits.total_nodes)
    if limits.actual_max_feature_nodes < limits.total_active_feats:
        col_read = torch.cat([selected_features, non_feature_nodes])
    else:
        col_read = torch.arange(limits.total_nodes)
    full_edges = torch.zeros(
        len(col_read), len(col_read), dtype=graph.edge_matrix.dtype
    )
    feature_row_order = graph.row_to_node_index[limits.n_logits : limits.st].argsort()
    full_edges[: limits.actual_max_feature_nodes] = graph.edge_matrix[
        limits.n_logits : limits.st
    ][feature_row_order][:, col_read]
    full_edges[-limits.n_logits :] = graph.edge_matrix[: limits.n_logits, :][:, col_read]
    runtime = inputs.runtime
    return Graph(
        input_string=runtime.model.tokenizer.decode(runtime.input_ids[: limits.n_pos]),
        input_tokens=runtime.input_ids[: limits.n_pos],
        logit_targets=runtime.targets.logit_targets,
        logit_probabilities=runtime.targets.logit_probabilities,
        vocab_size=runtime.targets.vocab_size,
        active_features=graph.activation_matrix.indices().T,
        activation_values=graph.activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edges.detach(),
        cfg=runtime.model.config,
        scan=runtime.model.scan,
    ), full_edges
