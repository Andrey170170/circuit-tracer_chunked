"""Compact graph tensor assembly for attribution Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from circuit_tracer.attribution.nnsight.phases.phase5_types import Phase5Config, Phase5Inputs
from circuit_tracer.attribution.nnsight.prefix_view import (
    _compact_nonfeature_column_counts,
    _compact_selected_feature_columns,
)


@dataclass(frozen=True)
class SelectedGraphFeatures:
    device_indices: torch.Tensor
    cpu_indices: torch.Tensor | None


@dataclass(frozen=True)
class CompactGraphAssembly:
    artifact: dict[str, object]
    selected_features: SelectedGraphFeatures


def select_graph_features(*, inputs: Phase5Inputs, config: Phase5Config) -> SelectedGraphFeatures:
    """Resolve graph columns and annotate semantic descriptors with final membership."""
    selected = torch.where(inputs.graph.visited)[0]
    if config.output_policy.use_compact_feature_row_store:
        assert inputs.graph.feature_row_store is not None
        selected = _compact_selected_feature_columns(
            selected, n_feature_columns=inputs.graph.feature_row_store.n_feature_columns
        )
    descriptors = inputs.diagnostics.feature_semantic_descriptors_payload
    if config.output_policy.capture_feature_semantic_descriptors and isinstance(descriptors, dict):
        from circuit_tracer.attribution.nnsight.phase_support import (
            _annotate_phase4_selection_on_feature_semantic_descriptors,
        )

        _annotate_phase4_selection_on_feature_semantic_descriptors(
            descriptors,
            selected_features=selected,
            active_features=inputs.graph.activation_matrix.indices().T,
            activation_values=inputs.graph.activation_matrix.values(),
        )
    cpu = selected.detach().to(device="cpu", dtype=torch.long) if config.output_policy.compact_output else None
    return SelectedGraphFeatures(selected, cpu)


def _materialize_compact_edges(
    *, inputs: Phase5Inputs, config: Phase5Config, selected_cpu: torch.Tensor,
    error_start: int, token_start: int, logit_start: int, nonfeature_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph, limits = inputs.graph, config.graph_limits
    if config.output_policy.use_compact_feature_row_store:
        assert graph.feature_row_store is not None and graph.nonfeature_row_store is not None
        feature_feature = graph.feature_row_store.materialize_dense_feature_slice(
            row_start=limits.n_logits, row_end=limits.st,
            selected_feature_columns=selected_cpu, phase="phase5")
        logit_feature = graph.feature_row_store.materialize_dense_feature_slice(
            row_start=0, row_end=limits.n_logits,
            selected_feature_columns=selected_cpu, phase="phase5")
        if int(graph.nonfeature_row_store.n_feature_columns) != nonfeature_count:
            raise ValueError("compact nonfeature row-store width does not match prefix-visible error/token column count")
        columns = torch.arange(nonfeature_count, dtype=torch.long)
        feature_nonfeature = graph.nonfeature_row_store.materialize_dense_feature_slice(
            row_start=limits.n_logits, row_end=limits.st,
            selected_feature_columns=columns, phase="phase5")
        logit_nonfeature = graph.nonfeature_row_store.materialize_dense_feature_slice(
            row_start=0, row_end=limits.n_logits,
            selected_feature_columns=columns, phase="phase5")
        n_error = token_start - error_start
        return (feature_feature, logit_feature, feature_nonfeature[:, :n_error],
                feature_nonfeature[:, n_error:], logit_nonfeature[:, :n_error],
                logit_nonfeature[:, n_error:])
    assert graph.edge_matrix is not None
    return (
        graph.edge_matrix[limits.n_logits:limits.st, selected_cpu].detach().cpu(),
        graph.edge_matrix[:limits.n_logits, selected_cpu].detach().cpu(),
        graph.edge_matrix[limits.n_logits:limits.st, error_start:token_start].detach().cpu(),
        graph.edge_matrix[limits.n_logits:limits.st, token_start:logit_start].detach().cpu(),
        graph.edge_matrix[:limits.n_logits, error_start:token_start].detach().cpu(),
        graph.edge_matrix[:limits.n_logits, token_start:logit_start].detach().cpu(),
    )


def assemble_compact_graph(
    *, inputs: Phase5Inputs, config: Phase5Config,
    selected_features: SelectedGraphFeatures,
) -> CompactGraphAssembly:
    """Materialize compact feature/nonfeature edge families and graph identity fields."""
    assert selected_features.cpu_indices is not None
    runtime, graph, limits = inputs.runtime, inputs.graph, config.graph_limits
    active_cpu = graph.activation_matrix.indices().T.detach().cpu()
    n_error, _n_token, n_nonfeature = _compact_nonfeature_column_counts(
        n_layers=int(runtime.model.cfg.n_layers), compact_token_count=limits.n_pos
    )
    error_start = int(active_cpu.shape[0])
    token_start = error_start + n_error
    logit_start = token_start + limits.n_pos
    edges = _materialize_compact_edges(
        inputs=inputs, config=config, selected_cpu=selected_features.cpu_indices,
        error_start=error_start, token_start=token_start, logit_start=logit_start,
        nonfeature_count=n_nonfeature,
    )
    artifact: dict[str, object] = {
        "input_string": runtime.model.tokenizer.decode(runtime.input_ids),
        "input_tokens": runtime.input_ids[:limits.n_pos].detach().cpu(),
        "full_input_tokens": runtime.input_ids.detach().cpu(),
        "logit_targets": runtime.targets.logit_targets,
        "logit_probabilities": runtime.targets.logit_probabilities.detach().cpu(),
        "vocab_size": runtime.targets.vocab_size,
        "active_features": active_cpu,
        "activation_values": graph.activation_matrix.values().detach().cpu(),
        "selected_features": selected_features.cpu_indices,
        "feature_row_node_indices": graph.row_to_node_index[limits.n_logits:limits.st].detach().cpu(),
        "logit_row_node_indices": graph.row_to_node_index[:limits.n_logits].detach().cpu(),
        "feature_feature_edges": edges[0], "logit_feature_edges": edges[1],
        "feature_error_edges": edges[2], "feature_token_edges": edges[3],
        "logit_error_edges": edges[4], "logit_token_edges": edges[5],
        "n_error_nodes": n_error, "n_token_nodes": limits.n_pos,
    }
    return CompactGraphAssembly(artifact, selected_features)
