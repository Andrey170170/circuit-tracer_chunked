"""Phase 5 orchestration for graph assembly, packaging, and publication."""

from __future__ import annotations

import time

from circuit_tracer.attribution.nnsight.phases.phase5_artifacts import (
    package_compact_artifacts,
)
from circuit_tracer.attribution.nnsight.phases.phase5_compact import (
    assemble_compact_graph,
    select_graph_features,
)
from circuit_tracer.attribution.nnsight.phases.phase5_full import assemble_full_graph
from circuit_tracer.attribution.nnsight.phases.phase5_publication import (
    finalize_compact_publication,
    finalize_full_publication,
)
from circuit_tracer.attribution.nnsight.phases.phase5_types import (
    BatchExecutionSummary,
    DiagnosticArtifacts,
    GraphAssemblyLimits,
    GraphAssemblyRuntime,
    GraphAssemblyState,
    GraphOutputOwnership,
    NumericExecutionSummary,
    OutputArtifactPolicy,
    Phase4PolicySummary,
    Phase4TimingSummary,
    Phase4WorkSummary,
    Phase5Config,
    Phase5Inputs,
    Phase5Result,
    ReplayArtifacts,
    RunProvenance,
)

__all__ = [
    "BatchExecutionSummary", "DiagnosticArtifacts", "GraphAssemblyLimits",
    "GraphAssemblyRuntime", "GraphAssemblyState", "GraphOutputOwnership",
    "NumericExecutionSummary", "OutputArtifactPolicy", "Phase4PolicySummary",
    "Phase4TimingSummary", "Phase4WorkSummary", "Phase5Config", "Phase5Inputs",
    "Phase5Result", "ReplayArtifacts", "RunProvenance", "run_phase5",
]


def run_phase5(*, inputs: Phase5Inputs, config: Phase5Config) -> Phase5Result:
    """Sequence Phase 5 domain operations without owning assembly logic."""
    phase5_start = time.perf_counter()
    selected = select_graph_features(inputs=inputs, config=config)
    if config.output_policy.compact_output:
        compact = assemble_compact_graph(
            inputs=inputs, config=config, selected_features=selected
        )
        artifact = package_compact_artifacts(
            assembly=compact, inputs=inputs, config=config
        )
        edge_matrix = finalize_compact_publication(
            artifact=artifact, selected_features=selected.device_indices,
            inputs=inputs, config=config, phase5_start=phase5_start,
        )
        return Phase5Result(artifact, artifact, edge_matrix)

    graph, full_edges = assemble_full_graph(
        inputs=inputs, config=config, selected_features=selected.device_indices
    )
    finalize_full_publication(
        full_edge_matrix=full_edges, inputs=inputs,
        config=config, phase5_start=phase5_start,
    )
    return Phase5Result(graph, None, inputs.graph.edge_matrix)
