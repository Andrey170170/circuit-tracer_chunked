from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from pathlib import Path

import pytest
import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.batch_execution import BatchAttributionRequest
from circuit_tracer.attribution.nnsight.context_state import ContextExecutionPolicy
from circuit_tracer.attribution.nnsight.phases.phase2 import (
    FeatureRowInfluencePolicy,
    FrontierBufferPolicy,
    Phase2Config,
    Phase2ExecutionPolicy,
    RowStoreLayout,
    RowStoreRuntime,
    TargetSelectionPolicy,
    run_phase2,
)
from circuit_tracer.attribution.nnsight.phases.phase5 import (
    BatchExecutionSummary,
    GraphAssemblyLimits,
    GraphAssemblyState,
    Phase4PolicySummary,
    Phase5Config,
    Phase5Inputs,
    run_phase5,
)
from circuit_tracer.observability import events as observability_events
from circuit_tracer.observability.lifecycle import TelemetryObserver
from circuit_tracer.replacement_model.model_adapter import resolve_model_adapter
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder.attribution_result import AttributionComponents
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


def _sparse_activations() -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.tensor([[0], [1], [2]]),
        torch.tensor([1.0]),
        size=(1, 2, 4),
    ).coalesce()


def test_attribution_components_enforce_cross_tensor_invariants() -> None:
    features = _sparse_activations()
    result = AttributionComponents(
        activation_matrix=features,
        reconstruction=torch.zeros(1, 2, 3),
        encoder_vectors=torch.zeros(1, 3),
        decoder_vectors=torch.zeros(1, 3),
        encoder_to_decoder_map=torch.zeros(1, dtype=torch.long),
        decoder_locations=torch.zeros(2, 1, dtype=torch.long),
    )
    assert result.active_feature_count == 1
    assert result.encoder_vectors.shape == (1, 3)

    with pytest.raises(ValueError, match="decoder locations"):
        AttributionComponents(
            activation_matrix=features,
            reconstruction=torch.zeros(1, 2, 3),
            encoder_vectors=torch.zeros(1, 3),
            decoder_vectors=torch.zeros(1, 3),
            encoder_to_decoder_map=torch.zeros(1, dtype=torch.long),
            decoder_locations=torch.zeros(2, 0, dtype=torch.long),
        )


def test_context_execution_policy_owns_residency_fallback() -> None:
    policy = ContextExecutionPolicy.resolve(
        chunked_decoder_state=None,
        encoder_vectors=torch.zeros(1, 3),
        error_vectors=torch.zeros(1, 2, 3),
        exact_encoder_residency="active_cpu",
        stage_encoder_vectors_on_cpu=None,
        stage_error_vectors_on_cpu=None,
        error_vector_prefetch_lookahead=0,
        chunked_feature_replay_window=0,
        row_subchunk_size=0,
    )
    assert policy.encoder_residency_effective == "lazy"
    assert policy.encoder_residency_fallback_reason is not None
    assert policy.error_vector_prefetch_lookahead == 1
    assert policy.chunked_feature_replay_window == 1
    assert policy.row_subchunk_size == 1


def test_batch_request_enforces_lane_and_column_invariants() -> None:
    request = BatchAttributionRequest(
        layers=torch.tensor([0, 1]),
        positions=torch.tensor([0]),
        inject_values=torch.zeros(2, 3),
        retain_graph=True,
        phase_label="test",
        feature_column_range=None,
        include_nonfeature=True,
    )
    with pytest.raises(ValueError, match="equal batch size"):
        request.validate(batch_capacity=2, active_features=4)


def test_model_variation_is_resolved_through_adapter_capabilities() -> None:
    chat = resolve_model_adapter(architecture="Gemma3ForCausalLM", has_chat_template=True)
    generic = resolve_model_adapter(architecture="GptOssForCausalLM", has_chat_template=False)
    assert chat.ignored_token_positions == slice(0, 4)
    assert chat.validate_preserved_prefix(torch.tensor([2, 105, 2364, 107]))
    assert generic.ignored_token_positions == slice(0, 1)
    assert generic.normalize_feature_output(torch.zeros(2, 3)).shape == (1, 2, 3)


def test_hotspot_entrypoints_delegate_to_domain_owners() -> None:
    context_init = inspect.getsource(AttributionContext.__init__)
    compute_batch = inspect.getsource(AttributionContext.compute_batch)
    configure = inspect.getsource(NNSightReplacementModel._configure_replacement_model)
    setup = inspect.getsource(NNSightReplacementModel.setup_attribution)
    components = inspect.getsource(CrossLayerTranscoder.compute_attribution_components)

    assert "tensor_state" in context_init
    assert "execute_observed_batch(" in compute_batch
    assert "configure_nnsight_replacement_model(" in configure
    assert "Phase0ActivationCapture.run(" in setup
    assert "AttributionSetupOperation(" in setup
    assert "AttributionComponents(" in components


def test_backward_mode_branching_stays_inside_strategy_owners() -> None:
    package_root = Path(inspect.getfile(AttributionContext)).parents[1]
    allowed = {
        package_root / "tracing" / "plan.py",
        package_root / "attribution" / "nnsight" / "backward_engines.py",
    }
    mode_literals = {
        "duplicated_lanes",
        "single_forward_batched_vjp",
        "single_forward_serial_vjp",
    }
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            constants = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            if constants & mode_literals:
                violations.append(f"{path.relative_to(package_root)}:{node.lineno}")
    assert not violations, "backward-mode conditionals escaped strategy owners: " + ", ".join(
        violations
    )


def test_replacement_model_hotspot_has_no_family_name_branch() -> None:
    tree = ast.parse(inspect.getsource(NNSightReplacementModel))
    string_literals = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any("gemma-3" in value or "gpt-oss" in value for value in string_literals)


def test_no_legacy_result_or_flat_context_constructor_surface() -> None:
    assert "__getitem__" not in AttributionComponents.__dict__
    assert "coerce" not in AttributionComponents.__dict__
    assert "_LEGACY_NAMES" not in AttributionComponents.__dict__
    assert list(inspect.signature(AttributionContext).parameters) == [
        "tensor_state",
        "execution_policy",
        "decoder_runtime",
        "numeric_policy",
    ]


def test_adapter_dispatch_does_not_accept_model_names() -> None:
    source = inspect.getsource(resolve_model_adapter)
    assert "model_name" not in source
    assert "endswith" not in source
    assert "has_chat_template" in inspect.signature(resolve_model_adapter).parameters


def test_phase2_and_phase5_use_bounded_domain_objects() -> None:
    bounded_types = (
        Phase2Config,
        TargetSelectionPolicy,
        FrontierBufferPolicy,
        FeatureRowInfluencePolicy,
        RowStoreLayout,
        RowStoreRuntime,
        Phase2ExecutionPolicy,
        Phase5Inputs,
        Phase5Config,
        GraphAssemblyState,
        GraphAssemblyLimits,
        BatchExecutionSummary,
        Phase4PolicySummary,
    )
    assert all(len(fields(domain_type)) <= 10 for domain_type in bounded_types)
    assert {field.name for field in fields(Phase2Config)} == {
        "targets",
        "phase0_replay",
        "phase3_replay",
        "frontier",
        "storage_layout",
        "storage_runtime",
        "execution",
    }
    assert {field.name for field in fields(Phase5Inputs)} == {
        "runtime",
        "graph",
        "replay",
        "diagnostics",
        "output",
    }


def test_phase2_and_phase5_orchestrators_are_small_and_explicit() -> None:
    expected_calls = {
        run_phase2: [
            "select_attribution_targets",
            "apply_phase0_replay",
            "record_target_replay_evidence",
            "plan_active_feature_storage",
            "open_row_storage",
            "record_storage_evidence",
            "load_phase3_replay",
        ],
        run_phase5: [
            "select_graph_features",
            "assemble_compact_graph",
            "package_compact_artifacts",
            "finalize_compact_publication",
            "assemble_full_graph",
            "finalize_full_publication",
        ],
    }
    for orchestrator, calls in expected_calls.items():
        source = inspect.getsource(orchestrator)
        assert len(source.splitlines()) <= 150
        positions = [source.index(f"{name}(") for name in calls]
        assert positions == sorted(positions)
        tree = ast.parse(source)
        complexity = 1 + sum(
            isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.BoolOp))
            for node in ast.walk(tree)
        )
        assert complexity <= 4


def test_phase2_and_phase5_operations_remain_bounded_and_owned() -> None:
    phases = Path(inspect.getfile(run_phase2)).parent
    operation_files = [
        *phases.glob("phase2_*.py"),
        *phases.glob("phase5_*.py"),
    ]
    for path in operation_files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 200, (
                    f"{path.name}:{node.name} exceeds the operation size limit"
                )

    phase2_source = Path(inspect.getfile(run_phase2)).read_text()
    phase5_source = Path(inspect.getfile(run_phase5)).read_text()
    assert "nnsight.replay import" not in phase2_source
    assert "nnsight.row_store import" not in phase2_source
    assert "from circuit_tracer.graph import Graph" not in phase5_source
    assert "nnsight.prefix_view import" not in phase5_source


def test_observability_has_no_service_locator_observations() -> None:
    event_source = inspect.getsource(observability_events)
    observer_source = inspect.getsource(TelemetryObserver.observe)
    assert "ConstructWithObserver" not in event_source
    assert "BindTraceObserver" not in event_source
    assert "factory(" not in observer_source
    assert "configure_trace_logging" not in observer_source
