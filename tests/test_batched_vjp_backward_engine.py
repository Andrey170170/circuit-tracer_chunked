from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from nnsight import NNsight

from circuit_tracer.attribution.nnsight.backward_engines import (
    BackwardEngineExecutionError,
    SingleForwardBatchedVjpEngine,
    SingleForwardSerialVjpEngine,
    resolve_backward_batch_engine,
)
from circuit_tracer.attribution.nnsight.batch_contract import (
    BatchAttributionRequest,
    BatchExecutionResult,
)
from circuit_tracer.attribution.nnsight.batch_execution import execute_observed_batch
from circuit_tracer.tracing import BackwardPlan
from circuit_tracer.tracing.plan import (
    BackwardEngineMode,
    BackwardExecutionTopology,
    ForwardGraphMode,
    VjpKernelMode,
)


class _Host:
    def __init__(self) -> None:
        torch.manual_seed(3)
        self.n_layers = 2
        self._input = torch.randn(1, 3, 2, requires_grad=True)
        resid0 = self._input * 1.25
        feature0 = torch.tanh(resid0 @ torch.tensor([[1.2, -0.4], [0.3, 0.8]]))
        resid1 = resid0 + feature0
        feature1 = torch.sin(resid1 @ torch.tensor([[0.7, 0.2], [-0.5, 1.1]]))
        resid2 = resid1 + feature1
        self._resid_activations = [resid0, resid1, resid2]
        self._feature_output_activations = [self._input, feature0, feature1]
        for tensor in (*self._resid_activations, *self._feature_output_activations):
            tensor.retain_grad()
        indices = torch.tensor([[0, 1], [0, 2], [1, 0]])
        values = torch.tensor([1.0, 1.0])
        self.activation_matrix = torch.sparse_coo_tensor(indices, values, (2, 3, 2))
        self.chunked_decoder_state = None
        self.diagnostic_mode = False
        self.capture_phase3_gradients = False
        self.phase3_gradient_replay_tensor = None
        self.phase3_gradient_replay_column_offset = 0
        self.phase3_gradient_captures = []
        self._batch_buffer = None
        self._row_size = 5
        self._chunked_feature_replay_window = 1
        self._produced_feature_range = None
        self._produce_nonfeature = True
        self._diagnostic_stats = {}
        self._trace_observer = None
        self._resource_sample_count_by_phase = {}
        self.events: list[dict[str, object]] = []

    def _clear_saved_grads(self) -> None:
        for tensor in (*self._resid_activations, *self._feature_output_activations):
            tensor.grad = None

    def _materialize_tensor(self, tensor, *, device=None, dtype=None):
        return tensor.to(device=device, dtype=dtype)

    def _flush_chunked_feature_grad_window(self, gradients, layers, **kwargs) -> None:
        raise AssertionError("non-chunked test host must not flush")

    def compute_feature_attributions(self, layer, grad, **kwargs) -> None:
        assert self._batch_buffer is not None
        self._batch_buffer[layer] = grad.sum(dim=(1, 2))

    def compute_error_attributions(self, layer, grads) -> None:
        assert self._batch_buffer is not None
        self._batch_buffer[2 + layer] = (grads.square()).sum(dim=(1, 2))

    def compute_token_attributions(self, grads) -> None:
        assert self._batch_buffer is not None
        self._batch_buffer[4] = grads[:, 0].sum(dim=1)

    def _add_layer_stat(self, key: str, layer: int, value: float) -> None:
        raise AssertionError("diagnostics disabled")

    def _add_stat(self, key: str, value: float) -> None:
        raise AssertionError("diagnostics disabled")

    def _emit_trace(self, event: str, **fields: object) -> None:
        self.events.append({"event": event, **fields})

    def _record_telemetry_event(self, **kwargs) -> None:
        self.events.append(dict(kwargs))


def _request() -> BatchAttributionRequest:
    return BatchAttributionRequest(
        layers=torch.tensor([2, 1, 2]),
        positions=torch.tensor([0, 2, 1]),
        inject_values=torch.tensor([[0.5, -0.2], [0.1, 0.9], [-0.3, 0.4]]),
        retain_graph=True,
        phase_label="test",
        feature_column_range=None,
        include_nonfeature=True,
    )


def _serial_reference(host: _Host, request: BatchAttributionRequest) -> torch.Tensor:
    rows = []
    for source_layer, position, inject in zip(
        request.layers.tolist(),
        request.positions.tolist(),
        request.inject_values,
        strict=True,
    ):
        output = host._resid_activations[source_layer]
        cotangent = torch.zeros_like(output)
        cotangent[0, position] = inject
        inputs = tuple(host._feature_output_activations[: source_layer + 1])
        gradients = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        row = torch.zeros(5)
        for layer in range(source_layer):
            grad = gradients[layer + 1]
            assert grad is not None
            row[layer] = grad.sum()
            row[2 + layer] = grad.square().sum()
        token = gradients[0]
        assert token is not None
        row[4] = token[0, 0].sum()
        rows.append(row)
    return torch.stack(rows)


def test_single_forward_batched_vjp_matches_serial_mixed_layer_reference() -> None:
    host = _Host()
    request = _request()
    expected = _serial_reference(host, request)

    result = SingleForwardBatchedVjpEngine(batch_capacity=3).execute(
        host,
        request,
        batch_call_index=1,
    )

    torch.testing.assert_close(result.rows, expected)
    assert result.engine_attrs["backward_engine_mode"] == "single_forward_batched_vjp"
    assert result.engine_attrs["forward_lane_count"] == 1
    assert result.engine_attrs["source_layer_group_count"] == 2
    assert result.engine_attrs["autograd_call_count"] == 2
    assert result.engine_attrs["vjp_requested_path"] == "autograd_batched"
    assert result.engine_attrs["vjp_effective_invocation"] == "torch.autograd.grad"
    assert result.engine_attrs["vjp_is_grads_batched"] is True
    assert result.engine_attrs["vjp_fallback_state"] == "unknown"
    assert (
        result.engine_attrs["vjp_fallback_observation_method"] == "direct_call_contract_and_success"
    )
    evidence = cast(dict[str, object], result.engine_attrs["vjp_execution_evidence"])
    assert evidence["successful_invocation_count"] == 2
    assert evidence["source_layer_group_count"] == 2
    assert evidence["fallback_state_reason"] == (
        "pytorch_has_no_programmatic_per_invocation_vmap_fallback_signal"
    )
    assert "vmap_fallback_observation" not in result.engine_attrs


def test_single_forward_serial_and_batched_vjp_match_on_same_graph() -> None:
    host = _Host()
    request = _request()

    batched = SingleForwardBatchedVjpEngine(batch_capacity=3).execute(
        host,
        request,
        batch_call_index=1,
    )
    serial = SingleForwardSerialVjpEngine(batch_capacity=3).execute(
        host,
        request,
        batch_call_index=2,
    )

    torch.testing.assert_close(serial.rows, batched.rows)
    assert serial.engine_attrs["backward_engine_mode"] == "single_forward_serial_vjp"
    assert serial.engine_attrs["forward_graph_mode"] == "single_lane"
    assert serial.engine_attrs["vjp_kernel_mode"] == "autograd_serial"
    assert serial.engine_attrs["forward_lane_count"] == 1
    assert serial.engine_attrs["source_layer_group_count"] == 2
    assert serial.engine_attrs["autograd_call_count"] == 3
    assert serial.engine_attrs["vjp_is_grads_batched"] is False
    assert serial.engine_attrs["vjp_fallback_state"] == "not_applicable"
    assert batched.engine_attrs["vjp_kernel_mode"] == "autograd_batched"
    assert batched.engine_attrs["autograd_call_count"] == 2


@pytest.mark.parametrize(
    "engine_type",
    [SingleForwardBatchedVjpEngine, SingleForwardSerialVjpEngine],
)
def test_single_forward_engines_reject_missing_required_gradient(
    engine_type: type[SingleForwardBatchedVjpEngine],
) -> None:
    host = _Host()
    disconnected = torch.zeros_like(
        host._feature_output_activations[2],
        requires_grad=True,
    )
    disconnected.retain_grad()
    host._feature_output_activations[2] = disconnected

    with pytest.raises(BackwardEngineExecutionError) as raised:
        engine_type(batch_capacity=3).execute(
            host,
            _request(),
            batch_call_index=1,
        )

    assert raised.value.stage == "contraction"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "required attribution gradient" in str(raised.value.__cause__)


def test_realized_nnsight_trace_tensor_supports_direct_batched_vjp() -> None:
    """Prove the new primitive works after NNSight has realized cached handles."""

    torch.manual_seed(17)
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2, bias=False),
    )
    model = NNsight(module)
    inputs = torch.randn(1, 3, requires_grad=True)
    cached: list[torch.Tensor] = []

    with model.trace(inputs):
        cached.append(model._children[0].output)

    hidden = cached[0]
    cotangents = torch.randn(3, *hidden.shape)
    (batched,) = torch.autograd.grad(
        hidden,
        (inputs,),
        grad_outputs=(cotangents,),
        retain_graph=True,
        is_grads_batched=True,
    )
    serial = torch.stack(
        [
            torch.autograd.grad(
                hidden,
                inputs,
                grad_outputs=cotangent,
                retain_graph=True,
            )[0]
            for cotangent in cotangents
        ]
    )

    assert isinstance(hidden, torch.Tensor)
    assert hidden.grad_fn is not None
    torch.testing.assert_close(batched, serial)


def test_single_forward_batched_vjp_merges_deferred_gradients_in_original_row_order() -> None:
    host = _Host()
    host.chunked_decoder_state = {}
    request = _request()

    expected_by_layer: list[list[torch.Tensor]] = [[], []]
    for source_layer, position, inject in zip(
        request.layers.tolist(),
        request.positions.tolist(),
        request.inject_values,
        strict=True,
    ):
        output = host._resid_activations[source_layer]
        cotangent = torch.zeros_like(output)
        cotangent[0, position] = inject
        inputs = tuple(host._feature_output_activations[: source_layer + 1])
        gradients = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        for layer in range(host.n_layers):
            if layer < source_layer:
                gradient = gradients[layer + 1]
                assert gradient is not None
                expected_by_layer[layer].append(gradient[0])
            else:
                expected_by_layer[layer].append(
                    torch.zeros_like(host._feature_output_activations[layer + 1][0])
                )

    result = SingleForwardBatchedVjpEngine(batch_capacity=3).execute(
        host,
        request,
        batch_call_index=2,
        defer_feature_vjps=True,
    )

    entry = result.feature_vjp_tape_entry
    assert entry is not None
    assert entry.batch_size == 3
    assert entry.row_buffer.shape == (host._row_size, 3)
    for actual, expected in zip(entry.gradients, expected_by_layer, strict=True):
        assert actual is not None
        torch.testing.assert_close(actual, torch.stack(expected))
    assert entry.host_nbytes == sum(
        gradient.numel() * gradient.element_size()
        for gradient in entry.gradients
        if gradient is not None
    )


def test_backward_plan_and_factory_keep_legacy_default_explicit() -> None:
    assert BackwardPlan().mode == "duplicated_lanes"
    legacy_topology = BackwardPlan().topology(batch_capacity=4)
    assert legacy_topology.logical_batch_capacity == 4
    assert legacy_topology.forward_graph_mode == "logical_capacity"
    assert legacy_topology.vjp_kernel_mode == "nnsight_injected"
    assert legacy_topology.forward_lane_count == 4
    assert (
        BackwardPlan(mode="single_forward_batched_vjp")
        .topology(batch_capacity=4)
        .forward_lane_count
        == 1
    )
    serial_plan = BackwardPlan(mode="single_forward_serial_vjp")
    assert serial_plan.forward_graph_mode == "single_lane"
    assert serial_plan.vjp_kernel_mode == "autograd_serial"
    assert serial_plan.topology(batch_capacity=4).forward_lane_count == 1
    explicit_serial = BackwardPlan(
        forward_graph_mode="single_lane",
        vjp_kernel_mode="autograd_serial",
    )
    assert explicit_serial == serial_plan
    assert (
        BackwardPlan().planner_batch_capacity(
            source_rows=16,
            feature_rows=32,
            feature_row_ceiling=128,
            logit_rows=8,
        )
        == 32
    )
    assert (
        BackwardPlan(mode="single_forward_batched_vjp").planner_batch_capacity(
            source_rows=16,
            feature_rows=32,
            feature_row_ceiling=128,
            logit_rows=8,
        )
        == 128
    )
    assert (
        resolve_backward_batch_engine(mode="duplicated_lanes", batch_capacity=4).forward_lane_count
        == 4
    )
    assert (
        resolve_backward_batch_engine(
            mode="single_forward_batched_vjp", batch_capacity=4
        ).forward_lane_count
        == 1
    )
    resolved_serial = resolve_backward_batch_engine(
        mode="single_forward_serial_vjp",
        batch_capacity=4,
    )
    assert isinstance(resolved_serial, SingleForwardSerialVjpEngine)
    assert resolved_serial.forward_graph_mode == "single_lane"
    assert resolved_serial.vjp_kernel_mode == "autograd_serial"
    assert resolved_serial.forward_lane_count == 1
    with pytest.raises(ValueError, match="backward engine mode"):
        BackwardPlan(mode=cast(Any, "fallback"))


@pytest.mark.parametrize(
    ("forward_graph_mode", "vjp_kernel_mode"),
    [
        ("logical_capacity", "autograd_batched"),
        ("logical_capacity", "autograd_serial"),
        ("single_lane", "nnsight_injected"),
    ],
)
def test_backward_plan_rejects_unsupported_component_combinations(
    forward_graph_mode: ForwardGraphMode,
    vjp_kernel_mode: VjpKernelMode,
) -> None:
    with pytest.raises(ValueError, match="unsupported backward execution combination"):
        BackwardPlan(
            forward_graph_mode=forward_graph_mode,
            vjp_kernel_mode=vjp_kernel_mode,
        )


def test_backward_plan_rejects_partial_or_conflicting_selection() -> None:
    with pytest.raises(ValueError, match="requires both"):
        BackwardPlan(forward_graph_mode="single_lane")
    with pytest.raises(ValueError, match="preset mode or explicit components"):
        BackwardPlan(
            mode="single_forward_serial_vjp",
            forward_graph_mode="single_lane",
            vjp_kernel_mode="autograd_serial",
        )
    with pytest.raises(ValueError, match="conflicts"):
        BackwardExecutionTopology.resolve_components(
            mode="single_forward_batched_vjp",
            forward_graph_mode="single_lane",
            vjp_kernel_mode="autograd_serial",
            batch_capacity=4,
        )


class _UnprintableFailure(RuntimeError):
    def __str__(self) -> str:
        raise KeyError("formatting should not mask the primary error")


@dataclass
class _FailingEngine:
    mode: BackwardEngineMode = "single_forward_batched_vjp"
    batch_capacity: int = 3
    forward_lane_count: int = 1
    forward_graph_mode: ForwardGraphMode = "single_lane"
    vjp_kernel_mode: VjpKernelMode = "autograd_batched"

    def execute(self, *args, **kwargs) -> BatchExecutionResult:
        raise _UnprintableFailure()


def test_batch_failure_telemetry_cannot_mask_unprintable_primary_error() -> None:
    host = _Host()
    with pytest.raises(_UnprintableFailure):
        execute_observed_batch(
            host,
            _request(),
            strategy=_FailingEngine(),
            batch_call_index=7,
        )

    failed = next(
        event for event in host.events if event.get("name") == "context.compute_batch.failed"
    )
    attrs = cast(dict[str, object], failed["attrs"])
    assert isinstance(attrs, dict)
    assert str(attrs["error_type"]).endswith("_UnprintableFailure")
    assert str(attrs["error_message"]).startswith("<unavailable: str raised")
    assert attrs["backward_engine_mode"] == "single_forward_batched_vjp"
