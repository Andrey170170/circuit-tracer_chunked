"""Typed contract shared by backward engines and batch observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from circuit_tracer.attribution.nnsight.feature_vjp_tape import FeatureVjpTapeEntry
from circuit_tracer.observability.events import TraceObserver


@dataclass(frozen=True)
class BatchAttributionRequest:
    layers: torch.Tensor
    positions: torch.Tensor
    inject_values: torch.Tensor
    retain_graph: bool
    phase_label: str
    feature_column_range: tuple[int, int] | None
    include_nonfeature: bool

    def validate(self, *, batch_capacity: int, active_features: int) -> int:
        batch_size = int(self.layers.numel())
        if self.layers.ndim != 1 or self.positions.ndim != 1:
            raise ValueError("layers and positions must be rank-1 tensors")
        if self.inject_values.ndim < 1:
            raise ValueError("inject_values must include a leading batch dimension")
        if (
            int(self.positions.numel()) != batch_size
            or int(self.inject_values.shape[0]) != batch_size
        ):
            raise ValueError("layers, positions, and injected values must have equal batch size")
        if batch_size <= 0 or batch_size > batch_capacity:
            raise ValueError(
                "compute_batch rows must be in [1, backward batch capacity] "
                f"(active={batch_size}, capacity={batch_capacity})"
            )
        if self.feature_column_range is not None:
            start, end = self.feature_column_range
            if start < 0 or end < start or end > active_features:
                raise ValueError("feature_column_range must lie within active feature columns")
        return batch_size


@dataclass(frozen=True)
class BatchExecutionResult:
    rows: torch.Tensor
    inject_values_nbytes: int
    batch_buffer_nbytes: int
    layers_in_batch: tuple[int, ...]
    chunked_feature_grad_window_peak: int
    feature_vjp_tape_entry: FeatureVjpTapeEntry | None = None
    engine_attrs: dict[str, object] = field(default_factory=dict)


class BatchExecutionHost(Protocol):
    n_layers: int
    activation_matrix: torch.Tensor
    chunked_decoder_state: dict[str, torch.Tensor] | None
    diagnostic_mode: bool
    capture_phase3_gradients: bool
    phase3_gradient_replay_tensor: torch.Tensor | None
    phase3_gradient_replay_column_offset: int
    phase3_gradient_captures: list[dict[str, torch.Tensor | int]]
    _resid_activations: list[torch.Tensor]
    _feature_output_activations: list[torch.Tensor]
    _batch_buffer: torch.Tensor | None
    _row_size: int
    _chunked_feature_replay_window: int
    _produced_feature_range: tuple[int, int] | None
    _produce_nonfeature: bool
    _diagnostic_stats: dict[str, object]
    _trace_observer: TraceObserver | None
    _resource_sample_count_by_phase: dict[str, int]

    def _clear_saved_grads(self) -> None: ...
    def _materialize_tensor(self, tensor, *, device=None, dtype=None): ...
    def _flush_chunked_feature_grad_window(self, gradients, layers, **kwargs) -> None: ...
    def compute_feature_attributions(self, layer, grad, **kwargs) -> None: ...
    def compute_error_attributions(self, layer, grads) -> None: ...
    def compute_token_attributions(self, grads) -> None: ...
    def _add_layer_stat(self, key: str, layer: int, value: float) -> None: ...
    def _add_stat(self, key: str, value: float) -> None: ...
    def _emit_trace(self, event: str, **fields: object) -> None: ...
    def _record_telemetry_event(self, **kwargs) -> None: ...
