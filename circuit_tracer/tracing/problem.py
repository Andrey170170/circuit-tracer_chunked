"""Scientific inputs and semantic choices for canonical circuit tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypeAlias

import torch

from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.attribution.targets import TargetSpec


Prompt = str | torch.Tensor | list[int]
AttributionTargets = Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None
PrefixViewMode = Literal["independent_prefix", "full_sequence_target_position"]


@dataclass(frozen=True)
class AllActiveSources:
    """Select every active feature as an eligible attribution source."""

    kind: Literal["all_active"] = field(default="all_active", init=False)
    version: Literal[1] = field(default=1, init=False)


@dataclass(frozen=True)
class TokenPositionSources:
    """Select active features at canonical causal token positions."""

    positions: tuple[int, ...]
    max_features_per_position: int | None = None
    kind: Literal["token_positions"] = field(default="token_positions", init=False)
    version: Literal[1] = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple):
            raise ValueError("source positions must be a tuple")
        if not self.positions:
            raise ValueError("source positions must contain at least one position")
        if any(isinstance(position, bool) or not isinstance(position, int) or position < 0 for position in self.positions):
            raise ValueError("source positions must contain nonnegative integers")
        if tuple(sorted(set(self.positions))) != self.positions:
            raise ValueError("source positions must be sorted and unique")
        cap = self.max_features_per_position
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
            raise ValueError("max_features_per_position must be a positive integer when provided")


SourceSelection: TypeAlias = AllActiveSources | TokenPositionSources


def compile_source_selection(
    selection: SourceSelection,
    activation_matrix: torch.Tensor,
    *,
    target_position: int,
) -> torch.Tensor:
    """Compile semantic source positions into canonical active-feature indices."""

    if (
        not activation_matrix.is_sparse
        or activation_matrix.ndim != 3
        or not activation_matrix.is_coalesced()
    ):
        raise ValueError("activation_matrix must be a coalesced sparse rank-3 tensor")
    if isinstance(target_position, bool) or not isinstance(target_position, int):
        raise ValueError("target_position must be a nonnegative integer")
    n_positions = int(activation_matrix.shape[1])
    if target_position < 0 or target_position >= n_positions:
        raise ValueError("target_position is out of range for the activation matrix")
    total_active = int(activation_matrix._nnz())
    if isinstance(selection, AllActiveSources):
        return torch.arange(total_active, dtype=torch.long)
    if not isinstance(selection, TokenPositionSources):
        raise ValueError("unsupported source selection value")
    if selection.positions[-1] >= n_positions:
        raise ValueError("source position is out of range for the activation matrix")
    if selection.positions[-1] > target_position:
        raise ValueError("future source position exceeds the causal target position")

    positions = activation_matrix.indices()[1].to(device="cpu")
    values = activation_matrix.values().detach().to(device="cpu")
    chosen: list[torch.Tensor] = []
    for position in selection.positions:
        global_indices = torch.nonzero(positions == position, as_tuple=False).flatten()
        cap = selection.max_features_per_position
        if cap is not None and global_indices.numel() > cap:
            # global_indices is ascending. Stable sorting therefore breaks
            # equal-|activation| ties by the canonical global feature index.
            order = torch.argsort(
                values[global_indices].abs(), descending=True, stable=True
            )
            global_indices = global_indices[order[:cap]]
        chosen.append(global_indices)
    if not chosen:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(chosen).sort().values.to(dtype=torch.long)


def _positive(name: str, value: int | float | None) -> None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be positive when provided")


def _nonnegative(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be nonnegative when provided")


@dataclass(frozen=True)
class PrefixViewTarget:
    """Semantic target selection for a causal prefix view."""

    mode: PrefixViewMode
    target_position: int

    def __post_init__(self) -> None:
        _positive("prefix view target_position", self.target_position)


@dataclass(frozen=True)
class AttributionProblem:
    """Model, input, target selection, and graph objective for one trace."""

    model: Any = field(repr=False, compare=False)
    prompt: Prompt
    targets: AttributionTargets = field(default=None, repr=False)
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    output_position: int | None = None
    prefix_view: PrefixViewTarget | None = None
    source_selection: SourceSelection = field(default_factory=AllActiveSources)

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("model is required")
        if isinstance(self.prompt, list) and not self.prompt:
            raise ValueError("prompt token ids cannot be empty")
        _positive("max_n_logits", self.max_n_logits)
        if not 0 < self.desired_logit_prob <= 1:
            raise ValueError("desired_logit_prob must be in (0, 1]")
        _nonnegative("output_position", self.output_position)
        if not isinstance(self.source_selection, (AllActiveSources, TokenPositionSources)):
            raise ValueError("source_selection must be a supported source selection value")
        if (
            self.prefix_view is not None
            and self.output_position is not None
            and self.output_position != self.prefix_view.target_position - 1
        ):
            raise ValueError("output_position must equal prefix view target_position - 1")


@dataclass(frozen=True)
class FrontierSemantics:
    """Strict frontier checkpoints and membership choices."""

    scheduler: Literal["locality", "planner_v1", "planner_v2", "legacy"] = "locality"
    refresh_policy: Literal["standard", "deferred_v1"] = "standard"
    refresh_interval_multiplier: int = 1
    ranker: Literal["argsort", "topk_v1"] = "argsort"
    phase3_buffer_relative_epsilon: float | None = None
    phase3_buffer_max_extra: int = 0
    phase4_buffer_relative_epsilon: float | None = None
    phase4_buffer_max_extra_per_refresh: int = 0
    phase4_buffer_max_extra_total: int = 0

    def __post_init__(self) -> None:
        _positive("refresh_interval_multiplier", self.refresh_interval_multiplier)
        for name in (
            "phase3_buffer_max_extra",
            "phase4_buffer_max_extra_per_refresh",
            "phase4_buffer_max_extra_total",
        ):
            _nonnegative(name, getattr(self, name))
        for name in ("phase3_buffer_relative_epsilon", "phase4_buffer_relative_epsilon"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when provided")


@dataclass(frozen=True)
class TraceSemantics:
    """Complete choices that are allowed to change the mathematical result."""

    source_batch_size: int = 512
    feature_batch_size: int | None = None
    logit_batch_size: int | None = None
    max_feature_nodes: int | None = None
    diagnostic_feature_cap: int | None = None
    update_interval: int = 4
    exact_trace_internal_dtype: Literal["fp32", "fp64"] = "fp32"
    phase0_activation_threshold_compare_mode: Literal[
        "baseline", "bf16", "fp32", "fp64"
    ] = "baseline"
    sparsification: SparsificationConfig | None = field(default=None, repr=False)
    frontier: FrontierSemantics = field(default_factory=FrontierSemantics)

    def __post_init__(self) -> None:
        for name in (
            "source_batch_size",
            "feature_batch_size",
            "logit_batch_size",
            "max_feature_nodes",
            "diagnostic_feature_cap",
            "update_interval",
        ):
            _positive(name, getattr(self, name))
