from dataclasses import dataclass
from typing import Any, cast

import torch

from circuit_tracer.attribution.attribute import attribute_phase0_stats


@dataclass
class FakeTranscoders:
    diagnostic_snapshot: dict[str, object] | None = None
    reset_called: bool = False

    def reset_diagnostic_stats(self) -> None:
        self.reset_called = True

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {} if self.diagnostic_snapshot is None else dict(self.diagnostic_snapshot)


class CleanupContext:
    def __init__(self, activation_matrix: torch.Tensor, setup_diagnostic_stats: dict[str, object]):
        self.activation_matrix = activation_matrix.coalesce()
        self.setup_diagnostic_stats = setup_diagnostic_stats
        self.cleanup_called = False

    def cleanup(self) -> None:
        self.cleanup_called = True


class CacheOnlyContext:
    def __init__(self, activation_matrix: torch.Tensor, setup_diagnostic_stats: dict[str, object]):
        self.activation_matrix = activation_matrix.coalesce()
        self.setup_diagnostic_stats = setup_diagnostic_stats
        self.clear_decoder_cache_called = False

    def clear_decoder_cache(self) -> None:
        self.clear_decoder_cache_called = True


class FakeModel:
    def __init__(self, backend: str, transcoders: FakeTranscoders, ctx) -> None:
        self.backend = backend
        self.transcoders = transcoders
        self._ctx = ctx
        self.setup_calls: list[tuple[object, dict[str, object]]] = []

    def ensure_tokenized(self, prompt: str) -> torch.Tensor:
        return torch.arange(len(prompt), dtype=torch.long)

    def setup_attribution(self, prompt, **kwargs):
        self.setup_calls.append((prompt, kwargs))
        return self._ctx


def _make_activation_matrix() -> torch.Tensor:
    return torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 2],
                [1, 2, 0, 3],
            ]
        ),
        values=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(2, 3, 5),
        check_invariants=True,
    ).coalesce()


def test_attribute_phase0_stats_returns_counts_timings_and_cleans_up() -> None:
    activation_matrix = _make_activation_matrix()
    ctx = CleanupContext(activation_matrix, {"token_count": 3})
    transcoders = FakeTranscoders(
        diagnostic_snapshot={"encode_sparse_seconds": 1.25, "reconstruction_seconds": 2.5}
    )
    model = FakeModel("nnsight", transcoders, ctx)

    stats = attribute_phase0_stats("abc", cast(Any, model))

    assert model.setup_calls == [
        (
            "abc",
            {"sparsification": None, "retain_full_logits": False},
        )
    ]
    assert transcoders.reset_called
    assert ctx.cleanup_called
    assert stats == {
        "token_count": 3,
        "prompt_token_count": 3,
        "total_active_features": 4,
        "active_features_by_layer": [2, 2],
        "active_features_by_token": [1, 2, 1],
        "phase0_encode_seconds": 1.25,
        "phase0_reconstruction_seconds": 2.5,
    }


def test_attribute_phase0_stats_falls_back_to_prompt_length_and_cache_cleanup() -> None:
    activation_matrix = _make_activation_matrix()
    ctx = CacheOnlyContext(activation_matrix, {})
    transcoders = FakeTranscoders(diagnostic_snapshot=None)
    model = FakeModel("transformerlens", transcoders, ctx)

    stats = attribute_phase0_stats([11, 22, 33], cast(Any, model))

    assert len(model.setup_calls) == 1
    setup_prompt, setup_kwargs = model.setup_calls[0]
    assert torch.equal(
        cast(torch.Tensor, setup_prompt), torch.tensor([11, 22, 33], dtype=torch.long)
    )
    assert setup_kwargs == {"sparsification": None}
    assert transcoders.reset_called
    assert ctx.clear_decoder_cache_called
    assert stats == {
        "token_count": 3,
        "prompt_token_count": 3,
        "total_active_features": 4,
        "active_features_by_layer": [2, 2],
        "active_features_by_token": [1, 2, 1],
        "phase0_encode_seconds": None,
        "phase0_reconstruction_seconds": None,
    }
