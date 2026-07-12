import pytest
import torch

from circuit_tracer.attribution.nnsight.row_replay import ReplayGraphLifecycle, RowRecipe, RowRecipeLedger
from circuit_tracer.transcoder.provider import TranscoderCapabilities, require_exact_row_replay_provider


def _ledger(rows: torch.Tensor):
    events: list[str] = []
    lifecycle = ReplayGraphLifecycle(
        reset=lambda: events.append("reset"),
        rebuild_forward=lambda: events.append("forward"),
        release=lambda: events.append("release"),
    )
    ledger = RowRecipeLedger(
        n_rows=rows.shape[0], n_feature_columns=rows.shape[1], dtype=rows.dtype,
        producer=lambda recipes, start, end: rows[
            [recipe.ordinal for recipe in recipes], start:end
        ],
        lifecycle=lifecycle, semantic_fingerprint={"s": 1}, execution_fingerprint={"e": 1},
        provider_fingerprint={"architecture": "clt"},
    )
    absolute = rows.abs()
    maxima = absolute.amax(dim=1)
    scaled = torch.where(maxima > 0, (absolute / maxima[:, None]).sum(dim=1), 0)
    for ordinal in range(rows.shape[0]):
        ledger.append_recipe(
            RowRecipe(ordinal, "logit" if ordinal == 0 else "feature", ordinal, ordinal,
                      stable_reference=("row", ordinal)),
            node_index=99 if ordinal == 0 else ordinal - 1,
            denominator=(maxima[ordinal : ordinal + 1], scaled[ordinal : ordinal + 1]),
        )
    return ledger, events


def test_replay_tiles_and_selected_projection_are_bitwise_equal() -> None:
    rows = torch.tensor([[0.2, 0.1, 0.0], [0.0, 0.4, 0.3]], dtype=torch.float64)
    ledger, events = _ledger(rows)
    assert torch.equal(ledger.read_tile(0, 2, 1, 3), rows[:, 1:3])
    selected = torch.tensor([2, 0, 2])
    assert torch.equal(
        ledger.materialize_dense_feature_slice(row_start=0, row_end=2, selected_feature_columns=selected),
        rows[:, selected],
    )
    assert events.count("reset") == events.count("forward") == events.count("release")
    snapshot = ledger.get_diagnostic_snapshot()
    assert snapshot["no_retained_full_feature_matrix"] is snapshot["no_kxn_file"] is True
    assert snapshot["request_bounds_enforced"] is False
    assert snapshot["apparent_file_bytes"] == snapshot["allocated_file_bytes"] == 0


def test_huge_ledger_has_no_kxn_allocation_or_file() -> None:
    lifecycle = ReplayGraphLifecycle(reset=lambda: None, rebuild_forward=lambda: None, release=lambda: None)
    ledger = RowRecipeLedger(
        n_rows=1_000_000, n_feature_columns=10_000_000, dtype=torch.float32,
        producer=lambda *_: pytest.fail("must not replay during construction"), lifecycle=lifecycle,
        semantic_fingerprint={}, execution_fingerprint={}, provider_fingerprint={},
    )
    assert ledger.path is None
    assert ledger.row_abs_max.shape == (1_000_000,)
    assert ledger.get_diagnostic_snapshot()["retained_recipe_bytes"] == 0


def test_release_runs_after_replay_failure_and_cleanup_is_idempotent() -> None:
    events: list[str] = []
    lifecycle = ReplayGraphLifecycle(
        reset=lambda: events.append("reset"), rebuild_forward=lambda: events.append("forward"),
        release=lambda: events.append("release"),
    )
    ledger = RowRecipeLedger(
        n_rows=1, n_feature_columns=1, dtype=torch.float32,
        producer=lambda *_: (_ for _ in ()).throw(RuntimeError("cancelled")), lifecycle=lifecycle,
        semantic_fingerprint={}, execution_fingerprint={}, provider_fingerprint={},
    )
    ledger.append_recipe(RowRecipe(0, "logit", 0, 0, stable_reference=("logit", 0)),
                         node_index=1, denominator=(torch.ones(1), torch.ones(1)))
    with pytest.raises(RuntimeError, match="cancelled"):
        ledger.read_tile(0, 1, 0, 1)
    assert events == ["reset", "forward", "release"]
    ledger.cleanup()
    ledger.cleanup()


def test_each_uncached_replay_request_rebuilds_then_releases_its_graph() -> None:
    rows = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    ledger, events = _ledger(rows)

    torch.testing.assert_close(ledger.read_tile(0, 1, 0, 1), rows[:, 0:1])
    torch.testing.assert_close(ledger.read_tile(0, 1, 1, 3), rows[:, 1:3])

    assert events == ["reset", "forward", "release", "reset", "forward", "release"]
    snapshot = ledger.get_diagnostic_snapshot()
    assert snapshot["graph_rebuild_count"] == 2
    assert snapshot["graph_forward_count"] == 2
    assert snapshot["graph_backward_count"] == 2
    assert snapshot["no_kxn_file"] is True
    assert snapshot["apparent_file_bytes"] == snapshot["allocated_file_bytes"] == 0


def test_replay_tile_batches_recipes_within_one_graph_lifecycle() -> None:
    rows = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    events: list[str] = []
    batch_sizes: list[int] = []
    lifecycle = ReplayGraphLifecycle(
        reset=lambda: events.append("reset"),
        rebuild_forward=lambda: events.append("forward"),
        release=lambda: events.append("release"),
    )

    def produce(recipes, start: int, end: int) -> torch.Tensor:
        batch_sizes.append(len(recipes))
        return rows[[recipe.ordinal for recipe in recipes], start:end]

    ledger = RowRecipeLedger(
        n_rows=5,
        n_feature_columns=4,
        dtype=torch.float32,
        producer=produce,
        lifecycle=lifecycle,
        semantic_fingerprint={},
        execution_fingerprint={},
        provider_fingerprint={},
        replay_batch_rows=2,
    )
    for ordinal in range(5):
        ledger.append_recipe(
            RowRecipe(ordinal, "feature", 0, 0, stable_reference=("row", ordinal)),
            node_index=ordinal,
            denominator=(torch.ones(1), torch.ones(1)),
        )

    assert torch.equal(ledger.read_tile(0, 5, 1, 4), rows[:, 1:4])
    assert batch_sizes == [2, 2, 1]
    assert events == ["reset", "forward", "release"]
    snapshot = ledger.get_diagnostic_snapshot()
    assert snapshot["replay_batch_count"] == 3
    assert snapshot["max_replay_batch_rows"] == 2
    assert snapshot["graph_backward_count"] == 3


def test_rebuild_failure_releases_and_resets_lifecycle() -> None:
    events: list[str] = []
    lifecycle = ReplayGraphLifecycle(
        reset=lambda: events.append("reset"),
        rebuild_forward=lambda: (_ for _ in ()).throw(RuntimeError("rebuild failed")),
        release=lambda: events.append("release"),
    )
    with pytest.raises(RuntimeError, match="rebuild failed"):
        lifecycle.begin_request()
    assert events == ["reset", "release"]
    lifecycle.cleanup()
    assert events == ["reset", "release"]


def test_rebuild_failure_unwraps_nnsight_style_original_exception() -> None:
    original = RuntimeError("actionable replay failure")
    wrapped = RuntimeError("broken formatter")
    wrapped.original = original  # type: ignore[attr-defined]
    lifecycle = ReplayGraphLifecycle(
        reset=lambda: None,
        rebuild_forward=lambda: (_ for _ in ()).throw(wrapped),
        release=lambda: None,
    )

    with pytest.raises(RuntimeError, match="actionable replay failure") as raised:
        lifecycle.begin_request()

    assert raised.value is original


def test_row_recipe_clones_cpu_injection_immutably() -> None:
    source = torch.tensor([1.0, 2.0])
    recipe = RowRecipe(0, "feature", 0, 0, injection=source)
    source.add_(10)
    assert torch.equal(recipe.injection, torch.tensor([1.0, 2.0]))


def test_replay_reader_enforces_configured_request_bounds() -> None:
    rows = torch.zeros((2, 4))
    lifecycle = ReplayGraphLifecycle(reset=lambda: None, rebuild_forward=lambda: None, release=lambda: None)
    ledger = RowRecipeLedger(
        n_rows=2, n_feature_columns=4, dtype=torch.float32,
        producer=lambda recipes, start, end: rows[
            [recipe.ordinal for recipe in recipes], start:end
        ],
        lifecycle=lifecycle, semantic_fingerprint={}, execution_fingerprint={},
        provider_fingerprint={}, max_request_rows=1, max_request_columns=2,
    )
    for ordinal in range(2):
        ledger.append_recipe(RowRecipe(ordinal, "feature", 0, 0, injection=torch.ones(1)),
                             node_index=ordinal, denominator=(torch.ones(1), torch.ones(1)))
    assert ledger.get_diagnostic_snapshot()["request_bounds_enforced"] is True
    with pytest.raises(ValueError, match="row bound"):
        ledger.read_tile(0, 2, 0, 1)
    with pytest.raises(ValueError, match="column bound"):
        ledger.read_tile(0, 1, 0, 3)
    selected = torch.tensor([3, 1, 3])
    actual = ledger.materialize_dense_feature_slice(
        row_start=0, row_end=2, selected_feature_columns=selected
    )
    torch.testing.assert_close(actual, rows[:, selected])


@pytest.mark.parametrize("architecture", ["clt", "plt"])
def test_provider_capability_gate_rejects_fallback(architecture: str) -> None:
    class Provider:
        def compute_attribution_components(self, *args):
            return None

        def get_decoder_chunk(self, *args):
            return None

        def decoder_output_layers_for_source(self, *args):
            return []

        def decoder_output_slot(self, *args):
            return 0

        def materialize_encoder_rows(self, *args):
            return None

    Provider.capabilities = TranscoderCapabilities(
        architecture=architecture, checkpoint_format="fake", supports_exact_chunked_provider=True,
        supports_exact_row_replay=True,
    )
    require_exact_row_replay_provider(Provider())
    Provider.capabilities = TranscoderCapabilities(
        architecture=architecture, checkpoint_format="fake", supports_exact_chunked_provider=True,
        supports_exact_row_replay=False,
    )
    with pytest.raises(ValueError, match="supports_exact_row_replay"):
        require_exact_row_replay_provider(Provider())
