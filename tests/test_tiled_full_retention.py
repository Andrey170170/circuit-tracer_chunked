import os

import pytest
import torch

from circuit_tracer.attribution.nnsight.row_store import (
    _ColumnTiledFeatureRowStore,
    _FileBackedFeatureRowStore,
)
from circuit_tracer.attribution.nnsight.tiled_rows import produce_and_store_tiled_rows
from circuit_tracer.attribution.nnsight.replay import _compute_row_denominator_scaled_l1
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_feature_influences_tiled,
)


def _append(store, rows: torch.Tensor) -> None:
    abs_rows = rows.abs()
    row_abs_max = abs_rows.amax(dim=1)
    row_l1_scaled = torch.where(row_abs_max > 0, (abs_rows / row_abs_max[:, None]).sum(dim=1), 0)
    store.append_rows(
        row_start=0,
        feature_rows=rows,
        row_denominator_scaled_l1=(row_abs_max, row_l1_scaled),
    )


def test_column_tiled_store_and_solver_match_full_file_oracle() -> None:
    rows = torch.tensor(
        [[0.2, 0.1, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.4]],
        dtype=torch.float64,
    )
    full = _FileBackedFeatureRowStore(n_rows=3, n_feature_columns=4, dtype=torch.float64)
    tiled = _ColumnTiledFeatureRowStore(
        n_rows=3, n_feature_columns=4, column_tile_size=2, dtype=torch.float64
    )
    try:
        _append(full, rows)
        _append(tiled, rows)
        args = dict(
            row_abs_sums=full.row_denominator_scaled_l1,
            logit_p=torch.tensor([1.0], dtype=torch.float64),
            row_to_node_index=torch.tensor([9, 0, 2]),
            n_feature_nodes=4,
            n_logits=1,
            compute_dtype=torch.float64,
        )
        expected = compute_partial_feature_influences_streaming(
            full.read_feature_rows, **args, row_chunk_size=2
        )
        telemetry: dict[str, int | str] = {}
        actual = compute_partial_feature_influences_tiled(
            tiled.read_tile,
            **args,
            row_tile_size=2,
            column_tile_size=2,
            telemetry=telemetry,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
        assert telemetry["maximum_materialized_tile_bytes"] <= 2 * 2 * 8
    finally:
        full.cleanup()
        tiled.cleanup()


def test_column_tiled_selected_projection_preserves_order_and_duplicates() -> None:
    rows = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    store = _ColumnTiledFeatureRowStore(
        n_rows=3, n_feature_columns=6, column_tile_size=2, dtype=torch.float32
    )
    try:
        _append(store, rows)
        selected = torch.tensor([5, 1, 5, 0])
        actual = store.materialize_dense_feature_slice(
            row_start=1, row_end=3, selected_feature_columns=selected
        )
        torch.testing.assert_close(actual, rows[1:3][:, selected])
    finally:
        store.cleanup()


def test_huge_shape_construction_is_sparse_and_never_materializes_full_matrix(tmp_path) -> None:
    store = _ColumnTiledFeatureRowStore(
        n_rows=1_000_000,
        n_feature_columns=10_000,
        column_tile_size=128,
        dtype=torch.float32,
        temp_root=tmp_path,
    )
    try:
        snapshot = store.get_diagnostic_snapshot()
        assert snapshot["apparent_file_bytes"] == 1_000_000 * 10_000 * 4
        assert snapshot["allocated_file_bytes"] < snapshot["apparent_file_bytes"]
        assert snapshot["maximum_materialized_tile_bytes"] == 0
    finally:
        path = store._tmpdir.name
        store.cleanup()
    assert not os.path.exists(path)
    with pytest.raises(RuntimeError, match="cleaned up"):
        store.read_tile(0, 1, 0, 1)


def test_tiled_solver_enforces_requested_tile_bound() -> None:
    maximum_shape = (0, 0)

    def reader(rs: int, re: int, cs: int, ce: int) -> torch.Tensor:
        nonlocal maximum_shape
        maximum_shape = max(maximum_shape, (re - rs, ce - cs))
        return torch.zeros((re - rs, ce - cs), dtype=torch.float32)

    result = compute_partial_feature_influences_tiled(
        reader,
        (torch.ones(100), torch.ones(100)),
        torch.ones(1),
        torch.cat((torch.tensor([999]), torch.arange(99))),
        n_feature_nodes=1_000,
        n_logits=1,
        row_tile_size=7,
        column_tile_size=11,
    )
    assert result.shape == (1_000,)
    assert maximum_shape <= (7, 11)


def test_true_tiled_production_streams_tiles_and_matches_full_row_denominator() -> None:
    full_rows = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, 0.25, -0.5]], dtype=torch.float64)

    class AllocationSpyContext:
        def __init__(self) -> None:
            self.maximum_feature_width = 0

        def produce_row_tiles(
            self, *args, feature_column_tile_size, consume_feature_tile, **kwargs
        ):
            del args, kwargs
            for start in range(0, 5, feature_column_tile_size):
                end = min(start + feature_column_tile_size, 5)
                tile = full_rows[:, start:end].clone()
                self.maximum_feature_width = max(self.maximum_feature_width, tile.shape[1])
                consume_feature_tile(start, end, tile)
            return full_rows[:, 5:].clone()

    ctx = AllocationSpyContext()
    feature = _ColumnTiledFeatureRowStore(
        n_rows=1, n_feature_columns=5, column_tile_size=2, dtype=torch.float64
    )
    nonfeature = _ColumnTiledFeatureRowStore(
        n_rows=1, n_feature_columns=2, column_tile_size=2, dtype=torch.float64
    )
    try:
        produced_nonfeature, denominator = produce_and_store_tiled_rows(
            ctx=ctx,
            layers=torch.tensor([0]),
            positions=torch.tensor([0]),
            inject_values=torch.ones((1, 1)),
            row_start=0,
            feature_row_store=feature,
            nonfeature_row_store=nonfeature,
            feature_column_tile_size=2,
            dtype=torch.float64,
            phase_label="phase4_features",
        )
        expected_denominator = _compute_row_denominator_scaled_l1(full_rows, dtype=torch.float64)
        torch.testing.assert_close(feature.read_feature_rows(0, 1), full_rows[:, :5])
        torch.testing.assert_close(produced_nonfeature, full_rows[:, 5:])
        torch.testing.assert_close(nonfeature.read_feature_rows(0, 1), full_rows[:, 5:])
        torch.testing.assert_close(denominator[0], expected_denominator[0])
        torch.testing.assert_close(denominator[1], expected_denominator[1])
        assert ctx.maximum_feature_width == 2
        snapshot = feature.get_diagnostic_snapshot()
        assert snapshot["full_width_production"] is False
        assert snapshot["max_produced_tile_columns"] == 2
        assert snapshot["feature_tile_count"] == 3
        assert snapshot["max_produced_tile_bytes"] <= 2 * 8
        assert nonfeature.get_diagnostic_snapshot()["nonfeature_tile_count"] == 1
    finally:
        feature.cleanup()
        nonfeature.cleanup()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiled_denominator_uses_canonical_global_max_chunk_order(dtype: torch.dtype) -> None:
    feature_rows = torch.full((2, 5003), 1e-7, dtype=dtype)
    feature_rows[0, 4097] = 1e10
    feature_rows[1, 3] = -1e8
    nonfeature_rows = torch.tensor([[3e-4, -2e5], [7e7, -9e-6]], dtype=dtype)

    class Context:
        def produce_row_tiles(self, *args, feature_column_tile_size, consume_feature_tile, **kwargs):
            del args, kwargs
            for start in range(0, feature_rows.shape[1], feature_column_tile_size):
                end = min(start + feature_column_tile_size, feature_rows.shape[1])
                consume_feature_tile(start, end, feature_rows[:, start:end])
            return nonfeature_rows

    feature = _ColumnTiledFeatureRowStore(
        n_rows=2, n_feature_columns=5003, column_tile_size=257, dtype=dtype
    )
    nonfeature = _ColumnTiledFeatureRowStore(
        n_rows=2, n_feature_columns=2, column_tile_size=2, dtype=dtype
    )
    try:
        _, actual = produce_and_store_tiled_rows(
            ctx=Context(), layers=torch.tensor([0, 0]), positions=torch.tensor([0, 0]),
            inject_values=torch.ones((2, 1)), row_start=0, feature_row_store=feature,
            nonfeature_row_store=nonfeature, feature_column_tile_size=257, dtype=dtype,
            phase_label="phase3_logits",
        )
        maximum = torch.maximum(feature_rows.abs().amax(dim=1), nonfeature_rows.abs().amax(dim=1))
        scaled = torch.zeros_like(maximum)
        for start in range(0, feature_rows.shape[1], 4096):
            scaled += (feature_rows[:, start : start + 4096].abs() / maximum[:, None]).sum(dim=1)
        scaled += (nonfeature_rows.abs() / maximum[:, None]).sum(dim=1)
        assert torch.equal(actual[0], maximum)
        assert torch.equal(actual[1], scaled)
    finally:
        feature.cleanup()
        nonfeature.cleanup()
