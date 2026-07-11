import os

import pytest
import torch

from circuit_tracer.attribution.nnsight.row_store import (
    _ColumnTiledFeatureRowStore,
    _FileBackedFeatureRowStore,
)
from circuit_tracer.graph import (
    compute_partial_feature_influences_streaming,
    compute_partial_feature_influences_tiled,
)


def _append(store, rows: torch.Tensor) -> None:
    abs_rows = rows.abs()
    row_abs_max = abs_rows.amax(dim=1)
    row_l1_scaled = torch.where(
        row_abs_max > 0, (abs_rows / row_abs_max[:, None]).sum(dim=1), 0
    )
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
