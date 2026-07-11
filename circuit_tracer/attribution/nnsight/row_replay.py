"""No-retention attribution-row recipes and deterministic replay readers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch


RowSourceKind = Literal["logit", "feature"]
ReplayTileProducer = Callable[["RowRecipe", int, int], torch.Tensor]
ReplayLifecycleHook = Callable[[], None]


@dataclass(frozen=True)
class RowRecipe:
    """Canonical information required to reproduce one attribution row."""

    ordinal: int
    source_kind: RowSourceKind
    layer: int
    position: int
    injection: torch.Tensor | None = None
    stable_reference: tuple[str, int] | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("row recipe ordinal must be >= 0")
        if (self.injection is None) == (self.stable_reference is None):
            raise ValueError("row recipe requires exactly one injection or stable reference")
        if self.injection is not None:
            injection = self.injection.detach().to(device="cpu").contiguous().clone()
            object.__setattr__(self, "injection", injection)

    @property
    def retained_bytes(self) -> int:
        if self.injection is None:
            return 0
        return int(self.injection.numel() * self.injection.element_size())


class ReplayGraphLifecycle:
    """Reset and deterministically rebuild the Phase-1 graph for replay requests."""

    def __init__(
        self,
        *,
        reset: ReplayLifecycleHook,
        rebuild_forward: ReplayLifecycleHook,
        release: ReplayLifecycleHook,
    ) -> None:
        self._reset = reset
        self._rebuild_forward = rebuild_forward
        self._release = release
        self.graph_rebuild_count = 0
        self.forward_count = 0
        self.backward_count = 0
        self._retained = False

    def begin_request(self) -> None:
        if self._retained:
            self.release()
        self._retained = True
        try:
            self._reset()
            self._rebuild_forward()
        except BaseException:
            self.release()
            raise
        self.graph_rebuild_count += 1
        self.forward_count += 1

    def record_backward(self) -> None:
        if not self._retained:
            raise RuntimeError("cannot record replay backward without a retained graph")
        self.backward_count += 1

    def release(self) -> None:
        if not self._retained:
            return
        try:
            self._release()
        finally:
            self._retained = False

    def cleanup(self) -> None:
        self.release()


class RowRecipeLedger:
    """O(K) replay metadata with an optional bounded tile cache.

    The ledger deliberately has no append API accepting a row matrix.  A replay
    producer may return only the requested tile, preventing accidental KxN
    retention in ``none_recompute`` mode.
    """

    def __init__(
        self,
        *,
        n_rows: int,
        n_feature_columns: int,
        dtype: torch.dtype,
        producer: ReplayTileProducer,
        lifecycle: ReplayGraphLifecycle,
        semantic_fingerprint: Mapping[str, object],
        execution_fingerprint: Mapping[str, object],
        provider_fingerprint: Mapping[str, object],
        tile_cache_bytes: int = 0,
        max_request_rows: int | None = None,
        max_request_columns: int | None = None,
    ) -> None:
        if n_rows < 0 or n_feature_columns < 0:
            raise ValueError("ledger dimensions must be nonnegative")
        if tile_cache_bytes < 0:
            raise ValueError("tile_cache_bytes must be >= 0")
        if max_request_rows is not None and max_request_rows <= 0:
            raise ValueError("max_request_rows must be > 0")
        if max_request_columns is not None and max_request_columns <= 0:
            raise ValueError("max_request_columns must be > 0")
        self.n_rows = int(n_rows)
        self.n_feature_columns = int(n_feature_columns)
        self.dtype = dtype
        self.row_abs_max = torch.zeros(n_rows, dtype=dtype)
        self.row_l1_scaled = torch.zeros(n_rows, dtype=dtype)
        self.row_to_node = torch.full((n_rows,), -1, dtype=torch.int64)
        self.semantic_fingerprint = dict(semantic_fingerprint)
        self.execution_fingerprint = dict(execution_fingerprint)
        self.provider_fingerprint = dict(provider_fingerprint)
        self._producer = producer
        self._lifecycle = lifecycle
        self._recipes: list[RowRecipe | None] = [None] * n_rows
        self._cache_limit = int(tile_cache_bytes)
        self._max_request_rows = max_request_rows
        self._max_request_columns = max_request_columns
        self._cache: OrderedDict[tuple[int, int, int, int], torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self._closed = False
        self._stats = {
            "replay_request_count": 0,
            "replay_tile_count": 0,
            "replay_row_count": 0,
            "max_tile_rows": 0,
            "max_tile_columns": 0,
            "max_tile_bytes": 0,
        }

    @property
    def path(self) -> None:
        return None

    def append_recipe(
        self,
        recipe: RowRecipe,
        *,
        node_index: int,
        denominator: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._ensure_open()
        ordinal = recipe.ordinal
        if ordinal >= self.n_rows:
            raise IndexError("row recipe ordinal exceeds ledger capacity")
        if self._recipes[ordinal] is not None:
            raise ValueError(f"row recipe ordinal {ordinal} was already recorded")
        row_abs_max, row_l1_scaled = denominator
        if row_abs_max.numel() != 1 or row_l1_scaled.numel() != 1:
            raise ValueError("append_recipe denominator must describe exactly one row")
        self._recipes[ordinal] = recipe
        self.row_to_node[ordinal] = int(node_index)
        self.row_abs_max[ordinal] = row_abs_max.detach().to(device="cpu", dtype=self.dtype)
        self.row_l1_scaled[ordinal] = row_l1_scaled.detach().to(device="cpu", dtype=self.dtype)

    def read_tile(
        self, row_start: int, row_end: int, column_start: int, column_end: int, **_: object
    ) -> torch.Tensor:
        self._ensure_open()
        self._validate_range(row_start, row_end, column_start, column_end)
        key = (row_start, row_end, column_start, column_end)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.clone()
        recipes = self._recipes[row_start:row_end]
        if any(recipe is None for recipe in recipes):
            raise RuntimeError("requested replay tile contains unrecorded rows")
        self._stats["replay_request_count"] += 1
        self._lifecycle.begin_request()
        tiles: list[torch.Tensor] = []
        try:
            for recipe in recipes:
                assert recipe is not None
                tile = self._producer(recipe, column_start, column_end)
                self._lifecycle.record_backward()
                expected = (1, column_end - column_start)
                if tuple(tile.shape) != expected:
                    raise ValueError(
                        f"replay producer returned shape {tuple(tile.shape)}; expected {expected}"
                    )
                tiles.append(tile.detach().to(device="cpu", dtype=self.dtype))
            result = torch.cat(tiles, dim=0) if tiles else torch.empty(
                (0, column_end - column_start), dtype=self.dtype
            )
        finally:
            self._lifecycle.release()
        tile_bytes = int(result.numel() * result.element_size())
        self._stats["replay_tile_count"] += 1
        self._stats["replay_row_count"] += row_end - row_start
        self._stats["max_tile_rows"] = max(self._stats["max_tile_rows"], row_end - row_start)
        self._stats["max_tile_columns"] = max(
            self._stats["max_tile_columns"], column_end - column_start
        )
        self._stats["max_tile_bytes"] = max(self._stats["max_tile_bytes"], tile_bytes)
        self._cache_put(key, result)
        return result

    def materialize_dense_feature_slice(
        self,
        *,
        row_start: int,
        row_end: int,
        selected_feature_columns: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        columns = selected_feature_columns.detach().to(device="cpu", dtype=torch.long)
        if columns.numel() == 0:
            return torch.empty((row_end - row_start, 0), dtype=self.dtype)
        pieces = [self.read_tile(row_start, row_end, int(column), int(column) + 1) for column in columns]
        return torch.cat(pieces, dim=1)

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        retained_recipe_bytes = sum(
            recipe.retained_bytes for recipe in self._recipes if recipe is not None
        )
        return {
            "feature_row_retention": "none_recompute",
            "no_retained_full_feature_matrix": True,
            "no_kxn_file": True,
            "request_bounds_enforced": bool(
                self._max_request_rows is not None and self._max_request_columns is not None
            ),
            "configured_max_request_rows": self._max_request_rows,
            "configured_max_request_columns": self._max_request_columns,
            "retained_recipe_bytes": retained_recipe_bytes,
            **self._stats,
            "graph_rebuild_count": self._lifecycle.graph_rebuild_count,
            "graph_forward_count": self._lifecycle.forward_count,
            "graph_backward_count": self._lifecycle.backward_count,
            "cache_bytes": self._cache_bytes,
            "apparent_file_bytes": 0,
            "allocated_file_bytes": 0,
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self._lifecycle.cleanup()
        finally:
            self._cache.clear()
            self._cache_bytes = 0
            self._closed = True

    def _cache_put(self, key: tuple[int, int, int, int], value: torch.Tensor) -> None:
        nbytes = int(value.numel() * value.element_size())
        if self._cache_limit == 0 or nbytes > self._cache_limit:
            return
        while self._cache and self._cache_bytes + nbytes > self._cache_limit:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= int(evicted.numel() * evicted.element_size())
        self._cache[key] = value.clone()
        self._cache_bytes += nbytes

    def _validate_range(self, rs: int, re: int, cs: int, ce: int) -> None:
        if not (0 <= rs <= re <= self.n_rows):
            raise IndexError("replay row range is out of bounds")
        if not (0 <= cs <= ce <= self.n_feature_columns):
            raise IndexError("replay column range is out of bounds")
        if self._max_request_rows is not None and re - rs > self._max_request_rows:
            raise ValueError("replay request exceeds configured row bound")
        if self._max_request_columns is not None and ce - cs > self._max_request_columns:
            raise ValueError("replay request exceeds configured column bound")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("row recipe ledger is closed")


class NonfeatureProjectionLedger(RowRecipeLedger):
    """Bounded replay metadata for compact error/token projections.

    This is deliberately a distinct owner from the feature ledger so Phase 5
    can request only selected nonfeature columns without creating a KxN file.
    """

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {
            **super().get_diagnostic_snapshot(),
            "projection_kind": "nonfeature_recompute",
        }
