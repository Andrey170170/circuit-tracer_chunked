"""File-backed dense attribution row storage and policy helpers."""

import os
import tempfile
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import torch

from circuit_tracer.attribution.nnsight.numerics import _row_abs_sums_to_scaled_l1
from circuit_tracer.utils.telemetry import TelemetryRecorder

_ROW_STORE_CACHE_CONTROL_DEFAULT: Literal["off"] = "off"


class FeatureRowStore(Protocol):
    """Storage contract for exact full-retention feature rows."""

    n_rows: int
    n_feature_columns: int
    row_abs_max: torch.Tensor
    row_l1_scaled: torch.Tensor

    def append_rows(self, *, row_start: int, feature_rows: torch.Tensor, **kwargs): ...
    def read_feature_rows(self, row_start: int, row_end: int, **kwargs) -> torch.Tensor: ...
    def read_tile(
        self, row_start: int, row_end: int, column_start: int, column_end: int, **kwargs
    ) -> torch.Tensor: ...
    def materialize_dense_feature_slice(self, **kwargs) -> torch.Tensor: ...
    def get_diagnostic_snapshot(self) -> dict[str, object]: ...
    def cleanup(self) -> None: ...

_ROW_STORE_TEMP_ROOT_POLICY_DEFAULT: Literal["default"] = "default"
_ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL: Literal["env_node_local"] = "env_node_local"
_ROW_STORE_TEMP_ROOT_POLICY_BY_NAME: dict[str, str] = {
    "default": _ROW_STORE_TEMP_ROOT_POLICY_DEFAULT,
    "tempfile_default": _ROW_STORE_TEMP_ROOT_POLICY_DEFAULT,
    "env_node_local": _ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL,
    "node_local": _ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL,
}
_ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_V1: Literal[
    "fadvise_dontneed_after_append_v1"
] = "fadvise_dontneed_after_append_v1"
_ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_AND_READ_V1: Literal[
    "fadvise_dontneed_after_append_and_read_v1"
] = "fadvise_dontneed_after_append_and_read_v1"
_RowStoreCacheControlMode = Literal[
    "off",
    "fadvise_dontneed_after_append_v1",
    "fadvise_dontneed_after_append_and_read_v1",
]
_ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE: dict[str, str] = {
    "off": "off",
    "fadvise_dontneed_after_append_v1": "fadvise_dontneed_after_append_v1",
    "fadvise_dontneed_after_append_and_read_v1": ("fadvise_dontneed_after_append_and_read_v1"),
}

@dataclass(frozen=True)
class _RowStoreCacheControlConfig:
    requested_mode: _RowStoreCacheControlMode
    effective_mode: _RowStoreCacheControlMode
    default_mode: Literal["off"]
    mode_applicable: bool
    effective_behavior: Literal["requested", "off_reference_execution"]
    fallback_reason: str | None


@dataclass(frozen=True)
class _RowStoreTempRootSelection:
    policy: Literal["default", "env_node_local"]
    requested_root: str | None
    selected_root: str | None
    selected_path: str
    fallback_reason: str | None



class _FileBackedFeatureRowStore:
    """Append-only dense feature-row store backed by a temporary memmap file."""

    def __init__(
        self,
        *,
        n_rows: int,
        n_feature_columns: int,
        dtype: torch.dtype,
        row_abs_sum_dtype: torch.dtype = torch.float32,
        read_chunk_cache_bytes: int = 0,
        prepared_read_cache_bytes: int = 0,
        row_store_cache_control_mode: _RowStoreCacheControlMode = "off",
        temp_root_policy: Literal["default", "env_node_local"] = "default",
        temp_root: str | os.PathLike[str] | None = None,
        preallocate: bool = False,
        telemetry_recorder: TelemetryRecorder | None = None,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported feature row store dtype: {dtype}")
        if row_abs_sum_dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported row_abs_sum_dtype: {row_abs_sum_dtype}")
        if row_store_cache_control_mode not in _ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE:
            allowed = ", ".join(sorted(_ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE))
            raise ValueError(
                "Unsupported row_store_cache_control_mode: "
                f"{row_store_cache_control_mode!r}; expected one of: {allowed}"
            )

        self.n_rows = n_rows
        self.n_feature_columns = n_feature_columns
        self.row_abs_max = torch.zeros(n_rows, dtype=row_abs_sum_dtype)
        self.row_l1_scaled = torch.zeros(n_rows, dtype=row_abs_sum_dtype)

        self._dtype = dtype
        self._np_dtype = np.float32 if dtype == torch.float32 else np.float64
        temp_root_selection, tmpdir = _select_row_store_temp_root(
            temp_root_policy=temp_root_policy,
            temp_root=temp_root,
        )
        self._tmpdir = tmpdir
        self._path = f"{self._tmpdir.name}/feature_rows.memmap"
        self._row_nbytes = int(np.dtype(self._np_dtype).itemsize * n_feature_columns)
        total_nbytes = int(self._row_nbytes * n_rows)
        preallocate_requested = bool(preallocate)
        posix_fallocate = getattr(os, "posix_fallocate", None)
        preallocate_status = "disabled"
        preallocate_error: str | None = None
        with open(self._path, "wb") as handle:
            handle.truncate(total_nbytes)
            if preallocate_requested:
                if callable(posix_fallocate):
                    try:
                        posix_fallocate(handle.fileno(), 0, total_nbytes)
                        preallocate_status = "succeeded"
                    except OSError as exc:
                        preallocate_status = "failed"
                        preallocate_error = repr(exc)
                else:
                    preallocate_status = "unavailable"
                    preallocate_error = "os.posix_fallocate is unavailable"
        self._write_fd: int | None = os.open(self._path, os.O_RDWR)
        self._row_store_cache_control_effective_mode = row_store_cache_control_mode
        self._fadvise_dontneed_after_append_enabled = bool(
            row_store_cache_control_mode
            in {
                _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_V1,
                _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_AND_READ_V1,
            }
        )
        self._fadvise_dontneed_after_read_enabled = bool(
            row_store_cache_control_mode
            == _ROW_STORE_CACHE_CONTROL_FADVISE_DONTNEED_AFTER_APPEND_AND_READ_V1
        )
        self._posix_fadvise = getattr(os, "posix_fadvise", None)
        self._posix_fadvise_dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        self._rows: np.memmap | None = np.memmap(
            self._path,
            mode="r+",
            dtype=self._np_dtype,
            shape=(n_rows, n_feature_columns),
        )
        self._read_chunk_cache_max_bytes = max(0, int(read_chunk_cache_bytes))
        self._read_chunk_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self._read_chunk_cache_nbytes = 0
        self._prepared_read_cache_max_bytes = max(0, int(prepared_read_cache_bytes))
        self._prepared_read_cache: OrderedDict[tuple[str, int, int, str, str], torch.Tensor] = (
            OrderedDict()
        )
        self._prepared_read_cache_nbytes = 0
        self._telemetry_recorder = telemetry_recorder
        self._closed = False
        self._diagnostic_stats: dict[str, object] = {
            "append_call_count": 0,
            "append_row_count": 0,
            "row_store_cache_control_effective_mode": (
                self._row_store_cache_control_effective_mode
            ),
            "row_store_cache_control_advisory_available": int(
                callable(self._posix_fadvise) and self._posix_fadvise_dontneed is not None
            ),
            "row_store_cache_control_advisory_call_count": 0,
            "row_store_cache_control_advisory_bytes": 0,
            "row_store_cache_control_advisory_failure_count": 0,
            "row_store_cache_control_advisory_unavailable_count": 0,
            "row_store_cache_control_advisory_skipped_count": 0,
            "row_store_cache_control_advisory_last_error": None,
            "row_store_cache_control_append_advisory_call_count": 0,
            "row_store_cache_control_append_advisory_bytes": 0,
            "row_store_cache_control_append_advisory_failure_count": 0,
            "row_store_cache_control_append_advisory_unavailable_count": 0,
            "row_store_cache_control_append_advisory_skipped_count": 0,
            "row_store_cache_control_read_advisory_call_count": 0,
            "row_store_cache_control_read_advisory_bytes": 0,
            "row_store_cache_control_read_advisory_failure_count": 0,
            "row_store_cache_control_read_advisory_unavailable_count": 0,
            "row_store_cache_control_read_advisory_skipped_count": 0,
            "temp_root_policy": temp_root_selection.policy,
            "temp_root_requested": temp_root_selection.requested_root,
            "temp_root_selected": temp_root_selection.selected_root,
            "temp_root_path": temp_root_selection.selected_path,
            "temp_root_fallback_reason": temp_root_selection.fallback_reason,
            "preallocate_requested": int(preallocate_requested),
            "preallocate_available": int(callable(posix_fallocate)),
            "preallocate_status": preallocate_status,
            "preallocate_error": preallocate_error,
            "preallocate_nbytes": int(total_nbytes) if preallocate_requested else 0,
            "read_call_count": 0,
            "read_row_count": 0,
            "read_last_row_start": None,
            "read_last_row_end": None,
            "read_cache_enabled": int(self._read_chunk_cache_max_bytes > 0),
            "read_cache_hit_count": 0,
            "read_cache_miss_count": 0,
            "read_cache_hit_row_count": 0,
            "read_cache_miss_row_count": 0,
            "read_cache_eviction_count": 0,
            "read_cache_store_attempt_count": 0,
            "read_cache_store_success_count": 0,
            "read_cache_store_skip_disabled_count": 0,
            "read_cache_store_skip_too_large_count": 0,
            "read_cache_entry_count": 0,
            "read_cache_nbytes": 0,
            "read_cache_max_bytes": int(self._read_chunk_cache_max_bytes),
            "prepared_read_cache_enabled": int(self._prepared_read_cache_max_bytes > 0),
            "prepared_read_cache_max_bytes": int(self._prepared_read_cache_max_bytes),
            "prepared_read_cache_entry_count": 0,
            "prepared_read_cache_nbytes": 0,
            "prepared_read_cache_hit_count": 0,
            "prepared_read_cache_miss_count": 0,
            "prepared_read_cache_hit_row_count": 0,
            "prepared_read_cache_miss_row_count": 0,
            "prepared_read_cache_eviction_count": 0,
            "prepared_read_cache_invalidation_count": 0,
            "prepared_read_cache_invalidation_entry_count": 0,
            "prepared_read_cache_store_attempt_count": 0,
            "prepared_read_cache_store_success_count": 0,
            "prepared_read_cache_store_skip_disabled_count": 0,
            "prepared_read_cache_store_skip_too_large_count": 0,
            "prepared_read_cache_prepare_elapsed_ms_total": 0.0,
            "materialize_call_count": 0,
            "materialize_row_count": 0,
            "materialize_column_count": 0,
            "materialize_last_row_start": None,
            "materialize_last_row_end": None,
        }

    def _telemetry_timer(
        self,
        *,
        name: str,
        phase: str | None,
        attrs: dict[str, object],
    ):
        if self._telemetry_recorder is None:
            return nullcontext()
        return self._telemetry_recorder.timer(
            scope="op",
            name=name,
            phase=phase,
            attrs=attrs,
        )

    @property
    def path(self) -> str:
        return self._path

    @property
    def nbytes(self) -> int:
        rows = self._require_open_rows()
        return int(rows.size * rows.dtype.itemsize)

    def _require_open_rows(self) -> np.memmap:
        if self._closed or self._rows is None:
            raise RuntimeError("feature row store has been cleaned up")
        return self._rows

    def _require_open_write_fd(self) -> int:
        if self._closed or self._write_fd is None:
            raise RuntimeError("feature row store has been cleaned up")
        return self._write_fd

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _sync_read_cache_snapshot(self) -> None:
        self._diagnostic_stats["read_cache_entry_count"] = int(len(self._read_chunk_cache))
        self._diagnostic_stats["read_cache_nbytes"] = int(self._read_chunk_cache_nbytes)

    def _sync_prepared_read_cache_snapshot(self) -> None:
        self._diagnostic_stats["prepared_read_cache_entry_count"] = int(
            len(self._prepared_read_cache)
        )
        self._diagnostic_stats["prepared_read_cache_nbytes"] = int(self._prepared_read_cache_nbytes)

    def _drop_read_chunk(self, key: tuple[int, int], *, count_eviction: bool = True) -> None:
        chunk = self._read_chunk_cache.pop(key, None)
        if chunk is None:
            return
        self._read_chunk_cache_nbytes = max(
            0,
            self._read_chunk_cache_nbytes - self._tensor_nbytes(chunk),
        )
        if count_eviction:
            self._diagnostic_stats["read_cache_eviction_count"] = (
                int(self._diagnostic_stats["read_cache_eviction_count"] or 0) + 1
            )

    def _drop_prepared_read_chunk(
        self,
        key: tuple[str, int, int, str, str],
        *,
        count_eviction: bool = True,
    ) -> None:
        chunk = self._prepared_read_cache.pop(key, None)
        if chunk is None:
            return
        self._prepared_read_cache_nbytes = max(
            0,
            self._prepared_read_cache_nbytes - self._tensor_nbytes(chunk),
        )
        if count_eviction:
            self._increment_diagnostic_counter("prepared_read_cache_eviction_count")

    def _insert_prepared_read_chunk(
        self,
        key: tuple[str, int, int, str, str],
        chunk: torch.Tensor,
    ) -> str:
        self._increment_diagnostic_counter("prepared_read_cache_store_attempt_count")
        if self._prepared_read_cache_max_bytes <= 0:
            self._increment_diagnostic_counter("prepared_read_cache_store_skip_disabled_count")
            return "disabled"
        chunk_nbytes = self._tensor_nbytes(chunk)
        if chunk_nbytes > self._prepared_read_cache_max_bytes:
            self._increment_diagnostic_counter("prepared_read_cache_store_skip_too_large_count")
            return "too_large"
        while (
            self._prepared_read_cache
            and self._prepared_read_cache_nbytes + chunk_nbytes
            > self._prepared_read_cache_max_bytes
        ):
            self._drop_prepared_read_chunk(next(iter(self._prepared_read_cache)))
        self._prepared_read_cache[key] = chunk
        self._prepared_read_cache.move_to_end(key)
        self._prepared_read_cache_nbytes += chunk_nbytes
        self._increment_diagnostic_counter("prepared_read_cache_store_success_count")
        return "stored"

    def _insert_read_chunk(self, key: tuple[int, int], chunk: torch.Tensor) -> str:
        self._diagnostic_stats["read_cache_store_attempt_count"] = (
            int(self._diagnostic_stats["read_cache_store_attempt_count"] or 0) + 1
        )
        if self._read_chunk_cache_max_bytes <= 0:
            self._diagnostic_stats["read_cache_store_skip_disabled_count"] = (
                int(self._diagnostic_stats["read_cache_store_skip_disabled_count"] or 0) + 1
            )
            return "disabled"

        chunk_nbytes = self._tensor_nbytes(chunk)
        if chunk_nbytes > self._read_chunk_cache_max_bytes:
            self._diagnostic_stats["read_cache_store_skip_too_large_count"] = (
                int(self._diagnostic_stats["read_cache_store_skip_too_large_count"] or 0) + 1
            )
            return "too_large"

        while (
            self._read_chunk_cache
            and self._read_chunk_cache_nbytes + chunk_nbytes > self._read_chunk_cache_max_bytes
        ):
            oldest_key = next(iter(self._read_chunk_cache))
            self._drop_read_chunk(oldest_key, count_eviction=True)

        self._read_chunk_cache[key] = chunk
        self._read_chunk_cache.move_to_end(key)
        self._read_chunk_cache_nbytes += chunk_nbytes
        self._diagnostic_stats["read_cache_store_success_count"] = (
            int(self._diagnostic_stats["read_cache_store_success_count"] or 0) + 1
        )
        return "stored"

    def _evict_overlapping_read_chunks(self, row_start: int, row_end: int) -> None:
        if not self._read_chunk_cache:
            return
        overlapping = [
            key for key in self._read_chunk_cache if key[0] < row_end and key[1] > row_start
        ]
        for key in overlapping:
            self._drop_read_chunk(key, count_eviction=True)

    def _evict_overlapping_prepared_read_chunks(self, row_start: int, row_end: int) -> None:
        if not self._prepared_read_cache:
            return
        overlapping = [
            key for key in self._prepared_read_cache if key[1] < row_end and key[2] > row_start
        ]
        if overlapping:
            self._increment_diagnostic_counter("prepared_read_cache_invalidation_count")
            self._increment_diagnostic_counter(
                "prepared_read_cache_invalidation_entry_count",
                delta=len(overlapping),
            )
        for key in overlapping:
            self._drop_prepared_read_chunk(key, count_eviction=False)

    def _increment_diagnostic_counter(self, key: str, *, delta: int = 1) -> None:
        self._diagnostic_stats[key] = int(self._diagnostic_stats.get(key, 0) or 0) + int(delta)

    def _apply_row_store_cache_control_advisory(
        self,
        *,
        write_fd: int,
        byte_offset: int,
        byte_length: int,
        enabled: bool,
        counter_prefix: str,
    ) -> None:
        if byte_length <= 0 or not enabled:
            self._increment_diagnostic_counter("row_store_cache_control_advisory_skipped_count")
            self._increment_diagnostic_counter(
                f"row_store_cache_control_{counter_prefix}_advisory_skipped_count"
            )
            return

        posix_fadvise = self._posix_fadvise
        dontneed_flag = self._posix_fadvise_dontneed
        if not callable(posix_fadvise) or dontneed_flag is None:
            self._diagnostic_stats["row_store_cache_control_advisory_available"] = 0
            self._increment_diagnostic_counter("row_store_cache_control_advisory_unavailable_count")
            self._increment_diagnostic_counter(
                f"row_store_cache_control_{counter_prefix}_advisory_unavailable_count"
            )
            return

        try:
            posix_fadvise(
                write_fd,
                int(byte_offset),
                int(byte_length),
                int(dontneed_flag),
            )
        except Exception as exc:
            self._increment_diagnostic_counter("row_store_cache_control_advisory_failure_count")
            self._increment_diagnostic_counter(
                f"row_store_cache_control_{counter_prefix}_advisory_failure_count"
            )
            self._diagnostic_stats["row_store_cache_control_advisory_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return

        self._increment_diagnostic_counter("row_store_cache_control_advisory_call_count")
        self._increment_diagnostic_counter(
            f"row_store_cache_control_{counter_prefix}_advisory_call_count"
        )
        self._increment_diagnostic_counter(
            "row_store_cache_control_advisory_bytes",
            delta=int(byte_length),
        )
        self._increment_diagnostic_counter(
            f"row_store_cache_control_{counter_prefix}_advisory_bytes",
            delta=int(byte_length),
        )

    def _apply_row_store_cache_control_after_append(
        self,
        *,
        write_fd: int,
        byte_offset: int,
        byte_length: int,
    ) -> None:
        self._apply_row_store_cache_control_advisory(
            write_fd=write_fd,
            byte_offset=byte_offset,
            byte_length=byte_length,
            enabled=self._fadvise_dontneed_after_append_enabled,
            counter_prefix="append",
        )

    def _apply_row_store_cache_control_after_safe_read(
        self,
        *,
        row_start: int,
        row_end: int,
    ) -> None:
        self._apply_row_store_cache_control_advisory(
            write_fd=self._require_open_write_fd(),
            byte_offset=int(row_start * self._row_nbytes),
            byte_length=int((row_end - row_start) * self._row_nbytes),
            enabled=self._fadvise_dontneed_after_read_enabled,
            counter_prefix="read",
        )

    def append_rows(
        self,
        *,
        row_start: int,
        feature_rows: torch.Tensor,
        row_denominator_scaled_l1: tuple[torch.Tensor, torch.Tensor] | None = None,
        full_row_abs_sums: torch.Tensor | None = None,
        phase: str | None = None,
    ) -> dict[str, float]:
        if feature_rows.ndim != 2:
            raise ValueError("feature_rows must be rank-2")
        row_count, n_feature_cols = feature_rows.shape
        if n_feature_cols != self.n_feature_columns:
            raise ValueError(
                "feature_rows second dimension must equal configured n_feature_columns"
            )
        if row_denominator_scaled_l1 is not None and full_row_abs_sums is not None:
            raise ValueError(
                "Provide either row_denominator_scaled_l1 or full_row_abs_sums, not both"
            )

        if row_denominator_scaled_l1 is not None:
            row_abs_max, row_l1_scaled = row_denominator_scaled_l1
            if row_abs_max.numel() != row_count:
                raise ValueError("row_abs_max length must equal number of feature_rows")
            if row_l1_scaled.numel() != row_count:
                raise ValueError("row_l1_scaled length must equal number of feature_rows")
        elif full_row_abs_sums is not None:
            if full_row_abs_sums.numel() != row_count:
                raise ValueError("full_row_abs_sums length must equal number of feature_rows")
            row_abs_max, row_l1_scaled = _row_abs_sums_to_scaled_l1(
                full_row_abs_sums,
                dtype=self.row_abs_max.dtype,
            )
        else:
            raise ValueError("row denominator data must be provided")

        row_end = row_start + row_count
        if row_start < 0 or row_end > self.n_rows:
            raise ValueError("row range is out of bounds for file-backed store")

        append_total_start = time.perf_counter()
        cpu_prepare_elapsed_ms = 0.0
        contiguous_elapsed_ms = 0.0
        numpy_elapsed_ms = 0.0
        pwrite_elapsed_ms = 0.0
        denominator_copy_elapsed_ms = 0.0

        with self._telemetry_timer(
            name="feature_row_store.append_rows",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_count,
                "feature_columns": n_feature_cols,
            },
        ):
            write_fd = self._require_open_write_fd()
            cpu_prepare_start = time.perf_counter()
            feature_rows_cpu = feature_rows.detach()
            if feature_rows_cpu.device.type != "cpu" or feature_rows_cpu.dtype != self._dtype:
                feature_rows_cpu = feature_rows_cpu.to(device="cpu", dtype=self._dtype)
            cpu_prepare_elapsed_ms = (time.perf_counter() - cpu_prepare_start) * 1000.0

            numpy_start = time.perf_counter()
            feature_rows_np = np.asarray(feature_rows_cpu.numpy(), dtype=self._np_dtype, order="C")
            numpy_elapsed_ms = (time.perf_counter() - numpy_start) * 1000.0
            if not feature_rows_np.flags.c_contiguous:
                contiguous_start = time.perf_counter()
                feature_rows_np = np.ascontiguousarray(feature_rows_np, dtype=self._np_dtype)
                contiguous_elapsed_ms = (time.perf_counter() - contiguous_start) * 1000.0
            payload = memoryview(feature_rows_np).cast("B")
            expected_nbytes = int(row_count * self._row_nbytes)
            if payload.nbytes != expected_nbytes:
                raise RuntimeError(
                    "feature row store append payload size mismatch: "
                    f"expected {expected_nbytes} bytes, got {payload.nbytes}"
                )

            byte_offset = int(row_start * self._row_nbytes)
            bytes_written = 0
            pwrite_start = time.perf_counter()
            while bytes_written < expected_nbytes:
                wrote = os.pwrite(write_fd, payload[bytes_written:], byte_offset + bytes_written)
                if wrote <= 0:
                    raise OSError("feature row store append write failed")
                bytes_written += wrote
            pwrite_elapsed_ms = (time.perf_counter() - pwrite_start) * 1000.0
            self._apply_row_store_cache_control_after_append(
                write_fd=write_fd,
                byte_offset=byte_offset,
                byte_length=expected_nbytes,
            )

            denominator_copy_start = time.perf_counter()
            row_abs_max_cpu = row_abs_max.detach()
            if (
                row_abs_max_cpu.device.type != "cpu"
                or row_abs_max_cpu.dtype != self.row_abs_max.dtype
            ):
                row_abs_max_cpu = row_abs_max_cpu.to(
                    device=self.row_abs_max.device,
                    dtype=self.row_abs_max.dtype,
                )
            row_l1_scaled_cpu = row_l1_scaled.detach()
            if (
                row_l1_scaled_cpu.device.type != "cpu"
                or row_l1_scaled_cpu.dtype != self.row_l1_scaled.dtype
            ):
                row_l1_scaled_cpu = row_l1_scaled_cpu.to(
                    device=self.row_l1_scaled.device,
                    dtype=self.row_l1_scaled.dtype,
                )

            self.row_abs_max[row_start:row_end] = row_abs_max_cpu
            self.row_l1_scaled[row_start:row_end] = row_l1_scaled_cpu
            denominator_copy_elapsed_ms = (time.perf_counter() - denominator_copy_start) * 1000.0

        self._diagnostic_stats["append_call_count"] = (
            int(self._diagnostic_stats["append_call_count"] or 0) + 1
        )
        self._diagnostic_stats["append_row_count"] = int(
            self._diagnostic_stats["append_row_count"] or 0
        ) + int(row_count)
        self._evict_overlapping_read_chunks(row_start, row_end)
        self._evict_overlapping_prepared_read_chunks(row_start, row_end)
        self._sync_read_cache_snapshot()
        self._sync_prepared_read_cache_snapshot()
        return {
            "row_store_append_cpu_prepare_elapsed_ms": float(cpu_prepare_elapsed_ms),
            "row_store_append_contiguous_elapsed_ms": float(contiguous_elapsed_ms),
            "row_store_append_numpy_elapsed_ms": float(numpy_elapsed_ms),
            "row_store_append_pwrite_elapsed_ms": float(pwrite_elapsed_ms),
            "row_store_append_denominator_copy_elapsed_ms": float(denominator_copy_elapsed_ms),
            "row_store_append_total_elapsed_ms": float(
                (time.perf_counter() - append_total_start) * 1000.0
            ),
        }

    def read_feature_rows(
        self,
        row_start: int,
        row_end: int,
        *,
        phase: str | None = None,
    ) -> torch.Tensor:
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for file-backed store")

        cache_key = (int(row_start), int(row_end))
        cached = self._read_chunk_cache.get(cache_key)
        cache_hit = cached is not None

        with self._telemetry_timer(
            name="feature_row_store.read_rows",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_end - row_start,
                "cache_hit": cache_hit,
            },
        ):
            if cached is not None:
                self._read_chunk_cache.move_to_end(cache_key)
                result = cached
            else:
                rows = self._require_open_rows()
                result = torch.from_numpy(np.asarray(rows[row_start:row_end], dtype=self._np_dtype))
                self._insert_read_chunk(cache_key, result)

        self._diagnostic_stats["read_call_count"] = (
            int(self._diagnostic_stats["read_call_count"] or 0) + 1
        )
        row_count = int(row_end - row_start)
        self._diagnostic_stats["read_row_count"] = (
            int(self._diagnostic_stats["read_row_count"] or 0) + row_count
        )
        self._diagnostic_stats["read_last_row_start"] = int(row_start)
        self._diagnostic_stats["read_last_row_end"] = int(row_end)
        if cache_hit:
            self._diagnostic_stats["read_cache_hit_count"] = (
                int(self._diagnostic_stats["read_cache_hit_count"] or 0) + 1
            )
            self._diagnostic_stats["read_cache_hit_row_count"] = (
                int(self._diagnostic_stats["read_cache_hit_row_count"] or 0) + row_count
            )
        else:
            self._diagnostic_stats["read_cache_miss_count"] = (
                int(self._diagnostic_stats["read_cache_miss_count"] or 0) + 1
            )
            self._diagnostic_stats["read_cache_miss_row_count"] = (
                int(self._diagnostic_stats["read_cache_miss_row_count"] or 0) + row_count
            )
        self._sync_read_cache_snapshot()
        return result

    def read_tile(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
        *,
        phase: str | None = None,
    ) -> torch.Tensor:
        if column_start < 0 or column_end < column_start or column_end > self.n_feature_columns:
            raise ValueError("requested column slice is out of bounds for file-backed store")
        return self.read_feature_rows(row_start, row_end, phase=phase)[:, column_start:column_end]

    def read_prepared_feature_rows(
        self,
        row_start: int,
        row_end: int,
        *,
        device,
        dtype: torch.dtype,
        phase: str | None = None,
    ) -> torch.Tensor:
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for file-backed store")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("prepared row dtype must be float32 or float64")
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and device_obj.index is None and torch.cuda.is_available():
            device_obj = torch.device("cuda", torch.cuda.current_device())
        cache_key = ("abs_v1", int(row_start), int(row_end), str(device_obj), str(dtype))
        cached = self._prepared_read_cache.get(cache_key)
        cache_hit = cached is not None
        row_count = int(row_end - row_start)
        with self._telemetry_timer(
            name="feature_row_store.read_prepared_rows",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_count,
                "device": str(device_obj),
                "dtype": str(dtype),
                "cache_hit": cache_hit,
            },
        ):
            if cached is not None:
                self._prepared_read_cache.move_to_end(cache_key)
                result = cached
            else:
                prepare_start = time.perf_counter()
                result = self.read_feature_rows(row_start, row_end, phase=phase)
                result = result.to(device=device_obj, dtype=dtype).abs()
                self._apply_row_store_cache_control_after_safe_read(
                    row_start=row_start,
                    row_end=row_end,
                )
                elapsed = (time.perf_counter() - prepare_start) * 1000.0
                self._diagnostic_stats["prepared_read_cache_prepare_elapsed_ms_total"] = float(
                    self._diagnostic_stats["prepared_read_cache_prepare_elapsed_ms_total"] or 0.0
                ) + float(elapsed)
                self._insert_prepared_read_chunk(cache_key, result)
        if cache_hit:
            self._increment_diagnostic_counter("prepared_read_cache_hit_count")
            self._increment_diagnostic_counter("prepared_read_cache_hit_row_count", delta=row_count)
        else:
            self._increment_diagnostic_counter("prepared_read_cache_miss_count")
            self._increment_diagnostic_counter(
                "prepared_read_cache_miss_row_count", delta=row_count
            )
        self._sync_prepared_read_cache_snapshot()
        return result

    def materialize_dense_feature_slice(
        self,
        *,
        row_start: int,
        row_end: int,
        selected_feature_columns: torch.Tensor,
        col_chunk_size: int = 2048,
        phase: str | None = None,
    ) -> torch.Tensor:
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for file-backed store")
        if col_chunk_size <= 0:
            raise ValueError("col_chunk_size must be > 0")

        n_rows = row_end - row_start
        n_cols = selected_feature_columns.numel()
        dense = torch.empty((n_rows, n_cols), dtype=self.row_abs_max.dtype)
        if n_rows == 0 or n_cols == 0:
            return dense

        selected_cols = selected_feature_columns.to(dtype=torch.long, device="cpu")
        if selected_cols.min() < 0 or selected_cols.max() >= self.n_feature_columns:
            raise ValueError("selected feature column indices must be in [0, n_feature_columns)")
        selected_cols_np = selected_cols.numpy()
        same_dtype_fast_path = dense.dtype == self._dtype

        with self._telemetry_timer(
            name="feature_row_store.materialize_dense_slice",
            phase=phase,
            attrs={
                "row_start": row_start,
                "row_end": row_end,
                "row_count": n_rows,
                "selected_columns": n_cols,
                "col_chunk_size": col_chunk_size,
            },
        ):
            rows = self._require_open_rows()
            row_slice = rows[row_start:row_end]
            dense_np = dense.numpy() if same_dtype_fast_path else None
            for col_start in range(0, n_cols, col_chunk_size):
                col_end = min(col_start + col_chunk_size, n_cols)
                cols_np = selected_cols_np[col_start:col_end]
                if same_dtype_fast_path:
                    assert dense_np is not None
                    np.take(
                        row_slice,
                        cols_np,
                        axis=1,
                        out=dense_np[:, col_start:col_end],
                    )
                else:
                    chunk_np = np.asarray(row_slice[:, cols_np], dtype=self._np_dtype)
                    dense[:, col_start:col_end] = torch.from_numpy(chunk_np)
            self._apply_row_store_cache_control_after_safe_read(
                row_start=row_start,
                row_end=row_end,
            )

        self._diagnostic_stats["materialize_call_count"] = (
            int(self._diagnostic_stats["materialize_call_count"] or 0) + 1
        )
        self._diagnostic_stats["materialize_row_count"] = int(
            self._diagnostic_stats["materialize_row_count"] or 0
        ) + int(n_rows)
        self._diagnostic_stats["materialize_column_count"] = int(
            self._diagnostic_stats["materialize_column_count"] or 0
        ) + int(n_cols)
        self._diagnostic_stats["materialize_last_row_start"] = int(row_start)
        self._diagnostic_stats["materialize_last_row_end"] = int(row_end)

        return dense

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return dict(self._diagnostic_stats)

    @property
    def row_denominator_scaled_l1(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.row_abs_max, self.row_l1_scaled)

    @property
    def row_abs_sums(self) -> torch.Tensor:
        """Backward-compatible legacy accessor (non-hot-path only)."""
        return self.row_abs_max * self.row_l1_scaled

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True

        rows = self._rows
        self._rows = None
        if rows is not None:
            try:
                rows.flush()
            except Exception:
                pass

        write_fd = self._write_fd
        self._write_fd = None
        if write_fd is not None:
            try:
                os.close(write_fd)
            except Exception:
                pass

        self._read_chunk_cache.clear()
        self._read_chunk_cache_nbytes = 0
        self._prepared_read_cache.clear()
        self._prepared_read_cache_nbytes = 0
        self._sync_read_cache_snapshot()
        self._sync_prepared_read_cache_snapshot()

        self._tmpdir.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


class _ColumnTiledFeatureRowStore:
    """Exact rows stored as independently auditable column-tile files.

    This Phase-D oracle still accepts full-width rows from ``compute_batch``.  Only
    storage and downstream influence reads are column bounded.
    """

    def __init__(
        self,
        *,
        n_rows: int,
        n_feature_columns: int,
        column_tile_size: int,
        dtype: torch.dtype,
        row_abs_sum_dtype: torch.dtype = torch.float32,
        temp_root_policy: Literal["default", "env_node_local"] = "default",
        temp_root: str | os.PathLike[str] | None = None,
        **_: object,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported feature row store dtype: {dtype}")
        if column_tile_size <= 0:
            raise ValueError("column_tile_size must be > 0")
        self.n_rows = int(n_rows)
        self.n_feature_columns = int(n_feature_columns)
        self.column_tile_size = int(column_tile_size)
        self.row_abs_max = torch.zeros(n_rows, dtype=row_abs_sum_dtype)
        self.row_l1_scaled = torch.zeros(n_rows, dtype=row_abs_sum_dtype)
        self._dtype = dtype
        self._np_dtype = np.float32 if dtype == torch.float32 else np.float64
        _, self._tmpdir = _select_row_store_temp_root(
            temp_root_policy=temp_root_policy, temp_root=temp_root
        )
        self._tiles: list[np.memmap] = []
        self._paths: list[str] = []
        self._closed = False
        self._max_materialized_tile_bytes = 0
        for column_start in range(0, n_feature_columns, column_tile_size):
            width = min(column_tile_size, n_feature_columns - column_start)
            path = os.path.join(self._tmpdir.name, f"columns_{column_start:012d}_{column_start + width:012d}.memmap")
            with open(path, "wb") as handle:
                handle.truncate(n_rows * width * np.dtype(self._np_dtype).itemsize)
            self._paths.append(path)
            self._tiles.append(
                np.memmap(path, mode="r+", dtype=self._np_dtype, shape=(n_rows, width))
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("feature row store has been cleaned up")

    @property
    def nbytes(self) -> int:
        self._require_open()
        return sum(os.path.getsize(path) for path in self._paths)

    @property
    def allocated_file_bytes(self) -> int:
        self._require_open()
        return sum(os.stat(path).st_blocks * 512 for path in self._paths)

    @property
    def row_denominator_scaled_l1(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.row_abs_max, self.row_l1_scaled

    def append_rows(
        self,
        *,
        row_start: int,
        feature_rows: torch.Tensor,
        row_denominator_scaled_l1: tuple[torch.Tensor, torch.Tensor] | None = None,
        full_row_abs_sums: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, float]:
        self._require_open()
        if feature_rows.shape != (feature_rows.shape[0], self.n_feature_columns):
            raise ValueError("feature_rows second dimension must equal configured n_feature_columns")
        row_end = row_start + feature_rows.shape[0]
        if row_start < 0 or row_end > self.n_rows:
            raise ValueError("row range is out of bounds for column-tiled store")
        cpu = feature_rows.detach().to(device="cpu", dtype=self._dtype)
        for tile_index, column_start in enumerate(
            range(0, self.n_feature_columns, self.column_tile_size)
        ):
            column_end = min(column_start + self.column_tile_size, self.n_feature_columns)
            tile = cpu[:, column_start:column_end].contiguous()
            self._max_materialized_tile_bytes = max(
                self._max_materialized_tile_bytes, tile.numel() * tile.element_size()
            )
            self._tiles[tile_index][row_start:row_end] = tile.numpy()
        if row_denominator_scaled_l1 is None:
            if full_row_abs_sums is None:
                raise ValueError("row denominator data must be provided")
            row_denominator_scaled_l1 = _row_abs_sums_to_scaled_l1(
                full_row_abs_sums, dtype=self.row_abs_max.dtype
            )
        self.row_abs_max[row_start:row_end] = row_denominator_scaled_l1[0].to("cpu")
        self.row_l1_scaled[row_start:row_end] = row_denominator_scaled_l1[1].to("cpu")
        return {}

    def read_tile(
        self, row_start: int, row_end: int, column_start: int, column_end: int, **_: object
    ) -> torch.Tensor:
        self._require_open()
        if row_start < 0 or row_end < row_start or row_end > self.n_rows:
            raise ValueError("requested row slice is out of bounds for column-tiled store")
        if column_start < 0 or column_end < column_start or column_end > self.n_feature_columns:
            raise ValueError("requested column slice is out of bounds for column-tiled store")
        if column_start == column_end:
            return torch.empty((row_end - row_start, 0), dtype=self._dtype)
        first = column_start // self.column_tile_size
        last = (column_end - 1) // self.column_tile_size
        result = torch.empty((row_end - row_start, column_end - column_start), dtype=self._dtype)
        for tile_index in range(first, last + 1):
            tile_start = tile_index * self.column_tile_size
            local_start = max(column_start, tile_start) - tile_start
            local_end = min(column_end, tile_start + self._tiles[tile_index].shape[1]) - tile_start
            output_start = max(column_start, tile_start) - column_start
            output_end = output_start + (local_end - local_start)
            source = torch.from_numpy(
                np.asarray(self._tiles[tile_index][row_start:row_end, local_start:local_end])
            )
            result[:, output_start:output_end].copy_(source)
        self._max_materialized_tile_bytes = max(
            self._max_materialized_tile_bytes, result.numel() * result.element_size()
        )
        return result

    def read_feature_rows(self, row_start: int, row_end: int, **kwargs) -> torch.Tensor:
        return self.read_tile(row_start, row_end, 0, self.n_feature_columns, **kwargs)

    def materialize_dense_feature_slice(
        self, *, row_start: int, row_end: int, selected_feature_columns: torch.Tensor, **_: object
    ) -> torch.Tensor:
        selected = selected_feature_columns.to(device="cpu", dtype=torch.long)
        result = torch.empty((row_end - row_start, selected.numel()), dtype=self._dtype)
        for output_column, source_column in enumerate(selected.tolist()):
            if source_column < 0 or source_column >= self.n_feature_columns:
                raise ValueError("selected feature column indices must be in [0, n_feature_columns)")
            result[:, output_column : output_column + 1] = self.read_tile(
                row_start, row_end, source_column, source_column + 1
            )
        return result

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "backend": "column_tiled_full_retention_v1",
            "full_width_production": True,
            "column_tile_size": self.column_tile_size,
            "tile_file_count": len(self._paths),
            "apparent_file_bytes": self.nbytes,
            "allocated_file_bytes": self.allocated_file_bytes,
            "maximum_materialized_tile_bytes": self._max_materialized_tile_bytes,
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        for tile in self._tiles:
            tile.flush()
        self._tiles.clear()
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass



def _resolve_row_store_cache_control(
    row_store_cache_control: str,
) -> _RowStoreCacheControlMode:
    normalized = str(row_store_cache_control).strip().lower()
    allowed_values = set(_ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE)
    if normalized not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"row_store_cache_control must be one of: {allowed} (got {row_store_cache_control!r})"
        )
    return cast(_RowStoreCacheControlMode, normalized)


def _resolve_row_store_cache_control_config(
    row_store_cache_control: str,
    *,
    compact_output: bool,
    supports_compact_row_store: bool,
) -> _RowStoreCacheControlConfig:
    requested_mode = _resolve_row_store_cache_control(row_store_cache_control)
    mode_applicable = bool(compact_output and supports_compact_row_store)
    fallback_reason: str | None = None
    if requested_mode != "off" and not mode_applicable:
        effective_mode = cast(_RowStoreCacheControlMode, "off")
        fallback_reason = (
            f"{requested_mode} requires compact_output=True and "
            "compact row-store provider support; falling back to off execution"
        )
    else:
        effective_mode = cast(
            _RowStoreCacheControlMode,
            _ROW_STORE_CACHE_CONTROL_EFFECTIVE_MODE_BY_MODE[requested_mode],
        )
    effective_behavior: Literal["requested", "off_reference_execution"] = (
        "requested" if requested_mode == effective_mode else "off_reference_execution"
    )
    return _RowStoreCacheControlConfig(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        default_mode=_ROW_STORE_CACHE_CONTROL_DEFAULT,
        mode_applicable=mode_applicable,
        effective_behavior=effective_behavior,
        fallback_reason=fallback_reason,
    )


def _build_row_store_cache_control_metadata(
    row_store_cache_control_config: _RowStoreCacheControlConfig,
) -> dict[str, object]:
    return {
        "row_store_cache_control_requested": row_store_cache_control_config.requested_mode,
        "row_store_cache_control": row_store_cache_control_config.requested_mode,
        "row_store_cache_control_default": row_store_cache_control_config.default_mode,
        "row_store_cache_control_effective": row_store_cache_control_config.effective_mode,
        "row_store_cache_control_applicable": bool(row_store_cache_control_config.mode_applicable),
        "row_store_cache_control_effective_behavior": (
            row_store_cache_control_config.effective_behavior
        ),
        "row_store_cache_control_fallback_reason": row_store_cache_control_config.fallback_reason,
        "row_store_cache_control_reference_execution": bool(
            row_store_cache_control_config.requested_mode
            != row_store_cache_control_config.effective_mode
        ),
    }


def _resolve_row_store_temp_root_policy(
    temp_root_policy: str,
) -> Literal["default", "env_node_local"]:
    normalized = str(temp_root_policy).strip().lower()
    if normalized not in _ROW_STORE_TEMP_ROOT_POLICY_BY_NAME:
        allowed = ", ".join(sorted(_ROW_STORE_TEMP_ROOT_POLICY_BY_NAME))
        raise ValueError(
            f"row_store_temp_root_policy must be one of: {allowed} (got {temp_root_policy!r})"
        )
    return cast(
        Literal["default", "env_node_local"],
        _ROW_STORE_TEMP_ROOT_POLICY_BY_NAME[normalized],
    )


def _is_existing_writable_dir(path: str | os.PathLike[str] | None) -> bool:
    if path is None:
        return False
    try:
        return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def _select_row_store_temp_root(
    *,
    temp_root_policy: str,
    temp_root: str | os.PathLike[str] | None,
) -> tuple[_RowStoreTempRootSelection, tempfile.TemporaryDirectory[str]]:
    policy = _resolve_row_store_temp_root_policy(temp_root_policy)
    requested_root = os.fspath(temp_root) if temp_root is not None else None
    fallback_reason: str | None = None

    if requested_root is not None:
        if _is_existing_writable_dir(requested_root):
            selected_root = requested_root
        else:
            selected_root = None
            fallback_reason = (
                f"requested temp_root is not an existing writable directory: {requested_root}"
            )
    elif policy == _ROW_STORE_TEMP_ROOT_POLICY_ENV_NODE_LOCAL:
        selected_root = None
        rejected: list[str] = []
        for label, candidate in (
            ("SLURM_TMPDIR", os.environ.get("SLURM_TMPDIR")),
            ("TMPDIR", os.environ.get("TMPDIR")),
            ("/tmp", "/tmp"),
        ):
            if _is_existing_writable_dir(candidate):
                selected_root = os.fspath(candidate)
                break
            if candidate:
                rejected.append(f"{label}={candidate!r}")
        if selected_root is None:
            fallback_reason = "no env_node_local candidate was an existing writable directory"
            if rejected:
                fallback_reason += f" (rejected: {', '.join(rejected)})"
    else:
        selected_root = None

    tmpdir = tempfile.TemporaryDirectory(prefix="ct_feature_rows_", dir=selected_root)
    return _RowStoreTempRootSelection(
        policy=policy,
        requested_root=requested_root,
        selected_root=selected_root,
        selected_path=tmpdir.name,
        fallback_reason=fallback_reason,
    ), tmpdir
