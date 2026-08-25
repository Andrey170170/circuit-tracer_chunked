"""Exact, bounded-memory evidence for canonical row denominators."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


ROW_DENOMINATOR_POLICY_ID = "canonical_scaled_l1_row_evidence_sha256_v2"
_AUDIT_ATTRIBUTE = "_correctness_row_denominator_audit"


class RowDenominatorAudit:
    """Retain exact digests of row-derived denominator batches, never row matrices."""

    def __init__(self) -> None:
        self._row_abs_max_raw_sha256 = hashlib.sha256()
        self._row_l1_scaled_raw_sha256 = hashlib.sha256()
        self._row_abs_max_dtype: str | None = None
        self._row_l1_scaled_dtype: str | None = None
        self._batch_count = 0
        self._next_row = 0

    @property
    def batch_count(self) -> int:
        return self._batch_count

    @property
    def rows_recorded(self) -> int:
        return self._next_row

    def record(
        self,
        *,
        row_start: int,
        denominator: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Record a canonical denominator produced directly from effective row values."""

        row_abs_max, row_l1_scaled = denominator
        if row_abs_max.ndim != 1 or row_l1_scaled.ndim != 1:
            raise ValueError("authoritative row denominators must be vectors")
        row_count = int(row_abs_max.numel())
        if row_count <= 0 or int(row_l1_scaled.numel()) != row_count:
            raise ValueError("authoritative row denominator vectors must align and be nonempty")
        if row_start != self._next_row:
            raise ValueError(
                "authoritative row denominator coverage must be contiguous: "
                f"expected row {self._next_row}, received {row_start}"
            )
        abs_max_dtype = str(row_abs_max.dtype)
        scaled_dtype = str(row_l1_scaled.dtype)
        if self._row_abs_max_dtype not in (None, abs_max_dtype):
            raise ValueError("authoritative row-maximum dtype changed across batches")
        if self._row_l1_scaled_dtype not in (None, scaled_dtype):
            raise ValueError("authoritative scaled-L1 dtype changed across batches")
        self._row_abs_max_dtype = abs_max_dtype
        self._row_l1_scaled_dtype = scaled_dtype
        self._row_abs_max_raw_sha256.update(_exact_tensor_bytes(row_abs_max))
        self._row_l1_scaled_raw_sha256.update(_exact_tensor_bytes(row_l1_scaled))
        self._batch_count += 1
        self._next_row += row_count

    def verify(
        self,
        *,
        row_abs_max: torch.Tensor,
        row_l1_scaled: torch.Tensor,
        expected_rows: int,
    ) -> dict[str, object]:
        """Compare stored bytes to authoritative digests and check scalar invariants."""

        if expected_rows < 0:
            raise ValueError("expected_rows must be non-negative")
        if row_abs_max.ndim != 1 or row_l1_scaled.ndim != 1:
            return _incomplete_evidence(batch_count=self.batch_count)

        available_rows = min(
            expected_rows,
            int(row_abs_max.numel()),
            int(row_l1_scaled.numel()),
        )
        can_compare = self.rows_recorded <= expected_rows and available_rows >= self.rows_recorded
        rows_checked = self.rows_recorded if can_compare else 0
        exact_match = False
        if can_compare and self._row_abs_max_dtype is not None and self._row_l1_scaled_dtype is not None:
            exact_match = (
                _exact_tensor_sha256(row_abs_max[:rows_checked])
                == _final_authoritative_sha256(
                    raw_sha256=self._row_abs_max_raw_sha256,
                    dtype=self._row_abs_max_dtype,
                    row_count=rows_checked,
                )
                and _exact_tensor_sha256(row_l1_scaled[:rows_checked])
                == _final_authoritative_sha256(
                    raw_sha256=self._row_l1_scaled_raw_sha256,
                    dtype=self._row_l1_scaled_dtype,
                    row_count=rows_checked,
                )
            )

        violation_count = rows_checked if rows_checked and not exact_match else 0
        if rows_checked and exact_match:
            maxima = row_abs_max[:rows_checked]
            scaled_l1 = row_l1_scaled[:rows_checked]
            violation_count = int((
                ~torch.isfinite(maxima)
                | ~torch.isfinite(scaled_l1)
                | (maxima < 0)
                | (scaled_l1 < 0)
                | ((maxima == 0) != (scaled_l1 == 0))
            ).sum().item())

        complete = (
            available_rows == expected_rows
            and self.rows_recorded == expected_rows
            and rows_checked == expected_rows
            and self.batch_count > 0
        )
        return {
            "complete": complete,
            "policy_id": ROW_DENOMINATOR_POLICY_ID,
            "rows_checked": rows_checked,
            "violation_count": violation_count,
            "authoritative_batch_count": self.batch_count,
        }


def enable_row_denominator_audit(store: Any | None) -> None:
    """Enable row-derived evidence on the feature-row owner for this trace."""

    if store is None:
        return
    if getattr(store, _AUDIT_ATTRIBUTE, None) is None:
        setattr(store, _AUDIT_ATTRIBUTE, RowDenominatorAudit())


def record_authoritative_row_denominator(
    store: Any | None,
    *,
    row_start: int,
    denominator: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Record only when correctness evidence was explicitly enabled in Phase 3."""

    audit = getattr(store, _AUDIT_ATTRIBUTE, None)
    if isinstance(audit, RowDenominatorAudit):
        stored_abs_max = getattr(store, "row_abs_max", None)
        stored_l1_scaled = getattr(store, "row_l1_scaled", None)
        if not isinstance(stored_abs_max, torch.Tensor) or not isinstance(
            stored_l1_scaled, torch.Tensor
        ):
            raise TypeError("row-denominator audit requires canonical stored vectors")
        canonical = (
            denominator[0].detach().to(device="cpu", dtype=stored_abs_max.dtype),
            denominator[1].detach().to(device="cpu", dtype=stored_l1_scaled.dtype),
        )
        audit.record(row_start=row_start, denominator=canonical)


def verify_row_denominator_evidence(
    store: Any | None,
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Return fail-closed evidence from the enabled audit and stored vectors."""

    audit = getattr(store, _AUDIT_ATTRIBUTE, None)
    row_abs_max = getattr(store, "row_abs_max", None)
    row_l1_scaled = getattr(store, "row_l1_scaled", None)
    if not isinstance(audit, RowDenominatorAudit):
        return _incomplete_evidence(batch_count=0)
    if not isinstance(row_abs_max, torch.Tensor) or not isinstance(
        row_l1_scaled, torch.Tensor
    ):
        return _incomplete_evidence(batch_count=audit.batch_count)
    return audit.verify(
        row_abs_max=row_abs_max,
        row_l1_scaled=row_l1_scaled,
        expected_rows=expected_rows,
    )


def _incomplete_evidence(*, batch_count: int) -> dict[str, object]:
    return {
        "complete": False,
        "policy_id": ROW_DENOMINATOR_POLICY_ID,
        "rows_checked": 0,
        "violation_count": 0,
        "authoritative_batch_count": batch_count,
    }


def _exact_tensor_sha256(value: torch.Tensor) -> str:
    raw = hashlib.sha256(_exact_tensor_bytes(value))
    return _final_authoritative_sha256(
        raw_sha256=raw,
        dtype=str(value.dtype),
        row_count=int(value.numel()),
    )


def _exact_tensor_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _final_authoritative_sha256(
    *,
    raw_sha256: Any,
    dtype: str,
    row_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(str((row_count,)).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw_sha256.digest())
    return digest.hexdigest()
