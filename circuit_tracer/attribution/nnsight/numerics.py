"""Leaf numerical helpers for NNSight attribution."""

import torch


_EXACT_TRACE_INTERNAL_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
    "torch.float64": torch.float64,
}


def _resolve_exact_trace_internal_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value in (torch.float32, torch.float64):
            return value
        raise ValueError(
            f"exact_trace_internal_dtype must be one of: fp32, fp64 (got dtype={value})"
        )

    normalized = str(value).strip().lower()
    resolved = _EXACT_TRACE_INTERNAL_DTYPE_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_EXACT_TRACE_INTERNAL_DTYPE_BY_NAME))
        raise ValueError(f"exact_trace_internal_dtype must be one of: {allowed} (got {value!r})")
    return resolved


def _exact_trace_internal_dtype_name(dtype: torch.dtype) -> str:
    resolved = _resolve_exact_trace_internal_dtype(dtype)
    return "fp32" if resolved == torch.float32 else "fp64"


def _row_abs_sums_to_scaled_l1(
    row_abs_sums: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert materialized row-L1 sums to the scaled denominator representation."""

    resolved_dtype = _resolve_exact_trace_internal_dtype(dtype)
    row_abs_sums_cpu = row_abs_sums.detach()
    if row_abs_sums_cpu.ndim != 1:
        raise ValueError("row_abs_sums must be rank-1")
    if row_abs_sums_cpu.device.type != "cpu" or row_abs_sums_cpu.dtype != torch.float64:
        row_abs_sums_cpu = row_abs_sums_cpu.to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(row_abs_sums_cpu).all().item():
        raise ValueError("row_abs_sums must be finite")
    if bool((row_abs_sums_cpu < 0).any().item()):
        raise ValueError("row_abs_sums must be non-negative")

    max_for_dtype = torch.full_like(row_abs_sums_cpu, torch.finfo(resolved_dtype).max)
    row_abs_max_f64 = torch.minimum(row_abs_sums_cpu, max_for_dtype)
    row_l1_scaled_f64 = torch.zeros_like(row_abs_sums_cpu)
    positive_rows = row_abs_sums_cpu > 0
    if bool(positive_rows.any().item()):
        row_l1_scaled_f64[positive_rows] = (
            row_abs_sums_cpu[positive_rows] / row_abs_max_f64[positive_rows]
        )
    row_abs_max = row_abs_max_f64.to(dtype=resolved_dtype).contiguous()
    row_l1_scaled = row_l1_scaled_f64.to(dtype=resolved_dtype).contiguous()
    return row_abs_max, row_l1_scaled
