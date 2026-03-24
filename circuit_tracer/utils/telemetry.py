from __future__ import annotations

from collections.abc import Mapping

import torch

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]


def _format_optional_gib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} GiB"


def get_memory_snapshot(device: torch.device | None = None) -> dict[str, float | None]:
    rss_gib = None
    if resource is not None:
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    snapshot: dict[str, float | None] = {
        "rss_gib": rss_gib,
        "cuda_allocated_gib": None,
        "cuda_reserved_gib": None,
        "cuda_max_allocated_gib": None,
        "cuda_max_reserved_gib": None,
    }

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                "cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "cuda_max_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
        )

    return snapshot


def format_memory_snapshot(
    device: torch.device | None = None, extra: Mapping[str, object] | None = None
) -> str:
    snapshot = get_memory_snapshot(device)
    parts = [
        f"rss={_format_optional_gib(snapshot['rss_gib'])}",
        f"cuda_alloc={_format_optional_gib(snapshot['cuda_allocated_gib'])}",
        f"cuda_reserved={_format_optional_gib(snapshot['cuda_reserved_gib'])}",
        f"cuda_peak_alloc={_format_optional_gib(snapshot['cuda_max_allocated_gib'])}",
        f"cuda_peak_reserved={_format_optional_gib(snapshot['cuda_max_reserved_gib'])}",
    ]
    if extra:
        parts.extend(f"{key}={value}" for key, value in extra.items())
    return ", ".join(parts)
