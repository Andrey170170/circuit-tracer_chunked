"""Bounded ownership for deferred Phase-4 feature VJP contraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class FeatureVjpTapeByteEstimate:
    """Conservative simultaneous ownership for one captured batch."""

    host_nbytes: int
    device_nbytes: int
    row_nbytes: int
    total_nbytes: int

    def __post_init__(self) -> None:
        parts = (self.host_nbytes, self.device_nbytes, self.row_nbytes)
        if any(value < 0 for value in parts):
            raise ValueError("FeatureVjpTape byte estimates must be >= 0")
        if self.total_nbytes != sum(parts):
            raise ValueError("FeatureVjpTape total_nbytes must equal all owned byte tiers")


@dataclass(frozen=True)
class FeatureVjpTapeEntry:
    """One immutable execution batch captured inside a semantic frontier."""

    batch_call_index: int
    gradients: tuple[torch.Tensor | None, ...]
    row_buffer: torch.Tensor
    batch_size: int
    host_nbytes: int
    device_nbytes: int
    row_nbytes: int
    total_nbytes: int
    pinned_host_nbytes: int
    pageable_host_nbytes: int
    pin_fallback_count: int = 0
    pin_fallback_reason: str | None = None

    def clear(self) -> None:
        """Release tensor storage deterministically after replay."""
        object.__setattr__(self, "gradients", ())
        object.__setattr__(self, "row_buffer", torch.empty(0))


class FeatureVjpTape:
    """Byte- and batch-bounded FIFO that never owns multiple frontiers."""

    def __init__(self, *, max_batches: int, max_bytes: int) -> None:
        if max_batches < 1:
            raise ValueError("FeatureVjpTape max_batches must be >= 1")
        if max_bytes < 0:
            raise ValueError("FeatureVjpTape max_bytes must be >= 0")
        if max_batches > 1 and max_bytes == 0:
            raise ValueError("FeatureVjpTape window > 1 requires a positive byte cap")
        self.max_batches = int(max_batches)
        self.max_bytes = int(max_bytes)
        self._entries: list[FeatureVjpTapeEntry] = []
        self.current_bytes = 0
        self.high_watermark_bytes = 0
        self.current_host_nbytes = 0
        self.current_device_nbytes = 0
        self.current_row_nbytes = 0
        self.current_pinned_host_nbytes = 0
        self.current_pageable_host_nbytes = 0
        self.high_watermark_host_nbytes = 0
        self.high_watermark_device_nbytes = 0
        self.high_watermark_row_nbytes = 0
        self.high_watermark_pinned_host_nbytes = 0
        self.high_watermark_pageable_host_nbytes = 0

    @property
    def entries(self) -> tuple[FeatureVjpTapeEntry, ...]:
        return tuple(self._entries)

    @property
    def batch_count(self) -> int:
        return len(self._entries)

    def can_accept(self, total_nbytes: int) -> bool:
        if total_nbytes < 0:
            raise ValueError("FeatureVjpTape entry size must be >= 0")
        return (
            self.batch_count < self.max_batches
            and (
                self.max_bytes == 0
                or self.current_bytes + total_nbytes <= self.max_bytes
            )
        )

    def append(self, entry: FeatureVjpTapeEntry) -> None:
        if not self.can_accept(entry.total_nbytes):
            raise BufferError(
                "FeatureVjpTape capacity exceeded "
                f"(batches={self.batch_count}/{self.max_batches}, "
                f"bytes={self.current_bytes}+{entry.total_nbytes}/{self.max_bytes})"
            )
        self._entries.append(entry)
        self.current_bytes += int(entry.total_nbytes)
        self.high_watermark_bytes = max(self.high_watermark_bytes, self.current_bytes)
        for tier in ("host", "device", "row", "pinned_host", "pageable_host"):
            current_name = f"current_{tier}_nbytes"
            high_name = f"high_watermark_{tier}_nbytes"
            current = int(getattr(self, current_name)) + int(
                getattr(entry, f"{tier}_nbytes")
            )
            setattr(self, current_name, current)
            setattr(self, high_name, max(int(getattr(self, high_name)), current))

    def clear(self) -> None:
        for entry in self._entries:
            entry.clear()
        self._entries.clear()
        self.current_bytes = 0
        self.current_host_nbytes = 0
        self.current_device_nbytes = 0
        self.current_row_nbytes = 0
        self.current_pinned_host_nbytes = 0
        self.current_pageable_host_nbytes = 0
