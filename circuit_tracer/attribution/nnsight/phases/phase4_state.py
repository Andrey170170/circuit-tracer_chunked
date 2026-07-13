"""Shared identity for Phase 4 feature-attribution operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureAttributionRun:
    """Feature-attribution run advanced by coherent domain operations."""

    inputs: Any
    config: Any
