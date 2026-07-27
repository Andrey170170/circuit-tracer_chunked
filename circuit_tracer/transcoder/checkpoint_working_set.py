"""Byte-budgeted checkpoint working-set admission and transition policy."""

from __future__ import annotations

from dataclasses import dataclass

from circuit_tracer.transcoder.checkpoint_assets import (
    CheckpointManifest,
    CheckpointRange,
)


@dataclass(frozen=True, slots=True)
class ProviderCheckpointLifecycle:
    """Provider-owned immutable checkpoint manifest and reviewed policy knobs."""

    manifest: CheckpointManifest
    prefault_budget_bytes: int = 0

    def __post_init__(self) -> None:
        if self.prefault_budget_bytes < 0:
            raise ValueError("prefault_budget_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class PhaseWorkingSetPlan:
    """Admitted byte ranges for one synchronous phase transition."""

    retain_bytes: int
    release: tuple[CheckpointRange, ...]
    prefault: tuple[CheckpointRange, ...]
    byte_budget: int
    available_headroom_bytes: int | None
    prefault_requested_bytes: int
    prefault_admitted_bytes: int
    prefault_refused_bytes: int
    fallback_reason: str | None = None

    @classmethod
    def admit(
        cls,
        manifest: CheckpointManifest,
        *,
        retain_bytes: int,
        release_roles: tuple[str, ...] = ("decoder",),
        prefault_roles: tuple[str, ...] = ("encoder", "refresh", "model_forward"),
        byte_budget: int = 0,
        available_headroom_bytes: int | None = None,
    ) -> "PhaseWorkingSetPlan":
        """Admit whole immutable ranges in role order without exceeding bytes."""

        retain_bytes = int(retain_bytes)
        byte_budget = int(byte_budget)
        if retain_bytes < 0 or byte_budget < 0:
            raise ValueError("working-set byte counts must be non-negative")
        if available_headroom_bytes is not None and available_headroom_bytes < 0:
            raise ValueError("available_headroom_bytes must be non-negative")

        release_role_set = frozenset(release_roles)
        release = tuple(
            byte_range
            for asset in manifest.assets
            for byte_range in asset.ranges
            if byte_range.role in release_role_set
        )
        candidates = tuple(
            byte_range
            for role in prefault_roles
            for asset in manifest.assets
            for byte_range in asset.ranges
            if byte_range.role == role
        )
        requested = sum(item.length for item in candidates)
        headroom_budget = (
            byte_budget
            if available_headroom_bytes is None
            else min(byte_budget, int(available_headroom_bytes))
        )
        remaining = max(0, headroom_budget - retain_bytes)
        admitted: list[CheckpointRange] = []
        for item in candidates:
            if item.length <= remaining:
                admitted.append(item)
                remaining -= item.length
        admitted_bytes = sum(item.length for item in admitted)
        fallback_reason = None
        if candidates and not admitted:
            fallback_reason = (
                "prefault_budget_nonpositive"
                if headroom_budget <= retain_bytes
                else "no_whole_prefault_range_fits"
            )
        elif admitted_bytes < requested:
            fallback_reason = "prefault_partially_admitted"
        return cls(
            retain_bytes=retain_bytes,
            release=release,
            prefault=tuple(admitted),
            byte_budget=byte_budget,
            available_headroom_bytes=available_headroom_bytes,
            prefault_requested_bytes=requested,
            prefault_admitted_bytes=admitted_bytes,
            prefault_refused_bytes=requested - admitted_bytes,
            fallback_reason=fallback_reason,
        )

