from __future__ import annotations

import os
import stat as stat_module
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Literal


class CheckpointAssetScope(str, Enum):
    """Filesystem ownership scope used to authorize page-cache advice."""

    JOB_PRIVATE = "job_private"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"

    @property
    def advice_eligible(self) -> bool:
        return self in {self.JOB_PRIVATE, self.EXCLUSIVE}


class CheckpointPageAdvice(str, Enum):
    PREFAULT = "prefault"
    RELEASE = "release"


CheckpointPageOutcome = Literal["issued", "refused", "unavailable", "error"]
PosixFadvise = Callable[[int, int, int, int], object]


@dataclass(frozen=True, slots=True)
class CheckpointRange:
    """One exact half-open byte range owned by a checkpoint asset and role."""

    asset_id: str
    role: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("checkpoint range asset_id must be non-empty")
        if not self.role:
            raise ValueError("checkpoint range role must be non-empty")
        if self.offset < 0:
            raise ValueError("checkpoint range offset must be non-negative")
        if self.length <= 0:
            raise ValueError(
                "checkpoint range length must be positive; zero-length file-wide advice is forbidden"
            )

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True, slots=True)
class CheckpointAsset:
    """Immutable file identity and the exact checkpoint ranges it owns."""

    asset_id: str
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    scope: CheckpointAssetScope
    ranges: tuple[CheckpointRange, ...]

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("checkpoint asset_id must be non-empty")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "scope", CheckpointAssetScope(self.scope))
        object.__setattr__(self, "ranges", tuple(self.ranges))
        if self.device < 0 or self.inode <= 0:
            raise ValueError("checkpoint asset device and inode must identify a real file")
        if self.size < 0:
            raise ValueError("checkpoint asset size must be non-negative")
        if not self.ranges:
            raise ValueError("checkpoint asset must own at least one byte range")

        ordered = sorted(self.ranges, key=lambda item: (item.offset, item.end))
        previous: CheckpointRange | None = None
        for item in ordered:
            if item.asset_id != self.asset_id:
                raise ValueError(
                    f"checkpoint range belongs to {item.asset_id!r}, not {self.asset_id!r}"
                )
            if item.end > self.size:
                raise ValueError(
                    f"checkpoint range {item.offset}:{item.end} exceeds file size {self.size}"
                )
            if previous is not None and item.offset < previous.end:
                raise ValueError(
                    "checkpoint ranges must not overlap: "
                    f"{previous.offset}:{previous.end} and {item.offset}:{item.end}"
                )
            previous = item

    @classmethod
    def from_path(
        cls,
        *,
        asset_id: str,
        path: str | os.PathLike[str],
        scope: CheckpointAssetScope,
        ranges: tuple[CheckpointRange, ...],
    ) -> CheckpointAsset:
        normalized_path = Path(path)
        fd = os.open(normalized_path, os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            if not stat_module.S_ISREG(stat.st_mode):
                raise ValueError(f"checkpoint asset is not a regular file: {normalized_path}")
        finally:
            os.close(fd)
        return cls(
            asset_id=asset_id,
            path=normalized_path,
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
            scope=scope,
            ranges=ranges,
        )

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(item.role for item in self.ranges)

    @property
    def has_mixed_roles(self) -> bool:
        return len(self.roles) > 1


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Validated checkpoint asset set keyed by stable logical asset id."""

    assets: tuple[CheckpointAsset, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", tuple(self.assets))
        asset_ids: set[str] = set()
        file_ids: set[tuple[int, int]] = set()
        for asset in self.assets:
            if asset.asset_id in asset_ids:
                raise ValueError(f"duplicate checkpoint asset_id: {asset.asset_id!r}")
            file_id = (asset.device, asset.inode)
            if file_id in file_ids:
                raise ValueError(
                    "one checkpoint file identity must have exactly one manifest owner: "
                    f"device={asset.device}, inode={asset.inode}"
                )
            asset_ids.add(asset.asset_id)
            file_ids.add(file_id)

    def asset(self, asset_id: str) -> CheckpointAsset:
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"unknown checkpoint asset: {asset_id!r}")

    def owning_asset(self, byte_range: CheckpointRange) -> CheckpointAsset:
        asset = self.asset(byte_range.asset_id)
        if byte_range not in asset.ranges:
            raise ValueError(
                "checkpoint range is not an exact member of its asset manifest; "
                "partial, enlarged, and forged ranges are forbidden"
            )
        return asset


@dataclass(frozen=True, slots=True)
class CheckpointPageTelemetry:
    advice: CheckpointPageAdvice
    outcome: CheckpointPageOutcome
    asset_id: str
    path: str
    device: int
    inode: int
    role: str
    offset: int
    length: int
    supported: bool
    effective: bool
    refused: bool
    attempted: bool
    issued: bool = False
    idempotent: bool = False
    reason: str | None = None
    error: str | None = None


class _Auto(Enum):
    VALUE = "auto"


_AUTO = _Auto.VALUE


class CheckpointPageLifecycle:
    """Apply safe, exact-range page-cache advice to private checkpoint files.

    Failures, unavailable platform support, stale file identities, and
    ineligible sharing scopes are observable no-ops. Advice is idempotent for a
    lifecycle instance so repeated phase cleanup cannot issue duplicate calls.
    """

    def __init__(
        self,
        manifest: CheckpointManifest,
        *,
        telemetry: Callable[[CheckpointPageTelemetry], None] | None = None,
        posix_fadvise: PosixFadvise | None | _Auto = _AUTO,
        prefault_flag: int | None | _Auto = _AUTO,
        release_flag: int | None | _Auto = _AUTO,
    ) -> None:
        self._manifest = manifest
        self._telemetry = telemetry
        resolved_fadvise = (
            getattr(os, "posix_fadvise", None)
            if posix_fadvise is _AUTO
            else posix_fadvise
        )
        self._posix_fadvise: PosixFadvise | None = (
            resolved_fadvise if callable(resolved_fadvise) else None
        )
        raw_flags = {
            CheckpointPageAdvice.PREFAULT: (
                getattr(os, "POSIX_FADV_WILLNEED", None)
                if prefault_flag is _AUTO
                else prefault_flag
            ),
            CheckpointPageAdvice.RELEASE: (
                getattr(os, "POSIX_FADV_DONTNEED", None)
                if release_flag is _AUTO
                else release_flag
            ),
        }
        self._flags: dict[CheckpointPageAdvice, int | None] = {
            advice: flag if isinstance(flag, int) else None
            for advice, flag in raw_flags.items()
        }
        self._completed: dict[
            tuple[CheckpointPageAdvice, CheckpointRange], CheckpointPageTelemetry
        ] = {}
        self._lock = Lock()

    def prefault(self, byte_range: CheckpointRange) -> CheckpointPageTelemetry:
        return self.advise(byte_range, CheckpointPageAdvice.PREFAULT)

    def release(self, byte_range: CheckpointRange) -> CheckpointPageTelemetry:
        return self.advise(byte_range, CheckpointPageAdvice.RELEASE)

    def advise(
        self,
        byte_range: CheckpointRange,
        advice: CheckpointPageAdvice,
    ) -> CheckpointPageTelemetry:
        advice = CheckpointPageAdvice(advice)
        key = (advice, byte_range)
        with self._lock:
            previous = self._completed.get(key)
            if previous is not None:
                result = replace(previous, attempted=False, idempotent=True)
                self._emit(result)
                return result

            result = self._advise_once(byte_range, advice)
            self._completed[key] = result
            self._emit(result)
            return result

    def _advise_once(
        self,
        byte_range: CheckpointRange,
        advice: CheckpointPageAdvice,
    ) -> CheckpointPageTelemetry:
        try:
            asset = self._manifest.owning_asset(byte_range)
        except (KeyError, ValueError) as exc:
            return self._result(
                byte_range,
                advice,
                asset=None,
                outcome="refused",
                supported=self._platform_supported(advice),
                refused=True,
                reason="range_not_manifest_owned",
                error=f"{type(exc).__name__}: {exc}",
            )

        supported = self._platform_supported(advice)
        if not asset.scope.advice_eligible:
            return self._result(
                byte_range,
                advice,
                asset=asset,
                outcome="refused",
                supported=supported,
                refused=True,
                reason=f"scope_{asset.scope.value}_is_not_advice_eligible",
            )
        if not supported:
            return self._result(
                byte_range,
                advice,
                asset=asset,
                outcome="unavailable",
                supported=False,
                reason="posix_fadvise_or_advice_flag_unavailable",
            )

        fd: int | None = None
        try:
            fd = os.open(asset.path, os.O_RDONLY)
            stat = os.fstat(fd)
            if (int(stat.st_dev), int(stat.st_ino)) != (asset.device, asset.inode):
                return self._result(
                    byte_range,
                    advice,
                    asset=asset,
                    outcome="refused",
                    supported=True,
                    refused=True,
                    attempted=False,
                    reason="open_file_identity_does_not_match_manifest",
                )
            if (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            ) != (
                asset.size,
                asset.mtime_ns,
                asset.ctime_ns,
            ):
                return self._result(
                    byte_range,
                    advice,
                    asset=asset,
                    outcome="refused",
                    supported=True,
                    refused=True,
                    attempted=False,
                    reason="open_file_metadata_does_not_match_manifest",
                )
            fadvise = self._posix_fadvise
            flag = self._flags[advice]
            assert fadvise is not None and flag is not None
            fadvise(
                fd,
                int(byte_range.offset),
                int(byte_range.length),
                flag,
            )
        except Exception as exc:
            return self._result(
                byte_range,
                advice,
                asset=asset,
                outcome="error",
                supported=True,
                attempted=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

        return self._result(
            byte_range,
            advice,
            asset=asset,
            outcome="issued",
            supported=True,
            attempted=True,
            issued=True,
        )

    def _platform_supported(self, advice: CheckpointPageAdvice) -> bool:
        return callable(self._posix_fadvise) and self._flags[advice] is not None

    def _result(
        self,
        byte_range: CheckpointRange,
        advice: CheckpointPageAdvice,
        *,
        asset: CheckpointAsset | None,
        outcome: CheckpointPageOutcome,
        supported: bool,
        effective: bool = False,
        refused: bool = False,
        attempted: bool = False,
        issued: bool = False,
        reason: str | None = None,
        error: str | None = None,
    ) -> CheckpointPageTelemetry:
        return CheckpointPageTelemetry(
            advice=advice,
            outcome=outcome,
            asset_id=byte_range.asset_id,
            path="" if asset is None else str(asset.path),
            device=-1 if asset is None else asset.device,
            inode=-1 if asset is None else asset.inode,
            role=byte_range.role,
            offset=byte_range.offset,
            length=byte_range.length,
            supported=supported,
            effective=effective,
            refused=refused,
            attempted=attempted,
            issued=issued,
            reason=reason,
            error=error,
        )

    def _emit(self, event: CheckpointPageTelemetry) -> None:
        if self._telemetry is not None:
            try:
                self._telemetry(event)
            except Exception:
                # Observability must not change checkpoint lifecycle semantics.
                pass
