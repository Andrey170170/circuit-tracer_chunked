"""Safe discovery of exact safetensors payload ranges for checkpoint paging."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from circuit_tracer.transcoder.checkpoint_assets import (
    CheckpointAsset,
    CheckpointAssetScope,
    CheckpointManifest,
    CheckpointRange,
)


CheckpointArchitecture = Literal["clt", "plt"]
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class CheckpointManifestDiagnosticCode(str, Enum):
    """Reasons a checkpoint was deliberately excluded from discovery."""

    OPEN_FAILED = "open_failed"
    INVALID_HEADER = "invalid_header"
    DUPLICATE_TENSOR_KEY = "duplicate_tensor_key"
    DUPLICATE_PATH = "duplicate_path"
    DUPLICATE_FILE_IDENTITY = "duplicate_file_identity"
    INVALID_MANIFEST = "invalid_manifest"


@dataclass(frozen=True, slots=True)
class CheckpointManifestDiagnostic:
    code: CheckpointManifestDiagnosticCode
    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class SafetensorsPayload:
    """A tensor's exact, non-empty physical byte range in a safetensors file."""

    key: str
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class SafetensorsHeaderResult:
    path: Path
    payloads: tuple[SafetensorsPayload, ...] = ()
    diagnostic: CheckpointManifestDiagnostic | None = None

    @property
    def ok(self) -> bool:
        return self.diagnostic is None


@dataclass(frozen=True, slots=True)
class CheckpointManifestDiscovery:
    """A manifest or explicit diagnostics; unsafe files never receive broad ranges."""

    manifest: CheckpointManifest | None
    diagnostics: tuple[CheckpointManifestDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.manifest is not None and not self.diagnostics


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_safetensors_payloads(path: str | os.PathLike[str]) -> SafetensorsHeaderResult:
    """Parse a local safetensors header without loading tensor payloads.

    All offsets returned are absolute file offsets.  Invalid or ambiguous headers
    return a diagnostic rather than a guessed/file-wide range.
    """

    normalized_path = Path(path)
    try:
        with normalized_path.open("rb") as handle:
            file_size = os.fstat(handle.fileno()).st_size
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > _MAX_HEADER_BYTES or header_size > file_size - 8:
                raise ValueError("safetensors header length is outside file bounds")
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError("truncated safetensors header")
    except OSError as exc:
        return SafetensorsHeaderResult(
            normalized_path,
            diagnostic=CheckpointManifestDiagnostic(
                CheckpointManifestDiagnosticCode.OPEN_FAILED, normalized_path, str(exc)
            ),
        )
    except ValueError as exc:
        return SafetensorsHeaderResult(
            normalized_path,
            diagnostic=CheckpointManifestDiagnostic(
                CheckpointManifestDiagnosticCode.INVALID_HEADER, normalized_path, str(exc)
            ),
        )

    try:
        parsed = json.loads(raw_header, object_pairs_hook=_duplicate_rejecting_object)
        if not isinstance(parsed, dict):
            raise ValueError("safetensors header must be a JSON object")
        data_start = 8 + header_size
        payloads: list[SafetensorsPayload] = []
        for key, entry in parsed.items():
            if key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"tensor {key!r} has a non-object header entry")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            ):
                raise ValueError(f"tensor {key!r} has invalid data_offsets")
            start, end = offsets
            if start < 0 or end <= start or data_start + end > file_size:
                raise ValueError(f"tensor {key!r} payload is outside file bounds")
            payloads.append(SafetensorsPayload(key, data_start + start, end - start))
        if not payloads:
            raise ValueError("safetensors file contains no non-empty tensor payloads")
        ordered = sorted(payloads, key=lambda payload: (payload.offset, payload.key))
        for previous, current in zip(ordered, ordered[1:]):
            if current.offset < previous.offset + previous.length:
                raise ValueError("safetensors tensor payloads overlap")
        return SafetensorsHeaderResult(normalized_path, tuple(ordered))
    except json.JSONDecodeError as exc:
        return SafetensorsHeaderResult(
            normalized_path,
            diagnostic=CheckpointManifestDiagnostic(
                CheckpointManifestDiagnosticCode.INVALID_HEADER, normalized_path, str(exc)
            ),
        )
    except ValueError as exc:
        code = (
            CheckpointManifestDiagnosticCode.DUPLICATE_TENSOR_KEY
            if str(exc).startswith("duplicate JSON key")
            else CheckpointManifestDiagnosticCode.INVALID_HEADER
        )
        return SafetensorsHeaderResult(
            normalized_path,
            diagnostic=CheckpointManifestDiagnostic(code, normalized_path, str(exc)),
        )


def checkpoint_range_role(tensor_key: str) -> str:
    """Classify tensor payloads without assuming any provider implementation."""

    if tensor_key.startswith(("W_dec", "w_dec")):
        return "decoder"
    if tensor_key.startswith(("W_enc", "w_enc")):
        return "encoder"
    return "refresh"


def build_checkpoint_manifest(
    architecture: CheckpointArchitecture,
    paths: Mapping[int, str | os.PathLike[str]] | Sequence[str | os.PathLike[str]],
    *,
    scope: CheckpointAssetScope = CheckpointAssetScope.SHARED,
) -> CheckpointManifestDiscovery:
    """Build an exact-range manifest for CLT or PLT checkpoint files.

    ``scope`` is explicit and deliberately defaults to ``SHARED``.  A malformed,
    repeated, or identity-duplicated file rejects the whole discovery result so
    callers cannot accidentally issue advice for a partial manifest.
    """

    if architecture not in ("clt", "plt"):
        raise ValueError(f"unsupported checkpoint architecture: {architecture!r}")
    normalized_scope = CheckpointAssetScope(scope)
    entries = sorted(paths.items()) if isinstance(paths, Mapping) else list(enumerate(paths))
    normalized_entries = [(int(index), Path(path)) for index, path in entries]
    seen_paths: set[Path] = set()
    diagnostics: list[CheckpointManifestDiagnostic] = []
    assets: list[CheckpointAsset] = []
    seen_file_ids: set[tuple[int, int]] = set()

    for layer, path in normalized_entries:
        absolute_path = path.absolute()
        if absolute_path in seen_paths:
            diagnostics.append(
                CheckpointManifestDiagnostic(
                    CheckpointManifestDiagnosticCode.DUPLICATE_PATH,
                    path,
                    f"checkpoint path is repeated for logical layer {layer}",
                )
            )
            continue
        seen_paths.add(absolute_path)
        header = parse_safetensors_payloads(path)
        if not header.ok:
            assert header.diagnostic is not None
            diagnostics.append(header.diagnostic)
            continue
        asset_id = f"{architecture}:layer:{layer}"
        ranges = tuple(
            CheckpointRange(asset_id, checkpoint_range_role(payload.key), payload.offset, payload.length)
            for payload in header.payloads
        )
        try:
            asset = CheckpointAsset.from_path(
                asset_id=asset_id, path=path, scope=normalized_scope, ranges=ranges
            )
        except (OSError, ValueError) as exc:
            diagnostics.append(
                CheckpointManifestDiagnostic(
                    CheckpointManifestDiagnosticCode.OPEN_FAILED, path, str(exc)
                )
            )
            continue
        file_id = (asset.device, asset.inode)
        if file_id in seen_file_ids:
            diagnostics.append(
                CheckpointManifestDiagnostic(
                    CheckpointManifestDiagnosticCode.DUPLICATE_FILE_IDENTITY,
                    path,
                    "checkpoint file identity is already owned by another logical layer",
                )
            )
            continue
        seen_file_ids.add(file_id)
        assets.append(asset)

    if diagnostics:
        return CheckpointManifestDiscovery(None, tuple(diagnostics))
    try:
        return CheckpointManifestDiscovery(CheckpointManifest(tuple(assets)))
    except ValueError as exc:
        return CheckpointManifestDiscovery(
            None,
            (
                CheckpointManifestDiagnostic(
                    CheckpointManifestDiagnosticCode.INVALID_MANIFEST,
                    Path("."),
                    str(exc),
                ),
            ),
        )
