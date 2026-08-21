from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.checkpoint_assets import CheckpointAssetScope
from circuit_tracer.transcoder.checkpoint_manifest import (
    CheckpointManifestDiagnosticCode,
    build_checkpoint_manifest,
    parse_safetensors_payloads,
)


def _save(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, path)


def test_builds_exact_clt_ranges_with_explicit_shared_scope(tmp_path: Path) -> None:
    path = tmp_path / "layer.safetensors"
    _save(
        path,
        {
            "W_dec_0": torch.ones(2, 3),
            "W_enc_0": torch.ones(3, 2),
            "b_dec_0": torch.ones(2),
        },
    )

    discovered = build_checkpoint_manifest("clt", {0: path})

    assert discovered.ok
    assert discovered.manifest is not None
    asset = discovered.manifest.asset("clt:layer:0")
    assert asset.scope is CheckpointAssetScope.SHARED
    assert asset.path == path
    assert {item.role for item in asset.ranges} == {"decoder", "encoder", "refresh"}
    header = parse_safetensors_payloads(path)
    assert header.ok
    assert {(item.offset, item.length) for item in asset.ranges} == {
        (item.offset, item.length) for item in header.payloads
    }


def test_plt_lowercase_keys_are_classified_and_ids_are_logical(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _save(first, {"w_dec": torch.ones(2, 3), "w_enc": torch.ones(3, 2)})
    _save(second, {"w_dec": torch.ones(2, 3), "w_enc": torch.ones(3, 2)})

    discovered = build_checkpoint_manifest("plt", [first, second], scope=CheckpointAssetScope.EXCLUSIVE)

    assert discovered.ok and discovered.manifest is not None
    assert [asset.asset_id for asset in discovered.manifest.assets] == ["plt:layer:0", "plt:layer:1"]
    assert all(asset.scope is CheckpointAssetScope.EXCLUSIVE for asset in discovered.manifest.assets)
    assert all(asset.roles == frozenset({"decoder", "encoder"}) for asset in discovered.manifest.assets)


@pytest.mark.parametrize("contents", [b"short", struct.pack("<Q", 1000) + b"{}"])
def test_invalid_headers_return_typed_diagnostics(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(contents)

    result = parse_safetensors_payloads(path)
    discovered = build_checkpoint_manifest("clt", [path])

    assert not result.ok
    assert result.diagnostic is not None
    assert result.diagnostic.code is CheckpointManifestDiagnosticCode.INVALID_HEADER
    assert not discovered.ok and discovered.manifest is None
    assert discovered.diagnostics[0].code is CheckpointManifestDiagnosticCode.INVALID_HEADER


def test_duplicate_json_tensor_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.safetensors"
    header = b'{"w_dec":{"data_offsets":[0,4]},"w_dec":{"data_offsets":[4,8]}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"12345678")

    result = parse_safetensors_payloads(path)

    assert not result.ok
    assert result.diagnostic is not None
    assert result.diagnostic.code is CheckpointManifestDiagnosticCode.DUPLICATE_TENSOR_KEY


def test_repeated_path_and_hard_link_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "source.safetensors"
    linked = tmp_path / "linked.safetensors"
    _save(path, {"w_dec": torch.ones(2, 3)})
    os.link(path, linked)

    repeated = build_checkpoint_manifest("plt", [path, path])
    linked_result = build_checkpoint_manifest("plt", [path, linked])

    assert repeated.manifest is None
    assert repeated.diagnostics[0].code is CheckpointManifestDiagnosticCode.DUPLICATE_PATH
    assert linked_result.manifest is None
    assert linked_result.diagnostics[0].code is CheckpointManifestDiagnosticCode.DUPLICATE_FILE_IDENTITY
