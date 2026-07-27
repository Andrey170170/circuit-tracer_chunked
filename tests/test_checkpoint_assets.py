from __future__ import annotations

import os
from mmap import PAGESIZE
from pathlib import Path
from threading import Thread

import pytest

from circuit_tracer.transcoder.checkpoint_assets import (
    CheckpointAsset,
    CheckpointAssetScope,
    CheckpointManifest,
    CheckpointPageAdvice,
    CheckpointPageLifecycle,
    CheckpointRange,
)


def _asset(
    path: Path,
    *,
    scope: CheckpointAssetScope = CheckpointAssetScope.JOB_PRIVATE,
    ranges: tuple[CheckpointRange, ...] | None = None,
) -> CheckpointAsset:
    if ranges is None:
        ranges = (CheckpointRange("weights", "decoder", 8, 16),)
    return CheckpointAsset.from_path(
        asset_id="weights",
        path=path,
        scope=scope,
        ranges=ranges,
    )


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(bytes(range(64)))
    return path


def test_asset_rejects_overlapping_and_wrong_owner_ranges(checkpoint_file: Path) -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        _asset(
            checkpoint_file,
            ranges=(
                CheckpointRange("weights", "encoder", 0, 16),
                CheckpointRange("weights", "decoder", 8, 16),
            ),
        )

    with pytest.raises(ValueError, match="belongs to"):
        _asset(
            checkpoint_file,
            ranges=(CheckpointRange("other", "decoder", 0, 8),),
        )


def test_asset_rejects_range_past_end_of_file(checkpoint_file: Path) -> None:
    with pytest.raises(ValueError, match="exceeds file size"):
        _asset(
            checkpoint_file,
            ranges=(CheckpointRange("weights", "decoder", 48, 17),),
        )


def test_manifest_rejects_duplicate_inode_even_through_hard_link(
    checkpoint_file: Path,
    tmp_path: Path,
) -> None:
    hard_link = tmp_path / "hard-link.safetensors"
    os.link(checkpoint_file, hard_link)
    first = _asset(checkpoint_file)
    second_range = CheckpointRange("other", "encoder", 0, 8)
    second = CheckpointAsset.from_path(
        asset_id="other",
        path=hard_link,
        scope=CheckpointAssetScope.EXCLUSIVE,
        ranges=(second_range,),
    )

    with pytest.raises(ValueError, match="exactly one manifest owner"):
        CheckpointManifest((first, second))


def test_mixed_role_file_uses_only_exact_target_ranges(checkpoint_file: Path) -> None:
    encoder = CheckpointRange("weights", "encoder", 0, 8)
    decoder = CheckpointRange("weights", "decoder", 16, 24)
    asset = _asset(checkpoint_file, ranges=(encoder, decoder))
    calls: list[tuple[int, int, int]] = []

    def fadvise(_fd: int, offset: int, length: int, flag: int) -> None:
        calls.append((offset, length, flag))

    lifecycle = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    )

    assert asset.has_mixed_roles
    prefault = lifecycle.prefault(encoder)
    release = lifecycle.release(decoder)
    assert prefault.outcome == "issued" and prefault.issued
    assert release.outcome == "issued" and release.issued
    assert not prefault.effective and not release.effective
    assert calls == [(0, 8, 11), (16, 24, 12)]
    assert all(length > 0 for _, length, _ in calls)


def test_advice_reports_exact_request_and_unverified_containing_page_span(
    tmp_path: Path,
) -> None:
    path = tmp_path / "page-span.safetensors"
    path.write_bytes(b"x" * (PAGESIZE + 16))
    byte_range = CheckpointRange("weights", "decoder", PAGESIZE - 2, 4)
    asset = _asset(path, ranges=(byte_range,))
    calls: list[tuple[int, int]] = []

    def fadvise(_fd: int, offset: int, length: int, _flag: int) -> None:
        calls.append((offset, length))

    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    ).release(byte_range)

    assert calls == [(PAGESIZE - 2, 4)]
    assert (event.offset, event.length) == (PAGESIZE - 2, 4)
    assert event.page_size == PAGESIZE
    assert event.page_span_offset == 0
    assert event.page_span_length == 2 * PAGESIZE
    assert event.kernel_effect_granularity == "page"
    assert not event.kernel_effect_verified
    assert event.issued and not event.effective


def test_shared_file_refuses_advice_without_calling_platform(checkpoint_file: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(
        checkpoint_file,
        scope=CheckpointAssetScope.SHARED,
        ranges=(byte_range,),
    )
    calls = 0

    def fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1

    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    ).release(byte_range)

    assert event.outcome == "refused"
    assert event.supported
    assert event.refused
    assert not event.effective
    assert not event.attempted
    assert calls == 0


def test_stale_inode_refuses_advice(checkpoint_file: Path, tmp_path: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"x" * 64)
    os.replace(replacement, checkpoint_file)
    calls = 0

    def fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1

    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    ).release(byte_range)

    assert event.outcome == "refused"
    assert event.reason == "open_file_identity_does_not_match_manifest"
    assert not event.attempted
    assert calls == 0


@pytest.mark.parametrize("mutation", ["rewrite", "truncate"])
def test_same_inode_mutation_refuses_advice(checkpoint_file: Path, mutation: str) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    original_inode = checkpoint_file.stat().st_ino
    with checkpoint_file.open("r+b") as handle:
        if mutation == "rewrite":
            handle.seek(0)
            handle.write(b"x" * 64)
        else:
            handle.truncate(48)
        handle.flush()
        os.fsync(handle.fileno())
    # A fast same-size rewrite can share a coarse filesystem timestamp with the
    # manifest capture. Make the metadata-identity change explicit so this test
    # exercises the guard rather than the timestamp resolution of its temp FS.
    before_utime = checkpoint_file.stat()
    os.utime(
        checkpoint_file,
        ns=(before_utime.st_atime_ns, before_utime.st_mtime_ns + 1_000_000_000),
    )
    assert checkpoint_file.stat().st_mtime_ns != asset.mtime_ns
    assert checkpoint_file.stat().st_ino == original_inode
    calls = 0

    def fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1

    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    ).release(byte_range)

    assert event.outcome == "refused"
    assert event.reason == "open_file_metadata_does_not_match_manifest"
    assert not event.attempted
    assert calls == 0


def test_forged_partial_range_is_refused(checkpoint_file: Path) -> None:
    owned = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(owned,))
    forged = CheckpointRange("weights", "decoder", 8, 8)
    calls = 0

    def fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1

    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    ).release(forged)

    assert event.outcome == "refused"
    assert event.reason == "range_not_manifest_owned"
    assert calls == 0


def test_prefault_release_and_advice_are_idempotent(checkpoint_file: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    calls: list[tuple[int, int, int]] = []
    events = []

    def fadvise(_fd: int, offset: int, length: int, flag: int) -> None:
        calls.append((offset, length, flag))

    lifecycle = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        telemetry=events.append,
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    )

    first_prefault = lifecycle.prefault(byte_range)
    second_prefault = lifecycle.advise(byte_range, CheckpointPageAdvice.PREFAULT)
    first_release = lifecycle.release(byte_range)
    second_release = lifecycle.release(byte_range)

    assert first_prefault.issued and first_release.issued
    assert not first_prefault.effective and not first_release.effective
    assert second_prefault.idempotent and not second_prefault.attempted
    assert second_release.idempotent and not second_release.attempted
    assert calls == [(8, 16, 11), (8, 16, 12)]
    assert events == [
        first_prefault,
        second_prefault,
        first_release,
        second_release,
    ]


def test_reentrant_telemetry_callback_does_not_deadlock(checkpoint_file: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    calls = 0
    callback_events = []
    nested_events = []
    reentered = False
    lifecycle: CheckpointPageLifecycle

    def fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1

    def telemetry(event) -> None:
        nonlocal reentered
        callback_events.append(event)
        if not reentered:
            reentered = True
            nested_events.append(lifecycle.release(byte_range))

    lifecycle = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        telemetry=telemetry,
        posix_fadvise=fadvise,
        prefault_flag=11,
        release_flag=12,
    )
    result = []
    thread = Thread(target=lambda: result.append(lifecycle.release(byte_range)), daemon=True)
    thread.start()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(result) == 1
    assert result[0].issued
    assert len(nested_events) == 1 and nested_events[0].idempotent
    assert [event.idempotent for event in callback_events] == [False, True]
    assert calls == 1


def test_unavailable_advice_is_observable_noop(checkpoint_file: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    event = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=None,
        prefault_flag=None,
        release_flag=None,
    ).prefault(byte_range)

    assert event.outcome == "unavailable"
    assert not event.supported
    assert not event.effective
    assert not event.refused
    assert not event.attempted
    assert event.error is None


def test_failed_advice_is_observable_noop_and_not_retried(checkpoint_file: Path) -> None:
    byte_range = CheckpointRange("weights", "decoder", 8, 16)
    asset = _asset(checkpoint_file, ranges=(byte_range,))
    calls = 0

    def failing_fadvise(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise OSError("filesystem rejected advice")

    lifecycle = CheckpointPageLifecycle(
        CheckpointManifest((asset,)),
        posix_fadvise=failing_fadvise,
        prefault_flag=11,
        release_flag=12,
    )
    first = lifecycle.release(byte_range)
    second = lifecycle.release(byte_range)

    assert first.outcome == "error"
    assert first.supported
    assert not first.effective
    assert first.attempted
    assert first.error == "OSError: filesystem rejected advice"
    assert second.idempotent and not second.attempted
    assert calls == 1
