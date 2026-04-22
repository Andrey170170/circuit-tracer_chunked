"""Unit tests for the prefix activation cache.

These tests exercise the cache's book-keeping logic — hit / miss /
invalidation paths, shape validation, fingerprint matching, and prefix
slicing — in isolation from the rest of the attribution pipeline.  They
run on CPU only and use small hand-built tensors, no model needed, so
they should complete in well under a second.

Covered behaviors (match the checklist in PR notes / BRIEF.md
deliverable 1):

1. Empty cache returns a miss.
2. Store + look up same prefix => full hit, all positions reusable.
3. Store short prefix, look up an extending prefix => partial hit.
4. Store a prefix, look up tokens that diverge mid-prefix => invalidate.
5. Look up a shorter prompt than what's cached => invalidate.
6. Fingerprint mismatch => invalidate.
7. ``extract_prefix_slices`` returns correctly-shaped slices.
8. Shape validation rejects non-3D tensors and mismatched position axes.
9. ``clear()`` drops the entry.
10. Hit / miss counters increment as expected.
"""

from __future__ import annotations

import pytest
import torch

from circuit_tracer.attribution.prefix_cache import (
    CacheDiagnostics,
    PrefixActivationCache,
)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #

N_LAYERS = 3
D_MODEL = 4
FINGERPRINT = "model=test|n_layers=3|d_model=4|dtype=float32|device=cpu"


def _make_tokens(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.long)


def _make_activation(n_pos: int, *, fill: float = 0.0) -> torch.Tensor:
    """Returns a deterministic (n_layers, n_pos, d_model) tensor with
    values that depend on ``(layer, pos, dim)`` so slicing errors are
    visible."""
    t = torch.zeros(N_LAYERS, n_pos, D_MODEL, dtype=torch.float32)
    for layer in range(N_LAYERS):
        for pos in range(n_pos):
            for dim in range(D_MODEL):
                t[layer, pos, dim] = layer * 100 + pos * 10 + dim + fill
    return t


def _populate(cache: PrefixActivationCache, token_ids: list[int]) -> None:
    tokens = _make_tokens(token_ids)
    cache.store(
        token_ids=tokens,
        mlp_in=_make_activation(len(token_ids), fill=0.0),
        mlp_out=_make_activation(len(token_ids), fill=0.5),
        model_fingerprint=FINGERPRINT,
    )


# --------------------------------------------------------------------- #
# 1. Empty cache                                                         #
# --------------------------------------------------------------------- #


def test_empty_cache_is_a_miss() -> None:
    cache = PrefixActivationCache()
    assert not cache.has_entry
    assert cache.cached_prefix_len == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    entry, diag = cache.lookup(
        _make_tokens([1, 2, 3]),
        model_fingerprint=FINGERPRINT,
    )

    assert entry is None
    assert diag.cache_state == "miss"
    assert diag.reused_positions == 0
    assert diag.recomputed_positions == 3
    assert cache.miss_count == 1
    assert cache.hit_count == 0


# --------------------------------------------------------------------- #
# 2. Full hit                                                            #
# --------------------------------------------------------------------- #


def test_full_hit_returns_entry() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3, 4, 5])

    assert cache.has_entry
    assert cache.cached_prefix_len == 5

    entry, diag = cache.lookup(
        _make_tokens([1, 2, 3, 4, 5]),
        model_fingerprint=FINGERPRINT,
    )

    assert entry is not None
    assert diag.cache_state == "full_hit"
    assert diag.reused_positions == 5
    assert diag.recomputed_positions == 0
    assert cache.hit_count == 1


# --------------------------------------------------------------------- #
# 3. Partial hit — extending prefix                                      #
# --------------------------------------------------------------------- #


def test_partial_hit_on_extending_prefix() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3])

    entry, diag = cache.lookup(
        _make_tokens([1, 2, 3, 7, 8]),
        model_fingerprint=FINGERPRINT,
    )

    assert entry is not None
    assert diag.cache_state == "partial_hit"
    assert diag.reused_positions == 3
    assert diag.recomputed_positions == 2
    assert cache.hit_count == 1


# --------------------------------------------------------------------- #
# 4. Diverging tokens invalidate                                         #
# --------------------------------------------------------------------- #


def test_diverging_tokens_invalidate() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3, 4, 5])

    entry, diag = cache.lookup(
        _make_tokens([1, 2, 9, 4, 5]),
        model_fingerprint=FINGERPRINT,
    )

    assert entry is None
    assert diag.cache_state == "invalidated"
    assert "do not match" in diag.reason
    assert not cache.has_entry  # cleared on invalidation
    assert cache.miss_count == 1


# --------------------------------------------------------------------- #
# 5. Shorter prompt invalidates                                          #
# --------------------------------------------------------------------- #


def test_shorter_prompt_invalidates() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3, 4, 5])

    entry, diag = cache.lookup(
        _make_tokens([1, 2]),
        model_fingerprint=FINGERPRINT,
    )

    assert entry is None
    assert diag.cache_state == "invalidated"
    assert "shorter than cached" in diag.reason
    assert not cache.has_entry


# --------------------------------------------------------------------- #
# 6. Fingerprint mismatch invalidates                                    #
# --------------------------------------------------------------------- #


def test_fingerprint_mismatch_invalidates() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3])

    entry, diag = cache.lookup(
        _make_tokens([1, 2, 3]),
        model_fingerprint="model=different|n_layers=6|d_model=8",
    )

    assert entry is None
    assert diag.cache_state == "invalidated"
    assert "fingerprint mismatch" in diag.reason
    assert not cache.has_entry


# --------------------------------------------------------------------- #
# 7. extract_prefix_slices returns correct shape and values              #
# --------------------------------------------------------------------- #


def test_extract_prefix_slices_shapes_and_values() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [10, 20, 30, 40, 50])

    entry, _ = cache.lookup(
        _make_tokens([10, 20, 30, 40, 50]),
        model_fingerprint=FINGERPRINT,
    )
    assert entry is not None

    mlp_in_slice, mlp_out_slice = PrefixActivationCache.extract_prefix_slices(
        entry,
        prefix_len=3,
        target_device="cpu",
    )

    # Shape: (n_layers, prefix_len, d_model)
    assert mlp_in_slice.shape == (N_LAYERS, 3, D_MODEL)
    assert mlp_out_slice.shape == (N_LAYERS, 3, D_MODEL)

    # Values: match the first 3 position rows of the stored tensor.
    expected_in = _make_activation(5, fill=0.0)[:, :3, :]
    expected_out = _make_activation(5, fill=0.5)[:, :3, :]
    assert torch.equal(mlp_in_slice, expected_in)
    assert torch.equal(mlp_out_slice, expected_out)


def test_extract_prefix_slices_rejects_out_of_range() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3])
    entry, _ = cache.lookup(
        _make_tokens([1, 2, 3]),
        model_fingerprint=FINGERPRINT,
    )
    assert entry is not None

    with pytest.raises(ValueError, match="out of range"):
        PrefixActivationCache.extract_prefix_slices(
            entry, prefix_len=10, target_device="cpu"
        )

    with pytest.raises(ValueError, match="out of range"):
        PrefixActivationCache.extract_prefix_slices(
            entry, prefix_len=-1, target_device="cpu"
        )


# --------------------------------------------------------------------- #
# 8. Shape validation                                                    #
# --------------------------------------------------------------------- #


def test_store_rejects_non_3d_tensor() -> None:
    cache = PrefixActivationCache()
    tokens = _make_tokens([1, 2, 3])

    flat = torch.zeros(3, D_MODEL)  # 2-D, wrong rank
    with pytest.raises(ValueError, match="must be 3-D"):
        cache.store(
            token_ids=tokens,
            mlp_in=flat,
            mlp_out=_make_activation(3),
            model_fingerprint=FINGERPRINT,
        )


def test_store_rejects_position_axis_mismatch() -> None:
    cache = PrefixActivationCache()
    tokens = _make_tokens([1, 2, 3])

    wrong = _make_activation(5)  # 5 positions instead of 3
    with pytest.raises(ValueError, match="position axis"):
        cache.store(
            token_ids=tokens,
            mlp_in=wrong,
            mlp_out=_make_activation(3),
            model_fingerprint=FINGERPRINT,
        )


def test_store_rejects_non_tensor_tokens() -> None:
    cache = PrefixActivationCache()
    with pytest.raises(TypeError, match="torch.Tensor"):
        cache.store(
            token_ids=[1, 2, 3],  # type: ignore[arg-type]
            mlp_in=_make_activation(3),
            mlp_out=_make_activation(3),
            model_fingerprint=FINGERPRINT,
        )


def test_store_rejects_non_1d_tokens() -> None:
    cache = PrefixActivationCache()
    with pytest.raises(ValueError, match="must be 1-D"):
        cache.store(
            token_ids=torch.tensor([[1, 2, 3]]),  # 2-D
            mlp_in=_make_activation(3),
            mlp_out=_make_activation(3),
            model_fingerprint=FINGERPRINT,
        )


# --------------------------------------------------------------------- #
# 9. clear() drops the entry                                             #
# --------------------------------------------------------------------- #


def test_clear_drops_the_entry() -> None:
    cache = PrefixActivationCache()
    _populate(cache, [1, 2, 3])
    assert cache.has_entry

    cache.clear()

    assert not cache.has_entry
    assert cache.cached_prefix_len == 0
    entry, diag = cache.lookup(
        _make_tokens([1, 2, 3]),
        model_fingerprint=FINGERPRINT,
    )
    assert entry is None
    assert diag.cache_state == "miss"


# --------------------------------------------------------------------- #
# 10. Hit/miss counters                                                  #
# --------------------------------------------------------------------- #


def test_hit_and_miss_counters_track_correctly() -> None:
    cache = PrefixActivationCache()
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    # First lookup against an empty cache -> miss
    cache.lookup(_make_tokens([1]), model_fingerprint=FINGERPRINT)
    assert cache.miss_count == 1

    # Populate and hit twice
    _populate(cache, [1, 2, 3])
    cache.lookup(_make_tokens([1, 2, 3]), model_fingerprint=FINGERPRINT)
    cache.lookup(_make_tokens([1, 2, 3, 4]), model_fingerprint=FINGERPRINT)
    assert cache.hit_count == 2

    # Invalidate with a divergent lookup -> miss
    cache.lookup(_make_tokens([1, 9, 3]), model_fingerprint=FINGERPRINT)
    assert cache.miss_count == 2
    assert not cache.has_entry


# --------------------------------------------------------------------- #
# Diagnostics round-trip to dict                                         #
# --------------------------------------------------------------------- #


def test_diagnostics_to_dict_has_stable_keys() -> None:
    diag = CacheDiagnostics(
        cached_prefix_len=5,
        reused_positions=3,
        recomputed_positions=2,
        cache_state="partial_hit",
        reason="example",
    )
    d = diag.to_dict()
    assert d == {
        "cached_prefix_len": 5,
        "reused_positions": 3,
        "recomputed_positions": 2,
        "cache_state": "partial_hit",
        "reason": "example",
    }
