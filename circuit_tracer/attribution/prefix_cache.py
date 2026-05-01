"""Prefix activation cache for temporal / consecutive-step tracing.

When an attribution workflow traces a model across consecutive generation
steps, each step's input is a prefix of the next step's input: only the
last token changes.  Because the underlying model is causally masked, the
MLP activations at prefix positions are mathematically identical on every
step.  The prefix cache stores those activations and offers them back to
``setup_attribution`` so the per-position forward-pass work on cached
positions can be reused instead of recomputed.

Design principles
-----------------

1. **Cache the pre-sparsification tensors only.**  The brief
   (``prefix_caching/docs/BRIEF.md``) is unambiguous that a cached trace must
   match an uncached trace in the parts that matter.  Sparsification picks
   a top-K candidate set that depends on the whole prompt's activation
   distribution, so it shifts at the boundary when the prompt grows.  To
   preserve the exact uncached result, we cache only the raw ``mlp_in``
   and ``mlp_out`` tensors and let sparsification run fresh every step.

2. **Strict prefix-equality invalidation.**  A cache entry is valid for a
   new prompt only if the new prompt begins with exactly the cached token
   sequence.  If a single cached token differs, the whole cache is dropped
   and recomputed.  This is a conservative rule that rules out silent
   reuse of stale state.

3. **Layer-slab storage.**  ``setup_attribution`` collects per-layer MLP
   inputs / outputs into tensors of shape ``(n_layers, n_pos, d_model)``.
   We keep that same shape in the cache and slice along the ``n_pos``
   axis when the caller asks for specific prefix positions.

4. **Opaque to sparsification.**  The cache does not know or care about
   feature IDs, decoder vectors, or the activation matrix.  All of that
   is recomputed by the transcoders on every call.  The cache only
   reduces the cost of the upstream forward pass.

Usage sketch (will be wired into ``setup_attribution`` in a follow-up
commit)::

    cache = PrefixActivationCache()
    for step in range(max_steps):
        tokens = current_prompt_tokens()
        ctx = model.setup_attribution(tokens, prefix_cache=cache)
        ...
        # After a successful setup, ``setup_attribution`` updates the cache
        # with the new full tensors so the next step can hit it.

The cache is single-threaded and meant for use inside one Python process
over the lifetime of one generation loop.  It is not a cross-process or
cross-run cache.  Its keys are tuples of token ids plus a coarse model
fingerprint; callers that load a different model into the same cache
object will see a fingerprint mismatch and fall back to a cold setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )


@dataclass(frozen=True, slots=True)
class CacheDiagnostics:
    """Summary of what happened on a single cache lookup.

    This is what the trace loop and the diagnostics panel in
    ``trace_pipeline_cached.py`` should surface per step so the fidelity
    study can tell "was this result genuinely reused" apart from "this
    result was recomputed from scratch."
    """

    cached_prefix_len: int
    reused_positions: int
    recomputed_positions: int
    cache_state: str  # "miss", "partial_hit", "full_hit", "invalidated"
    reason: str  # human-readable explanation

    def to_dict(self) -> dict[str, object]:
        return {
            "cached_prefix_len": self.cached_prefix_len,
            "reused_positions": self.reused_positions,
            "recomputed_positions": self.recomputed_positions,
            "cache_state": self.cache_state,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _CacheEntry:
    """Internal record of the activations stored for a specific prefix.

    Stored tensors are on CPU by default to avoid growing VRAM pressure as
    the generation loop proceeds.  Callers move them back to the model's
    device on hit; storage and transport costs stay inside the cache.

    ``past_key_values`` is the HuggingFace KV state captured from the
    cold forward pass, normalized to ``tuple[tuple[Tensor K, Tensor V],
    ...]`` (per-layer).  When set, a follow-up call can ask the model to
    skip the forward pass over the cached positions by passing the KV
    back into ``model.trace(input_ids=new_tokens, past_key_values=...)``.
    """

    token_ids: tuple[int, ...]
    model_fingerprint: str
    mlp_in: torch.Tensor
    mlp_out: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    @property
    def has_kv(self) -> bool:
        return self.past_key_values is not None


class PrefixCacheMiss(Exception):
    """Raised from internal helpers when the cache does not satisfy the
    caller's request.  Not surfaced to end users; ``setup_attribution``
    catches it and falls back to full recomputation.
    """


class PrefixActivationCache:
    """Store raw (pre-sparsification) MLP activations keyed by prefix.

    Thread-unsafe, in-process only.  Designed to be instantiated once at
    the top of a temporal-tracing loop and passed into every
    ``setup_attribution`` call as ``prefix_cache=``.

    Exactly one entry is retained at a time.  Consecutive generation steps
    always extend the current prefix, so keeping a single entry is enough
    to serve all typical lookups; keeping more would only waste host RAM.

    Parameters
    ----------
    storage_device:
        Device to hold cached tensors on.  Defaults to ``"cpu"`` so the
        cache does not compete for VRAM with the attribution pipeline.
        Callers can pass ``"cuda"`` to avoid host transfers at the cost
        of VRAM.
    """

    def __init__(self, *, storage_device: str | torch.device = "cpu") -> None:
        self._storage_device = torch.device(storage_device)
        self._entry: _CacheEntry | None = None
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Fingerprint construction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_model_fingerprint(
        model: "NNSightReplacementModel",
    ) -> str:
        """Produce a short string that identifies a model configuration.

        Two models with the same fingerprint are assumed to produce
        byte-identical MLP activations for the same token sequence.  The
        fingerprint deliberately mixes the model name, transcoder name,
        dtype, and device; it is not a cryptographic hash.  Callers who
        need stronger isolation should instantiate one cache per
        (model, transcoder) pair.
        """

        parts: list[str] = []
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            parts.append(f"model={getattr(cfg, 'model_name', type(model).__name__)}")
            parts.append(f"n_layers={getattr(cfg, 'n_layers', '?')}")
            parts.append(f"d_model={getattr(cfg, 'd_model', '?')}")
        transcoders = getattr(model, "transcoders", None)
        if transcoders is not None:
            parts.append(
                f"transcoder={getattr(transcoders, 'transcoder_name', type(transcoders).__name__)}"
            )
        parts.append(f"dtype={getattr(model, 'dtype', '?')}")
        parts.append(f"device={getattr(model, 'device', '?')}")
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    @property
    def has_entry(self) -> bool:
        return self._entry is not None

    @property
    def cached_prefix_len(self) -> int:
        return 0 if self._entry is None else self._entry.prefix_len

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    def clear(self) -> None:
        """Drop the cached entry (e.g. between completions)."""
        self._entry = None

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        token_ids: torch.Tensor,
        *,
        model_fingerprint: str,
    ) -> tuple[_CacheEntry | None, CacheDiagnostics]:
        """Check whether the cached entry can serve ``token_ids``.

        Returns a ``(entry, diagnostics)`` pair.  ``entry`` is ``None`` on
        a miss or an invalidation; callers fall back to a full cold
        compute.  ``diagnostics`` is always populated so the caller can
        log it without branching.

        Validity rule: the cached prefix must be a strict prefix of
        ``token_ids`` (same tokens at every cached position), and the
        stored fingerprint must match.  If ``token_ids`` is shorter than
        the cached prefix, that counts as a miss — we never "partially"
        use a cached suffix.
        """

        self._coerce_token_tensor(token_ids)
        tokens = tuple(int(t) for t in token_ids.tolist())
        new_len = len(tokens)

        if self._entry is None:
            self._misses += 1
            return None, CacheDiagnostics(
                cached_prefix_len=0,
                reused_positions=0,
                recomputed_positions=new_len,
                cache_state="miss",
                reason="cache empty",
            )

        if self._entry.model_fingerprint != model_fingerprint:
            self._misses += 1
            reason = (
                f"fingerprint mismatch "
                f"(cached={self._entry.model_fingerprint!r}, "
                f"new={model_fingerprint!r})"
            )
            self._entry = None
            return None, CacheDiagnostics(
                cached_prefix_len=0,
                reused_positions=0,
                recomputed_positions=new_len,
                cache_state="invalidated",
                reason=reason,
            )

        cached = self._entry.token_ids
        cached_len = len(cached)

        if new_len < cached_len:
            self._misses += 1
            self._entry = None
            return None, CacheDiagnostics(
                cached_prefix_len=cached_len,
                reused_positions=0,
                recomputed_positions=new_len,
                cache_state="invalidated",
                reason=(
                    f"new prompt is shorter than cached prefix "
                    f"({new_len} < {cached_len})"
                ),
            )

        if tokens[:cached_len] != cached:
            self._misses += 1
            self._entry = None
            return None, CacheDiagnostics(
                cached_prefix_len=cached_len,
                reused_positions=0,
                recomputed_positions=new_len,
                cache_state="invalidated",
                reason="cached tokens do not match new prompt prefix",
            )

        self._hits += 1
        reused = cached_len
        recomputed = new_len - cached_len
        state = "full_hit" if recomputed == 0 else "partial_hit"
        return self._entry, CacheDiagnostics(
            cached_prefix_len=cached_len,
            reused_positions=reused,
            recomputed_positions=recomputed,
            cache_state=state,
            reason=f"reused {reused} of {new_len} positions",
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        *,
        token_ids: torch.Tensor,
        mlp_in: torch.Tensor,
        mlp_out: torch.Tensor,
        model_fingerprint: str,
        past_key_values: object | None = None,
    ) -> None:
        """Replace the cached entry with fresh activations.

        ``mlp_in`` and ``mlp_out`` are expected to have shape
        ``(n_layers, n_pos, d_model)`` with ``n_pos == len(token_ids)``.
        Tensors are moved to ``storage_device`` and detached from the
        autograd graph before storage.

        ``past_key_values``, when given, is normalized to
        ``tuple[tuple[Tensor K, Tensor V], ...]`` and stored alongside
        the MLP tensors.  Accepted forms are:
        * legacy tuple-of-tuples, ``((K, V), (K, V), ...)``
        * HF ``DynamicCache`` (any object exposing ``.key_cache`` and
          ``.value_cache`` lists, or ``.to_legacy_cache()``).
        Anything else is rejected with ``TypeError``.
        """

        self._coerce_token_tensor(token_ids)
        tokens = tuple(int(t) for t in token_ids.tolist())
        n_pos = len(tokens)

        self._validate_shape(mlp_in, n_pos, name="mlp_in")
        self._validate_shape(mlp_out, n_pos, name="mlp_out")

        normalized_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        if past_key_values is not None:
            normalized_kv = self._normalize_past_key_values(
                past_key_values, n_pos=n_pos
            )

        self._entry = _CacheEntry(
            token_ids=tokens,
            model_fingerprint=model_fingerprint,
            mlp_in=mlp_in.detach().to(self._storage_device, copy=True),
            mlp_out=mlp_out.detach().to(self._storage_device, copy=True),
            past_key_values=normalized_kv,
        )

    # ------------------------------------------------------------------
    # Utilities for callers that consume a hit
    # ------------------------------------------------------------------

    @staticmethod
    def extract_prefix_slices(
        entry: _CacheEntry,
        *,
        prefix_len: int,
        target_device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mlp_in[:, :prefix_len], mlp_out[:, :prefix_len])``
        on the target device.

        This is the primary helper for the (future) ``setup_attribution``
        integration: on a hit, the caller calls ``extract_prefix_slices``
        to get the reusable tensor rows, computes only the new positions
        fresh, concatenates along the position axis, and continues as
        before.
        """
        if prefix_len < 0 or prefix_len > entry.prefix_len:
            raise ValueError(
                f"prefix_len {prefix_len} out of range [0, {entry.prefix_len}]"
            )
        device = torch.device(target_device)
        mlp_in_slice = entry.mlp_in[:, :prefix_len].to(device)
        mlp_out_slice = entry.mlp_out[:, :prefix_len].to(device)
        return mlp_in_slice, mlp_out_slice

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def load_past_key_values(
        self,
        entry: _CacheEntry,
        *,
        target_device: torch.device | str,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
        """Move the cached ``past_key_values`` to ``target_device``.

        Returns ``None`` if the entry has no KV stored.  The returned
        tuple is a fresh device-resident copy; the cache keeps its CPU
        copy intact so the next step can still read it.
        """
        if entry.past_key_values is None:
            return None
        device = torch.device(target_device)
        return tuple(
            (k.to(device), v.to(device)) for (k, v) in entry.past_key_values
        )

    def _normalize_past_key_values(
        self,
        past_key_values: object,
        *,
        n_pos: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Coerce HF KV-cache objects into a per-layer tuple-of-tuples.

        Stored tensors are detached, moved to ``storage_device``, and
        copied so the cache owns its own memory.
        """
        # Tuple-of-tuples (legacy form HF still emits with
        # ``return_legacy_cache=True`` or when use_cache produces a
        # past tuple directly).
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        if isinstance(past_key_values, tuple):
            for i, layer in enumerate(past_key_values):
                if not (isinstance(layer, tuple) and len(layer) == 2):
                    raise TypeError(
                        f"past_key_values[{i}] must be a (K, V) tuple, "
                        f"got {type(layer).__name__}"
                    )
                k, v = layer
                if not (isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)):
                    raise TypeError(
                        f"past_key_values[{i}] entries must be tensors"
                    )
                layers.append((k, v))
        elif hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return self._normalize_past_key_values(legacy, n_pos=n_pos)
        elif hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ):
            keys = list(past_key_values.key_cache)
            values = list(past_key_values.value_cache)
            if len(keys) != len(values):
                raise TypeError(
                    f"DynamicCache key/value layer count mismatch: "
                    f"{len(keys)} vs {len(values)}"
                )
            for k, v in zip(keys, values):
                layers.append((k, v))
        else:
            raise TypeError(
                f"Unsupported past_key_values type: "
                f"{type(past_key_values).__name__}"
            )

        # Sanity: KV seq-len axis should equal prefix length.
        # HF convention: K/V shape is (batch, n_heads, seq_len, head_dim).
        for i, (k, v) in enumerate(layers):
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError(
                    f"past_key_values[{i}]: expected 4-D K/V "
                    f"(batch, n_heads, seq_len, head_dim), got K={tuple(k.shape)}, "
                    f"V={tuple(v.shape)}"
                )
            if k.shape[2] != n_pos or v.shape[2] != n_pos:
                raise ValueError(
                    f"past_key_values[{i}]: seq_len axis must equal "
                    f"prefix length {n_pos}, got K seq_len={k.shape[2]}, "
                    f"V seq_len={v.shape[2]}"
                )

        return tuple(
            (
                k.detach().to(self._storage_device, copy=True),
                v.detach().to(self._storage_device, copy=True),
            )
            for (k, v) in layers
        )

    @staticmethod
    def _coerce_token_tensor(token_ids: torch.Tensor) -> None:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(
                f"token_ids must be a torch.Tensor, got {type(token_ids).__name__}"
            )
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D, got shape {tuple(token_ids.shape)}"
            )

    @staticmethod
    def _validate_shape(tensor: torch.Tensor, n_pos: int, *, name: str) -> None:
        if tensor.ndim != 3:
            raise ValueError(
                f"{name} must be 3-D (n_layers, n_pos, d_model), "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != n_pos:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)} but expected "
                f"position axis == {n_pos}"
            )
