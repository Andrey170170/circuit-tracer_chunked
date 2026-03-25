# Research usage notes for this fork

This fork is tuned for **single-GPU, memory-bounded, exact tracing** of **GemmaScope-2 cross-layer transcoders (CLTs)**.

## What changed relative to upstream

- GemmaScope-2 CLTs use an **exact chunked decoder** path automatically.
- Full decoder expansion is avoided during attribution setup.
- Optional double-pass sparsification can screen feature candidates before reconstruction and later attribution.
- Phase 4 partial-influence computation no longer performs an implicit full dense CUDA copy.
- Verbose attribution runs emit phase-level time and memory telemetry.

The repo-facing API is intentionally kept close to upstream:

- `ReplacementModel.from_pretrained(...)` still works as before.
- `attribute(...)` still works as before.
- Existing graph consumers should continue to work with the produced `Graph` objects.

## Recommended usage for GemmaScope-2 CLTs

Use:

- `backend="nnsight"`
- `dtype=torch.bfloat16`
- `lazy_encoder=True`
- `lazy_decoder=True`
- small initial `batch_size`
- conservative `max_feature_nodes` for smoke runs

Example:

```python
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer import SparsificationConfig
from circuit_tracer.attribution.attribute_nnsight import attribute

model = ReplacementModel.from_pretrained(
    "google/gemma-3-1b-pt",
    "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine",
    backend="nnsight",
    dtype=torch.bfloat16,
    lazy_encoder=True,
    lazy_decoder=True,
)

graph = attribute(
    "If Alice has 3 apples and buys 2 more, she has",
    model,
    max_n_logits=4,
    batch_size=16,
    max_feature_nodes=128,
    offload="cpu",
    verbose=True,
)
```

To enable early screening on retained candidates, pass a sparsification config:

```python
graph = attribute(
    prompt,
    model,
    batch_size=16,
    max_feature_nodes=128,
    sparsification=SparsificationConfig(
        per_layer_position_topk=4,
        global_cap=512,
    ),
)
```

This keeps `attribute(...)` backward compatible when sparsification is omitted, while allowing phase 0 reconstruction and later attribution to reuse the same retained candidate set.

## Offload behavior

- Model/component offload still works.
- For the exact GemmaScope-2 chunked path, **transcoder offload is intentionally skipped during attribution**.
- This is required so decoder chunks remain readable during backward scoring.

So if you pass `offload="cpu"` or `offload="disk"`, expect it to help with model memory pressure, but not by moving the active GemmaScope-2 transcoder itself away during attribution.

## Telemetry and logging

With `verbose=True`, attribution logs include:

- per-phase wall time
- process RSS
- CUDA allocated/reserved memory
- peak CUDA allocated/reserved memory
- key matrix shapes

This is intended for SLURM/HPC debugging.

For deeper bottleneck-finding runs, this fork also supports:

- `profile=True`: emit batch-level profiling logs for attribution
- `profile_log_interval=N`: log every N batches while profiling
- `diagnostic_feature_cap=K`: debug-only early active-feature cap for scaling experiments
- `sparsification=SparsificationConfig(...)`: early candidate screening before reconstruction

Example diagnostic comparison:

```python
graph = attribute(
    prompt,
    model,
    batch_size=16,
    max_feature_nodes=128,
    profile=True,
    profile_log_interval=1,
)
```

And to test whether lazy decoder loading is the dominant cost, compare otherwise identical runs with:

- `lazy_decoder=True`
- `lazy_decoder=False`

The profiling logs report, where applicable:

- setup/precompute timing inside `setup_attribution`
- live `TRACE ...` progress lines during long-running setup/reconstruction/chunked attribution work
- per-batch timing in Phases 3 and 4
- `compute_batch` cumulative timing
- per-layer feature/error attribution timing
- decoder load count and decoder load time
- chunk counts and chunked attribution timing by layer
- sparsification candidate counts, per-layer retained counts, and retained activation-mass proxy

## HPC guidance

- Do **not** run large-model or GPU-heavy validation on shared login nodes.
- Use batch jobs for GPU tests and smoke runs.
- A reference script is included at:

```bash
scripts/slurm/test_gemmascope2_chunked_a100.sbatch
```

- Prefer `uv` commands in jobs and local environments.

Example:

```bash
uv run pytest -q tests/test_partial_influences.py tests/test_gemmascope2_chunked.py
```

CLI profiling example:

```bash
uv run circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine" \
  --backend nnsight \
  --dtype bf16 \
  --lazy-encoder \
  --profile \
  --profile-log-interval 1 \
  --no-lazy-decoder \
  --diagnostic-feature-cap 256 \
  --graph_output_path /tmp/debug_graph.pt
```

## Known fork-specific caveats

- The exact chunked path is enabled specifically for **GemmaScope-2 CLTs**.
- Some GemmaScope-2 configs contain a duplicated final shard path; the loader normalizes this automatically.
- The NNSight backend remains more operationally fragile than TransformerLens, so keep an eye on SLURM logs for tracing/runtime issues.

## Validation status in this repo

Validated in this fork with:

- lightweight CPU tests for chunked loading and partial influences
- A100 40GB SLURM smoke test for Gemma-3-1B + GemmaScope-2 CLT exact chunked tracing

If you scale up prompt counts, steps, or feature budgets, increase them gradually and watch the emitted telemetry.
