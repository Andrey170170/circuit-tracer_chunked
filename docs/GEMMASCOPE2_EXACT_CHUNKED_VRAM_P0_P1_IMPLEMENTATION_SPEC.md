# GemmaScope-2 exact chunked VRAM P0/P1 implementation spec

## 1. Problem statement

GemmaScope-2 CLT attribution through the exact chunked decoder path (`backend="nnsight"`) is functional, but single-GPU VRAM is still tight enough to limit routine use. This spec turns the current memory analysis into an implementation plan for the lowest-risk fixes first.

Goal: reduce peak CUDA memory without changing attribution semantics, public call sites, or graph outputs.

Primary touchpoints:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/transcoder/cross_layer_transcoder.py`
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`

Implementation preference:

- use pinned CPU memory for staged tensors where copies are frequent
- use `non_blocking=True` for CPU↔GPU transfers when tensor lifetime and stream ordering are already explicit
- keep transfer points phase-boundary driven, not ad hoc

## 2. Scope / non-goals

### In scope

- Exact chunked GemmaScope-2 CLT runs on a single GPU.
- P0 and P1 memory reductions only.
- Runtime telemetry that proves where VRAM is spent.
- Teardown/cleanup for run-scoped decoder cache and attribution buffers.

### Non-goals

- Changing attribution math or graph schema.
- Broad offload redesign or whole-model CPU migration.
- Multi-GPU support.
- Generalizing the plan to non-GemmaScope-2 backends.

## 3. Proposed approach

### P0 — low-risk, should land first

#### P0.1 CPU-stage `encoder_vecs` in exact chunked mode

- Rationale: `encoder_vecs` is one of the longest-lived GPU residents and does not need full-device residency between chunked uses.
- Main code changes:
  - stage encoder vectors into pinned CPU storage after setup/reconstruction
  - materialize only the active layer/chunk slice on GPU for the current step
  - add explicit transfer helpers in `context_nnsight.py`
- Invariants / correctness constraints:
  - values must remain bitwise-equivalent after round-trip staging for the same dtype/device path
  - transfer order must preserve chunk evaluation order
  - staged tensors must not be mutated while resident on CPU
- Expected VRAM impact: high; should lower steady-state resident memory across the attribution loop.
- Expected performance impact: small copy overhead, usually offset by better VRAM headroom and fewer OOM retries.
- Suggested tests:
  - small exact run with encoder staging on/off and identical outputs
  - memory snapshot before/after setup and before/after Phase 4 batches
  - transfer ordering test for a multi-chunk prompt

Likely file touchpoints:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/transcoder/cross_layer_transcoder.py` (if staging helpers live with chunk logic)

#### P0.2 Retain only last-token logits

- Rationale: most exact attribution runs only consume the final-position logits; keeping the full tensor is avoidable VRAM pressure.
- Main code changes:
  - store only last-token logits on the hot path when the target is final-position attribution
  - keep full logits behind an explicit opt-in path for consumers that need them
  - separate “logit source” from “logit retention” in context bookkeeping
- Invariants / correctness constraints:
  - exact outputs must match the full-logits path for last-token targets
  - full-logit retention must still be available when required by call-site intent
  - no change to graph schema or node ordering
- Expected VRAM impact: high for common last-token workflows.
- Expected performance impact: small positive to neutral; less tensor residency and reduced book-keeping.
- Suggested tests:
  - last-token attribution equality with full-logits fallback
  - non-last-token path still retains full logits when explicitly requested
  - phase-boundary memory comparison

Likely file touchpoints:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/attribution/attribute_nnsight.py`

#### P0.3 Release phase-0 temporaries earlier / reduce phase-0 overlap

- Rationale: setup tensors often overlap with early attribution tensors longer than necessary, inflating peak CUDA allocated/reserved.
- Main code changes:
  - insert explicit release points at the end of Phase 0 and after materialization handoffs
  - reduce scope of setup-local temporaries in `attribute_nnsight.py`
  - clear references in context objects immediately after replacement state is live
- Invariants / correctness constraints:
  - release must happen only after downstream tensors are fully materialized
  - no tensor needed for backward scoring may be freed early
  - repeated runs must not leak stale references into the next prompt
- Expected VRAM impact: medium to high, especially on prompts with large setup footprints.
- Expected performance impact: neutral or slightly positive due to lower peak pressure and fewer allocator stalls.
- Suggested tests:
  - compare cuda_alloc/cuda_reserved right before and after phase handoff
  - run with verbose/profile logs and verify early-release markers
  - smoke test repeated prompts to catch stale references

Likely file touchpoints:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/attribution/context_nnsight.py`

#### P0.4 Decoder-cache usefulness guardrails

- Rationale: a small cache budget can churn without helping; it should not consume VRAM by default when reuse is poor.
- Main code changes:
  - keep cache opt-in and explicit
  - track hit/miss/eviction/resident-bytes telemetry
  - add a simple guardrail that warns and disables cache when reuse is ineffective for the run
- Invariants / correctness constraints:
  - cache on/off must not change outputs
  - guardrail must not trigger on noisy startup alone; it should use observed run telemetry
  - cache teardown must still occur even after auto-disable
- Expected VRAM impact: medium when cache churn is currently wasting residency.
- Expected performance impact: positive when cache is ineffective; neutral when disabled.
- Suggested tests:
  - cache-enabled run with near-zero hits should warn and disable
  - cache-enabled run with useful reuse should remain enabled
  - teardown clears resident cache bytes

Likely file touchpoints:

- `circuit_tracer/transcoder/cross_layer_transcoder.py`
- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`

#### P0.5 Explicit teardown cleanup of attribution context

- Rationale: run-scoped tensors and decoder state should not survive prompt teardown or accumulate across repeated runs.
- Main code changes:
  - add explicit cleanup entry points in attribution context and transcoder cache
  - clear staged CPU/GPU tensors, cache state, and phase-local buffers deterministically
  - make cleanup idempotent
- Invariants / correctness constraints:
  - cleanup must not affect already-computed outputs
  - repeated teardown calls must be safe
  - next run must start from a clean context
- Expected VRAM impact: medium; prevents “sticky” residency across prompts.
- Expected performance impact: neutral; may slightly improve allocator stability in long sessions.
- Suggested tests:
  - run two prompts back-to-back and verify no leftover resident cache/buffers
  - call teardown twice and verify no error
  - inspect final cuda_alloc/cuda_reserved after teardown

Likely file touchpoints:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/transcoder/cross_layer_transcoder.py`
- `circuit_tracer/attribution/attribute_nnsight.py`

### P1 — moderate refactor, after P0 is stable

#### P1.1 CPU-stage and prefetch `error_vectors` by layer

- Rationale: error vectors are large, layer-shaped working state; they do not need to sit on GPU all at once.
- Main code changes:
  - stage error vectors in pinned CPU memory
  - prefetch the next layer slice to GPU only when the current slice is nearly consumed
  - keep prefetch explicit and bounded by a small lookahead window
- Invariants / correctness constraints:
  - layer order must stay exact
  - prefetch must not change values or attribution outputs
  - peak residency must be bounded by the configured window
- Expected VRAM impact: high if error vectors are currently a major long-lived resident.
- Expected performance impact: neutral to slightly positive; better overlap if prefetch is well-tuned.
- Suggested tests:
  - compare outputs with prefetch on/off
  - verify slice residency never exceeds the configured window
  - benchmark a small lookahead versus no lookahead

Likely file touchpoints:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/attribution/attribute_nnsight.py`

#### P1.2 Split `feature_batch_size` from `logit_batch_size`

- Rationale: feature attribution and logit attribution have different memory footprints, so one batch size is too blunt.
- Main code changes:
  - add separate knobs in the attribution configuration path
  - preserve existing behavior by defaulting both values to the current single batch size
  - thread the two values to the relevant phase-specific loops
- Invariants / correctness constraints:
  - default path must behave exactly like today
  - both batch sizes must preserve the same numerical result as a single batch size would
  - no new required argument at call sites
- Expected VRAM impact: medium; allows the hottest phase to shrink independently.
- Expected performance impact: mixed; better memory headroom may cost or save time depending on the chosen split.
- Suggested tests:
  - default config reproduces current behavior
  - feature/logit split produces identical outputs
  - benchmark memory/time sensitivity across a few splits

Likely file touchpoints:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`

#### P1.3 Bound chunked feature grad residency

- Rationale: chunked feature-grad/replay buffers should stay small and explicit; unbounded residency defeats the chunking work.
- Main code changes:
  - define a fixed replay window size for chunked feature grad work
  - release each chunk immediately after use
  - avoid any full-batch materialization of replay buffers
- Invariants / correctness constraints:
  - per-chunk results must stitch together exactly
  - residency limits must be enforced even under large batches
  - no implicit fallback to dense full-batch storage
- Expected VRAM impact: medium to high for long attribution runs.
- Expected performance impact: neutral to slightly positive if smaller resident buffers reduce allocator churn.
- Suggested tests:
  - large-batch smoke test with bounded chunk residency
  - compare chunked vs non-chunked outputs on a small case
  - verify peak memory scales with configured window, not batch shape

Likely file touchpoints:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/attribution/attribute_nnsight.py`

#### P1.4 Diagnostics and profiling updates

- Rationale: after memory changes land, the logs need to prove where VRAM moved and whether the tradeoffs are worth it.
- Main code changes:
  - log cuda_alloc/cuda_reserved at key phase boundaries
  - add before/after markers for setup, phase handoff, batch start/end, and teardown
  - include numerical consistency checks in profile or test-mode runs
- Invariants / correctness constraints:
  - diagnostics must be optional and low overhead when disabled
  - logging must not change outputs or tensor lifetimes
  - metrics should be comparable across runs with the same prompt/config
- Expected VRAM impact: indirect only; improves visibility, not memory by itself.
- Expected performance impact: small overhead in verbose/profile mode only.
- Suggested tests:
  - confirm logs emit phase-boundary cuda_alloc/cuda_reserved values
  - compare numerical outputs across the old and new paths
  - verify profile logging can be toggled off cleanly

Likely file touchpoints:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/transcoder/cross_layer_transcoder.py`

### Module-level implementation notes

- `attribute_nnsight.py`
  - own phase boundaries, cleanup ordering, and telemetry emission
  - release phase-0 buffers as soon as the next phase can proceed

- `context_nnsight.py`
  - manage active GPU slices for `encoder_vecs`, `error_vectors`, and replay buffers
  - record per-phase memory and batch-level timing

- `cross_layer_transcoder.py`
  - manage cache budget, hit/miss/eviction accounting, and deterministic teardown
  - support cache-disable fallback when reuse is ineffective

- `replacement_model_nnsight.py`
  - keep loader/API behavior stable while exposing the new memory policy

## 4. Rollout guidance

1. Land all P0 items first.
2. Benchmark before/after on representative prompts and batch sizes.
3. If P0 is stable, land P1.1 through P1.4 incrementally, not as one large change.
4. Re-run the same benchmarks after each P1 item to isolate the effect.
5. Keep any performance regressions or numerical drift blocking until explained.

## 5. Validation guidance

Use the same prompt/config pair before and after each change.

Measure:

- `cuda_alloc` and `cuda_reserved` immediately before and after Phase 0
- the same metrics at the Phase 0 → Phase 1 handoff
- the same metrics before and after the hot Phase 4 batch loop
- final teardown memory after cleanup

Check:

- exact numerical consistency for logits and attribution outputs on a small deterministic prompt
- unchanged graph outputs and node ordering
- no cache/state residue after teardown
- bounded peak memory under repeated runs

## 6. Acceptance criteria

This work is complete when all of the following hold:

- Peak CUDA allocated/reserved memory drops in representative GemmaScope-2 exact chunked runs.
- Graph outputs are unchanged for the same prompt/config.
- `attribute(...)` and `ReplacementModel.from_pretrained(...)` remain callable without new required arguments.
- Decoder cache is either measurably useful or auto-disabled/warned when it churns.
- Teardown leaves no run-scoped cache or attribution buffer resident.
- Verbose/profile logs clearly show phase timing and memory residency.

## 7. Risks and open questions

- How far can `encoder_vecs` and `error_vectors` be staged before copy overhead offsets the VRAM win?
- What cache hit rate is high enough to justify keeping the cross-batch decoder cache enabled?
- Will separating feature and logit batch sizing complicate the public API enough to need a named config object?
- Are any tensors shared across phases tightly enough that early release would break exactness or teardown?

## 8. Assumptions

- Exact chunked GemmaScope-2 CLT tracing remains the target path.
- `decoder_chunk_size=1024` stays the baseline unless measurements prove otherwise.
- `lazy_encoder=True` and `lazy_decoder=True` remain the recommended defaults for this fork.
- P0 should be safe to implement without changing scientific behavior.

## 9. Implementation status (Mar 2026)

Implemented in this fork:

- exact chunked `setup_attribution(...)` now retains last-token logits by default, with `retain_full_logits=True` available as an opt-in for internal callers that need full logits
- exact chunked attribution stages large run-scoped tensors more conservatively and tears them down explicitly after each run
- attribution score accumulation now uses `float32` buffers to avoid split-batch drift from `bfloat16` accumulation order
- `attribute(...)` now accepts optional `feature_batch_size` and `logit_batch_size` overrides for Phase 4 and Phase 3 microbatching without introducing new required arguments
- decoder cache guardrails, extra phase-boundary memory logging, and SLURM validation coverage were added

Activation / user-facing behavior:

- no new required flags are needed for the GemmaScope-2 exact chunked path; the memory-policy changes activate automatically when using GemmaScope-2 CLTs with `backend="nnsight"`
- existing `attribute(...)` call sites continue to work unchanged
- users only need the new split-batch knobs when tuning memory/runtime tradeoffs on the exact chunked path
