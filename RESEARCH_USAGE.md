# Research usage notes for this fork

This fork supports single-GPU, memory-bounded exact tracing of GemmaScope-2 CLTs through the Phase C2 typed API. The legacy compatibility interfaces were removed.

## Canonical request and result

```python
import torch
from circuit_tracer import (AttributionProblem, ExecutionConstraints, ObservabilityPolicy, ReplacementModel, TraceRequest, TraceSemantics, trace_one)

model = ReplacementModel.from_pretrained("google/gemma-3-1b-pt", "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine", backend="nnsight", dtype=torch.bfloat16, lazy_encoder=True, lazy_decoder=True)
request = TraceRequest(
    problem=AttributionProblem(model=model, prompt="If Alice has 3 apples and buys 2 more, she has", max_n_logits=4),
    semantics=TraceSemantics(source_batch_size=16, max_feature_nodes=128),
    execution=ExecutionConstraints(offload="cpu", observability=ObservabilityPolicy(verbose=True, profile=True, profile_log_interval=1)),
)
result = trace_one(request)
if result.status.value != "succeeded":
    raise RuntimeError(result.telemetry_summary)
graph = result.graph
print(result.semantic_fingerprint, result.execution_fingerprint)
print(result.telemetry_summary)
```

`TraceResult` is the provenance boundary: it carries `status`, `graph` (and `output`), independent semantic and execution fingerprints, structured telemetry, and an admission report when produced. Persist the fingerprints and telemetry; do not infer success from a partially written graph artifact.

## Policy ownership

`AttributionProblem` owns the scientific input: model, prompt, optional `targets`, logit-selection objective, and output position. `TraceSemantics` owns choices that can alter the mathematical result, including source/feature batching semantics, maximum retained feature nodes, sparsification, precision, and frontier membership/ranking.

`ExecutionConstraints` owns physical policy: session capacity and microbatches, row storage and staging, replay windows, execution mechanisms, offload, decoder-cache budgeting, and observability. Execution-only changes get a distinct execution fingerprint while preserving the semantic fingerprint, enabling comparisons across hardware and memory policies without conflating them with a changed scientific trace.

`ObservabilityPolicy`, nested in `ExecutionConstraints`, controls verbose rendering, profiling, bounded event capture, JSONL telemetry paths, and provenance/debug capture. It does not define scientific meaning.

## Targets and sessions

Pass targets through `AttributionProblem(targets=...)`: `None` selects salient logits, a sequence of strings selects token targets, a tensor selects token IDs, and `CustomTarget` / `TargetSpec` values select custom residual directions.

```python
from circuit_tracer import SessionWindow, open_session
session = open_session(request, window=SessionWindow(max_prefix_len=64))
try:
    # Submit the session's typed prefix operations here.
    pass
finally:
    session.close()
```

## GemmaScope-2 operational guidance

Use `backend="nnsight"`, `dtype=torch.bfloat16`, and lazy encoder/decoder loading. The runtime chooses exact chunked decoder handling; callers express only problem semantics and physical constraints. Start with small `TraceSemantics.source_batch_size` and conservative `max_feature_nodes` for smoke work. Adjust `ExecutionConstraints` rather than silently changing the scientific request when capacity is constrained.

```python
from circuit_tracer import DecoderCachePolicy, ExecutionConstraints, SessionPlan
execution = ExecutionConstraints(session=SessionPlan(decoder_cache=DecoderCachePolicy(enabled=True, max_bytes=2 * 1024**3)))
```

The decoder cache is physical session policy. A diagnostic feature cap changes semantics and is appropriate only for explicitly diagnostic runs. Run GPU/model work in scheduled jobs and retain the typed telemetry with the job record.
