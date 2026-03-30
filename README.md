# circuit-tracer

This library implements tools for finding circuits using features from (cross-layer) MLP transcoders, as originally introduced by [Ameisen et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [Lindsey et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

Our library performs three main tasks. 
1. Given a model with pre-trained transcoders, it finds the circuit / attribution graph; i.e., it computes the direct effect that each non-zero transcoder feature, transcoder error node, and input token has on each other non-zero transcoder feature and output logit.
2. Given an attribution graph, it visualizes this graph and allows you to annotate these features.
3. Enables interventions on a model's transcoder features using the insights gained from the attribution graph; i.e. you can set features to arbitrary values, and observe how model output changes.

## Getting Started
One quick way to start is to try our [tutorial notebook](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb)! 

If you are using this fork specifically for memory-bounded GemmaScope-2 CLT tracing, see [RESEARCH_USAGE.md](RESEARCH_USAGE.md).

You can also find circuits and visualize them in one of three ways:
1. Use `circuit-tracer` on [Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C23_12237_10%2C20_15589_9%2C16_25_9%2C14_2268_9%2C18_8959_10%2C4_13154_9%2C7_6861_9%2C19_1445_10%2CE_2329_7%2CE_6037_4%2C0_13727_7%2C6_4012_7%2C17_7178_10%2C15_4494_4%2C6_4662_4%2C4_7671_4%2C3_13984_4%2C1_1000_4%2C19_7477_9%2C18_6101_10%2C16_4298_10%2C7_691_10&supernodes=%5B%5B%22state%22%2C%226_4012_7%22%2C%220_13727_7%22%5D%2C%5B%22preposition+followed+by+place+name%22%2C%2219_1445_10%22%2C%2218_6101_10%22%5D%2C%5B%22Texas%22%2C%2220_15589_10%22%2C%2220_15589_9%22%2C%2219_7477_9%22%2C%2216_25_9%22%2C%224_13154_9%22%2C%2214_2268_9%22%2C%227_6861_9%22%5D%2C%5B%22capital+%2F+capital+cities%22%2C%2215_4494_4%22%2C%226_4662_4%22%2C%224_7671_4%22%2C%223_13984_4%22%2C%221_1000_4%22%2C%2221_5943_10%22%2C%2217_7178_10%22%2C%227_691_10%22%2C%2216_4298_10%22%5D%5D&pruningThreshold=0.6&clickedId=21_5943_10&densityThreshold=0.99) - no installation required! Just click on `+ New Graph` to create your own, or use the drop-down menu to select an existing graph.
2. Run `circuit-tracer` via a Python script or Jupyter notebook. Start with our [tutorial notebook](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb). This will work on Colab with the GPU resources provided for free by default - just click on the Colab badge! Check out the **Demos** section below for more tutorials. You can also run these demo notebooks locally, with your own compute.
3. Run `circuit-tracer` via the command-line interface. This can only be done with your own compute. For more on how to do that, see **Command-Line Interface**. 

Working with Gemma-2 (2B) is possible with relatively limited GPU resources; Colab GPUs have 15GB of RAM. More GPU RAM will allow you to do less offloading, and to use a larger batch size. 

Currently, intervening on models with respect to the transcoder features you discover in your graphs is possible both when using `circuit-tracer` in a script or notebook, or on Neuronpedia for Gemma-2 (2B). To perform interventions on Neuronpedia, ensure at least one node is pinned, then click "Steer" in the subgraph.

### Installation
We recommend using [`uv`](https://docs.astral.sh/uv/) for local development and reproducible environments.

```bash
uv sync
```

For editable/dev installs, use:

```bash
uv sync --extra dev
```

### Demos
We include some demos showing how to use our library in the `demos` folder. The main demo is [`demos/circuit_tracing_tutorial.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb), which replicates two of the findings from [this paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) using Gemma 2 (2B). All demos except for the Llama demo can be run on Colab.

We also make two simple demos of attribution and intervention available, for those who want to learn more about how to use the library:
- [`demos/attribute_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/attribute_demo.ipynb): Demonstrates how to find circuits and visualize them. 
- [`demos/attribution_targets_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/attribution_targets_demo.ipynb): Demonstrates how to find circuits by specifying attribution targets, i.e. specific logits (or related quantities) that you wish to attribute from. 
- [`demos/intervention_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/intervention_demo.ipynb): Demonstrates how to perform interventions on models. 

We finally provide demos that dig deeper into specific, pre-computed and pre-annotated attribution graphs, performing interventions to demonstrate the correctness of the annotated graph:
- [`demos/gemma_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb): Explores graphs from Gemma 2 (2B).
- [`demos/gemma_it_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_it_demo.ipynb): Explores graphs from instruction-tuned Gemma 2 (2B), using transcoders from the base model.
- [`demos/llama_demo.ipynb`](https://github.com/safety-research/circuit-tracer/blob/main/demos/llama_demo.ipynb): Explores graphs from Llama 3.2 (1B). Not supported on Colab.

We also provide a number of annotated attribution graphs for both models, which can be found at the top of their two demo notebooks.

## Usage
### Available Transcoders
The following transcoders are available for use with `circuit-tracer`; this means that the transcoder weights and features are both available (so features will load properly when you run the visualization server). You can use the HuggingFace repo name (e.g. `mntss/gemma-scope-transcoders`) as the `transcoders` argument of `ReplacementModel.from_pretrained`, or as the argument of `--transcoder_set` in the CLI. 
- Gemma-2 (2B): [PLTs](https://huggingface.co/mntss/gemma-scope-transcoders) (originally from [GemmaScope](https://huggingface.co/google/gemma-scope)) and CLTs with 2 feature counts: [426K](https://huggingface.co/mntss/clt-gemma-2-2b-426k) and [2.5M](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M)
- Llama-3.2 (1B): [PLTs](https://huggingface.co/mntss/transcoder-Llama-3.2-1B) and [CLTs](https://huggingface.co/mntss/clt-llama-3.2-1b-524k)
- Qwen-3 PLTs: for Qwen-3 [0.6B](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0), [1.7B](https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0), [4B](https://huggingface.co/mwhanna/qwen3-4b-transcoders), [8B](https://huggingface.co/mwhanna/qwen3-8b-transcoders), and [14B](https://huggingface.co/mwhanna/qwen3-14b-transcoders-lowl0)
- [GPT-OSS (20B) CLT](https://huggingface.co/mntss/clt-131k)
- Gemma-3 PLTs (originally from [GemmaScope-2](https://huggingface.co/google/gemma-scope-2)) can be found [here for models of size 270M, 1B, 4B, 12B, and 27B, PT and IT](https://huggingface.co/collections/mwhanna/gemma-scope-2-transcoders-circuit-tracer). These require using the `nnsight` backend.

### GemmaScope-2 CLT usage in this fork
This fork adds a memory-bounded, exact tracing path for GemmaScope-2 cross-layer transcoders (CLTs) on a single GPU.

- `attribute(...)` and `ReplacementModel.from_pretrained(...)` remain backwards compatible at the call site.
- For GemmaScope-2 CLTs, exact chunked decoder handling is enabled automatically.
- The exact chunked NNSight path now stages large run-scoped attribution tensors more aggressively and accumulates attribution scores in `float32` for better split-batch numerical stability.
- By default, exact chunked `setup_attribution(...)` now retains only last-token logits; use `retain_full_logits=True` only if you explicitly need the full sequence logits from that internal setup call.
- Optional double-pass sparsification can now screen candidates before reconstruction and reuse the same retained set during later attribution.
- GemmaScope-2 CLTs should be used with `backend="nnsight"`.
- The loader also tolerates the duplicated final shard path present in some GemmaScope-2 configs.

No activation flag is required for the new VRAM policy changes: if you are using a GemmaScope-2 CLT with `backend="nnsight"`, the memory-saving path is already active.

Recommended single-GPU starting point for GemmaScope-2 CLTs:

```python
import torch

from circuit_tracer import ReplacementModel
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

Optional split-batch tuning for the exact chunked path:

```python
graph = attribute(
    "If Alice has 3 apples and buys 2 more, she has",
    model,
    max_n_logits=4,
    batch_size=16,
    feature_batch_size=8,
    logit_batch_size=4,
    max_feature_nodes=128,
    offload="cpu",
)
```

- `batch_size` still controls the main trace width and remains the default for both phases.
- `feature_batch_size` optionally shrinks only Phase 4 feature batches.
- `logit_batch_size` optionally shrinks only Phase 3 logit batches.
- You only need these new knobs when tuning memory/runtime tradeoffs; existing calls keep working unchanged.

Phase 0 stats-only helper:

```python
from circuit_tracer import attribute_phase0_stats

phase0_stats = attribute_phase0_stats(
    "If Alice has 3 apples and buys 2 more, she has",
    model,
)

print(phase0_stats)
```

This runs only the setup / Phase 0 portion of the pipeline and returns a compact dict:

```python
{
    "token_count": 10,
    "prompt_token_count": 10,
    "total_active_features": 123456,
    "active_features_by_layer": [...],
    "active_features_by_token": [...],
    "phase0_encode_seconds": 1.23,
    "phase0_reconstruction_seconds": 4.56,
}
```

Use this when you want prompt-level scaling/count analysis without running Phases 1-4 or building a full attribution graph.

Optional Phase 4 cross-batch decoder cache:

- available for the exact chunked GemmaScope-2 CLT path
- **disabled by default**
- requires an explicit `cross_batch_decoder_cache_bytes` budget
- intended for repeated-batch Phase 4 runs where decoder reloads dominate
- once enabled, it remains active for the full attribution run unless you explicitly reset/clear it

Script example:

```python
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt
from circuit_tracer.utils.hf_utils import resolve_transcoder_paths

config = {
    "repo_id": "mwhanna/gemma-scope-2-1b-pt",
    "subfolder": "clt/width_262k_l0_medium_affine",
    "scan": "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine",
    "feature_input_hook": "hook_resid_mid",
    "feature_output_hook": "hook_mlp_out",
}

transcoders = load_gemma_scope_2_clt(
    resolve_transcoder_paths(config),
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    lazy_encoder=True,
    lazy_decoder=True,
    decoder_chunk_size=1024,
    cross_batch_decoder_cache_bytes=2 * 1024**3,
)

model = ReplacementModel.from_pretrained_and_transcoders(
    "google/gemma-3-1b-pt",
    transcoders,
    backend="nnsight",
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
```

If you maintain a transcoder `config.yaml`, you can also set:

```yaml
cross_batch_decoder_cache_bytes: 2147483648
```

See [RESEARCH_USAGE.md](RESEARCH_USAGE.md) for operational guidance and profiling notes.

Optional early sparsification example:

```python
from circuit_tracer import SparsificationConfig

graph = attribute(
    "If Alice has 3 apples and buys 2 more, she has",
    model,
    max_n_logits=4,
    batch_size=16,
    max_feature_nodes=128,
    sparsification=SparsificationConfig(
        per_layer_position_topk=4,
        global_cap=512,
    ),
    offload="cpu",
    verbose=True,
)
```

Operational notes for this forked path:

- `lazy_encoder=True` and `lazy_decoder=True` are recommended for GemmaScope-2 CLTs.
- `offload="cpu"` or `offload="disk"` can still help for model components, but transcoder offload is intentionally skipped during exact chunked decoder attribution so decoder slices remain readable during backward scoring.
- With `verbose=True`, phase-level runtime and memory telemetry is emitted to logs (RSS plus CUDA allocated/reserved where available), which is useful for SLURM debugging.
- The exact chunked path now keeps only last-token logits by default during `setup_attribution(...)`, stages `encoder_vecs`/`error_vectors` more conservatively, avoids the previous `torch.cat(...)` encoder-vector peak during Phase 0, and cleans up run-scoped attribution buffers/caches during teardown automatically.
- `SparsificationConfig(per_layer_position_topk=..., global_cap=...)` uses a per-layer-per-position activation screen first, then an optional global cap as a safety valve.
- For deeper profiling, `attribute(..., profile=True, profile_log_interval=1)` emits setup/precompute diagnostics, live `TRACE ...` progress lines for long-running work, batch-level diagnostics including decoder load counts/timing and chunked attribution timing, and sparsification retention summaries when sparsification is enabled.
- When the cross-batch decoder cache is enabled, profiling also reports decoder cache hits, misses, evictions, and resident bytes.
- For scaling experiments only, `attribute(..., diagnostic_feature_cap=K)` applies a debug-only early active-feature cap before attribution rows are computed. This changes semantics and should not be used for final scientific traces.
- `create_graph_files(...)` now accepts `prune_device=...` if you want pruning to happen on a specific device explicitly.


### Choosing a Backend
By default, `circuit-tracer` creates a `ReplacementModel` that inherits from the `TransformerLens` `HookedTransformer` class. However, `TransformerLens` does not support all HuggingFace models; it only supports those implemented in `TransformerLens`. 

Creating a `ReplacementModel` with `backend='nnsight'` will create an `nnsight`-backed `ReplacementModel` that inherits from its `LanguageModel` class; this supports most HuggingFace models. That is, you can create an `nnsight` `ReplacementModel` using `ReplacementModel.from_pretrained(model_name, backend='nnsight')`. Note, however, that the `nnsight` backend is still experimental: it is slower and less memory-efficient, and may not provide all of the functionality of the `TransformerLens` version.

### Caching
In order to use the `lazy_decoder` and `lazy_encoder` options on transcoders, they must be stored in `circuit-tracer`-compatible format. While many transcoders have been uploaded in that format to HuggingFace, this requires large amounts of storage. `circuit-tracer` now supports instead creating a local cache of models, by calling e.g.

```python
from circuit_tracer.utils.caching import save_transcoders_to_cache

hf_ref = "mwhanna/gemma-scope-2-27b-pt/transcoder_all/width_262k_l0_small"
cache_dir = '~/.cache/'
save_transcoders_to_cache(hf_ref, cache_dir=cache_dir)
```

You can also empty the cache using `circuit_tracer.utils.caching.empty_cache`.

For GemmaScope-2 CLTs, cached loads keep the fork's exact chunked decoder behavior enabled automatically.

## Command-Line Interface

The unified CLI performs the complete 3-step process for finding and visualizing circuits:

### 3-Step Process
1. **Attribution**: Runs the attribution algorithm to find the circuit/attribution graph, computing direct effects between transcoder features, error nodes, tokens, and output logits.
2. **Graph File Creation**: Prunes the attribution graph to remove low-effect nodes and edges, then converts it to JSON format suitable for visualization.
3. **Local Server**: Starts a local web server to visualize and interact with the graph in your browser.

### Basic Usage
To find a circuit, create the graph files, and start up a local server, use the command:

```
circuit-tracer attribute --prompt [prompt] --transcoder_set [transcoder_set] --slug [slug] --graph_file_dir [directory] --slug [slug] --graph_file_dir [graph_file_dir] --server
```

It will tell you where the server is serving (something like `localhost:[port]`). If you run this command on a remote machine, make sure to enable port forwarding, so you can see the graphs locally!

### Mandatory Arguments
**Attribution**
- `--prompt` (`-p`): The input prompt to analyze
- `--transcoder_set` (`-t`): The set of transcoders to use for attribution. Options:
  - HuggingFace repository ID (e.g., `mntss/gemma-scope-transcoders`, `username/repo-name@revision`)
  - Convenience shortcuts: `gemma` (GemmaScope transcoders) or `llama` (ReLU transcoders)

**Graph File Creation**

These are required if you want to run a local web server for visualization:
- `--slug`: A name/identifier for your analysis run
- `--graph_file_dir`: Directory path where JSON graph files will be saved

You can also save the raw attribution graph (to be loaded and used in Python later):
- `--graph_output_path` (`-o`): Path to save the raw attribution graph (`.pt` file)

You must set `--slug` and `--graph_file_dir`, or `--graph_output_path`, or both! Otherwise the CLI will output nothing.

**Local Server**
- `--server`: Start a local web server for graph visualization

### Optional Arguments

**Attribution Parameters:**
- `--model` (`-m`): Model architecture (auto-inferred for `gemma` and `llama` presets)
- `--max_n_logits` (default: 10): Maximum number of logit nodes to attribute from
- `--desired_logit_prob` (default: 0.95): Cumulative probability threshold for top logits
- `--batch_size` (default: 256): Batch size for backward passes
- `--max_feature_nodes`: Maximum number of feature nodes (defaults to 7500)
- `--dtype`: Datatype in which to load the model / transcoders (allowed: `float32/fp32`, `float16/fp16`, `bfloat16/bf16`)
- `--offload`: Memory optimization option (`cpu`, `disk`, or `None`)
- `--verbose`: Display detailed progress information
- `--profile`: Emit batch-level diagnostic profiling logs
- `--profile-log-interval`: Log every N batches when profiling
- `--diagnostic-feature-cap`: Debug-only early active-feature cap for profiling/scaling experiments
- `--no-lazy-decoder`: Disable lazy decoder loading for comparison/profiling runs

For GemmaScope-2 CLTs in this fork, `--verbose` is especially useful because it emits phase-level timing and memory telemetry to the console / SLURM logs.

**Graph Pruning Parameters:**
- `--node_threshold` (default: 0.8): Keeps minimum nodes with cumulative influence ≥ threshold
- `--edge_threshold` (default: 0.98): Keeps minimum edges with cumulative influence ≥ threshold

**Server Parameters:**
- `--port` (default: 8041): Port for the local server

### Examples

**Complete workflow with visualization:**
```
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set gemma \
  --slug gemma-demo \
  --graph_file_dir ./graph_files \
  --server
```

**Attribution only (save raw graph):**
```
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set llama \
  --graph_output_path france_capital.pt
```

### Graph Annotation
When using the `--server` option, your browser will open to a local visualization interface. The interface is the same as in [the original papers](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (frontend available [here](https://github.com/anthropics/attribution-graphs-frontend)).
- **Select a node**: Click on a node.
- **Pin / unpin a node to subgraph pane**: Ctrl+click/Commmand+click the node.
- **Annotate a node**: Click on the "Edit" button on the right side of the window while a node is selected to edit its annotation.
- **Group nodes**: Hold G and click on nodes to group them together into a supernode. Hold G and click on the x next to a supernode to ungroup all of them.
- **Annotate supernode / node group**: click on the label below the supernode to edit the supernode annotation.

## Cite
You can cite this library as follows:
```
@misc{circuit-tracer,
  author = {Hanna, Michael and Piotrowski, Mateusz and Lindsey, Jack and Ameisen, Emmanuel},
  title = {circuit-tracer},
  howpublished = {\url{https://github.com/decoderesearch/circuit-tracer}},
  note = {The first two authors contributed equally and are listed alphabetically.},
  year = {2025}
}
```
or cite the paper [here](https://aclanthology.org/2025.blackboxnlp-1.14/).
