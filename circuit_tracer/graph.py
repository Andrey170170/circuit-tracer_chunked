"""Graph data structures for attribution results."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import NamedTuple
import warnings

import torch

from circuit_tracer.utils.tl_nnsight_mapping import (
    convert_nnsight_config_to_transformerlens,
    UnifiedConfig,
)
from circuit_tracer.attribution.targets import LogitTarget


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_targets: list[LogitTarget]
    active_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    selected_features: torch.Tensor
    activation_values: torch.Tensor
    logit_probabilities: torch.Tensor
    vocab_size: int
    cfg: UnifiedConfig
    scan: str | list[str] | None
    n_pos: int

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        active_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg,
        selected_features: torch.Tensor,
        activation_values: torch.Tensor,
        logit_targets: list[LogitTarget],
        logit_probabilities: torch.Tensor,
        scan: str | list[str] | None = None,
        vocab_size: int | None = None,
    ):
        """
        A graph object containing the adjacency matrix describing the direct effect of each
        node on each other. Nodes are either non-zero transcoder features, transcoder errors,
        tokens, or logits. They are stored in the order [active_features[0], ...,
        active_features[n-1], error[layer0][position0], error[layer0][position1], ...,
        error[layer l - 1][position t-1], tokens[0], ..., tokens[t-1], logits[top-1 logit],
        ..., logits[top-k logit]].

        Args:
            input_string (str): The input string attributed.
            input_tokens (torch.Tensor): The input tokens attributed.
            active_features (torch.Tensor): A tensor of shape (n_active_features, 3)
                containing the indices (layer, pos, feature_idx) of the non-zero features
                of the model on the given input string.
            adjacency_matrix (torch.Tensor): The adjacency matrix. Organized as
                [active_features, error_nodes, embed_nodes, logit_nodes], where there are
                model.cfg.n_layers * len(input_tokens) error nodes, len(input_tokens) embed
                nodes, len(logit_targets) logit nodes. The rows represent target nodes, while
                columns represent source nodes.
            cfg: The cfg of the model.
            selected_features (torch.Tensor): Indices into active_features for selected nodes.
            activation_values (torch.Tensor): Activation values for selected features.
            logit_targets: List of LogitTarget records describing each logit target.
            logit_probabilities: Tensor of logit target probabilities/weights.
            scan (Union[str,List[str]] | None, optional): The identifier of the
                transcoders used in the graph. Without a scan, the graph cannot be uploaded
                (since we won't know what transcoders were used). Defaults to None
            vocab_size: Vocabulary size. If not provided, defaults to cfg.d_vocab.
        """
        self.logit_targets = logit_targets
        self.logit_probabilities = logit_probabilities
        self.vocab_size = vocab_size if vocab_size is not None else cfg.d_vocab

        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        # Convert cfg to UnifiedConfig (handles both HookedTransformerConfig and NNSight configs)
        self.cfg = convert_nnsight_config_to_transformerlens(cfg)
        self.n_pos = len(input_tokens)
        self.active_features = active_features
        self.input_tokens = input_tokens
        if scan is None:
            print("Graph loaded without scan to identify it. Uploading will not be possible.")
        self.scan = scan
        self.selected_features = selected_features
        self.activation_values = activation_values

    def to(self, device):
        """Send all relevant tensors to the device (cpu, cuda, etc.)

        Args:
            device (_type_): device to send tensors
        """
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.active_features = self.active_features.to(device)
        # logit_targets is list[LogitTarget], no device transfer needed
        self.logit_probabilities = self.logit_probabilities.to(device)

    @property
    def logit_token_ids(self) -> torch.Tensor:
        """Tensor of logit target token IDs.

        Returns token IDs for logit targets on the same device as other graph tensors.

        Returns:
            torch.Tensor: Long tensor of vocabulary indices
        """
        return torch.tensor(
            [target.vocab_idx for target in self.logit_targets],
            dtype=torch.long,
            device=self.logit_probabilities.device,
        )

    @property
    def logit_tokens(self) -> torch.Tensor:
        """Get logit target token IDs tensor (legacy compatibility).

        .. deprecated::
            Use `logit_token_ids` property instead. This is an alias for backward compatibility.

        Raises:
            ValueError: If any targets have virtual indices
        """
        warnings.warn(
            "logit_tokens property is deprecated. Use logit_token_ids property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.logit_token_ids

    def to_pt(self, path: str):
        """Saves the graph at the given path

        Args:
            path (str): The path where the graph will be saved. Should end in .pt
        """
        d = {
            "input_string": self.input_string,
            "adjacency_matrix": self.adjacency_matrix,
            "cfg": self.cfg,
            "active_features": self.active_features,
            "logit_targets": self.logit_targets,
            "logit_probabilities": self.logit_probabilities,
            "vocab_size": self.vocab_size,
            "input_tokens": self.input_tokens,
            "selected_features": self.selected_features,
            "activation_values": self.activation_values,
            "scan": self.scan,
        }
        torch.save(d, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        """Load a graph (saved using graph.to_pt) from a .pt file at the given path.

        Handles backward compatibility with older serialized graphs that stored
        logit_targets as a torch.Tensor of token IDs.

        Args:
            path (str): The path of the Graph to load
            map_location (str, optional): the device to load the graph onto.
                Defaults to 'cpu'.

        Returns:
            Graph: the Graph saved at the specified path
        """
        d = torch.load(path, weights_only=False, map_location=map_location)
        # BC: convert legacy tensor logit_targets to LogitTarget list
        lt = d.get("logit_targets")
        if isinstance(lt, torch.Tensor):
            d["logit_targets"] = [
                LogitTarget(token_str="", vocab_idx=int(idx)) for idx in lt.tolist()
            ]
        return Graph(**d)


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    # Normally we calculate total influence B using A + A^2 + ... or (I - A)^-1 - I,
    # and do logit_weights @ B
    # But it's faster / more efficient to compute logit_weights @ A + logit_weights @ A^2
    # as follows:

    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float):
    # Find score threshold that keeps the desired fraction of total influence
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index: int = int(torch.searchsorted(cumulative_score, threshold).item())
    # make sure we don't go out of bounds (only really happens at threshold=1.0)
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


class PruneResult(NamedTuple):
    node_mask: torch.Tensor  # Boolean tensor indicating which nodes to keep
    edge_mask: torch.Tensor  # Boolean tensor indicating which edges to keep
    cumulative_scores: torch.Tensor  # Tensor of cumulative influence scores for each node


def prune_graph(
    graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98
) -> PruneResult:
    """Prunes a graph by removing nodes and edges with low influence on the output logits.

    Args:
        graph: The graph to prune
        node_threshold: Keep nodes that contribute to this fraction of total influence
        edge_threshold: Keep edges that contribute to this fraction of total influence

    Returns:
        Tuple containing:
        - node_mask: Boolean tensor indicating which nodes to keep
        - edge_mask: Boolean tensor indicating which edges to keep
        - cumulative_scores: Tensor of cumulative influence scores for each node
    """

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    # Extract dimensions
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Calculate node influence and apply threshold
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    # Always keep tokens and logits
    node_mask[-n_logits - n_tokens :] = True

    # Create pruned matrix with selected nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0
    # we could also do iterative pruning here (see below)

    # Calculate edge influence and apply threshold
    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)

    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    old_node_mask = node_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    # Ensure feature nodes have incoming edges
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # iteratively prune until all nodes missing incoming / outgoing edges are gone
    # (each pruning iteration potentially opens up new candidates for pruning)
    # this should not take more than n_layers + 1 iterations
    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        # Ensure feature and error nodes have outgoing edges
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        # Ensure feature nodes have incoming edges
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    # Calculate cumulative influence scores
    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)


def compute_graph_scores(graph: Graph) -> tuple[float, float]:
    """Compute metrics for evaluating how well the graph captures the model's computation.
    This function calculates two complementary scores that measure how much of the model's
    computation flows through interpretable feature nodes versus reconstruction error nodes:
    1. Replacement Score: Measures the fraction of end-to-end influence from input tokens
       to output logits that flows through feature nodes rather than error nodes. This is
       a strict metric that rewards complete explanations where tokens influence logits
       entirely through features.
    2. Completeness Score: Measures the fraction of incoming edges to all nodes (weighted
       by each node's influence on the output) that originate from feature or token nodes
       rather than error nodes. This metric gives partial credit for nodes that are mostly
       explained by features, even if some error influence remains.
    Args:
        graph: The computation graph containing nodes for features, errors, tokens, and logits,
               along with their connections and influence weights.
    Returns:
        tuple[float, float]: A tuple containing:
            - replacement_score: Fraction of token-to-logit influence through features (0-1)
            - completeness_score: Weighted fraction of non-error inputs across all nodes (0-1)
    Note:
        Higher scores indicate better model interpretability, with 1.0 representing perfect
        reconstruction where all computation flows through interpretable features. Lower
        scores indicate more reliance on error nodes, suggesting incomplete feature coverage.
    """
    n_logits = len(graph.logit_targets)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers
    token_end = error_end + n_tokens

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    non_error_fractions = 1 - normalized_matrix[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()


def compute_partial_influences(
    edge_matrix: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    max_iter: int = 128,
    device=None,
    row_chunk_size: int = 4096,
):
    """Compute partial influences using power iteration method.

    This function calculates the influence of each node on the output logits
    based on the edge weights in the graph.

    Args:
        edge_matrix: The edge weight matrix.
        logit_p: The logit probabilities.
        row_to_node_index: Mapping from row indices to node indices.
        max_iter: Maximum number of iterations for convergence.
        device: Device to perform computation on.

    Returns:
        torch.Tensor: Influence values for each node.

    Raises:
        RuntimeError: If computation fails to converge within max_iter.
    """
    device = device or edge_matrix.device
    working_matrix = edge_matrix if edge_matrix.device == device else edge_matrix.to(device)
    working_row_index = (
        row_to_node_index if row_to_node_index.device == device else row_to_node_index.to(device)
    ).long()
    working_logit_p = logit_p if logit_p.device == device else logit_p.to(device)

    influences = torch.zeros(working_matrix.shape[1], device=device, dtype=working_matrix.dtype)
    prod = torch.zeros(working_matrix.shape[1], device=device, dtype=working_matrix.dtype)
    prod[-len(working_logit_p) :] = working_logit_p.to(dtype=working_matrix.dtype)

    for _ in range(max_iter):
        next_prod = torch.zeros_like(prod)
        for start in range(0, working_matrix.shape[0], row_chunk_size):
            end = min(start + row_chunk_size, working_matrix.shape[0])
            chunk = working_matrix[start:end].abs()
            if not chunk.numel():
                continue

            row_weights = prod[working_row_index[start:end]]
            if not row_weights.any():
                continue

            chunk /= chunk.sum(dim=1, keepdim=True).clamp(min=1e-8)
            next_prod += row_weights @ chunk

        prod = next_prod
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def compute_partial_feature_influences(
    feature_edge_matrix: torch.Tensor,
    row_abs_sums: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    *,
    n_feature_nodes: int,
    n_logits: int,
    max_iter: int = 128,
    device=None,
    row_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute feature-only partial influences from compact row storage.

    This is mathematically equivalent to ``compute_partial_influences(...)[ :n_feature_nodes ]``
    when:
      - rows correspond to logit + feature source nodes,
      - only feature-column edges are materialized, and
      - ``row_abs_sums`` stores exact L1 row sums over *all* original columns.

    Args:
        feature_edge_matrix: Dense matrix of shape ``(n_rows, n_feature_nodes)`` containing
            feature-column edge values for each stored row.
        row_abs_sums: Exact absolute row sums over the original full edge rows
            (shape ``(n_rows,)``). These preserve normalization semantics even when
            non-feature columns are not materialized.
        logit_p: Logit probabilities / weights for the first ``n_logits`` rows.
        row_to_node_index: Mapping from row index to original node index.
        n_feature_nodes: Number of active feature columns in the original graph.
        n_logits: Number of logit rows at the start of the stored rows.
        max_iter: Maximum number of power-iteration steps.
        device: Device to run computation on.
        row_chunk_size: Row chunk size for batched matrix-vector products.

    Returns:
        Tensor of shape ``(n_feature_nodes,)`` with partial influence values.

    Raises:
        ValueError: If inputs are inconsistent.
        RuntimeError: If computation fails to converge within ``max_iter``.
    """

    if n_feature_nodes < 0:
        raise ValueError("n_feature_nodes must be >= 0")
    if n_logits < 0:
        raise ValueError("n_logits must be >= 0")
    if feature_edge_matrix.ndim != 2:
        raise ValueError("feature_edge_matrix must be rank-2")

    n_rows, feature_cols = feature_edge_matrix.shape
    if feature_cols != n_feature_nodes:
        raise ValueError(
            "feature_edge_matrix second dimension must equal n_feature_nodes "
            f"({feature_cols} != {n_feature_nodes})"
        )
    if row_abs_sums.numel() != n_rows:
        raise ValueError("row_abs_sums length must equal number of rows")
    if row_to_node_index.numel() != n_rows:
        raise ValueError("row_to_node_index length must equal number of rows")
    if n_logits > n_rows:
        raise ValueError("n_logits must be <= number of rows")
    if logit_p.numel() != n_logits:
        raise ValueError("logit_p length must equal n_logits")

    device = device or feature_edge_matrix.device
    working_feature_matrix = (
        feature_edge_matrix
        if feature_edge_matrix.device == device
        else feature_edge_matrix.to(device)
    )
    working_row_abs_sums = (
        row_abs_sums if row_abs_sums.device == device else row_abs_sums.to(device)
    ).to(dtype=working_feature_matrix.dtype)
    working_row_index = (
        row_to_node_index if row_to_node_index.device == device else row_to_node_index.to(device)
    ).long()
    working_logit_p = logit_p if logit_p.device == device else logit_p.to(device)

    feature_row_node_index = working_row_index[n_logits:]
    if feature_row_node_index.numel():
        if feature_row_node_index.min() < 0 or feature_row_node_index.max() >= n_feature_nodes:
            raise ValueError(
                "feature row node indices must be in [0, n_feature_nodes) for compact influences"
            )

    influences = torch.zeros(n_feature_nodes, device=device, dtype=working_feature_matrix.dtype)
    row_weights = torch.zeros(n_rows, device=device, dtype=working_feature_matrix.dtype)
    row_weights[:n_logits] = working_logit_p.to(dtype=working_feature_matrix.dtype)

    for _ in range(max_iter):
        next_feature_prod = torch.zeros_like(influences)
        for start in range(0, n_rows, row_chunk_size):
            end = min(start + row_chunk_size, n_rows)
            chunk = working_feature_matrix[start:end].abs()
            if not chunk.numel():
                continue

            chunk_row_weights = row_weights[start:end]
            if not chunk_row_weights.any():
                continue

            denom = working_row_abs_sums[start:end].clamp(min=1e-8)
            next_feature_prod += (chunk_row_weights / denom) @ chunk

        if not next_feature_prod.any():
            break

        influences += next_feature_prod
        row_weights.zero_()
        if feature_row_node_index.numel():
            row_weights[n_logits:] = next_feature_prod[feature_row_node_index]
    else:
        raise RuntimeError("Failed to converge")

    return influences


def compute_partial_feature_influences_streaming(
    row_reader: Callable[[int, int], torch.Tensor],
    row_abs_sums: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    *,
    n_feature_nodes: int,
    n_logits: int,
    max_iter: int = 128,
    device=None,
    row_chunk_size: int = 4096,
    chunk_cache_max_bytes: int = 0,
    chunk_reuse_stats: dict[str, int] | None = None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute feature-only partial influences from streamed dense row chunks.

    This computes the same quantity as ``compute_partial_feature_influences`` while
    reading feature rows incrementally via ``row_reader``. It is intended for
    file-backed row stores where materializing the full dense feature matrix in RAM
    is undesirable.

    Args:
        row_reader: Callable receiving ``(row_start, row_end)`` and returning a dense
            tensor of shape ``(row_end - row_start, n_feature_nodes)`` for that row range.
        row_abs_sums: Exact absolute row sums over the original full rows.
        logit_p: Logit probabilities / weights for the first ``n_logits`` rows.
        row_to_node_index: Mapping from row index to original node index.
        n_feature_nodes: Number of active feature columns in the original graph.
        n_logits: Number of logit rows at the start of the stored rows.
        max_iter: Maximum number of power-iteration steps.
        device: Device to run computation on.
        row_chunk_size: Row chunk size used for streamed reads.
        chunk_cache_max_bytes: Strict byte budget for optional solver-local
            row-chunk reuse cache. ``0`` disables solver-local caching.
        chunk_reuse_stats: Optional output dictionary populated with lightweight
            chunk reuse counters for diagnostics.
        compute_dtype: Optional explicit compute dtype for influence math. When
            omitted, defaults to ``row_abs_sums.dtype`` for backward compatibility.

    Returns:
        Tensor of shape ``(n_feature_nodes,)`` with partial influence values.

    Raises:
        ValueError: If inputs are inconsistent.
        RuntimeError: If computation fails to converge within ``max_iter``.
    """

    if n_feature_nodes < 0:
        raise ValueError("n_feature_nodes must be >= 0")
    if n_logits < 0:
        raise ValueError("n_logits must be >= 0")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be > 0")
    if chunk_cache_max_bytes < 0:
        raise ValueError("chunk_cache_max_bytes must be >= 0")

    n_rows = row_abs_sums.numel()
    if row_to_node_index.numel() != n_rows:
        raise ValueError("row_to_node_index length must equal row_abs_sums length")
    if n_logits > n_rows:
        raise ValueError("n_logits must be <= number of rows")
    if logit_p.numel() != n_logits:
        raise ValueError("logit_p length must equal n_logits")

    if compute_dtype is not None and compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64 when provided")

    device = device or row_abs_sums.device
    dtype = row_abs_sums.dtype if compute_dtype is None else compute_dtype
    working_row_abs_sums = (
        row_abs_sums if row_abs_sums.device == device else row_abs_sums.to(device)
    )
    working_row_abs_sums = working_row_abs_sums.to(dtype=dtype)
    working_row_index = (
        row_to_node_index if row_to_node_index.device == device else row_to_node_index.to(device)
    ).long()
    working_logit_p = logit_p if logit_p.device == device else logit_p.to(device)

    feature_row_node_index = working_row_index[n_logits:]
    if feature_row_node_index.numel():
        if feature_row_node_index.min() < 0 or feature_row_node_index.max() >= n_feature_nodes:
            raise ValueError(
                "feature row node indices must be in [0, n_feature_nodes) for compact influences"
            )

    influences = torch.zeros(n_feature_nodes, device=device, dtype=dtype)
    row_weights = torch.zeros(n_rows, device=device, dtype=dtype)
    row_weights[:n_logits] = working_logit_p.to(dtype=dtype)
    denom = working_row_abs_sums.clamp(min=1e-8)
    chunk_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
    cache_enabled = bool(chunk_cache_max_bytes > 0)
    chunk_cache_nbytes = 0
    chunk_request_count = 0
    chunk_cache_hit_count = 0
    chunk_cache_miss_count = 0
    chunk_cache_eviction_count = 0
    chunk_cache_store_success_count = 0
    chunk_cache_store_skip_disabled_count = 0
    chunk_cache_store_skip_too_large_count = 0

    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _drop_oldest_chunk() -> None:
        nonlocal chunk_cache_nbytes, chunk_cache_eviction_count
        oldest_key = next(iter(chunk_cache))
        dropped = chunk_cache.pop(oldest_key)
        chunk_cache_nbytes = max(0, chunk_cache_nbytes - _tensor_nbytes(dropped))
        chunk_cache_eviction_count += 1

    for _ in range(max_iter):
        next_feature_prod = torch.zeros_like(influences)
        for start in range(0, n_rows, row_chunk_size):
            end = min(start + row_chunk_size, n_rows)
            chunk_row_weights = row_weights[start:end]
            if not bool(chunk_row_weights.any()):
                continue

            chunk_request_count += 1
            cache_key = (start, end)
            cached_chunk = chunk_cache.get(cache_key) if cache_enabled else None
            if cached_chunk is None:
                chunk = row_reader(start, end)
                if chunk.ndim != 2 or chunk.shape != (end - start, n_feature_nodes):
                    raise ValueError(
                        "row_reader must return shape "
                        f"({end - start}, {n_feature_nodes}) for rows [{start}, {end})"
                    )
                if chunk.device != device:
                    chunk = chunk.to(device)
                chunk = chunk.to(dtype=dtype).abs()
                if not cache_enabled:
                    chunk_cache_store_skip_disabled_count += 1
                else:
                    chunk_nbytes = _tensor_nbytes(chunk)
                    if chunk_nbytes > chunk_cache_max_bytes:
                        chunk_cache_store_skip_too_large_count += 1
                    else:
                        while (
                            chunk_cache
                            and chunk_cache_nbytes + chunk_nbytes > chunk_cache_max_bytes
                        ):
                            _drop_oldest_chunk()
                        chunk_cache[cache_key] = chunk
                        chunk_cache.move_to_end(cache_key)
                        chunk_cache_nbytes += chunk_nbytes
                        chunk_cache_store_success_count += 1
                chunk_cache_miss_count += 1
            else:
                chunk = cached_chunk
                chunk_cache.move_to_end(cache_key)
                chunk_cache_hit_count += 1

            next_feature_prod += (chunk_row_weights / denom[start:end]) @ chunk

        if not bool(next_feature_prod.any()):
            break

        influences += next_feature_prod
        row_weights.zero_()
        if feature_row_node_index.numel():
            row_weights[n_logits:] = next_feature_prod[feature_row_node_index]
    else:
        raise RuntimeError("Failed to converge")

    if chunk_reuse_stats is not None:
        chunk_reuse_stats.clear()
        chunk_reuse_stats.update(
            {
                "chunk_request_count": int(chunk_request_count),
                "chunk_cache_enabled": int(cache_enabled),
                "chunk_cache_max_bytes": int(chunk_cache_max_bytes),
                "chunk_cache_hit_count": int(chunk_cache_hit_count),
                "chunk_cache_miss_count": int(chunk_cache_miss_count),
                "row_reader_call_count": int(chunk_cache_miss_count),
                "chunk_cache_eviction_count": int(chunk_cache_eviction_count),
                "chunk_cache_store_success_count": int(chunk_cache_store_success_count),
                "chunk_cache_store_skip_disabled_count": int(chunk_cache_store_skip_disabled_count),
                "chunk_cache_store_skip_too_large_count": int(
                    chunk_cache_store_skip_too_large_count
                ),
                "chunk_cache_unique_entries": int(len(chunk_cache)),
                "chunk_cache_nbytes": int(chunk_cache_nbytes),
            }
        )

    return influences
