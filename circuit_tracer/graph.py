"""Graph data structures for attribution results."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import NamedTuple
import time
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
    row_denominator = _compute_row_denominator_scaled_l1(normalized, already_abs=True)
    normalized_row_weights = _normalize_row_weights_from_denominator(
        torch.ones(normalized.shape[0], device=normalized.device, dtype=normalized.dtype),
        denom_mode="scaled_row_l1",
        denom_primary=row_denominator.row_abs_max,
        denom_secondary=row_denominator.row_l1_scaled,
        clamp_epsilon=1e-10,
    )
    return normalized_row_weights.unsqueeze(1) * normalized


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


class RowL1ScaledDenominator(NamedTuple):
    """Stable row-denominator representation.

    ``row_l1 = row_abs_max * row_l1_scaled`` where:
      - ``row_abs_max = max(abs(row))``
      - ``row_l1_scaled = sum(abs(row) / row_abs_max)`` for non-zero rows,
        and ``0`` for all-zero rows.
    """

    row_abs_max: torch.Tensor
    row_l1_scaled: torch.Tensor


def _compute_row_denominator_scaled_l1(
    row_values: torch.Tensor,
    *,
    already_abs: bool = False,
) -> RowL1ScaledDenominator:
    abs_values = row_values if already_abs else row_values.abs()
    if abs_values.ndim != 2:
        raise ValueError("row_values must be rank-2")

    n_rows = int(abs_values.shape[0])
    if abs_values.shape[1] == 0:
        zeros = torch.zeros(n_rows, device=abs_values.device, dtype=abs_values.dtype)
        return RowL1ScaledDenominator(row_abs_max=zeros, row_l1_scaled=zeros)

    row_abs_max = abs_values.amax(dim=1)
    row_l1_scaled = torch.zeros_like(row_abs_max)
    nonzero_rows = (row_abs_max > 0) & torch.isfinite(row_abs_max)
    if bool(nonzero_rows.any()):
        scaled_rows = abs_values[nonzero_rows] / row_abs_max[nonzero_rows].unsqueeze(1)
        row_l1_scaled[nonzero_rows] = scaled_rows.sum(dim=1)

    infinite_rows = torch.isinf(row_abs_max)
    if bool(infinite_rows.any()):
        row_l1_scaled[infinite_rows] = 1

    return RowL1ScaledDenominator(row_abs_max=row_abs_max, row_l1_scaled=row_l1_scaled)


def _normalize_row_weights_from_denominator(
    row_weights: torch.Tensor,
    *,
    denom_mode: str,
    denom_primary: torch.Tensor,
    denom_secondary: torch.Tensor | None,
    clamp_epsilon: float = 1e-8,
) -> torch.Tensor:
    if denom_mode == "raw_l1":
        return row_weights / denom_primary.clamp(min=clamp_epsilon)

    if denom_secondary is None:
        raise ValueError("scaled_row_l1 denominator requires secondary component")

    row_abs_max = denom_primary
    row_l1_scaled = denom_secondary
    scaled_threshold = torch.where(
        row_abs_max > 0,
        torch.full_like(row_abs_max, clamp_epsilon) / row_abs_max,
        torch.full_like(row_abs_max, float("inf")),
    )
    nan_denominator = torch.isnan(row_abs_max) | torch.isnan(row_l1_scaled)
    infinite_denominator = ~nan_denominator & (
        torch.isinf(row_abs_max) | torch.isinf(row_l1_scaled)
    )
    finite_denominator = ~(nan_denominator | infinite_denominator)
    use_scaled_denominator = (
        finite_denominator
        & (row_abs_max > 0)
        & (row_l1_scaled > 0)
        & (row_l1_scaled >= scaled_threshold)
    )
    use_clamped_epsilon = finite_denominator & (
        (row_abs_max <= 0) | (row_l1_scaled <= 0) | (row_l1_scaled < scaled_threshold)
    )

    normalized = torch.empty_like(row_weights)
    if bool(use_scaled_denominator.any()):
        normalized[use_scaled_denominator] = (
            row_weights[use_scaled_denominator] / row_abs_max[use_scaled_denominator]
        ) / row_l1_scaled[use_scaled_denominator]
    if bool(use_clamped_epsilon.any()):
        normalized[use_clamped_epsilon] = row_weights[use_clamped_epsilon] / clamp_epsilon
    if bool(infinite_denominator.any()):
        normalized[infinite_denominator] = 0
    if bool(nan_denominator.any()):
        normalized[nan_denominator] = torch.full_like(row_weights[nan_denominator], float("nan"))
    return normalized


def _resolve_row_denominator(
    row_denominator: torch.Tensor | RowL1ScaledDenominator | tuple[torch.Tensor, torch.Tensor],
    *,
    expected_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    if isinstance(row_denominator, torch.Tensor):
        if row_denominator.numel() != expected_rows:
            raise ValueError("row_abs_sums length must equal number of rows")
        return (
            "raw_l1",
            row_denominator.to(device=device, dtype=dtype),
            None,
        )

    if isinstance(row_denominator, tuple) and len(row_denominator) == 2:
        row_abs_max, row_l1_scaled = row_denominator
        if not isinstance(row_abs_max, torch.Tensor) or not isinstance(row_l1_scaled, torch.Tensor):
            raise TypeError(
                "row_abs_sums tuple form must be (row_abs_max: Tensor, row_l1_scaled: Tensor)"
            )
        if row_abs_max.numel() != expected_rows:
            raise ValueError("row_abs_max length must equal number of rows")
        if row_l1_scaled.numel() != expected_rows:
            raise ValueError("row_l1_scaled length must equal number of rows")
        return (
            "scaled_row_l1",
            row_abs_max.to(device=device, dtype=dtype),
            row_l1_scaled.to(device=device, dtype=dtype),
        )

    raise TypeError("row_abs_sums must be a Tensor or a (row_abs_max, row_l1_scaled) tuple")


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

            chunk_row_denominator = _compute_row_denominator_scaled_l1(chunk, already_abs=True)
            normalized_row_weights = _normalize_row_weights_from_denominator(
                row_weights,
                denom_mode="scaled_row_l1",
                denom_primary=chunk_row_denominator.row_abs_max,
                denom_secondary=chunk_row_denominator.row_l1_scaled,
            )
            next_prod += normalized_row_weights @ chunk

        prod = next_prod
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def compute_partial_feature_influences(
    feature_edge_matrix: torch.Tensor,
    row_abs_sums: torch.Tensor | RowL1ScaledDenominator | tuple[torch.Tensor, torch.Tensor],
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
      - ``row_abs_sums`` stores exact L1 row denominators over *all* original columns.

    Args:
        feature_edge_matrix: Dense matrix of shape ``(n_rows, n_feature_nodes)`` containing
            feature-column edge values for each stored row.
        row_abs_sums: Either
            (1) exact absolute row sums over original full rows (shape ``(n_rows,)``), or
            (2) stable scaled-row-L1 pair ``(row_abs_max, row_l1_scaled)`` where
                ``row_l1 = row_abs_max * row_l1_scaled``.
            Both preserve normalization semantics even when non-feature columns are
            not materialized.
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
    denom_mode, denom_primary, denom_secondary = _resolve_row_denominator(
        row_abs_sums,
        expected_rows=n_rows,
        device=device,
        dtype=working_feature_matrix.dtype,
    )
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

            normalized_row_weights = _normalize_row_weights_from_denominator(
                chunk_row_weights,
                denom_mode=denom_mode,
                denom_primary=denom_primary[start:end],
                denom_secondary=(
                    denom_secondary[start:end] if denom_secondary is not None else None
                ),
            )
            next_feature_prod += normalized_row_weights @ chunk

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
    row_abs_sums: torch.Tensor | RowL1ScaledDenominator | tuple[torch.Tensor, torch.Tensor],
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    *,
    n_feature_nodes: int,
    n_logits: int,
    max_iter: int = 128,
    device=None,
    row_chunk_size: int = 4096,
    chunk_cache_max_bytes: int = 0,
    chunk_reuse_stats: dict[str, int | float | str] | None = None,
    compute_dtype: torch.dtype | None = None,
    active_row_only_chunks: bool = False,
) -> torch.Tensor:
    """Compute feature-only partial influences from streamed dense row chunks.

    This computes the same quantity as ``compute_partial_feature_influences`` while
    reading feature rows incrementally via ``row_reader``. It is intended for
    file-backed row stores where materializing the full dense feature matrix in RAM
    is undesirable.

    Args:
        row_reader: Callable receiving ``(row_start, row_end)`` and returning a dense
            tensor of shape ``(row_end - row_start, n_feature_nodes)`` for that row range.
        row_abs_sums: Either exact absolute row sums over original full rows or
            a stable scaled-row-L1 pair ``(row_abs_max, row_l1_scaled)``.
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
            chunk reuse counters and elapsed timings for diagnostics.
        compute_dtype: Optional explicit compute dtype for influence math. When
            omitted, defaults to the primary denominator dtype for backward
            compatibility.
        active_row_only_chunks: When ``True``, preserve fixed row-chunk windows but
            read only contiguous non-zero subranges inside each active chunk.
            Default ``False`` preserves legacy fixed-chunk behavior.

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

    if isinstance(row_abs_sums, torch.Tensor):
        n_rows = row_abs_sums.numel()
        denominator_dtype = row_abs_sums.dtype
    elif isinstance(row_abs_sums, tuple) and len(row_abs_sums) == 2:
        n_rows = row_abs_sums[0].numel()
        denominator_dtype = row_abs_sums[0].dtype
    else:
        raise TypeError("row_abs_sums must be a Tensor or a (row_abs_max, row_l1_scaled) tuple")
    if row_to_node_index.numel() != n_rows:
        raise ValueError("row_to_node_index length must equal row_abs_sums length")
    if n_logits > n_rows:
        raise ValueError("n_logits must be <= number of rows")
    if logit_p.numel() != n_logits:
        raise ValueError("logit_p length must equal n_logits")

    if compute_dtype is not None and compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64 when provided")
    if not isinstance(active_row_only_chunks, bool):
        raise ValueError("active_row_only_chunks must be a bool")

    if isinstance(row_abs_sums, torch.Tensor):
        row_denominator_device = row_abs_sums.device
    else:
        row_denominator_device = row_abs_sums[0].device

    device = device or row_denominator_device
    dtype = denominator_dtype if compute_dtype is None else compute_dtype
    denom_mode, denom_primary, denom_secondary = _resolve_row_denominator(
        row_abs_sums,
        expected_rows=n_rows,
        device=device,
        dtype=dtype,
    )
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
    chunk_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
    # Active-row chunks are iteration-specific because the stored fixed-shape chunk
    # contains zero-filled inactive rows that may differ on later iterations.
    # Never reuse those chunks from the solver-local cache.
    cache_enabled = bool(chunk_cache_max_bytes > 0) and not active_row_only_chunks
    chunk_cache_nbytes = 0
    chunk_request_count = 0
    chunk_cache_hit_count = 0
    chunk_cache_miss_count = 0
    chunk_cache_eviction_count = 0
    chunk_cache_store_success_count = 0
    chunk_cache_store_skip_disabled_count = 0
    chunk_cache_store_skip_too_large_count = 0
    active_row_chunk_count = 0
    active_row_range_count = 0
    row_reader_row_count = 0
    row_reader_call_count = 0
    row_weight_nonzero_row_count = 0
    row_weight_zero_row_count = 0
    row_reader_elapsed_ms_total = 0.0
    normalization_elapsed_ms_total = 0.0
    matmul_elapsed_ms_total = 0.0
    iteration_count = 0
    solver_start = time.perf_counter()
    row_chunk_strategy = (
        "active_row_contiguous_chunks" if active_row_only_chunks else "fixed_row_chunks"
    )

    def _iter_active_row_subranges(
        chunk_start: int,
        chunk_end: int,
        current_row_weights: torch.Tensor,
    ):
        active_rows = (
            torch.nonzero(
                current_row_weights[chunk_start:chunk_end] != 0,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        if not active_rows:
            return

        run_start = active_rows[0]
        run_end = run_start + 1
        for row_offset in active_rows[1:]:
            if row_offset == run_end:
                run_end += 1
                continue

            yield chunk_start + run_start, chunk_start + run_end
            run_start = row_offset
            run_end = row_offset + 1

        yield chunk_start + run_start, chunk_start + run_end

    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _drop_oldest_chunk() -> None:
        nonlocal chunk_cache_nbytes, chunk_cache_eviction_count
        oldest_key = next(iter(chunk_cache))
        dropped = chunk_cache.pop(oldest_key)
        chunk_cache_nbytes = max(0, chunk_cache_nbytes - _tensor_nbytes(dropped))
        chunk_cache_eviction_count += 1

    for _ in range(max_iter):
        iteration_count += 1
        next_feature_prod = torch.zeros_like(influences)
        iteration_nonzero_rows = int(torch.count_nonzero(row_weights).item())
        row_weight_nonzero_row_count += iteration_nonzero_rows
        row_weight_zero_row_count += int(max(0, n_rows - iteration_nonzero_rows))

        if not active_row_only_chunks:
            for start in range(0, n_rows, row_chunk_size):
                end = min(start + row_chunk_size, n_rows)
                chunk_row_weights = row_weights[start:end]
                if not bool(chunk_row_weights.any()):
                    continue

                active_row_chunk_count += 1
                chunk_request_count += 1
                cache_key = (start, end)
                cached_chunk = chunk_cache.get(cache_key) if cache_enabled else None
                if cached_chunk is None:
                    row_reader_start = time.perf_counter()
                    chunk = row_reader(start, end)
                    row_reader_elapsed_ms_total += (time.perf_counter() - row_reader_start) * 1000.0
                    row_reader_call_count += 1
                    row_reader_row_count += int(end - start)
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

                normalization_start = time.perf_counter()
                normalized_row_weights = _normalize_row_weights_from_denominator(
                    chunk_row_weights,
                    denom_mode=denom_mode,
                    denom_primary=denom_primary[start:end],
                    denom_secondary=(
                        denom_secondary[start:end] if denom_secondary is not None else None
                    ),
                )
                normalization_elapsed_ms_total += (
                    time.perf_counter() - normalization_start
                ) * 1000.0

                matmul_start = time.perf_counter()
                next_feature_prod += normalized_row_weights @ chunk
                matmul_elapsed_ms_total += (time.perf_counter() - matmul_start) * 1000.0
        else:
            for start in range(0, n_rows, row_chunk_size):
                end = min(start + row_chunk_size, n_rows)
                chunk_row_weights = row_weights[start:end]
                active_row_subranges = list(_iter_active_row_subranges(start, end, row_weights))
                if not active_row_subranges:
                    continue

                active_row_chunk_count += 1
                active_row_range_count += len(active_row_subranges)
                chunk_request_count += 1
                cache_key = (start, end)
                cached_chunk = chunk_cache.get(cache_key) if cache_enabled else None
                if cached_chunk is None:
                    chunk = torch.zeros((end - start, n_feature_nodes), device=device, dtype=dtype)
                    for sub_start, sub_end in active_row_subranges:
                        row_reader_start = time.perf_counter()
                        subchunk = row_reader(sub_start, sub_end)
                        row_reader_elapsed_ms_total += (
                            time.perf_counter() - row_reader_start
                        ) * 1000.0
                        row_reader_call_count += 1
                        row_reader_row_count += int(sub_end - sub_start)
                        if subchunk.ndim != 2 or subchunk.shape != (
                            sub_end - sub_start,
                            n_feature_nodes,
                        ):
                            raise ValueError(
                                "row_reader must return shape "
                                f"({sub_end - sub_start}, {n_feature_nodes}) for rows "
                                f"[{sub_start}, {sub_end})"
                            )
                        if subchunk.device != device:
                            subchunk = subchunk.to(device)
                        subchunk = subchunk.to(dtype=dtype).abs()
                        chunk[sub_start - start : sub_end - start] = subchunk
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

                normalization_start = time.perf_counter()
                normalized_row_weights = _normalize_row_weights_from_denominator(
                    chunk_row_weights,
                    denom_mode=denom_mode,
                    denom_primary=denom_primary[start:end],
                    denom_secondary=(
                        denom_secondary[start:end] if denom_secondary is not None else None
                    ),
                )
                normalization_elapsed_ms_total += (
                    time.perf_counter() - normalization_start
                ) * 1000.0

                matmul_start = time.perf_counter()
                next_feature_prod += normalized_row_weights @ chunk
                matmul_elapsed_ms_total += (time.perf_counter() - matmul_start) * 1000.0

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
        solver_elapsed_ms_total = (time.perf_counter() - solver_start) * 1000.0
        chunk_reuse_stats.update(
            {
                "chunk_request_count": int(chunk_request_count),
                "chunk_cache_enabled": int(cache_enabled),
                "chunk_cache_max_bytes": int(chunk_cache_max_bytes),
                "chunk_cache_hit_count": int(chunk_cache_hit_count),
                "chunk_cache_miss_count": int(chunk_cache_miss_count),
                "row_reader_call_count": int(row_reader_call_count),
                "chunk_cache_eviction_count": int(chunk_cache_eviction_count),
                "chunk_cache_store_success_count": int(chunk_cache_store_success_count),
                "chunk_cache_store_skip_disabled_count": int(chunk_cache_store_skip_disabled_count),
                "chunk_cache_store_skip_too_large_count": int(
                    chunk_cache_store_skip_too_large_count
                ),
                "chunk_cache_unique_entries": int(len(chunk_cache)),
                "chunk_cache_nbytes": int(chunk_cache_nbytes),
                "row_chunk_strategy": row_chunk_strategy,
                "active_row_only_chunks": int(active_row_only_chunks),
                "active_row_chunk_count": int(active_row_chunk_count),
                "active_row_range_count": int(active_row_range_count),
                "row_weight_nonzero_row_count": int(row_weight_nonzero_row_count),
                "row_weight_zero_row_count": int(row_weight_zero_row_count),
                "row_reader_overread_zero_row_count": int(
                    max(0, row_reader_row_count - row_weight_nonzero_row_count)
                ),
                "row_reader_row_count": int(row_reader_row_count),
                "iteration_count": int(iteration_count),
                "solver_elapsed_ms_total": float(solver_elapsed_ms_total),
                "row_reader_elapsed_ms_total": float(row_reader_elapsed_ms_total),
                "normalization_elapsed_ms_total": float(normalization_elapsed_ms_total),
                "matmul_elapsed_ms_total": float(matmul_elapsed_ms_total),
            }
        )

    return influences
