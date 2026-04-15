import torch

from circuit_tracer.graph import (
    compute_partial_feature_influences,
    compute_partial_feature_influences_streaming,
    compute_partial_influences,
)


def compute_partial_influences_reference(
    edge_matrix: torch.Tensor,
    logit_p: torch.Tensor,
    row_to_node_index: torch.Tensor,
    max_iter: int = 128,
) -> torch.Tensor:
    normalized_matrix = edge_matrix.abs()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], dtype=edge_matrix.dtype)
    prod = torch.zeros(edge_matrix.shape[1], dtype=edge_matrix.dtype)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index.long()] @ normalized_matrix
        if not prod.any():
            break
        influences += prod

    return influences


def test_compute_partial_influences_matches_dense_reference():
    edge_matrix = torch.tensor(
        [
            [0.6, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([0.7, 0.3], dtype=torch.float32)
    row_to_node_index = torch.tensor([4, 3, 2], dtype=torch.int32)

    expected = compute_partial_influences_reference(edge_matrix, logit_p, row_to_node_index)
    actual = compute_partial_influences(
        edge_matrix,
        logit_p,
        row_to_node_index,
        device=torch.device("cpu"),
        row_chunk_size=2,
    )

    assert actual.device.type == "cpu"
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_compute_partial_influences_keeps_cpu_ownership_by_default():
    edge_matrix = torch.tensor([[0.4, 0.6, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    row_to_node_index = torch.tensor([2, 1], dtype=torch.int32)

    influences = compute_partial_influences(edge_matrix, logit_p, row_to_node_index)

    assert influences.device.type == "cpu"


def test_compute_partial_feature_influences_matches_dense_feature_slice():
    n_features = 4
    n_sinks = 3
    n_logits = 2
    total_nodes = n_features + n_sinks + n_logits

    edge_matrix = torch.tensor(
        [
            # logit rows
            [0.2, -0.1, 0.0, 0.3, 0.7, 0.0, -0.2, 0.0, 0.0],
            [0.1, 0.0, -0.2, 0.0, 0.2, -0.4, 0.3, 0.0, 0.0],
            # feature rows
            [0.4, -0.1, 0.2, 0.0, 0.6, -0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, -0.5, -0.8, 0.0, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([0.75, 0.25], dtype=torch.float32)
    # First two rows are logits (node indices at tail), then two visited feature rows
    row_to_node_index = torch.tensor(
        [total_nodes - 2, total_nodes - 1, 1, 3],
        dtype=torch.int32,
    )

    expected_full = compute_partial_influences(
        edge_matrix,
        logit_p,
        row_to_node_index,
        device=torch.device("cpu"),
        row_chunk_size=2,
    )

    actual_feature_only = compute_partial_feature_influences(
        edge_matrix[:, :n_features],
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
        device=torch.device("cpu"),
        row_chunk_size=2,
    )

    assert torch.allclose(actual_feature_only, expected_full[:n_features], atol=1e-6, rtol=1e-6)


def test_compute_partial_feature_influences_uses_full_row_abs_sums_for_normalization():
    n_features = 3
    n_sinks = 2
    n_logits = 1
    total_nodes = n_features + n_sinks + n_logits

    # Large sink mass should dilute feature influence exactly as in dense path.
    edge_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.0, 100.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 50.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    row_to_node_index = torch.tensor([total_nodes - 1, 0], dtype=torch.int32)

    expected = compute_partial_influences(edge_matrix, logit_p, row_to_node_index)

    compact = compute_partial_feature_influences(
        edge_matrix[:, :n_features],
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
    )

    # If row_abs_sums ignored sink columns this would fail (much larger values).
    assert torch.allclose(compact, expected[:n_features], atol=1e-6, rtol=1e-6)


def _dense_row_reader(dense_feature_rows: torch.Tensor):
    def _read_rows(row_start: int, row_end: int) -> torch.Tensor:
        return dense_feature_rows[row_start:row_end]

    return _read_rows


def test_compute_partial_feature_influences_streaming_matches_dense_companion():
    n_features = 4
    n_sinks = 3
    n_logits = 2
    total_nodes = n_features + n_sinks + n_logits

    edge_matrix = torch.tensor(
        [
            [0.2, -0.1, 0.0, 0.3, 0.7, 0.0, -0.2, 0.0, 0.0],
            [0.1, 0.0, -0.2, 0.0, 0.2, -0.4, 0.3, 0.0, 0.0],
            [0.4, -0.1, 0.2, 0.0, 0.6, -0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, -0.5, -0.8, 0.0, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([0.75, 0.25], dtype=torch.float32)
    row_to_node_index = torch.tensor(
        [total_nodes - 2, total_nodes - 1, 1, 3],
        dtype=torch.int32,
    )

    dense_expected = compute_partial_feature_influences(
        edge_matrix[:, :n_features],
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
    )

    streaming_actual = compute_partial_feature_influences_streaming(
        _dense_row_reader(edge_matrix[:, :n_features]),
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
        row_chunk_size=2,
    )

    assert torch.allclose(streaming_actual, dense_expected, atol=1e-6, rtol=1e-6)


def test_compute_partial_feature_influences_streaming_ignores_rows_beyond_prefix():
    n_features = 3
    n_sinks = 2
    n_logits = 1
    total_nodes = n_features + n_sinks + n_logits

    # Last row is outside the evaluated prefix and should not affect output.
    edge_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 2.0, 0.5, 0.0, 5.0, 0.0],
            [0.3, 0.0, 0.1, 0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    row_to_node_index = torch.tensor([total_nodes - 1, 0, 2, 1], dtype=torch.int32)

    prefix_rows = 3
    dense_expected = compute_partial_feature_influences(
        edge_matrix[:prefix_rows, :n_features],
        edge_matrix[:prefix_rows].abs().sum(dim=1),
        logit_p,
        row_to_node_index[:prefix_rows],
        n_feature_nodes=n_features,
        n_logits=n_logits,
    )

    streaming_actual = compute_partial_feature_influences_streaming(
        _dense_row_reader(edge_matrix[:, :n_features]),
        edge_matrix[:prefix_rows].abs().sum(dim=1),
        logit_p,
        row_to_node_index[:prefix_rows],
        n_feature_nodes=n_features,
        n_logits=n_logits,
        row_chunk_size=2,
    )

    assert torch.allclose(streaming_actual, dense_expected, atol=1e-6, rtol=1e-6)


def test_compute_partial_feature_influences_streaming_handles_zero_feature_rows():
    n_features = 3
    n_sinks = 2
    n_logits = 1
    total_nodes = n_features + n_sinks + n_logits

    edge_matrix = torch.tensor(
        [
            # Logit row with no feature edges but non-zero sink mass.
            [0.0, 0.0, 0.0, 5.0, 1.0, 0.0],
            # One visited feature row.
            [1.0, 0.5, 0.0, 0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_p = torch.tensor([1.0], dtype=torch.float32)
    row_to_node_index = torch.tensor([total_nodes - 1, 0], dtype=torch.int32)

    dense_expected = compute_partial_feature_influences(
        edge_matrix[:, :n_features],
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
    )

    streaming_actual = compute_partial_feature_influences_streaming(
        _dense_row_reader(edge_matrix[:, :n_features]),
        edge_matrix.abs().sum(dim=1),
        logit_p,
        row_to_node_index,
        n_feature_nodes=n_features,
        n_logits=n_logits,
        row_chunk_size=2,
    )

    assert torch.allclose(streaming_actual, dense_expected, atol=1e-6, rtol=1e-6)
