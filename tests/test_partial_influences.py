import torch

from circuit_tracer.graph import compute_partial_influences


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
