import torch

from circuit_tracer.attribution.attribute_nnsight import _build_phase3_frontier_buffer_metadata


def test_phase3_frontier_buffer_expands_near_cutoff_candidates():
    metadata = _build_phase3_frontier_buffer_metadata(
        seed_feature_influences=torch.tensor([10.0, 9.0, 8.0, 7.96, 7.8, 1.0]),
        base_max_feature_nodes=3,
        total_active_features=6,
        relative_epsilon=0.01,
        max_extra=2,
    )

    assert metadata["status"] == "expanded"
    assert metadata["effective"] is True
    assert metadata["extra_feature_count"] == 1
    assert metadata["actual_max_feature_nodes"] == 4
    assert metadata["cutoff_score"] == 8.0
    assert metadata["near_cutoff_counts"]["0.01"] == 1


def test_phase3_frontier_buffer_nonpositive_cutoff_is_conservative():
    metadata = _build_phase3_frontier_buffer_metadata(
        seed_feature_influences=torch.tensor([1.0, 0.0, -0.001, -0.002]),
        base_max_feature_nodes=2,
        total_active_features=4,
        relative_epsilon=0.05,
        max_extra=10,
    )

    assert metadata["status"] == "fallback"
    assert metadata["fallback_reason"] == "nonpositive_cutoff_score"
    assert metadata["extra_feature_count"] == 0
    assert metadata["actual_max_feature_nodes"] == 2
