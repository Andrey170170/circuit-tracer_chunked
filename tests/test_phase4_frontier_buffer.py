import torch

from circuit_tracer.attribution.attribute_nnsight import (
    _build_phase4_frontier_buffer_decision,
)


def _decision(**kwargs):
    defaults = dict(
        candidate_scores=torch.tensor([10.0, 9.0, 8.0, 7.96, 7.8, 1.0]),
        base_frontier_size=3,
        actual_max_feature_nodes=3,
        capacity_feature_nodes=6,
        total_active_features=6,
        used_total=0,
        epsilon=0.01,
        max_per_refresh=2,
        max_total=3,
        refresh_index=0,
        visited_before=0,
    )
    defaults.update(kwargs)
    return _build_phase4_frontier_buffer_decision(**defaults)


def test_phase4_frontier_buffer_expands_near_cutoff_candidates():
    decision = _decision()

    assert decision["effective"] is True
    assert decision["extra_feature_count"] == 1
    assert decision["expanded_frontier_size"] == 4
    event = decision["event"]
    assert event["cutoff_score"] == 8.0
    assert event["near_cutoff_counts"]["0.01"] == 1


def test_phase4_frontier_buffer_respects_caps():
    decision = _decision(
        candidate_scores=torch.tensor([10.0, 9.0, 8.0, 7.99, 7.98, 7.97]),
        capacity_feature_nodes=4,
        max_per_refresh=3,
        max_total=3,
    )

    assert decision["extra_feature_count"] == 1
    assert decision["expanded_frontier_size"] == 4


def test_phase4_frontier_buffer_nonpositive_cutoff_fallback():
    decision = _decision(
        candidate_scores=torch.tensor([1.0, 0.0, -0.001, -0.002]),
        base_frontier_size=2,
        actual_max_feature_nodes=2,
        capacity_feature_nodes=4,
        total_active_features=4,
        epsilon=0.05,
    )

    assert decision["effective"] is False
    assert decision["extra_feature_count"] == 0
    assert decision["event"]["fallback_reason"] == "nonpositive_cutoff_score"


def test_phase4_frontier_buffer_capacity_exhausted_fallback():
    decision = _decision(actual_max_feature_nodes=6, capacity_feature_nodes=6)

    assert decision["effective"] is False
    assert decision["event"]["fallback_reason"] == "capacity_or_budget_exhausted"
