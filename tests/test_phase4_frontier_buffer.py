import pytest
import torch

from circuit_tracer.attribution.nnsight.phase4_policy import (
    _compute_phase4_rank_selection_max_feature_nodes_cap_bound,
    _select_phase4_frontier_rank_selection,
)
from circuit_tracer.attribution.nnsight.phase_support import _build_phase4_frontier_buffer_decision


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


def test_phase4_rank_selection_reports_binding_cutoff_margin():
    selection = _select_phase4_frontier_rank_selection(
        feature_influences=torch.tensor([10.0, 9.0, 8.0, 7.999996, 1.0], dtype=torch.float64),
        visited=torch.zeros(5, dtype=torch.bool),
        frontier_size=3,
        ranker_mode="argsort",
    )

    assert selection.cutoff_score == 8.0
    assert selection.cutoff_gap == pytest.approx(4e-6)
    assert selection.relative_cutoff_gap == pytest.approx(
        selection.cutoff_gap / selection.cutoff_score
    )
    assert selection.near_cutoff_epsilon == 1e-6
    assert selection.near_cutoff_count == 1


def test_phase4_rank_selection_topk_v1_reports_unselected_ties_and_near_cutoff_gap():
    selection = _select_phase4_frontier_rank_selection(
        feature_influences=torch.tensor([7.0, 0.5, 8.0, 1.0, 7.9999992, 8.0], dtype=torch.float64),
        visited=torch.zeros(6, dtype=torch.bool),
        frontier_size=1,
        ranker_mode="topk_v1",
    )

    assert selection.cutoff_score == 8.0
    assert selection.cutoff_gap == 0.0
    assert selection.relative_cutoff_gap == 0.0
    assert selection.near_cutoff_epsilon == 1e-6
    assert selection.near_cutoff_count == 2
    assert selection.tie_count_at_cutoff == 2
    assert selection.tie_at_cutoff is True


def test_phase4_rank_selection_nonbinding_has_no_cutoff_gap():
    selection = _select_phase4_frontier_rank_selection(
        feature_influences=torch.tensor([3.0, 2.0]),
        visited=torch.zeros(2, dtype=torch.bool),
        frontier_size=5,
        ranker_mode="argsort",
    )

    assert selection.cutoff_score == 2.0
    assert selection.cutoff_gap is None
    assert selection.relative_cutoff_gap is None
    assert selection.near_cutoff_epsilon is None
    assert selection.near_cutoff_count == 0


def test_phase4_rank_selection_empty_has_null_margin_telemetry():
    selection = _select_phase4_frontier_rank_selection(
        feature_influences=torch.tensor([3.0, 2.0]),
        visited=torch.ones(2, dtype=torch.bool),
        frontier_size=5,
        ranker_mode="argsort",
    )

    assert selection.cutoff_score is None
    assert selection.cutoff_gap is None
    assert selection.relative_cutoff_gap is None
    assert selection.near_cutoff_count == 0


def test_phase4_rank_selection_nonpositive_cutoff_has_no_relative_gap():
    selection = _select_phase4_frontier_rank_selection(
        feature_influences=torch.tensor([1.0, 0.0, -0.5]),
        visited=torch.zeros(3, dtype=torch.bool),
        frontier_size=2,
        ranker_mode="argsort",
    )

    assert selection.cutoff_score == 0.0
    assert selection.cutoff_gap == 0.5
    assert selection.relative_cutoff_gap is None
    assert selection.near_cutoff_count == 0


def test_phase4_rank_selection_max_feature_nodes_cap_bound_requires_budget_and_frontier_room():
    assert (
        _compute_phase4_rank_selection_max_feature_nodes_cap_bound(
            candidate_count=5,
            actual_max_feature_nodes=3,
            n_visited=0,
            max_frontier_size=3,
        )
        is True
    )


def test_phase4_rank_selection_max_feature_nodes_cap_bound_ignores_queue_window_only_clipping():
    assert (
        _compute_phase4_rank_selection_max_feature_nodes_cap_bound(
            candidate_count=5,
            actual_max_feature_nodes=10,
            n_visited=0,
            max_frontier_size=3,
        )
        is False
    )
