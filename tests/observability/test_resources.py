import torch

import circuit_tracer.observability.resources as resources


def test_resource_snapshot_soft_fails_without_cgroup(monkeypatch) -> None:
    monkeypatch.setattr(resources, "_resolve_cgroup_memory_dir", lambda: None)

    snapshot = resources.get_memory_snapshot(torch.device("cpu"))

    assert snapshot["cgroup_memory_current_gib"] is None
    assert snapshot["cgroup_memory_peak_gib"] is None


def test_memory_attrs_and_numeric_diff_are_available_directly() -> None:
    attrs = resources.build_memory_before_after_attrs(
        before={"rss_current_gib": 1.0},
        after={"rss_current_gib": 1.5},
        keys=("rss_current_gib",),
    )

    assert attrs["memory_delta_rss_current_gib"] == 0.5
    assert resources.diff_numeric_metrics({"nested": {"count": 2}}, {"nested": {"count": 5}}) == {
        "nested.count": 3
    }
