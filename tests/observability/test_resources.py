import torch

import circuit_tracer.observability.resources as resources


def test_resource_snapshot_soft_fails_without_cgroup(monkeypatch) -> None:
    monkeypatch.setattr(resources, "_resolve_cgroup_memory_dir", lambda: None)

    snapshot = resources.get_memory_snapshot(torch.device("cpu"))

    assert snapshot["cgroup_memory_current_gib"] is None
    assert snapshot["cgroup_memory_peak_gib"] is None


def test_resource_snapshot_maps_cgroup_v1_to_generic_memory_fields(
    monkeypatch, tmp_path
) -> None:
    gib = 1024**3
    job = tmp_path / "job"
    task = job / "task"
    task.mkdir(parents=True)
    (job / "memory.limit_in_bytes").write_text(str(10 * gib), encoding="utf-8")
    (task / "memory.limit_in_bytes").write_text(str(2**63 - 4096), encoding="utf-8")
    (task / "memory.usage_in_bytes").write_text(str(3 * gib), encoding="utf-8")
    (task / "memory.max_usage_in_bytes").write_text(str(5 * gib), encoding="utf-8")
    (task / "memory.stat").write_text(
        "\n".join(
            (
                f"total_rss {2 * gib}",
                f"total_cache {gib}",
                f"total_active_file {gib // 2}",
                f"total_inactive_file {gib // 4}",
                "total_shmem 4096",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(resources, "_resolve_cgroup_memory_dir", lambda: str(task))

    snapshot = resources.get_memory_snapshot(torch.device("cpu"))

    assert snapshot["cgroup_memory_current_gib"] == 3.0
    assert snapshot["cgroup_memory_limit_gib"] == 10.0
    assert snapshot["cgroup_memory_headroom_gib"] == 7.0
    assert snapshot["cgroup_memory_peak_gib"] == 5.0
    assert snapshot["cgroup_memory_anon_gib"] == 2.0
    assert snapshot["cgroup_memory_file_gib"] == 1.0
    assert snapshot["cgroup_memory_active_file_gib"] == 0.5
    assert snapshot["cgroup_memory_inactive_file_gib"] == 0.25
    assert snapshot["cgroup_memory_shmem_gib"] == 4096 / gib


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
