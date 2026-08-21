from __future__ import annotations

import pytest

from circuit_tracer.governor import discover_host_budget


MIB = 1024**2


def test_explicit_paths_and_slurm_choose_most_restrictive_finite_limit(tmp_path):
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    v2.write_text(str(12 * MIB))
    v1.write_text(str(20 * MIB))
    result = discover_host_budget(
        16 * MIB,
        environ={"SLURM_MEM_PER_NODE": "32"},
        cgroup_v2_path=v2,
        cgroup_v1_path=v1,
        proc_self_cgroup_path=None,
    )
    assert result.budget_bytes == 12 * MIB
    assert result.source == "cgroup_v2"
    assert {(item.source, item.bytes) for item in result.candidates} == {
        ("explicit_override", 16 * MIB),
        ("cgroup_v2", 12 * MIB),
        ("cgroup_v1", 20 * MIB),
        ("slurm_mem_per_node", 32 * MIB),
    }


def test_unlimited_missing_and_malformed_limits_are_warnings(tmp_path):
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    v2.write_text("max")
    v1.write_text(str(1 << 62))
    result = discover_host_budget(
        environ={"SLURM_MEM_PER_CPU": "bad", "SLURM_CPUS_ON_NODE": "8"},
        cgroup_v2_path=v2,
        cgroup_v1_path=v1,
        proc_self_cgroup_path=None,
    )
    assert result.budget_bytes is None
    assert any("cgroup_v2: unlimited" in warning for warning in result.warnings)
    assert any("cgroup_v1: unlimited sentinel" in warning for warning in result.warnings)
    assert any("malformed per-CPU" in warning for warning in result.warnings)


def test_slurm_per_cpu_multiplies_allocated_cpus(tmp_path):
    result = discover_host_budget(
        environ={"SLURM_MEM_PER_CPU": "4096", "SLURM_JOB_CPUS_PER_NODE": "8(x2)"},
        cgroup_v2_path=tmp_path / "missing-v2",
        cgroup_v1_path=tmp_path / "missing-v1",
        proc_self_cgroup_path=None,
    )
    assert result.source == "slurm_mem_per_cpu"
    assert result.budget_bytes == 4096 * 8 * MIB


def test_v2_unlimited_child_inherits_finite_parent_limit(tmp_path):
    root = tmp_path / "cgroup-v2"
    relative = "user.slice/user-1653998.slice/session-42.scope"
    child = root / relative
    parent = child.parent
    (root / "user.slice").mkdir(parents=True)
    child.mkdir(parents=True)
    (child / "memory.max").write_text("max")
    (parent / "memory.max").write_text(str(16 * MIB))
    (root / "user.slice/memory.max").write_text(str(32 * MIB))
    (root / "memory.max").write_text(str(64 * MIB))
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(f"0::/{relative}\n")

    result = discover_host_budget(
        environ={},
        cgroup_v2_path=None,
        cgroup_v1_path=None,
        proc_self_cgroup_path=proc,
        cgroup_v2_root=root,
        cgroup_v1_root=tmp_path / "unused-v1",
    )
    assert result.budget_bytes == 16 * MIB
    assert result.source == f"cgroup_v2:{parent / 'memory.max'}"
    assert any("session-42.scope/memory.max: unlimited" in item for item in result.warnings)


def test_nested_chpc_style_v1_walks_all_memory_ancestors(tmp_path):
    root = tmp_path / "cgroup-v1-memory"
    relative = "slurm/uid_1653998/job_987/step_0"
    leaf = root / relative
    job = leaf.parent
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text(str(48 * MIB))
    (job / "memory.limit_in_bytes").write_text(str(32 * MIB))
    (root / "slurm").mkdir(exist_ok=True)
    (root / "slurm/memory.limit_in_bytes").write_text(str(64 * MIB))
    (root / "memory.limit_in_bytes").write_text(str(128 * MIB))
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(f"5:cpu,cpuacct:/other\n7:memory:/{relative}\n")

    result = discover_host_budget(
        environ={},
        cgroup_v2_path=None,
        cgroup_v1_path=None,
        proc_self_cgroup_path=proc,
        cgroup_v2_root=tmp_path / "unused-v2",
        cgroup_v1_root=root,
    )
    assert result.budget_bytes == 32 * MIB
    assert result.source == f"cgroup_v1:{job / 'memory.limit_in_bytes'}"


def test_nonpositive_explicit_override_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        discover_host_budget(
            0,
            environ={},
            cgroup_v2_path=tmp_path / "missing-v2",
            cgroup_v1_path=tmp_path / "missing-v1",
            proc_self_cgroup_path=None,
        )
