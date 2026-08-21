from __future__ import annotations

from circuit_tracer.observability.resources import _get_linux_resource_snapshot


def _write_proc(proc, *, minor: int = 11, major: int = 13) -> None:
    # Fields after comm begin at field 3; minflt and majflt are fields 10 and 12.
    (proc / "stat").write_text(
        f"1 (python worker) S 0 0 0 0 0 0 {minor} 0 {major} 0 0 0\n",
        encoding="utf-8",
    )
    (proc / "io").write_text(
        "rchar: 1\nwchar: 2\nsyscr: 3\nsyscw: 4\nread_bytes: 5\nwrite_bytes: 6\n",
        encoding="utf-8",
    )
    (proc / "smaps_rollup").write_text(
        "Rss: 100 kB\nAnonymous: 40 kB\n",
        encoding="utf-8",
    )


def test_resource_snapshot_parses_cgroup_v1_and_proc(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_proc(proc)
    cgroup = tmp_path / "v1"
    cgroup.mkdir()
    (cgroup / "memory.usage_in_bytes").write_text("100\n", encoding="utf-8")
    (cgroup / "memory.failcnt").write_text("7\n", encoding="utf-8")
    (cgroup / "memory.stat").write_text(
        "\n".join(
            (
                "total_rss 1",
                "total_cache 2",
                "pgfault 3",
                "pgmajfault 4",
                "pgpgin 5",
                "pgpgout 6",
            )
        ),
        encoding="utf-8",
    )

    snapshot = _get_linux_resource_snapshot(
        proc_self_dir=str(proc), cgroup_dir=str(cgroup), cgroup_version=1
    )

    assert snapshot["resource_snapshot_schema_version"] == 1
    assert snapshot["proc_minor_faults"] == 11
    assert snapshot["proc_major_faults"] == 13
    assert snapshot["proc_read_bytes"] == 5
    assert snapshot["proc_smaps_anonymous_bytes"] == 40 * 1024
    assert snapshot["proc_smaps_file_backed_bytes"] == 60 * 1024
    assert snapshot["cgroup_v1_total_cache"] == 2
    assert snapshot["cgroup_v1_memory_failcnt"] == 7


def test_resource_snapshot_parses_cgroup_v2_workingset_and_reclaim(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_proc(proc)
    cgroup = tmp_path / "v2"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("100\n", encoding="utf-8")
    (cgroup / "memory.stat").write_text(
        "anon 10\nfile 20\nworkingset_refault_file 30\npgscan_direct 40\n",
        encoding="utf-8",
    )

    snapshot = _get_linux_resource_snapshot(
        proc_self_dir=str(proc), cgroup_dir=str(cgroup), cgroup_version=2
    )

    assert snapshot["cgroup_version"] == 2
    assert snapshot["cgroup_v2_anon"] == 10
    assert snapshot["cgroup_v2_file"] == 20
    assert snapshot["cgroup_v2_workingset_refault_file"] == 30
    assert snapshot["cgroup_v2_pgscan_direct"] == 40


def test_resource_snapshot_records_explicit_unavailable_provenance(tmp_path) -> None:
    snapshot = _get_linux_resource_snapshot(
        proc_self_dir=str(tmp_path / "missing"),
        cgroup_dir=str(tmp_path / "missing-cgroup"),
        cgroup_version=2,
    )

    assert snapshot["resource_snapshot_available"] is False
    unavailable = snapshot["resource_unavailable_fields"]
    assert "proc_minor_faults" in unavailable
    assert "cgroup_v2_anon" in unavailable
