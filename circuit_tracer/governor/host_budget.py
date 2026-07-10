from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_V1_MEMORY_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
CGROUP_V1_ROOT = Path("/sys/fs/cgroup/memory")
_V1_UNLIMITED_THRESHOLD = 1 << 60
_MIB = 1024 * 1024


@dataclass(frozen=True)
class HostBudgetCandidate:
    source: str
    bytes: int

    def __post_init__(self) -> None:
        if self.bytes <= 0:
            raise ValueError("candidate bytes must be positive")


@dataclass(frozen=True)
class HostBudgetDiscovery:
    budget_bytes: int | None
    source: str | None
    candidates: tuple[HostBudgetCandidate, ...]
    warnings: tuple[str, ...] = ()


def _read_raw(path: Path, source: str, warnings: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        warnings.append(f"{source}: missing {path}")
    except OSError as error:
        warnings.append(f"{source}: could not read {path}: {error}")
    return None


def _read_limit(path: Path, source: str, warnings: list[str]) -> HostBudgetCandidate | None:
    raw = _read_raw(path, source, warnings)
    if raw is None:
        return None
    if not raw:
        warnings.append(f"{source}: empty value")
        return None
    if raw.lower() == "max":
        warnings.append(f"{source}: unlimited")
        return None
    try:
        value = int(raw)
    except ValueError:
        warnings.append(f"{source}: malformed value {raw!r}")
        return None
    if value <= 0:
        warnings.append(f"{source}: nonpositive value {value}")
        return None
    if source.startswith("cgroup_v1") and value >= _V1_UNLIMITED_THRESHOLD:
        warnings.append(f"{source}: unlimited sentinel {value}")
        return None
    return HostBudgetCandidate(source, value)


def _parse_cpus(raw: str) -> int:
    return int(raw.split("(", 1)[0])


def _slurm_candidate(
    environ: Mapping[str, str], warnings: list[str]
) -> HostBudgetCandidate | None:
    per_node = environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            value = int(per_node) * _MIB
            if value <= 0:
                raise ValueError
        except ValueError:
            warnings.append(f"slurm: malformed SLURM_MEM_PER_NODE={per_node!r}")
        else:
            return HostBudgetCandidate("slurm_mem_per_node", value)
    per_cpu = environ.get("SLURM_MEM_PER_CPU")
    if not per_cpu:
        warnings.append("slurm: memory variables missing")
        return None
    cpu_raw = environ.get("SLURM_CPUS_ON_NODE") or environ.get("SLURM_JOB_CPUS_PER_NODE")
    if not cpu_raw:
        warnings.append("slurm: SLURM_MEM_PER_CPU set but CPU count missing")
        return None
    try:
        value = int(per_cpu) * _parse_cpus(cpu_raw) * _MIB
        if value <= 0:
            raise ValueError
    except ValueError:
        warnings.append(
            "slurm: malformed per-CPU memory or CPU count "
            f"({per_cpu!r}, {cpu_raw!r})"
        )
        return None
    return HostBudgetCandidate("slurm_mem_per_cpu", value)


def _parse_proc_cgroup(raw: str, warnings: list[str]) -> tuple[str | None, str | None]:
    v2_path: str | None = None
    v1_path: str | None = None
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            warnings.append(f"proc_cgroup: malformed line {line!r}")
            continue
        _, controllers, path = parts
        if not controllers:
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path
    return v2_path, v1_path


def _ancestor_limit_paths(root: Path, relative: str, filename: str) -> tuple[Path, ...]:
    leaf = root / relative.lstrip("/")
    paths: list[Path] = []
    current = leaf
    while current == root or current.is_relative_to(root):
        paths.append(current / filename)
        if current == root:
            break
        current = current.parent
    return tuple(paths)


def discover_host_budget(
    explicit_override_bytes: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cgroup_v2_path: Path | None = CGROUP_V2_MEMORY_MAX,
    cgroup_v1_path: Path | None = CGROUP_V1_MEMORY_LIMIT,
    proc_self_cgroup_path: Path | None = PROC_SELF_CGROUP,
    cgroup_v2_root: Path = CGROUP_V2_ROOT,
    cgroup_v1_root: Path = CGROUP_V1_ROOT,
) -> HostBudgetDiscovery:
    """Discover the minimum finite effective host limit across cgroups and Slurm."""

    warnings: list[str] = []
    candidates: list[HostBudgetCandidate] = []
    visited: set[Path] = set()
    if explicit_override_bytes is not None:
        if explicit_override_bytes <= 0:
            raise ValueError("explicit_override_bytes must be positive")
        candidates.append(HostBudgetCandidate("explicit_override", explicit_override_bytes))

    for path, source in (
        (cgroup_v2_path, "cgroup_v2"),
        (cgroup_v1_path, "cgroup_v1"),
    ):
        if path is not None:
            visited.add(path)
            candidate = _read_limit(path, source, warnings)
            if candidate is not None:
                candidates.append(candidate)

    if proc_self_cgroup_path is not None:
        raw = _read_raw(proc_self_cgroup_path, "proc_cgroup", warnings)
        if raw is not None:
            v2_relative, v1_relative = _parse_proc_cgroup(raw, warnings)
            nested = (
                (
                    "cgroup_v2",
                    _ancestor_limit_paths(cgroup_v2_root, v2_relative, "memory.max")
                    if v2_relative is not None
                    else (),
                ),
                (
                    "cgroup_v1",
                    _ancestor_limit_paths(
                        cgroup_v1_root, v1_relative, "memory.limit_in_bytes"
                    )
                    if v1_relative is not None
                    else (),
                ),
            )
            for kind, paths in nested:
                for path in paths:
                    if path in visited:
                        continue
                    visited.add(path)
                    source = f"{kind}:{path}"
                    candidate = _read_limit(path, source, warnings)
                    if candidate is not None:
                        candidates.append(candidate)

    slurm = _slurm_candidate(os.environ if environ is None else environ, warnings)
    if slurm is not None:
        candidates.append(slurm)
    if not candidates:
        warnings.append("no finite host-memory limit discovered")
        return HostBudgetDiscovery(None, None, (), tuple(warnings))
    chosen = min(candidates, key=lambda candidate: (candidate.bytes, candidate.source))
    return HostBudgetDiscovery(
        chosen.bytes,
        chosen.source,
        tuple(sorted(candidates, key=lambda candidate: candidate.source)),
        tuple(warnings),
    )
