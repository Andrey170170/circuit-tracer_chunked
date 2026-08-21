from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import cast

import torch

from circuit_tracer.observability.events import CudaMemoryProbe, CudaMemorySnapshot

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]


_DEFAULT_MEMORY_ATTR_KEYS: tuple[str, ...] = (
    "rss_current_gib",
    "proc_rss_anon_gib",
    "proc_rss_file_gib",
    "cgroup_memory_current_gib",
    "cgroup_memory_anon_gib",
    "cgroup_memory_file_gib",
    "cuda_allocated_gib",
    "cuda_reserved_gib",
    "proc_minor_faults",
    "proc_major_faults",
    "proc_read_bytes",
    "proc_read_syscalls",
    "proc_write_bytes",
    "proc_write_syscalls",
    "proc_smaps_anonymous_bytes",
    "proc_smaps_file_backed_bytes",
)

def _format_optional_gib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} GiB"


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def _kib_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**2)


def _read_file_first_line(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()
    except (FileNotFoundError, OSError):
        return None


def _parse_memory_bytes_value(raw_value: str | None) -> int | None:
    if not raw_value:
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"", "max"}:
        return None
    try:
        value = int(normalized)
    except ValueError:
        return None
    return value if value >= 0 else None


def _read_memory_bytes_file(path: str) -> int | None:
    return _parse_memory_bytes_value(_read_file_first_line(path))


def _read_memory_stat_file(path: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                key, value_text = parts
                parsed = _parse_memory_bytes_value(value_text)
                if parsed is not None:
                    stats[key] = parsed
    except (FileNotFoundError, OSError):
        return {}
    return stats


def _read_key_value_file(path: str) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" in line:
                    key, raw = line.split(":", 1)
                else:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    key, raw = parts
                token = raw.strip().split()[0] if raw.strip() else ""
                try:
                    values[key] = int(token)
                except ValueError:
                    continue
    except (FileNotFoundError, OSError):
        pass
    return values


def _get_linux_resource_snapshot(
    *,
    proc_self_dir: str = "/proc/self",
    cgroup_dir: str | None = None,
    cgroup_version: int | None = None,
) -> dict[str, object]:
    """Collect versioned fault, I/O, mapping, and cgroup counters with provenance."""

    unavailable: list[str] = []
    snapshot: dict[str, object] = {"resource_snapshot_schema_version": 1}
    stat = _read_file_first_line(os.path.join(proc_self_dir, "stat"))
    tail = stat[stat.rfind(")") + 2 :].split() if stat and ")" in stat else []
    for key, index in (("proc_minor_faults", 7), ("proc_major_faults", 9)):
        try:
            snapshot[key] = int(tail[index])
        except (IndexError, ValueError):
            snapshot[key] = None
            unavailable.append(key)

    io = _read_key_value_file(os.path.join(proc_self_dir, "io"))
    for source, target in (
        ("read_bytes", "proc_read_bytes"),
        ("syscr", "proc_read_syscalls"),
        ("write_bytes", "proc_write_bytes"),
        ("syscw", "proc_write_syscalls"),
    ):
        snapshot[target] = io.get(source)
        if source not in io:
            unavailable.append(target)

    smaps = _read_key_value_file(os.path.join(proc_self_dir, "smaps_rollup"))
    anonymous_kib = smaps.get("Anonymous")
    rss_kib = smaps.get("Rss")
    snapshot["proc_smaps_anonymous_bytes"] = (
        anonymous_kib * 1024 if anonymous_kib is not None else None
    )
    snapshot["proc_smaps_file_backed_bytes"] = (
        max(rss_kib - anonymous_kib, 0) * 1024
        if rss_kib is not None and anonymous_kib is not None
        else None
    )
    for key in ("proc_smaps_anonymous_bytes", "proc_smaps_file_backed_bytes"):
        if snapshot[key] is None:
            unavailable.append(key)

    resolved = cgroup_dir if cgroup_dir is not None else _resolve_cgroup_memory_dir()
    if cgroup_version is None and resolved is not None:
        if os.path.isfile(os.path.join(resolved, "memory.current")):
            cgroup_version = 2
        elif os.path.isfile(os.path.join(resolved, "memory.usage_in_bytes")):
            cgroup_version = 1
    snapshot["cgroup_version"] = cgroup_version
    snapshot["cgroup_source"] = resolved
    cgroup_fields: dict[str, int | None] = {}
    if resolved is None or cgroup_version not in {1, 2}:
        unavailable.append("cgroup")
    elif cgroup_version == 1:
        stats = _read_memory_stat_file(os.path.join(resolved, "memory.stat"))
        for key in (
            "total_rss",
            "total_cache",
            "pgfault",
            "pgmajfault",
            "pgpgin",
            "pgpgout",
        ):
            cgroup_fields[f"cgroup_v1_{key}"] = stats.get(key)
        cgroup_fields["cgroup_v1_memory_failcnt"] = _read_memory_bytes_file(
            os.path.join(resolved, "memory.failcnt")
        )
    else:
        stats = _read_memory_stat_file(os.path.join(resolved, "memory.stat"))
        selected = ("anon", "file") + tuple(
            key
            for key in stats
            if key.startswith("workingset_")
            or key.startswith("pgscan")
            or key.startswith("pgsteal")
        )
        for key in dict.fromkeys(selected):
            cgroup_fields[f"cgroup_v2_{key}"] = stats.get(key)
    snapshot.update(cgroup_fields)
    for key, value in cgroup_fields.items():
        if value is None:
            unavailable.append(key)
    snapshot["resource_unavailable_fields"] = tuple(sorted(set(unavailable)))
    snapshot["resource_snapshot_available"] = not unavailable
    return snapshot


def _get_current_rss_gib_from_proc() -> float | None:
    """Best-effort Linux RSS snapshot using /proc/self/statm."""

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0:
        return None

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.readline().split()
        if len(fields) < 2:
            return None
        resident_pages = int(fields[1])
    except (FileNotFoundError, OSError, ValueError):
        return None

    return (resident_pages * page_size) / (1024**3)


def _get_process_status_memory_gib_from_proc() -> dict[str, float | None]:
    status_fields: dict[str, float | None] = {
        "proc_rss_gib": None,
        "proc_rss_anon_gib": None,
        "proc_rss_file_gib": None,
        "proc_rss_shmem_gib": None,
    }
    field_map = {
        "VmRSS": "proc_rss_gib",
        "RssAnon": "proc_rss_anon_gib",
        "RssFile": "proc_rss_file_gib",
        "RssShmem": "proc_rss_shmem_gib",
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                attr_name = field_map.get(key)
                if attr_name is None:
                    continue
                value_parts = raw_value.strip().split()
                if not value_parts:
                    continue
                try:
                    kib_value = int(value_parts[0])
                except ValueError:
                    continue
                if kib_value < 0:
                    continue
                status_fields[attr_name] = _kib_to_gib(kib_value)
    except (FileNotFoundError, OSError):
        return status_fields

    return status_fields


def _resolve_cgroup_memory_dir() -> str | None:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            cgroup_lines = handle.readlines()
    except (FileNotFoundError, OSError):
        cgroup_lines = []

    unified_path: str | None = None
    legacy_memory_path: str | None = None
    for line in cgroup_lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        if controllers == "":
            unified_path = rel_path
            break
        controller_set = set(filter(None, controllers.split(",")))
        if "memory" in controller_set:
            legacy_memory_path = rel_path

    if unified_path is not None:
        base = "/sys/fs/cgroup"
        candidate = (
            os.path.normpath(os.path.join(base, unified_path.lstrip("/")))
            if unified_path not in {"", "/"}
            else base
        )
        if os.path.isdir(candidate):
            return candidate

    if legacy_memory_path is not None:
        for base in ("/sys/fs/cgroup/memory", "/sys/fs/cgroup"):
            candidate = (
                os.path.normpath(os.path.join(base, legacy_memory_path.lstrip("/")))
                if legacy_memory_path not in {"", "/"}
                else base
            )
            if os.path.isdir(candidate):
                return candidate

    fallback = "/sys/fs/cgroup"
    if os.path.isfile(os.path.join(fallback, "memory.current")):
        return fallback
    return None


def _get_cgroup_memory_snapshot_gib() -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
        "cgroup_memory_current_gib": None,
        "cgroup_memory_limit_gib": None,
        "cgroup_memory_headroom_gib": None,
        "cgroup_memory_peak_gib": None,
        "cgroup_memory_anon_gib": None,
        "cgroup_memory_file_gib": None,
        "cgroup_memory_active_file_gib": None,
        "cgroup_memory_inactive_file_gib": None,
        "cgroup_memory_shmem_gib": None,
        "cgroup_memory_slab_reclaimable_gib": None,
        "cgroup_memory_slab_unreclaimable_gib": None,
    }
    cgroup_dir = _resolve_cgroup_memory_dir()
    if cgroup_dir is None:
        return snapshot

    is_v2 = os.path.isfile(os.path.join(cgroup_dir, "memory.current"))
    is_v1 = os.path.isfile(os.path.join(cgroup_dir, "memory.usage_in_bytes"))
    if is_v2:
        current_bytes = _read_memory_bytes_file(os.path.join(cgroup_dir, "memory.current"))
        limit_bytes = _read_memory_bytes_file(os.path.join(cgroup_dir, "memory.max"))
        peak_bytes = _read_memory_bytes_file(os.path.join(cgroup_dir, "memory.peak"))
    elif is_v1:
        current_bytes = _read_memory_bytes_file(
            os.path.join(cgroup_dir, "memory.usage_in_bytes")
        )
        peak_bytes = _read_memory_bytes_file(
            os.path.join(cgroup_dir, "memory.max_usage_in_bytes")
        )
        # Slurm commonly leaves the leaf task cgroup effectively unlimited and
        # applies the allocation limit at the enclosing step/job cgroup.
        # Resolve the smallest finite inherited limit instead of reporting the
        # kernel's huge v1 unlimited sentinel as real headroom.
        finite_limits: list[int] = []
        current_dir = os.path.abspath(cgroup_dir)
        while True:
            candidate = _read_memory_bytes_file(
                os.path.join(current_dir, "memory.limit_in_bytes")
            )
            if candidate is not None and candidate < 1 << 60:
                finite_limits.append(candidate)
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent
        limit_bytes = min(finite_limits) if finite_limits else None
    else:
        return snapshot

    snapshot["cgroup_memory_current_gib"] = _bytes_to_gib(current_bytes)
    snapshot["cgroup_memory_limit_gib"] = _bytes_to_gib(limit_bytes)
    snapshot["cgroup_memory_headroom_gib"] = _bytes_to_gib(
        max(limit_bytes - current_bytes, 0)
        if current_bytes is not None and limit_bytes is not None
        else None
    )
    snapshot["cgroup_memory_peak_gib"] = _bytes_to_gib(peak_bytes)

    memory_stats = _read_memory_stat_file(os.path.join(cgroup_dir, "memory.stat"))
    if is_v2:
        stat_keys = {
            "anon": "cgroup_memory_anon_gib",
            "file": "cgroup_memory_file_gib",
            "active_file": "cgroup_memory_active_file_gib",
            "inactive_file": "cgroup_memory_inactive_file_gib",
            "shmem": "cgroup_memory_shmem_gib",
            "slab_reclaimable": "cgroup_memory_slab_reclaimable_gib",
            "slab_unreclaimable": "cgroup_memory_slab_unreclaimable_gib",
        }
        for stat_key, attr_name in stat_keys.items():
            snapshot[attr_name] = _bytes_to_gib(memory_stats.get(stat_key))
    else:
        stat_keys = {
            "rss": "cgroup_memory_anon_gib",
            "cache": "cgroup_memory_file_gib",
            "active_file": "cgroup_memory_active_file_gib",
            "inactive_file": "cgroup_memory_inactive_file_gib",
            "shmem": "cgroup_memory_shmem_gib",
            "slab_reclaimable": "cgroup_memory_slab_reclaimable_gib",
            "slab_unreclaimable": "cgroup_memory_slab_unreclaimable_gib",
        }
        for stat_key, attr_name in stat_keys.items():
            value = memory_stats.get(f"total_{stat_key}", memory_stats.get(stat_key))
            snapshot[attr_name] = _bytes_to_gib(value)

    return snapshot


def _coerce_finite_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def build_memory_snapshot_attrs(
    snapshot: Mapping[str, object] | None,
    *,
    keys: Sequence[str] | None = None,
    prefix: str = "memory",
) -> dict[str, float | None]:
    selected_keys = tuple(keys) if keys is not None else _DEFAULT_MEMORY_ATTR_KEYS
    attrs: dict[str, float | None] = {}
    for key in selected_keys:
        value = snapshot.get(key) if snapshot is not None else None
        attrs[f"{prefix}_{key}"] = _coerce_finite_float(value)
    return attrs


def build_memory_before_after_attrs(
    *,
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    keys: Sequence[str] | None = None,
    before_prefix: str = "memory_before",
    after_prefix: str = "memory_after",
    delta_prefix: str = "memory_delta",
) -> dict[str, float | None]:
    selected_keys = tuple(keys) if keys is not None else _DEFAULT_MEMORY_ATTR_KEYS
    attrs: dict[str, float | None] = {}
    attrs.update(build_memory_snapshot_attrs(before, keys=selected_keys, prefix=before_prefix))
    attrs.update(build_memory_snapshot_attrs(after, keys=selected_keys, prefix=after_prefix))
    for key in selected_keys:
        before_value = _coerce_finite_float(before.get(key) if before is not None else None)
        after_value = _coerce_finite_float(after.get(key) if after is not None else None)
        attrs[f"{delta_prefix}_{key}"] = (
            (after_value - before_value)
            if before_value is not None and after_value is not None
            else None
        )
    return attrs


def get_memory_snapshot(device: torch.device | None = None) -> dict[str, object]:
    rss_current_gib = _get_current_rss_gib_from_proc()
    rss_gib = None
    if resource is not None:
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    snapshot: dict[str, float | None] = {
        "rss_current_gib": rss_current_gib,
        "rss_gib": rss_gib,
        "proc_rss_gib": None,
        "proc_rss_anon_gib": None,
        "proc_rss_file_gib": None,
        "proc_rss_shmem_gib": None,
        "cgroup_memory_current_gib": None,
        "cgroup_memory_limit_gib": None,
        "cgroup_memory_headroom_gib": None,
        "cgroup_memory_peak_gib": None,
        "cgroup_memory_anon_gib": None,
        "cgroup_memory_file_gib": None,
        "cgroup_memory_active_file_gib": None,
        "cgroup_memory_inactive_file_gib": None,
        "cgroup_memory_shmem_gib": None,
        "cgroup_memory_slab_reclaimable_gib": None,
        "cgroup_memory_slab_unreclaimable_gib": None,
        "cuda_allocated_gib": None,
        "cuda_reserved_gib": None,
        "cuda_max_allocated_gib": None,
        "cuda_max_reserved_gib": None,
    }
    snapshot.update(_get_process_status_memory_gib_from_proc())
    snapshot.update(_get_cgroup_memory_snapshot_gib())
    snapshot.update(_get_linux_resource_snapshot())

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                "cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "cuda_max_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
        )

    return snapshot


def probe_cuda_memory(probe: CudaMemoryProbe) -> CudaMemorySnapshot:
    """Execute one typed CUDA allocator operation for domain code."""

    if not torch.cuda.is_available():
        return CudaMemorySnapshot(available=False)

    device = torch.cuda.current_device() if probe.device is None else probe.device
    if probe.operation == "reset_peak":
        torch.cuda.reset_peak_memory_stats(device)
    elif probe.operation == "synchronize":
        torch.cuda.synchronize(device)
    elif probe.operation != "snapshot":
        raise ValueError(f"unsupported CUDA memory probe: {probe.operation}")

    return CudaMemorySnapshot(
        available=True,
        current_reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        total_bytes=int(torch.cuda.get_device_properties(device).total_memory),
    )


def format_memory_snapshot(
    device: torch.device | None = None, extra: Mapping[str, object] | None = None
) -> str:
    snapshot = get_memory_snapshot(device)
    parts = [
        f"rss={_format_optional_gib(snapshot['rss_gib'])}",
        f"rss_current={_format_optional_gib(snapshot['rss_current_gib'])}",
        f"proc_anon={_format_optional_gib(snapshot['proc_rss_anon_gib'])}",
        f"proc_file={_format_optional_gib(snapshot['proc_rss_file_gib'])}",
        f"cg_current={_format_optional_gib(snapshot['cgroup_memory_current_gib'])}",
        f"cg_peak={_format_optional_gib(snapshot['cgroup_memory_peak_gib'])}",
        f"cg_anon={_format_optional_gib(snapshot['cgroup_memory_anon_gib'])}",
        f"cg_file={_format_optional_gib(snapshot['cgroup_memory_file_gib'])}",
        f"cuda_alloc={_format_optional_gib(snapshot['cuda_allocated_gib'])}",
        f"cuda_reserved={_format_optional_gib(snapshot['cuda_reserved_gib'])}",
        f"cuda_peak_alloc={_format_optional_gib(snapshot['cuda_max_allocated_gib'])}",
        f"cuda_peak_reserved={_format_optional_gib(snapshot['cuda_max_reserved_gib'])}",
    ]
    if extra:
        parts.extend(f"{key}={value}" for key, value in extra.items())
    return ", ".join(parts)


def flatten_numeric_metrics(
    metrics: Mapping[str, object], prefix: str = ""
) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_numeric_metrics(cast(Mapping[str, object], value), full_key))
        elif isinstance(value, Number) and not isinstance(value, bool):
            flat[full_key] = cast(float | int, value)
    return flat


def diff_numeric_metrics(
    before: Mapping[str, object] | None, after: Mapping[str, object]
) -> dict[str, float | int]:
    after_flat = flatten_numeric_metrics(after)
    if before is None:
        return after_flat

    before_flat = flatten_numeric_metrics(before)
    diff: dict[str, float | int] = {}
    for key, value in after_flat.items():
        baseline = before_flat.get(key, 0)
        delta = value - baseline
        if delta:
            diff[key] = delta
    return diff


def format_numeric_metrics(metrics: Mapping[str, object], limit: int | None = None) -> str:
    flat = flatten_numeric_metrics(metrics)
    items = list(flat.items())
    if limit is not None:
        items = items[:limit]
    return ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in items
    )
