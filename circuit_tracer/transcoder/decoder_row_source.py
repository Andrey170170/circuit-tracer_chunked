"""Provider-neutral, exact selective decoder-row access for safetensors."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import ceil, prod
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

import torch

from circuit_tracer.transcoder.checkpoint_manifest import parse_safetensors_payloads


_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
}


def safetensors_decoder_tensor_supported(
    path: str | os.PathLike[str],
    tensor_key: str,
    *,
    expected_shape: tuple[int, ...],
) -> bool:
    """Return whether one authoritative tensor has a supported exact layout."""

    header = parse_safetensors_payloads(path)
    if not header.ok:
        return False
    payload = next(
        (item for item in header.payloads if item.key == tensor_key),
        None,
    )
    if payload is None or payload.dtype not in _DTYPES:
        return False
    if payload.shape != expected_shape or len(payload.shape) not in {2, 3}:
        return False
    return prod(payload.shape) * _DTYPES[payload.dtype][1] == payload.length


class DecoderRowOrder(str, Enum):
    CALLER = "caller"
    SORTED_UNIQUE = "sorted_unique"


class DecoderRowRefusalCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_KEY = "invalid_key"
    UNSUPPORTED_LAYOUT = "unsupported_layout"
    SOURCE_MISMATCH = "source_mismatch"
    BUDGET_TOO_SMALL = "budget_too_small"
    PER_ROW_REQUESTS = "per_row_requests"
    UNSUPPORTED_DESTINATION = "unsupported_destination"
    RELEASED = "released"


@dataclass(frozen=True, order=True, slots=True)
class DecoderRowKey:
    """One decoder vector, independent of provider topology."""

    source_layer: int
    feature_id: int
    output_slot: int = 0


@dataclass(frozen=True, slots=True)
class DecoderTensorSpec:
    """Provider declaration binding a source layer to one checkpoint tensor."""

    source_layer: int
    path: Path
    tensor_key: str
    expected_source_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class DecoderTensorIdentity:
    source_layer: int
    path: str
    tensor_key: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    payload_offset: int
    payload_length: int
    dtype: str
    shape: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DecoderRowSourceFingerprint:
    value: str
    tensors: tuple[DecoderTensorIdentity, ...]


@dataclass(frozen=True, slots=True)
class DecoderRowEstimate:
    occurrence_row_count: int
    unique_row_count: int
    requested_row_bytes: int
    occurrence_row_bytes: int
    backend_materialized_bytes: int
    backend_request_count: int
    block_count: int
    mapping_count: int
    range_count: int
    read_count: int
    planned_overfetch_ratio: float
    output_bytes: int
    temporary_staging_high_water_bytes: int


@dataclass(frozen=True, slots=True)
class DecoderRowPlan:
    caller_keys: tuple[DecoderRowKey, ...]
    sorted_unique_keys: tuple[DecoderRowKey, ...]
    inverse_order: tuple[int, ...]
    destination: torch.device
    order: DecoderRowOrder
    output_dtype: torch.dtype
    estimate: DecoderRowEstimate


@dataclass(frozen=True, slots=True)
class DecoderRowRefusal:
    code: DecoderRowRefusalCode
    reason: str


@dataclass(frozen=True, slots=True)
class DecoderRowPlanResult:
    plan: DecoderRowPlan | None = None
    refusal: DecoderRowRefusal | None = None

    @property
    def ok(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True, slots=True)
class DecoderRowMaterializationTelemetry:
    outcome: Literal["materialized"]
    occurrence_row_count: int
    unique_row_count: int
    requested_row_bytes: int
    occurrence_row_bytes: int
    backend_materialized_bytes: int
    backend_request_count: int
    block_count: int
    mapping_count: int
    mapping_open_count: int
    range_count: int
    read_count: int
    planned_overfetch_ratio: float
    output_bytes: int
    temporary_staging_high_water_bytes: int
    planning_seconds: float
    mapping_seconds: float
    fault_read_seconds: float
    gather_seconds: float
    reorder_seconds: float
    h2d_seconds: float
    h2d_bytes: int
    total_seconds: float
    os_physical_read_estimate: int | None = None


@dataclass(frozen=True, slots=True)
class DecoderRowReleaseTelemetry:
    outcome: Literal["released", "already_released"]
    reason: str
    mapping_count: int
    handle_count: int
    mapped_address_span_bytes: int
    release_seconds: float


@dataclass(frozen=True, slots=True)
class DecoderRowSeed:
    rows: torch.Tensor
    keys: tuple[DecoderRowKey, ...]
    telemetry: DecoderRowMaterializationTelemetry


DecoderRowTelemetry = (
    DecoderRowMaterializationTelemetry | DecoderRowReleaseTelemetry | DecoderRowRefusal
)


class DecoderRowSource(Protocol):
    """Provider-facing selective row-source contract."""

    @property
    def fingerprint(self) -> DecoderRowSourceFingerprint: ...

    def estimate(
        self,
        keys: Sequence[DecoderRowKey],
        *,
        destination: str | torch.device = "cpu",
        order: DecoderRowOrder = DecoderRowOrder.CALLER,
        output_dtype: torch.dtype | None = None,
    ) -> DecoderRowPlanResult: ...

    def materialize(
        self,
        keys: Sequence[DecoderRowKey],
        *,
        destination: str | torch.device = "cpu",
        order: DecoderRowOrder = DecoderRowOrder.CALLER,
        output_dtype: torch.dtype | None = None,
    ) -> DecoderRowSeed | DecoderRowRefusal: ...

    def release(self, reason: str) -> DecoderRowReleaseTelemetry: ...


@dataclass(frozen=True, slots=True)
class _Tensor:
    identity: DecoderTensorIdentity
    torch_dtype: torch.dtype
    element_bytes: int
    feature_count: int
    output_count: int
    vector_width: int

    @property
    def row_bytes(self) -> int:
        return self.vector_width * self.element_bytes


class MappedSafetensorsDecoderRowSource:
    """Long-lived mappings with bounded, sorted vectorized gathers.

    Supported tensors are contiguous rank-2 ``[feature, width]`` and rank-3
    ``[feature, output_slot, width]`` payloads in BF16, F16, or F32.
    Mappings use ``ACCESS_COPY`` solely to give PyTorch a writable buffer view;
    gathered rows are copies and the source never writes to the mapping.
    """

    def __init__(
        self,
        specs: Sequence[DecoderTensorSpec],
        *,
        max_staging_bytes: int,
        telemetry: Callable[[DecoderRowTelemetry], None] | None = None,
    ) -> None:
        if max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive")
        self._max_staging_bytes = int(max_staging_bytes)
        self._telemetry = telemetry
        self._tensors = self._inspect_specs(specs)
        self._descriptors: dict[Path, int] = {}
        try:
            for tensor in self._tensors.values():
                path = Path(tensor.identity.path)
                if path in self._descriptors:
                    continue
                descriptor = os.open(path, os.O_RDONLY)
                stat = os.fstat(descriptor)
                observed = (stat.st_dev, stat.st_ino, stat.st_size)
                expected = (
                    tensor.identity.device,
                    tensor.identity.inode,
                    tensor.identity.size,
                )
                if observed != expected:
                    os.close(descriptor)
                    raise ValueError(f"decoder source changed while opening {path}")
                self._descriptors[path] = descriptor
        except BaseException:
            for descriptor in self._descriptors.values():
                os.close(descriptor)
            raise
        self._mappings: dict[Path, mmap.mmap] = {}
        self._released = False
        identities = tuple(item.identity for _, item in sorted(self._tensors.items()))
        self._fingerprint = DecoderRowSourceFingerprint(
            _fingerprint([identity.fingerprint for identity in identities]), identities
        )

    @property
    def fingerprint(self) -> DecoderRowSourceFingerprint:
        return self._fingerprint

    def estimate(
        self,
        keys: Sequence[DecoderRowKey],
        *,
        destination: str | torch.device = "cpu",
        order: DecoderRowOrder = DecoderRowOrder.CALLER,
        output_dtype: torch.dtype | None = None,
    ) -> DecoderRowPlanResult:
        if self._released:
            return self._refuse(DecoderRowRefusalCode.RELEASED, "decoder row source is released")
        try:
            device = torch.device(destination)
        except (RuntimeError, TypeError) as exc:
            return self._refuse(DecoderRowRefusalCode.UNSUPPORTED_DESTINATION, str(exc))
        if device.type not in {"cpu", "cuda"}:
            return self._refuse(
                DecoderRowRefusalCode.UNSUPPORTED_DESTINATION,
                f"destination must be cpu or cuda, got {device}",
            )
        try:
            normalized_order = DecoderRowOrder(order)
        except ValueError:
            return self._refuse(
                DecoderRowRefusalCode.INVALID_KEY,
                f"unsupported decoder row order: {order!r}",
            )
        normalized = tuple(keys)
        if not normalized:
            return self._refuse(DecoderRowRefusalCode.INVALID_KEY, "at least one row key is required")
        for key in normalized:
            tensor = self._tensors.get(key.source_layer)
            if tensor is None:
                return self._refuse(
                    DecoderRowRefusalCode.INVALID_SOURCE,
                    f"unknown decoder source layer {key.source_layer}",
                )
            if not 0 <= key.feature_id < tensor.feature_count:
                return self._refuse(
                    DecoderRowRefusalCode.INVALID_KEY,
                    f"feature_id {key.feature_id} is outside source layer "
                    f"{key.source_layer} coverage [0, {tensor.feature_count})",
                )
            if not 0 <= key.output_slot < tensor.output_count:
                return self._refuse(
                    DecoderRowRefusalCode.INVALID_KEY,
                    f"output_slot {key.output_slot} is outside source layer "
                    f"{key.source_layer} coverage [0, {tensor.output_count})",
                )

        sorted_unique = tuple(sorted(set(normalized)))
        unique_index = {key: index for index, key in enumerate(sorted_unique)}
        inverse = tuple(unique_index[key] for key in normalized)
        dtypes = {self._tensors[key.source_layer].torch_dtype for key in sorted_unique}
        widths = {self._tensors[key.source_layer].vector_width for key in sorted_unique}
        if len(dtypes) != 1 or len(widths) != 1:
            return self._refuse(
                DecoderRowRefusalCode.UNSUPPORTED_LAYOUT,
                "one materialization requires a common dtype and vector width",
            )
        raw_dtype = next(iter(dtypes))
        normalized_output_dtype = raw_dtype if output_dtype is None else output_dtype
        if normalized_output_dtype not in {item[0] for item in _DTYPES.values()}:
            return self._refuse(
                DecoderRowRefusalCode.UNSUPPORTED_LAYOUT,
                f"unsupported decoder output dtype: {normalized_output_dtype}",
            )
        output_element_bytes = torch.empty(
            (), dtype=normalized_output_dtype
        ).element_size()

        groups: dict[int, list[DecoderRowKey]] = defaultdict(list)
        for key in sorted_unique:
            groups[key.source_layer].append(key)
        backend_requests = 0
        block_ids: set[tuple[int, int]] = set()
        backend_bytes = 0
        staging_high_water = 0
        for source_layer, group in groups.items():
            tensor = self._tensors[source_layer]
            capacity = self._max_staging_bytes // tensor.row_bytes
            if capacity < 1:
                return self._refuse(
                    DecoderRowRefusalCode.BUDGET_TOO_SMALL,
                    f"staging budget {self._max_staging_bytes} is smaller than one "
                    f"{tensor.row_bytes}-byte decoder row",
                )
            if len(group) > 1 and capacity < 2:
                return self._refuse(
                    DecoderRowRefusalCode.PER_ROW_REQUESTS,
                    "planned gather would degenerate to one backend request per row",
                )
            backend_requests += ceil(len(group) / capacity)
            chunk_rows = min(len(group), capacity)
            raw_staging_bytes = chunk_rows * tensor.row_bytes
            cast_staging_bytes = (
                chunk_rows * tensor.vector_width * output_element_bytes
                if normalized_output_dtype != tensor.torch_dtype
                else 0
            )
            staging_high_water = max(
                staging_high_water,
                raw_staging_bytes + cast_staging_bytes,
            )
            backend_bytes += len(group) * tensor.row_bytes
            for key in group:
                flat_row = key.feature_id * tensor.output_count + key.output_slot
                start = tensor.identity.payload_offset + flat_row * tensor.row_bytes
                end = start + tensor.row_bytes - 1
                first_page = start // mmap.PAGESIZE
                last_page = end // mmap.PAGESIZE
                block_ids.update(
                    (source_layer, page) for page in range(first_page, last_page + 1)
                )
        occurrence_bytes = sum(
            self._tensors[key.source_layer].row_bytes for key in normalized
        )
        estimate = DecoderRowEstimate(
            occurrence_row_count=len(normalized),
            unique_row_count=len(sorted_unique),
            requested_row_bytes=backend_bytes,
            occurrence_row_bytes=occurrence_bytes,
            backend_materialized_bytes=backend_bytes,
            backend_request_count=backend_requests,
            block_count=len(block_ids),
            mapping_count=len(groups),
            range_count=0,
            read_count=0,
            planned_overfetch_ratio=0.0,
            output_bytes=(
                len(normalized) * next(iter(widths)) * output_element_bytes
                if normalized_order is DecoderRowOrder.CALLER
                else len(sorted_unique) * next(iter(widths)) * output_element_bytes
            ),
            temporary_staging_high_water_bytes=staging_high_water,
        )
        return DecoderRowPlanResult(
            plan=DecoderRowPlan(
                normalized,
                sorted_unique,
                inverse,
                device,
                normalized_order,
                normalized_output_dtype,
                estimate,
            )
        )

    def materialize(
        self,
        keys: Sequence[DecoderRowKey],
        *,
        destination: str | torch.device = "cpu",
        order: DecoderRowOrder = DecoderRowOrder.CALLER,
        output_dtype: torch.dtype | None = None,
    ) -> DecoderRowSeed | DecoderRowRefusal:
        started = perf_counter()
        planned = self.estimate(
            keys,
            destination=destination,
            order=order,
            output_dtype=output_dtype,
        )
        planning_done = perf_counter()
        if not planned.ok:
            assert planned.refusal is not None
            return planned.refusal
        assert planned.plan is not None
        plan = planned.plan
        mismatch = self._verify_identities()
        if mismatch is not None:
            return mismatch

        grouped: dict[int, list[tuple[int, DecoderRowKey]]] = defaultdict(list)
        for sorted_index, key in enumerate(plan.sorted_unique_keys):
            grouped[key.source_layer].append((sorted_index, key))
        first_tensor = self._tensors[plan.sorted_unique_keys[0].source_layer]
        unique_rows = torch.empty(
            (len(plan.sorted_unique_keys), first_tensor.vector_width),
            dtype=plan.output_dtype,
        )
        mapping_seconds = 0.0
        gather_seconds = 0.0
        opened = 0
        for source_layer, entries in grouped.items():
            tensor = self._tensors[source_layer]
            mapping_started = perf_counter()
            mapping, was_opened = self._mapping_for(Path(tensor.identity.path))
            mapping_seconds += perf_counter() - mapping_started
            opened += int(was_opened)
            capacity = self._max_staging_bytes // tensor.row_bytes
            for start in range(0, len(entries), capacity):
                chunk = entries[start : start + capacity]
                flat_ids = torch.tensor(
                    [
                        key.feature_id * tensor.output_count + key.output_slot
                        for _, key in chunk
                    ],
                    dtype=torch.int64,
                )
                gather_started = perf_counter()
                raw = torch.frombuffer(
                    mapping,
                    dtype=tensor.torch_dtype,
                    count=prod(tensor.identity.shape),
                    offset=tensor.identity.payload_offset,
                ).view(-1, tensor.vector_width)
                rows = raw.index_select(0, flat_ids)
                del raw
                if rows.dtype != plan.output_dtype:
                    converted_rows = rows.to(dtype=plan.output_dtype)
                    del rows
                    rows = converted_rows
                destination_ids = torch.tensor(
                    [sorted_index for sorted_index, _ in chunk], dtype=torch.int64
                )
                unique_rows.index_copy_(0, destination_ids, rows)
                del destination_ids, flat_ids, rows
                gather_seconds += perf_counter() - gather_started

        reorder_started = perf_counter()
        if plan.order is DecoderRowOrder.CALLER:
            rows = unique_rows.index_select(0, torch.tensor(plan.inverse_order, dtype=torch.int64))
            output_keys = plan.caller_keys
        else:
            rows = unique_rows
            output_keys = plan.sorted_unique_keys
        reorder_seconds = perf_counter() - reorder_started

        h2d_started = perf_counter()
        h2d_bytes = 0
        if plan.destination.type != "cpu":
            rows = rows.to(plan.destination)
            h2d_bytes = rows.numel() * rows.element_size()
        h2d_seconds = perf_counter() - h2d_started
        total_seconds = perf_counter() - started
        estimate = plan.estimate
        event = DecoderRowMaterializationTelemetry(
            outcome="materialized",
            occurrence_row_count=estimate.occurrence_row_count,
            unique_row_count=estimate.unique_row_count,
            requested_row_bytes=estimate.requested_row_bytes,
            occurrence_row_bytes=estimate.occurrence_row_bytes,
            backend_materialized_bytes=estimate.backend_materialized_bytes,
            backend_request_count=estimate.backend_request_count,
            block_count=estimate.block_count,
            mapping_count=estimate.mapping_count,
            mapping_open_count=opened,
            range_count=estimate.range_count,
            read_count=estimate.read_count,
            planned_overfetch_ratio=estimate.planned_overfetch_ratio,
            output_bytes=estimate.output_bytes,
            temporary_staging_high_water_bytes=estimate.temporary_staging_high_water_bytes,
            planning_seconds=planning_done - started,
            mapping_seconds=mapping_seconds,
            # Page faults occur inside the vectorized gather and cannot be
            # attributed separately without perturbing the access pattern.
            fault_read_seconds=0.0,
            gather_seconds=gather_seconds,
            reorder_seconds=reorder_seconds,
            h2d_seconds=h2d_seconds,
            h2d_bytes=h2d_bytes,
            total_seconds=total_seconds,
        )
        self._emit(event)
        return DecoderRowSeed(rows, output_keys, event)

    def release(self, reason: str) -> DecoderRowReleaseTelemetry:
        started = perf_counter()
        if self._released:
            event = DecoderRowReleaseTelemetry(
                "already_released", reason, 0, 0, 0, perf_counter() - started
            )
            self._emit(event)
            return event
        mapping_count = len(self._mappings)
        handle_count = len(self._descriptors)
        mapping_bytes = sum(len(mapping) for mapping in self._mappings.values())
        for mapping in self._mappings.values():
            mapping.close()
        for descriptor in self._descriptors.values():
            os.close(descriptor)
        self._mappings.clear()
        self._descriptors.clear()
        self._released = True
        event = DecoderRowReleaseTelemetry(
            "released",
            reason,
            mapping_count,
            handle_count,
            mapping_bytes,
            perf_counter() - started,
        )
        self._emit(event)
        return event

    def _inspect_specs(self, specs: Sequence[DecoderTensorSpec]) -> dict[int, _Tensor]:
        result: dict[int, _Tensor] = {}
        for spec in specs:
            if spec.source_layer in result:
                raise ValueError(f"duplicate decoder source layer {spec.source_layer}")
            header = parse_safetensors_payloads(spec.path)
            if not header.ok:
                assert header.diagnostic is not None
                raise ValueError(
                    f"cannot inspect decoder tensor {spec.path}: {header.diagnostic.message}"
                )
            payload = next((item for item in header.payloads if item.key == spec.tensor_key), None)
            if payload is None:
                raise ValueError(f"tensor {spec.tensor_key!r} is absent from {spec.path}")
            dtype = _DTYPES.get(payload.dtype)
            if dtype is None or len(payload.shape) not in {2, 3}:
                raise ValueError(
                    f"unsupported decoder tensor layout dtype={payload.dtype}, shape={payload.shape}"
                )
            torch_dtype, element_bytes = dtype
            if prod(payload.shape) * element_bytes != payload.length:
                raise ValueError("decoder tensor shape/dtype does not match payload length")
            stat = spec.path.stat()
            identity_data = {
                "source_layer": spec.source_layer,
                "path": str(spec.path.absolute()),
                "tensor_key": spec.tensor_key,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "payload_offset": payload.offset,
                "payload_length": payload.length,
                "dtype": payload.dtype,
                "shape": payload.shape,
            }
            fingerprint = _fingerprint(identity_data)
            if (
                spec.expected_source_fingerprint is not None
                and spec.expected_source_fingerprint != fingerprint
            ):
                raise ValueError(
                    f"decoder source fingerprint mismatch for layer {spec.source_layer}: "
                    f"expected {spec.expected_source_fingerprint}, got {fingerprint}"
                )
            identity = DecoderTensorIdentity(
                **identity_data,
                fingerprint=fingerprint,
            )
            output_count = 1 if len(payload.shape) == 2 else payload.shape[1]
            result[spec.source_layer] = _Tensor(
                identity,
                torch_dtype,
                element_bytes,
                payload.shape[0],
                output_count,
                payload.shape[-1],
            )
        if not result:
            raise ValueError("at least one decoder tensor spec is required")
        return result

    def _verify_identities(self) -> DecoderRowRefusal | None:
        for tensor in self._tensors.values():
            identity = tensor.identity
            try:
                stat = os.stat(identity.path)
                descriptor_stat = os.fstat(self._descriptors[Path(identity.path)])
            except OSError as exc:
                return self._refuse(DecoderRowRefusalCode.SOURCE_MISMATCH, str(exc)).refusal
            observed = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            expected = (
                identity.device,
                identity.inode,
                identity.size,
                identity.mtime_ns,
                identity.ctime_ns,
            )
            if observed != expected:
                return self._refuse(
                    DecoderRowRefusalCode.SOURCE_MISMATCH,
                    f"decoder checkpoint identity changed for {identity.path}",
                ).refusal
            descriptor_observed = (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
                descriptor_stat.st_size,
                descriptor_stat.st_mtime_ns,
                descriptor_stat.st_ctime_ns,
            )
            if descriptor_observed != expected:
                return self._refuse(
                    DecoderRowRefusalCode.SOURCE_MISMATCH,
                    f"open decoder checkpoint identity changed for {identity.path}",
                ).refusal
        return None

    def _mapping_for(self, path: Path) -> tuple[mmap.mmap, bool]:
        existing = self._mappings.get(path)
        if existing is not None:
            return existing, False
        mapping = mmap.mmap(self._descriptors[path], 0, access=mmap.ACCESS_COPY)
        self._mappings[path] = mapping
        return mapping, True

    def _refuse(self, code: DecoderRowRefusalCode, reason: str) -> DecoderRowPlanResult:
        refusal = DecoderRowRefusal(code, reason)
        self._emit(refusal)
        return DecoderRowPlanResult(refusal=refusal)

    def _emit(self, event: DecoderRowTelemetry) -> None:
        if self._telemetry is not None:
            try:
                self._telemetry(event)
            except Exception:
                # Observability is best-effort and must not change row semantics
                # or prevent deterministic resource release.
                pass


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
