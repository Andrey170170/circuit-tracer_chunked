from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from circuit_tracer.attribution.nnsight.execution import AttributionExecution
from circuit_tracer.observability.events import MemoryDelta, MemorySnapshot, TraceEvent
from circuit_tracer.transcoder.checkpoint_assets import (
    CheckpointAsset,
    CheckpointAssetScope,
    CheckpointManifest,
    CheckpointRange,
)
from circuit_tracer.transcoder.checkpoint_working_set import (
    PhaseWorkingSetPlan,
    ProviderCheckpointLifecycle,
)


def _manifest(path: Path, *, scope: CheckpointAssetScope) -> CheckpointManifest:
    path.write_bytes(b"x" * 128)
    asset_id = "checkpoint"
    return CheckpointManifest(
        (
            CheckpointAsset.from_path(
                asset_id=asset_id,
                path=path,
                scope=scope,
                ranges=(
                    CheckpointRange(asset_id, "decoder", 0, 32),
                    CheckpointRange(asset_id, "encoder", 32, 48),
                    CheckpointRange(asset_id, "refresh", 80, 16),
                ),
            ),
        )
    )


def test_working_set_admission_is_byte_bounded_and_role_prioritized(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path / "weights.safetensors",
        scope=CheckpointAssetScope.SHARED,
    )

    plan = PhaseWorkingSetPlan.admit(
        manifest,
        retain_bytes=16,
        byte_budget=80,
        available_headroom_bytes=72,
    )

    assert [item.role for item in plan.release] == ["decoder"]
    assert [item.role for item in plan.prefault] == ["encoder"]
    assert plan.prefault_requested_bytes == 64
    assert plan.prefault_admitted_bytes == 48
    assert plan.prefault_refused_bytes == 16
    assert plan.fallback_reason == "prefault_partially_admitted"


class _Observer:
    def __init__(self) -> None:
        self.events: list[object] = []

    def observe(self, event: object) -> object:
        self.events.append(event)
        if isinstance(event, MemorySnapshot):
            return {"cgroup_memory_headroom_gib": 1.0, "proc_rss_file_gib": 2.0}
        if isinstance(event, MemoryDelta):
            return {"proc_rss_file_gib_delta": -0.25}
        return event


class _Provider:
    def __init__(
        self,
        manifest: CheckpointManifest,
        order: list[str],
    ) -> None:
        self.checkpoint_lifecycle = ProviderCheckpointLifecycle(
            manifest,
            prefault_budget_bytes=128,
        )
        self._order = order

    def close_decoder_checkpoint_handles(self) -> None:
        self._order.append("provider_handles_closed")


class _Context:
    def __init__(self, provider: _Provider, order: list[str]) -> None:
        self.decoder_provider = provider
        self._order = order

    def seal_active_decoder_rows_for_checkpoint_transition(self) -> int:
        self._order.append("active_rows_sealed")
        return 16

    def close_owned_decoder_resources_for_checkpoint_transition(self) -> None:
        self._order.append("owned_decoder_resources_closed")

    def get_diagnostic_snapshot(self) -> dict[str, object]:
        return {"active_encoder_bytes": 8}


def test_runtime_transition_seals_and_closes_before_shared_advice_refusal(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    manifest = _manifest(
        tmp_path / "shared.safetensors",
        scope=CheckpointAssetScope.SHARED,
    )
    observer = _Observer()
    ctx = _Context(_Provider(manifest, order), order)
    execution = AttributionExecution(
        prepared=SimpleNamespace(
            diagnostics=SimpleNamespace(observer=observer),
        ),
        scope=SimpleNamespace(),
        operations=SimpleNamespace(),
        phase0=SimpleNamespace(ctx=ctx),
    )
    execution._refresh_active_decoder_row_execution_metadata = lambda: None  # type: ignore[method-assign]

    execution.apply_checkpoint_working_set_transition()

    assert order == [
        "active_rows_sealed",
        "owned_decoder_resources_closed",
        "provider_handles_closed",
    ]
    trace_events = [event for event in observer.events if isinstance(event, TraceEvent)]
    page_events = [event for event in trace_events if event.name.startswith("checkpoint.page.")]
    assert page_events
    assert all(event.attrs["outcome"] == "refused" for event in page_events)
    assert all(event.attrs["reason"] == "scope_shared_is_not_advice_eligible" for event in page_events)
    assert any(event.name == "checkpoint.working_set.transitioned" for event in trace_events)


def test_runtime_transition_refuses_before_close_when_rows_are_not_sealed(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    manifest = _manifest(
        tmp_path / "shared.safetensors",
        scope=CheckpointAssetScope.SHARED,
    )
    observer = _Observer()
    ctx = _Context(_Provider(manifest, order), order)

    def refuse_seal() -> int:
        order.append("seal_refused")
        raise RuntimeError("coverage mismatch")

    ctx.seal_active_decoder_rows_for_checkpoint_transition = refuse_seal  # type: ignore[method-assign]
    execution = AttributionExecution(
        prepared=SimpleNamespace(
            diagnostics=SimpleNamespace(observer=observer),
        ),
        scope=SimpleNamespace(),
        operations=SimpleNamespace(),
        phase0=SimpleNamespace(ctx=ctx),
    )

    execution.apply_checkpoint_working_set_transition()

    assert order == ["seal_refused"]
    assert any(
        isinstance(event, TraceEvent)
        and event.name == "checkpoint.working_set.refused"
        and event.attrs["reason"] == "active_decoder_rows_not_sealed"
        for event in observer.events
    )
