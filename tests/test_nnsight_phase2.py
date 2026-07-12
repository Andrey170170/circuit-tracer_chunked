from types import SimpleNamespace
from collections.abc import Callable
from typing import cast

import pytest
import torch

import circuit_tracer.attribution.attribute_nnsight as attribute_nnsight
from circuit_tracer.attribution.nnsight.phases.phase2 import _make_replay_lifecycle
from circuit_tracer.attribution.nnsight.phases.phase2 import Phase2Inputs, Phase2ResourceOwner
from circuit_tracer.attribution.nnsight.row_store import _FileBackedFeatureRowStore
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel


class FakeStore:
    def __init__(self, error: BaseException | None = None) -> None:
        self.cleanup_calls = 0
        self.error = error

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.error is not None:
            raise self.error


class FakeContext:
    def __init__(self, error: BaseException | None = None) -> None:
        self.cleanup_calls = 0
        self.error = error

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.error is not None:
            raise self.error


class FakeLifecycleObserver:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.recorder = object()
        self.phase_events: list[dict[str, object]] = []
        self.calls: list[str] = []
        self.failures = failures or set()

    def run(self, **payload: object) -> None:
        name = str(payload["name"])
        self.calls.append(name)
        if name in self.failures:
            raise RuntimeError(f"{name} failed")

    def phase(self, **payload: object) -> None:
        name = str(payload["name"])
        self.calls.append(name)
        self.phase_events.append(payload)
        if name in self.failures:
            raise RuntimeError(f"{name} failed")

    def close_export(self, **_: object) -> dict[str, object]:
        self.calls.append("close_export")
        if "close_export" in self.failures:
            raise RuntimeError("close_export failed")
        return {"summary": {}, "events": []}

    def attach_exception(self, error: BaseException, _: object) -> None:
        self.calls.append("attach_exception")
        if "attach_exception" in self.failures:
            raise RuntimeError("attach_exception failed")
        setattr(error, "telemetry_export", {})

    def render_human_summary(self, **_: object) -> None:
        pass


def _configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ctx: FakeContext,
    observer: FakeLifecycleObserver,
    run_phase2: Callable[..., object],
) -> None:
    capabilities = SimpleNamespace(
        architecture="fake",
        checkpoint_format="fake",
        supports_compact_row_store=False,
        supports_decoder_chunk_cache=False,
        supports_exact_encoder_residency=False,
        decoder_output_topology="fake",
    )
    phase0_result = SimpleNamespace(
        ctx=ctx,
        input_ids=torch.tensor([1, 2]),
        n_input_pos=2,
        output_position=1,
        trace_input_ids=torch.tensor([1, 2]),
        activation_matrix=torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.int64), torch.empty(0), (1, 2, 1)
        ).coalesce(),
    )
    monkeypatch.setattr(attribute_nnsight, "get_transcoder_capabilities", lambda _: capabilities)
    monkeypatch.setattr(attribute_nnsight, "require_exact_chunked_provider", lambda _: False)
    monkeypatch.setattr(attribute_nnsight.TelemetryObserver, "create", lambda **_: observer)
    monkeypatch.setattr(attribute_nnsight, "_log_memory_boundary", lambda *_, **__: None)
    monkeypatch.setattr(attribute_nnsight, "run_phase0", lambda **_: phase0_result)
    monkeypatch.setattr(attribute_nnsight, "_run_phase1_forward_pass", lambda **_: None)
    monkeypatch.setattr(attribute_nnsight, "run_phase2", run_phase2)


def _run_attribution(ctx: FakeContext) -> None:
    model = SimpleNamespace(device=torch.device("cpu"), transcoders=object())
    attribute_nnsight._run_attribution(
        model=cast(NNSightReplacementModel, model),
        prompt=[1, 2],
        attribution_targets=None,
        max_n_logits=1,
        desired_logit_prob=0.0,
        batch_size=1,
        feature_batch_size=None,
        logit_batch_size=None,
        max_feature_nodes=None,
        offload=None,
        verbose=False,
        offload_handles=[],
        logger=SimpleNamespace(info=lambda *_, **__: None),
        profile=False,
    )


def test_phase2_second_store_failure_uses_owner_fallback_once(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = FakeContext()
    observer = FakeLifecycleObserver()
    feature_store = FakeStore()
    original_error = OSError("second row-store construction failed")

    def fail_after_feature_store(*, inputs: Phase2Inputs, **_: object) -> None:
        owner = inputs.resource_owner
        owner.feature_row_store = cast(_FileBackedFeatureRowStore, feature_store)
        raise original_error

    _configure_runtime(monkeypatch, ctx=ctx, observer=observer, run_phase2=fail_after_feature_store)

    with pytest.raises(OSError, match="second row-store construction failed") as raised:
        _run_attribution(ctx)

    assert raised.value is original_error
    assert feature_store.cleanup_calls == 1
    assert ctx.cleanup_calls == 1
    teardown = next(event for event in observer.phase_events if event["name"] == "teardown.cleanup")
    assert teardown["attrs"] == {"ctx_present": True, "feature_row_store": True}


def test_phase2_failure_after_both_stores_uses_owner_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FakeContext()
    observer = FakeLifecycleObserver()
    feature_store = FakeStore()
    nonfeature_store = FakeStore()
    original_error = RuntimeError("post-store Phase 2 failure")

    def fail_after_both_stores(*, inputs: Phase2Inputs, **_: object) -> None:
        owner = inputs.resource_owner
        owner.feature_row_store = cast(_FileBackedFeatureRowStore, feature_store)
        owner.nonfeature_row_store = cast(_FileBackedFeatureRowStore, nonfeature_store)
        raise original_error

    _configure_runtime(monkeypatch, ctx=ctx, observer=observer, run_phase2=fail_after_both_stores)

    with pytest.raises(RuntimeError, match="post-store Phase 2 failure") as raised:
        _run_attribution(ctx)

    assert raised.value is original_error
    assert feature_store.cleanup_calls == 1
    assert nonfeature_store.cleanup_calls == 1
    assert ctx.cleanup_calls == 1


def test_phase2_returned_stores_cleanup_once_after_ownership_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FakeContext()
    observer = FakeLifecycleObserver()
    feature_store = FakeStore()
    nonfeature_store = FakeStore()
    original_error = RuntimeError("failure after Phase 2 result transfer")

    class ResultAfterOwnershipTransfer:
        targets = ()
        activation_matrix = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.int64), torch.empty(0), (1, 2, 1)
        ).coalesce()
        feat_layers = torch.empty(0, dtype=torch.int64)
        feat_pos = torch.empty(0, dtype=torch.int64)
        feat_ids = torch.empty(0, dtype=torch.int64)
        n_layers = 1
        n_pos = 2
        total_active_feats = 0
        logit_offset = 0
        n_logits = 0
        total_nodes = 0
        base_max_feature_nodes = 0
        actual_max_feature_nodes = 0
        row_store_capacity_feature_nodes = 0
        feature_row_store = feature_store
        nonfeature_row_store = nonfeature_store

        @property
        def edge_matrix(self) -> object:
            raise original_error

    def return_stores(*, inputs: Phase2Inputs, **_: object) -> ResultAfterOwnershipTransfer:
        owner = inputs.resource_owner
        owner.feature_row_store = cast(_FileBackedFeatureRowStore, feature_store)
        owner.nonfeature_row_store = cast(_FileBackedFeatureRowStore, nonfeature_store)
        return ResultAfterOwnershipTransfer()

    _configure_runtime(monkeypatch, ctx=ctx, observer=observer, run_phase2=return_stores)

    with pytest.raises(RuntimeError, match="failure after Phase 2 result transfer") as raised:
        _run_attribution(ctx)

    assert raised.value is original_error
    assert feature_store.cleanup_calls == 1
    assert nonfeature_store.cleanup_calls == 1
    assert ctx.cleanup_calls == 1


def test_phase2_resource_owner_starts_empty() -> None:
    assert Phase2ResourceOwner() == Phase2ResourceOwner()


def test_replay_lifecycle_restores_offloaded_modules_once_before_rebuild() -> None:
    events: list[str] = []
    ctx = SimpleNamespace(
        reset_saved_graph_handles=lambda: events.append("reset"),
        rebuild_saved_graph_handles=lambda: events.append("rebuild"),
        release_saved_graph_handles=lambda: events.append("release"),
    )
    handles = [lambda: events.append("restore-1"), lambda: events.append("restore-2")]
    lifecycle = _make_replay_lifecycle(ctx, handles)

    lifecycle.begin_request()
    lifecycle.release()
    lifecycle.begin_request()
    lifecycle.release()

    assert events == [
        "reset",
        "restore-1",
        "restore-2",
        "rebuild",
        "release",
        "reset",
        "rebuild",
        "release",
    ]
    assert handles == []


def test_lifecycle_cleanup_failures_do_not_replace_primary_and_all_actions_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("primary tracing failure")
    ctx = FakeContext(RuntimeError("context cleanup failed"))
    feature_store = FakeStore(RuntimeError("feature cleanup failed"))
    nonfeature_store = FakeStore(RuntimeError("nonfeature cleanup failed"))
    observer = FakeLifecycleObserver(
        {"teardown.cleanup", "attribute.failed", "close_export", "attach_exception"}
    )

    def fail_after_both_stores(*, inputs: Phase2Inputs, **_: object) -> None:
        inputs.resource_owner.feature_row_store = cast(
            _FileBackedFeatureRowStore, feature_store
        )
        inputs.resource_owner.nonfeature_row_store = cast(
            _FileBackedFeatureRowStore, nonfeature_store
        )
        raise primary

    _configure_runtime(
        monkeypatch, ctx=ctx, observer=observer, run_phase2=fail_after_both_stores
    )

    with pytest.raises(RuntimeError, match="primary tracing failure") as raised:
        _run_attribution(ctx)

    assert raised.value is primary
    assert feature_store.cleanup_calls == 1
    assert nonfeature_store.cleanup_calls == 1
    assert ctx.cleanup_calls == 1
    assert observer.calls[-4:] == [
        "teardown.cleanup",
        "attribute.failed",
        "close_export",
        "attach_exception",
    ]
    notes = getattr(primary, "__notes__", [])
    assert len(notes) == 7


def test_cancellation_is_preserved_through_lifecycle_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = KeyboardInterrupt("cancelled")
    ctx = FakeContext()
    observer = FakeLifecycleObserver()

    def cancel(*, inputs: Phase2Inputs, **_: object) -> None:
        raise cancellation

    _configure_runtime(monkeypatch, ctx=ctx, observer=observer, run_phase2=cancel)

    with pytest.raises(KeyboardInterrupt, match="cancelled") as raised:
        _run_attribution(ctx)

    assert raised.value is cancellation
    assert ctx.cleanup_calls == 1
    assert "attribute.failed" in observer.calls
    assert "attach_exception" in observer.calls


def test_preflight_rejection_does_not_create_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attribute_nnsight.TelemetryObserver,
        "create",
        lambda **_: pytest.fail("observer created before preflight completed"),
    )
    model = SimpleNamespace(device=torch.device("cpu"), transcoders=object())

    with pytest.raises(ValueError, match="feature_batch_size_max"):
        attribute_nnsight._run_attribution(
            model=cast(NNSightReplacementModel, model),
            prompt=[1, 2],
            attribution_targets=None,
            max_n_logits=1,
            desired_logit_prob=0.0,
            batch_size=2,
            feature_batch_size=2,
            logit_batch_size=None,
            max_feature_nodes=None,
            offload=None,
            verbose=False,
            offload_handles=[],
            logger=SimpleNamespace(info=lambda *_, **__: None),
            feature_batch_size_max=1,
        )


def test_cleanup_only_failures_raise_exception_group() -> None:
    first = RuntimeError("feature cleanup failed")
    second = OSError("sink close failed")

    with pytest.raises(
        attribute_nnsight.ExceptionGroup, match="lifecycle cleanup failed"
    ) as raised:
        attribute_nnsight._raise_cleanup_failures([first, second])

    assert raised.value.exceptions == (first, second)
