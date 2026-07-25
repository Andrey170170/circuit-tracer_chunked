from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from circuit_tracer.attribution.nnsight.preparation import (
    BatchMechanisms,
    FrontierMechanisms,
    NumericMechanisms,
    ProviderMechanisms,
    ReplayMechanisms,
    _effective_execution_identity,
)
from circuit_tracer.attribution.nnsight.session_controls import NNSightSessionControls
from circuit_tracer.execution_identity import ExecutionIdentityState
from circuit_tracer.tracing.plan import (
    DecoderCachePolicy,
    ExecutionConstraints,
    FrontierExpansionPlan,
    RowStoragePlan,
    SessionPlan,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities


def _identity(
    *,
    feature_batch_size: int = 8,
    planner_status: str = "disabled",
    compact_row_store: bool = True,
    row_backend: str = "full_file",
    decoder_cache_bytes: int = 0,
    tape_window: int = 1,
    tape_bytes: int = 0,
):
    provider = ProviderMechanisms(
        capabilities=TranscoderCapabilities(
            architecture="clt",
            checkpoint_format="test",
            supports_exact_chunked_provider=True,
            supports_compact_row_store=compact_row_store,
            supports_decoder_chunk_cache=True,
            supports_exact_encoder_residency=True,
        ),
        exact_chunked=True,
        compact_row_store=compact_row_store,
        decoder_chunk_cache=True,
        exact_encoder_residency=True,
        use_compact_feature_row_store=compact_row_store,
    )
    numerics = NumericMechanisms(
        exact_dtype=torch.float32,
        exact_dtype_name="fp32",
        internal_precision_requested="fp32",
        dtype_map={
            "feature_row_storage_dtype": "fp32",
            "row_abs_sum_dtype": "fp32",
            "influence_compute_dtype": "fp32",
            "planner_compute_dtype": "fp32",
            "shadow_debug_compute_dtype": "fp32",
        },
        feature_row_storage_dtype=torch.float32,
        row_abs_sum_dtype=torch.float32,
        influence_compute_dtype=torch.float32,
        planner_compute_dtype=torch.float32,
        shadow_debug_compute_dtype=torch.float32,
        activation_compare_mode="fp32",
    )
    replay = ReplayMechanisms(
        phase0_mode="disabled",
        phase0_context_policy="strict",
        phase0_bundle_path=None,
        phase3_gradient_mode="disabled",
        phase3_gradient_bundle_path=None,
        phase3_row_mode="disabled",
        phase3_row_bundle_path=None,
        phase3_validation_policy="strict",
    )
    controls = NNSightSessionControls(
        session_capacity=feature_batch_size,
        phase3_microbatch_max_rows=feature_batch_size,
        phase4_execution_batch_max_rows=feature_batch_size,
        metadata={},
    )
    batches = BatchMechanisms(
        phase1_config=SimpleNamespace(
            effective_policy="legacy",
            effective_batch_size_max=None,
            fallback_reason=None,
        ),
        phase1_metadata={},
        source_batch_size=feature_batch_size,
        feature_batch_size=feature_batch_size,
        logit_batch_size=feature_batch_size,
        max_phase4_feature_batch_size=32,
        planner_enabled=planner_status != "disabled",
        planner_status=planner_status,
        planner_skip_reason=None,
        session_controls=controls,
        trace_batch_size=feature_batch_size,
    )
    mode = "v1" if compact_row_store else "off"
    frontier = FrontierMechanisms(
        scheduler=SimpleNamespace(
            effective_mode="locality",
            effective_version="locality_v1",
            effective_policy="fixed_frontier_locality",
        ),
        refresh_optimization=SimpleNamespace(
            effective_mode=mode, effective_version=f"{mode}_v1"
        ),
        refresh_policy=SimpleNamespace(
            effective_policy="standard",
            effective_interval_multiplier=1,
            fallback_reason=None,
        ),
        ranker=SimpleNamespace(effective_mode="argsort"),
        row_executor=SimpleNamespace(
            effective_mode="batched", effective_version="batched_v1"
        ),
        row_reduction=SimpleNamespace(
            effective_mode="gpu_v1" if compact_row_store else "off",
            effective_version="gpu_v1_staged" if compact_row_store else "off_v1",
        ),
        row_store_cache_control=SimpleNamespace(
            effective_mode="off", fallback_reason=None
        ),
        exact_encoder_residency=SimpleNamespace(
            effective_mode="active_cpu" if compact_row_store else "lazy",
            fallback_reason=None if compact_row_store else "unsupported provider",
        ),
        exact_encoder_residency_metadata={},
        execution_metadata={},
        refresh_aux_applicable=compact_row_store,
        prepared_chunk_cache_bytes_effective=4096 if compact_row_store else 0,
        active_row_accumulation_effective="direct_v1" if compact_row_store else "zero_fill",
        refresh_aux_fallback_reason=None if compact_row_store else "not_applicable",
        feature_vjp_tape_enabled=tape_window > 1 and row_backend == "full_file",
        feature_vjp_tape_batch_window_effective=(
            tape_window if tape_window > 1 and row_backend == "full_file" else 1
        ),
        feature_vjp_tape_max_bytes_effective=(
            tape_bytes if tape_window > 1 and row_backend == "full_file" else 0
        ),
        feature_vjp_tape_fallback_reason=(
            None
            if tape_window > 1 and row_backend == "full_file"
            else (
                "requires_full_file_backend"
                if tape_window > 1
                else "window_one_streaming_fallback"
            )
        ),
    )
    plan = SimpleNamespace(
        execution=ExecutionConstraints(
            session=SessionPlan(
                decoder_cache=DecoderCachePolicy(
                    enabled=decoder_cache_bytes > 0,
                    max_bytes=decoder_cache_bytes or None,
                )
            ),
            storage=RowStoragePlan(full_retention_backend=row_backend),
            frontier=FrontierExpansionPlan(
                feature_vjp_tape_batch_window=tape_window,
                feature_vjp_tape_max_bytes=tape_bytes,
            ),
        )
    )
    return _effective_execution_identity(provider, numerics, replay, batches, frontier, plan)


def test_effective_execution_descriptor_is_stable_and_json_serializable() -> None:
    first = _identity(planner_status="executed")
    second = _identity(planner_status="executed")

    assert first.fingerprint == second.fingerprint
    assert first.descriptor is not None
    payload = first.descriptor.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["batches"]["feature_batch_planner_status"] == "executed"


def test_effective_identity_changes_with_planner_outcome_and_batch_size() -> None:
    base = _identity(feature_batch_size=8, planner_status="executed")
    resized = _identity(feature_batch_size=16, planner_status="executed")
    skipped = _identity(feature_batch_size=8, planner_status="skipped_no_headroom")

    assert len({base.fingerprint, resized.fingerprint, skipped.fingerprint}) == 3


def test_effective_identity_changes_with_provider_dependent_fallbacks() -> None:
    supported = _identity(compact_row_store=True)
    fallback = _identity(compact_row_store=False)

    assert supported.fingerprint != fallback.fingerprint
    assert fallback.descriptor is not None
    assert fallback.descriptor.frontier["refresh_aux_fallback_reason"] == "not_applicable"


def test_effective_identity_covers_storage_and_decoder_mechanisms() -> None:
    full = _identity(row_backend="full_file")
    tiled = _identity(row_backend="column_tiled_v1")
    cached = _identity(row_backend="full_file", decoder_cache_bytes=4096)

    assert len({full.fingerprint, tiled.fingerprint, cached.fingerprint}) == 3
    assert tiled.descriptor is not None
    assert tiled.descriptor.storage["backend"] == "column_tiled_v1"
    assert cached.descriptor is not None
    assert cached.descriptor.decoder["cache_max_bytes"] == 4096


def test_effective_identity_changes_with_feature_vjp_tape_physical_controls() -> None:
    streaming = _identity()
    taped = _identity(tape_window=2, tape_bytes=4096)

    assert streaming.fingerprint != taped.fingerprint
    assert taped.descriptor is not None
    assert taped.descriptor.frontier["feature_vjp_tape_enabled"] is True
    assert taped.descriptor.frontier["feature_vjp_tape_batch_window_effective"] == 2
    assert taped.descriptor.frontier["feature_vjp_tape_max_bytes_effective"] == 4096


def test_effective_identity_records_unsupported_tape_storage_as_explicit_noop() -> None:
    tiled = _identity(
        row_backend="column_tiled_v1",
        tape_window=2,
        tape_bytes=4096,
    )

    assert tiled.descriptor is not None
    assert tiled.descriptor.frontier["feature_vjp_tape_enabled"] is False
    assert tiled.descriptor.frontier["feature_vjp_tape_batch_window_requested"] == 2
    assert tiled.descriptor.frontier["feature_vjp_tape_batch_window_effective"] == 1
    assert (
        tiled.descriptor.frontier["feature_vjp_tape_fallback_reason"]
        == "requires_full_file_backend"
    )


def test_effective_identity_state_records_allowed_runtime_revisions() -> None:
    initial = _identity(feature_batch_size=8)
    revised = _identity(feature_batch_size=16)
    state = ExecutionIdentityState("requested")

    state.mark_effective(initial)
    state.revise_effective(revised)
    state.revise_effective(revised)

    assert state.effective == revised
    assert state.effective_revisions == [initial, revised]
