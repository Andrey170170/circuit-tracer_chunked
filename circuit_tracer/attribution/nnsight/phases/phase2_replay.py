"""Replay bundle validation and application for attribution Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal, cast

import torch

from circuit_tracer.attribution.nnsight.replay import (
    _build_phase0_activation_matrix_from_loaded_bundle,
    _build_phase0_donor_bundle_payload,
    _build_phase0_replay_metadata,
    _build_phase0_replay_validation_context,
    _build_phase3_replay_metadata,
    _extract_clt_constants_hash_from_snapshot,
    _hash_index_tensor,
    _hash_tensor_raw_bytes,
    _load_phase0_donor_bundle_npz,
    _load_phase3_gradient_donor_bundle_npz,
    _load_phase3_row_donor_bundle_npz,
)
from circuit_tracer.attribution.nnsight.phases.phase2_targets import TargetSelection
from circuit_tracer.observability.events import DiagnosticSnapshot, TraceEvent, TraceObserver


@dataclass(frozen=True)
class Phase0ReplayPolicy:
    mode: str
    donor_bundle_path: str | os.PathLike[str] | None
    context_policy: str
    capture_bundle: bool


@dataclass(frozen=True)
class Phase3ReplayPolicy:
    gradient_mode: str
    gradient_bundle_path: str | os.PathLike[str] | None
    row_mode: str
    row_bundle_path: str | os.PathLike[str] | None
    validation_policy: str


@dataclass(frozen=True)
class Phase0ReplayState:
    activation_matrix: torch.Tensor
    validation_context: dict[str, object]
    metadata: dict[str, object]
    donor_bundle_payload: dict[str, object] | None


@dataclass(frozen=True)
class Phase3ReplayState:
    gradient_metadata: dict[str, object]
    row_metadata: dict[str, object]
    loaded_row_donor_bundle: dict[str, object] | None


def apply_phase0_replay(
    *,
    model: Any,
    ctx: Any,
    input_ids: torch.Tensor,
    host_activation_matrix: torch.Tensor,
    targets: TargetSelection,
    observer: TraceObserver,
    policy: Phase0ReplayPolicy,
) -> Phase0ReplayState:
    """Validate and apply the Phase-0 donor, then capture effective state if requested."""
    host_activation_matrix = host_activation_matrix.coalesce()
    snapshot = cast(
        dict[str, object] | None,
        observer.observe(DiagnosticSnapshot(model.transcoders)),
    )
    validation_context = _build_phase0_replay_validation_context(
        input_tokens=input_ids,
        target_token_ids=targets.target_token_ids,
        activation_matrix=host_activation_matrix,
        clt_constants_hash=_extract_clt_constants_hash_from_snapshot(snapshot),
    )
    host_hashes = {
        key: validation_context.get(key)
        for key in (
            "input_tokens_hash",
            "target_token_ids_hash",
            "active_feature_membership_hash_raw_order",
            "active_feature_membership_hash_canonical",
            "clt_constants_hash",
        )
    }
    activation_matrix = host_activation_matrix
    if policy.mode == "donor_phase0":
        assert policy.donor_bundle_path is not None
        loaded = _load_phase0_donor_bundle_npz(
            policy.donor_bundle_path,
            context_policy=cast(Literal["strict", "warn"], policy.context_policy),
            validation_context=validation_context,
        )
        donor_matrix = _build_phase0_activation_matrix_from_loaded_bundle(
            loaded, device=host_activation_matrix.device
        )
        replace_state = getattr(ctx, "replace_phase0_activation_state", None)
        if not callable(replace_state):
            raise RuntimeError(
                "Attribution context does not support Phase-0 activation-state replacement"
            )
        replace_state(donor_matrix)
        activation_matrix = ctx.activation_matrix.coalesce()
        validation = cast(dict[str, object], loaded.get("validation_metadata", {}))
        dtype_metadata = cast(dict[str, object], loaded.get("dtype_metadata", {}))
        warnings = [
            str(item) for item in cast(list[object], validation.get("warnings", []))
        ]
        status = "applied_with_warnings" if warnings else "applied"
        metadata = _build_phase0_replay_metadata(
            mode=policy.mode,
            status=status,
            donor_bundle_path=policy.donor_bundle_path,
            context_policy=policy.context_policy,
            validation_warnings=warnings,
            validation_failure_count=int(
                cast(int, validation.get("validation_failure_count", len(warnings)))
            ),
            dtype_metadata=dtype_metadata,
            host_hashes=host_hashes,
            donor_hashes={
                "computed": cast(dict[str, object], validation.get("computed_hashes", {})),
                "stored": cast(dict[str, object], validation.get("stored_hashes", {})),
            },
            host_active_feature_count=int(host_activation_matrix._nnz()),
            donor_active_feature_count=int(activation_matrix._nnz()),
            replay_single_step_intended=True,
            note="single-step intended replay mode",
        )
        attrs = {
            "phase0_replay_mode": policy.mode,
            "phase0_replay_status": status,
            "context_policy": policy.context_policy,
            "validation_warning_count": len(warnings),
            "dtype_roundtrip_loss": bool(dtype_metadata.get("dtype_roundtrip_loss", False)),
            "host_active_feature_count": int(host_activation_matrix._nnz()),
            "donor_active_feature_count": int(activation_matrix._nnz()),
        }
    else:
        metadata = _build_phase0_replay_metadata(
            mode=policy.mode,
            status="disabled",
            donor_bundle_path=None,
            context_policy=policy.context_policy,
            host_hashes=host_hashes,
            host_active_feature_count=int(host_activation_matrix._nnz()),
            replay_single_step_intended=True,
            note="single-step intended replay mode",
        )
        attrs = {"phase0_replay_mode": policy.mode, "phase0_replay_status": "disabled"}
    observer.observe(TraceEvent(scope="phase", name="phase2.phase0_replay", phase="phase2", attrs=attrs))

    payload = None
    if policy.capture_bundle:
        valid_mask = (targets.target_token_ids >= 0) & (
            targets.target_token_ids < int(targets.output_logits.shape[0])
        )
        target_logits = (
            targets.output_logits[targets.target_token_ids[valid_mask]]
            if bool(valid_mask.any().item())
            else None
        )
        payload = _build_phase0_donor_bundle_payload(
            activation_matrix=activation_matrix,
            input_tokens=input_ids,
            target_token_ids=targets.target_token_ids,
            target_probabilities=targets.targets.logit_probabilities,
            target_logits=target_logits,
            transcoder_diagnostic_snapshot=cast(
                dict[str, object] | None,
                observer.observe(DiagnosticSnapshot(model.transcoders)),
            ),
            status=(
                "captured_replayed_effective_state"
                if policy.mode != "disabled"
                else "captured"
            ),
        )
        payload["replayed_effective_state"] = policy.mode != "disabled"
        payload["phase0_replay_mode"] = policy.mode
    return Phase0ReplayState(activation_matrix, validation_context, metadata, payload)


def load_phase3_replay(
    *,
    ctx: Any,
    targets: TargetSelection,
    activation_matrix: torch.Tensor,
    n_layers: int,
    n_pos: int,
    trace_batch_size: int,
    policy: Phase3ReplayPolicy,
    validation_context: dict[str, object],
) -> Phase3ReplayState:
    """Validate Phase-3 gradient/row donors and attach replay state to the context."""
    total_active_feats = int(activation_matrix._nnz())
    common_host_hashes = {
        "target_token_ids_hash": validation_context.get("target_token_ids_hash"),
        "active_features_hash": _hash_index_tensor(
            activation_matrix.indices().T.detach().cpu().reshape(-1)
        ),
        "activation_values_hash": _hash_tensor_raw_bytes(activation_matrix.values()),
        "active_feature_count": total_active_feats,
    }
    gradient_metadata = _build_phase3_replay_metadata(
        replay_kind="phase3_gradient_replay_v1",
        mode=policy.gradient_mode,
        status="disabled" if policy.gradient_mode == "disabled" else "pending",
        donor_bundle_path=policy.gradient_bundle_path,
        validation_policy=policy.validation_policy,
        source="host_computed" if policy.gradient_mode == "disabled" else None,
    )
    if policy.gradient_mode == "donor":
        assert policy.gradient_bundle_path is not None
        loaded_gradient = _load_phase3_gradient_donor_bundle_npz(
            policy.gradient_bundle_path,
            target_token_ids=targets.target_token_ids,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            expected_n_layers=n_layers,
            expected_gradient_batch_size=trace_batch_size,
            expected_n_positions=n_pos,
            expected_d_model=int(targets.targets.logit_vectors.shape[-1]),
            validation_policy=cast(Literal["strict"], policy.validation_policy),
        )
        setattr(ctx, "phase3_gradient_replay_tensor", loaded_gradient["gradients"])
        setattr(ctx, "phase3_gradient_replay_status", "applied")
        validation = cast(dict[str, object], loaded_gradient.get("validation_metadata", {}))
        gradient_metadata = _build_phase3_replay_metadata(
            replay_kind="phase3_gradient_replay_v1", mode=policy.gradient_mode,
            status="applied", donor_bundle_path=policy.gradient_bundle_path,
            validation_policy=policy.validation_policy,
            validation_failure_count=int(cast(int, validation.get("validation_failure_count", 0))),
            donor_hashes=cast(dict[str, object], validation.get("stored_hashes", {})),
            host_hashes=common_host_hashes, source="donor_gradient_bundle",
            note="feature/error gradients replayed from donor; token gradient remains host-computed",
        )
    else:
        setattr(ctx, "phase3_gradient_replay_tensor", None)
        setattr(ctx, "phase3_gradient_replay_status", "disabled")

    row_metadata = _build_phase3_replay_metadata(
        replay_kind="phase3_row_replay_v1", mode=policy.row_mode,
        status="disabled" if policy.row_mode == "disabled" else "pending",
        donor_bundle_path=policy.row_bundle_path, validation_policy=policy.validation_policy,
        source="host_computed" if policy.row_mode == "disabled" else None,
    )
    loaded_row = None
    if policy.row_mode == "donor":
        assert policy.row_bundle_path is not None
        loaded_row = _load_phase3_row_donor_bundle_npz(
            policy.row_bundle_path,
            target_token_ids=targets.target_token_ids,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            expected_total_active_features=total_active_feats,
            validation_policy=cast(Literal["strict"], policy.validation_policy),
        )
        validation = cast(dict[str, object], loaded_row.get("validation_metadata", {}))
        row_metadata = _build_phase3_replay_metadata(
            replay_kind="phase3_row_replay_v1", mode=policy.row_mode, status="applied",
            donor_bundle_path=policy.row_bundle_path,
            validation_policy=policy.validation_policy,
            validation_failure_count=int(cast(int, validation.get("validation_failure_count", 0))),
            donor_hashes=cast(dict[str, object], validation.get("stored_hashes", {})),
            host_hashes=common_host_hashes, source="donor_row_bundle_override",
            note=("donor row bundle overrides feature rows and row normalizers; "
                  "dense token/error columns remain host-computed"),
        )
    return Phase3ReplayState(gradient_metadata, row_metadata, loaded_row)
