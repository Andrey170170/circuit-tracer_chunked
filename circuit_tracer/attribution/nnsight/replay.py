"""Replay policy, donor-bundle serialization, and validation helpers."""

import hashlib
import os
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import torch

from circuit_tracer.attribution.nnsight.numerics import _resolve_exact_trace_internal_dtype
from circuit_tracer.attribution.nnsight.telemetry import _hash_json_payload


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _compute_row_denominator_scaled_l1(
    row_values: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
    preserve_device: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stable row-L1 denominator representation.

    Returns ``(row_abs_max, row_l1_scaled)`` where
    ``row_l1 = row_abs_max * row_l1_scaled`` for each row.
    """

    resolved_dtype = _resolve_exact_trace_internal_dtype(dtype)
    row_values_cpu = row_values.detach()
    if row_values_cpu.ndim != 2:
        raise ValueError("row_values must be rank-2")
    target_device = row_values_cpu.device if preserve_device else torch.device("cpu")
    if row_values_cpu.device != target_device or row_values_cpu.dtype != resolved_dtype:
        row_values_cpu = row_values_cpu.to(device=target_device, dtype=resolved_dtype)

    n_rows = int(row_values_cpu.shape[0])
    n_cols = int(row_values_cpu.shape[1])
    if n_cols == 0:
        row_abs_max = torch.zeros(n_rows, dtype=resolved_dtype, device=target_device)
        row_l1_scaled = torch.zeros(n_rows, dtype=resolved_dtype, device=target_device)
        return row_abs_max, row_l1_scaled

    # Two-pass chunked reduction to avoid materializing a full abs() matrix copy.
    col_chunk_size = min(max(n_cols, 1), 4096)
    row_abs_max = torch.zeros(n_rows, dtype=resolved_dtype, device=target_device)
    for col_start in range(0, n_cols, col_chunk_size):
        col_end = min(col_start + col_chunk_size, n_cols)
        chunk_abs_max = row_values_cpu[:, col_start:col_end].abs().amax(dim=1)
        row_abs_max = torch.maximum(row_abs_max, chunk_abs_max)

    row_l1_scaled = torch.zeros_like(row_abs_max)
    nonzero_rows = (row_abs_max > 0) & torch.isfinite(row_abs_max)
    if bool(nonzero_rows.any()):
        nonzero_denom = row_abs_max[nonzero_rows].unsqueeze(1)
        nonzero_scaled_sum = torch.zeros(
            nonzero_denom.shape[0], dtype=resolved_dtype, device=target_device
        )
        for col_start in range(0, n_cols, col_chunk_size):
            col_end = min(col_start + col_chunk_size, n_cols)
            chunk = row_values_cpu[nonzero_rows, col_start:col_end].abs()
            nonzero_scaled_sum += (chunk / nonzero_denom).sum(dim=1)
        row_l1_scaled[nonzero_rows] = nonzero_scaled_sum

    infinite_rows = torch.isinf(row_abs_max)
    if bool(infinite_rows.any()):
        row_l1_scaled[infinite_rows] = 1
    return row_abs_max, row_l1_scaled


def _compute_row_abs_sums(
    row_values: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Backward-compatible helper for non-hot-path diagnostics/tests."""

    row_abs_max, row_l1_scaled = _compute_row_denominator_scaled_l1(row_values, dtype=dtype)
    return row_abs_max * row_l1_scaled


_PHASE0_REPLAY_MODE_BY_NAME: dict[str, str] = {
    "disabled": "disabled",
    "donor_phase0": "donor_phase0",
}


_PHASE0_DONOR_CONTEXT_POLICY_BY_NAME: dict[str, str] = {
    "strict": "strict",
    "warn": "warn",
}


_PHASE3_REPLAY_MODE_BY_NAME: dict[str, str] = {
    "disabled": "disabled",
    "donor": "donor",
}


_PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME: dict[str, str] = {
    "strict": "strict",
}


def _resolve_phase0_replay_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _PHASE0_REPLAY_MODE_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_PHASE0_REPLAY_MODE_BY_NAME))
        raise ValueError(f"phase0_replay_mode must be one of: {allowed} (got {value!r})")
    return resolved


def _resolve_phase0_donor_context_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _PHASE0_DONOR_CONTEXT_POLICY_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_PHASE0_DONOR_CONTEXT_POLICY_BY_NAME))
        raise ValueError(f"phase0_donor_context_policy must be one of: {allowed} (got {value!r})")
    return resolved


def _resolve_phase3_replay_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _PHASE3_REPLAY_MODE_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_PHASE3_REPLAY_MODE_BY_NAME))
        raise ValueError(f"phase3 replay mode must be one of: {allowed} (got {value!r})")
    return resolved


def _resolve_phase3_replay_validation_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    resolved = _PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(_PHASE3_REPLAY_VALIDATION_POLICY_BY_NAME))
        raise ValueError(
            f"phase3_replay_validation_policy must be one of: {allowed} (got {value!r})"
        )
    return resolved


def _hash_sparse_membership_indices(
    indices: torch.Tensor,
    *,
    shape: Sequence[int],
    canonicalize: bool,
) -> str:
    indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    hasher = hashlib.blake2s(digest_size=8)
    hasher.update(np.asarray(list(shape), dtype=np.int64).tobytes())
    if indices_cpu.numel() == 0:
        hasher.update(b"empty")
        return hasher.hexdigest()

    if not canonicalize:
        hasher.update(indices_cpu.numpy().tobytes())
        return hasher.hexdigest()

    if len(shape) != 3:
        raise ValueError(f"Expected sparse membership shape of length 3, got {shape}")

    _, n_positions, n_features = [int(v) for v in shape]
    n_positions_n_features = int(n_positions) * int(n_features)
    flat_membership = (
        indices_cpu[0] * n_positions_n_features + indices_cpu[1] * int(n_features) + indices_cpu[2]
    )
    flat_membership_sorted = torch.sort(flat_membership).values.contiguous()
    hasher.update(flat_membership_sorted.numpy().tobytes())
    return hasher.hexdigest()


def _hash_index_tensor(indices: torch.Tensor) -> str:
    indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.blake2s(indices_cpu.numpy().tobytes(), digest_size=8).hexdigest()


def _hash_float_tensor(values: torch.Tensor, *, dtype: torch.dtype = torch.float64) -> str:
    values_cpu = values.detach().to(device="cpu", dtype=dtype).contiguous()
    return hashlib.blake2s(values_cpu.numpy().tobytes(), digest_size=8).hexdigest()


def _hash_tensor_raw_bytes(values: torch.Tensor) -> str:
    values_cpu = values.detach().to(device="cpu").contiguous()
    raw_byte_view = values_cpu.view(torch.uint8)
    return hashlib.blake2s(raw_byte_view.numpy().tobytes(), digest_size=8).hexdigest()


def _extract_clt_constants_hash_from_snapshot(
    transcoder_diagnostic_snapshot: dict[str, object] | None,
) -> str | None:
    if not isinstance(transcoder_diagnostic_snapshot, dict):
        return None

    def _extract_from_boundary_fingerprints(payload: object) -> str | None:
        if not isinstance(payload, dict):
            return None
        constants = payload.get("transcoder_constant_fingerprints")
        if isinstance(constants, dict):
            global_hash = constants.get("global_hash")
            if isinstance(global_hash, str) and global_hash:
                return global_hash
        global_hashes = payload.get("global_hashes")
        if isinstance(global_hashes, dict):
            constants_hash = global_hashes.get("transcoder_constants_global_hash")
            if isinstance(constants_hash, str) and constants_hash:
                return constants_hash
        return None

    boundary_hash = _extract_from_boundary_fingerprints(
        transcoder_diagnostic_snapshot.get("phase0_boundary_fingerprints")
    )
    if boundary_hash:
        return boundary_hash

    threshold_membership = transcoder_diagnostic_snapshot.get("phase0_threshold_membership")
    if isinstance(threshold_membership, dict):
        constants = threshold_membership.get("transcoder_constant_fingerprints")
        if isinstance(constants, dict):
            global_hash = constants.get("global_hash")
            if isinstance(global_hash, str) and global_hash:
                return global_hash
        global_hashes = threshold_membership.get("global_hashes")
        if isinstance(global_hashes, dict):
            constants_hash = global_hashes.get("transcoder_constants_global_hash")
            if isinstance(constants_hash, str) and constants_hash:
                return constants_hash
    return None


def _build_phase0_donor_bundle_payload(
    *,
    activation_matrix: torch.Tensor,
    input_tokens: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_probabilities: torch.Tensor | None,
    target_logits: torch.Tensor | None,
    transcoder_diagnostic_snapshot: dict[str, object] | None,
    status: str,
) -> dict[str, object]:
    activation_matrix_coalesced = activation_matrix.coalesce()
    activation_indices = (
        activation_matrix_coalesced.indices().detach().to(device="cpu", dtype=torch.int64)
    )
    activation_values = activation_matrix_coalesced.values().detach().to(device="cpu").contiguous()
    active_features = activation_indices.T.contiguous()
    activation_shape = [int(dim) for dim in activation_matrix_coalesced.shape]

    n_layers = int(activation_shape[0]) if activation_shape else 0
    layer_counts = (
        torch.bincount(active_features[:, 0], minlength=n_layers)
        if active_features.numel() > 0
        else torch.zeros((n_layers,), dtype=torch.int64)
    )

    input_tokens_cpu = input_tokens.detach().to(device="cpu", dtype=torch.int64).contiguous()
    target_token_ids_cpu = (
        target_token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    )
    target_probabilities_cpu = (
        target_probabilities.detach().to(device="cpu").contiguous()
        if isinstance(target_probabilities, torch.Tensor)
        else None
    )
    target_logits_cpu = (
        target_logits.detach().to(device="cpu").contiguous()
        if isinstance(target_logits, torch.Tensor)
        else None
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "replay_kind": "phase0_active_features_v1",
        "status": str(status),
        "active_features": active_features,
        "activation_values": activation_values,
        "activation_values_dtype": str(activation_values.dtype).replace("torch.", ""),
        "activation_matrix_shape": activation_shape,
        "active_feature_count": int(activation_values.numel()),
        "active_feature_membership_hash_raw_order": _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_shape,
            canonicalize=False,
        ),
        "active_feature_membership_hash_canonical": _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_shape,
            canonicalize=True,
        ),
        "active_feature_values_hash": _hash_tensor_raw_bytes(activation_values),
        "active_feature_layer_counts": layer_counts,
        "input_tokens": input_tokens_cpu,
        "input_token_count": int(input_tokens_cpu.numel()),
        "input_tokens_hash": _hash_index_tensor(input_tokens_cpu),
        "target_token_ids": target_token_ids_cpu,
        "target_count": int(target_token_ids_cpu.numel()),
        "target_token_ids_hash": (
            _hash_index_tensor(target_token_ids_cpu) if target_token_ids_cpu.numel() > 0 else None
        ),
        "provenance": {
            "transcoder_diagnostic_snapshot_hash": (
                _hash_json_payload(transcoder_diagnostic_snapshot)
                if isinstance(transcoder_diagnostic_snapshot, dict)
                else None
            ),
        },
    }

    if target_probabilities_cpu is not None:
        payload["target_probabilities"] = target_probabilities_cpu
        payload["target_probability_hash"] = _hash_float_tensor(
            target_probabilities_cpu,
            dtype=torch.float64,
        )

    if target_logits_cpu is not None:
        payload["target_logits"] = target_logits_cpu
        payload["target_logit_hash"] = _hash_float_tensor(
            target_logits_cpu,
            dtype=torch.float64,
        )

    clt_constants_hash = _extract_clt_constants_hash_from_snapshot(transcoder_diagnostic_snapshot)
    if clt_constants_hash is not None:
        payload["clt_constants_hash"] = clt_constants_hash

    return payload


def _phase0_npz_scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _phase0_npz_optional_str(value: object) -> str | None:
    scalar = _phase0_npz_scalar(value)
    if scalar is None:
        return None
    text = str(scalar).replace("torch.", "").strip()
    return text if text else None


def _phase0_npz_int(value: object, *, default: int = 0) -> int:
    scalar = _phase0_npz_scalar(value)
    if scalar is None:
        return int(default)
    try:
        return int(scalar)
    except (TypeError, ValueError):
        return int(default)


def _phase0_to_int64_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    array = np.asarray(value)
    if array.size == 0:
        return torch.empty((0,), dtype=torch.int64)
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.int64)


def _load_phase0_donor_bundle_npz(
    donor_bundle_path: str | os.PathLike[str],
    *,
    context_policy: Literal["strict", "warn"] = "strict",
    validation_context: dict[str, object] | None = None,
) -> dict[str, object]:
    policy = str(context_policy).strip().lower()
    if policy not in {"strict", "warn"}:
        raise ValueError("context_policy must be one of {'strict', 'warn'}")

    with np.load(donor_bundle_path, allow_pickle=False) as donor_npz:
        donor_payload = {key: donor_npz[key] for key in donor_npz.files}

    schema_version = _phase0_npz_int(donor_payload.get("schema_version"), default=0)
    replay_kind = _phase0_npz_optional_str(donor_payload.get("replay_kind")) or ""
    status = _phase0_npz_optional_str(donor_payload.get("status")) or ""

    activation_values_dtype_recorded = (
        _phase0_npz_optional_str(donor_payload.get("activation_values_dtype")) or ""
    ).lower()
    if activation_values_dtype_recorded == "torch.bfloat16":
        activation_values_dtype_recorded = "bfloat16"

    active_features_array = np.ascontiguousarray(
        np.asarray(donor_payload.get("active_features", np.empty((0, 3), dtype=np.int64)))
    )
    if active_features_array.ndim == 1 and active_features_array.size == 0:
        active_features_array = np.empty((0, 3), dtype=np.int64)
    active_features = torch.from_numpy(active_features_array).to(dtype=torch.int64)

    activation_values_array = np.ascontiguousarray(
        np.asarray(donor_payload.get("activation_values", np.empty((0,), dtype=np.float32)))
    ).reshape(-1)
    activation_values = torch.from_numpy(activation_values_array)

    raw_uint16_array = np.ascontiguousarray(
        np.asarray(
            donor_payload.get("activation_values_raw_uint16", np.empty((0,), dtype=np.uint16)),
            dtype=np.uint16,
        )
    ).reshape(-1)

    exact_bfloat16_reconstructed = False
    dtype_roundtrip_loss = False
    if activation_values_dtype_recorded == "bfloat16":
        if raw_uint16_array.size == activation_values_array.size and raw_uint16_array.size > 0:
            activation_values = torch.from_numpy(raw_uint16_array).view(torch.bfloat16).clone()
            exact_bfloat16_reconstructed = True
        else:
            dtype_roundtrip_loss = True

    activation_matrix_shape = tuple(
        int(v)
        for v in np.asarray(
            donor_payload.get("activation_matrix_shape", np.empty((0,), dtype=np.int64))
        )
        .astype(np.int64, copy=False)
        .reshape(-1)
        .tolist()
    )
    active_feature_count = _phase0_npz_int(
        donor_payload.get("active_feature_count"),
        default=int(active_features.shape[0]) if active_features.ndim == 2 else 0,
    )

    input_tokens = _phase0_to_int64_tensor(
        donor_payload.get("input_tokens", np.empty((0,), dtype=np.int64))
    )
    target_token_ids = _phase0_to_int64_tensor(
        donor_payload.get("target_token_ids", np.empty((0,), dtype=np.int64))
    )

    input_token_count = _phase0_npz_int(donor_payload.get("input_token_count"), default=0)
    target_count = _phase0_npz_int(donor_payload.get("target_count"), default=0)

    stored_membership_hash_raw = _phase0_npz_optional_str(
        donor_payload.get("active_feature_membership_hash_raw_order")
    )
    stored_membership_hash_canonical = _phase0_npz_optional_str(
        donor_payload.get("active_feature_membership_hash_canonical")
    )
    stored_values_hash = _phase0_npz_optional_str(donor_payload.get("active_feature_values_hash"))
    stored_input_tokens_hash = _phase0_npz_optional_str(donor_payload.get("input_tokens_hash"))
    stored_target_token_ids_hash = _phase0_npz_optional_str(
        donor_payload.get("target_token_ids_hash")
    )
    stored_clt_constants_hash = _phase0_npz_optional_str(donor_payload.get("clt_constants_hash"))

    computed_input_tokens_hash = _hash_index_tensor(input_tokens)
    computed_target_token_ids_hash = (
        _hash_index_tensor(target_token_ids) if target_token_ids.numel() > 0 else None
    )

    computed_membership_hash_raw: str | None = None
    computed_membership_hash_canonical: str | None = None
    if (
        active_features.ndim == 2
        and active_features.shape[1] == 3
        and len(activation_matrix_shape) == 3
    ):
        activation_indices = active_features.T.contiguous()
        computed_membership_hash_raw = _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_matrix_shape,
            canonicalize=False,
        )
        computed_membership_hash_canonical = _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_matrix_shape,
            canonicalize=True,
        )

    computed_values_hash: str | None = None
    if exact_bfloat16_reconstructed:
        computed_values_hash = _hash_tensor_raw_bytes(activation_values)

    validation_issues: list[str] = []

    if schema_version != 1:
        validation_issues.append(f"schema_version mismatch (expected 1, got {schema_version})")
    if replay_kind != "phase0_active_features_v1":
        validation_issues.append(
            f"replay_kind mismatch (expected 'phase0_active_features_v1', got {replay_kind!r})"
        )

    if active_features.ndim != 2 or active_features.shape[1] != 3:
        validation_issues.append(
            f"active_features must have shape [N, 3] (got {tuple(active_features.shape)})"
        )

    if len(activation_matrix_shape) != 3:
        validation_issues.append(
            f"activation_matrix_shape must have length 3 (got {activation_matrix_shape})"
        )

    row_count = int(active_features.shape[0]) if active_features.ndim == 2 else 0
    value_count = int(activation_values.numel())
    if value_count != row_count:
        validation_issues.append(
            f"activation_values length mismatch (values={value_count}, active_features={row_count})"
        )
    if active_feature_count != row_count:
        validation_issues.append(
            "active_feature_count mismatch "
            f"(declared={active_feature_count}, active_features={row_count})"
        )
    if active_feature_count != value_count:
        validation_issues.append(
            "active_feature_count mismatch "
            f"(declared={active_feature_count}, activation_values={value_count})"
        )

    if computed_membership_hash_raw is None:
        validation_issues.append(
            "unable to compute active_feature_membership_hash_raw_order "
            "because active_features/activation_matrix_shape are invalid"
        )
    else:
        if not stored_membership_hash_raw:
            validation_issues.append("missing active_feature_membership_hash_raw_order")
        elif stored_membership_hash_raw != computed_membership_hash_raw:
            validation_issues.append(
                "active_feature_membership_hash_raw_order mismatch "
                f"(stored={stored_membership_hash_raw}, computed={computed_membership_hash_raw})"
            )

    if computed_membership_hash_canonical is None:
        validation_issues.append(
            "unable to compute active_feature_membership_hash_canonical "
            "because active_features/activation_matrix_shape are invalid"
        )
    else:
        if not stored_membership_hash_canonical:
            validation_issues.append("missing active_feature_membership_hash_canonical")
        elif stored_membership_hash_canonical != computed_membership_hash_canonical:
            validation_issues.append(
                "active_feature_membership_hash_canonical mismatch "
                "(stored="
                f"{stored_membership_hash_canonical}, computed={computed_membership_hash_canonical})"
            )

    if exact_bfloat16_reconstructed:
        if not stored_values_hash:
            validation_issues.append("missing active_feature_values_hash")
        elif computed_values_hash != stored_values_hash:
            validation_issues.append(
                "active_feature_values_hash mismatch "
                f"(stored={stored_values_hash}, computed={computed_values_hash})"
            )

    if input_token_count != int(input_tokens.numel()):
        validation_issues.append(
            f"input_token_count mismatch (declared={input_token_count}, actual={int(input_tokens.numel())})"
        )
    if not stored_input_tokens_hash:
        validation_issues.append("missing input_tokens_hash")
    elif stored_input_tokens_hash != computed_input_tokens_hash:
        validation_issues.append(
            "input_tokens_hash mismatch "
            f"(stored={stored_input_tokens_hash}, computed={computed_input_tokens_hash})"
        )

    if target_count != int(target_token_ids.numel()):
        validation_issues.append(
            f"target_count mismatch (declared={target_count}, actual={int(target_token_ids.numel())})"
        )
    if target_token_ids.numel() > 0:
        if not stored_target_token_ids_hash:
            validation_issues.append("missing target_token_ids_hash")
        elif stored_target_token_ids_hash != computed_target_token_ids_hash:
            validation_issues.append(
                "target_token_ids_hash mismatch "
                "(stored="
                f"{stored_target_token_ids_hash}, computed={computed_target_token_ids_hash})"
            )

    if isinstance(validation_context, dict):
        expected_input_tokens = validation_context.get("input_tokens")
        if expected_input_tokens is not None:
            expected_input_tokens_tensor = _phase0_to_int64_tensor(expected_input_tokens).reshape(
                -1
            )
            if not torch.equal(expected_input_tokens_tensor, input_tokens.reshape(-1)):
                validation_issues.append("input_tokens mismatch with validation_context")

        expected_input_hash = _phase0_npz_optional_str(validation_context.get("input_tokens_hash"))
        if expected_input_hash is not None and expected_input_hash != computed_input_tokens_hash:
            validation_issues.append(
                "input_tokens_hash mismatch with validation_context "
                f"(expected={expected_input_hash}, computed={computed_input_tokens_hash})"
            )

        expected_target_ids = validation_context.get("target_token_ids")
        if expected_target_ids is not None:
            expected_target_tensor = _phase0_to_int64_tensor(expected_target_ids).reshape(-1)
            if not torch.equal(expected_target_tensor, target_token_ids.reshape(-1)):
                validation_issues.append("target_token_ids mismatch with validation_context")

        expected_target_hash = _phase0_npz_optional_str(
            validation_context.get("target_token_ids_hash")
        )
        if (
            expected_target_hash is not None
            and expected_target_hash != computed_target_token_ids_hash
        ):
            validation_issues.append(
                "target_token_ids_hash mismatch with validation_context "
                f"(expected={expected_target_hash}, computed={computed_target_token_ids_hash})"
            )

        expected_clt_hash = _phase0_npz_optional_str(validation_context.get("clt_constants_hash"))
        if (
            expected_clt_hash is not None
            and stored_clt_constants_hash is not None
            and expected_clt_hash != stored_clt_constants_hash
        ):
            validation_issues.append(
                "clt_constants_hash mismatch with validation_context "
                f"(expected={expected_clt_hash}, bundle={stored_clt_constants_hash})"
            )

    if validation_issues and policy == "strict":
        raise ValueError("Phase-0 donor bundle validation failed: " + "; ".join(validation_issues))

    dtype_metadata = {
        "activation_values_dtype_recorded": activation_values_dtype_recorded,
        "activation_values_dtype_loaded": str(activation_values.dtype).replace("torch.", ""),
        "activation_values_raw_uint16_present": bool(raw_uint16_array.size > 0),
        "exact_bfloat16_reconstruction": bool(exact_bfloat16_reconstructed),
        "dtype_roundtrip_loss": bool(dtype_roundtrip_loss),
    }

    validation_metadata = {
        "context_policy": policy,
        "warnings": list(validation_issues) if policy == "warn" else [],
        "validation_failure_count": int(len(validation_issues)),
        "validated": bool(len(validation_issues) == 0),
        "computed_hashes": {
            "active_feature_membership_hash_raw_order": computed_membership_hash_raw,
            "active_feature_membership_hash_canonical": computed_membership_hash_canonical,
            "active_feature_values_hash": computed_values_hash,
            "input_tokens_hash": computed_input_tokens_hash,
            "target_token_ids_hash": computed_target_token_ids_hash,
        },
        "stored_hashes": {
            "active_feature_membership_hash_raw_order": stored_membership_hash_raw,
            "active_feature_membership_hash_canonical": stored_membership_hash_canonical,
            "active_feature_values_hash": stored_values_hash,
            "input_tokens_hash": stored_input_tokens_hash,
            "target_token_ids_hash": stored_target_token_ids_hash,
            "clt_constants_hash": stored_clt_constants_hash,
        },
    }

    return {
        "schema_version": schema_version,
        "replay_kind": replay_kind,
        "status": status,
        "active_features": active_features,
        "activation_values": activation_values,
        "activation_matrix_shape": activation_matrix_shape,
        "active_feature_count": active_feature_count,
        "input_tokens": input_tokens,
        "target_token_ids": target_token_ids,
        "dtype_metadata": dtype_metadata,
        "validation_metadata": validation_metadata,
    }


def _build_phase0_replay_validation_context(
    *,
    input_tokens: torch.Tensor,
    target_token_ids: torch.Tensor,
    activation_matrix: torch.Tensor,
    clt_constants_hash: str | None,
) -> dict[str, object]:
    activation_matrix = activation_matrix.coalesce()
    activation_indices = activation_matrix.indices()
    activation_shape = tuple(int(dim) for dim in activation_matrix.shape)
    input_tokens_cpu = input_tokens.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    target_token_ids_cpu = target_token_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)

    return {
        "input_tokens": input_tokens_cpu,
        "input_tokens_hash": _hash_index_tensor(input_tokens_cpu),
        "target_token_ids": target_token_ids_cpu,
        "target_token_ids_hash": (
            _hash_index_tensor(target_token_ids_cpu)
            if int(target_token_ids_cpu.numel()) > 0
            else None
        ),
        "active_feature_membership_hash_raw_order": _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_shape,
            canonicalize=False,
        ),
        "active_feature_membership_hash_canonical": _hash_sparse_membership_indices(
            activation_indices,
            shape=activation_shape,
            canonicalize=True,
        ),
        "clt_constants_hash": clt_constants_hash,
    }


def _build_phase0_activation_matrix_from_loaded_bundle(
    loaded_bundle: dict[str, object],
    *,
    device: torch.device,
) -> torch.Tensor:
    active_features = cast(torch.Tensor, loaded_bundle["active_features"])
    activation_values = cast(torch.Tensor, loaded_bundle["activation_values"])
    activation_shape = tuple(int(dim) for dim in loaded_bundle["activation_matrix_shape"])
    if len(activation_shape) != 3:
        raise ValueError(
            "loaded Phase-0 donor activation_matrix_shape must have length 3 "
            f"(got {activation_shape})"
        )
    if active_features.ndim != 2 or active_features.shape[1] != 3:
        raise ValueError(
            "loaded Phase-0 donor active_features must have shape [N, 3] "
            f"(got {tuple(active_features.shape)})"
        )

    activation_indices = active_features.T.to(device=device, dtype=torch.int64, non_blocking=True)
    activation_values = activation_values.to(device=device, non_blocking=True)
    return torch.sparse_coo_tensor(
        activation_indices,
        activation_values,
        size=activation_shape,
        device=device,
        dtype=activation_values.dtype,
    ).coalesce()


def _build_phase0_replay_metadata(
    *,
    mode: str,
    status: str,
    donor_bundle_path: str | os.PathLike[str] | None,
    context_policy: str,
    validation_warnings: list[str] | None = None,
    validation_failure_count: int = 0,
    dtype_metadata: dict[str, object] | None = None,
    host_hashes: dict[str, object] | None = None,
    donor_hashes: dict[str, object] | None = None,
    host_active_feature_count: int | None = None,
    donor_active_feature_count: int | None = None,
    replay_single_step_intended: bool = True,
    note: str | None = None,
) -> dict[str, object]:
    donor_path_text = os.fspath(donor_bundle_path) if donor_bundle_path is not None else None
    warnings = list(validation_warnings or [])
    warning_count = int(len(warnings))
    if warning_count == 0:
        warning_count = int(max(validation_failure_count, 0))

    return {
        "schema_version": 1,
        "status": str(status),
        "mode": str(mode),
        "context_policy": str(context_policy),
        "donor_bundle_path": donor_path_text,
        "donor_bundle_basename": (
            os.path.basename(donor_path_text) if isinstance(donor_path_text, str) else None
        ),
        "validation_warnings": warnings,
        "validation_warning_count": int(warning_count),
        "validation_failure_count": int(validation_failure_count),
        "dtype_metadata": dict(dtype_metadata) if isinstance(dtype_metadata, dict) else {},
        "host_active_feature_count": host_active_feature_count,
        "donor_active_feature_count": donor_active_feature_count,
        "host_hashes": dict(host_hashes) if isinstance(host_hashes, dict) else {},
        "donor_hashes": dict(donor_hashes) if isinstance(donor_hashes, dict) else {},
        "replay_single_step_intended": bool(replay_single_step_intended),
        "note": note,
    }


def _phase3_npz_optional_str(value: object) -> str | None:
    return _phase0_npz_optional_str(value)


def _phase3_npz_int(value: object, *, default: int = 0) -> int:
    return _phase0_npz_int(value, default=default)


def _build_phase3_replay_metadata(
    *,
    replay_kind: str,
    mode: str,
    status: str,
    donor_bundle_path: str | os.PathLike[str] | None,
    validation_policy: str,
    validation_failure_count: int = 0,
    error: str | None = None,
    donor_hashes: dict[str, object] | None = None,
    host_hashes: dict[str, object] | None = None,
    source: str | None = None,
    note: str | None = None,
) -> dict[str, object]:
    donor_path_text = os.fspath(donor_bundle_path) if donor_bundle_path is not None else None
    return {
        "schema_version": 1,
        "replay_kind": str(replay_kind),
        "mode": str(mode),
        "status": str(status),
        "validation_policy": str(validation_policy),
        "donor_bundle_path": donor_path_text,
        "donor_bundle_basename": (
            os.path.basename(donor_path_text) if isinstance(donor_path_text, str) else None
        ),
        "validation_failure_count": int(validation_failure_count),
        "error": error,
        "donor_hashes": dict(donor_hashes) if isinstance(donor_hashes, dict) else {},
        "host_hashes": dict(host_hashes) if isinstance(host_hashes, dict) else {},
        "source": source,
        "note": note,
    }


def _load_phase3_gradient_donor_bundle_npz(
    donor_bundle_path: str | os.PathLike[str],
    *,
    target_token_ids: torch.Tensor,
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    expected_n_layers: int,
    expected_gradient_batch_size: int,
    expected_n_positions: int,
    expected_d_model: int,
    validation_policy: Literal["strict"] = "strict",
) -> dict[str, object]:
    if validation_policy != "strict":
        raise ValueError("Phase-3 replay currently supports only strict validation")

    with np.load(donor_bundle_path, allow_pickle=False) as donor_npz:
        donor_payload = {key: donor_npz[key] for key in donor_npz.files}

    schema_version = _phase3_npz_int(donor_payload.get("schema_version"), default=0)
    capture_kind = _phase3_npz_optional_str(donor_payload.get("capture_kind")) or ""
    status = _phase3_npz_optional_str(donor_payload.get("status")) or ""
    gradients_array = np.ascontiguousarray(
        np.asarray(donor_payload.get("gradients", np.empty((0, 0, 0, 0), dtype=np.float32)))
    )
    gradients = torch.from_numpy(gradients_array).to(dtype=torch.float32)
    donor_target_ids = _phase0_to_int64_tensor(
        donor_payload.get("target_token_ids", np.empty((0,), dtype=np.int64))
    ).reshape(-1)

    target_token_ids_cpu = target_token_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    active_features_cpu = active_features.detach().to(device="cpu", dtype=torch.int64)
    activation_values_cpu = activation_values.detach().to(device="cpu")
    computed_target_hash = (
        _hash_index_tensor(donor_target_ids) if donor_target_ids.numel() else None
    )
    stored_target_hash = _phase3_npz_optional_str(donor_payload.get("target_token_ids_hash"))
    active_feature_count = _phase3_npz_int(
        donor_payload.get("active_feature_count"), default=int(active_features_cpu.shape[0])
    )
    computed_active_hash = _hash_index_tensor(active_features_cpu.reshape(-1))
    stored_active_hash = _phase3_npz_optional_str(donor_payload.get("active_features_hash"))
    computed_activation_hash = _hash_tensor_raw_bytes(activation_values_cpu)
    stored_activation_hash = _phase3_npz_optional_str(donor_payload.get("activation_values_hash"))
    stored_gradient_hash = _phase3_npz_optional_str(donor_payload.get("gradient_hash"))
    computed_gradient_hash = _hash_float_tensor(gradients, dtype=torch.float32)

    validation_issues: list[str] = []
    if schema_version not in {1, 2}:
        validation_issues.append(f"schema_version mismatch (expected 1 or 2, got {schema_version})")
    expected_capture_kind = (
        "phase3_gradient_bundle_v1" if schema_version == 1 else "phase3_gradient_bundle_v2"
    )
    if capture_kind != expected_capture_kind:
        validation_issues.append(
            f"capture_kind mismatch (expected {expected_capture_kind!r}, got {capture_kind!r})"
        )
    if status not in {"captured", "captured_replayed_effective_state"}:
        validation_issues.append(f"unexpected status {status!r}")
    if not torch.equal(donor_target_ids, target_token_ids_cpu):
        validation_issues.append("target_token_ids mismatch with runtime targets")
    if not stored_target_hash:
        validation_issues.append("missing target_token_ids_hash")
    if stored_target_hash and computed_target_hash and stored_target_hash != computed_target_hash:
        validation_issues.append(
            "target_token_ids_hash mismatch within donor bundle "
            f"(stored={stored_target_hash}, computed={computed_target_hash})"
        )
    if active_feature_count != int(active_features_cpu.shape[0]):
        validation_issues.append(
            "active_feature_count mismatch "
            f"(declared={active_feature_count}, runtime={int(active_features_cpu.shape[0])})"
        )
    if not stored_active_hash:
        validation_issues.append("missing active_features_hash")
    elif stored_active_hash != computed_active_hash:
        validation_issues.append(
            "active_features_hash mismatch with runtime active feature order "
            f"(donor={stored_active_hash}, runtime={computed_active_hash})"
        )
    if not stored_activation_hash:
        validation_issues.append("missing activation_values_hash")
    elif stored_activation_hash != computed_activation_hash:
        validation_issues.append(
            "activation_values_hash mismatch with runtime activation values "
            f"(donor={stored_activation_hash}, runtime={computed_activation_hash})"
        )
    if gradients.ndim != 4:
        validation_issues.append(
            f"gradients must have shape [layers, batch, positions, d_model] (got={tuple(gradients.shape)})"
        )
    elif (
        int(gradients.shape[0]) != int(expected_n_layers)
        or int(gradients.shape[1])
        != int(expected_gradient_batch_size if schema_version == 1 else target_token_ids_cpu.numel())
        or int(gradients.shape[2]) != int(expected_n_positions)
        or int(gradients.shape[3]) != int(expected_d_model)
    ):
        validation_issues.append(
            "gradients shape mismatch "
            "(expected layers="
            f"{int(expected_n_layers)}, batch={int(expected_gradient_batch_size if schema_version == 1 else target_token_ids_cpu.numel())}, "
            f"positions={int(expected_n_positions)}, d_model={int(expected_d_model)}; "
            f"got={tuple(gradients.shape)})"
        )
    if schema_version == 2:
        recorded_width = _phase3_npz_int(donor_payload.get("canonical_target_width"), default=-1)
        if recorded_width != int(target_token_ids_cpu.numel()):
            validation_issues.append(
                "canonical_target_width mismatch "
                f"(declared={recorded_width}, runtime={int(target_token_ids_cpu.numel())})"
            )
    elif int(gradients.shape[1]) < int(target_token_ids_cpu.numel()):
        validation_issues.append(
            "gradients batch width is smaller than target token count "
            f"(batch={int(gradients.shape[1])}, targets={int(target_token_ids_cpu.numel())})"
        )
    if not torch.isfinite(gradients).all().item():
        validation_issues.append("gradients contain nonfinite values")
    if not stored_gradient_hash:
        validation_issues.append("missing gradient_hash")
    elif stored_gradient_hash != computed_gradient_hash:
        validation_issues.append(
            "gradient_hash mismatch within donor bundle "
            f"(stored={stored_gradient_hash}, computed={computed_gradient_hash})"
        )

    if validation_issues:
        raise ValueError(
            "Phase-3 gradient donor bundle validation failed: " + "; ".join(validation_issues)
        )

    return {
        "status": status,
        "gradients": gradients,
        "validation_metadata": {
            "validated": True,
            "validation_failure_count": 0,
            "computed_hashes": {
                "target_token_ids_hash": computed_target_hash,
                "active_features_hash": computed_active_hash,
                "activation_values_hash": computed_activation_hash,
                "gradient_hash": computed_gradient_hash,
            },
            "stored_hashes": {
                "target_token_ids_hash": stored_target_hash,
                "active_features_hash": stored_active_hash,
                "activation_values_hash": stored_activation_hash,
                "gradient_hash": stored_gradient_hash,
            },
        },
    }


def _load_phase3_row_donor_bundle_npz(
    donor_bundle_path: str | os.PathLike[str],
    *,
    target_token_ids: torch.Tensor,
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    expected_total_active_features: int,
    validation_policy: Literal["strict"] = "strict",
) -> dict[str, object]:
    if validation_policy != "strict":
        raise ValueError("Phase-3 replay currently supports only strict validation")

    with np.load(donor_bundle_path, allow_pickle=False) as donor_npz:
        donor_payload = {key: donor_npz[key] for key in donor_npz.files}

    schema_version = _phase3_npz_int(donor_payload.get("schema_version"), default=0)
    capture_kind = _phase3_npz_optional_str(donor_payload.get("capture_kind")) or ""
    status = _phase3_npz_optional_str(donor_payload.get("status")) or ""
    feature_rows_array = np.ascontiguousarray(
        np.asarray(donor_payload.get("phase3_feature_rows", np.empty((0, 0), dtype=np.float32)))
    )
    feature_rows = torch.from_numpy(feature_rows_array).to(dtype=torch.float32)
    row_abs_sums = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(donor_payload.get("row_abs_sums", np.empty((0,), dtype=np.float64)))
        )
    ).to(dtype=torch.float64)
    feature_abs_sums = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(donor_payload.get("feature_abs_sums", np.empty((0,), dtype=np.float64)))
        )
    ).to(dtype=torch.float64)
    error_abs_sums = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(donor_payload.get("error_abs_sums", np.empty((0,), dtype=np.float64)))
        )
    ).to(dtype=torch.float64)
    token_abs_sums = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(donor_payload.get("token_abs_sums", np.empty((0,), dtype=np.float64)))
        )
    ).to(dtype=torch.float64)
    donor_target_ids = _phase0_to_int64_tensor(
        donor_payload.get("target_token_ids", np.empty((0,), dtype=np.int64))
    ).reshape(-1)

    target_token_ids_cpu = target_token_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    active_features_cpu = active_features.detach().to(device="cpu", dtype=torch.int64)
    activation_values_cpu = activation_values.detach().to(device="cpu")
    computed_target_hash = (
        _hash_index_tensor(donor_target_ids) if donor_target_ids.numel() else None
    )
    stored_target_hash = _phase3_npz_optional_str(donor_payload.get("target_token_ids_hash"))
    computed_active_hash = _hash_index_tensor(active_features_cpu.reshape(-1))
    stored_active_hash = _phase3_npz_optional_str(donor_payload.get("active_features_hash"))
    computed_activation_hash = _hash_tensor_raw_bytes(activation_values_cpu)
    stored_activation_hash = _phase3_npz_optional_str(donor_payload.get("activation_values_hash"))
    stored_row_hash = _phase3_npz_optional_str(donor_payload.get("row_hash"))
    computed_row_hash = _hash_float_tensor(feature_rows, dtype=torch.float32)
    stored_row_abs_hash = _phase3_npz_optional_str(donor_payload.get("row_abs_sum_hash"))
    computed_row_abs_hash = _hash_float_tensor(row_abs_sums, dtype=torch.float64)
    active_feature_count = _phase3_npz_int(
        donor_payload.get("active_feature_count"), default=int(feature_rows.shape[1])
    )
    total_active_features = _phase3_npz_int(
        donor_payload.get("total_active_features"), default=int(feature_rows.shape[1])
    )

    validation_issues: list[str] = []
    if schema_version != 1:
        validation_issues.append(f"schema_version mismatch (expected 1, got {schema_version})")
    if capture_kind != "phase3_row_bundle_v1":
        validation_issues.append(
            f"capture_kind mismatch (expected 'phase3_row_bundle_v1', got {capture_kind!r})"
        )
    if status not in {"captured", "captured_replayed_effective_state"}:
        validation_issues.append(f"unexpected status {status!r}")
    if not torch.equal(donor_target_ids, target_token_ids_cpu):
        validation_issues.append("target_token_ids mismatch with runtime targets")
    if not stored_target_hash:
        validation_issues.append("missing target_token_ids_hash")
    if stored_target_hash and computed_target_hash and stored_target_hash != computed_target_hash:
        validation_issues.append(
            "target_token_ids_hash mismatch within donor bundle "
            f"(stored={stored_target_hash}, computed={computed_target_hash})"
        )
    expected_shape = (int(target_token_ids_cpu.numel()), int(expected_total_active_features))
    if tuple(feature_rows.shape) != expected_shape:
        validation_issues.append(
            f"phase3_feature_rows shape mismatch (expected={expected_shape}, got={tuple(feature_rows.shape)})"
        )
    if tuple(row_abs_sums.shape) != (int(target_token_ids_cpu.numel()),):
        validation_issues.append(
            "row_abs_sums shape mismatch "
            f"(expected={(int(target_token_ids_cpu.numel()),)}, got={tuple(row_abs_sums.shape)})"
        )
    for name, values in (
        ("feature_abs_sums", feature_abs_sums),
        ("error_abs_sums", error_abs_sums),
        ("token_abs_sums", token_abs_sums),
    ):
        if tuple(values.shape) != (int(target_token_ids_cpu.numel()),):
            validation_issues.append(
                f"{name} shape mismatch (expected={(int(target_token_ids_cpu.numel()),)}, got={tuple(values.shape)})"
            )
    if active_feature_count != int(expected_total_active_features):
        validation_issues.append(
            "active_feature_count mismatch "
            f"(declared={active_feature_count}, runtime={int(expected_total_active_features)})"
        )
    if total_active_features != int(expected_total_active_features):
        validation_issues.append(
            "total_active_features mismatch "
            f"(declared={total_active_features}, runtime={int(expected_total_active_features)})"
        )
    if not stored_active_hash:
        validation_issues.append("missing active_features_hash")
    elif stored_active_hash != computed_active_hash:
        validation_issues.append(
            "active_features_hash mismatch with runtime active feature order "
            f"(donor={stored_active_hash}, runtime={computed_active_hash})"
        )
    if not stored_activation_hash:
        validation_issues.append("missing activation_values_hash")
    elif stored_activation_hash != computed_activation_hash:
        validation_issues.append(
            "activation_values_hash mismatch with runtime activation values "
            f"(donor={stored_activation_hash}, runtime={computed_activation_hash})"
        )
    if not torch.isfinite(feature_rows).all().item():
        validation_issues.append("phase3_feature_rows contain nonfinite values")
    if not torch.isfinite(row_abs_sums).all().item():
        validation_issues.append("row_abs_sums contain nonfinite values")
    if not torch.isfinite(feature_abs_sums).all().item():
        validation_issues.append("feature_abs_sums contain nonfinite values")
    if not torch.isfinite(error_abs_sums).all().item():
        validation_issues.append("error_abs_sums contain nonfinite values")
    if not torch.isfinite(token_abs_sums).all().item():
        validation_issues.append("token_abs_sums contain nonfinite values")
    if tuple(feature_abs_sums.shape) == (int(target_token_ids_cpu.numel()),):
        computed_feature_abs_sums = _compute_row_abs_sums(feature_rows, dtype=torch.float64)
        if not torch.allclose(feature_abs_sums, computed_feature_abs_sums, rtol=1e-5, atol=1e-6):
            validation_issues.append(
                "feature_abs_sums do not match phase3_feature_rows absolute sums"
            )
    if (
        tuple(row_abs_sums.shape)
        == tuple(feature_abs_sums.shape)
        == tuple(error_abs_sums.shape)
        == tuple(token_abs_sums.shape)
    ):
        split_total = feature_abs_sums + error_abs_sums + token_abs_sums
        if not torch.allclose(row_abs_sums, split_total, rtol=1e-5, atol=1e-6):
            validation_issues.append("row_abs_sums do not match feature/error/token split sums")
    if not stored_row_hash:
        validation_issues.append("missing row_hash")
    elif stored_row_hash != computed_row_hash:
        validation_issues.append(
            f"row_hash mismatch within donor bundle (stored={stored_row_hash}, computed={computed_row_hash})"
        )
    if not stored_row_abs_hash:
        validation_issues.append("missing row_abs_sum_hash")
    elif stored_row_abs_hash != computed_row_abs_hash:
        validation_issues.append(
            "row_abs_sum_hash mismatch within donor bundle "
            f"(stored={stored_row_abs_hash}, computed={computed_row_abs_hash})"
        )

    if validation_issues:
        raise ValueError(
            "Phase-3 row donor bundle validation failed: " + "; ".join(validation_issues)
        )

    return {
        "status": status,
        "phase3_feature_rows": feature_rows,
        "row_abs_sums": row_abs_sums,
        "feature_abs_sums": feature_abs_sums,
        "error_abs_sums": error_abs_sums,
        "token_abs_sums": token_abs_sums,
        "validation_metadata": {
            "validated": True,
            "validation_failure_count": 0,
            "computed_hashes": {
                "target_token_ids_hash": computed_target_hash,
                "active_features_hash": computed_active_hash,
                "activation_values_hash": computed_activation_hash,
                "row_hash": computed_row_hash,
                "row_abs_sum_hash": computed_row_abs_hash,
            },
            "stored_hashes": {
                "target_token_ids_hash": stored_target_hash,
                "active_features_hash": stored_active_hash,
                "activation_values_hash": stored_activation_hash,
                "row_hash": stored_row_hash,
                "row_abs_sum_hash": stored_row_abs_hash,
            },
        },
    }


def _build_phase3_seed_influence_topk(
    *,
    ranked_feature_indices: torch.Tensor,
    seed_feature_influences: torch.Tensor,
    feat_layers: torch.Tensor,
    feat_positions: torch.Tensor,
    feat_ids: torch.Tensor,
    top_k: int = 8,
) -> list[dict[str, object]]:
    ranked_cpu = ranked_feature_indices.detach().to(device="cpu", dtype=torch.int64)
    influences_cpu = seed_feature_influences.detach().to(device="cpu", dtype=torch.float64)
    layers_cpu = feat_layers.detach().to(device="cpu", dtype=torch.int64)
    positions_cpu = feat_positions.detach().to(device="cpu", dtype=torch.int64)
    feat_ids_cpu = feat_ids.detach().to(device="cpu", dtype=torch.int64)

    limit = min(max(int(top_k), 0), int(ranked_cpu.numel()))
    if limit == 0:
        return []

    entries: list[dict[str, object]] = []
    for rank in range(limit):
        feature_idx = int(ranked_cpu[rank].item())
        entries.append(
            {
                "rank": rank + 1,
                "feature_index": feature_idx,
                "influence": float(influences_cpu[feature_idx].item()),
                "layer": int(layers_cpu[feature_idx].item()),
                "position": int(positions_cpu[feature_idx].item()),
                "feature_id": int(feat_ids_cpu[feature_idx].item()),
            }
        )
    return entries


def _build_phase3_seed_bundle_payload(
    *,
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    seed_feature_influences: torch.Tensor,
    frontier_pre_locality: torch.Tensor,
    frontier_post_locality: torch.Tensor,
    queue_size: int,
    actual_max_feature_nodes: int,
    total_active_features: int,
    status: str,
    planner_compute_dtype: torch.dtype,
    influence_compute_dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "status": status,
        "active_features": active_features.detach().to(device="cpu", dtype=torch.int64),
        "activation_values": activation_values.detach().to(device="cpu"),
        "seed_feature_influences": seed_feature_influences.detach().to(device="cpu"),
        "frontier_pre_locality": frontier_pre_locality.detach().to(device="cpu", dtype=torch.int64),
        "frontier_post_locality": frontier_post_locality.detach().to(
            device="cpu", dtype=torch.int64
        ),
        "queue_size": int(queue_size),
        "actual_max_feature_nodes": int(actual_max_feature_nodes),
        "total_active_features": int(total_active_features),
        "planner_compute_dtype": _dtype_to_name(planner_compute_dtype),
        "influence_compute_dtype": _dtype_to_name(influence_compute_dtype),
    }


def _build_phase3_gradient_bundle_payload(
    *,
    gradient_captures: list[dict[str, object]],
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_probabilities: torch.Tensor,
    status: str,
) -> dict[str, object]:
    target_token_ids_cpu = target_token_ids.detach().to(device="cpu", dtype=torch.int64)
    target_probabilities_cpu = target_probabilities.detach().to(device="cpu")
    active_features_cpu = active_features.detach().to(device="cpu", dtype=torch.int64)
    activation_values_cpu = activation_values.detach().to(device="cpu")

    gradient_tensors: list[torch.Tensor] = []
    layer_masks: list[torch.Tensor] = []
    batch_call_indices: list[int] = []
    for capture in gradient_captures:
        gradients = capture.get("gradients")
        layer_mask = capture.get("layer_mask")
        if not isinstance(gradients, torch.Tensor) or not isinstance(layer_mask, torch.Tensor):
            continue
        gradient_tensors.append(gradients.detach().to(device="cpu", dtype=torch.float32))
        layer_masks.append(layer_mask.detach().to(device="cpu", dtype=torch.bool))
        batch_call_indices.append(int(capture.get("batch_call_index", len(batch_call_indices))))

    if gradient_tensors:
        gradients_by_layer = torch.cat(gradient_tensors, dim=1).contiguous()
        layer_mask = torch.stack(layer_masks, dim=0).any(dim=0).to(dtype=torch.bool)
    else:
        gradients_by_layer = torch.empty((0, 0, 0, 0), dtype=torch.float32)
        layer_mask = torch.empty((0,), dtype=torch.bool)

    per_layer_abs_sum: list[float] = []
    per_layer_max_abs: list[float] = []
    per_layer_nonfinite_count: list[int] = []
    per_layer_hashes: list[str] = []
    for layer_idx in range(int(gradients_by_layer.shape[0])):
        layer_values = gradients_by_layer[layer_idx]
        per_layer_hashes.append(_hash_float_tensor(layer_values, dtype=torch.float32))
        finite = torch.isfinite(layer_values)
        per_layer_nonfinite_count.append(int((~finite).sum().item()))
        abs_values = layer_values.detach().abs()
        per_layer_abs_sum.append(float(abs_values.sum(dtype=torch.float64).item()))
        per_layer_max_abs.append(float(abs_values.max().item()) if abs_values.numel() else 0.0)

    return {
        "schema_version": 2,
        "status": status,
        "capture_kind": "phase3_gradient_bundle_v2",
        "gradient_batch_representation": "canonical_target_width_v1",
        "canonical_target_width": int(target_token_ids_cpu.numel()),
        "target_token_ids": target_token_ids_cpu,
        "target_probabilities": target_probabilities_cpu,
        "target_token_ids_hash": _hash_index_tensor(target_token_ids_cpu),
        "target_probability_hash": _hash_float_tensor(
            target_probabilities_cpu,
            dtype=torch.float64,
        ),
        "active_feature_count": int(active_features_cpu.shape[0]),
        "active_features_hash": _hash_index_tensor(active_features_cpu.reshape(-1)),
        "activation_values_hash": _hash_tensor_raw_bytes(activation_values_cpu),
        "gradients": gradients_by_layer,
        "layer_mask": layer_mask,
        "batch_call_indices": torch.tensor(batch_call_indices, dtype=torch.int64),
        "per_layer_abs_sum": torch.tensor(per_layer_abs_sum, dtype=torch.float64),
        "per_layer_max_abs": torch.tensor(per_layer_max_abs, dtype=torch.float64),
        "per_layer_nonfinite_count": torch.tensor(per_layer_nonfinite_count, dtype=torch.int64),
        "per_layer_hashes": per_layer_hashes,
        "gradient_hash": _hash_float_tensor(gradients_by_layer, dtype=torch.float32),
    }


def _build_phase3_row_bundle_payload(
    *,
    feature_rows: list[torch.Tensor],
    row_abs_sums: list[torch.Tensor],
    feature_abs_sums: list[torch.Tensor],
    error_abs_sums: list[torch.Tensor],
    token_abs_sums: list[torch.Tensor],
    active_features: torch.Tensor,
    activation_values: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_probabilities: torch.Tensor,
    total_active_features: int,
    error_column_count: int,
    token_column_count: int,
    status: str,
) -> dict[str, object]:
    target_token_ids_cpu = target_token_ids.detach().to(device="cpu", dtype=torch.int64)
    target_probabilities_cpu = target_probabilities.detach().to(device="cpu")
    active_features_cpu = active_features.detach().to(device="cpu", dtype=torch.int64)
    activation_values_cpu = activation_values.detach().to(device="cpu")

    feature_rows_cpu = (
        torch.cat(
            [row.detach().to(device="cpu", dtype=torch.float32) for row in feature_rows], dim=0
        )
        if feature_rows
        else torch.empty((0, int(total_active_features)), dtype=torch.float32)
    )
    row_abs_sums_cpu = (
        torch.cat(
            [row.detach().to(device="cpu", dtype=torch.float64) for row in row_abs_sums], dim=0
        )
        if row_abs_sums
        else torch.empty((0,), dtype=torch.float64)
    )
    feature_abs_sums_cpu = (
        torch.cat(
            [row.detach().to(device="cpu", dtype=torch.float64) for row in feature_abs_sums], dim=0
        )
        if feature_abs_sums
        else torch.empty((0,), dtype=torch.float64)
    )
    error_abs_sums_cpu = (
        torch.cat(
            [row.detach().to(device="cpu", dtype=torch.float64) for row in error_abs_sums], dim=0
        )
        if error_abs_sums
        else torch.empty((0,), dtype=torch.float64)
    )
    token_abs_sums_cpu = (
        torch.cat(
            [row.detach().to(device="cpu", dtype=torch.float64) for row in token_abs_sums], dim=0
        )
        if token_abs_sums
        else torch.empty((0,), dtype=torch.float64)
    )

    return {
        "schema_version": 1,
        "status": status,
        "capture_kind": "phase3_row_bundle_v1",
        "target_token_ids": target_token_ids_cpu,
        "target_probabilities": target_probabilities_cpu,
        "target_token_ids_hash": _hash_index_tensor(target_token_ids_cpu),
        "target_probability_hash": _hash_float_tensor(
            target_probabilities_cpu,
            dtype=torch.float64,
        ),
        "active_feature_count": int(active_features_cpu.shape[0]),
        "active_features_hash": _hash_index_tensor(active_features_cpu.reshape(-1)),
        "activation_values_hash": _hash_tensor_raw_bytes(activation_values_cpu),
        "phase3_feature_rows": feature_rows_cpu,
        "row_abs_sums": row_abs_sums_cpu,
        "feature_abs_sums": feature_abs_sums_cpu,
        "error_abs_sums": error_abs_sums_cpu,
        "token_abs_sums": token_abs_sums_cpu,
        "total_active_features": int(total_active_features),
        "error_column_count": int(error_column_count),
        "token_column_count": int(token_column_count),
        "row_hash": _hash_float_tensor(feature_rows_cpu, dtype=torch.float32),
        "row_abs_sum_hash": _hash_float_tensor(row_abs_sums_cpu, dtype=torch.float64),
    }
