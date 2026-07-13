from __future__ import annotations

import numpy as np
import pytest
import torch

from circuit_tracer.attribution.nnsight.replay import (
    _build_phase3_gradient_bundle_payload,
    _hash_float_tensor,
    _hash_index_tensor,
    _hash_tensor_raw_bytes,
    _load_phase3_gradient_donor_bundle_npz,
    _load_phase3_row_donor_bundle_npz,
)
from circuit_tracer.attribution.context_nnsight import _slice_phase3_gradient_replay_batch


def test_phase3_gradient_bundle_v2_replays_canonical_target_width_not_session_capacity(tmp_path):
    target_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    activation_values = torch.tensor([1.0], dtype=torch.float32)
    gradients = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
    payload = _build_phase3_gradient_bundle_payload(
        gradient_captures=[{"gradients": gradients, "layer_mask": torch.tensor([True, True])}],
        active_features=active_features,
        activation_values=activation_values,
        target_token_ids=target_ids,
        target_probabilities=torch.ones(3),
        status="captured",
    )
    path = tmp_path / "capacity8_targets3_physical2.npz"
    np.savez_compressed(path, **payload)

    loaded = _load_phase3_gradient_donor_bundle_npz(
        path,
        target_token_ids=target_ids,
        active_features=active_features,
        activation_values=activation_values,
        expected_n_layers=2,
        expected_gradient_batch_size=8,
        expected_n_positions=1,
        expected_d_model=1,
    )

    assert payload["schema_version"] == 2
    assert payload["canonical_target_width"] == 3
    assert torch.equal(loaded["gradients"], gradients)


def test_phase3_gradient_donor_validation_rejects_target_mismatch(tmp_path):
    gradients = torch.zeros((2, 4, 3, 5), dtype=torch.float32)
    donor_targets = torch.tensor([10], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 0, 3]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    path = tmp_path / "gradient.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_gradient_bundle_v1"),
        target_token_ids=donor_targets.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(donor_targets)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(3, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(active_features.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        gradients=gradients.numpy(),
        layer_mask=np.array([True, True]),
        batch_call_indices=np.array([1], dtype=np.int64),
        per_layer_abs_sum=np.zeros((2,), dtype=np.float64),
        per_layer_max_abs=np.zeros((2,), dtype=np.float64),
        per_layer_nonfinite_count=np.zeros((2,), dtype=np.int64),
        per_layer_hashes=np.array(["a", "b"]),
        gradient_hash=np.array(_hash_float_tensor(gradients, dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="target_token_ids mismatch"):
        _load_phase3_gradient_donor_bundle_npz(
            path,
            target_token_ids=torch.tensor([11], dtype=torch.int64),
            active_features=active_features,
            activation_values=activation_values,
            expected_n_layers=2,
            expected_gradient_batch_size=4,
            expected_n_positions=3,
            expected_d_model=5,
        )


def test_phase3_row_donor_validation_rejects_active_feature_hash_mismatch(tmp_path):
    target_ids = torch.tensor([10], dtype=torch.int64)
    donor_active = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    runtime_active = torch.tensor([[0, 0, 1], [0, 1, 3]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    rows = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    row_abs_sums = torch.tensor([0.75], dtype=torch.float64)
    path = tmp_path / "row.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_row_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(donor_active.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        phase3_feature_rows=rows.numpy(),
        row_abs_sums=row_abs_sums.numpy(),
        feature_abs_sums=np.array([0.75], dtype=np.float64),
        error_abs_sums=np.array([0.0], dtype=np.float64),
        token_abs_sums=np.array([0.0], dtype=np.float64),
        total_active_features=np.array(2, dtype=np.int64),
        error_column_count=np.array(0, dtype=np.int64),
        token_column_count=np.array(0, dtype=np.int64),
        row_hash=np.array(_hash_float_tensor(rows, dtype=torch.float32)),
        row_abs_sum_hash=np.array(_hash_float_tensor(row_abs_sums, dtype=torch.float64)),
    )

    with pytest.raises(ValueError, match="active_features_hash mismatch"):
        _load_phase3_row_donor_bundle_npz(
            path,
            target_token_ids=target_ids,
            active_features=runtime_active,
            activation_values=activation_values,
            expected_total_active_features=2,
        )


def test_phase3_gradient_donor_validation_rejects_active_feature_hash_mismatch(tmp_path):
    gradients = torch.zeros((2, 1, 3, 5), dtype=torch.float32)
    target_ids = torch.tensor([10], dtype=torch.int64)
    donor_active = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    runtime_active = torch.tensor([[0, 0, 1], [0, 1, 3]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    path = tmp_path / "gradient_active_mismatch.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_gradient_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(donor_active.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        gradients=gradients.numpy(),
        layer_mask=np.array([True, True]),
        batch_call_indices=np.array([1], dtype=np.int64),
        per_layer_abs_sum=np.zeros((2,), dtype=np.float64),
        per_layer_max_abs=np.zeros((2,), dtype=np.float64),
        per_layer_nonfinite_count=np.zeros((2,), dtype=np.int64),
        per_layer_hashes=np.array(["a", "b"]),
        gradient_hash=np.array(_hash_float_tensor(gradients, dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="active_features_hash mismatch"):
        _load_phase3_gradient_donor_bundle_npz(
            path,
            target_token_ids=target_ids,
            active_features=runtime_active,
            activation_values=activation_values,
            expected_n_layers=2,
            expected_gradient_batch_size=1,
            expected_n_positions=3,
            expected_d_model=5,
        )


def test_phase3_gradient_donor_validation_accepts_trace_batch_width(tmp_path):
    gradients = torch.zeros((2, 2, 3, 5), dtype=torch.float32)
    target_ids = torch.tensor([10], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    path = tmp_path / "gradient_trace_batch.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_gradient_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(active_features.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        gradients=gradients.numpy(),
        layer_mask=np.array([True, True]),
        batch_call_indices=np.array([1], dtype=np.int64),
        per_layer_abs_sum=np.zeros((2,), dtype=np.float64),
        per_layer_max_abs=np.zeros((2,), dtype=np.float64),
        per_layer_nonfinite_count=np.zeros((2,), dtype=np.int64),
        per_layer_hashes=np.array(["a", "b"]),
        gradient_hash=np.array(_hash_float_tensor(gradients, dtype=torch.float32)),
    )

    loaded = _load_phase3_gradient_donor_bundle_npz(
        path,
        target_token_ids=target_ids,
        active_features=active_features,
        activation_values=activation_values,
        expected_n_layers=2,
        expected_gradient_batch_size=2,
        expected_n_positions=3,
        expected_d_model=5,
    )

    assert torch.equal(loaded["gradients"], gradients)


def test_phase3_gradient_donor_validation_rejects_wrong_trace_batch_width(tmp_path):
    gradients = torch.zeros((2, 2, 3, 5), dtype=torch.float32)
    target_ids = torch.tensor([10], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    path = tmp_path / "gradient_wrong_batch_width.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_gradient_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(active_features.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        gradients=gradients.numpy(),
        layer_mask=np.array([True, True]),
        batch_call_indices=np.array([1], dtype=np.int64),
        per_layer_abs_sum=np.zeros((2,), dtype=np.float64),
        per_layer_max_abs=np.zeros((2,), dtype=np.float64),
        per_layer_nonfinite_count=np.zeros((2,), dtype=np.int64),
        per_layer_hashes=np.array(["a", "b"]),
        gradient_hash=np.array(_hash_float_tensor(gradients, dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="gradients shape mismatch"):
        _load_phase3_gradient_donor_bundle_npz(
            path,
            target_token_ids=target_ids,
            active_features=active_features,
            activation_values=activation_values,
            expected_n_layers=2,
            expected_gradient_batch_size=1,
            expected_n_positions=3,
            expected_d_model=5,
        )


def test_phase3_row_donor_validation_rejects_missing_active_feature_hash(tmp_path):
    target_ids = torch.tensor([10], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    rows = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    row_abs_sums = torch.tensor([0.75], dtype=torch.float64)
    path = tmp_path / "row_missing_hash.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_row_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        phase3_feature_rows=rows.numpy(),
        row_abs_sums=row_abs_sums.numpy(),
        feature_abs_sums=np.array([0.75], dtype=np.float64),
        error_abs_sums=np.array([0.0], dtype=np.float64),
        token_abs_sums=np.array([0.0], dtype=np.float64),
        total_active_features=np.array(2, dtype=np.int64),
        error_column_count=np.array(0, dtype=np.int64),
        token_column_count=np.array(0, dtype=np.int64),
        row_hash=np.array(_hash_float_tensor(rows, dtype=torch.float32)),
        row_abs_sum_hash=np.array(_hash_float_tensor(row_abs_sums, dtype=torch.float64)),
    )

    with pytest.raises(ValueError, match="missing active_features_hash"):
        _load_phase3_row_donor_bundle_npz(
            path,
            target_token_ids=target_ids,
            active_features=active_features,
            activation_values=activation_values,
            expected_total_active_features=2,
        )


def test_phase3_row_donor_validation_rejects_feature_abs_sum_mismatch(tmp_path):
    target_ids = torch.tensor([10], dtype=torch.int64)
    active_features = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    activation_values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    rows = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    row_abs_sums = torch.tensor([0.75], dtype=torch.float64)
    path = tmp_path / "row_bad_feature_abs.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int64),
        status=np.array("captured"),
        capture_kind=np.array("phase3_row_bundle_v1"),
        target_token_ids=target_ids.numpy(),
        target_probabilities=np.array([1.0], dtype=np.float32),
        target_token_ids_hash=np.array(_hash_index_tensor(target_ids)),
        target_probability_hash=np.array("unused"),
        active_feature_count=np.array(2, dtype=np.int64),
        active_features_hash=np.array(_hash_index_tensor(active_features.reshape(-1))),
        activation_values_hash=np.array(_hash_tensor_raw_bytes(activation_values)),
        phase3_feature_rows=rows.numpy(),
        row_abs_sums=row_abs_sums.numpy(),
        feature_abs_sums=np.array([0.5], dtype=np.float64),
        error_abs_sums=np.array([0.25], dtype=np.float64),
        token_abs_sums=np.array([0.0], dtype=np.float64),
        total_active_features=np.array(2, dtype=np.int64),
        error_column_count=np.array(0, dtype=np.int64),
        token_column_count=np.array(0, dtype=np.int64),
        row_hash=np.array(_hash_float_tensor(rows, dtype=torch.float32)),
        row_abs_sum_hash=np.array(_hash_float_tensor(row_abs_sums, dtype=torch.float64)),
    )

    with pytest.raises(ValueError, match="feature_abs_sums do not match"):
        _load_phase3_row_donor_bundle_npz(
            path,
            target_token_ids=target_ids,
            active_features=active_features,
            activation_values=activation_values,
            expected_total_active_features=2,
        )


def test_phase3_gradient_replay_slices_target_batches_by_offset():
    replay_gradients = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)

    sliced = _slice_phase3_gradient_replay_batch(
        replay_gradients,
        layer=1,
        column_offset=2,
        batch_size=2,
    )

    assert torch.equal(sliced, replay_gradients[1, 2:4])


def test_phase3_gradient_replay_slice_rejects_short_final_batch():
    replay_gradients = torch.zeros((2, 3, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="batch slice shape mismatch"):
        _slice_phase3_gradient_replay_batch(
            replay_gradients,
            layer=1,
            column_offset=2,
            batch_size=2,
        )
