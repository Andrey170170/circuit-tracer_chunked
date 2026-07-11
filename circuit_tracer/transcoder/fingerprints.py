import hashlib
import json
from typing import cast

import numpy as np
import torch

from circuit_tracer.transcoder.activation_functions import JumpReLU


class FingerprintMixin:
    @staticmethod
    def _resolve_phase0_activation_threshold_compare_mode(mode: str) -> str:
        aliases = {
            "baseline": "baseline",
            "default": "baseline",
            "bf16": "bf16",
            "bfloat16": "bf16",
            "fp32": "fp32",
            "float32": "fp32",
            "torch.float32": "fp32",
            "fp64": "fp64",
            "float64": "fp64",
            "torch.float64": "fp64",
        }
        normalized = str(mode).strip().lower()
        resolved = aliases.get(normalized)
        if resolved is None:
            allowed = "baseline, bf16, fp32, fp64"
            raise ValueError(
                "phase0_activation_threshold_compare_mode must be one of "
                f"{{{allowed}}} (got {mode!r})"
            )
        return resolved

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        if dtype == torch.bfloat16:
            return "bfloat16"
        if dtype == torch.float32:
            return "float32"
        if dtype == torch.float64:
            return "float64"
        return str(dtype)

    @staticmethod
    def _hash_json_payload(payload: object) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:16]

    @staticmethod
    def _hash_tensor_payload(
        values: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> str:
        resolved = values.detach()
        if dtype is not None:
            resolved = resolved.to(dtype=dtype)
        resolved_cpu = resolved.to(device="cpu").contiguous()
        hasher = hashlib.blake2s(digest_size=8)
        hasher.update(np.asarray(list(resolved_cpu.shape), dtype=np.int64).tobytes())
        hasher.update(resolved_cpu.numpy().tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _build_compact_tensor_stats(
        values: torch.Tensor,
        *,
        epsilon: float = 1e-12,
    ) -> dict[str, object]:
        flat = values.detach().to(device="cpu", dtype=torch.float64).flatten()
        count = int(flat.numel())
        if count == 0:
            return {
                "count": 0,
                "finite_count": 0,
                "nonfinite_count": 0,
                "nan_count": 0,
                "posinf_count": 0,
                "neginf_count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "abs_sum": 0.0,
                "abs_max": None,
                "effective_nonzero_count": 0,
                "effective_zero_count": 0,
                "epsilon": float(epsilon),
            }

        finite_mask = torch.isfinite(flat)
        finite_values = flat[finite_mask]
        abs_flat = flat.abs()
        effective_nonzero_count = int((abs_flat > epsilon).sum().item())
        return {
            "count": count,
            "finite_count": int(finite_mask.sum().item()),
            "nonfinite_count": int(count - int(finite_mask.sum().item())),
            "nan_count": int(torch.isnan(flat).sum().item()),
            "posinf_count": int(torch.isposinf(flat).sum().item()),
            "neginf_count": int(torch.isneginf(flat).sum().item()),
            "min": float(finite_values.min().item()) if finite_values.numel() > 0 else None,
            "max": float(finite_values.max().item()) if finite_values.numel() > 0 else None,
            "mean": float(finite_values.mean().item()) if finite_values.numel() > 0 else None,
            "abs_sum": float(abs_flat.sum().item()),
            "abs_max": float(abs_flat.max().item()) if abs_flat.numel() > 0 else None,
            "effective_nonzero_count": effective_nonzero_count,
            "effective_zero_count": int(count - effective_nonzero_count),
            "epsilon": float(epsilon),
        }

    @staticmethod
    def _hash_sparse_membership_indices_2d(
        indices: torch.Tensor,
        *,
        shape: tuple[int, int],
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

        flat = indices_cpu[:, 0] * int(shape[1]) + indices_cpu[:, 1]
        hasher.update(torch.sort(flat).values.contiguous().numpy().tobytes())
        return hasher.hexdigest()

    def _build_sampled_tensor_fingerprint(
        self,
        values: torch.Tensor,
        *,
        sample_limit: int = 4096,
        hash_dtype: torch.dtype = torch.float32,
    ) -> dict[str, object]:
        flat = values.detach().flatten()
        total_count = int(flat.numel())
        sample_limit = max(1, int(sample_limit))
        if total_count == 0:
            sample = flat
            sample_stride = 1
        elif total_count <= sample_limit:
            sample = flat
            sample_stride = 1
        else:
            sample_count = min(total_count, sample_limit)
            sample_indices = (
                torch.linspace(
                    0,
                    total_count - 1,
                    steps=sample_count,
                    device=flat.device,
                )
                .round()
                .to(dtype=torch.int64)
            )
            sample = flat.index_select(0, sample_indices)
            sample_stride = max(1, total_count // sample_count)

        return {
            "shape": list(values.shape),
            "dtype": self._dtype_name(values.dtype),
            "element_count": total_count,
            "sample_count": int(sample.numel()),
            "sample_stride": int(sample_stride),
            "sample_hash": self._hash_tensor_payload(sample, dtype=hash_dtype),
            "sample_hash_dtype": self._dtype_name(hash_dtype),
            "sample_stats": self._build_compact_tensor_stats(sample, epsilon=1e-12),
        }

    def _build_layer_constant_fingerprint(
        self,
        *,
        layer_id: int,
        encoder_weights: torch.Tensor,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "layer": int(layer_id),
            "encoder_weight": self._build_sampled_tensor_fingerprint(
                encoder_weights,
                sample_limit=4096,
            ),
            "encoder_bias_hash_fp32": self._hash_tensor_payload(
                self.b_enc[layer_id],
                dtype=torch.float32,
            ),
            "encoder_bias_stats": self._build_compact_tensor_stats(
                self.b_enc[layer_id],
                epsilon=1e-12,
            ),
        }
        if isinstance(self.activation_function, JumpReLU):
            threshold = cast(JumpReLU, self.activation_function).threshold[layer_id]
            payload["threshold_hash_fp32"] = self._hash_tensor_payload(
                threshold,
                dtype=torch.float32,
            )
            payload["threshold_stats"] = self._build_compact_tensor_stats(
                threshold,
                epsilon=1e-12,
            )
        payload["layer_constant_hash"] = self._hash_json_payload(payload)
        return payload

    def configure_phase0_activation_threshold_compare(
        self,
        *,
        mode: str = "baseline",
        collect_diagnostics: bool = False,
        sample_limit_per_layer: int = 3,
    ) -> None:
        resolved_mode = self._resolve_phase0_activation_threshold_compare_mode(mode)
        self._phase0_activation_threshold_compare_mode = resolved_mode
        self._phase0_threshold_membership_debug_enabled = bool(collect_diagnostics)
        self._phase0_threshold_membership_sample_limit_per_layer = max(
            0,
            int(sample_limit_per_layer),
        )

    def _resolve_phase0_compare_dtype(self) -> torch.dtype | None:
        mode = self._phase0_activation_threshold_compare_mode
        if mode == "bf16":
            return torch.bfloat16
        if mode == "fp32":
            return torch.float32
        if mode == "fp64":
            return torch.float64
        return None

    def _compute_jump_relu_mask(
        self,
        *,
        layer_id: int,
        features: torch.Tensor,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        thresholds = cast(JumpReLU, self.activation_function).threshold[layer_id]
        compare_dtype = self._resolve_phase0_compare_dtype()
        if compare_dtype is None:
            compare_features = features
            compare_thresholds = thresholds
            compare_dtype_name = self._dtype_name(features.dtype)
        else:
            compare_features = features.to(dtype=compare_dtype)
            compare_thresholds = thresholds.to(device=features.device, dtype=compare_dtype)
            compare_dtype_name = self._dtype_name(compare_dtype)

        mask = compare_features > compare_thresholds
        self._diagnostic_stats["phase0_activation_threshold_compare_dtype"] = compare_dtype_name

        if not collect_diagnostics:
            return mask, None

        pre_activation_fingerprint = self._build_sampled_tensor_fingerprint(
            features,
            sample_limit=4096,
            hash_dtype=torch.float32,
        )

        margin_cpu = (
            (compare_features - compare_thresholds)
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        abs_margin_flat = margin_cpu.abs().flatten()
        mask_flat_cpu = mask.detach().to(device="cpu").flatten()
        mask_2d_cpu = mask.detach().to(device="cpu")
        mask_u8 = mask_2d_cpu.to(dtype=torch.uint8)
        compare_margin_fingerprint = self._build_sampled_tensor_fingerprint(
            margin_cpu,
            sample_limit=4096,
            hash_dtype=torch.float64,
        )
        total_entries = int(mask_flat_cpu.numel())
        active_entries = int(mask_flat_cpu.sum().item())

        active_indices = torch.nonzero(mask_2d_cpu, as_tuple=False)
        mask_membership_hash_raw = self._hash_sparse_membership_indices_2d(
            active_indices,
            shape=(int(mask.shape[0]), int(mask.shape[1])),
            canonicalize=False,
        )
        mask_membership_hash_canonical = self._hash_sparse_membership_indices_2d(
            active_indices,
            shape=(int(mask.shape[0]), int(mask.shape[1])),
            canonicalize=True,
        )

        near_counts_by_epsilon: dict[str, int] = {}
        near_active_counts_by_epsilon: dict[str, int] = {}
        near_inactive_counts_by_epsilon: dict[str, int] = {}
        for epsilon in self._phase0_threshold_near_epsilons:
            epsilon_key = f"abs_lte_{epsilon:.0e}"
            near_mask = abs_margin_flat <= float(epsilon)
            near_count = int(near_mask.sum().item())
            near_active_count = int((near_mask & mask_flat_cpu).sum().item())
            near_counts_by_epsilon[epsilon_key] = near_count
            near_active_counts_by_epsilon[epsilon_key] = near_active_count
            near_inactive_counts_by_epsilon[epsilon_key] = near_count - near_active_count

        sample_limit = min(
            self._phase0_threshold_membership_sample_limit_per_layer,
            total_entries,
        )
        borderline_samples: list[dict[str, object]] = []
        if sample_limit > 0 and total_entries > 0:
            _, sample_indices = torch.topk(
                abs_margin_flat,
                k=sample_limit,
                largest=False,
            )
            margin_flat = margin_cpu.flatten()
            raw_features_flat = features.detach().to(device="cpu", dtype=torch.float64).flatten()
            thresholds_flat = thresholds.detach().to(device="cpu", dtype=torch.float64).flatten()
            for rank, flat_idx in enumerate(sample_indices.tolist(), start=1):
                flat_idx_int = int(flat_idx)
                pos_idx = flat_idx_int // self.d_transcoder
                feat_idx = flat_idx_int % self.d_transcoder
                borderline_samples.append(
                    {
                        "rank": rank,
                        "position": int(pos_idx),
                        "feature_id": int(feat_idx),
                        "active": bool(mask_flat_cpu[flat_idx_int].item()),
                        "pre_activation": float(raw_features_flat[flat_idx_int].item()),
                        "threshold": float(thresholds_flat[feat_idx].item()),
                        "compare_margin": float(margin_flat[flat_idx_int].item()),
                        "abs_compare_margin": float(abs_margin_flat[flat_idx_int].item()),
                    }
                )

        return mask, {
            "layer": int(layer_id),
            "compare_mode": self._phase0_activation_threshold_compare_mode,
            "compare_dtype": compare_dtype_name,
            "pre_activation_hash_fp32": pre_activation_fingerprint["sample_hash"],
            "pre_activation_stats": pre_activation_fingerprint["sample_stats"],
            "pre_activation_fingerprint": pre_activation_fingerprint,
            "compare_margin_hash_fp64": compare_margin_fingerprint["sample_hash"],
            "compare_margin_stats": compare_margin_fingerprint["sample_stats"],
            "compare_margin_fingerprint": compare_margin_fingerprint,
            "mask_value_hash_u8": self._hash_tensor_payload(mask_u8, dtype=torch.uint8),
            "mask_membership_hash_raw_order": mask_membership_hash_raw,
            "mask_membership_hash_canonical": mask_membership_hash_canonical,
            "total_entries": total_entries,
            "active_entries": active_entries,
            "inactive_entries": int(total_entries - active_entries),
            "near_counts_by_epsilon": near_counts_by_epsilon,
            "near_active_counts_by_epsilon": near_active_counts_by_epsilon,
            "near_inactive_counts_by_epsilon": near_inactive_counts_by_epsilon,
            "borderline_samples": borderline_samples,
        }
