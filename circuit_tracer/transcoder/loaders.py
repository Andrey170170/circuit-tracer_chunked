from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from circuit_tracer.transcoder.checkpoint_assets import CheckpointAssetScope
from circuit_tracer.transcoder.checkpoint_manifest import build_checkpoint_manifest
from circuit_tracer.transcoder.checkpoint_working_set import ProviderCheckpointLifecycle
from circuit_tracer.utils import get_default_device

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


DEFAULT_EXACT_DECODER_CHUNK_SIZE = 1024
DEFAULT_CROSS_BATCH_DECODER_CACHE_BYTES = 8589934592


def load_clt(
    clt_path: str,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
    exact_chunked_decoder: bool = False,
    decoder_chunk_size: int = DEFAULT_EXACT_DECODER_CHUNK_SIZE,
    cross_batch_decoder_cache_bytes: int | None = None,
    checkpoint_asset_scope: CheckpointAssetScope = CheckpointAssetScope.SHARED,
    checkpoint_prefault_budget_bytes: int = 0,
) -> CrossLayerTranscoder:
    """Load a cross-layer transcoder from safetensors files.

    Args:
        clt_path: Path to directory containing W_enc_*.safetensors and W_dec_*.safetensors files
        dtype: Data type for loaded tensors
        lazy_decoder: Whether to load decoder weights on-demand
        lazy_encoder: Whether to load encoder weights on-demand
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        scan: Optional identifier for feature visualization
        device: Device to load tensors to (defaults to auto-detected)

    Returns:
        CrossLayerTranscoder: Loaded transcoder instance
    """
    if device is None:
        device = get_default_device()

    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

    state_dict = _load_state_dict(clt_path, lazy_decoder, lazy_encoder, device, dtype)

    # Infer dimensions from loaded tensors
    n_layers = state_dict["b_dec"].shape[0]
    d_transcoder = state_dict["b_enc"].shape[1]
    d_model = state_dict["b_dec"].shape[1]

    act_fn = "jump_relu" if "activation_function.threshold" in state_dict else "relu"

    manifest_discovery = build_checkpoint_manifest(
        "clt",
        _standard_clt_checkpoint_paths(clt_path),
        scope=checkpoint_asset_scope,
    )
    checkpoint_lifecycle = (
        ProviderCheckpointLifecycle(
            manifest_discovery.manifest,
            prefault_budget_bytes=checkpoint_prefault_budget_bytes,
        )
        if manifest_discovery.manifest is not None
        else None
    )

    # Create instance and load state dict
    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function=act_fn,
            skip_connection=state_dict.get("W_skip") is not None,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=torch.device("meta"),
            dtype=dtype,
            clt_path=clt_path,
            exact_chunked_decoder=exact_chunked_decoder,
            decoder_chunk_size=decoder_chunk_size,
            cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
            checkpoint_lifecycle=checkpoint_lifecycle,
        )

    instance.load_state_dict(state_dict, assign=True)
    instance.checkpoint_manifest_diagnostics = manifest_discovery.diagnostics

    return instance


def load_gemma_scope_2_clt(
    paths: dict[int, str],
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
    decoder_chunk_size: int = DEFAULT_EXACT_DECODER_CHUNK_SIZE,
    cross_batch_decoder_cache_bytes: int | None = None,
    checkpoint_asset_scope: CheckpointAssetScope = CheckpointAssetScope.SHARED,
    checkpoint_prefault_budget_bytes: int = 0,
) -> CrossLayerTranscoder:
    """Load a CrossLayerTranscoder from a GemmaScope2 JumpReLUMultiLayerSAE checkpoint.

    Args:
        path: Path to the checkpoint file
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        scan: Optional identifier for feature visualization
        device: Device to load to
        dtype: Data type to use
        lazy_decoder: Whether to lazily load decoder weights from per-layer safetensors files
        lazy_encoder: Whether to lazily load encoder weights from per-layer safetensors files

    Returns:
        CrossLayerTranscoder: The loaded transcoder
    """
    if device is None:
        device = get_default_device()

    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

    ordered_layers = sorted(paths)
    if ordered_layers != list(range(len(ordered_layers))):
        raise ValueError("GemmaScope-2 CLT paths must be indexed contiguously from 0")

    normalized_path_list: list[str] = []
    for layer_idx in ordered_layers:
        path = paths[layer_idx]
        if not normalized_path_list or normalized_path_list[-1] != path:
            normalized_path_list.append(path)

    paths = {layer_idx: path for layer_idx, path in enumerate(normalized_path_list)}

    with safe_open(paths[0], framework="pt", device="cpu") as f:
        d_model, d_transcoder = f.get_slice("w_enc").get_shape()
        has_skip = "affine_skip_connection" in f.keys()

    n_layers = len(paths)

    state_dict = {
        "b_enc": torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype),
        "b_dec": torch.zeros(n_layers, d_model, device=device, dtype=dtype),
        "activation_function.threshold": torch.zeros(
            n_layers, 1, d_transcoder, device=device, dtype=dtype
        ),
    }

    if not lazy_encoder:
        state_dict["W_enc"] = torch.zeros(
            n_layers, d_transcoder, d_model, device=device, dtype=dtype
        )

    if has_skip:
        state_dict["W_skip"] = torch.zeros(n_layers, d_model, d_model, device=device, dtype=dtype)

    for i in range(n_layers):
        with safe_open(paths[i], framework="pt", device=str(device)) as f:
            state_dict["b_enc"][i] = f.get_tensor("b_enc").to(dtype=dtype)
            state_dict["b_dec"][i] = f.get_tensor("b_dec").to(dtype=dtype)
            state_dict["activation_function.threshold"][i] = (
                f.get_tensor("threshold").to(dtype=dtype).unsqueeze(0)
            )

            if not lazy_encoder:
                state_dict["W_enc"][i] = f.get_tensor("w_enc").transpose(-1, -2).to(dtype=dtype)

            if not lazy_decoder:
                state_dict[f"W_dec.{i}"] = f.get_tensor("w_dec")[:, i:, :].to(dtype=dtype)

            if has_skip:
                state_dict["W_skip"][i] = f.get_tensor("affine_skip_connection").to(dtype=dtype)

    manifest_discovery = build_checkpoint_manifest(
        "clt",
        paths,
        scope=checkpoint_asset_scope,
    )
    checkpoint_lifecycle = (
        ProviderCheckpointLifecycle(
            manifest_discovery.manifest,
            prefault_budget_bytes=checkpoint_prefault_budget_bytes,
        )
        if manifest_discovery.manifest is not None
        else None
    )

    # Create instance
    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function="jump_relu",
            skip_connection=("W_skip" in state_dict),
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=torch.device("meta"),
            dtype=dtype,
            layer_paths=paths if (lazy_encoder or lazy_decoder) else None,
            weight_format="gemmascope2",
            exact_chunked_decoder=True,
            decoder_chunk_size=decoder_chunk_size,
            cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
            checkpoint_lifecycle=checkpoint_lifecycle,
        )

    instance.load_state_dict(state_dict, assign=True)
    instance.checkpoint_manifest_diagnostics = manifest_discovery.diagnostics

    return instance


def _standard_clt_checkpoint_paths(clt_path: str) -> tuple[str, ...]:
    """Return only the exact local safetensors files owned by a standard CLT."""

    encoders = sorted(glob.glob(os.path.join(clt_path, "W_enc_*.safetensors")))
    decoders = sorted(glob.glob(os.path.join(clt_path, "W_dec_*.safetensors")))
    return tuple((*encoders, *decoders))


def _load_state_dict(
    clt_path, lazy_decoder=True, lazy_encoder=False, device=None, dtype=torch.bfloat16
):
    if device is None:
        device = get_default_device()

    enc_files = glob.glob(os.path.join(clt_path, "W_enc_*.safetensors"))
    n_layers = len(enc_files)

    # Get dimensions from first file
    dec_file = "W_enc_0.safetensors"
    with safe_open(os.path.join(clt_path, dec_file), framework="pt", device=str(device)) as f:
        d_transcoder, d_model = f.get_slice("W_enc_0").get_shape()
        has_threshold = "threshold_0" in f.keys()

    # Preallocate tensors
    b_dec = torch.zeros(n_layers, d_model, device=device, dtype=dtype)
    b_enc = torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)

    state_dict = {"b_dec": b_dec, "b_enc": b_enc}

    if has_threshold:
        state_dict["activation_function.threshold"] = torch.zeros(
            n_layers, 1, d_transcoder, device=device, dtype=dtype
        )

    # Only create W_enc if not lazy
    if not lazy_encoder:
        W_enc = torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
        state_dict["W_enc"] = W_enc

    # Load all layers
    for i in range(n_layers):
        enc_file = f"W_enc_{i}.safetensors"
        with safe_open(os.path.join(clt_path, enc_file), framework="pt", device=str(device)) as f:
            b_dec[i] = f.get_tensor(f"b_dec_{i}").to(dtype)
            b_enc[i] = f.get_tensor(f"b_enc_{i}").to(dtype)

            # Only load W_enc if not lazy
            if not lazy_encoder:
                W_enc[i] = f.get_tensor(f"W_enc_{i}").to(dtype)

            if has_threshold:
                threshold = f.get_tensor(f"threshold_{i}").to(dtype)
                state_dict["activation_function.threshold"][i] = threshold.unsqueeze(0)

        # Load W_dec for this layer if not lazy
        if not lazy_decoder:
            dec_file = os.path.join(clt_path, f"W_dec_{i}.safetensors")
            with safe_open(dec_file, framework="pt", device=str(device)) as f:
                state_dict[f"W_dec.{i}"] = f.get_tensor(f"W_dec_{i}").to(dtype)

    return state_dict
