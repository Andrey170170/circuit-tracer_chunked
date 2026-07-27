from __future__ import annotations

import time

import torch

import circuit_tracer.replacement_model.attribution_setup as attribution_setup_module
from circuit_tracer.replacement_model.attribution_setup import (
    AttributionSetupInput,
    AttributionSetupOperation,
    AttributionSetupOptions,
    Phase0ActivationCapture,
)
from circuit_tracer.transcoder.attribution_result import AttributionComponents
from circuit_tracer.transcoder.provider import TranscoderCapabilities


class _CltLikeProvider:
    capabilities = TranscoderCapabilities(
        architecture="clt",
        checkpoint_format="synthetic-clt",
    )

    def __init__(self) -> None:
        self.component_kwargs: dict[str, object] | None = None

    def compute_attribution_components(self, mlp_inputs, zero_positions, **kwargs):
        del zero_positions
        self.component_kwargs = dict(kwargs)
        return AttributionComponents(
            activation_matrix=torch.sparse_coo_tensor(
                torch.empty((3, 0), dtype=torch.long),
                torch.empty((0,), dtype=mlp_inputs.dtype),
                size=(1, int(mlp_inputs.shape[1]), 1),
            ),
            reconstruction=torch.zeros_like(mlp_inputs),
            encoder_vectors=torch.empty((0, mlp_inputs.shape[-1]), dtype=mlp_inputs.dtype),
            decoder_vectors=torch.empty((0, mlp_inputs.shape[-1]), dtype=mlp_inputs.dtype),
            encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
            decoder_locations=torch.empty((2, 0), dtype=torch.long),
        )


class _FakeContext:
    def __init__(self, **kwargs) -> None:
        self.encoder_vecs = kwargs["tensor_state"].encoder_vectors
        self.logit_retention = "last_token"
        self.exact_encoder_staging_destination = None
        self.exact_encoder_pinned_requested = False
        self.exact_encoder_pinned_effective = False
        self.exact_encoder_pinning_success = None
        self.exact_encoder_pinning_failure_reason = None
        self.setup_diagnostic_stats: dict[str, object] = {}
        self.sparsification_stats: dict[str, object] | None = None


class _Model:
    def __init__(self, transcoders: _CltLikeProvider) -> None:
        self.transcoders = transcoders
        self.zero_positions = slice(0, 1)
        self.embed_weight = torch.zeros((8, 3))
        self.device = "cpu"


def test_clt_setup_does_not_receive_active_decoder_row_kwargs(monkeypatch) -> None:
    provider = _CltLikeProvider()
    model = _Model(provider)
    monkeypatch.setattr(attribution_setup_module, "AttributionContext", _FakeContext)
    capture = Phase0ActivationCapture(
        mlp_inputs=torch.zeros((1, 2, 3)),
        mlp_outputs=torch.zeros((1, 2, 3)),
        logits=torch.zeros((2, 8)),
        elapsed_seconds=0.0,
    )
    options = AttributionSetupOptions(
        sparsification=None,
        retain_full_logits=False,
        chunked_feature_replay_window=4,
        error_vector_prefetch_lookahead=2,
        stage_encoder_vectors_on_cpu=None,
        stage_error_vectors_on_cpu=None,
        row_subchunk_size=None,
        exact_encoder_residency="lazy",
        internal_precision_requested=None,
        resolved_dtype_map=None,
        decoder_chunk_cache=None,
        decoder_cache_fingerprint=None,
        decoder_active_row_residency=True,
        decoder_active_row_max_bytes=1024,
    )

    AttributionSetupOperation(
        model=model,
        setup_input=AttributionSetupInput.resolve(torch.tensor([1, 2]), None),
        capture=capture,
        options=options,
        setup_started_at=time.perf_counter(),
        phase0_input_fingerprints=None,
        trace_observer=None,
    ).run()

    assert provider.component_kwargs == {"sparsification": None}


def test_setup_binds_decoder_row_seed_to_caller_provider(monkeypatch) -> None:
    from circuit_tracer.transcoder.attribution_result import DecoderRowSeed
    from circuit_tracer.transcoder.provider import provider_fingerprint

    provider = _CltLikeProvider()
    provider.capabilities = TranscoderCapabilities(
        architecture="plt",
        checkpoint_format="synthetic-plt",
        supports_active_decoder_row_residency=True,
    )
    provider.scan = "canonical-provider"
    seed = DecoderRowSeed(
        layers=(),
        source_fingerprint={"unstable": "method-trace-boundary"},
        occurrence_estimated_bytes=0,
        capture_seconds=0.0,
        shared_traversal_bytes=0,
        shared_decoder_load_count=0,
        shared_decoder_load_bytes=0,
    )

    def _components(mlp_inputs, zero_positions, **kwargs):
        del zero_positions, kwargs
        return AttributionComponents(
            activation_matrix=torch.sparse_coo_tensor(
                torch.empty((3, 0), dtype=torch.long),
                torch.empty((0,), dtype=mlp_inputs.dtype),
                size=(1, int(mlp_inputs.shape[1]), 1),
            ),
            reconstruction=torch.zeros_like(mlp_inputs),
            encoder_vectors=torch.empty((0, mlp_inputs.shape[-1]), dtype=mlp_inputs.dtype),
            decoder_vectors=torch.empty((0, mlp_inputs.shape[-1]), dtype=mlp_inputs.dtype),
            encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
            decoder_locations=torch.empty((2, 0), dtype=torch.long),
            chunked_decoder_state={
                "source_layers": torch.empty((0,), dtype=torch.long),
                "positions": torch.empty((0,), dtype=torch.long),
                "feature_ids": torch.empty((0,), dtype=torch.long),
                "activation_values": torch.empty((0,), dtype=mlp_inputs.dtype),
            },
            decoder_row_seed=seed,
        )

    provider.compute_attribution_components = _components
    capture = Phase0ActivationCapture(
        mlp_inputs=torch.zeros((1, 2, 3)),
        mlp_outputs=torch.zeros((1, 2, 3)),
        logits=torch.zeros((2, 8)),
        elapsed_seconds=0.0,
    )
    options = AttributionSetupOptions(
        sparsification=None,
        retain_full_logits=False,
        chunked_feature_replay_window=4,
        error_vector_prefetch_lookahead=2,
        stage_encoder_vectors_on_cpu=None,
        stage_error_vectors_on_cpu=None,
        row_subchunk_size=None,
        exact_encoder_residency="lazy",
        internal_precision_requested=None,
        resolved_dtype_map=None,
        decoder_chunk_cache=None,
        decoder_cache_fingerprint=None,
        decoder_active_row_residency=True,
        decoder_active_row_max_bytes=1024,
    )

    class _CapturingContext(_FakeContext):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.decoder_row_seed = kwargs["decoder_runtime"].decoder_row_seed

    monkeypatch.setattr(attribution_setup_module, "AttributionContext", _CapturingContext)
    context = AttributionSetupOperation(
        model=_Model(provider),
        setup_input=AttributionSetupInput.resolve(torch.tensor([1, 2]), None),
        capture=capture,
        options=options,
        setup_started_at=time.perf_counter(),
        phase0_input_fingerprints=None,
        trace_observer=None,
    ).run()

    assert context.decoder_row_seed.source_fingerprint == provider_fingerprint(provider)
    assert seed.source_fingerprint != context.decoder_row_seed.source_fingerprint
