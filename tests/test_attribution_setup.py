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


def test_active_cpu_setup_materializes_encoder_rows_directly_on_cpu(monkeypatch) -> None:
    class _ExactProvider(_CltLikeProvider):
        capabilities = TranscoderCapabilities(
            architecture="clt",
            checkpoint_format="synthetic-clt",
            supports_exact_chunked_provider=True,
            supports_exact_encoder_residency=True,
        )

        def __init__(self) -> None:
            super().__init__()
            self.materialize_device: torch.device | None = None

        def compute_attribution_components(self, mlp_inputs, zero_positions, **kwargs):
            del zero_positions
            self.component_kwargs = dict(kwargs)
            indices = torch.tensor([[0, 0], [0, 1], [2, 1]])
            activation = torch.sparse_coo_tensor(
                indices,
                torch.tensor([1.0, 2.0]),
                size=(1, 2, 4),
            ).coalesce()
            return AttributionComponents(
                activation_matrix=activation,
                reconstruction=torch.zeros_like(mlp_inputs),
                encoder_vectors=torch.empty((0, mlp_inputs.shape[-1])),
                decoder_vectors=torch.empty((0, mlp_inputs.shape[-1])),
                encoder_to_decoder_map=torch.empty((0,), dtype=torch.long),
                decoder_locations=torch.empty((2, 0), dtype=torch.long),
                chunked_decoder_state={
                    "source_layers": activation.indices()[0],
                    "positions": activation.indices()[1],
                    "feature_ids": activation.indices()[2],
                    "activation_values": activation.values(),
                },
            )

        def materialize_encoder_rows(self, source_layers, feature_ids, *, device=None):
            self.materialize_device = torch.device(device)
            rows = torch.stack(
                [source_layers.to(torch.float32), feature_ids.to(torch.float32)], dim=1
            )
            return rows.to(device=self.materialize_device)

        def get_decoder_chunk(self, layer_id, chunk_id, **kwargs):
            del layer_id, chunk_id, kwargs
            return torch.empty((0, 1, 2))

        def decoder_output_layers_for_source(self, source_layer, active_output_layers=None):
            del active_output_layers
            return [source_layer]

        def decoder_output_slot(self, source_layer, output_layer):
            if source_layer != output_layer:
                raise ValueError("synthetic provider is same-layer")
            return 0

    provider = _ExactProvider()
    monkeypatch.setattr(attribution_setup_module, "AttributionContext", _FakeContext)
    capture = Phase0ActivationCapture(
        mlp_inputs=torch.zeros((1, 2, 2)),
        mlp_outputs=torch.zeros((1, 2, 2)),
        logits=torch.zeros((1, 2, 8)),
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
        exact_encoder_residency="active_cpu",
        internal_precision_requested=None,
        resolved_dtype_map=None,
        decoder_chunk_cache=None,
        decoder_cache_fingerprint=None,
        decoder_active_row_residency=False,
        decoder_active_row_max_bytes=0,
    )

    context = AttributionSetupOperation(
        model=_Model(provider),
        setup_input=AttributionSetupInput.resolve(torch.tensor([1, 2]), None),
        capture=capture,
        options=options,
        setup_started_at=time.perf_counter(),
        phase0_input_fingerprints=None,
        trace_observer=None,
    ).run()

    assert provider.component_kwargs == {
        "sparsification": None,
        "materialize_encoder_vecs": False,
    }
    assert provider.materialize_device == torch.device("cpu")
    assert context.encoder_vecs.device.type == "cpu"
    assert torch.equal(context.encoder_vecs, torch.tensor([[0.0, 2.0], [0.0, 1.0]]))


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
