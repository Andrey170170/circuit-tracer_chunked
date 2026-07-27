from __future__ import annotations

import gc
import weakref
import pytest
import torch

from circuit_tracer.attribution.context_nnsight import AttributionContext
import circuit_tracer.attribution.nnsight.active_decoder_rows as active_decoder_rows_module
from circuit_tracer.attribution.nnsight.context_state import (
    AttributionTensorState,
    ContextExecutionPolicy,
    ContextNumericPolicy,
    DecoderRuntime,
)
from circuit_tracer.tracing.plan import FrontierExpansionPlan
from circuit_tracer.transcoder.attribution_result import DecoderRowSeed, DecoderRowSeedLayer
from circuit_tracer.transcoder.provider import TranscoderCapabilities, provider_fingerprint


class SyntheticPltProvider:
    def __init__(
        self,
        blocks: dict[int, torch.Tensor],
        *,
        chunk_size: int = 2,
        supports_residency: bool = True,
        architecture: str = "plt",
        source_id: str = "synthetic-a",
    ) -> None:
        self.blocks = blocks
        self.scan = source_id
        self.decoder_chunk_size = chunk_size
        self.n_layers = max(blocks) + 1
        self.d_transcoder = max(int(block.shape[0]) for block in blocks.values())
        self.d_model = int(next(iter(blocks.values())).shape[-1])
        self.dtype = next(iter(blocks.values())).dtype
        self.load_calls: list[tuple[int, int]] = []
        self.clear_calls = 0
        self._stats = {
            "decoder_load_count": 0,
            "decoder_load_bytes": 0,
        }
        self.capabilities = TranscoderCapabilities(
            architecture=architecture,  # type: ignore[arg-type]
            checkpoint_format="synthetic",
            supports_exact_chunked_provider=True,
            supports_compact_row_store=True,
            supports_decoder_chunk_cache=True,
            supports_encoder_row_materialization=True,
            supports_active_decoder_row_residency=supports_residency,
            decoder_output_topology="same_layer",
            default_decoder_chunk_size=chunk_size,
        )

    def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
        del max_bytes, fingerprint
        return {}

    def clear_decoder_block_cache(self, cache) -> None:
        self.clear_calls += 1
        if cache is not None:
            cache.clear()

    def decoder_output_layers_for_source(self, source_layer: int, active_output_layers=None):
        if active_output_layers is not None and source_layer not in active_output_layers:
            return []
        return [source_layer]

    def decoder_output_slot(self, source_layer: int, output_layer: int) -> int:
        if output_layer != source_layer:
            raise ValueError("synthetic PLT provider only supports same-layer output")
        return 0

    def get_decoder_chunk(self, layer_id: int, chunk_id: int, decoder_cache=None, **kwargs):
        del kwargs
        key = (int(layer_id), int(chunk_id))
        if decoder_cache is not None and key in decoder_cache:
            return decoder_cache[key]
        start = chunk_id * self.decoder_chunk_size
        stop = min(start + self.decoder_chunk_size, self.blocks[layer_id].shape[0])
        page = self.blocks[layer_id][start:stop]
        self.load_calls.append(key)
        self._stats["decoder_load_count"] += 1
        self._stats["decoder_load_bytes"] += int(page.numel() * page.element_size())
        if decoder_cache is not None:
            decoder_cache[key] = page
        return page

    def materialize_encoder_rows(self, source_layers, feature_ids):
        return torch.stack((source_layers, feature_ids), dim=-1).to(torch.float32)

    def get_diagnostic_snapshot(self):
        return dict(self._stats)

    def record_decoder_prefetch_event(self, event: str, **attrs) -> None:
        del event, attrs


def _activation_state():
    source_layers = torch.tensor([0, 0, 0, 1, 1, 1])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    feature_ids = torch.tensor([5, 1, 3, 4, 0, 2])
    values = torch.tensor([0.5, -1.25, 2.0, 0.75, -0.5, 1.5])
    activation = torch.sparse_coo_tensor(
        torch.stack((source_layers, positions, feature_ids)),
        values,
        size=(2, 3, 6),
        check_invariants=True,
    ).coalesce()
    return activation


def _provider(
    *,
    supports_residency: bool = True,
    architecture: str = "plt",
    source_id: str = "synthetic-a",
):
    blocks = {
        0: torch.arange(6 * 1 * 4, dtype=torch.float32).reshape(6, 1, 4).div(7),
        1: torch.arange(6 * 1 * 4, dtype=torch.float32).reshape(6, 1, 4).add(50).div(11),
    }
    return SyntheticPltProvider(
        blocks,
        supports_residency=supports_residency,
        architecture=architecture,
        source_id=source_id,
    )


def _context(
    provider: SyntheticPltProvider,
    *,
    row_subchunk_size: int = 2,
    decoder_row_seed: DecoderRowSeed | None = None,
    decoder_row_seed_refusal_reason: str | None = None,
    decoder_row_seed_estimated_bytes: int | None = None,
) -> AttributionContext:
    activation = _activation_state()
    indices = activation.indices()
    return AttributionContext(
        tensor_state=AttributionTensorState(
            activation_matrix=activation,
            error_vectors=torch.zeros(2, 3, 4),
            token_vectors=torch.zeros(3, 4),
            decoder_vectors=torch.empty((0, 4)),
            encoder_vectors=torch.zeros((activation._nnz(), 4)),
            encoder_to_decoder_map=torch.empty(0, dtype=torch.long),
            decoder_locations=torch.empty((2, 0), dtype=torch.long),
            logits=torch.zeros(1),
        ),
        execution_policy=ContextExecutionPolicy.resolve(
            chunked_decoder_state={
                "source_layers": indices[0],
                "positions": indices[1],
                "feature_ids": indices[2],
                "activation_values": activation.values(),
            },
            encoder_vectors=torch.zeros((activation._nnz(), 4)),
            error_vectors=torch.zeros(2, 3, 4),
            exact_encoder_residency="lazy",
            stage_encoder_vectors_on_cpu=False,
            stage_error_vectors_on_cpu=False,
            error_vector_prefetch_lookahead=1,
            chunked_feature_replay_window=1,
            row_subchunk_size=row_subchunk_size,
        ),
        decoder_runtime=DecoderRuntime.resolve(
            provider=provider,
            chunked_state={
                "source_layers": indices[0],
                "positions": indices[1],
                "feature_ids": indices[2],
                "activation_values": activation.values(),
            },
            decoder_row_seed=decoder_row_seed,
            decoder_row_seed_refusal_reason=decoder_row_seed_refusal_reason,
            decoder_row_seed_estimated_bytes=decoder_row_seed_estimated_bytes,
        ),
        numeric_policy=ContextNumericPolicy(),
    )


def _decoder_row_seed(provider: SyntheticPltProvider) -> DecoderRowSeed:
    activation = _activation_state()
    source_layers, _, feature_ids = activation.indices()
    layers = []
    for layer in range(2):
        ids = torch.unique(feature_ids[source_layers == layer], sorted=True)
        layers.append(
            DecoderRowSeedLayer(
                source_layer=layer,
                output_layers=(layer,),
                feature_ids=ids.cpu(),
                rows=provider.blocks[layer][ids].cpu(),
            )
        )
    return DecoderRowSeed(
        layers=tuple(layers),
        source_fingerprint=provider_fingerprint(provider),
        occurrence_estimated_bytes=6 * 4 * torch.float32.itemsize,
        capture_seconds=0.25,
        shared_traversal_bytes=1234,
        shared_decoder_load_count=6,
        shared_decoder_load_bytes=1234,
    )


def _grad_batches(row_count: int):
    first = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).div(5)
    second = first.flip((0, 1)).mul(-0.75)
    return [
        ((first, first.flip(-1)), torch.zeros(row_count, 2), 0),
        ((second, second.flip(-1)), torch.zeros(row_count, 2), 1),
    ]


def test_plt_active_row_build_matches_selected_provider_rows() -> None:
    provider = _provider()
    ctx = _context(provider)

    assert ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=True,
        max_bytes=10_000,
    )
    owner = ctx._active_decoder_rows
    assert owner is not None
    feature_ids = ctx.chunked_decoder_state["feature_ids"]
    for layer in range(2):
        block = owner.layers[layer]
        assert block is not None
        expected = provider.blocks[layer][
            feature_ids[block.global_row_start : block.global_row_end]
        ]
        assert torch.equal(block.rows, expected)
    assert owner.active_row_bytes == 6 * 1 * 4 * torch.float32.itemsize
    assert owner.build_traversal_bytes == sum(
        provider.blocks[layer].numel() * provider.blocks[layer].element_size()
        for layer in provider.blocks
    )


@pytest.mark.parametrize("produced_range", [None, (1, 5)])
def test_resident_rows_match_page_scan_for_multi_chunk_nonmonotonic_rows(
    produced_range: tuple[int, int] | None,
) -> None:
    page_provider = _provider()
    resident_provider = _provider()
    page_ctx = _context(page_provider)
    resident_ctx = _context(resident_provider)
    if produced_range is not None:
        page_ctx._produced_feature_range = produced_range
        resident_ctx._produced_feature_range = produced_range
    row_count = 6 if produced_range is None else produced_range[1] - produced_range[0]
    page_batches = _grad_batches(row_count)
    resident_batches = _grad_batches(row_count)

    page_ctx._compute_chunked_feature_attributions_from_grad_batches(page_batches)
    assert resident_ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=True,
        max_bytes=10_000,
    )
    build_load_count = len(resident_provider.load_calls)
    resident_ctx._compute_chunked_feature_attributions_from_grad_batches(resident_batches)

    for (_, page_rows, _), (_, resident_rows, _) in zip(
        page_batches, resident_batches, strict=True
    ):
        assert torch.equal(resident_rows, page_rows)
    assert len(resident_provider.load_calls) == build_load_count
    assert page_provider.load_calls


def test_active_row_budget_refusal_keeps_page_scan_fallback() -> None:
    provider = _provider()
    baseline_provider = _provider()
    ctx = _context(provider)
    baseline_ctx = _context(baseline_provider)
    assert not ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=True,
        max_bytes=1,
    )
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_residency_requested"] is True
    assert diagnostics["decoder_active_row_residency_effective"] is False
    assert diagnostics["decoder_active_row_fallback_reason"] == "estimated_bytes_exceed_max"
    assert provider.load_calls == []

    batches = _grad_batches(6)
    baseline_batches = _grad_batches(6)
    ctx._compute_chunked_feature_attributions_from_grad_batches(batches)
    baseline_ctx._compute_chunked_feature_attributions_from_grad_batches(baseline_batches)
    for (_, refused_rows, _), (_, baseline_rows, _) in zip(batches, baseline_batches, strict=True):
        assert torch.equal(refused_rows, baseline_rows)
    assert provider.load_calls


def test_active_row_builder_failure_releases_ephemeral_pages() -> None:
    class FailingProvider(SyntheticPltProvider):
        def __init__(self) -> None:
            baseline = _provider()
            super().__init__(baseline.blocks)
            self.ephemeral_page_refs: list[weakref.ReferenceType[torch.Tensor]] = []

        def get_decoder_chunk(self, layer_id, chunk_id, decoder_cache=None, **kwargs):
            if self.ephemeral_page_refs:
                raise RuntimeError("synthetic decoder page failure")
            page = (
                super()
                .get_decoder_chunk(layer_id, chunk_id, decoder_cache=decoder_cache, **kwargs)
                .clone()
            )
            self.ephemeral_page_refs.append(weakref.ref(page))
            return page

    provider = FailingProvider()
    ctx = _context(provider)
    with pytest.raises(RuntimeError, match="synthetic decoder page failure"):
        ctx.prepare_active_decoder_rows(
            requested=True,
            enabled=True,
            max_bytes=10_000,
        )
    gc.collect()

    assert ctx._active_decoder_rows is None
    assert provider.ephemeral_page_refs
    assert all(page_ref() is None for page_ref in provider.ephemeral_page_refs)


def test_unsupported_clt_explicitly_falls_back_without_building() -> None:
    provider = _provider(supports_residency=False, architecture="clt")
    ctx = _context(provider)
    assert not ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=False,
        max_bytes=0,
        fallback_reason="provider_capability_unavailable",
    )
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_residency_requested"] is True
    assert diagnostics["decoder_active_row_residency_effective"] is False
    assert diagnostics["decoder_active_row_fallback_reason"] == ("provider_capability_unavailable")
    assert provider.load_calls == []


def test_active_row_cleanup_and_state_replacement_release_owner() -> None:
    provider = _provider()
    ctx = _context(provider)
    assert ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=True,
        max_bytes=10_000,
    )
    ctx.replace_phase0_activation_state(ctx.activation_matrix.clone())
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_owner_count"] == 0
    assert diagnostics["decoder_active_row_bytes"] == 0
    assert diagnostics["decoder_active_row_fallback_reason"] == "phase0_state_replacement"

    assert ctx.prepare_active_decoder_rows(
        requested=True,
        enabled=True,
        max_bytes=10_000,
    )
    ctx.cleanup()
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_owner_count"] == 0
    assert diagnostics["decoder_active_row_bytes"] == 0
    assert diagnostics["decoder_active_row_fallback_reason"] == "cleanup"
    assert provider.clear_calls >= 1


def test_state_signature_mismatch_releases_and_uses_page_fallback() -> None:
    provider = _provider()
    ctx = _context(provider)
    assert ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    build_load_count = len(provider.load_calls)
    ctx.chunked_decoder_state["feature_ids"].add_(0)
    batches = _grad_batches(6)
    ctx._compute_chunked_feature_attributions_from_grad_batches(batches)
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_owner_count"] == 0
    assert diagnostics["decoder_active_row_fallback_reason"] == "state_signature_mismatch"
    assert len(provider.load_calls) > build_load_count


def test_frontier_active_row_byte_cap_validation() -> None:
    assert FrontierExpansionPlan().decoder_active_row_residency is False
    with pytest.raises(ValueError, match="decoder_active_row_max_bytes"):
        FrontierExpansionPlan(decoder_active_row_max_bytes=-1)


def test_phase0_seed_adoption_materializes_without_decoder_page_loads() -> None:
    provider = _provider()
    ctx = _context(provider, decoder_row_seed=_decoder_row_seed(provider))

    assert ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    assert provider.load_calls == []
    owner = ctx._active_decoder_rows
    assert owner is not None
    feature_ids = ctx.chunked_decoder_state["feature_ids"]
    for layer in range(2):
        block = owner.layers[layer]
        assert block is not None
        expected = provider.blocks[layer][
            feature_ids[block.global_row_start : block.global_row_end]
        ]
        assert torch.equal(block.rows, expected)
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_build_source"] == "phase0_fused_seed"
    assert diagnostics["decoder_active_row_build_decoder_load_count"] == 0
    assert diagnostics["decoder_active_row_seed_materialization_h2d_bytes"] == 96
    assert diagnostics["decoder_active_row_build_count"] == 1


def test_phase0_seed_covers_diagnostic_subset_and_missing_key_falls_back() -> None:
    subset_provider = _provider()
    subset_ctx = _context(subset_provider, decoder_row_seed=_decoder_row_seed(subset_provider))
    subset_ctx.apply_diagnostic_feature_cap(2)
    assert subset_ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    assert subset_provider.load_calls == []

    fallback_provider = _provider()
    fallback_ctx = _context(
        fallback_provider, decoder_row_seed=_decoder_row_seed(fallback_provider)
    )
    fallback_ctx.chunked_decoder_state["feature_ids"][0] = 2
    assert fallback_ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    diagnostics = fallback_ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_seed_missing_keys"] == 1
    assert diagnostics["decoder_active_row_seed_fallback_reason"] == "seed_missing_keys"
    assert diagnostics["decoder_active_row_build_source"] == "page_scan_after_seed_miss"
    assert diagnostics["decoder_active_row_seed_shared_traversal_bytes"] == 1234
    assert diagnostics["decoder_active_row_seed_shared_decoder_load_count"] == 6
    assert fallback_provider.load_calls


def test_disabled_phase0_seed_is_released_without_materialization() -> None:
    provider = _provider()
    ctx = _context(provider, decoder_row_seed=_decoder_row_seed(provider))
    assert not ctx.prepare_active_decoder_rows(
        requested=True, enabled=False, max_bytes=10_000, fallback_reason="disabled_after_replan"
    )
    assert ctx._decoder_row_seed is None
    assert provider.load_calls == []
    assert ctx.get_diagnostic_snapshot()["decoder_active_row_seed_available"] is False


def test_phase0_seed_source_mismatch_scans_current_provider_and_preserves_evidence() -> None:
    seed_provider = _provider(source_id="seed-source")
    current_provider = _provider(source_id="current-source")
    current_provider.blocks = {
        layer: rows.add(1000) for layer, rows in current_provider.blocks.items()
    }
    ctx = _context(current_provider, decoder_row_seed=_decoder_row_seed(seed_provider))

    assert ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_seed_source_mismatch"] is True
    assert diagnostics["decoder_active_row_seed_fallback_reason"] == "seed_source_mismatch"
    assert diagnostics["decoder_active_row_build_source"] == (
        "page_scan_after_seed_source_mismatch"
    )
    assert diagnostics["decoder_active_row_seed_shared_traversal_bytes"] == 1234
    assert diagnostics["decoder_active_row_seed_shared_decoder_load_count"] == 6
    assert diagnostics["decoder_active_row_seed_materialization_h2d_bytes"] == 0
    assert current_provider.load_calls
    owner = ctx._active_decoder_rows
    assert owner is not None
    for layer in range(2):
        block = owner.layers[layer]
        assert block is not None
        feature_ids = ctx.chunked_decoder_state["feature_ids"][
            block.global_row_start : block.global_row_end
        ]
        assert torch.equal(block.rows, current_provider.blocks[layer][feature_ids])


def test_phase0_seed_materialization_failure_releases_prior_layer_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SecondLayerFailingProvider(SyntheticPltProvider):
        layer_one_queries = 0

        def decoder_output_layers_for_source(self, source_layer, active_output_layers=None):
            if source_layer == 1:
                self.layer_one_queries += 1
                if self.layer_one_queries > 1:
                    raise RuntimeError("synthetic second-layer materialization failure")
            return super().decoder_output_layers_for_source(source_layer, active_output_layers)

    baseline = _provider()
    provider = SecondLayerFailingProvider(baseline.blocks)
    seed = _decoder_row_seed(provider)
    row_refs = []
    original_post_init = active_decoder_rows_module.ActiveDecoderLayerRows.__post_init__

    def observe_rows(self):
        original_post_init(self)
        row_refs.append(weakref.ref(self.rows))

    monkeypatch.setattr(
        active_decoder_rows_module.ActiveDecoderLayerRows, "__post_init__", observe_rows
    )
    ctx = _context(provider, decoder_row_seed=seed)
    with pytest.raises(RuntimeError, match="second-layer materialization failure"):
        ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=10_000)
    gc.collect()
    assert ctx._active_decoder_rows is None
    assert row_refs and all(row_ref() is None for row_ref in row_refs)


def test_phase0_seed_over_cap_then_subset_uses_explicit_page_scan_refusal() -> None:
    provider = _provider()
    ctx = _context(
        provider,
        decoder_row_seed=None,
        decoder_row_seed_refusal_reason="phase0_occurrence_bytes_exceed_max",
        decoder_row_seed_estimated_bytes=96,
    )
    ctx.apply_diagnostic_feature_cap(2)

    assert ctx.prepare_active_decoder_rows(requested=True, enabled=True, max_bytes=32)
    diagnostics = ctx.get_diagnostic_snapshot()
    assert diagnostics["decoder_active_row_build_source"] == "page_scan"
    assert diagnostics["decoder_active_row_seed_capture_refusal_reason"] == (
        "phase0_occurrence_bytes_exceed_max"
    )
    assert diagnostics["decoder_active_row_seed_phase0_estimated_bytes"] == 96
    assert diagnostics["decoder_active_row_seed_source_mismatch"] is False
