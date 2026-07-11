import inspect

import torch

from circuit_tracer.transcoder.cross_layer_transcoder import (
    DecoderChunkCache,
    _load_state_dict,
    load_clt,
    load_gemma_scope_2_clt,
)
from circuit_tracer.transcoder.decoder_cache import DecoderChunkCache as ExtractedDecoderChunkCache
from circuit_tracer.transcoder.loaders import (
    _load_state_dict as extracted_load_state_dict,
    load_clt as extracted_load_clt,
    load_gemma_scope_2_clt as extracted_load_gemma_scope_2_clt,
)


def test_extracted_decoder_cache_is_reexported_without_behavior_change():
    assert DecoderChunkCache is ExtractedDecoderChunkCache
    cache = DecoderChunkCache(8, fingerprint=("checkpoint", 1))

    assert cache.put((0, 0), torch.zeros(2)) == []
    assert cache.get((0, 0)) is not None
    evicted = cache.put((0, 1), torch.zeros(2))

    assert evicted == [((0, 0), 8)]
    assert cache.bytes_resident == 8
    assert cache.fingerprint == ("checkpoint", 1)


def test_extracted_loaders_are_reexported_with_exact_signatures():
    pairs = (
        (load_clt, extracted_load_clt),
        (load_gemma_scope_2_clt, extracted_load_gemma_scope_2_clt),
        (_load_state_dict, extracted_load_state_dict),
    )
    for reexported, extracted in pairs:
        assert reexported is extracted
        assert inspect.signature(reexported) == inspect.signature(extracted)
