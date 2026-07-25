from __future__ import annotations

from contextlib import nullcontext
from threading import Event, Lock

import pytest
import torch

import circuit_tracer.attribution.context_nnsight as context_module
from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.decoder_page_prefetch import (
    DecoderPagePrefetch,
)


class _Provider:
    decoder_device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], str]] = []
        self.loads = 0
        self.cache_hits = 0
        self.in_flight = 0
        self.in_flight_peak = 0
        self.in_flight_bytes = 0
        self.in_flight_bytes_peak = 0
        self.consumes = 0
        self.host_waits = 0
        self.fail_key: tuple[int, int] | None = None
        self.started = Event()
        self.release = Event()
        self.block_key: tuple[int, int] | None = None
        self._lock = Lock()

    def decoder_chunk_nbytes(self, _source_layer: int, _chunk_id: int) -> int:
        return 16

    def get_decoder_chunk(
        self,
        source_layer: int,
        chunk_id: int,
        decoder_cache=None,
        *,
        request_kind: str = "demand",
    ) -> torch.Tensor:
        key = (source_layer, chunk_id)
        with self._lock:
            self.calls.append((key, request_kind))
        cached = decoder_cache.get(key) if decoder_cache is not None else None
        if cached is not None:
            self.cache_hits += 1
            return cached
        if key == self.block_key:
            self.started.set()
            assert self.release.wait(timeout=2)
        if key == self.fail_key:
            raise RuntimeError("page load failed")
        self.loads += 1
        result = torch.full((2, 2), float(source_layer * 10 + chunk_id))
        if decoder_cache is not None:
            decoder_cache[key] = result
        return result

    def record_decoder_prefetch_event(self, event: str, **attrs: object) -> None:
        nbytes = int(attrs.get("nbytes", 0))
        with self._lock:
            if event == "schedule":
                self.in_flight += 1
                self.in_flight_bytes += nbytes
                self.in_flight_peak = max(self.in_flight_peak, self.in_flight)
                self.in_flight_bytes_peak = max(
                    self.in_flight_bytes_peak, self.in_flight_bytes
                )
            elif event == "consume":
                self.consumes += 1
                self.host_waits += int(bool(attrs.get("host_waited", False)))
                self.in_flight -= 1
                self.in_flight_bytes -= nbytes
            elif event == "release":
                self.in_flight -= 1
                self.in_flight_bytes -= nbytes


class _DepthZeroProvider:
    """A legacy exact provider with no prefetch-specific attributes."""

    def get_decoder_chunk(
        self, source_layer: int, chunk_id: int, decoder_cache=None
    ) -> torch.Tensor:
        del decoder_cache
        return torch.tensor([source_layer, chunk_id])


def test_depth_zero_is_provider_agnostic_noop() -> None:
    provider = _DepthZeroProvider()
    with DecoderPagePrefetch(provider=provider, decoder_cache=None, depth=0) as pages:
        assert torch.equal(pages.get(2, 3), torch.tensor([2, 3]))
        pages.schedule(9, 9)


def _bare_context() -> AttributionContext:
    ctx = object.__new__(AttributionContext)
    ctx.decoder_provider = object()
    ctx.decoder_chunk_cache = None
    ctx.decoder_page_prefetch_depth = 1
    ctx._decoder_page_prefetch = None
    return ctx


def test_context_preserves_primary_error_when_unconsumed_prefetch_close_fails(
    monkeypatch,
) -> None:
    class _FailingClose:
        def __init__(self, **_kwargs) -> None:
            pass

        def close(self) -> None:
            raise OSError("unconsumed prefetch load failed")

    def _fail_replay(self, *_args, **_kwargs) -> None:
        raise RuntimeError("contraction failed")

    monkeypatch.setattr(context_module, "DecoderPagePrefetch", _FailingClose)
    monkeypatch.setattr(
        AttributionContext,
        "_compute_chunked_feature_attributions_from_grad_batches_impl",
        _fail_replay,
    )
    ctx = _bare_context()

    with pytest.raises(RuntimeError, match="contraction failed") as exc_info:
        ctx._compute_chunked_feature_attributions_from_grad_batches([])

    assert ctx._decoder_page_prefetch is None
    assert any(
        "unconsumed prefetch load failed" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


def test_context_clears_prefetch_owner_when_close_is_primary_error(monkeypatch) -> None:
    class _FailingClose:
        def __init__(self, **_kwargs) -> None:
            pass

        def close(self) -> None:
            raise OSError("prefetch close failed")

    monkeypatch.setattr(context_module, "DecoderPagePrefetch", _FailingClose)
    monkeypatch.setattr(
        AttributionContext,
        "_compute_chunked_feature_attributions_from_grad_batches_impl",
        lambda *_args, **_kwargs: None,
    )
    ctx = _bare_context()

    with pytest.raises(OSError, match="prefetch close failed"):
        ctx._compute_chunked_feature_attributions_from_grad_batches([])

    assert ctx._decoder_page_prefetch is None


def test_depth_one_prefetch_preserves_page_order_and_values() -> None:
    provider = _Provider()
    with DecoderPagePrefetch(provider=provider, decoder_cache=None, depth=1) as pages:
        values = []
        for chunk_id in range(3):
            page = pages.get(0, chunk_id)
            if chunk_id < 2:
                pages.schedule(0, chunk_id + 1)
            values.append(float(page.sum()))

    assert values == [0.0, 4.0, 8.0]
    assert provider.calls == [
        ((0, 0), "demand"),
        ((0, 1), "prefetch"),
        ((0, 2), "prefetch"),
    ]
    assert provider.in_flight_peak == 1
    assert provider.in_flight_bytes_peak == 16
    assert provider.in_flight == 0
    assert provider.consumes == 2


def test_prefetch_runs_while_current_page_is_available() -> None:
    provider = _Provider()
    provider.block_key = (0, 1)
    with DecoderPagePrefetch(provider=provider, decoder_cache=None, depth=1) as pages:
        current = pages.get(0, 0)
        pages.schedule(0, 1)
        assert provider.started.wait(timeout=2)
        assert float(current.sum()) == 0.0
        provider.release.set()
        assert float(pages.get(0, 1).sum()) == 4.0


def test_cuda_page_records_consumer_stream_after_wait(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeTensor:
        device = torch.device("cuda")

        def record_stream(self, stream) -> None:
            assert stream == "consumer-stream"
            calls.append("record_stream")

    class _CudaProvider:
        decoder_device = torch.device("cuda")

        def decoder_chunk_nbytes(self, _source_layer: int, _chunk_id: int) -> int:
            return 16

        def get_decoder_chunk(self, *_args, **_kwargs):
            return _FakeTensor()

    class _FakeEvent:
        def record(self, stream) -> None:
            assert stream == "producer-stream"
            calls.append("event_record")

        def synchronize(self) -> None:
            calls.append("event_synchronize")

    class _ConsumerStream:
        def wait_event(self, event) -> None:
            assert isinstance(event, _FakeEvent)
            calls.append("wait_event")

        def __eq__(self, other: object) -> bool:
            return other == "consumer-stream"

    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: "producer-stream")
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda **_kwargs: _ConsumerStream()
    )

    with DecoderPagePrefetch(
        provider=_CudaProvider(), decoder_cache=None, depth=1
    ) as pages:
        pages.schedule(0, 1)
        result = pages.get(0, 1)

    assert isinstance(result, _FakeTensor)
    assert calls == ["event_record", "wait_event", "record_stream"]


def test_cached_prefetch_does_not_duplicate_load() -> None:
    provider = _Provider()
    cached = torch.ones((2, 2))
    cache = {(0, 1): cached}
    with DecoderPagePrefetch(provider=provider, decoder_cache=cache, depth=1) as pages:
        pages.get(0, 0)
        pages.schedule(0, 1)
        result = pages.get(0, 1)

    assert result is cached
    assert provider.loads == 1
    assert provider.cache_hits == 1
    assert provider.calls[-1] == ((0, 1), "prefetch")


def test_prefetch_exception_releases_owned_page_slot() -> None:
    provider = _Provider()
    provider.fail_key = (0, 1)
    with pytest.raises(RuntimeError, match="page load failed"):
        with DecoderPagePrefetch(provider=provider, decoder_cache=None, depth=1) as pages:
            pages.get(0, 0)
            pages.schedule(0, 1)
            pages.get(0, 1)

    assert provider.in_flight == 0
    assert provider.in_flight_bytes == 0

    # The failed future is consumed by get(); close is idempotent and does not
    # re-raise or mask the original load failure.
    pages.close()
