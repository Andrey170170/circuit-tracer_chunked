"""Bounded decoder-page prefetch for exact chunked attribution replay."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any

import torch


@dataclass(frozen=True)
class _LoadedDecoderPage:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event | None


@dataclass(frozen=True)
class _ConsumedDecoderPage:
    tensor: torch.Tensor
    completion_event: torch.cuda.Event | None
    nbytes: int


class DecoderPagePrefetch:
    """Own at most one decoder page loaded ahead of the consumer.

    The extra residency bound is ``depth * max_decoder_page_bytes`` beyond the
    page currently consumed. Depth one is intentionally the only supported
    implementation: it overlaps the next independent safetensors page load with
    contraction of the current page without becoming another decoder cache.
    """

    def __init__(self, *, provider: Any, decoder_cache: Any, depth: int) -> None:
        if depth not in (0, 1):
            raise ValueError(f"decoder page prefetch supports depth 0 or 1, got {depth}")
        self._provider = provider
        self._decoder_cache = decoder_cache
        self._depth = int(depth)
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="decoder-page-prefetch")
            if depth == 1
            else None
        )
        device = (
            torch.device(provider.decoder_device)
            if depth == 1
            else torch.device("cpu")
        )
        self._cuda_stream = (
            torch.cuda.Stream(device=device) if depth == 1 and device.type == "cuda" else None
        )
        self._future: Future[_LoadedDecoderPage] | None = None
        self._key: tuple[int, int] | None = None
        self._scheduled_nbytes = 0
        self._active_tensor: torch.Tensor | None = None
        self._active_nbytes = 0
        self._consumed: _ConsumedDecoderPage | None = None
        self._closed = False
        if self._depth == 1:
            self._record("owner_open")

    def __enter__(self) -> DecoderPagePrefetch:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def get(self, source_layer: int, chunk_id: int) -> torch.Tensor:
        key = (int(source_layer), int(chunk_id))
        if self._depth == 0:
            return self._provider.get_decoder_chunk(
                *key,
                decoder_cache=self._decoder_cache,
            )
        if self._active_tensor is not None:
            raise RuntimeError("finish the active decoder page before requesting the next page")
        self._retire_consumed()
        if self._future is None:
            tensor = self._provider.get_decoder_chunk(
                *key,
                decoder_cache=self._decoder_cache,
            )
            self._accept(tensor, nbytes=int(self._provider.decoder_chunk_nbytes(*key)))
            return tensor
        if key != self._key:
            raise RuntimeError(
                f"decoder prefetch traversal mismatch: expected {self._key}, got {key}"
            )

        future = self._future
        waited = not future.done()
        wait_start = time.perf_counter()
        try:
            loaded = future.result()
            wait_seconds = time.perf_counter() - wait_start
            if loaded.ready_event is not None:
                consumer_stream = torch.cuda.current_stream(device=loaded.tensor.device)
                consumer_stream.wait_event(loaded.ready_event)
                loaded.tensor.record_stream(consumer_stream)
        except BaseException:
            self._future = None
            self._key = None
            self._record("release", nbytes=self._scheduled_nbytes)
            self._scheduled_nbytes = 0
            raise
        self._future = None
        self._key = None
        self._record(
            "consume",
            nbytes=self._scheduled_nbytes,
            host_waited=waited,
            host_wait_seconds=wait_seconds,
        )
        self._accept(loaded.tensor, nbytes=self._scheduled_nbytes)
        self._scheduled_nbytes = 0
        return loaded.tensor

    def finish(self, tensor: torch.Tensor) -> None:
        """Mark contraction of the active page complete on the consumer stream."""
        if self._depth == 0:
            return
        if tensor is not self._active_tensor:
            raise RuntimeError("decoder page finish does not match the active page")
        completion_event = None
        if tensor.device.type == "cuda":
            consumer_stream = torch.cuda.current_stream(device=tensor.device)
            completion_event = torch.cuda.Event()
            completion_event.record(consumer_stream)
        nbytes = self._active_nbytes
        self._consumed = _ConsumedDecoderPage(tensor, completion_event, nbytes)
        self._active_tensor = None
        self._active_nbytes = 0
        self._record("consumer_finish", nbytes=nbytes)

    def schedule(self, source_layer: int, chunk_id: int) -> None:
        if self._depth == 0:
            return
        if self._closed:
            raise RuntimeError("cannot schedule on a closed decoder prefetch owner")
        if self._future is not None:
            raise RuntimeError(f"decoder prefetch already owns in-flight page {self._key}")
        assert self._executor is not None
        key = (int(source_layer), int(chunk_id))
        nbytes = int(self._provider.decoder_chunk_nbytes(*key))
        self._key = key
        self._scheduled_nbytes = nbytes
        self._record("schedule", nbytes=nbytes)
        try:
            self._future = self._executor.submit(self._load, key)
        except BaseException:
            self._key = None
            self._scheduled_nbytes = 0
            self._record("release", nbytes=nbytes)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            if self._active_tensor is not None:
                try:
                    self.finish(self._active_tensor)
                except BaseException as error:
                    first_error = error
                    self._active_tensor = None
                    self._active_nbytes = 0
            try:
                self._retire_consumed()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(
                        f"consumer page retirement also failed: {type(error).__name__}: {error}"
                    )
            if self._future is not None:
                try:
                    loaded = self._future.result()
                    if loaded.ready_event is not None:
                        loaded.ready_event.synchronize()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    else:
                        first_error.add_note(
                            f"prefetched page cleanup also failed: {type(error).__name__}: {error}"
                        )
                finally:
                    self._record("release", nbytes=self._scheduled_nbytes)
                    self._future = None
                    self._key = None
                    self._scheduled_nbytes = 0
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._cuda_stream = None
            if self._depth == 1:
                self._record("owner_close")
        if first_error is not None:
            raise first_error

    def _load(self, key: tuple[int, int]) -> _LoadedDecoderPage:
        device = torch.device(self._provider.decoder_device)
        if device.type != "cuda":
            return _LoadedDecoderPage(
                self._provider.get_decoder_chunk(
                    *key,
                    decoder_cache=self._decoder_cache,
                    request_kind="prefetch",
                ),
                None,
            )

        with torch.cuda.device(device):
            stream = self._cuda_stream
            assert stream is not None
            with torch.cuda.stream(stream):
                tensor = self._provider.get_decoder_chunk(
                    *key,
                    decoder_cache=self._decoder_cache,
                    request_kind="prefetch",
                )
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
        return _LoadedDecoderPage(tensor, ready_event)

    def _record(self, event: str, **attrs: object) -> None:
        recorder = getattr(self._provider, "record_decoder_prefetch_event", None)
        if callable(recorder):
            recorder(event, **attrs)

    def _accept(self, tensor: torch.Tensor, *, nbytes: int) -> None:
        self._active_tensor = tensor
        self._active_nbytes = int(nbytes)
        self._record("handoff", nbytes=nbytes)

    def _retire_consumed(self) -> None:
        consumed = self._consumed
        if consumed is None:
            return
        waited = False
        wait_start = time.perf_counter()
        if consumed.completion_event is not None:
            waited = not consumed.completion_event.query()
            consumed.completion_event.synchronize()
        wait_seconds = time.perf_counter() - wait_start
        self._consumed = None
        self._record(
            "consumer_retire",
            nbytes=consumed.nbytes,
            backpressure_waited=waited,
            backpressure_wait_seconds=wait_seconds,
        )
