from collections import OrderedDict

import torch


class DecoderChunkCache:
    def __init__(self, max_bytes: int, *, fingerprint: object | None = None) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.fingerprint = fingerprint
        self.bytes_resident = 0
        self._entries: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def get(self, key: tuple[int, int]) -> torch.Tensor | None:
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[int, int], value: torch.Tensor) -> list[tuple[tuple[int, int], int]]:
        value_nbytes = self._tensor_nbytes(value)
        evicted: list[tuple[tuple[int, int], int]] = []
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.bytes_resident -= self._tensor_nbytes(existing)

        if self.max_bytes <= 0 or value_nbytes > self.max_bytes:
            return evicted

        while self._entries and self.bytes_resident + value_nbytes > self.max_bytes:
            evicted_key, evicted_value = self._entries.popitem(last=False)
            evicted_nbytes = self._tensor_nbytes(evicted_value)
            self.bytes_resident -= evicted_nbytes
            evicted.append((evicted_key, evicted_nbytes))

        self._entries[key] = value
        self.bytes_resident += value_nbytes
        return evicted

    def clear(self) -> None:
        self._entries.clear()
        self.bytes_resident = 0
