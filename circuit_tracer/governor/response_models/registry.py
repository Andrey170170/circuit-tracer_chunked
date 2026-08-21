"""Explicit response-model family registry."""

from __future__ import annotations

from .contracts import ModelFamily


class ModelRegistry:
    def __init__(self) -> None:
        self._families: dict[str, ModelFamily] = {}

    def register(self, family: ModelFamily) -> None:
        if not family.kind or family.kind in self._families:
            raise ValueError(f"duplicate or empty response model kind: {family.kind!r}")
        self._families[family.kind] = family

    def get(self, kind: str) -> ModelFamily:
        try:
            return self._families[kind]
        except KeyError as error:
            raise ValueError(f"unregistered response model kind: {kind!r}") from error

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._families))
