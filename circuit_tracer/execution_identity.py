"""Backend-neutral requested and prepared execution identity contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from circuit_tracer.governor.contracts import canonical_json


@dataclass(frozen=True)
class EffectiveExecutionDescriptor:
    """Stable, JSON-serializable description of prepared backend mechanisms."""

    schema_version: int
    backend: Literal["nnsight", "transformerlens"]
    provider: Mapping[str, Any] = field(default_factory=dict)
    numerics: Mapping[str, Any] = field(default_factory=dict)
    replay: Mapping[str, Any] = field(default_factory=dict)
    batches: Mapping[str, Any] = field(default_factory=dict)
    frontier: Mapping[str, Any] = field(default_factory=dict)
    decoder: Mapping[str, Any] = field(default_factory=dict)
    storage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the same canonical primitive tree used for fingerprinting."""
        return json.loads(canonical_json(self))


@dataclass(frozen=True)
class EffectiveExecutionIdentity:
    """Prepared execution descriptor and its stable fingerprint."""

    descriptor: EffectiveExecutionDescriptor | None
    fingerprint: str


@dataclass
class ExecutionIdentityState:
    """Runner-owned requested identity, populated with effective identity on preparation."""

    requested_fingerprint: str
    effective: EffectiveExecutionIdentity | None = None
    effective_revisions: list[EffectiveExecutionIdentity] = field(default_factory=list)

    def mark_effective(self, identity: EffectiveExecutionIdentity) -> None:
        if self.effective is not None and self.effective != identity:
            raise RuntimeError("effective execution identity is already set")
        self.effective = identity
        if not self.effective_revisions:
            self.effective_revisions.append(identity)

    def revise_effective(self, identity: EffectiveExecutionIdentity) -> None:
        """Record an effective identity produced by an allowed planning epoch."""
        if self.effective is None:
            self.mark_effective(identity)
            return
        if self.effective == identity:
            return
        self.effective = identity
        self.effective_revisions.append(identity)

    def mark_requested_as_effective(self) -> None:
        self.mark_effective(
            EffectiveExecutionIdentity(descriptor=None, fingerprint=self.requested_fingerprint)
        )

    @property
    def effective_fingerprint(self) -> str | None:
        return None if self.effective is None else self.effective.fingerprint

    @property
    def execution_fingerprint(self) -> str:
        """Compatibility identity: effective when prepared, otherwise requested."""
        return self.effective_fingerprint or self.requested_fingerprint
