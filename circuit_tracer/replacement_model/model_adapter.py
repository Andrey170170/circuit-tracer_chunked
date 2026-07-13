"""Explicit model capabilities used by the NNSight replacement runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class NNSightModelAdapter:
    """Model-specific behavior required by replacement-model setup."""

    architecture: str
    ignored_token_positions: slice = field(default_factory=lambda: slice(0, 1))
    required_token_prefix: tuple[int, ...] | None = None

    def normalize_feature_output(self, output: torch.Tensor) -> torch.Tensor:
        """Return the canonical ``(batch, position, model)`` activation shape."""
        if output.ndim == 2:
            return output.unsqueeze(0)
        if output.ndim != 3:
            raise ValueError(
                "feature output must have rank 2 or 3 "
                f"(architecture={self.architecture!r}, shape={tuple(output.shape)})"
            )
        return output

    def validate_preserved_prefix(self, tokens: torch.Tensor) -> bool:
        prefix = self.required_token_prefix
        if prefix is None:
            return False
        expected = torch.tensor(prefix, dtype=tokens.dtype, device=tokens.device)
        if tokens.numel() < expected.numel() or not torch.equal(tokens[: expected.numel()], expected):
            raise ValueError(
                "tokenized chat input does not contain the prefix required by "
                f"{self.architecture}"
            )
        return True


@dataclass(frozen=True)
class Gemma3NNSightModelAdapter(NNSightModelAdapter):
    """Gemma 3 prompt semantics selected by explicit tokenizer capability."""

    @classmethod
    def create(cls, *, architecture: str, has_chat_template: bool) -> "Gemma3NNSightModelAdapter":
        if has_chat_template:
            return cls(
                architecture=architecture,
                ignored_token_positions=slice(0, 4),
                required_token_prefix=(2, 105, 2364, 107),
            )
        return cls(architecture=architecture)


def resolve_model_adapter(
    *, architecture: str, has_chat_template: bool
) -> NNSightModelAdapter:
    """Resolve the explicit capability adapter for a supported HF architecture."""
    adapter_factories = {
        "Gemma3ForCausalLM": Gemma3NNSightModelAdapter.create,
        "Gemma3ForConditionalGeneration": Gemma3NNSightModelAdapter.create,
    }
    factory = adapter_factories.get(architecture)
    if factory is not None:
        return factory(
            architecture=architecture,
            has_chat_template=has_chat_template,
        )
    return NNSightModelAdapter(architecture=architecture)
