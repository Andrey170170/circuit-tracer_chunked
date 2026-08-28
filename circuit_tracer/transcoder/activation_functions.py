from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SELECTED_FEATURE_ACTIVATION_CAPABILITY = "supports_independent_feature_activation"


def require_independent_feature_activation(activation_function: object) -> None:
    """Reject activations whose result for one feature depends on peer features."""
    explicitly_supported = getattr(
        activation_function, SELECTED_FEATURE_ACTIVATION_CAPABILITY, None
    )
    if explicitly_supported is True:
        return
    if activation_function in (F.relu, torch.relu) or isinstance(
        activation_function, nn.ReLU
    ):
        return

    activation_name = type(activation_function).__name__
    raise ValueError(
        "selected-feature activation requires an explicitly supported featurewise "
        f"activation; {activation_name} may couple values across the feature axis"
    )


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    supports_independent_feature_activation = True

    def __init__(self, threshold: float | torch.Tensor, bandwidth: float = 2) -> None:
        super().__init__()
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold)

        self.threshold = nn.Parameter(threshold)
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)  # type: ignore

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, bandwidth={self.bandwidth}"


class TopK(nn.Module):
    supports_independent_feature_activation = False

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)
