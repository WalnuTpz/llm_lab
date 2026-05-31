from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_lab.modules.activations import activation_fn, silu
from llm_lab.modules.linear import Linear


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        activation: str = "relu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.activation = activation_fn(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)))


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(math.ceil((8.0 / 3.0) * d_model))
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def build_ffn(
    ffn_type: str,
    d_model: int,
    d_ff: int,
    *,
    activation: str = "relu",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if ffn_type == "relu":
        return FeedForward(d_model, d_ff, activation=activation, device=device, dtype=dtype)
    if ffn_type == "swiglu":
        return SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
    raise ValueError(f"unsupported ffn_type for dense builder: {ffn_type}")
