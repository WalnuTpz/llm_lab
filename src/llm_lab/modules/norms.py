from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float * rms
        return (out.to(dtype=x.dtype)) * self.weight


def build_norm(
    norm_type: str,
    dim: int,
    eps: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    raise ValueError(f"unknown norm_type: {norm_type}")
