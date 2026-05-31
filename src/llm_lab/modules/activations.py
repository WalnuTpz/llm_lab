from __future__ import annotations

import torch
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def activation_fn(name: str):
    if name == "relu":
        return torch.relu
    if name == "silu":
        return silu
    raise ValueError(f"unknown activation: {name}")
