from __future__ import annotations

import math

import torch
from torch import nn


def init_linear_weight(weight: nn.Parameter | torch.Tensor, in_features: int, out_features: int) -> None:
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


def init_embedding_weight(weight: nn.Parameter | torch.Tensor) -> None:
    d_model = weight.shape[-1]
    std = 1.0 / math.sqrt(d_model)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
