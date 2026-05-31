from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.modules.init import init_embedding_weight, init_linear_weight


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        init_linear_weight(self.weight, in_features, out_features)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            y = y + self.bias
        return y


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        init_embedding_weight(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
