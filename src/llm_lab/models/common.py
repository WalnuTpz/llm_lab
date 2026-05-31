from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.modules.linear import Linear, TokenEmbedding


class LMHeadMixin:
    token_embeddings: TokenEmbedding
    lm_head: Linear | None
    tie_embeddings: bool

    def project_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            return x @ self.token_embeddings.weight.transpose(0, 1)
        if self.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings is false")
        return self.lm_head(x)


def make_token_positions(token_ids: Tensor) -> Tensor:
    batch, seq_len = token_ids.shape
    return torch.arange(seq_len, device=token_ids.device).expand(batch, seq_len)


def build_lm_head(
    vocab_size: int,
    d_model: int,
    tie_embeddings: bool,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Linear | None:
    if tie_embeddings:
        return None
    return Linear(d_model, vocab_size, device=device, dtype=dtype)
