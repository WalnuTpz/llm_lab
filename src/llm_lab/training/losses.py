from __future__ import annotations

import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size).float()
    targets_1d = targets.reshape(-1).long()
    log_denom = torch.logsumexp(logits_2d, dim=-1)
    gold = logits_2d.gather(dim=-1, index=targets_1d[:, None]).squeeze(-1)
    return (log_denom - gold).mean()


def perplexity(loss: Tensor) -> Tensor:
    return torch.exp(loss.float())
