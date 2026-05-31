from __future__ import annotations

import torch
from torch import Tensor


def top_p_sample(logits: Tensor, *, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax((logits / temperature).float(), dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cdf = sorted_probs.cumsum(dim=-1)
        keep = cdf - sorted_probs < top_p
        keep[..., 0] = True
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(dim=-1, index=sorted_idx, src=sorted_probs)
    return torch.multinomial(probs, num_samples=1)
