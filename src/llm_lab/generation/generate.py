from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.generation.sampler import top_p_sample


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    *,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    model.eval()
    out = prompt_ids
    for _ in range(max_new_tokens):
        context = out[:, -context_length:]
        logits = model(context)
        next_id = top_p_sample(logits[:, -1, :], temperature=temperature, top_p=top_p)
        out = torch.cat([out, next_id], dim=1)
        if eos_token_id is not None and bool((next_id == eos_token_id).all()):
            break
    return out
