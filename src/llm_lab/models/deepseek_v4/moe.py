from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_lab.modules import HashRouter, MoEFeedForward, SwiGLUFeedForward


class HashMoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        active_experts: int,
        *,
        shared_experts: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.router = HashRouter(num_experts, active_experts)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype) for _ in range(shared_experts)]
        )

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        top_indices, weights = self.router(token_ids)
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_out = out.reshape(-1, out.shape[-1])
        flat_indices = top_indices.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts).to(dtype=x.dtype)
        for expert_id, expert in enumerate(self.experts):
            selected = flat_indices == expert_id
            if not selected.any():
                continue
            token_rows, expert_slots = selected.nonzero(as_tuple=True)
            flat_out[token_rows] += expert(flat_x[token_rows]) * flat_weights[token_rows, expert_slots, None]
        if self.shared:
            shared_out = sum(expert(x) for expert in self.shared)
            out = out + shared_out / math.sqrt(len(self.shared))
        return out

    def active_parameters_per_token(self) -> int:
        expert_params = sum(p.numel() for p in self.experts[0].parameters()) if self.experts else 0
        shared_params = sum(p.numel() for expert in self.shared for p in expert.parameters())
        return self.active_experts * expert_params + shared_params


def build_deepseek_moe(
    layer_type: str,
    d_model: int,
    d_ff: int,
    num_experts: int,
    active_experts: int,
    shared_experts: int,
    *,
    score_fn: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if layer_type == "hash_moe":
        return HashMoEFeedForward(
            d_model,
            d_ff,
            num_experts,
            active_experts,
            shared_experts=shared_experts,
            device=device,
            dtype=dtype,
        )
    if layer_type == "moe":
        return MoEFeedForward(
            d_model,
            d_ff,
            num_experts,
            active_experts,
            shared_experts=shared_experts,
            score_fn=score_fn,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"unsupported deepseek_v4 moe layer_type: {layer_type}")
