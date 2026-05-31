from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_lab.modules.ffn import SwiGLUFeedForward
from llm_lab.modules.linear import Linear


class TopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        active_experts: int,
        *,
        score_fn: str = "softmax",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if active_experts > num_experts:
            raise ValueError("active_experts cannot exceed num_experts")
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.score_fn = score_fn
        self.gate = Linear(d_model, num_experts, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.gate(x)
        top_values, top_indices = torch.topk(logits, k=self.active_experts, dim=-1)
        if self.score_fn == "softmax":
            weights = torch.softmax(top_values.float(), dim=-1).to(dtype=x.dtype)
        elif self.score_fn == "sqrtsoftplus":
            weights = torch.sqrt(torch.nn.functional.softplus(top_values.float())).to(dtype=x.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            raise ValueError(f"unknown router score_fn: {self.score_fn}")
        return top_indices, weights


class HashRouter(nn.Module):
    def __init__(self, num_experts: int, active_experts: int = 1) -> None:
        super().__init__()
        if active_experts > num_experts:
            raise ValueError("active_experts cannot exceed num_experts")
        self.num_experts = num_experts
        self.active_experts = active_experts

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        base = token_ids.long() % self.num_experts
        offsets = torch.arange(self.active_experts, device=token_ids.device)
        indices = (base[..., None] + offsets) % self.num_experts
        weights = torch.full(indices.shape, 1.0 / self.active_experts, device=token_ids.device, dtype=torch.float32)
        return indices, weights


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        active_experts: int,
        *,
        shared_experts: int = 0,
        score_fn: str = "softmax",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.shared_experts = shared_experts
        self.router = TopKRouter(
            d_model,
            num_experts,
            active_experts,
            score_fn=score_fn,
            device=device,
            dtype=dtype,
        )
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype) for _ in range(shared_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        top_indices, weights = self.router(x)
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
            expert_out = expert(flat_x[token_rows])
            flat_out[token_rows] += expert_out * flat_weights[token_rows, expert_slots, None]

        if self.shared:
            shared_out = sum(expert(x) for expert in self.shared)
            out = out + shared_out / math.sqrt(len(self.shared))
        return out

    def active_parameters_per_token(self) -> int:
        expert_params = sum(p.numel() for p in self.experts[0].parameters()) if self.experts else 0
        shared_params = sum(p.numel() for expert in self.shared for p in expert.parameters())
        router_params = sum(p.numel() for p in self.router.parameters())
        return router_params + self.active_experts * expert_params + shared_params
