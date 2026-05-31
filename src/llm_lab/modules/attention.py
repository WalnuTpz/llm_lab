from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_lab.modules.linear import Linear
from llm_lab.modules.positions import RotaryEmbedding


class QKNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        rope: RotaryEmbedding | None = None,
        qk_norm: bool = False,
        local_window: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.rope = rope
        self.q_norm = QKNorm() if qk_norm else None
        self.k_norm = QKNorm() if qk_norm else None

        self.q_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.o_proj = Linear(num_heads * self.head_dim, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).expand(batch, seq_len)
            pos = token_positions[:, None, :]
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_mask = causal_attention_mask(seq_len, device=x.device, local_window=self.local_window)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        y = torch.matmul(probs, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(y)


def causal_attention_mask(seq_len: int, *, device: torch.device, local_window: int | None = None) -> Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    mask = k <= q
    if local_window is not None:
        mask = mask & (k >= q - local_window + 1)
    return mask[None, None, :, :]
