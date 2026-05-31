from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm_lab.modules import Linear, MultiHeadAttention, RotaryEmbedding


class CompressedHybridAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        rope: RotaryEmbedding,
        compression_ratio: int = 4,
        topk: int | None = None,
        qk_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.compression_ratio = max(1, compression_ratio)
        self.topk = topk
        self.rope = rope
        self.qk_norm = qk_norm
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.head_dim, device=device, dtype=dtype)
        self.sink = nn.Parameter(torch.zeros(num_heads, 1, 1, device=device, dtype=dtype))
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x)[:, None, :, :]
        v = self.v_proj(x)[:, None, :, :]
        pos = token_positions[:, None, :]
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        if self.qk_norm:
            q = _rms_unit(q)
            k = _rms_unit(k)

        k_comp, v_comp, comp_positions = _compress_kv(k, v, self.compression_ratio)
        causal = comp_positions[None, None, None, :] <= token_positions[:, None, :, None]
        scores = torch.matmul(q, k_comp.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.sink[None, :, :, :]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if self.topk is not None and self.topk < scores.shape[-1]:
            scores = _mask_to_topk(scores, self.topk)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        y = torch.matmul(probs, v_comp.expand(batch, self.num_heads, -1, -1))
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(y)


class DeepSeekV4Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        layer_type: str,
        *,
        rope: RotaryEmbedding,
        local_window: int,
        compression_ratio: int,
        topk: int | None,
        qk_norm: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "sliding_attention":
            self.attn = MultiHeadAttention(
                d_model,
                num_heads,
                num_kv_heads=1,
                rope=rope,
                qk_norm=qk_norm,
                local_window=local_window,
                device=device,
                dtype=dtype,
            )
        elif layer_type in {"compressed_sparse_attention", "heavily_compressed_attention"}:
            effective_topk = topk if layer_type == "compressed_sparse_attention" else None
            effective_ratio = compression_ratio if layer_type == "compressed_sparse_attention" else compression_ratio * 2
            self.attn = CompressedHybridAttention(
                d_model,
                num_heads,
                rope=rope,
                compression_ratio=effective_ratio,
                topk=effective_topk,
                qk_norm=qk_norm,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"unsupported deepseek_v4 attention layer_type: {layer_type}")

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        return self.attn(x, token_positions)


def _rms_unit(x: Tensor) -> Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (x.float() * scale).to(dtype=x.dtype)


def _compress_kv(k: Tensor, v: Tensor, ratio: int) -> tuple[Tensor, Tensor, Tensor]:
    batch, heads, seq_len, dim = k.shape
    pad = (ratio - (seq_len % ratio)) % ratio
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    groups = k.shape[2] // ratio
    k_comp = k.view(batch, heads, groups, ratio, dim).mean(dim=3)
    v_comp = v.view(batch, heads, groups, ratio, dim).mean(dim=3)
    positions = torch.arange(groups, device=k.device) * ratio + ratio - 1
    positions = positions.clamp_max(seq_len - 1)
    return k_comp, v_comp, positions


def _mask_to_topk(scores: Tensor, topk: int) -> Tensor:
    values, indices = torch.topk(scores, k=topk, dim=-1)
    masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
    return masked.scatter(dim=-1, index=indices, src=values)
