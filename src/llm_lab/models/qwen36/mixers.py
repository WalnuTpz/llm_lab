from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.modules import Linear, MultiHeadAttention, RotaryEmbedding


class GatedDeltaNetMixer(nn.Module):
    """Pure PyTorch Gated DeltaNet-like causal linear mixer.

    This preserves the Qwen3.6-style linear-mixer role without attempting to
    reproduce the production kernel exactly.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.gate_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self._shape(torch.nn.functional.elu(self.q_proj(x)) + 1.0, batch, seq_len)
        k = self._shape(torch.nn.functional.elu(self.k_proj(x)) + 1.0, batch, seq_len)
        v = self._shape(self.v_proj(x), batch, seq_len)
        gate = torch.sigmoid(self._shape(self.gate_proj(x), batch, seq_len))

        kv = torch.einsum("b h t d, b h t e -> b h t d e", k, v)
        kv_cumsum = kv.cumsum(dim=2)
        k_cumsum = k.cumsum(dim=2)
        numerator = torch.einsum("b h t d, b h t d e -> b h t e", q, kv_cumsum)
        denominator = torch.einsum("b h t d, b h t d -> b h t", q, k_cumsum).clamp_min(1e-6)
        y = numerator / denominator[..., None]
        y = y * gate
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(y)

    def _shape(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class GatedFullAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rope: RotaryEmbedding,
        qk_norm: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            num_kv_heads=num_kv_heads,
            rope=rope,
            qk_norm=qk_norm,
            device=device,
            dtype=dtype,
        )
        self.gate_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        return self.attn(x, token_positions) * torch.sigmoid(self.gate_proj(x))
