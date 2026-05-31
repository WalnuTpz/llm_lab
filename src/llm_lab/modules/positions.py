from __future__ import annotations

import torch
from torch import Tensor, nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        if dtype is not None:
            pe = pe.to(dtype=dtype)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: Tensor) -> Tensor:
        return self.pe[token_positions]


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        partial_rotary_factor: float = 1.0,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        rotary_dim = int(dim * partial_rotary_factor)
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        if rotary_dim > dim:
            raise ValueError("rotary dimension cannot exceed input dimension")
        self.dim = dim
        self.rotary_dim = rotary_dim
        inv_freq = theta ** (-torch.arange(0, rotary_dim // 2, device=device, dtype=torch.float32) * 2.0 / rotary_dim)
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if self.rotary_dim == 0:
            return x
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        y_rot = torch.stack((y_even, y_odd), dim=-1).flatten(-2)
        return torch.cat((y_rot, x_pass), dim=-1)
