from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.common import LMHeadMixin, build_lm_head, make_token_positions
from llm_lab.modules import MultiHeadAttention, RotaryEmbedding, SwiGLUFeedForward, TokenEmbedding, build_norm


class ModernDecoderBlock(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        rope: RotaryEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            cfg.d_model,
            cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            rope=rope,
            qk_norm=cfg.qk_norm,
            local_window=cfg.local_window,
            device=device,
            dtype=dtype,
        )
        self.ln2 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(cfg.d_model, cfg.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class ModernDecoderLM(nn.Module, LMHeadMixin):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tie_embeddings = cfg.tie_embeddings
        self.token_embeddings = TokenEmbedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        rope = RotaryEmbedding(
            cfg.d_model // cfg.num_heads,
            cfg.context_length,
            theta=cfg.rope_theta,
            partial_rotary_factor=cfg.partial_rotary_factor,
            device=device,
        )
        self.layers = nn.ModuleList(
            [ModernDecoderBlock(cfg, rope=rope, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        )
        self.final_norm = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.lm_head = build_lm_head(cfg.vocab_size, cfg.d_model, cfg.tie_embeddings, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        token_positions = make_token_positions(token_ids)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.project_logits(self.final_norm(x))

    def active_parameters_per_token(self) -> int:
        return sum(p.numel() for p in self.parameters())
