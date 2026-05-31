from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.common import LMHeadMixin, build_lm_head, make_token_positions
from llm_lab.modules import FeedForward, MultiHeadAttention, SinusoidalPositionEmbedding, TokenEmbedding, build_norm


class OriginalTransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            cfg.d_model,
            cfg.num_heads,
            num_kv_heads=cfg.num_heads,
            rope=None,
            qk_norm=False,
            local_window=None,
            device=device,
            dtype=dtype,
        )
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, activation="relu", device=device, dtype=dtype)
        self.ln1 = build_norm("layernorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.ln2 = build_norm("layernorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class OriginalTransformerLM(nn.Module, LMHeadMixin):
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
        self.position_embeddings = SinusoidalPositionEmbedding(
            cfg.context_length,
            cfg.d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList([OriginalTransformerBlock(cfg, device=device, dtype=dtype) for _ in range(cfg.num_layers)])
        self.final_norm = build_norm("layernorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.lm_head = build_lm_head(cfg.vocab_size, cfg.d_model, cfg.tie_embeddings, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        token_positions = make_token_positions(token_ids)
        x = self.token_embeddings(token_ids) + self.position_embeddings(token_positions)
        for layer in self.layers:
            x = layer(x)
        return self.project_logits(self.final_norm(x))

    def active_parameters_per_token(self) -> int:
        return sum(p.numel() for p in self.parameters())
