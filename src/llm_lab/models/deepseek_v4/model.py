from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.common import LMHeadMixin, build_lm_head, make_token_positions
from llm_lab.models.deepseek_v4.attention import DeepSeekV4Attention
from llm_lab.models.deepseek_v4.moe import build_deepseek_moe
from llm_lab.modules import Linear, RotaryEmbedding, TokenEmbedding, build_norm


class MultiStreamResidual(nn.Module):
    def __init__(self, d_model: int, streams: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.streams = streams
        self.mix = Linear(d_model, d_model, device=device, dtype=dtype) if streams > 1 else None

    def forward(self, x: Tensor, update: Tensor) -> Tensor:
        if self.mix is None:
            return x + update
        return x + update + self.mix(update) / self.streams


class DeepSeekV4Block(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        attention_type: str,
        moe_type: str,
        *,
        rope: RotaryEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.moe_type = moe_type
        self.ln1 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.attn = DeepSeekV4Attention(
            cfg.d_model,
            cfg.num_heads,
            attention_type,
            rope=rope,
            local_window=cfg.local_window or 128,
            compression_ratio=cfg.compression_ratio or 4,
            topk=cfg.compressed_topk,
            qk_norm=cfg.qk_norm,
            device=device,
            dtype=dtype,
        )
        self.resid1 = MultiStreamResidual(cfg.d_model, cfg.residual_streams, device=device, dtype=dtype)
        self.ln2 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        if cfg.num_experts is None or cfg.active_experts is None:
            raise ValueError("deepseek_v4 requires num_experts and active_experts")
        self.ffn = build_deepseek_moe(
            moe_type,
            cfg.d_model,
            cfg.expert_d_ff or cfg.d_ff,
            cfg.num_experts,
            cfg.active_experts,
            cfg.shared_experts,
            score_fn=cfg.router_score,
            device=device,
            dtype=dtype,
        )
        self.resid2 = MultiStreamResidual(cfg.d_model, cfg.residual_streams, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor, token_ids: Tensor) -> Tensor:
        x = self.resid1(x, self.attn(self.ln1(x), token_positions))
        h = self.ln2(x)
        if self.moe_type == "hash_moe":
            x = self.resid2(x, self.ffn(h, token_ids))
        else:
            x = self.resid2(x, self.ffn(h))
        return x

    def active_parameters_per_token(self) -> int:
        ffn_active = self.ffn.active_parameters_per_token()
        other = sum(p.numel() for name, p in self.named_parameters() if not name.startswith("ffn."))
        return other + ffn_active


class DeepSeekV4LM(nn.Module, LMHeadMixin):
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
        attention_types = deepseek_attention_layer_types(cfg)
        moe_types = deepseek_moe_layer_types(cfg)
        self.token_embeddings = TokenEmbedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        rope = RotaryEmbedding(
            cfg.d_model // cfg.num_heads,
            cfg.context_length,
            theta=cfg.rope_theta,
            partial_rotary_factor=cfg.partial_rotary_factor,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                DeepSeekV4Block(cfg, attn_type, moe_type, rope=rope, device=device, dtype=dtype)
                for attn_type, moe_type in zip(attention_types, moe_types, strict=True)
            ]
        )
        self.final_norm = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.lm_head = build_lm_head(cfg.vocab_size, cfg.d_model, cfg.tie_embeddings, device=device, dtype=dtype)
        self.mtp_heads = nn.ModuleList(
            [Linear(cfg.d_model, cfg.vocab_size, device=device, dtype=dtype) for _ in range(cfg.mtp_layers)]
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self._hidden_states(token_ids)
        return self.project_logits(hidden)

    def forward_with_mtp(self, token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = self._hidden_states(token_ids)
        logits = self.project_logits(hidden)
        mtp_logits = [head(hidden) for head in self.mtp_heads]
        return logits, mtp_logits

    def _hidden_states(self, token_ids: Tensor) -> Tensor:
        token_positions = make_token_positions(token_ids)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions, token_ids)
        return self.final_norm(x)

    def active_parameters_per_token(self) -> int:
        base = sum(p.numel() for name, p in self.named_parameters() if not name.startswith("layers."))
        return base + sum(layer.active_parameters_per_token() for layer in self.layers)


def deepseek_attention_layer_types(cfg: ModelConfig) -> list[str]:
    if cfg.layer_types:
        return list(cfg.layer_types)
    pattern = ["sliding_attention", "compressed_sparse_attention", "sliding_attention", "heavily_compressed_attention"]
    return [pattern[i % len(pattern)] for i in range(cfg.num_layers)]


def deepseek_moe_layer_types(cfg: ModelConfig) -> list[str]:
    if cfg.moe_layer_types:
        return list(cfg.moe_layer_types)
    return ["hash_moe" if i < min(2, cfg.num_layers) else "moe" for i in range(cfg.num_layers)]
