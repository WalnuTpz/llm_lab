from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.common import LMHeadMixin, build_lm_head, make_token_positions
from llm_lab.models.qwen36.mixers import GatedDeltaNetMixer, GatedFullAttention
from llm_lab.modules import Linear, RotaryEmbedding, SwiGLUFeedForward, TokenEmbedding, build_norm


class Qwen36Block(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        layer_type: str,
        *,
        rope: RotaryEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.ln1 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        if layer_type == "linear_attention":
            self.mixer = GatedDeltaNetMixer(cfg.d_model, cfg.num_heads, device=device, dtype=dtype)
        elif layer_type == "full_attention":
            if cfg.num_kv_heads is None:
                raise ValueError("qwen36 full_attention requires num_kv_heads")
            self.mixer = GatedFullAttention(
                cfg.d_model,
                cfg.num_heads,
                cfg.num_kv_heads,
                rope=rope,
                qk_norm=cfg.qk_norm,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"unsupported qwen36 layer_type: {layer_type}")
        self.ln2 = build_norm("rmsnorm", cfg.d_model, cfg.eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(cfg.d_model, cfg.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        h = self.ln1(x)
        if self.layer_type == "full_attention":
            x = x + self.mixer(h, token_positions)
        else:
            x = x + self.mixer(h)
        x = x + self.ffn(self.ln2(x))
        return x


class Qwen36LM(nn.Module, LMHeadMixin):
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
        layer_types = qwen36_layer_types(cfg)
        self.token_embeddings = TokenEmbedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        rope = RotaryEmbedding(
            cfg.d_model // cfg.num_heads,
            cfg.context_length,
            theta=cfg.rope_theta,
            partial_rotary_factor=cfg.partial_rotary_factor,
            device=device,
        )
        self.layers = nn.ModuleList(
            [Qwen36Block(cfg, layer_type, rope=rope, device=device, dtype=dtype) for layer_type in layer_types]
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
            x = layer(x, token_positions)
        return self.final_norm(x)

    def active_parameters_per_token(self) -> int:
        return sum(p.numel() for p in self.parameters())


def qwen36_layer_types(cfg: ModelConfig) -> list[str]:
    if cfg.layer_types:
        return list(cfg.layer_types)
    interval = cfg.full_attention_interval or 4
    layer_types = []
    for i in range(cfg.num_layers):
        layer_types.append("full_attention" if (i + 1) % interval == 0 else "linear_attention")
    return layer_types
