from __future__ import annotations

import torch
from torch import nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.modern_decoder import ModernDecoderLM
from llm_lab.models.original_transformer import OriginalTransformerLM


def build_model(
    cfg: ModelConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    cfg.validate()
    if cfg.architecture == "original_transformer":
        return OriginalTransformerLM(cfg, device=device, dtype=dtype)
    if cfg.architecture == "modern_decoder":
        return ModernDecoderLM(cfg, device=device, dtype=dtype)
    raise NotImplementedError(f"architecture is not implemented yet: {cfg.architecture}")
