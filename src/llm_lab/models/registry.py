from __future__ import annotations

import torch
from torch import nn

from llm_lab.config.schema import ModelConfig
from llm_lab.models.deepseek_v4 import DeepSeekV4LM
from llm_lab.models.modern_decoder import ModernDecoderLM
from llm_lab.models.original_transformer import OriginalTransformerLM
from llm_lab.models.qwen36 import Qwen36LM


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
    if cfg.architecture == "qwen36":
        return Qwen36LM(cfg, device=device, dtype=dtype)
    if cfg.architecture == "deepseek_v4":
        return DeepSeekV4LM(cfg, device=device, dtype=dtype)
    raise NotImplementedError(f"architecture is not implemented yet: {cfg.architecture}")
