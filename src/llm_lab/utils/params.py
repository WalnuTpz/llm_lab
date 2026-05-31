from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class ParameterReport:
    total_parameters: int
    trainable_parameters: int
    embedding_parameters: int
    non_embedding_parameters: int
    active_parameters_per_token: int | None = None


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_report(module: nn.Module, *, active_parameters_per_token: int | None = None) -> ParameterReport:
    total = count_parameters(module)
    trainable = count_parameters(module, trainable_only=True)
    embedding = sum(p.numel() for name, p in module.named_parameters() if "embedding" in name or "embed" in name)
    return ParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        embedding_parameters=embedding,
        non_embedding_parameters=total - embedding,
        active_parameters_per_token=active_parameters_per_token,
    )
