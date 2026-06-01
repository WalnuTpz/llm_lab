from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
    x, targets = get_batch_with_future_targets(
        dataset,
        batch_size,
        context_length,
        device,
        num_targets=1,
    )
    return x, targets[0]


def get_batch_with_future_targets(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    *,
    num_targets: int,
) -> tuple[Tensor, list[Tensor]]:
    if num_targets <= 0:
        raise ValueError("num_targets must be positive")
    required_length = context_length + num_targets
    if len(dataset) < required_length:
        raise ValueError("dataset length must fit context_length plus target offsets")
    starts = torch.randint(0, len(dataset) - required_length + 1, (batch_size,), device="cpu")
    offsets = torch.arange(required_length, device="cpu")
    idx = starts[:, None] + offsets[None, :]
    seq = torch.as_tensor(dataset[idx], dtype=torch.long)
    x = seq[:, :context_length].to(device)
    targets = [seq[:, offset : offset + context_length].to(device) for offset in range(1, num_targets + 1)]
    return x, targets
