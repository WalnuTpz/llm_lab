from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
    if len(dataset) <= context_length:
        raise ValueError("dataset length must exceed context_length")
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,), device="cpu")
    offsets = torch.arange(context_length + 1, device="cpu")
    idx = starts[:, None] + offsets[None, :]
    seq = torch.as_tensor(dataset[idx], dtype=torch.long)
    return seq[:, :-1].to(device), seq[:, 1:].to(device)
