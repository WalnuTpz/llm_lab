from __future__ import annotations

import math


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if step < warmup_iters:
        return max_lr * step / max(warmup_iters, 1)
    if step <= cosine_cycle_iters:
        progress = (step - warmup_iters) / max(cosine_cycle_iters - warmup_iters, 1)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr
