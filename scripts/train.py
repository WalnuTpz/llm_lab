from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from llm_lab.config import load_config
from llm_lab.data import get_batch
from llm_lab.models import build_model
from llm_lab.training import cosine_lr, cross_entropy
from llm_lab.utils import dtype_from_str, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or smoke-test an LLM lab model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on random tokens instead of dataset files.")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.training.device)
    dtype = safe_dtype_for_device(dtype_from_str(args.dtype or cfg.training.dtype), device)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    model = build_model(cfg.model, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.adam_eps,
        weight_decay=cfg.training.weight_decay,
    )

    max_iters = args.max_iters or (1 if args.smoke else cfg.training.max_iters)
    if args.smoke:
        dataset = np.random.randint(0, cfg.model.vocab_size, size=cfg.model.context_length * 8 + 1, dtype=np.int64)
    else:
        if cfg.data.train_data is None:
            raise ValueError("train_data must be set unless --smoke is used")
        dataset = np.load(cfg.data.train_data, mmap_mode="r")

    model.train()
    start = time.perf_counter()
    for step in range(max_iters):
        lr = cosine_lr(
            step,
            cfg.training.learning_rate,
            cfg.training.min_learning_rate,
            cfg.training.warmup_iters,
            cfg.training.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(dataset, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        print(f"step={step} lr={lr:.6g} loss={float(loss.detach().cpu()):.6f}")
    elapsed = time.perf_counter() - start
    tokens = max_iters * cfg.training.batch_size * cfg.model.context_length
    print(f"tokens_per_second={tokens / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
