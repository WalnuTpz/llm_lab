from __future__ import annotations

import argparse
import time

import torch

from llm_lab.config import load_config
from llm_lab.models import build_model
from llm_lab.utils import dtype_from_str, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark configured model forward pass.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    dtype = safe_dtype_for_device(dtype_from_str(args.dtype or cfg.training.dtype), device)
    model = build_model(cfg.model, device=device, dtype=dtype).eval()
    batch_size = args.batch_size or cfg.training.batch_size
    context_length = args.context_length or cfg.model.context_length
    x = torch.randint(0, cfg.model.vocab_size, (batch_size, context_length), device=device)

    with torch.no_grad():
        model(x)
        start = time.perf_counter()
        for _ in range(args.steps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    tokens = args.steps * batch_size * context_length
    print(f"tokens_per_second={tokens / max(elapsed, 1e-9):.2f}")
    if device.type == "cuda":
        print(f"max_memory_allocated={torch.cuda.max_memory_allocated(device)}")


if __name__ == "__main__":
    main()
