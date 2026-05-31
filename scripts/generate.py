from __future__ import annotations

import argparse

import torch

from llm_lab.config import load_config
from llm_lab.generation import generate
from llm_lab.models import build_model
from llm_lab.utils import dtype_from_str, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token ids from a configured model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-ids", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    dtype = safe_dtype_for_device(dtype_from_str(args.dtype or cfg.training.dtype), device)
    model = build_model(cfg.model, device=device, dtype=dtype)
    ids = [int(part.strip()) for part in args.prompt_ids.split(",") if part.strip()]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        prompt,
        args.max_new_tokens,
        context_length=cfg.model.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(",".join(str(i) for i in out[0].tolist()))


if __name__ == "__main__":
    main()
