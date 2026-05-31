from __future__ import annotations

import argparse

import torch

from llm_lab.config import load_config
from llm_lab.data import ByteLevelBPETokenizer, ByteTokenizer
from llm_lab.generation import generate
from llm_lab.models import build_model
from llm_lab.utils import dtype_from_str, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token ids from a configured model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default=None)
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
    tokenizer = ByteLevelBPETokenizer.load(cfg.data.tokenizer_path) if cfg.data.tokenizer_path else ByteTokenizer()
    if tokenizer.vocab_size > cfg.model.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds model vocab_size {cfg.model.vocab_size}"
        )
    if args.prompt is not None:
        ids = tokenizer.encode(args.prompt)
    else:
        ids = [int(part.strip()) for part in args.prompt_ids.split(",") if part.strip()]
    if ids and max(ids) >= cfg.model.vocab_size:
        raise ValueError(
            f"prompt contains token id {max(ids)} but configured vocab_size is {cfg.model.vocab_size}; "
            "use a config with a larger vocab or pass smaller --prompt-ids"
        )
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        prompt,
        args.max_new_tokens,
        context_length=cfg.model.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    ids_out = out[0].tolist()
    print(",".join(str(i) for i in ids_out))
    if args.prompt is not None:
        print(tokenizer.decode(ids_out))


if __name__ == "__main__":
    main()
