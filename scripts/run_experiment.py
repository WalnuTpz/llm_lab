from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_CONFIGS = {
    "original_transformer": "configs/original_transformer.yaml",
    "modern_decoder": "configs/modern_decoder.yaml",
    "qwen36": "configs/qwen36.yaml",
    "deepseek_v4": "configs/deepseek_v4.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one formal LLM Lab experiment config.")
    parser.add_argument("model", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the train command without running it.")
    args = parser.parse_args()

    command = [sys.executable, "scripts/train.py", "--config", MODEL_CONFIGS[args.model]]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.dtype is not None:
        command.extend(["--dtype", args.dtype])
    if args.max_iters is not None:
        command.extend(["--max-iters", str(args.max_iters)])
    if args.resume is not None:
        command.append("--resume")
        if args.resume != "latest":
            command.append(args.resume)
    if args.smoke:
        command.append("--smoke")
    if args.no_checkpoint:
        command.append("--no-checkpoint")

    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
