from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from llm_lab.data import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize text into a .npy array with the byte tokenizer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--add-eot", action="store_true")
    args = parser.parse_args()

    tokenizer = ByteTokenizer()
    input_path = Path(args.input)
    ids = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line.rstrip("\n"), add_eot=args.add_eot))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(ids, dtype=np.int32))
    print(f"wrote {len(ids)} tokens to {output_path}")


if __name__ == "__main__":
    main()
