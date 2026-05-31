from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from llm_lab.data import ByteLevelBPETokenizer, iter_text_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize local text/jsonl files into a single .npy token array.")
    parser.add_argument("--input", nargs="+", required=True, help="Input .txt/.md/.jsonl files or directories.")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer directory or tokenizer.json file.")
    parser.add_argument("--output", required=True, help="Output .npy path.")
    parser.add_argument("--jsonl-field", default="text")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories.")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum documents to tokenize.")
    parser.add_argument("--add-eot", action="store_true", help="Append EOT after each document.")
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "int32", "int64"])
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_path)
    dtype = np.dtype(args.dtype)
    _ensure_tokenizer_fits_dtype(tokenizer, dtype)

    total_tokens, total_docs = _count_tokens(
        args.input,
        tokenizer=tokenizer,
        jsonl_field=args.jsonl_field,
        recursive=args.recursive,
        max_docs=args.max_docs,
        add_eot=args.add_eot,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(total_tokens,))

    offset = 0
    for text in iter_text_documents(
        args.input,
        jsonl_field=args.jsonl_field,
        recursive=args.recursive,
        max_docs=args.max_docs,
    ):
        ids = tokenizer.encode(text, add_eot=args.add_eot)
        next_offset = offset + len(ids)
        tokens[offset:next_offset] = np.asarray(ids, dtype=dtype)
        offset = next_offset
    tokens.flush()

    print(f"wrote tokens: {output_path}")
    print(f"documents: {total_docs}")
    print(f"tokens: {total_tokens}")
    print(f"dtype: {dtype}")


def _count_tokens(
    paths: list[str],
    *,
    tokenizer: ByteLevelBPETokenizer,
    jsonl_field: str,
    recursive: bool,
    max_docs: int | None,
    add_eot: bool,
) -> tuple[int, int]:
    total_tokens = 0
    total_docs = 0
    for text in iter_text_documents(paths, jsonl_field=jsonl_field, recursive=recursive, max_docs=max_docs):
        total_tokens += len(tokenizer.encode(text, add_eot=add_eot))
        total_docs += 1
    if total_tokens == 0:
        raise ValueError("tokenized dataset is empty")
    return total_tokens, total_docs


def _ensure_tokenizer_fits_dtype(tokenizer: ByteLevelBPETokenizer, dtype: np.dtype) -> None:
    if not np.issubdtype(dtype, np.integer):
        raise TypeError("token dtype must be an integer dtype")
    max_id = tokenizer.vocab_size - 1
    dtype_max = np.iinfo(dtype).max
    if max_id > dtype_max:
        raise ValueError(f"tokenizer max id {max_id} does not fit in dtype {dtype}")


if __name__ == "__main__":
    main()
