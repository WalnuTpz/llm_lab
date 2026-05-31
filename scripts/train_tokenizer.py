from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.data import iter_text_documents, train_byte_level_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from local text/jsonl files.")
    parser.add_argument("--input", nargs="+", required=True, help="Input .txt/.md/.jsonl files or directories.")
    parser.add_argument("--output-dir", required=True, help="Directory where tokenizer.json will be written.")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--name", default="byte_bpe_16k")
    parser.add_argument("--jsonl-field", default="text")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum training characters to sample.")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum documents to read.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories.")
    parser.add_argument("--eot-token", default="<|endoftext|>")
    args = parser.parse_args()

    texts = iter_text_documents(
        args.input,
        jsonl_field=args.jsonl_field,
        recursive=args.recursive,
        max_docs=args.max_docs,
    )
    tokenizer = train_byte_level_bpe(
        texts,
        vocab_size=args.vocab_size,
        name=args.name,
        eot_token=args.eot_token,
        max_chars=args.max_chars,
    )
    tokenizer_path = tokenizer.save(Path(args.output_dir))
    print(f"wrote tokenizer: {tokenizer_path}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"eot_token_id: {tokenizer.eot_token_id}")


if __name__ == "__main__":
    main()
