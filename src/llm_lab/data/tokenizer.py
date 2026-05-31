from __future__ import annotations

from collections.abc import Iterable, Iterator


class ByteTokenizer:
    """Minimal byte-level tokenizer for unified smoke/data pipelines."""

    def __init__(self, eot_token_id: int = 256) -> None:
        self.eot_token_id = eot_token_id
        self.vocab_size = eot_token_id + 1

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_eot:
            ids.append(self.eot_token_id)
        return ids

    def encode_iterable(self, texts: Iterable[str], *, add_eot: bool = False) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text, add_eot=add_eot)

    def decode(self, ids: Iterable[int]) -> str:
        byte_values = [idx for idx in ids if 0 <= idx < 256]
        return bytes(byte_values).decode("utf-8", errors="replace")
