from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex
import tiktoken
import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


DEFAULT_BPE_PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_EOT_TOKEN = "<|endoftext|>"


@dataclass(slots=True)
class ByteLevelBPETokenizer:
    name: str
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int] = field(default_factory=dict)
    eot_token: str = DEFAULT_EOT_TOKEN
    _encoding: tiktoken.Encoding = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._encoding = tiktoken.Encoding(
            self.name,
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
            explicit_n_vocab=self.vocab_size,
        )

    @property
    def vocab_size(self) -> int:
        max_merge = max(self.mergeable_ranks.values(), default=-1)
        max_special = max(self.special_tokens.values(), default=-1)
        return max(max_merge, max_special) + 1

    @property
    def eot_token_id(self) -> int | None:
        return self.special_tokens.get(self.eot_token)

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        ids = self._encoding.encode_ordinary(text)
        if add_eot:
            if self.eot_token_id is None:
                raise ValueError("add_eot=True requires an eot token in special_tokens")
            ids.append(self.eot_token_id)
        return ids

    def encode_iterable(self, texts: Iterable[str], *, add_eot: bool = False) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text, add_eot=add_eot)

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        token_ids = list(ids)
        if skip_special and self.special_tokens:
            special_ids = set(self.special_tokens.values())
            token_ids = [idx for idx in token_ids if idx not in special_ids]
        return self._encoding.decode(token_ids)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix == ".json":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            tokenizer_path = output_path / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if tokenizer_path.name == "tokenizer.json":
            config_path = tokenizer_path.parent / "tokenizer_config.yaml"
            config_path.write_text(yaml.safe_dump(self.to_config_dict(), sort_keys=False), encoding="utf-8")
        return tokenizer_path

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "byte_level_bpe",
            "vocab_size": self.vocab_size,
            "pat_str": self.pat_str,
            "eot_token": self.eot_token,
            "eot_token_id": self.eot_token_id,
            "special_tokens": dict(self.special_tokens),
            "tokenizer_file": "tokenizer.json",
        }

    def to_dict(self) -> dict[str, Any]:
        mergeable = [
            {"token": base64.b64encode(token).decode("ascii"), "rank": rank}
            for token, rank in sorted(self.mergeable_ranks.items(), key=lambda item: item[1])
        ]
        return {
            "name": self.name,
            "type": "byte_level_bpe",
            "pat_str": self.pat_str,
            "special_tokens": dict(self.special_tokens),
            "eot_token": self.eot_token,
            "mergeable_ranks": mergeable,
        }

    @classmethod
    def load(cls, path: str | Path) -> ByteLevelBPETokenizer | HFByteLevelBPETokenizer:
        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        if raw.get("type") != "byte_level_bpe":
            if raw.get("model", {}).get("type") == "BPE":
                return HFByteLevelBPETokenizer.load(tokenizer_path)
            raise ValueError(f"unsupported tokenizer type: {raw.get('type')}")
        mergeable = {
            base64.b64decode(item["token"].encode("ascii")): int(item["rank"])
            for item in raw["mergeable_ranks"]
        }
        return cls(
            name=raw["name"],
            pat_str=raw["pat_str"],
            mergeable_ranks=mergeable,
            special_tokens={str(k): int(v) for k, v in raw.get("special_tokens", {}).items()},
            eot_token=raw.get("eot_token", DEFAULT_EOT_TOKEN),
        )


@dataclass(slots=True)
class HFByteLevelBPETokenizer:
    name: str
    tokenizer: Tokenizer
    eot_token: str = DEFAULT_EOT_TOKEN

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def eot_token_id(self) -> int | None:
        token_id = self.tokenizer.token_to_id(self.eot_token)
        return int(token_id) if token_id is not None else None

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_eot:
            if self.eot_token_id is None:
                raise ValueError("add_eot=True requires an eot token in special_tokens")
            ids.append(self.eot_token_id)
        return ids

    def encode_iterable(self, texts: Iterable[str], *, add_eot: bool = False) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text, add_eot=add_eot)

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix == ".json":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            tokenizer_path = output_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        if tokenizer_path.name == "tokenizer.json":
            config_path = tokenizer_path.parent / "tokenizer_config.yaml"
            config_path.write_text(yaml.safe_dump(self.to_config_dict(), sort_keys=False), encoding="utf-8")
        return tokenizer_path

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "huggingface_byte_level_bpe",
            "vocab_size": self.vocab_size,
            "eot_token": self.eot_token,
            "eot_token_id": self.eot_token_id,
            "tokenizer_file": "tokenizer.json",
            "backend": "tokenizers",
        }

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, eot_token: str = DEFAULT_EOT_TOKEN) -> HFByteLevelBPETokenizer:
        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config_path = tokenizer_path.parent / "tokenizer_config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            name = name or config.get("name")
            eot_token = config.get("eot_token", eot_token)
        return cls(name=name or tokenizer_path.parent.name or "hf_byte_bpe", tokenizer=tokenizer, eot_token=eot_token)


def train_hf_byte_level_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 16_384,
    name: str = "byte_bpe_16k",
    eot_token: str = DEFAULT_EOT_TOKEN,
    max_chars: int | None = None,
) -> HFByteLevelBPETokenizer:
    if vocab_size < 257:
        raise ValueError("vocab_size must be at least 257 to include all bytes and one EOT token")

    tokenizer = Tokenizer(BPE(unk_token=None, fuse_unk=False))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size - 1,
        min_frequency=2,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_limited_chars(texts, max_chars=max_chars), trainer=trainer)
    tokenizer.add_special_tokens([eot_token])
    wrapped = HFByteLevelBPETokenizer(name=name, tokenizer=tokenizer, eot_token=eot_token)
    if wrapped.vocab_size != vocab_size:
        raise ValueError(f"trained tokenizer vocab_size {wrapped.vocab_size} does not match requested {vocab_size}")
    return wrapped


def train_byte_level_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 16_384,
    name: str = "byte_bpe_16k",
    eot_token: str = DEFAULT_EOT_TOKEN,
    pat_str: str = DEFAULT_BPE_PAT_STR,
    max_chars: int | None = None,
) -> ByteLevelBPETokenizer:
    if vocab_size < 257:
        raise ValueError("vocab_size must be at least 257 to include all bytes and one EOT token")

    training_text = _collect_training_text(texts, max_chars=max_chars)
    if not training_text:
        raise ValueError("cannot train tokenizer on empty text")

    merge_vocab_size = vocab_size - 1
    mergeable_ranks = _train_mergeable_ranks(training_text, merge_vocab_size, pat_str)
    special_tokens = {eot_token: merge_vocab_size}
    return ByteLevelBPETokenizer(
        name=name,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
        eot_token=eot_token,
    )


def _iter_limited_chars(texts: Iterable[str], *, max_chars: int | None) -> Iterator[str]:
    total_chars = 0
    for text in texts:
        if max_chars is not None and total_chars >= max_chars:
            break
        if max_chars is not None and total_chars + len(text) > max_chars:
            text = text[: max_chars - total_chars]
        total_chars += len(text)
        if text:
            yield text


def _collect_training_text(texts: Iterable[str], *, max_chars: int | None) -> str:
    chunks: list[str] = []
    total_chars = 0
    for text in texts:
        if max_chars is not None and total_chars >= max_chars:
            break
        if max_chars is not None and total_chars + len(text) > max_chars:
            text = text[: max_chars - total_chars]
        chunks.append(text)
        total_chars += len(text)
    return "\n".join(chunks)


def _train_mergeable_ranks(text: str, vocab_size: int, pat_str: str) -> dict[bytes, int]:
    ranks = {bytes([idx]): idx for idx in range(256)}
    word_counts = _pretokenize(text, pat_str)
    if not word_counts:
        raise ValueError("tokenizer training text produced no regex pieces")

    while len(ranks) < vocab_size:
        pair_stats = _pair_counts(word_counts)
        if not pair_stats:
            raise ValueError(
                f"training sample is too small to reach vocab_size={vocab_size + 1}; "
                f"stopped at mergeable tokens={len(ranks)}"
            )
        best_pair = None
        best_count = -1
        for pair, count in pair_stats.items():
            token = pair[0] + pair[1]
            if token in ranks:
                continue
            if count > best_count:
                best_pair = pair
                best_count = count
        if best_pair is None:
            raise ValueError(
                f"training sample cannot produce more unique BPE merges; stopped at mergeable tokens={len(ranks)}"
            )

        merged_token = best_pair[0] + best_pair[1]
        ranks[merged_token] = len(ranks)
        word_counts = _merge_word_counts(word_counts, best_pair, merged_token)
    return ranks


def _pretokenize(text: str, pat_str: str) -> Counter[tuple[bytes, ...]]:
    pieces = regex.findall(pat_str, text)
    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece in pieces:
        token_bytes = tuple(bytes([byte]) for byte in piece.encode("utf-8"))
        if token_bytes:
            counts[token_bytes] += 1
    return counts


def _pair_counts(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        for left, right in zip(word[:-1], word[1:]):
            counts[(left, right)] += freq
    return counts


def _merge_word_counts(
    word_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> Counter[tuple[bytes, ...]]:
    merged_counts: Counter[tuple[bytes, ...]] = Counter()
    for word, freq in word_counts.items():
        merged_word: list[bytes] = []
        idx = 0
        while idx < len(word):
            if idx < len(word) - 1 and word[idx] == pair[0] and word[idx + 1] == pair[1]:
                merged_word.append(merged_token)
                idx += 2
            else:
                merged_word.append(word[idx])
                idx += 1
        merged_counts[tuple(merged_word)] += freq
    return merged_counts
