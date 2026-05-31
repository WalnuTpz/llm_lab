from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from llm_lab.data import ByteLevelBPETokenizer, train_byte_level_bpe


ROOT = Path(__file__).resolve().parents[1]


def test_byte_level_bpe_roundtrip_save_load(tmp_path: Path):
    training_text = _training_text()
    tokenizer = train_byte_level_bpe([training_text], vocab_size=320, name="test_bpe")

    text = "hello 世界 token42"
    ids = tokenizer.encode(text, add_eot=True)
    assert tokenizer.eot_token_id == 319
    assert ids[-1] == tokenizer.eot_token_id
    assert max(ids) < tokenizer.vocab_size
    assert tokenizer.decode(ids) == text

    tokenizer.save(tmp_path / "bpe")
    loaded = ByteLevelBPETokenizer.load(tmp_path / "bpe")
    assert loaded.vocab_size == 320
    assert loaded.encode(text, add_eot=True) == ids
    assert loaded.decode(ids) == text


def test_train_and_tokenize_dataset_scripts(tmp_path: Path):
    raw_path = tmp_path / "sample.jsonl"
    rows = [{"text": _training_text()}, {"text": "held out document 世界 token17"}]
    raw_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    tokenizer_dir = tmp_path / "tokenizer"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            "--input",
            str(raw_path),
            "--output-dir",
            str(tokenizer_dir),
            "--vocab-size",
            "320",
            "--name",
            "script_test_bpe",
        ],
        cwd=ROOT,
        check=True,
    )

    output_path = tmp_path / "tokens.npy"
    subprocess.run(
        [
            sys.executable,
            "scripts/tokenize_dataset.py",
            "--input",
            str(raw_path),
            "--tokenizer-path",
            str(tokenizer_dir),
            "--output",
            str(output_path),
            "--add-eot",
            "--dtype",
            "uint16",
        ],
        cwd=ROOT,
        check=True,
    )

    tokens = np.load(output_path, mmap_mode="r")
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_dir)
    assert tokens.dtype == np.uint16
    assert len(tokens) > 0
    assert int(tokens.max()) < tokenizer.vocab_size
    assert int((tokens == tokenizer.eot_token_id).sum()) == len(rows)


def _training_text() -> str:
    parts = []
    for i in range(500):
        parts.append(f"token{i} alpha{i % 17} beta{i % 31} hello world byte pair encoding 世界 {i}\n")
    return "".join(parts)
