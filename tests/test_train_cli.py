from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from llm_lab.data import train_byte_level_bpe
from scripts.train import _effective_lr_schedule


ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_real_data_checkpoint_and_resume(tmp_path: Path):
    tokenizer = train_byte_level_bpe([_training_text()], vocab_size=320, name="train_cli_bpe")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)

    train_tokens = (np.arange(256, dtype=np.uint16) % tokenizer.vocab_size).astype(np.uint16)
    val_tokens = (np.arange(128, dtype=np.uint16) % tokenizer.vocab_size).astype(np.uint16)
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    checkpoint_dir = tmp_path / "checkpoints"
    run_dir = tmp_path / "runs"
    config_path = tmp_path / "train_real_data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "train_real_data_smoke",
                "model": {
                    "architecture": "modern_decoder",
                    "vocab_size": tokenizer.vocab_size,
                    "context_length": 8,
                    "d_model": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "d_ff": 64,
                    "tie_embeddings": True,
                    "norm_type": "rmsnorm",
                    "activation": "swiglu",
                    "ffn_type": "swiglu",
                    "attention_type": "gqa",
                    "position_encoding": "rope",
                    "qk_norm": True,
                },
                "data": {
                    "train_data": str(train_path),
                    "val_data": str(val_path),
                    "tokenizer_path": str(tokenizer_dir),
                    "dataset_dtype": "uint16",
                },
                "training": {
                    "batch_size": 2,
                    "max_iters": 2,
                    "learning_rate": 0.001,
                    "min_learning_rate": 0.0001,
                    "warmup_iters": 1,
                    "cosine_cycle_iters": 4,
                    "dtype": "float32",
                    "device": "cpu",
                    "grad_accum_steps": 2,
                    "seed": 123,
                },
                "runtime": {
                    "log_interval": 1,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "checkpoint_interval": 1,
                    "checkpoint_dir": str(checkpoint_dir),
                    "run_dir": str(run_dir),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _run_train(config_path, max_iters=2)
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "step_00000002.pt").exists()

    _run_train(config_path, max_iters=3, resume=True)
    assert (checkpoint_dir / "step_00000003.pt").exists()
    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == 3

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record["type"] == "train" and record["step"] == 1 for record in records)
    first_train = next(record for record in records if record["type"] == "train" and record["step"] == 1)
    assert first_train["grad_accum_steps"] == 2
    assert first_train["tokens"] == 2 * 8 * 2
    assert any(record["type"] == "eval" and record["step"] == 2 for record in records)
    assert any(record["type"] == "train" and record["step"] == 3 for record in records)
    assert any(record["type"] == "eval" and record["step"] == 3 for record in records)


def test_train_cli_records_mtp_loss_components(tmp_path: Path):
    run_dir = tmp_path / "runs"
    config_path = tmp_path / "qwen_mtp_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "qwen_mtp_smoke",
                "model": {
                    "architecture": "qwen36",
                    "vocab_size": 64,
                    "context_length": 8,
                    "d_model": 32,
                    "num_layers": 4,
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "d_ff": 64,
                    "tie_embeddings": True,
                    "norm_type": "rmsnorm",
                    "activation": "swiglu",
                    "ffn_type": "swiglu",
                    "attention_type": "hybrid",
                    "position_encoding": "rope",
                    "partial_rotary_factor": 0.5,
                    "full_attention_interval": 4,
                    "mtp_layers": 1,
                    "mtp_loss_weight": 0.1,
                },
                "training": {
                    "batch_size": 2,
                    "max_iters": 1,
                    "learning_rate": 0.001,
                    "min_learning_rate": 0.0001,
                    "warmup_iters": 1,
                    "cosine_cycle_iters": 1,
                    "dtype": "float32",
                    "device": "cpu",
                    "seed": 123,
                },
                "runtime": {
                    "log_interval": 1,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "checkpoint_interval": 1,
                    "checkpoint_dir": str(tmp_path / "checkpoints"),
                    "run_dir": str(run_dir),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--smoke",
            "--max-iters",
            "1",
            "--no-checkpoint",
        ],
        cwd=ROOT,
        check=True,
    )
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_record = next(record for record in records if record["type"] == "run")
    train_record = next(record for record in records if record["type"] == "train")
    assert run_record["mtp_layers"] == 1
    assert run_record["mtp_loss_weight"] == 0.1
    assert train_record["next_token_loss"] > 0
    assert train_record["mtp_loss"] > 0
    expected_loss = train_record["next_token_loss"] + 0.1 * train_record["mtp_loss"]
    assert abs(train_record["loss"] - expected_loss) < 1e-6


def test_max_iters_override_scales_default_lr_schedule():
    assert _effective_lr_schedule(
        config_max_iters=1000,
        max_iters=6000,
        config_warmup_iters=100,
        config_cosine_cycle_iters=1000,
    ) == (600, 6000)


def test_custom_lr_schedule_survives_max_iters_override():
    assert _effective_lr_schedule(
        config_max_iters=1000,
        max_iters=6000,
        config_warmup_iters=100,
        config_cosine_cycle_iters=2000,
    ) == (100, 2000)


def _run_train(config_path: Path, *, max_iters: int, resume: bool = False) -> None:
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--max-iters",
        str(max_iters),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


def _training_text() -> str:
    return " ".join(f"openwebtext tiny tokenizer fixture token{i}" for i in range(600))
