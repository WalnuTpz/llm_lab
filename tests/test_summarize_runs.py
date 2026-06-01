from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.summarize_runs import summarize_runs


def test_summarize_runs_reads_metrics_and_checkpoints(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "modern_decoder"
    run_dir.mkdir(parents=True)
    records = [
        {
            "type": "run",
            "name": "modern_decoder_100m",
            "architecture": "modern_decoder",
            "git_commit": "abc123",
            "total_parameters": 100,
            "active_parameters_per_token": 100,
            "batch_size": 2,
            "context_length": 8,
            "grad_accum_steps": 4,
        },
        {"type": "train", "step": 1, "loss": 4.0, "tokens": 64, "tokens_per_second": 1000.0},
        {"type": "eval", "step": 1, "val_loss": 4.2},
        {"type": "resume", "step": 1, "checkpoint": "latest.pt"},
        {
            "type": "train",
            "step": 2,
            "loss": 3.5,
            "next_token_loss": 3.2,
            "mtp_loss": 3.0,
            "tokens": 128,
            "tokens_per_second": 1100.0,
        },
        {"type": "eval", "step": 2, "val_loss": 3.9},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "modern_decoder"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_00000001.pt").write_bytes(b"")
    (checkpoint_dir / "step_00000002.pt").write_bytes(b"")
    (checkpoint_dir / "latest.pt").write_bytes(b"")

    summaries = summarize_runs(runs_root, checkpoints_root=tmp_path / "checkpoints")

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["name"] == "modern_decoder_100m"
    assert summary["architecture"] == "modern_decoder"
    assert summary["grad_accum_steps"] == 4
    assert summary["last_step"] == 2
    assert summary["tokens"] == 128
    assert summary["last_train_loss"] == 3.5
    assert summary["last_next_token_loss"] == 3.2
    assert summary["last_mtp_loss"] == 3.0
    assert summary["last_val_loss"] == 3.9
    assert summary["val_ppl"] == math.exp(3.9)
    assert summary["resumes"] == 1
    assert summary["checkpoint_count"] == 2
    assert summary["latest_checkpoint"] is not None
