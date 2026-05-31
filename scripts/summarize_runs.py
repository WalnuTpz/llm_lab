from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLM Lab metrics.jsonl runs.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--format", choices=["table", "jsonl"], default="table")
    args = parser.parse_args()

    summaries = summarize_runs(Path(args.runs_root), checkpoints_root=Path(args.checkpoints_root))
    if args.format == "jsonl":
        for summary in summaries:
            print(json.dumps(summary, sort_keys=True))
    else:
        _print_table(summaries)


def summarize_runs(runs_root: Path, *, checkpoints_root: Path | None = None) -> list[dict[str, Any]]:
    metrics_files = sorted(runs_root.rglob("metrics.jsonl")) if runs_root.exists() else []
    summaries = []
    for metrics_path in metrics_files:
        records = _read_metrics(metrics_path)
        if not records:
            continue
        run_record = next((record for record in records if record.get("type") == "run"), {})
        train_records = [record for record in records if record.get("type") == "train"]
        eval_records = [record for record in records if record.get("type") == "eval"]
        resume_records = [record for record in records if record.get("type") == "resume"]
        last_train = train_records[-1] if train_records else {}
        last_eval = eval_records[-1] if eval_records else {}
        architecture = run_record.get("architecture") or metrics_path.parent.name
        checkpoint_info = _checkpoint_info(architecture, checkpoints_root)
        summaries.append(
            {
                "run_dir": str(metrics_path.parent),
                "name": run_record.get("name", metrics_path.parent.name),
                "architecture": architecture,
                "git_commit": run_record.get("git_commit"),
                "total_parameters": run_record.get("total_parameters"),
                "active_parameters_per_token": run_record.get("active_parameters_per_token"),
                "batch_size": run_record.get("batch_size"),
                "context_length": run_record.get("context_length"),
                "grad_accum_steps": run_record.get("grad_accum_steps"),
                "last_step": last_train.get("step"),
                "last_train_loss": last_train.get("loss"),
                "last_val_loss": last_eval.get("val_loss"),
                "tokens": last_train.get("tokens"),
                "tokens_per_second": last_train.get("tokens_per_second"),
                "resumes": len(resume_records),
                **checkpoint_info,
            }
        )
    return summaries


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metrics JSON at {path}:{line_number}") from exc
    return records


def _checkpoint_info(architecture: object, checkpoints_root: Path | None) -> dict[str, object]:
    if checkpoints_root is None or architecture is None:
        return {"latest_checkpoint": None, "checkpoint_count": 0}
    checkpoint_dir = checkpoints_root / str(architecture)
    if not checkpoint_dir.exists():
        return {"latest_checkpoint": None, "checkpoint_count": 0}
    step_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    latest = checkpoint_dir / "latest.pt"
    return {
        "latest_checkpoint": str(latest) if latest.exists() else None,
        "checkpoint_count": len(step_checkpoints),
    }


def _print_table(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        print("no runs found")
        return
    columns = [
        "name",
        "architecture",
        "last_step",
        "tokens",
        "last_train_loss",
        "last_val_loss",
        "tokens_per_second",
        "checkpoint_count",
    ]
    widths = {
        column: max(len(column), *(len(_format_cell(row.get(column))) for row in summaries))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in summaries:
        print("  ".join(_format_cell(row.get(column)).ljust(widths[column]) for column in columns))


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
