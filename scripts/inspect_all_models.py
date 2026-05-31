from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from llm_lab.config import load_config
from llm_lab.models import build_model
from llm_lab.utils import dtype_from_str, parameter_report, resolve_device, safe_dtype_for_device


DEFAULT_CONFIGS = [
    "configs/original_transformer.yaml",
    "configs/modern_decoder.yaml",
    "configs/qwen36.yaml",
    "configs/deepseek_v4.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all configured formal models.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--device", default="meta", help="Use meta for allocation-free parameter inspection.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--format", choices=["table", "jsonl"], default="table")
    args = parser.parse_args()

    rows = [_inspect_config(Path(path), device_name=args.device, dtype_name=args.dtype) for path in args.configs]
    if args.format == "jsonl":
        for row in rows:
            print(_json_dumps(row))
    else:
        _print_table(rows)


def _inspect_config(path: Path, *, device_name: str, dtype_name: str) -> dict[str, object]:
    cfg = load_config(path)
    device = torch.device("meta") if device_name == "meta" else resolve_device(device_name)
    dtype = dtype_from_str(dtype_name)
    dtype = dtype if device.type == "meta" else safe_dtype_for_device(dtype, device)
    model = build_model(cfg.model, device=device, dtype=dtype)
    active = model.active_parameters_per_token() if hasattr(model, "active_parameters_per_token") else None
    report = parameter_report(model, active_parameters_per_token=active)
    return {
        "config": str(path),
        "name": cfg.name,
        "architecture": cfg.model.architecture,
        "device": str(device),
        "dtype": str(dtype),
        **asdict(report),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        "name",
        "architecture",
        "total_parameters",
        "active_parameters_per_token",
        "embedding_parameters",
        "non_embedding_parameters",
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def _json_dumps(row: dict[str, object]) -> str:
    import json

    return json.dumps(row, sort_keys=True)


if __name__ == "__main__":
    main()
