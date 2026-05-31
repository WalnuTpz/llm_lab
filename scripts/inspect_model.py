from __future__ import annotations

import argparse
from dataclasses import asdict

from llm_lab.config import load_config
from llm_lab.models import build_model
from llm_lab.utils import dtype_from_str, parameter_report, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a configured model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    dtype = dtype_from_str(args.dtype or cfg.training.dtype)
    dtype = safe_dtype_for_device(dtype, device)
    model = build_model(cfg.model, device=device, dtype=dtype)
    active = model.active_parameters_per_token() if hasattr(model, "active_parameters_per_token") else None
    report = parameter_report(model, active_parameters_per_token=active)

    print(f"name: {cfg.name}")
    print(f"architecture: {cfg.model.architecture}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    for key, value in asdict(report).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
