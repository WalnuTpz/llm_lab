from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from llm_lab.config.schema import ExperimentConfig, experiment_from_dict


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = load_mapping(config_path)
    return experiment_from_dict(raw)


def load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError(f"unsupported config suffix: {config_path.suffix}")
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError("config file must contain a mapping at the top level")
    return loaded
