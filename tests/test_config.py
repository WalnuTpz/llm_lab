from __future__ import annotations

from pathlib import Path

from llm_lab.config import load_config


CONFIG_DIR = Path(__file__).parent / "fixtures" / "configs"
FORMAL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_load_all_smoke_configs():
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.name
        assert cfg.model.vocab_size == 64
        assert cfg.training.device == "cpu"


def test_load_all_formal_configs():
    for path in FORMAL_CONFIG_DIR.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.name
        assert cfg.model.vocab_size == 16_384
