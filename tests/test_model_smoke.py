from __future__ import annotations

from pathlib import Path

import torch

from llm_lab.config import load_config
from llm_lab.models import build_model
from llm_lab.training import cross_entropy
from llm_lab.utils import parameter_report


CONFIG_DIR = Path(__file__).parent / "fixtures" / "configs"


def test_all_models_forward_backward_and_parameter_report():
    torch.manual_seed(0)
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(path)
        model = build_model(cfg.model)
        x = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.context_length))
        y = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.context_length))
        logits = model(x)
        assert logits.shape == (2, cfg.model.context_length, cfg.model.vocab_size)
        loss = cross_entropy(logits, y)
        loss.backward()
        active = model.active_parameters_per_token() if hasattr(model, "active_parameters_per_token") else None
        report = parameter_report(model, active_parameters_per_token=active)
        assert report.total_parameters > 0
        assert report.trainable_parameters == report.total_parameters
        assert report.active_parameters_per_token is not None
