from __future__ import annotations

import numpy as np
import torch

from llm_lab.data import get_batch_with_future_targets


def test_get_batch_with_future_targets_offsets_targets():
    torch.manual_seed(0)
    dataset = np.arange(128, dtype=np.int64)
    x, targets = get_batch_with_future_targets(
        dataset,
        batch_size=4,
        context_length=8,
        device="cpu",
        num_targets=3,
    )

    assert x.shape == (4, 8)
    assert len(targets) == 3
    for offset, target in enumerate(targets, start=1):
        assert target.shape == (4, 8)
        torch.testing.assert_close(target, x + offset)
