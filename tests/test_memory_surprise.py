import math

import pytest
import torch

from jepa_game_agent.memory import LatentMemory
from jepa_game_agent.surprise import SurpriseMeter


def test_memory_nearest_and_ring_buffer() -> None:
    mem = LatentMemory(dim=3, capacity=2)
    assert mem.nearest(torch.ones(3)) == []
    mem.add(torch.tensor([1.0, 0, 0]), step=0, info={"water": 1.0})
    mem.add(torch.tensor([0, 1.0, 0]), step=1)
    sim, m = mem.nearest(torch.tensor([0.9, 0.1, 0]))[0]
    assert m.step == 0 and m.info["water"] == 1.0 and sim > 0.9
    mem.add(torch.tensor([0, 0, 1.0]), step=2)  # overwrites step 0
    assert len(mem) == 2 and {m.step for _, m in mem.nearest(torch.ones(3), k=5)} == {1, 2}
    with pytest.raises(ValueError):
        mem.add(torch.ones(4), step=3)


def test_surprise_meter_flags_outliers() -> None:
    meter = SurpriseMeter(decay=0.9, warmup=10)
    gen = torch.Generator().manual_seed(0)
    scores = [meter.update(0.02 + 0.002 * torch.randn(1, generator=gen).item()) for _ in range(50)]
    assert all(s == 0.0 for s in scores[:10]) and max(abs(s) for s in scores[10:]) < 6
    assert meter.update(0.3) > 10  # a teleport-sized error
    with pytest.raises(ValueError):
        meter.update(math.nan)
    with pytest.raises(ValueError):
        SurpriseMeter(decay=1.0)
