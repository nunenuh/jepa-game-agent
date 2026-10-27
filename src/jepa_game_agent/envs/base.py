"""The interface every game world implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]  # (H, W, C), 0-255


@dataclass(frozen=True)
class StepResult:
    frame: Frame
    reward: float
    done: bool
    info: dict[str, float] = field(default_factory=dict)


class GameEnv(Protocol):
    num_actions: int
    frame_shape: tuple[int, int, int]

    def reset(self, seed: int | None = None) -> Frame: ...

    def step(self, action: int) -> StepResult: ...
