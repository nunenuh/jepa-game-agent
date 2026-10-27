"""The agent interface and the baselines every result is compared with."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from jepa_game_agent.envs.base import Frame

NOOP = 0  # action 0 is "do nothing" in GridWorld and in Craftax


class Agent(Protocol):
    def reset(self, goal_frames: Sequence[Frame]) -> None:
        """A new task starts; goal_frames show what success looks like (one frame, or several to average)."""
        ...

    def act(self, frame: Frame) -> int: ...


class StillAgent:
    """Does nothing: succeeds only if the goal is already reached."""

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        pass

    def act(self, frame: Frame) -> int:
        return NOOP


class RandomAgent:
    def __init__(self, num_actions: int, seed: int = 0) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        self.num_actions, self._rng = num_actions, np.random.default_rng(seed)

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        pass

    def act(self, frame: Frame) -> int:
        return int(self._rng.integers(self.num_actions))


class ReplayAgent:
    """Plays the recorded actions open loop, then does nothing: the upper bound for reach tasks."""

    def __init__(self, actions: Sequence[int]) -> None:
        self.actions, self._t = list(actions), 0

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        self._t = 0

    def act(self, frame: Frame) -> int:
        a = self.actions[self._t] if self._t < len(self.actions) else NOOP
        self._t += 1
        return a
