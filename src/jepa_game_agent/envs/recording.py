"""An env wrapper that keeps the first worlds it plays as episodes (to watch them later)."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.base import Frame, GameEnv, StepResult


class RecordingEnv:
    """Passes everything through to `env`; records frames and actions of the first `limit` worlds."""

    def __init__(self, env: GameEnv, limit: int) -> None:
        self.env, self.limit = env, limit
        self._done: list[Episode] = []
        self._now: tuple[int, list[Frame], list[int], list[str]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, seed: int | None = None) -> Frame:
        self._close()
        frame = self.env.reset(seed)
        if len(self._done) < self.limit:
            self._now = (-1 if seed is None else seed, [frame], [], [])
        return frame

    def step(self, action: int) -> StepResult:
        res = self.env.step(action)
        if self._now is not None:
            _, frames, actions, got = self._now
            frames.append(res.frame)
            actions.append(action)
            got += [a for a in res.achievements if a not in got]
        return res

    def _close(self) -> None:
        if self._now is not None:
            seed, frames, actions, got = self._now
            n = len(actions)
            self._done.append(Episode(np.stack(frames), np.array(actions, np.int64), np.zeros(n, np.float32),
                                      seed=seed, achievements=tuple(got)))  # fmt: skip
            self._now = None

    def episodes(self) -> list[Episode]:
        """The recorded worlds so far, the one being played included."""
        self._close()
        return list(self._done)
