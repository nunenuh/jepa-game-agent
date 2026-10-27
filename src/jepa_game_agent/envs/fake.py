"""A tiny deterministic grid world with pixel frames, for tests and for debugging the pipeline on a CPU."""

from __future__ import annotations

import numpy as np

from jepa_game_agent.envs.base import Frame, StepResult

MOVES = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # NOOP, LEFT, RIGHT, UP, DOWN
AGENT, ITEM = (255, 200, 0), (0, 160, 255)


class GridWorld:
    """The agent (yellow) walks on a size x size grid; reaching the item (blue) gives reward 1 and moves it."""

    num_actions = len(MOVES)

    def __init__(self, size: int = 8, cell: int = 4, max_steps: int = 100, seed: int = 0) -> None:
        if size < 2 or cell < 1 or max_steps < 1:
            raise ValueError("size >= 2, cell >= 1 and max_steps >= 1 required")
        self.size, self.cell, self.max_steps = size, cell, max_steps
        self.frame_shape = (size * cell, size * cell, 3)
        self._rng = np.random.default_rng(seed)
        self.agent = (0, 0)
        self.item = (size - 1, size - 1)
        self.t = 0

    def reset(self, seed: int | None = None) -> Frame:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent = self._random_cell()
        self.item = self._random_cell(avoid=self.agent)
        self.t = 0
        return self.render()

    def step(self, action: int) -> StepResult:
        if action not in MOVES:
            raise ValueError(f"action must be in 0..{self.num_actions - 1}, got {action}")
        dr, dc = MOVES[action]
        r, c = self.agent
        self.agent = (min(max(r + dr, 0), self.size - 1), min(max(c + dc, 0), self.size - 1))
        reward = 0.0
        if self.agent == self.item:
            reward = 1.0
            self.item = self._random_cell(avoid=self.agent)
        self.t += 1
        return StepResult(
            self.render(),
            reward,
            self.t >= self.max_steps,
            {"agent_row": self.agent[0], "agent_col": self.agent[1]},
            ("collect_item",) if reward else (),
        )

    def state_key(self) -> tuple[int, ...]:
        return self.agent

    def set_agent(self, row: int, col: int) -> Frame:
        """Teleport the agent (an 'impossible' event, for surprise tests)."""
        self.agent = (row, col)
        return self.render()

    def render(self) -> Frame:
        frame = np.zeros(self.frame_shape, dtype=np.uint8)
        for (r, c), colour in ((self.item, ITEM), (self.agent, AGENT)):
            frame[r * self.cell : (r + 1) * self.cell, c * self.cell : (c + 1) * self.cell] = colour
        return frame

    def _random_cell(self, avoid: tuple[int, int] | None = None) -> tuple[int, int]:
        while True:
            cell = (int(self._rng.integers(self.size)), int(self._rng.integers(self.size)))
            if cell != avoid:
                return cell
