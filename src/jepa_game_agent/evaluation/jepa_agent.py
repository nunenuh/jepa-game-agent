"""The JEPA agent: plan toward a goal picture by imagining futures in latent space (model-predictive control)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.envs.base import Frame
from jepa_game_agent.models.world_model import WorldModel
from jepa_game_agent.planning.cem import plan_to_goal


class JepaAgent:
    """Each step: encode the frame, plan `cfg.horizon` actions whose imagined end is nearest the goal latent,
    play the first one, plan again next step.

    hud_rows > 0: the goal is "this view, with the goal frames' bottom hud_rows pixel rows" (Craftax's inventory
    bar), rebuilt every step, so the goal asks for an inventory change instead of a remembered place.
    weights (D,): per-latent-dimension cost scale passed to the planner (None = all 1).
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: PlannerConfig,
        device: str | torch.device = "cpu",
        hud_rows: int = 0,
        weights: torch.Tensor | None = None,
    ) -> None:
        if hud_rows < 0:
            raise ValueError("hud_rows must be >= 0")
        self.model, self.cfg, self.device, self.hud_rows = model.eval().to(device), cfg, device, hud_rows
        self.goal: torch.Tensor | None = None
        self._goal_frames: np.ndarray | None = None
        self.weights = None if weights is None else weights.to(device)

    def _encode(self, frames: Sequence[Frame]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return self.model.encode(x)

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        if len(goal_frames) == 0:
            raise ValueError("need at least one goal frame")
        self.goal = self._encode(goal_frames).mean(0)
        self._goal_frames = np.stack(goal_frames)

    def _see(self, frame: Frame) -> torch.Tensor:
        """Encode the frame (and rebuild a HUD goal around it); returns its latent."""
        if self.goal is None or self._goal_frames is None:
            raise RuntimeError("call reset(goal_frames) before act")
        if self.hud_rows:
            if self.hud_rows >= frame.shape[0]:
                raise ValueError(f"hud_rows {self.hud_rows} must be below the frame height {frame.shape[0]}")
            mixed = self._goal_frames.copy()
            mixed[:, : -self.hud_rows] = frame[: -self.hud_rows]
            self.goal = self._encode(list(mixed)).mean(0)
        return self._encode([frame])[0]

    def act(self, frame: Frame) -> int:
        now = self._see(frame)
        assert self.goal is not None
        return int(plan_to_goal(self.model, now, self.goal, self.cfg, self.weights)[0])
