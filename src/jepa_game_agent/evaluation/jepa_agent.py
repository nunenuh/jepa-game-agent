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
    play the first one, plan again next step."""

    def __init__(self, model: WorldModel, cfg: PlannerConfig, device: str | torch.device = "cpu") -> None:
        self.model, self.cfg, self.device = model.eval().to(device), cfg, device
        self.goal: torch.Tensor | None = None

    def _encode(self, frames: Sequence[Frame]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return self.model.encode(x)

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        if len(goal_frames) == 0:
            raise ValueError("need at least one goal frame")
        self.goal = self._encode(goal_frames).mean(0)

    def act(self, frame: Frame) -> int:
        if self.goal is None:
            raise RuntimeError("call reset(goal_frames) before act")
        now = self._encode([frame])[0]
        return int(plan_to_goal(self.model, now, self.goal, self.cfg)[0])
