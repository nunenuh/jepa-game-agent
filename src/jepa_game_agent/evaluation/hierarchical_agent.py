"""Two-level agent: every `stride` steps the top plans toward the goal and sets a subgoal one chunk ahead;
in between, the bottom planner (the flat JepaAgent's) walks to that subgoal. A subgoal no closer to the goal
than where the agent stands is no progress, so then the bottom heads for the goal itself."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.envs.base import Frame
from jepa_game_agent.evaluation.jepa_agent import JepaAgent
from jepa_game_agent.models.top import TopModel
from jepa_game_agent.models.world_model import WorldModel
from jepa_game_agent.planning.cem import plan_to_goal
from jepa_game_agent.planning.hierarchy import plan_chunks_to_goal


class HierarchicalAgent(JepaAgent):
    def __init__(
        self,
        model: WorldModel,
        top: TopModel,
        top_cfg: PlannerConfig,
        bottom_cfg: PlannerConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model, bottom_cfg, device)
        self.top, self.top_cfg = top.eval().to(device), top_cfg
        self.subgoal: torch.Tensor | None = None
        self.top_plans, self._t = 0, 0

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        super().reset(goal_frames)
        self.subgoal, self.top_plans, self._t = None, 0, 0

    def act(self, frame: Frame) -> int:
        if self.goal is None:
            raise RuntimeError("call reset(goal_frames) before act")
        now = self._encode([frame])[0]
        if self._t % self.top.cfg.stride == 0:
            _, sub = plan_chunks_to_goal(self.top, now, self.goal, self.top_cfg)
            progress = (sub - self.goal).norm() < (now - self.goal).norm()
            self.subgoal = sub if progress else self.goal
            self.top_plans += 1
        self._t += 1
        assert self.subgoal is not None
        return int(plan_to_goal(self.model, now, self.subgoal, self.cfg)[0])
