"""Two-level agent: every `stride` steps the top plans toward the goal and sets a subgoal one chunk ahead;
in between, the bottom planner (the flat JepaAgent's) walks to that subgoal. A subgoal no closer to the goal
than where the agent stands is no progress, so then the bottom heads for the goal itself.

bank (M, D) of real latents (e.g. encoded training frames): each subgoal is snapped to its nearest bank latent,
so the bottom walks toward something that really exists rather than an imagined blur."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.envs.base import Frame
from jepa_game_agent.evaluation.jepa_agent import JepaAgent
from jepa_game_agent.models.prior import ActionPrior
from jepa_game_agent.models.readout import Readout
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
        hud_rows: int = 0,
        bank: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        prior: ActionPrior | None = None,
        prior_mix: float = 0.1,
        goal_as: str = "picture",
        reach: float = 0.5,
        travel: str = "after",
        readout: Readout | None = None,
        survival: float = 0.0,
        readout_model: WorldModel | None = None,
        critic: Readout | None = None,
        danger: float = 0.0,
    ) -> None:
        super().__init__(
            model, bottom_cfg, device, hud_rows, weights, prior, prior_mix, goal_as, reach, travel, readout,
            survival, readout_model, critic, danger,
        )  # fmt: skip
        if bank is not None and (bank.dim() != 2 or bank.size(1) != top.cfg.latent_dim):
            raise ValueError(f"bank must have shape (M, {top.cfg.latent_dim})")
        self.top, self.top_cfg = top.eval().to(device), top_cfg
        self.bank = None if bank is None else bank.to(device)
        self.subgoal: torch.Tensor | None = None
        self.top_plans, self._t = 0, 0

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        super().reset(goal_frames)
        self.subgoal, self.top_plans, self._t = None, 0, 0

    def act(self, frame: Frame) -> int:
        now = self._see(frame)
        assert self.goal is not None
        change = self._change_plan(now)  # goal_as='both': a doable change beats walking on
        if change is not None:
            self._t = 0  # replan the route from wherever the change leaves us
            return int(change[0])
        if self._t % self.top.cfg.stride == 0:
            _, sub = plan_chunks_to_goal(self.top, now, self.goal, self.top_cfg, self.weights)
            if self.bank is not None:
                sub = self.bank[(self.bank - sub).pow(2).sum(-1).argmin()]
            w = 1.0 if self.weights is None else self.weights
            progress = ((sub - self.goal).pow(2) * w).sum() < ((now - self.goal).pow(2) * w).sum()
            self.subgoal = sub if progress else self.goal
            self.top_plans += 1
        self._t += 1
        assert self.subgoal is not None
        return int(
            plan_to_goal(self.model, now, self.subgoal, self.cfg, self.weights, self._prior(now), self._extra(now))[0]
        )
