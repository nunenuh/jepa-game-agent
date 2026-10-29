"""The JEPA agent: plan toward a goal picture by imagining futures in latent space (model-predictive control)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.envs.base import Frame
from jepa_game_agent.models.prior import ActionPrior
from jepa_game_agent.models.world_model import WorldModel
from jepa_game_agent.planning.cem import plan_to_goal


class JepaAgent:
    """Each step: encode the frame, plan `cfg.horizon` actions whose imagined end is nearest the goal latent,
    play the first one, plan again next step.

    hud_rows > 0: the goal is "this view, with the goal frames' bottom hud_rows pixel rows" (Craftax's inventory
    bar), rebuilt every step, so the goal asks for an inventory change instead of a remembered place.
    weights (D,): per-latent-dimension cost scale passed to the planner (None = all 1).
    prior: action prior for the current latent, blended with uniform by prior_mix, where CEM starts sampling.
    goal_as="change": goal frames are (2, H, W, C) before/after pairs; the goal is "this latent plus their mean
    change", rebuilt every step, so it asks for what the achievement changes, not for a place.
    goal_as="both": travel toward the picture (the pairs' after frames), but each step first plan for the change;
    if that plan gets within reach * |change|^2 of it, play it (the change is doable from here).
    goal_as="blend": goal = the mean of the picture and the change goal (same as adding the two costs).
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: PlannerConfig,
        device: str | torch.device = "cpu",
        hud_rows: int = 0,
        weights: torch.Tensor | None = None,
        prior: ActionPrior | None = None,
        prior_mix: float = 0.1,
        goal_as: str = "picture",
        reach: float = 0.5,
    ) -> None:
        if hud_rows < 0:
            raise ValueError("hud_rows must be >= 0")
        if goal_as not in ("picture", "change", "both", "blend"):
            raise ValueError(f"goal_as must be picture, change, both or blend, got {goal_as!r}")
        if goal_as != "picture" and hud_rows:
            raise ValueError(f"goal_as={goal_as!r} and hud_rows can't be combined")
        if not 0.0 < reach <= 1.0:
            raise ValueError("reach must be in (0, 1]")
        self.goal_as, self.reach = goal_as, reach
        self._delta: torch.Tensor | None = None
        self._picture: torch.Tensor | None = None
        self.change_goal: torch.Tensor | None = None
        self.model, self.cfg, self.device, self.hud_rows = model.eval().to(device), cfg, device, hud_rows
        self.goal: torch.Tensor | None = None
        self._goal_frames: np.ndarray | None = None
        self.weights = None if weights is None else weights.to(device)
        self.prior = None if prior is None else prior.eval().to(device)
        self.prior_mix = prior_mix

    def _prior(self, now: torch.Tensor) -> torch.Tensor | None:
        return None if self.prior is None else self.prior.probs(now, self.prior_mix)

    def _encode(self, frames: Sequence[Frame]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return self.model.encode(x)

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        if len(goal_frames) == 0:
            raise ValueError("need at least one goal frame")
        if self.goal_as != "picture":
            pairs = np.stack(goal_frames)
            if pairs.ndim != 5 or pairs.shape[1] != 2:
                raise ValueError(f"goal_as={self.goal_as!r} needs (2, H, W, C) before/after pairs")
            after = self._encode(list(pairs[:, 1]))
            self._delta = (after - self._encode(list(pairs[:, 0]))).mean(0)
            self._picture = after.mean(0)
            self.goal = self._delta if self.goal_as == "change" else self._picture  # change: set per step
            self._goal_frames = pairs[:, 1]
            return
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
        now = self._encode([frame])[0]
        if self._delta is not None:
            self.change_goal = now + self._delta
            if self.goal_as == "change":
                self.goal = self.change_goal
            elif self.goal_as == "blend":
                assert self._picture is not None
                self.goal = (self._picture + self.change_goal) / 2
        return now

    def _sq(self, x: torch.Tensor) -> torch.Tensor:
        return (x.pow(2) if self.weights is None else x.pow(2) * self.weights).sum(-1)

    def _change_plan(self, now: torch.Tensor) -> torch.Tensor | None:
        """goal_as='both': a plan for the change if it gets within reach * |change|^2 of it, else None."""
        if self.goal_as != "both" or self.change_goal is None or self._delta is None:
            return None
        seq = plan_to_goal(self.model, now, self.change_goal, self.cfg, self.weights, self._prior(now))
        with torch.no_grad():
            path = self.model.rollout(now.unsqueeze(0), seq.unsqueeze(0))[0]
        best = self._sq(path - self.change_goal).min()
        return seq if best <= self.reach * self._sq(self._delta) else None

    def act(self, frame: Frame) -> int:
        now = self._see(frame)
        assert self.goal is not None
        change = self._change_plan(now)
        if change is not None:
            return int(change[0])
        return int(plan_to_goal(self.model, now, self.goal, self.cfg, self.weights, self._prior(now))[0])
