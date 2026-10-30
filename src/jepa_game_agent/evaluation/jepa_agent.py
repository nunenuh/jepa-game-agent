"""The JEPA agent: plan toward a goal picture by imagining futures in latent space (model-predictive control)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.envs.base import Frame
from jepa_game_agent.models.prior import ActionPrior
from jepa_game_agent.models.readout import Readout
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
    travel: which picture both/blend travel to, "after" (how having it looks) or "before" (where the change
    starts, e.g. facing a tree with enough wood).
    readout + survival: add survival * (health lost vs now, summed over the imagined path) to every plan's
    cost, health read off the latents by `readout` (LeCun's built-in "pain" cost).
    critic: a second world model (e.g. one that sees the status bar) that imagines the same action sequences;
    the readout then reads health off its latents instead of the planning model's.
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
        travel: str = "after",
        readout: Readout | None = None,
        survival: float = 0.0,
        critic: WorldModel | None = None,
    ) -> None:
        if hud_rows < 0:
            raise ValueError("hud_rows must be >= 0")
        if goal_as not in ("picture", "change", "both", "blend"):
            raise ValueError(f"goal_as must be picture, change, both or blend, got {goal_as!r}")
        if goal_as != "picture" and hud_rows:
            raise ValueError(f"goal_as={goal_as!r} and hud_rows can't be combined")
        if not 0.0 < reach <= 1.0:
            raise ValueError("reach must be in (0, 1]")
        if travel not in ("after", "before"):
            raise ValueError(f"travel must be 'after' or 'before', got {travel!r}")
        self.travel = travel
        if survival < 0 or (survival > 0 and readout is None):
            raise ValueError("survival must be >= 0 and needs a readout")
        self.readout = None if readout is None else readout.eval().to(device)
        self.survival = survival
        if critic is not None and readout is None:
            raise ValueError("a critic is only read through a readout")
        self.critic = None if critic is None else critic.eval().to(device)
        self._critic_now: torch.Tensor | None = None
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

    def _frames_to_tensor(self, frames: Sequence[Frame]) -> torch.Tensor:
        return torch.from_numpy(np.stack(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255

    def _encode(self, frames: Sequence[Frame]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(self._frames_to_tensor(frames))

    def reset(self, goal_frames: Sequence[Frame]) -> None:
        if len(goal_frames) == 0:
            raise ValueError("need at least one goal frame")
        if self.goal_as != "picture":
            pairs = np.stack(goal_frames)
            if pairs.ndim != 5 or pairs.shape[1] != 2:
                raise ValueError(f"goal_as={self.goal_as!r} needs (2, H, W, C) before/after pairs")
            before, after = self._encode(list(pairs[:, 0])), self._encode(list(pairs[:, 1]))
            self._delta = (after - before).mean(0)
            self._picture = (before if self.travel == "before" else after).mean(0)
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
        if self.critic is not None:
            with torch.no_grad():
                self._critic_now = self.critic.encode(self._frames_to_tensor([frame]))[0]
        if self._delta is not None:
            self.change_goal = now + self._delta
            if self.goal_as == "change":
                self.goal = self.change_goal
            elif self.goal_as == "blend":
                assert self._picture is not None
                self.goal = (self._picture + self.change_goal) / 2
        return now

    def _extra(self, now: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
        """Survival cost for plans from `now`: health lost along the imagined path (0 if it doesn't drop).
        With a critic, the path is re-imagined by the critic from its own latent of the current frame."""
        if self.readout is None or self.survival == 0:
            return None
        head, w, critic, c_now = self.readout, self.survival, self.critic, self._critic_now
        start = now if critic is None or c_now is None else c_now
        with torch.no_grad():
            h_now = head(start)[0]

        def cost(path: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                if critic is not None and c_now is not None:
                    path = critic.rollout(c_now.expand(len(seqs), -1), seqs)
                lost: torch.Tensor = w * (h_now - head(path)[..., 0]).clamp(min=0).sum(-1)
            return lost

        return cost

    def _sq(self, x: torch.Tensor) -> torch.Tensor:
        return (x.pow(2) if self.weights is None else x.pow(2) * self.weights).sum(-1)

    def _change_plan(self, now: torch.Tensor) -> torch.Tensor | None:
        """goal_as='both': a plan for the change if it gets within reach * |change|^2 of it, else None."""
        if self.goal_as != "both" or self.change_goal is None or self._delta is None:
            return None
        seq = plan_to_goal(
            self.model, now, self.change_goal, self.cfg, self.weights, self._prior(now), self._extra(now)
        )
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
        return int(
            plan_to_goal(self.model, now, self.goal, self.cfg, self.weights, self._prior(now), self._extra(now))[0]
        )
