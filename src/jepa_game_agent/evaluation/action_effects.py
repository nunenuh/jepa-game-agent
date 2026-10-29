"""Does the world model know what an action does? For recorded transitions (z_t, a_t, z_t+1), predict one step
ahead for every action and rank the action really taken (1 = best):

- next rank: by distance of each prediction to the real next latent (does the model know the action's effect?)
- goal rank: by distance to a goal latent (would the planner pick that action when chasing that goal?)

Ties count in the action's favour: rank = 1 + number of actions strictly closer."""

from __future__ import annotations

from typing import Any

import torch

from jepa_game_agent.models.world_model import WorldModel


def _ranks(pred: torch.Tensor, target: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """pred (N, A, D), target (N, D) or (D,) -> rank of actions[i] per row (N,)."""
    d = (pred - target.unsqueeze(-2)).pow(2).sum(-1)  # (N, A)
    mine = d.gather(1, actions.unsqueeze(1))
    return (d < mine).sum(1) + 1


@torch.no_grad()
def rank_actions(
    model: WorldModel,
    z_now: torch.Tensor,
    z_next: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor | None = None,
) -> dict[str, Any]:
    """z_now, z_next (N, D), actions (N,) int, goal (D,) optional."""
    n, a = len(z_now), model.cfg.num_actions
    every = torch.arange(a, device=z_now.device).repeat(n).unsqueeze(1)  # (N*A, 1)
    pred = model.rollout(z_now.repeat_interleave(a, 0), every)[:, 0].reshape(n, a, -1)
    nxt = _ranks(pred, z_next, actions)
    out: dict[str, Any] = {
        "n": n,
        "next_rank": nxt.tolist(),
        "mean_next_rank": nxt.float().mean().item(),
        "next_top1": (nxt == 1).float().mean().item(),
        "goal_rank": None,
    }
    if goal is not None:
        g = _ranks(pred, goal.expand(n, -1), actions)
        out |= {"goal_rank": g.tolist(), "mean_goal_rank": g.float().mean().item(),
                "goal_top1": (g == 1).float().mean().item()}  # fmt: skip
    return out
