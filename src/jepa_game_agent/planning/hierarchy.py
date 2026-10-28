"""Top-level planning: categorical CEM over action chunks with the top model; its first imagined latent is the
subgoal the bottom planner walks to."""

from __future__ import annotations

import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.models.top import TopModel
from jepa_game_agent.planning.cem import categorical_cem


def plan_chunks_to_goal(
    top: TopModel, now: torch.Tensor, goal: torch.Tensor, cfg: PlannerConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """cfg.horizon chunks of top.cfg.stride actions whose imagined path gets closest to the goal soonest.

    Returns the flat action sequence (horizon * stride,) and the subgoal: the latent after the first chunk (D,).
    """
    stride = top.cfg.stride

    def cost(seqs: torch.Tensor) -> torch.Tensor:
        path = top.rollout(now.expand(len(seqs), -1), seqs.view(len(seqs), cfg.horizon, stride))
        return (path - goal).pow(2).sum(-1).sum(-1)

    flat = PlannerConfig(cfg.horizon * stride, cfg.samples, cfg.elites, cfg.iterations, cfg.smoothing, cfg.seed)
    seq, _ = categorical_cem(cost, top.cfg.num_actions, flat, device=now.device)
    subgoal = top.rollout(now.unsqueeze(0), seq[:stride].view(1, 1, stride))[0, 0]
    return seq, subgoal
