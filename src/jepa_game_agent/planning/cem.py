"""Cross-entropy method over discrete action sequences (Craftax actions are discrete)."""

from __future__ import annotations

from collections.abc import Callable

import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.models.world_model import WorldModel

CostFn = Callable[[torch.Tensor], torch.Tensor]  # (N, K) int action sequences -> (N,) cost, lower is better


def categorical_cem(
    cost_fn: CostFn, num_actions: int, cfg: PlannerConfig, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, float]:
    """Keep one categorical distribution per step; sample, keep the cheapest, move the distribution to them.

    Returns the cheapest sequence seen (K,) and its cost.
    """
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    probs = torch.full((cfg.horizon, num_actions), 1.0 / num_actions, device=device)
    best_seq, best_cost = torch.zeros(cfg.horizon, dtype=torch.long, device=device), float("inf")
    for _ in range(cfg.iterations):
        seqs = torch.multinomial(probs, cfg.samples, replacement=True, generator=gen).T  # (N, K)
        costs = cost_fn(seqs)
        if costs.shape != (cfg.samples,):
            raise ValueError(f"cost_fn must return shape ({cfg.samples},), got {tuple(costs.shape)}")
        order = costs.argsort()
        if costs[order[0]].item() < best_cost:
            best_cost, best_seq = costs[order[0]].item(), seqs[order[0]].clone()
        elites = seqs[order[: cfg.elites]]
        freq = torch.nn.functional.one_hot(elites, num_actions).float().mean(0)  # (K, A)
        probs = (1 - cfg.smoothing) * probs + cfg.smoothing * freq
    return best_seq, best_cost


def plan_to_goal(model: WorldModel, now: torch.Tensor, goal: torch.Tensor, cfg: PlannerConfig) -> torch.Tensor:
    """Action sequence (K,) whose imagined final latent is closest to the goal latent. now, goal: (D,)."""

    def cost(seqs: torch.Tensor) -> torch.Tensor:
        final = model.rollout(now.expand(len(seqs), -1), seqs)[:, -1]
        return (final - goal).pow(2).sum(-1)

    seq, _ = categorical_cem(cost, model.cfg.num_actions, cfg, device=now.device)
    return seq
