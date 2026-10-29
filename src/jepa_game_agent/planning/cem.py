"""Cross-entropy method over discrete action sequences (Craftax actions are discrete)."""

from __future__ import annotations

from collections.abc import Callable

import torch

from jepa_game_agent.config import PlannerConfig
from jepa_game_agent.models.world_model import WorldModel

CostFn = Callable[[torch.Tensor], torch.Tensor]  # (N, K) int action sequences -> (N,) cost, lower is better


def categorical_cem(
    cost_fn: CostFn,
    num_actions: int,
    cfg: PlannerConfig,
    device: str | torch.device = "cpu",
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Keep one categorical distribution per step; sample, keep the cheapest, move the distribution to them.

    init (A,): starting distribution for every step (default uniform), e.g. an action prior.
    Returns the cheapest sequence seen (K,) and its cost.
    """
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init is None:
        probs = torch.full((cfg.horizon, num_actions), 1.0 / num_actions, device=device)
    elif init.shape != (num_actions,):
        raise ValueError(f"init must have shape ({num_actions},), got {tuple(init.shape)}")
    else:
        probs = init.to(device).float().expand(cfg.horizon, -1).clone()
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


def plan_to_goal(
    model: WorldModel,
    now: torch.Tensor,
    goal: torch.Tensor,
    cfg: PlannerConfig,
    weights: torch.Tensor | None = None,
    prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Action sequence (K,) whose imagined latents get closest to the goal latent soonest. now, goal: (D,).

    Cost = distance to the goal summed over every imagined step (not only the last): agents play just the
    first action and plan again, so a plan that detours and comes back would waste that action.
    weights (D,): per-dimension cost scale, e.g. 0 on the view and 1 on the inventory part of a HUD latent.
    prior (A,): where CEM starts sampling (default uniform), so actions players use here get tried first.
    """

    def cost(seqs: torch.Tensor) -> torch.Tensor:
        path = model.rollout(now.expand(len(seqs), -1), seqs)  # (N, K, D)
        sq = (path - goal).pow(2)
        return (sq if weights is None else sq * weights).sum(-1).sum(-1)

    seq, _ = categorical_cem(cost, model.cfg.num_actions, cfg, device=now.device, init=prior)
    return seq
