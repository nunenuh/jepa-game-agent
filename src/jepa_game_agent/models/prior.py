"""Action prior: which actions players press in a state (a latent), learned from recorded games.

Seeds the planner's sampling, so rare but decisive actions (place a table, craft) get tried where players
used them; the world model still decides which plan reaches the goal."""

from __future__ import annotations

import torch
from torch import nn


class ActionPrior(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.latent_dim, self.num_actions, self.hidden = latent_dim, num_actions, hidden
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, num_actions))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., D) latents -> (..., A) action logits."""
        out: torch.Tensor = self.net(z)
        return out

    @torch.no_grad()
    def probs(self, z: torch.Tensor, mix: float = 0.0) -> torch.Tensor:
        """(..., A) probabilities, blended with uniform by `mix` so no action is ruled out."""
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        p: torch.Tensor = (1 - mix) * self(z).softmax(-1) + mix / self.num_actions
        return p
