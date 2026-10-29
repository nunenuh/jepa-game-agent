"""Body sense: read the player's vitals (health, food, drink, energy) off a latent.

The world model sees these only as icons in the status bar; a small head trained on labels from replayed
games tells the planner how an imagined future would leave the player (for a survival cost)."""

from __future__ import annotations

import torch
from torch import nn

VITALS = ("health", "food", "drink", "energy")


class Readout(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.latent_dim, self.hidden = latent_dim, hidden
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, len(VITALS)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., D) latents -> (..., 4) vitals in VITALS order (Craftax scale 0-9)."""
        out: torch.Tensor = self.net(z)
        return out
