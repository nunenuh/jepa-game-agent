"""Body sense: read the player's vitals (health, food, drink, energy) off a latent.

The world model sees these only as icons in the status bar; a small head trained on labels from replayed
games tells the planner how an imagined future would leave the player (for a survival cost).
With outputs=1 the same head is the critic: it predicts the health about to be lost (danger before pain)."""

from __future__ import annotations

import torch
from torch import nn

VITALS = ("health", "food", "drink", "energy")


class Readout(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 256, outputs: int = len(VITALS)) -> None:
        super().__init__()
        self.latent_dim, self.hidden, self.outputs = latent_dim, hidden, outputs
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, outputs))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., D) latents -> (..., 4) vitals in VITALS order (Craftax scale 0-9), or (..., 1) danger."""
        out: torch.Tensor = self.net(z)
        return out
