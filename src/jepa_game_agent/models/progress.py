"""Sense of progress: read the item counts off the status bar in a latent.

Labels (the game's inventory) only train this head; while playing, it reads pixels through the encoder, so the
agent can tell for itself when a stage of the recipe is done. See docs/DESIGN_RECIPE.md."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ProgressHead(nn.Module):
    """latent (..., D) -> item counts (..., len(items)). On a spatial latent (grid > 0) it reads only the status
    bar's tokens, the bottom hud_rows rows: the world view has no say in what the bar shows."""

    def __init__(
        self,
        latent_dim: int,
        items: Sequence[str],
        hidden: int = 256,
        grid: int = 0,
        token_dim: int = 0,
        hud_rows: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim, self.items, self.hidden = latent_dim, tuple(items), hidden
        self.grid, self.token_dim, self.hud_rows = grid, token_dim, hud_rows
        self.first = (grid - hud_rows) * grid * token_dim if grid else 0
        self.net = nn.Sequential(nn.Linear(latent_dim - self.first, hidden), nn.GELU(), nn.Linear(hidden, len(items)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(z[..., self.first :])
        return out

    @torch.no_grad()
    def read(self, z: torch.Tensor) -> dict[str, float]:
        """One latent (D,) -> {item: count}."""
        return dict(zip(self.items, self(z).tolist(), strict=True))
