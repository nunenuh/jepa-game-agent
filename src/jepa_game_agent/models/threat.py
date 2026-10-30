"""Threat sense: is there a zombie on this tile? A small conv head over the tile tokens of a spatial (v4) latent.

Run on imagined tokens it says where danger will be, so the planner can avoid futures with a zombie next to the
player (like the amygdala reading the cortex's predictions). Labels (zombie positions) only train this head."""

from __future__ import annotations

import torch
from torch import nn


def around(grid: int, center: tuple[int, int]) -> list[int]:
    """Token indices of the 8 tiles next to `center` (row, col) in a grid x grid token layout."""
    r0, c0 = center
    return [r * grid + c for r in (r0 - 1, r0, r0 + 1) for c in (c0 - 1, c0, c0 + 1)
            if (r, c) != center and 0 <= r < grid and 0 <= c < grid]  # fmt: skip


class ThreatHead(nn.Module):
    """A 3 x 3 conv over the grid of tile tokens: each tile is read with its 8 neighbours, the same weights for
    every tile (the encoder spreads a zombie over neighbouring tokens: one token alone read 'zombie next to the
    player' at AUC 0.72, a 3 x 3 neighbourhood at 0.97)."""

    def __init__(self, token_dim: int, grid: int, hidden: int = 64) -> None:
        super().__init__()
        self.token_dim, self.grid, self.hidden = token_dim, grid, hidden
        self.net = nn.Sequential(nn.Conv2d(token_dim, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., grid * grid * token_dim) flattened tokens -> (..., grid * grid) logits, 'zombie on this tile'."""
        lead, g = z.shape[:-1], self.grid
        x = z.reshape(-1, g, g, self.token_dim).permute(0, 3, 1, 2)
        out: torch.Tensor = self.net(x).reshape(*lead, g * g)
        return out
