"""Threat sense: is there a zombie on this tile? One tiny head shared by every tile token of a spatial (v4) latent.

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
    def __init__(self, token_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.token_dim, self.hidden = token_dim, hidden
        self.net = nn.Sequential(nn.Linear(token_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., N * token_dim) flattened tokens -> (..., N) logits, 'zombie on this tile'."""
        tokens = z.reshape(*z.shape[:-1], -1, self.token_dim)
        out: torch.Tensor = self.net(tokens)[..., 0]
        return out
