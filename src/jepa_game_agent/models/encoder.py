"""Frame -> one latent vector."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Small CNN for game frames of any size (63x63 Craftax-Classic, 130x110 Craftax, tiny test grids).

    hud_rows > 0: the bottom hud_rows pixel rows (a status / inventory bar) go through their own small CNN that
    keeps a 2 x 9 grid of slots, and fill the last hud_dim latent dims; the view above fills the rest. Pooling the
    whole frame to 4 x 4 otherwise washes the few inventory pixels out.
    """

    def __init__(self, channels: int, latent_dim: int, hud_rows: int = 0, hud_dim: int = 0) -> None:
        super().__init__()
        self.hud_rows = hud_rows
        map_dim = latent_dim - hud_dim if hud_rows else latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, map_dim), nn.LayerNorm(map_dim))
        self.hud: nn.Module | None = None
        if hud_rows:
            self.hud = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 9)),
                nn.Flatten(),
                nn.Linear(32 * 18, hud_dim),
                nn.LayerNorm(hud_dim),
            )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(..., C, H, W) in [0, 1] -> (..., latent_dim)."""
        lead = frames.shape[:-3]
        x = frames.reshape(-1, *frames.shape[-3:])
        if self.hud is None:
            z: torch.Tensor = self.proj(self.conv(x))
        else:
            view, bar = x[..., : -self.hud_rows, :], x[..., -self.hud_rows :, :]
            z = torch.cat([self.proj(self.conv(view)), self.hud(bar)], -1)
        return z.reshape(*lead, -1)
