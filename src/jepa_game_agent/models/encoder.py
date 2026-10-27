"""Frame -> one latent vector."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Small CNN for game frames of any size (63x63 Craftax-Classic, 130x110 Craftax, tiny test grids)."""

    def __init__(self, channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(..., C, H, W) in [0, 1] -> (..., latent_dim)."""
        lead = frames.shape[:-3]
        z: torch.Tensor = self.proj(self.conv(frames.reshape(-1, *frames.shape[-3:])))
        return z.reshape(*lead, -1)
