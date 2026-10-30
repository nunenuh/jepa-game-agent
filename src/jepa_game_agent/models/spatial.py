"""v4: a spatial latent, one token per game tile (docs/DESIGN_V4.md).

A Craftax-Classic frame is a 9 x 9 grid of 7 x 7-pixel tiles; v2's pooled vector blurs small sprites
(zombies, facing, health hearts), so here every tile keeps its own small latent, as V-JEPA 2-AC and DINO-WM keep
one latent per image patch. Tokens are flattened to one vector outside the model, so planners and heads work
unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialEncoder(nn.Module):
    """(..., C, H, W) in [0, 1] -> (..., grid * grid * token_dim): one token per tile, row by row.

    Full-resolution convolutions, average-pooled per tile, then one 3 x 3 mixing step between neighbouring tiles,
    so a token sees its own tile and the tiles around it, nothing farther."""

    def __init__(self, channels: int, grid: int, token_dim: int) -> None:
        super().__init__()
        self.grid, self.token_dim = grid, token_dim
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(grid),
            nn.Conv2d(64, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, token_dim, 1),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lead = frames.shape[:-3]
        x = self.net(frames.reshape(-1, *frames.shape[-3:]))  # (N, d, g, g)
        z: torch.Tensor = self.norm(x.permute(0, 2, 3, 1))  # (N, g, g, d)
        return z.reshape(*lead, -1)


class SpatialPredictor(nn.Module):
    """Same contract as Predictor: latents (B, T, N * d), actions (B, T) -> next latents (B, T, N * d).

    A transformer over every token of the last T frames; a token may attend to all tokens of its own frame and
    earlier frames, never later ones (block-causal). The action is added to every token of its frame."""

    def __init__(
        self, num_actions: int, tokens: int, token_dim: int, context: int, layers: int, heads: int, width: int
    ) -> None:
        super().__init__()
        self.tokens, self.token_dim, self.context = tokens, token_dim, context
        self.inp = nn.Linear(token_dim, width)
        self.action = nn.Embedding(num_actions, width)
        self.tile_pos = nn.Parameter(torch.randn(1, 1, tokens, width) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, context, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, token_dim)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b, t = latents.shape[:2]
        if t > self.context:
            raise ValueError(f"sequence length {t} > context {self.context}")
        if actions.shape != latents.shape[:2]:
            raise ValueError("actions must have shape (B, T) matching latents")
        x = self.inp(latents.reshape(b, t, self.tokens, self.token_dim))
        x = x + self.action(actions).unsqueeze(2) + self.tile_pos + self.time_pos[:, :t]
        frame = torch.arange(t, device=latents.device).repeat_interleave(self.tokens)
        mask = torch.zeros(t * self.tokens, t * self.tokens, device=latents.device)
        mask = mask.masked_fill(frame.unsqueeze(0) > frame.unsqueeze(1), float("-inf"))  # no later frames
        y = self.blocks(x.reshape(b, t * self.tokens, -1), mask=mask)
        out: torch.Tensor = self.out(y).reshape(b, t, -1)
        return out
