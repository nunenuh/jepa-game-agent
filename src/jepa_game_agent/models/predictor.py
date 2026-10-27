"""(latents so far, actions taken) -> the next latent, causally."""

from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    """Causal transformer: output t predicts latent t+1 from latents 0..t and actions 0..t."""

    def __init__(self, num_actions: int, latent_dim: int, context: int, layers: int, heads: int) -> None:
        super().__init__()
        self.context = context
        self.action = nn.Embedding(num_actions, latent_dim)
        self.pos = nn.Parameter(torch.randn(1, context, latent_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            latent_dim, heads, dim_feedforward=4 * latent_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """latents (B, T, D), actions (B, T) int -> predicted next latents (B, T, D). T <= context."""
        t = latents.size(1)
        if t > self.context:
            raise ValueError(f"sequence length {t} > context {self.context}")
        if actions.shape != latents.shape[:2]:
            raise ValueError("actions must have shape (B, T) matching latents")
        x = latents + self.action(actions) + self.pos[:, :t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=latents.device)
        y: torch.Tensor = self.out(self.blocks(x, mask=mask, is_causal=True))
        return y
