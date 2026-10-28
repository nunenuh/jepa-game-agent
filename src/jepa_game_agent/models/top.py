"""Top level of the hierarchy: predicts the latent `stride` game steps ahead from an action chunk."""

from __future__ import annotations

import torch
from torch import nn

from jepa_game_agent.config import TopConfig
from jepa_game_agent.models.predictor import Predictor


class TopModel(nn.Module):
    """(latents, action chunks) -> latents one chunk later. Works on the frozen bottom encoder's latents,
    so its predictions are goals the bottom planner understands."""

    def __init__(self, cfg: TopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.action = nn.Embedding(cfg.num_actions, d)
        self.slot = nn.Parameter(torch.randn(1, 1, cfg.stride, d) * 0.02)  # position inside the chunk: order matters
        self.chunk = nn.Linear(cfg.stride * d, d)
        self.core = Predictor(1, d, cfg.context, cfg.layers, cfg.heads)  # actions enter through `chunk` instead

    def forward(self, latents: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """latents (B, T, D), chunks (B, T, stride) int -> predicted next-chunk latents (B, T, D)."""
        if chunks.shape != (*latents.shape[:2], self.cfg.stride):
            raise ValueError(f"chunks must have shape (B, T, {self.cfg.stride})")
        a = (self.action(chunks) + self.slot).flatten(2)  # (B, T, stride * D)
        no_action = torch.zeros(latents.shape[:2], dtype=torch.long, device=latents.device)
        y: torch.Tensor = self.core(latents + self.chunk(a), no_action)
        return y

    @torch.no_grad()
    def rollout(self, start: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """start (B, D), chunks (B, K, stride) -> imagined latents after each chunk (B, K, D)."""
        lat, out = start.unsqueeze(1), []
        for k in range(chunks.size(1)):
            ctx = self.cfg.context
            nxt = self(lat[:, -ctx:], chunks[:, max(0, k + 1 - ctx) : k + 1])[:, -1]
            out.append(nxt)
            lat = torch.cat([lat, nxt.unsqueeze(1)], 1)
        return torch.stack(out, 1)
