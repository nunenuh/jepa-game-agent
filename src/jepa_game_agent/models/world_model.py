"""LeWM-style JEPA world model: encoder + action-conditioned predictor, trained end to end with SIGReg."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from jepa_game_agent.config import ModelConfig
from jepa_game_agent.models.encoder import Encoder
from jepa_game_agent.models.predictor import Predictor
from jepa_game_agent.models.sigreg import sigreg


@dataclass(frozen=True)
class Losses:
    total: torch.Tensor
    prediction: torch.Tensor
    sigreg: torch.Tensor


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.channels, cfg.latent_dim)
        self.predictor = Predictor(
            cfg.num_actions, cfg.latent_dim, cfg.context, cfg.predictor_layers, cfg.predictor_heads
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(frames)
        return z

    def loss(self, frames: torch.Tensor, actions: torch.Tensor) -> Losses:
        """frames (B, T+1, C, H, W), actions (B, T): predict every next latent from the ones before."""
        z = self.encode(frames)
        pred = self.predictor(z[:, :-1], actions)
        prediction = F.mse_loss(pred, z[:, 1:])
        reg = sigreg(z.reshape(-1, z.size(-1)))
        return Losses(prediction + self.cfg.sigreg_weight * reg, prediction, reg)

    @torch.no_grad()
    def rollout(
        self,
        start: torch.Tensor,
        actions: torch.Tensor,
        history: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Imagine the future: start (B, D) and actions (B, K) -> predicted latents (B, K, D).

        Optional history: earlier latents (B, H, D) and the actions taken after each (B, H), oldest first.
        Only the last `context` steps are fed to the predictor.
        """
        if (history is None) != (history_actions is None):
            raise ValueError("give both history and history_actions, or neither")
        lat = start.unsqueeze(1) if history is None else torch.cat([history, start.unsqueeze(1)], 1)
        acts = actions[:, :0] if history_actions is None else history_actions
        out = []
        for k in range(actions.size(1)):
            acts = torch.cat([acts, actions[:, k : k + 1]], 1)
            ctx = self.cfg.context
            nxt = self.predictor(lat[:, -ctx:], acts[:, -ctx:])[:, -1]
            out.append(nxt)
            lat = torch.cat([lat, nxt.unsqueeze(1)], 1)
        return torch.stack(out, 1)

    @torch.no_grad()
    def surprise(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-step prediction error (B, T): how unexpected each next frame was. High = 'that can't happen'."""
        z = self.encode(frames)
        err: torch.Tensor = (self.predictor(z[:, :-1], actions) - z[:, 1:]).pow(2).mean(-1)
        return err
