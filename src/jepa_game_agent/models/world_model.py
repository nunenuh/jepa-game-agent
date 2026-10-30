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
from jepa_game_agent.models.spatial import SpatialEncoder, SpatialPredictor


@dataclass(frozen=True)
class Losses:
    total: torch.Tensor
    prediction: torch.Tensor
    sigreg: torch.Tensor
    inverse: torch.Tensor  # inverse-dynamics cross-entropy (0 when the head is off)


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder: nn.Module
        self.predictor: nn.Module
        if cfg.grid:  # v4: one token per tile, flattened to latent_dim outside the model
            self.encoder = SpatialEncoder(cfg.channels, cfg.grid, cfg.token_dim)
            self.predictor = SpatialPredictor(cfg.num_actions, cfg.grid * cfg.grid, cfg.token_dim, cfg.context,
                                              cfg.predictor_layers, cfg.predictor_heads, cfg.width)  # fmt: skip
        else:
            self.encoder = Encoder(cfg.channels, cfg.latent_dim, cfg.hud_rows, cfg.hud_dim)
            self.predictor = Predictor(
                cfg.num_actions, cfg.latent_dim, cfg.context, cfg.predictor_layers, cfg.predictor_heads
            )
        # Inverse dynamics: which action led from z_t to z_t+1? Forces the latent to keep what actions change
        # (e.g. the player's facing in Craftax). Only built when used, so older checkpoints load unchanged.
        self.inverse: nn.Module | None = None
        if cfg.inverse_weight > 0:
            d, hidden = cfg.latent_dim, min(cfg.latent_dim, 256)  # v2: hidden = d = 128, unchanged
            self.inverse = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, cfg.num_actions))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(frames)
        return z

    def loss(self, frames: torch.Tensor, actions: torch.Tensor) -> Losses:
        """frames (B, T+1, C, H, W), actions (B, T): predict every next latent from the ones before."""
        z = self.encode(frames)
        pred = self.predictor(z[:, :-1], actions)
        prediction = F.mse_loss(pred, z[:, 1:])
        reg = sigreg(self._reg_rows(z))
        inverse = torch.zeros((), device=z.device)
        if self.inverse is not None:
            logits = self.inverse(torch.cat([z[:, :-1], z[:, 1:]], -1))
            inverse = F.cross_entropy(logits.reshape(-1, logits.size(-1)), actions.reshape(-1))
        total = prediction + self.cfg.sigreg_weight * reg + self.cfg.inverse_weight * inverse
        return Losses(total, prediction, reg, inverse)

    def _reg_rows(self, z: torch.Tensor) -> torch.Tensor:
        """Rows SIGReg keeps Gaussian: one latent per frame, or (v4) per-tile tokens, as many as there are frames
        (sampled), so the regularizer's strength matches v2's."""
        frames = z.reshape(-1, z.size(-1))
        if not self.cfg.grid:
            return frames
        tokens = frames.reshape(-1, self.cfg.token_dim)
        return tokens[torch.randint(len(tokens), (max(len(frames), 2),), device=z.device)]

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
