"""Train the action prior on frozen-encoder latents: latent z_t -> the action the player pressed at t."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from jepa_game_agent.models.prior import ActionPrior


def train_prior(
    prior: ActionPrior,
    latents: Sequence[NDArray[Any]],
    actions: Sequence[NDArray[Any]],
    steps: int = 5_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """latents: one (T+1, D) array per episode; actions: the matching (T,) arrays. Returns the loss history."""
    for z, a in zip(latents, actions, strict=True):
        if len(z) != len(a) + 1:
            raise ValueError(f"each episode needs T+1 latents for T actions, got {len(z)} and {len(a)}")
    z_all = torch.from_numpy(np.concatenate([z[:-1] for z in latents])).float().to(device)
    a_all = torch.from_numpy(np.concatenate(list(actions))).long().to(device)
    prior = prior.to(device).train()
    opt = torch.optim.AdamW(prior.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(a_all), (batch_size,), generator=gen).to(device)
        loss = F.cross_entropy(prior(z_all[idx]), a_all[idx])
        opt.zero_grad()
        torch.autograd.backward(loss)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "loss": loss.item()})
    prior.eval()
    return history


def save_prior(path: Path, prior: ActionPrior, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"latent_dim": prior.latent_dim, "num_actions": prior.num_actions, "hidden": prior.hidden}
    torch.save({"model": prior.state_dict(), "cfg": cfg, "history": history}, path)


def load_prior(path: Path, device: str | torch.device = "cpu") -> tuple[ActionPrior, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior = ActionPrior(**ckpt["cfg"]).to(device).eval()
    prior.load_state_dict(ckpt["model"])
    return prior, ckpt
