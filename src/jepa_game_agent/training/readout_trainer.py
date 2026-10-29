"""Train the vitals readout on frozen-encoder latents; labels come from replaying recorded games."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.readout import VITALS, Readout


def replay_vitals(env: Any, episode: Episode) -> NDArray[np.float32]:
    """Replay the episode (same seed, same actions) and read the env's vitals before each action and after
    the last: (T+1, 4) in VITALS order. Needs an env with `vitals()` (CraftaxEnv)."""
    env.reset(episode.seed)
    rows = [[float(env.vitals()[k]) for k in VITALS]]
    for a in episode.actions:
        env.step(int(a))
        rows.append([float(env.vitals()[k]) for k in VITALS])
    return np.asarray(rows, dtype=np.float32)


def train_readout(
    head: Readout,
    latents: Sequence[NDArray[Any]],
    labels: Sequence[NDArray[Any]],
    steps: int = 5_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """latents: (T+1, D) per episode; labels: matching (T+1, 4). Mean-squared error; returns the history."""
    if len(latents) != len(labels) or any(len(z) != len(y) for z, y in zip(latents, labels, strict=False)):
        raise ValueError("need one (T+1, 4) label array per (T+1, D) latent array")
    z_all = torch.from_numpy(np.concatenate(list(latents))).float().to(device)
    y_all = torch.from_numpy(np.concatenate(list(labels))).float().to(device)
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(y_all), (batch_size,), generator=gen).to(device)
        loss = F.mse_loss(head(z_all[idx]), y_all[idx])
        opt.zero_grad()
        torch.autograd.backward(loss)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "loss": loss.item()})
    head.eval()
    return history


def save_readout(path: Path, head: Readout, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"latent_dim": head.latent_dim, "hidden": head.hidden}
    torch.save({"model": head.state_dict(), "cfg": cfg, "history": history}, path)


def load_readout(path: Path, device: str | torch.device = "cpu") -> tuple[Readout, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = Readout(**ckpt["cfg"]).to(device).eval()
    head.load_state_dict(ckpt["model"])
    return head, ckpt
