"""Train the progress head on frozen-encoder latents; labels come from replaying recorded games."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.progress import ProgressHead


def replay_inventory(env: Any, episode: Episode, items: Sequence[str]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Replayed frames (T'+1, H, W, C) and the item counts for each (T'+1, len(items)), stopping if the game ends.
    Labels must go with the replayed frames (see readout_trainer.replay_vitals)."""
    frames = [np.asarray(env.reset(episode.seed), dtype=np.uint8)]
    rows = [[float(env.inventory()[i]) for i in items]]
    for a in episode.actions:
        res = env.step(int(a))
        frames.append(np.asarray(res.frame, dtype=np.uint8))
        rows.append([float(env.inventory()[i]) for i in items])
        if res.done:
            break
    return np.stack(frames), np.asarray(rows, dtype=np.float32)


def train_progress(
    head: ProgressHead,
    latents: Sequence[NDArray[Any]],
    counts: Sequence[NDArray[Any]],
    steps: int = 2_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """latents (T+1, D) per episode, counts (T+1, items): squared error on the counts."""
    if len(latents) != len(counts) or any(len(z) != len(c) for z, c in zip(latents, counts, strict=False)):
        raise ValueError("need one (T+1, items) count array per (T+1, D) latent array")
    z_all = torch.from_numpy(np.concatenate(list(latents))).float().to(device)
    y_all = torch.from_numpy(np.concatenate(list(counts))).float().to(device)
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


def save_progress(path: Path, head: ProgressHead, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"latent_dim": head.latent_dim, "items": list(head.items), "hidden": head.hidden, "grid": head.grid,
           "token_dim": head.token_dim, "hud_rows": head.hud_rows}  # fmt: skip
    torch.save({"model": head.state_dict(), "cfg": cfg, "history": history}, path)


def load_progress(path: Path, device: str | torch.device = "cpu") -> tuple[ProgressHead, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = ProgressHead(**ckpt["cfg"]).to(device).eval()
    head.load_state_dict(ckpt["model"])
    return head, ckpt
