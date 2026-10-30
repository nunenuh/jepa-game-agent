"""Train the threat head on frozen spatial latents; labels are zombie positions read while replaying games."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.threat import ThreatHead


def view_to_grid(view: NDArray[Any], grid: int) -> NDArray[np.float32]:
    """(rows, cols) view of zombies -> (grid * grid,) per token, the view in the top-left (Craftax-Classic: the
    7 x 9 world view is the top 7 of 9 token rows; the status-bar rows below stay 0)."""
    out = np.zeros((grid, grid), np.float32)
    r, c = min(view.shape[0], grid), min(view.shape[1], grid)
    out[:r, :c] = view[:r, :c]
    return out.reshape(-1)


def rank_auc(scores: NDArray[Any], labels: NDArray[Any]) -> float:
    """ROC AUC from ranks (ties share their mean rank), for millions of scores."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2)[inverse]  # 1-based, ties share their mean rank
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        raise ValueError("need at least one positive and one negative")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def replay_threats(env: Any, episode: Episode, grid: int) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Replay the episode's seed and actions: the replayed frames (T'+1, H, W, C) and each one's zombie map
    (T'+1, grid * grid), stopping if the replayed game ends early. Needs an env with `zombie_tiles()`."""
    frames = [np.asarray(env.reset(episode.seed), dtype=np.uint8)]
    maps = [view_to_grid(env.zombie_tiles(), grid)]
    for a in episode.actions:
        res = env.step(int(a))
        frames.append(np.asarray(res.frame, dtype=np.uint8))
        maps.append(view_to_grid(env.zombie_tiles(), grid))
        if res.done:
            break
    return np.stack(frames), np.stack(maps)


def train_threat(
    head: ThreatHead,
    latents: Sequence[NDArray[Any]],
    maps: Sequence[NDArray[Any]],
    steps: int = 2_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """latents (T+1, N * d) per episode, maps (T+1, N) 0/1: binary cross-entropy per tile."""
    if len(latents) != len(maps) or any(len(z) != len(m) for z, m in zip(latents, maps, strict=False)):
        raise ValueError("need one (T+1, N) map per (T+1, N * d) latent array")
    z_all = torch.from_numpy(np.concatenate(list(latents))).float().to(device)
    y_all = torch.from_numpy(np.concatenate(list(maps))).float().to(device)
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(y_all), (batch_size,), generator=gen).to(device)
        loss = F.binary_cross_entropy_with_logits(head(z_all[idx]), y_all[idx])
        opt.zero_grad()
        torch.autograd.backward(loss)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "loss": loss.item()})
    head.eval()
    return history


def save_threat(path: Path, head: ThreatHead, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"token_dim": head.token_dim, "grid": head.grid, "hidden": head.hidden}
    torch.save({"model": head.state_dict(), "cfg": cfg, "history": history}, path)


def load_threat(path: Path, device: str | torch.device = "cpu") -> tuple[ThreatHead, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = ThreatHead(**ckpt["cfg"]).to(device).eval()
    head.load_state_dict(ckpt["model"])
    return head, ckpt
