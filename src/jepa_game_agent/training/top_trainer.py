"""Train the top level on the frozen bottom encoder's latents (no images: fast), resumable like `train`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from jepa_game_agent.config import TopConfig, TrainConfig
from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.top import TopModel
from jepa_game_agent.models.world_model import WorldModel


@torch.no_grad()
def encode_episodes(
    model: WorldModel, episodes: Sequence[Episode], device: str | torch.device = "cpu", batch: int = 1024
) -> list[NDArray[np.float32]]:
    """Every frame of every episode through the (frozen) bottom encoder: one (T+1, D) array per episode."""
    model = model.eval().to(device)
    out = []
    for e in episodes:
        parts = []
        for i in range(0, len(e.frames), batch):
            x = torch.from_numpy(e.frames[i : i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
            parts.append(model.encode(x).cpu())
        out.append(torch.cat(parts).numpy().astype(np.float32))
    return out


class TopWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Latents (steps+1, D) `stride` apart and the action chunks between them (steps, stride)."""

    def __init__(
        self, latents: Sequence[NDArray[Any]], actions: Sequence[NDArray[Any]], stride: int, steps: int
    ) -> None:
        self.latents, self.actions, self.stride, self.steps = list(latents), list(actions), stride, steps
        span = stride * steps
        self.index = [(i, t) for i, z in enumerate(self.latents) for t in range(len(z) - span)]
        if not self.index:
            raise ValueError(f"no episode has {span} transitions")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, t = self.index[k]
        s, n = self.stride, self.steps
        z = torch.from_numpy(np.ascontiguousarray(self.latents[i][t : t + s * n + 1 : s]))
        chunks = torch.from_numpy(self.actions[i][t : t + s * n].reshape(n, s)).long()
        return z, chunks


def save_top(path: Path, top: TopModel, opt: torch.optim.Optimizer, step: int, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": top.state_dict(), "opt": opt.state_dict(), "cfg": asdict(top.cfg),
                "history": history}, tmp)  # fmt: skip
    tmp.replace(path)


def load_top(path: Path, device: str | torch.device = "cpu") -> tuple[TopModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    top = TopModel(TopConfig(**ckpt["cfg"])).to(device)
    top.load_state_dict(ckpt["model"])
    return top, ckpt


def train_top(
    top: TopModel,
    data: TopWindowDataset,
    cfg: TrainConfig,
    checkpoint: Path | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """MSE between predicted and real next-chunk latents, until cfg.steps (counting earlier sessions)."""
    torch.manual_seed(cfg.seed)
    top.to(device).train()
    opt = torch.optim.AdamW(top.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, history = 0, []
    if checkpoint is not None and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        top.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, history = int(ckpt["step"]), list(ckpt["history"])
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, drop_last=len(data) > cfg.batch_size)
    while step < cfg.steps:
        for z, chunks in loader:
            z, chunks = z.to(device), chunks.to(device)
            loss = F.mse_loss(top(z[:, :-1], chunks), z[:, 1:])
            opt.zero_grad()
            torch.autograd.backward(loss)
            opt.step()
            step += 1
            history.append({"step": step, "loss": loss.item()})
            if checkpoint is not None and (step % cfg.checkpoint_every == 0 or step == cfg.steps):
                save_top(checkpoint, top, opt, step, history)
            if step >= cfg.steps:
                break
    return history
