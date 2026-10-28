"""A plain training loop that checkpoints often and resumes where it stopped."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_game_agent.config import ModelConfig, TrainConfig
from jepa_game_agent.data.episodes import WindowDataset
from jepa_game_agent.models.world_model import WorldModel


def save_checkpoint(
    path: Path, model: WorldModel, opt: torch.optim.Optimizer, step: int, history: list[dict[str, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "model_cfg": asdict(model.cfg),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a session killed mid-save keeps the last good file


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[WorldModel, dict[str, object]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WorldModel(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def train(
    model: WorldModel,
    data: WindowDataset,
    cfg: TrainConfig,
    checkpoint: Path | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Train until cfg.steps (counting steps done in earlier sessions). Returns the loss history."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, history = 0, []
    if checkpoint is not None and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, history = int(ckpt["step"]), list(ckpt["history"])
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, drop_last=len(data) > cfg.batch_size)
    while step < cfg.steps:
        for frames, actions in loader:
            losses = model.loss(frames.to(device), actions.to(device))
            opt.zero_grad()
            torch.autograd.backward(losses.total)
            opt.step()
            step += 1
            history.append(
                {
                    "step": step,
                    "loss": losses.total.item(),
                    "prediction": losses.prediction.item(),
                    "sigreg": losses.sigreg.item(),
                    "inverse": losses.inverse.item(),
                }
            )
            if checkpoint is not None and (step % cfg.checkpoint_every == 0 or step == cfg.steps):
                save_checkpoint(checkpoint, model, opt, step, history)
            if step >= cfg.steps:
                break
    return history
