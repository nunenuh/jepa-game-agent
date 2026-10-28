"""Surprise test: is the world model more surprised by an impossible jump than by what really came next?

Each pair shares a real history (window frames and actions). "Normal" ends with the real next frame;
"jump" ends with a frame of the same game at least `min_gap` steps away: a teleport within the same world.
Score = the model's prediction error on that last step; ROC-AUC says how well it tells the two apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.world_model import WorldModel


def roc_auc(negatives: Sequence[float], positives: Sequence[float]) -> float:
    """P(a random positive scores above a random negative); ties count half."""
    if not negatives or not positives:
        raise ValueError("need at least one negative and one positive score")
    neg, pos = np.asarray(negatives)[None, :], np.asarray(positives)[:, None]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _pairs(
    episodes: Sequence[Episode], window: int, n: int, min_gap: int, rng: np.random.Generator
) -> tuple[Any, Any, Any]:
    eligible = [e for e in episodes if len(e) >= window + min_gap]
    if not eligible:
        raise ValueError(f"no episode has window + min_gap = {window + min_gap} transitions")
    normal: list[Any] = []
    jump: list[Any] = []
    actions: list[Any] = []
    while len(normal) < n:
        e = eligible[rng.integers(len(eligible))]
        t = int(rng.integers(len(e) - window + 1))
        nxt = t + window
        far = [s for s in range(len(e.frames)) if abs(s - nxt) >= min_gap]
        if not far:
            continue
        clip = e.frames[t : nxt + 1]
        normal.append(clip)
        jump.append(np.concatenate([clip[:-1], e.frames[far[rng.integers(len(far))]][None]]))
        actions.append(e.actions[t:nxt])
    return np.stack(normal), np.stack(jump), np.stack(actions)


@torch.no_grad()
def _last_step_surprise(
    model: WorldModel, frames: Any, actions: Any, device: str | torch.device, batch: int = 256
) -> list[float]:
    out: list[float] = []
    for i in range(0, len(frames), batch):
        f = torch.from_numpy(frames[i : i + batch]).to(device).permute(0, 1, 4, 2, 3).float() / 255
        a = torch.from_numpy(actions[i : i + batch]).to(device).long()
        out += model.surprise(f, a)[:, -1].cpu().tolist()
    return out


def surprise_auc(
    model: WorldModel,
    episodes: Sequence[Episode],
    window: int = 8,
    n: int = 1000,
    min_gap: int = 50,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """AUC of last-step surprise, jump (positive) vs normal (negative), over n pairs."""
    normal, jump, actions = _pairs(episodes, window, n, min_gap, np.random.default_rng(seed))
    model = model.eval().to(device)
    s_normal = _last_step_surprise(model, normal, actions, device)
    s_jump = _last_step_surprise(model, jump, actions, device)
    return {
        "auc": roc_auc(s_normal, s_jump),
        "pairs": n,
        "mean_normal": float(np.mean(s_normal)),
        "mean_jump": float(np.mean(s_jump)),
    }
