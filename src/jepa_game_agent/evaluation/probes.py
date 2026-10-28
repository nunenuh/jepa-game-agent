"""Linear probes: fit latent -> game state (position, facing, inventory) on some episodes, score R² on others.

High R² means the world model's latent keeps that piece of state in a linearly readable form; the planner can
only steer toward what the latent keeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from jepa_game_agent.data.episodes import Policy
from jepa_game_agent.envs.base import GameEnv
from jepa_game_agent.models.world_model import WorldModel


def ridge_r2(
    z_train: NDArray[Any], y_train: NDArray[Any], z_test: NDArray[Any], y_test: NDArray[Any], l2: float = 1e-2
) -> NDArray[np.float64]:
    """R² per target column of a ridge regression (with bias). NaN where the test target is constant."""
    x_tr = np.hstack([z_train, np.ones((len(z_train), 1))])
    x_te = np.hstack([z_test, np.ones((len(z_test), 1))])
    w = np.linalg.solve(x_tr.T @ x_tr + l2 * np.eye(x_tr.shape[1]), x_tr.T @ y_train)
    res = ((x_te @ w - y_test) ** 2).sum(0)
    tot = ((y_test - y_test.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, 1 - res / tot, np.nan)


def collect_states(
    env: GameEnv, policy: Policy, seeds: Sequence[int], max_steps: int
) -> tuple[NDArray[np.uint8], NDArray[np.int64], NDArray[np.int64]]:
    """Play one episode per seed; every frame (reset included) with its state key and episode index."""
    rng = np.random.default_rng(0)
    frames, keys, episode = [], [], []
    for i, seed in enumerate(seeds):
        frame = env.reset(seed)
        for t in range(max_steps + 1):
            frames.append(frame)
            keys.append(env.state_key())
            episode.append(i)
            if t == max_steps:
                break
            result = env.step(policy(frame, rng))
            frame = result.frame
            if result.done:
                frames.append(frame)
                keys.append(env.state_key())
                episode.append(i)
                break
    return np.stack(frames), np.asarray(keys, dtype=np.int64), np.asarray(episode, dtype=np.int64)


@torch.no_grad()
def _encode(model: WorldModel, frames: NDArray[np.uint8], device: str | torch.device, batch: int = 512) -> Any:
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i : i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        out.append(model.encode(x).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def run_probes(
    model: WorldModel,
    env: GameEnv,
    policy: Policy,
    seeds: Sequence[int],
    max_steps: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Collect states, encode them, fit on 4 of every 5 episodes and score R² on the 5th (split by episode)."""
    frames, keys, episode = collect_states(env, policy, seeds, max_steps)
    z = _encode(model.eval().to(device), frames, device)
    test = episode % 5 == 0
    r2 = ridge_r2(z[~test], keys[~test].astype(np.float64), z[test], keys[test].astype(np.float64))
    return {
        "r2": [None if np.isnan(v) else float(v) for v in r2],
        "train_frames": int((~test).sum()),
        "test_frames": int(test.sum()),
        "test_episodes": len(np.unique(episode[test])),
    }
