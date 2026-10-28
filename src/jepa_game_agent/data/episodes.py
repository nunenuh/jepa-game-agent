"""Episodes as numpy arrays: frames (T+1, H, W, C) uint8 and actions (T,) int64."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from jepa_game_agent.envs.base import Frame, GameEnv

Policy = Callable[[Frame, np.random.Generator], int]


@dataclass(frozen=True)
class Episode:
    frames: NDArray[np.uint8]  # (T+1, H, W, C): the frame before each action, plus the last one
    actions: NDArray[np.int64]  # (T,)
    rewards: NDArray[np.float32]  # (T,)
    seed: int = -1  # reset seed: replaying it with the same actions rebuilds every frame (-1: unknown)
    achievements: tuple[str, ...] = ()  # unlocked during the episode, in order

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.dtype != np.uint8:
            raise ValueError("frames must be uint8 with shape (T+1, H, W, C)")
        if len(self.frames) != len(self.actions) + 1 or len(self.actions) != len(self.rewards):
            raise ValueError("need len(frames) == len(actions) + 1 == len(rewards) + 1")

    def __len__(self) -> int:
        return len(self.actions)


SPLITS = ("train", "val", "test")


def split_of(seed: int) -> str:
    """Split by episode, from its reset seed: seed % 20 == 0 is test, == 1 is val, the rest train (90/5/5)."""
    if seed < 0:
        raise ValueError("episode has no seed, can't assign a split")
    return {0: "test", 1: "val"}.get(seed % 20, "train")


def random_policy(num_actions: int) -> Policy:
    return lambda _frame, rng: int(rng.integers(num_actions))


def collect_episode(env: GameEnv, policy: Policy, max_steps: int, seed: int) -> Episode:
    """Play one episode (until done or max_steps) and record it."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    rng = np.random.default_rng(seed)
    frames, actions, rewards, unlocked = [env.reset(seed)], [], [], []
    for _ in range(max_steps):
        action = policy(frames[-1], rng)
        result = env.step(action)
        frames.append(result.frame)
        actions.append(action)
        rewards.append(result.reward)
        unlocked += [a for a in result.achievements if a not in unlocked]
        if result.done:
            break
    return Episode(
        np.stack(frames),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        seed=seed,
        achievements=tuple(unlocked),
    )


def data_card(episodes: Sequence[Episode]) -> dict[str, object]:
    """Summary for a dataset: size, episode lengths, how many episodes unlocked each achievement."""
    if not episodes:
        raise ValueError("no episodes")
    lengths = [len(e) for e in episodes]
    counts: dict[str, int] = {}
    for e in episodes:
        for a in e.achievements:
            counts[a] = counts.get(a, 0) + 1
    return {
        "episodes": len(episodes),
        "transitions": sum(lengths),
        "length": {"min": min(lengths), "mean": round(sum(lengths) / len(lengths), 1), "max": max(lengths)},
        "achievements": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def save_episodes(path: str | Path, episodes: Sequence[Episode]) -> None:
    """One compressed .npz; episodes are concatenated with their lengths stored alongside."""
    if not episodes:
        raise ValueError("nothing to save")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.concatenate([e.frames for e in episodes]),
        actions=np.concatenate([e.actions for e in episodes]),
        rewards=np.concatenate([e.rewards for e in episodes]),
        lengths=np.asarray([len(e) for e in episodes], dtype=np.int64),
        seeds=np.asarray([e.seed for e in episodes], dtype=np.int64),
        achievements=np.asarray([json.dumps(list(e.achievements)) for e in episodes]),
    )


def load_episodes(path: str | Path) -> list[Episode]:
    with np.load(path) as d:
        frames, actions, rewards, lengths = d["frames"], d["actions"], d["rewards"], d["lengths"]
        seeds = d["seeds"] if "seeds" in d else np.full(len(lengths), -1)
        achievements = [tuple(json.loads(a)) for a in d["achievements"]] if "achievements" in d else None
    out, f0, a0 = [], 0, 0
    for i, n in enumerate(lengths.tolist()):
        ach = achievements[i] if achievements is not None else ()
        out.append(
            Episode(
                frames[f0 : f0 + n + 1],
                actions[a0 : a0 + n],
                rewards[a0 : a0 + n],
                seed=int(seeds[i]),
                achievements=ach,
            )
        )
        f0, a0 = f0 + n + 1, a0 + n
    return out


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Every run of `window` consecutive transitions: frames (window+1, C, H, W) in [0, 1], actions (window,)."""

    def __init__(self, episodes: Sequence[Episode], window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.episodes, self.window = list(episodes), window
        self.index = [(i, t) for i, e in enumerate(self.episodes) for t in range(len(e) - window + 1)]
        if not self.index:
            raise ValueError(f"no episode has {window} transitions")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, t = self.index[k]
        e = self.episodes[i]
        frames = torch.from_numpy(e.frames[t : t + self.window + 1]).permute(0, 3, 1, 2).float() / 255
        return frames, torch.from_numpy(e.actions[t : t + self.window])
