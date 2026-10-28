"""Achievement goals as pictures: the frames right after an achievement unlocked in recorded games.

Craftax's reward per step = (first-time unlocks) + 0.1 * (health change). Health rises by at most 1 per step
(+0.1) but can fall a lot (a zombie hit while asleep: -0.7), so the unlocks in a step are ceil(reward - 0.1),
matched in order to the episode's recorded unlock list.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.base import Frame


def unlock_steps(episode: Episode) -> dict[str, int]:
    """Achievement -> the step at which it unlocked. Extra unit rewards (repeatable events) are ignored."""
    steps = [t for t, r in enumerate(episode.rewards) for _ in range(max(0, math.ceil(float(r) - 0.1 - 1e-4)))]
    if len(steps) < len(episode.achievements):
        raise ValueError(f"{len(episode.achievements)} achievements but {len(steps)} unlock rewards")
    return {name: steps[i] for i, name in enumerate(episode.achievements)}


def achievement_goal_frames(episodes: Sequence[Episode], name: str, n: int = 64, seed: int = 0) -> list[Frame]:
    """Up to n frames (from different episodes) right after `name` unlocked: what having it looks like."""
    found = []
    for e in episodes:
        if name in e.achievements:
            try:
                found.append(e.frames[unlock_steps(e)[name] + 1])
            except ValueError:  # unlock rewards don't add up (e.g. a deadly hit on the unlock step): skip
                continue
    if not found:
        raise ValueError(f"no episode unlocks {name!r}")
    pick = np.random.default_rng(seed).permutation(len(found))[:n]
    return [found[i] for i in sorted(pick)]
