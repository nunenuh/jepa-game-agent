"""Task definitions (docs/DESIGN.md D5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.base import GameEnv


@dataclass(frozen=True)
class ReachTask:
    """Family A: from step `start` of a recorded episode, reach the state len(goal_actions) steps later.

    The start is rebuilt by replaying `prefix` from the reset `seed`; the goal by replaying `goal_actions` after it.
    """

    seed: int
    start: int
    prefix: tuple[int, ...]
    goal_actions: tuple[int, ...]


@dataclass(frozen=True)
class AchievementTask:
    """Family B: from a fresh reset with `seed`, unlock `achievement` within `budget` steps."""

    achievement: str
    seed: int
    budget: int = 200


def make_reach_tasks(
    episodes: Sequence[Episode], k: int, n: int, seed: int, env: GameEnv | None = None
) -> list[ReachTask]:
    """n tasks with goal offset k, starts drawn uniformly over every valid (episode, step); deterministic.

    With `env`, each candidate is replayed and dropped unless `usable` (goal differs from start, and the
    replayed game doesn't end before the goal); candidates are drawn until n remain or none are left.
    """
    if k < 1 or n < 1:
        raise ValueError("k and n must be >= 1")
    valid = [(i, t) for i, e in enumerate(episodes) if e.seed >= 0 for t in range(len(e) - k + 1)]
    if not valid:
        raise ValueError(f"no episode with a known seed has {k} steps")
    order = np.random.default_rng(seed).permutation(len(valid))
    tasks = []
    for p in order.tolist():
        i, t = valid[p]
        acts = episodes[i].actions.tolist()
        task = ReachTask(episodes[i].seed, t, tuple(acts[:t]), tuple(acts[t : t + k]))
        if env is None or usable(env, task):
            tasks.append(task)
        if len(tasks) == n:
            break
    return sorted(tasks, key=lambda x: (x.seed, x.start))


def usable(env: GameEnv, task: ReachTask) -> bool:
    """Can the task be scored? Its goal must differ from its start, and the replayed game must not end
    before the goal step (the evaluators stop at the end of a game, so such a goal is unreachable)."""
    env.reset(task.seed)
    for a in task.prefix:
        if env.step(a).done:
            return False
    start = env.state_key()
    for i, a in enumerate(task.goal_actions):
        if env.step(a).done and i < len(task.goal_actions) - 1:
            return False
    return env.state_key() != start
