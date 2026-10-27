"""Score the baseline agents on both task families; the numbers every learned agent is compared with."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.base import GameEnv
from jepa_game_agent.evaluation.agents import RandomAgent, ReplayAgent, StillAgent
from jepa_game_agent.evaluation.runner import evaluate_achievement, evaluate_reach
from jepa_game_agent.evaluation.tasks import AchievementTask, make_reach_tasks


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_baselines(
    env: GameEnv,
    episodes: Sequence[Episode],
    ks: Sequence[int],
    tasks: int,
    achievements: Sequence[str] = (),
    resets: int = 50,
    budget: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Reach tasks (random / still / replay) for every k, achievement tasks (random / still) per name."""
    reach: dict[str, dict[str, Any]] = {}
    for k in ks:
        reach_tasks = make_reach_tasks(episodes, k=k, n=tasks, seed=seed, env=env)
        reach[f"k={k}"] = {
            "random": evaluate_reach(lambda: env, reach_tasks, lambda t: RandomAgent(env.num_actions, seed=seed)),
            "still": evaluate_reach(lambda: env, reach_tasks, lambda t: StillAgent()),
            "replay": evaluate_reach(lambda: env, reach_tasks, lambda t: ReplayAgent(t.goal_actions)),
        }
    ach: dict[str, dict[str, Any]] = {}
    for name in achievements:
        ach_tasks = [AchievementTask(name, seed=seed + i, budget=budget) for i in range(resets)]
        ach[name] = {
            "random": evaluate_achievement(lambda: env, ach_tasks, lambda t: RandomAgent(env.num_actions, seed=seed)),
            "still": evaluate_achievement(lambda: env, ach_tasks, lambda t: StillAgent()),
        }
    return {
        "reach": reach,
        "achievement": ach,
        "config": {
            "ks": list(ks),
            "tasks": tasks,
            "achievements": list(achievements),
            "resets": resets,
            "budget": budget,
            "seed": seed,
        },
        "commit": git_commit(),
    }


def summary(results: dict[str, Any]) -> dict[str, Any]:
    """Success rates only."""
    return {
        family: {
            task: {agent: r["success_rate"] for agent, r in agents.items()} for task, agents in results[family].items()
        }
        for family in ("reach", "achievement")
    }
