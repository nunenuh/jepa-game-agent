"""Run agents on tasks and score them. Results are plain dicts, ready for JSON."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from jepa_game_agent.envs.base import Frame, GameEnv
from jepa_game_agent.evaluation.agents import Agent
from jepa_game_agent.evaluation.tasks import AchievementTask, ReachTask

T = TypeVar("T")


def _replay(env: GameEnv, seed: int, actions: Sequence[int]) -> Frame:
    frame = env.reset(seed)
    for a in actions:
        frame = env.step(a).frame
    return frame


def evaluate_reach(
    make_env: Callable[[], GameEnv],
    tasks: Sequence[ReachTask],
    make_agent: Callable[[ReachTask], Agent],
    budget_factor: int = 2,
) -> dict[str, Any]:
    """Success = the env's state_key equals the goal's at any step within budget_factor * k steps."""
    successes, steps, start_keys, goal_keys = [], [], [], []
    for task in tasks:
        env = make_env()
        _replay(env, task.seed, task.prefix)
        start_key = env.state_key()
        goal_frame = _replay(env, task.seed, task.prefix + task.goal_actions)
        goal_key = env.state_key()
        frame = _replay(env, task.seed, task.prefix)
        agent = make_agent(task)
        agent.reset([goal_frame])
        hit = None
        for step in range(1, budget_factor * len(task.goal_actions) + 1):
            result = env.step(agent.act(frame))
            frame = result.frame
            if env.state_key() == goal_key:
                hit = step
                break
            if result.done:
                break
        successes.append(hit is not None)
        steps.append(hit)
        start_keys.append(start_key)
        goal_keys.append(goal_key)
    return {
        "family": "reach",
        "tasks": len(tasks),
        "budget_factor": budget_factor,
        "success_rate": sum(successes) / len(tasks) if tasks else 0.0,
        "successes": successes,
        "steps_to_success": steps,
        "start_keys": start_keys,
        "goal_keys": goal_keys,
    }


def evaluate_achievement(
    make_env: Callable[[], GameEnv],
    tasks: Sequence[AchievementTask],
    make_agent: Callable[[AchievementTask], Agent],
    goal_frames: Sequence[Frame] = (),
) -> dict[str, Any]:
    """Success = the env reports the task's achievement as unlocked within the budget."""
    successes, steps = [], []
    for task in tasks:
        env = make_env()
        frame = env.reset(task.seed)
        agent = make_agent(task)
        agent.reset(list(goal_frames))
        hit = None
        for step in range(1, task.budget + 1):
            result = env.step(agent.act(frame))
            frame = result.frame
            if task.achievement in result.achievements:
                hit = step
                break
            if result.done:
                break
        successes.append(hit is not None)
        steps.append(hit)
    return {
        "family": "achievement",
        "tasks": len(tasks),
        "success_rate": sum(successes) / len(tasks) if tasks else 0.0,
        "successes": successes,
        "steps_to_success": steps,
    }
