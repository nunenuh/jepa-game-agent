"""Run agents on tasks and score them. Results are plain dicts, ready for JSON."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChainStage:
    """One subgoal of a chain, given as its goal frames. Done when the achievement `name` unlocks, or, with an
    item, when the inventory holds at least `count` of it (e.g. a table needs 2 wood: collect_wood, wood, 2)."""

    name: str
    frames: Sequence[Frame]
    item: str | None = None
    count: int = 0

    @property
    def label(self) -> str:
        return self.name if self.item is None else f"{self.name}:{self.count}"


def _stage_done(
    st: ChainStage, got: Sequence[str], env: GameEnv, counts: Callable[[GameEnv], Mapping[str, int]] | None
) -> bool:
    if st.item is None:
        return st.name in got
    assert counts is not None
    return counts(env).get(st.item, 0) >= st.count


def evaluate_chain(
    make_env: Callable[[], GameEnv],
    tasks: Sequence[AchievementTask],
    make_agent: Callable[[AchievementTask], Agent],
    chain: Sequence[ChainStage | tuple[str, Sequence[Frame]]],
    counts: Callable[[GameEnv], Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Pursue the chain's stages in order, each given as its goal frames; the agent is handed the first stage
    not yet done (one already done when reached is skipped; done stays done). Success = every stage done
    within the budget. counts(env) -> inventory, needed for stages that count items."""
    stages = [c if isinstance(c, ChainStage) else ChainStage(*c) for c in chain]
    if not stages:
        raise ValueError("empty chain")
    if counts is None and any(st.item is not None for st in stages):
        raise ValueError("stages that count items need counts(env)")
    labels = [st.label for st in stages]
    successes, steps, reached, unlocked, lengths, ended = [], [], [], [], [], []
    for task in tasks:
        env = make_env()
        frame = env.reset(task.seed)
        agent = make_agent(task)
        got: list[str] = []
        stage = 0
        agent.reset(list(stages[0].frames))
        hit, step, how = None, 0, "budget"
        for step in range(1, task.budget + 1):
            result = env.step(agent.act(frame))
            frame = result.frame
            got += [a for a in result.achievements if a not in got]
            before = stage
            while stage < len(stages) and _stage_done(stages[stage], got, env, counts):
                stage += 1
            if stage == len(stages):
                hit, how = step, "success"
                break
            if stage != before:
                agent.reset(list(stages[stage].frames))
            if result.done:  # Craftax: the player died
                how = "done"
                break
        lengths.append(step)
        ended.append(how)
        successes.append(hit is not None)
        steps.append(hit)
        reached.append(labels[:stage])
        unlocked.append(got)
    n = len(tasks)
    return {
        "family": "chain",
        "chain": labels,
        "tasks": n,
        "success_rate": sum(successes) / n if n else 0.0,
        "successes": successes,
        "steps_to_success": steps,
        "reached": reached,
        "unlocked": unlocked,
        "lengths": lengths,
        "ended": ended,
        "stage_rates": {lb: sum(lb in r for r in reached) / n if n else 0.0 for lb in labels},
    }
