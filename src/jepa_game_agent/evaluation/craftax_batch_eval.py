"""Craftax evaluation with every task at once (jax.vmap). Needs the craftax extra.

Same tasks and success rule as `evaluation.runner`, but all tasks share one batched game: start states are
rebuilt by replaying each task's recorded actions together (shorter histories wait: their game and key don't
move), then the baseline agent plays every task at once. Agrees with the one-game runner (tested); minutes
instead of hours for a few hundred tasks.

Random actions use JAX's generator, so the random baseline's individual successes differ from the one-game
runner's (same distribution, different draws).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from jepa_game_agent.evaluation.agents import NOOP
from jepa_game_agent.evaluation.tasks import AchievementTask, ReachTask

AGENTS = ("random", "still", "replay")


def _env(name: str) -> tuple[Any, Any]:
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(name, auto_reset=False)
    return env, env.default_params


def _reset(env: Any, params: Any, seeds: Sequence[int]) -> tuple[Any, Any]:
    """Keys and states after reset, each world built on its own (batched reset depends on the batch size)."""
    import jax
    import jax.numpy as jnp

    keys, states = [], []
    reset = jax.jit(env.reset)
    for s in seeds:
        key, sub = jax.random.split(jax.random.PRNGKey(s))
        keys.append(key)
        states.append(reset(sub, params)[1])
    return jnp.stack(keys), jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def _state_keys(states: Any) -> Any:
    """(N, 15) int: position (2), facing (1), 12 inventory counts, as `CraftaxEnv.state_key`."""
    import dataclasses

    import jax.numpy as jnp

    inv = [getattr(states.inventory, f.name) for f in dataclasses.fields(states.inventory)]
    return jnp.concatenate([states.player_position, states.player_direction[:, None], jnp.stack(inv, -1)], -1).astype(
        jnp.int32
    )


def _stepper(env: Any, params: Any) -> Any:
    """One masked batched step: games where `active` is False keep their key and state."""
    import jax
    import jax.numpy as jnp

    split = jax.vmap(jax.random.split)
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def masked(keys: Any, states: Any, actions: Any, active: Any) -> Any:
        pair = split(keys)
        _obs, new_states, _r, dones, _info = step(pair[:, 1], states, actions, params)
        keys = jnp.where(active[:, None], pair[:, 0], keys)
        pick = lambda new, old: jnp.where(active.reshape(-1, *([1] * (new.ndim - 1))), new, old)  # noqa: E731
        return keys, jax.tree.map(pick, new_states, states), dones & active

    return masked


def _replay(env: Any, params: Any, keys: Any, states: Any, actions: list[tuple[int, ...]]) -> tuple[Any, Any]:
    """Advance game i by actions[i] (ragged), all together."""
    import jax
    import jax.numpy as jnp

    longest = max((len(a) for a in actions), default=0)
    if longest == 0:
        return keys, states
    padded = np.full((len(actions), longest), NOOP, dtype=np.int32)
    lengths = np.asarray([len(a) for a in actions])
    for i, a in enumerate(actions):
        padded[i, : len(a)] = a
    step = _stepper(env, params)

    def body(carry: Any, t: Any) -> Any:
        keys, states = carry
        keys, states, _ = step(keys, states, jnp.asarray(padded)[:, t], t < jnp.asarray(lengths))
        return (keys, states), None

    (keys, states), _ = jax.lax.scan(body, (keys, states), jnp.arange(longest))
    return keys, states


def _start_and_goal(name: str, tasks: Sequence[ReachTask]) -> tuple[Any, Any, Any, Any, Any]:
    env, params = _env(name)
    keys, states = _reset(env, params, [t.seed for t in tasks])
    keys, states = _replay(env, params, keys, states, [t.prefix for t in tasks])
    start_keys = _state_keys(states)
    _, goal_states = _replay(env, params, keys, states, [t.goal_actions for t in tasks])
    return env, params, (keys, states), start_keys, _state_keys(goal_states)


def nontrivial(name: str, tasks: Sequence[ReachTask]) -> list[bool]:
    """For each task: does reaching the goal require changing the state key at all?"""
    _, _, _, start, goal = _start_and_goal(name, tasks)
    return [bool(x) for x in np.asarray((start != goal).any(-1))]


def _play(
    env: Any,
    params: Any,
    keys: Any,
    states: Any,
    agent: str,
    steps: int,
    num_actions: int,
    seed: int,
    scripted: np.ndarray[Any, Any] | None,
    done_check: Any,
) -> np.ndarray[Any, Any]:
    """Run the agent for `steps`; return (N,) first step where done_check(states) holds (0 = never)."""
    import jax
    import jax.numpy as jnp

    step = _stepper(env, params)
    n = keys.shape[0]
    script = jnp.asarray(scripted if scripted is not None else np.zeros((n, 1), np.int32))

    def body(carry: Any, t: Any) -> Any:
        keys, states, alive, hit, pkey = carry
        pkey, akey = jax.random.split(pkey)
        if agent == "random":
            actions = jax.random.randint(akey, (n,), 0, num_actions)
        elif agent == "replay":
            actions = jnp.where(t < script.shape[1], script[:, jnp.minimum(t, script.shape[1] - 1)], NOOP)
        else:
            actions = jnp.full((n,), NOOP)
        keys, states, dones = step(keys, states, actions.astype(jnp.int32), alive)
        ok = alive & done_check(states) & (hit == 0)
        hit = jnp.where(ok, t + 1, hit)
        alive = alive & ~dones & (hit == 0)
        return (keys, states, alive, hit, pkey), None

    init = (keys, states, jnp.ones(n, bool), jnp.zeros(n, jnp.int32), jax.random.PRNGKey(seed))
    (_, _, _, hit, _), _ = jax.lax.scan(body, init, jnp.arange(steps))
    return np.asarray(hit)


def batch_reach(
    name: str, tasks: Sequence[ReachTask], agent: str, budget_factor: int = 2, seed: int = 0
) -> dict[str, Any]:
    """Reach tasks of one goal offset k for one baseline agent; same result format as `evaluate_reach`."""
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}")
    ks = {len(t.goal_actions) for t in tasks}
    if len(ks) != 1:
        raise ValueError("all tasks must share one goal offset k")
    k = ks.pop()
    env, params, (keys, states), start, goal = _start_and_goal(name, tasks)
    scripted = np.asarray([t.goal_actions for t in tasks], dtype=np.int32) if agent == "replay" else None
    hit = _play(
        env,
        params,
        keys,
        states,
        agent,
        budget_factor * k,
        int(env.action_space(params).n),
        seed,
        scripted,
        lambda s: (_state_keys(s) == goal).all(-1),
    )
    return {
        "family": "reach",
        "evaluator": "batch",
        "tasks": len(tasks),
        "budget_factor": budget_factor,
        "success_rate": float((hit > 0).mean()) if len(tasks) else 0.0,
        "successes": [bool(h > 0) for h in hit],
        "steps_to_success": [int(h) if h > 0 else None for h in hit],
        "start_keys": [tuple(int(v) for v in row) for row in np.asarray(start)],
        "goal_keys": [tuple(int(v) for v in row) for row in np.asarray(goal)],
    }


def batch_achievement(name: str, tasks: Sequence[AchievementTask], agent: str, seed: int = 0) -> dict[str, Any]:
    """Achievement tasks (fresh resets) for the random or still agent; same format as `evaluate_achievement`."""
    if agent not in ("random", "still"):
        raise ValueError("agent must be 'random' or 'still'")
    from jepa_game_agent.data.craftax_batch import _achievement_names

    names = _achievement_names(name)
    budgets = {t.budget for t in tasks}
    if len(budgets) != 1:
        raise ValueError("all tasks must share one budget")
    idx = np.asarray([names.index(t.achievement) for t in tasks])
    env, params = _env(name)
    keys, states = _reset(env, params, [t.seed for t in tasks])
    import jax.numpy as jnp

    col = jnp.asarray(idx)
    hit = _play(
        env,
        params,
        keys,
        states,
        agent,
        budgets.pop(),
        int(env.action_space(params).n),
        seed,
        None,
        lambda s: jnp.take_along_axis(s.achievements, col[:, None], 1)[:, 0],
    )
    return {
        "family": "achievement",
        "evaluator": "batch",
        "tasks": len(tasks),
        "success_rate": float((hit > 0).mean()) if len(tasks) else 0.0,
        "successes": [bool(h > 0) for h in hit],
        "steps_to_success": [int(h) if h > 0 else None for h in hit],
    }


def run_baselines_batch(
    name: str,
    episodes: Sequence[Any],
    ks: Sequence[int],
    tasks: int,
    achievements: Sequence[str] = (),
    resets: int = 50,
    budget: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """`baselines.run_baselines` for Craftax, every task at once. Trivial reach tasks are skipped."""
    from jepa_game_agent.evaluation.baselines import git_commit
    from jepa_game_agent.evaluation.tasks import make_reach_tasks

    reach: dict[str, dict[str, Any]] = {}
    for k in ks:
        candidates = make_reach_tasks(episodes, k=k, n=4 * tasks, seed=seed)
        keep = nontrivial(name, candidates)
        chosen = [t for t, ok in zip(candidates, keep, strict=True) if ok][:tasks]
        reach[f"k={k}"] = {agent: batch_reach(name, chosen, agent, seed=seed) for agent in AGENTS}
    ach: dict[str, dict[str, Any]] = {}
    for a in achievements:
        ach_tasks = [AchievementTask(a, seed=seed + i, budget=budget) for i in range(resets)]
        ach[a] = {agent: batch_achievement(name, ach_tasks, agent, seed=seed) for agent in ("random", "still")}
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
            "evaluator": "batch",
        },
        "commit": git_commit(),
    }
