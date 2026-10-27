"""Collect many Craftax games at once on the GPU (jax.vmap + lax.scan). Needs the craftax extra.

Each game uses the same key schedule as `envs.craftax.CraftaxEnv` (split once at reset, once per step),
so any episode collected here can be replayed there from its seed and actions: the same game, with
frames equal up to 1/255 (batched rendering can round a shaded pixel differently).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from jepa_game_agent.data.episodes import Episode

BatchPolicy = Callable[[Any, Any, Any], Any]  # (frames (N, H, W, C) in [0, 1], game states, jax key) -> actions (N,)


def craftax_num_actions(name: str) -> int:
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(name, auto_reset=False)
    return int(env.action_space(env.default_params).n)


def random_batch_policy(num_actions: int) -> BatchPolicy:
    import jax

    def policy(frames: Any, states: Any, key: Any) -> Any:
        return jax.random.randint(key, (frames.shape[0],), 0, num_actions)

    return policy


def with_epsilon(policy: BatchPolicy, epsilon: float, num_actions: int) -> BatchPolicy:
    """Replace each action by a random one with probability epsilon (variety in the data)."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    import jax
    import jax.numpy as jnp

    def noisy(frames: Any, states: Any, key: Any) -> Any:
        k_policy, k_coin, k_rand = jax.random.split(key, 3)
        actions = policy(frames, states, k_policy)
        coin = jax.random.uniform(k_coin, actions.shape) < epsilon
        return jnp.where(coin, jax.random.randint(k_rand, actions.shape, 0, num_actions), actions)

    return noisy


def _achievement_names(name: str) -> list[str]:
    if "Classic" in name:
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.constants import Achievement
    return [a.name.lower() for a in sorted(Achievement, key=lambda a: a.value)]


def collect_craftax_batch(
    name: str,
    seeds: Sequence[int],
    max_steps: int,
    policy: BatchPolicy | None = None,
    chunk: int = 100,
    policy_seed: int = 0,
) -> list[Episode]:
    """Play one game per seed, all together, until each dies or max_steps; return one Episode per seed.

    Frames move to the host as uint8 every `chunk` steps, so GPU memory stays bounded.
    """
    if not seeds or max_steps < 1 or chunk < 1:
        raise ValueError("need at least one seed, max_steps >= 1 and chunk >= 1")
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(name, auto_reset=False)
    params = env.default_params
    policy = policy or random_batch_policy(int(env.action_space(params).n))
    split = jax.vmap(jax.random.split)

    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    pair = split(keys)
    keys, subs = pair[:, 0], pair[:, 1]
    # One reset per game, then stack: Craftax's world generation under vmap depends on the batch size
    # (the same seed can get a different world in a batch of 32), which would break replay.
    reset = jax.jit(env.reset)
    worlds = [reset(subs[i], params) for i in range(len(seeds))]
    obs, states = jax.tree.map(lambda *xs: jnp.stack(xs), *worlds)

    def one_step(carry: Any, _: Any) -> Any:
        keys, states, obs, pkey = carry
        pkey, akey = jax.random.split(pkey)
        actions = policy(obs, states, akey).astype(jnp.int32)
        pair = split(keys)
        keys, subs = pair[:, 0], pair[:, 1]
        obs, states, rewards, dones, _info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(subs, states, actions, params)
        out = ((obs * 255).round().astype(jnp.uint8), actions, rewards, dones, states.achievements)
        return (keys, states, obs, pkey), out

    def scan_chunk(carry: Any) -> Any:
        return jax.lax.scan(one_step, carry, None, length=chunk)

    run_chunk = jax.jit(scan_chunk)

    first = np.asarray((obs * 255).round().astype(jnp.uint8))
    carry = (keys, states, obs, jax.random.PRNGKey(policy_seed))
    parts: list[tuple[np.ndarray[Any, Any], ...]] = []
    done_any = np.zeros(len(seeds), dtype=bool)
    steps = 0
    while steps < max_steps and not done_any.all():
        carry, out = run_chunk(carry)
        out_np = tuple(np.asarray(x) for x in out)
        parts.append(out_np)
        done_any |= out_np[3].any(0)
        steps += chunk

    frames, actions, rewards, dones, ach = (np.concatenate([p[i] for p in parts])[:max_steps] for i in range(5))
    names = _achievement_names(name)
    episodes = []
    for i, seed in enumerate(seeds):
        end = int(np.argmax(dones[:, i])) + 1 if dones[:, i].any() else len(dones)
        got = ach[:end, i]  # (T, 22) flags after each step
        first_step = np.where(got.any(0), got.argmax(0), np.iinfo(np.int64).max)
        unlocked = tuple(names[j] for j in np.argsort(first_step, kind="stable") if got[:, j].any())
        episodes.append(
            Episode(
                np.concatenate([first[i : i + 1], frames[:end, i]]),
                actions[:end, i].astype(np.int64),
                rewards[:end, i].astype(np.float32),
                seed=int(seed),
                achievements=unlocked,
            )
        )
    return episodes
