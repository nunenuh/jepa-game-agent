"""A PPO player trained with Craftax_Baselines (MIT), exported as plain weights. Needs the craftax extra.

Craftax_Baselines' `ActorCritic` is an MLP on the symbolic observation: three tanh layers then logits
(`Dense_0`..`Dense_3`; `Dense_4`.. are the critic, unused for acting). The player reads the symbolic view of
the game state while the dataset records pixels of the same state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jepa_game_agent.data.craftax_batch import BatchPolicy

ACTOR_LAYERS = 4


def save_actor_params(variables: dict[str, Any], path: str | Path) -> None:
    """Flax variables ({"params": {"Dense_0": {"kernel", "bias"}, ...}}) -> one .npz of the actor layers."""
    params = variables.get("params", variables)
    arrays = {
        f"Dense_{i}/{w}": np.asarray(params[f"Dense_{i}"][w]) for i in range(ACTOR_LAYERS) for w in ("kernel", "bias")
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]


def mlp_logits(params: dict[str, Any], x: Any) -> Any:
    import jax.numpy as jnp

    for i in range(ACTOR_LAYERS - 1):
        x = jnp.tanh(x @ params[f"Dense_{i}/kernel"] + params[f"Dense_{i}/bias"])
    return x @ params[f"Dense_{ACTOR_LAYERS - 1}/kernel"] + params[f"Dense_{ACTOR_LAYERS - 1}/bias"]


def symbolic_obs_fn(name: str) -> Any:
    """Batched game states of a pixel env -> the symbolic observations of the same states."""
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    sym = make_craftax_env_from_name(name.replace("Pixels", "Symbolic"), auto_reset=False)
    return jax.vmap(sym.get_obs)


def load_mlp_policy(path: str | Path, name: str) -> BatchPolicy:
    """Sample actions from the exported PPO actor, as a BatchPolicy for `collect_craftax_batch`."""
    import jax
    import jax.numpy as jnp

    with np.load(path) as d:
        params = {k: jnp.asarray(d[k]) for k in d.files}
    obs_of = symbolic_obs_fn(name)

    def policy(frames: Any, states: Any, key: Any) -> Any:
        return jax.random.categorical(key, mlp_logits(params, obs_of(states)))

    return policy
