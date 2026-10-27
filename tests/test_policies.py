"""PPO player exported as plain weights: save, load, play. Needs the craftax extra (JAX)."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("craftax")

from jepa_game_agent.data.craftax_batch import collect_craftax_batch
from jepa_game_agent.policies.mlp import load_mlp_policy, mlp_logits, save_actor_params, symbolic_obs_fn

pytestmark = pytest.mark.craftax
NAME = "Craftax-Classic-Pixels-v1"
SIZES = [1345, 16, 16, 16, 17]


def flax_like_params(rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    """Same layout as Craftax_Baselines' ActorCritic: Dense_0..3 actor, Dense_4..6 critic."""
    p = {
        f"Dense_{i}": {
            "kernel": rng.normal(size=(a, b)).astype(np.float32) * 0.1,
            "bias": rng.normal(size=b).astype(np.float32) * 0.1,
        }
        for i, (a, b) in enumerate(pairwise(SIZES))
    }
    p |= {f"Dense_{i}": {"kernel": np.zeros((16, 1), np.float32), "bias": np.zeros(1, np.float32)} for i in (4, 5, 6)}
    return p


def numpy_logits(params: dict[str, dict[str, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for i in range(3):
        x = np.tanh(x @ params[f"Dense_{i}"]["kernel"] + params[f"Dense_{i}"]["bias"])
    return x @ params["Dense_3"]["kernel"] + params["Dense_3"]["bias"]


def test_save_keeps_only_the_actor(tmp_path: Path) -> None:
    params = flax_like_params(np.random.default_rng(0))
    save_actor_params({"params": params}, tmp_path / "ppo.npz")
    with np.load(tmp_path / "ppo.npz") as d:
        assert sorted(d.files) == sorted(f"Dense_{i}/{w}" for i in range(4) for w in ("kernel", "bias"))


def test_logits_match_a_numpy_reference(tmp_path: Path) -> None:
    import jax.numpy as jnp

    params = flax_like_params(np.random.default_rng(1))
    save_actor_params({"params": params}, tmp_path / "ppo.npz")
    with np.load(tmp_path / "ppo.npz") as d:
        loaded = {k: jnp.asarray(d[k]) for k in d.files}
    x = np.random.default_rng(2).random((5, 1345)).astype(np.float32)
    assert np.allclose(np.asarray(mlp_logits(loaded, jnp.asarray(x))), numpy_logits(params, x), atol=1e-4)


def test_symbolic_obs_from_pixel_states() -> None:
    import jax

    from jepa_game_agent.envs.craftax import CraftaxEnv

    env = CraftaxEnv(NAME)
    env.reset(0)
    states = jax.tree.map(lambda x: np.stack([x, x]), env.state)
    assert symbolic_obs_fn(NAME)(states).shape == (2, 1345)


def test_mlp_policy_plays_in_the_batch_collector(tmp_path: Path) -> None:
    save_actor_params({"params": flax_like_params(np.random.default_rng(3))}, tmp_path / "ppo.npz")
    policy = load_mlp_policy(tmp_path / "ppo.npz", NAME)
    a = collect_craftax_batch(NAME, seeds=[0, 1], max_steps=15, chunk=15, policy=policy, policy_seed=0)
    b = collect_craftax_batch(NAME, seeds=[0, 1], max_steps=15, chunk=15, policy=policy, policy_seed=0)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.actions, y.actions) and x.actions.min() >= 0 and x.actions.max() < 17


def test_cli_collects_with_a_saved_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from jepa_game_agent.cli import main

    save_actor_params({"params": flax_like_params(np.random.default_rng(4))}, tmp_path / "ppo.npz")
    out = tmp_path / "cx.npz"
    main(
        [
            "collect",
            "--env",
            NAME,
            "--episodes",
            "2",
            "--max-steps",
            "10",
            "--policy",
            str(tmp_path / "ppo.npz"),
            "--epsilon",
            "0.2",
            "--out",
            str(out),
        ]
    )
    card = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert card["policy"] == str(tmp_path / "ppo.npz") and card["epsilon"] == 0.2 and out.exists()


def test_cli_epsilon_needs_a_policy(tmp_path: Path) -> None:
    from jepa_game_agent.cli import main

    with pytest.raises(SystemExit, match="needs --policy"):
        main(["collect", "--env", NAME, "--episodes", "1", "--epsilon", "0.2", "--out", str(tmp_path / "x.npz")])


def test_craftax_num_actions() -> None:
    from jepa_game_agent.data.craftax_batch import craftax_num_actions

    assert craftax_num_actions(NAME) == 17
