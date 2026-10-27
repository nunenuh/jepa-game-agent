"""Craftax batch collector. Needs the craftax extra; the first call compiles for ~1 min."""

import numpy as np
import pytest

pytest.importorskip("craftax")

from jepa_game_agent.data.craftax_batch import collect_craftax_batch
from jepa_game_agent.data.episodes import collect_episode, random_policy
from jepa_game_agent.envs.craftax import CraftaxEnv

pytestmark = pytest.mark.craftax
NAME = "Craftax-Classic-Pixels-v1"


@pytest.fixture(scope="module")
def batch():
    return collect_craftax_batch(NAME, seeds=[3, 4, 5], max_steps=120, chunk=40, policy_seed=0)


def test_batch_shapes(batch) -> None:
    assert [e.seed for e in batch] == [3, 4, 5]
    for e in batch:
        assert 1 <= len(e) <= 120 and e.frames.shape[1:] == (63, 63, 3) and e.frames.dtype == np.uint8
        assert e.actions.min() >= 0 and e.actions.max() < 17


def test_batch_episode_replays_exactly(batch) -> None:
    """Same seed + same actions through the one-game wrapper gives the same game, terminal frame included.

    Batched rendering can round a shaded pixel differently, so frames may differ by at most 1 / 255.
    """
    env = CraftaxEnv(NAME)
    for e in batch:
        frames = [env.reset(e.seed)]
        for a in e.actions:
            frames.append(env.step(int(a)).frame)
        diff = np.abs(np.stack(frames).astype(int) - e.frames.astype(int))
        assert diff.max() <= 1


def test_terminal_step_keeps_the_old_world() -> None:
    """When the player dies, the env must not silently start a new world (Craftax's auto-reset does)."""
    env = CraftaxEnv(NAME)
    ep = collect_episode(env, random_policy(17), max_steps=3000, seed=0)
    assert len(ep) < 3000  # the random player died
    assert int(env.state.timestep) == len(ep)  # an auto-reset would have put the clock back to 0


def test_with_epsilon_mixes_in_random_actions() -> None:
    import jax
    import jax.numpy as jnp

    from jepa_game_agent.data.craftax_batch import with_epsilon

    always_zero = lambda frames, key: jnp.zeros(frames.shape[0], dtype=jnp.int32)  # noqa: E731
    frames = jnp.zeros((4000, 1, 1, 3))
    acts = np.asarray(with_epsilon(always_zero, 0.2, 17)(frames, jax.random.PRNGKey(0)))
    changed = (acts != 0).mean()
    assert 0.15 < changed < 0.22  # 20% replaced, 1/17 of those land on 0 again
    assert (np.asarray(with_epsilon(always_zero, 0.0, 17)(frames, jax.random.PRNGKey(0))) == 0).all()
    with pytest.raises(ValueError):
        with_epsilon(always_zero, 1.5, 17)


def test_set_state_stages_a_teleport() -> None:
    env = CraftaxEnv(NAME)
    before = env.reset(0)
    moved = env.state.replace(player_position=env.state.player_position + np.array([3, 0]))
    after = env.set_state(moved)
    assert int(env.state.player_position[0]) == int(moved.player_position[0])
    assert after.shape == before.shape and not np.array_equal(after, before)


def test_cli_collects_craftax_in_batches(tmp_path, capsys) -> None:
    import json

    from jepa_game_agent.cli import main
    from jepa_game_agent.data.episodes import load_episodes

    out = tmp_path / "cx.npz"
    main(["collect", "--env", NAME, "--episodes", "3", "--max-steps", "60", "--out", str(out)])
    card = json.loads(capsys.readouterr().out.strip().splitlines()[-1])  # JAX may print its own lines first
    eps = load_episodes(out)
    assert card["episodes"] == 3 and card["collector"] == "batch"
    assert card["transitions"] == sum(len(e) for e in eps)
    assert [e.seed for e in eps] == [0, 1, 2]
    assert set(card["achievements"]) == {a for e in eps for a in e.achievements}
