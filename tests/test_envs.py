import numpy as np
import pytest

from jepa_game_agent.envs import GridWorld, make_env


def test_gridworld_is_deterministic() -> None:
    a, b = GridWorld(seed=1), GridWorld(seed=1)
    assert np.array_equal(a.reset(seed=3), b.reset(seed=3))
    for action in [1, 2, 3, 4, 0]:
        assert np.array_equal(a.step(action).frame, b.step(action).frame)


def test_gridworld_frame_and_moves(env: GridWorld) -> None:
    frame = env.reset(seed=0)
    assert frame.shape == env.frame_shape == (16, 16, 3) and frame.dtype == np.uint8
    env.set_agent(0, 0)
    env.item = (3, 3)
    assert env.step(1).info["agent_col"] == 0  # walls stop the agent
    assert env.step(2).info["agent_col"] == 1


def test_gridworld_reward_and_done() -> None:
    env = GridWorld(size=2, cell=1, max_steps=3, seed=0)
    env.reset(seed=0)
    env.set_agent(0, 0)
    env.item = (0, 1)
    result = env.step(2)
    assert result.reward == 1.0 and env.item != env.agent
    env.step(0)
    assert env.step(0).done


def test_gridworld_rejects_bad_input(env: GridWorld) -> None:
    with pytest.raises(ValueError):
        env.step(9)
    with pytest.raises(ValueError):
        GridWorld(size=1)


def test_make_env() -> None:
    assert isinstance(make_env("gridworld"), GridWorld)
    with pytest.raises(ValueError):
        make_env("pong")


@pytest.mark.craftax
def test_craftax_env() -> None:
    pytest.importorskip("craftax")
    env = make_env("Craftax-Classic-Pixels-v1")
    assert env.frame_shape == (63, 63, 3) and env.num_actions == 17
    result = env.step(0)
    assert result.frame.dtype == np.uint8 and len(result.info) == 22
    inv = env.inventory()  # type: ignore[attr-defined]
    assert len(inv) == 12 and inv["wood"] == 0 and all(isinstance(v, int) for v in inv.values())
    vit = env.vitals()  # type: ignore[attr-defined]
    assert vit["health"] == 9 and vit["in_lava"] is False and vit["zombies_near"] >= 0
    assert {"food", "drink", "energy", "sleeping", "light", "zombies_alive"} <= set(vit)
