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


@pytest.mark.craftax
def test_craftax_zombie_tiles_match_where_the_zombie_is_drawn() -> None:
    """zombie_tiles(): (7, 9) view grid, player at (3, 4); a zombie one tile right of the player is at (3, 5),
    and that is the tile whose pixels change when it is drawn."""
    pytest.importorskip("craftax")
    import jax.numpy as jnp

    env = make_env("Craftax-Classic-Pixels-v1")
    env.reset(seed=0)
    st = env.state  # type: ignore[attr-defined]
    z = st.zombies
    none = z.replace(mask=jnp.zeros_like(z.mask))
    empty = env.set_state(st.replace(zombies=none))  # type: ignore[attr-defined]
    assert not env.zombie_tiles().any()  # type: ignore[attr-defined]
    pos = z.position.at[0].set(st.player_position + jnp.array([0, 1]))
    one = z.replace(position=pos, mask=jnp.zeros_like(z.mask).at[0].set(True))
    frame = env.set_state(st.replace(zombies=one))  # type: ignore[attr-defined]
    tiles = env.zombie_tiles()  # type: ignore[attr-defined]
    assert tiles.shape == (7, 9) and tiles.sum() == 1 and tiles[3, 5]
    diff = np.abs(frame.astype(int) - empty.astype(int)).sum(-1)
    assert diff[21:28, 35:42].sum() > 0 and diff[:49].sum() == diff[21:28, 35:42].sum()


@pytest.mark.craftax
def test_craftax_position_and_blocks_for_scoring_memory() -> None:
    """position() and block_map() (scoring only): moving right adds 1 to the column unless blocked, and the
    block map knows which tiles are trees."""
    pytest.importorskip("craftax")
    env = make_env("Craftax-Classic-Pixels-v1")
    env.reset(seed=0)
    start = env.position()  # type: ignore[attr-defined]
    blocks = env.block_map()  # type: ignore[attr-defined]
    env.step(2)  # RIGHT
    r, c = env.position()  # type: ignore[attr-defined]
    ahead = blocks[start[0], start[1] + 1]
    walkable = {env.block_id(n) for n in ("GRASS", "SAND", "PATH")}  # type: ignore[attr-defined]
    assert (r, c) == ((start[0], start[1] + 1) if ahead in walkable else start)
    assert blocks.shape == (64, 64) and (blocks == env.block_id("TREE")).any()  # type: ignore[attr-defined]


@pytest.mark.craftax
def test_craftax_reports_an_unlock_on_the_step_it_happens() -> None:
    """Craftax's own Achievements/ flags are only set on the step the game ends; ours must not wait for that."""
    pytest.importorskip("craftax")
    env = make_env("Craftax-Classic-Pixels-v1")
    env.reset(seed=0)
    st = env.state  # type: ignore[attr-defined]
    env.set_state(st.replace(inventory=st.inventory.replace(wood=9)))  # type: ignore[attr-defined]
    for turn in (1, 2, 3, 4):  # face each way until the table fits
        env.step(turn)
        res = env.step(8)  # PLACE_TABLE
        if env.inventory()["wood"] < 9:  # type: ignore[attr-defined]
            break
    assert env.inventory()["wood"] < 9  # type: ignore[attr-defined]
    assert "place_table" in res.achievements and not res.done
    assert "place_table" in env.step(0).achievements  # stays reported, like the game's own list
    st = env.state  # type: ignore[attr-defined]
    env.set_state(st.replace(player_health=0))  # type: ignore[attr-defined]
    last = env.step(0)
    assert last.done and "place_table" in last.achievements  # the step the game ends, too
    env.reset(seed=1)
    assert "place_table" not in env.step(0).achievements  # a new world starts clean
