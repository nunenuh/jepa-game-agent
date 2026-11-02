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


@pytest.mark.craftax
def test_the_chain_moves_on_the_moment_a_stage_is_done_in_the_real_game() -> None:
    """The real game through the real chain runner: the agent is given each stage right after finishing the last,
    while still alive. With unlocks only seen at death, it stayed on place_table forever."""
    pytest.importorskip("craftax")
    from craftax.craftax_classic.constants import Action

    from jepa_game_agent.evaluation import AchievementTask, evaluate_chain
    from jepa_game_agent.evaluation.runner import ChainStage

    env = make_env("Craftax-Classic-Pixels-v1")

    class Rich:
        """The real game, starting with wood to spend."""

        def reset(self, seed: int) -> np.ndarray:
            env.reset(seed)
            st = env.state  # type: ignore[attr-defined]
            return env.set_state(st.replace(inventory=st.inventory.replace(wood=9)))  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(env, name)

    class Crafter:
        """Turns and places a table until one fits, then makes a pickaxe. Stages are told apart by their goal frame."""

        def __init__(self) -> None:
            self.stages_given: list[int] = []
            self.t = 0

        def reset(self, frames: list[np.ndarray]) -> None:
            self.stages_given.append(int(frames[0][0, 0, 0]))

        def act(self, frame: np.ndarray) -> int:
            self.t += 1
            if self.stages_given[-1] == 2:
                return int(Action.MAKE_WOOD_PICKAXE.value)
            return int(Action.PLACE_TABLE.value) if self.t % 2 == 0 else 1 + (self.t // 2) % 4

    agent = Crafter()
    picture = lambda i: [np.full((2, 2, 3), i, np.uint8)]  # noqa: E731
    chain = [ChainStage("collect_wood", picture(0), "wood", 2), ChainStage("place_table", picture(1)),
             ChainStage("make_wood_pickaxe", picture(2))]  # fmt: skip
    res = evaluate_chain(lambda: Rich(), [AchievementTask("make_wood_pickaxe", seed=0, budget=30)],  # type: ignore[arg-type,return-value]
                         lambda t: agent, chain, counts=lambda e: e.inventory())  # type: ignore[attr-defined,arg-type,return-value]  # fmt: skip
    assert res["successes"] == [True] and res["ended"] == ["success"]
    assert agent.stages_given == [0, 1, 2]


@pytest.mark.craftax
def test_own_sense_catches_a_real_table_with_action_plus_wood_drop() -> None:
    pytest.importorskip("craftax")
    from jepa_game_agent.evaluation.runner import ChainStage
    from jepa_game_agent.evaluation.stages import StageTracker

    env = make_env("Craftax-Classic-Pixels-v1")
    frame = env.reset(seed=0)
    st = env.state  # type: ignore[attr-defined]
    env.set_state(st.replace(inventory=st.inventory.replace(wood=9)))  # type: ignore[attr-defined]
    place = env.action_id("PLACE_TABLE")  # type: ignore[attr-defined]
    pic = [frame]
    t = StageTracker([ChainStage("place_table", pic)], lambda f: env.inventory(), {"place_table": place})  # type: ignore[attr-defined]
    t.start(frame)
    stage, placed_at, seen_at = 0, None, None
    for i in range(20):
        action = place if i % 2 else 1 + (i // 2) % 4
        res = env.step(action)
        if placed_at is None and "place_table" in res.achievements:
            placed_at = i
        stage = t.update(res.frame, action)
        if stage == 1:
            seen_at = i
            break
    assert placed_at is not None and seen_at is not None and seen_at - placed_at <= 1
