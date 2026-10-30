import numpy as np
import pytest

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.base import StepResult
from jepa_game_agent.envs.fake import GridWorld
from jepa_game_agent.evaluation import (
    AchievementTask,
    RandomAgent,
    ReplayAgent,
    StillAgent,
    evaluate_achievement,
    evaluate_chain,
    evaluate_reach,
    make_reach_tasks,
)
from jepa_game_agent.evaluation.runner import ChainStage


def make_env() -> GridWorld:
    return GridWorld(size=4, cell=4, max_steps=200, seed=0)


def test_state_key_follows_the_agent() -> None:
    env = make_env()
    env.reset(seed=0)
    env.set_agent(1, 1)
    before = env.state_key()
    env.step(2)
    assert env.state_key() != before and env.state_key() == (1, 2)


def test_grid_item_counts_as_an_achievement() -> None:
    env = make_env()
    env.reset(seed=0)
    env.set_agent(0, 0)
    env.item = (0, 1)
    assert env.step(2).achievements == ("collect_item",)
    assert env.step(0).achievements == ()


def test_make_reach_tasks(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=10, seed=0)
    assert len(tasks) == 10
    for t in tasks:
        ep = next(e for e in episodes if e.seed == t.seed)
        assert len(t.goal_actions) == 5 and t.start + 5 <= len(ep)
        assert list(t.prefix) == ep.actions[: t.start].tolist()
        assert list(t.goal_actions) == ep.actions[t.start : t.start + 5].tolist()
    assert make_reach_tasks(episodes, k=5, n=10, seed=0) == tasks  # deterministic
    with pytest.raises(ValueError):
        make_reach_tasks(episodes, k=100, n=1, seed=0)  # no episode is long enough


def test_replaying_the_recorded_actions_always_reaches_the_goal(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=20, seed=1)
    res = evaluate_reach(make_env, tasks, lambda task: ReplayAgent(task.goal_actions))
    assert res["success_rate"] == 1.0 and res["successes"] == [True] * 20 and res["budget_factor"] == 2


def test_standing_still_only_wins_when_the_goal_is_the_start(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=20, seed=2)
    res = evaluate_reach(make_env, tasks, lambda task: StillAgent())
    assert res["successes"] == [s == g for s, g in zip(res["start_keys"], res["goal_keys"], strict=True)]


def test_random_agent_is_reproducible(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=10, seed=3)
    a = evaluate_reach(make_env, tasks, lambda task: RandomAgent(5, seed=0))
    b = evaluate_reach(make_env, tasks, lambda task: RandomAgent(5, seed=0))
    assert a["successes"] == b["successes"] and 0.0 <= a["success_rate"] <= 1.0


def test_achievement_tasks() -> None:
    class WalkToItem:
        """Scripted: read the agent and item positions from the env, walk there."""

        def __init__(self, env: GridWorld) -> None:
            self.env = env

        def reset(self, goal_frames: list[np.ndarray]) -> None:
            pass

        def act(self, frame: np.ndarray) -> int:
            (r, c), (ir, ic) = self.env.agent, self.env.item
            return 4 if ir > r else 3 if ir < r else 2 if ic > c else 1

    tasks = [AchievementTask("collect_item", seed=s, budget=20) for s in range(5)]
    env = make_env()
    walk = evaluate_achievement(lambda: env, tasks, lambda task: WalkToItem(env))
    still = evaluate_achievement(make_env, tasks, lambda task: StillAgent())
    assert walk["success_rate"] == 1.0 and still["success_rate"] == 0.0
    assert walk["steps_to_success"][0] is not None and still["steps_to_success"] == [None] * 5


def test_reach_tasks_can_skip_trivial_goals(episodes: list[Episode]) -> None:
    """With an env, tasks whose goal state equals the start state are dropped: standing still can't win."""
    tasks = make_reach_tasks(episodes, k=2, n=15, seed=0, env=make_env())
    assert len(tasks) == 15
    res = evaluate_reach(make_env, tasks, lambda task: StillAgent())
    assert res["success_rate"] == 0.0
    assert all(s != g for s, g in zip(res["start_keys"], res["goal_keys"], strict=True))


def test_reach_tasks_skip_goals_past_the_end_of_the_game() -> None:
    """A goal the replayed game can't reach alive (it ends first) is unusable: even replay would fail it."""
    from jepa_game_agent.data.episodes import collect_episode, random_policy
    from jepa_game_agent.evaluation.tasks import ReachTask, usable

    env = make_env()  # ends after max_steps=200
    ep = collect_episode(env, random_policy(5), max_steps=200, seed=7)
    acts = ep.actions.tolist()
    inside = ReachTask(7, 190, tuple(acts[:190]), tuple(acts[190:195]))
    past_end = ReachTask(7, 198, tuple(acts[:198]), (*acts[198:200], 2, 2, 2))
    assert usable(env, past_end) is False
    assert usable(env, inside) == (
        evaluate_reach(make_env, [inside], lambda t: StillAgent())["start_keys"][0]
        != evaluate_reach(make_env, [inside], lambda t: StillAgent())["goal_keys"][0]
    )
    res = evaluate_reach(make_env, [past_end], lambda task: ReplayAgent(task.goal_actions))
    assert res["success_rate"] == 0.0  # why such tasks must be skipped


class _ScriptedEnv:
    """Unlocks achievements on fixed step numbers: {step: names}."""

    num_actions = 1
    frame_shape = (2, 2, 3)

    def __init__(self, script: dict[int, tuple[str, ...]]) -> None:
        self.script, self.t = script, 0

    def reset(self, seed: int) -> np.ndarray:
        self.t = 0
        return np.zeros(self.frame_shape, np.uint8)

    def step(self, action: int) -> StepResult:
        self.t += 1
        return StepResult(np.zeros(self.frame_shape, np.uint8), 0.0, False, achievements=self.script.get(self.t, ()))


class _GoalLog:
    def __init__(self) -> None:
        self.goals: list[int] = []

    def reset(self, goal_frames: list[np.ndarray]) -> None:
        self.goals.append(int(goal_frames[0][0, 0, 0]))

    def act(self, frame: np.ndarray) -> int:
        return 0


def test_chain_moves_to_the_next_subgoal_and_skips_ones_already_unlocked() -> None:
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b", "c"))]
    log = _GoalLog()
    env = _ScriptedEnv({2: ("b",), 3: ("a",), 6: ("c",)})  # b unlocks early, while chasing a
    res = evaluate_chain(lambda: env, [AchievementTask("c", seed=0, budget=10)], lambda t: log, chain)
    assert log.goals == [0, 2]  # chased a, then c; never b (already had it)
    assert res["successes"] == [True] and res["steps_to_success"] == [6]
    assert res["reached"] == [["a", "b", "c"]] and res["unlocked"] == [["b", "a", "c"]]
    assert res["stage_rates"] == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_chain_reports_how_far_each_world_got() -> None:
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b"))]
    tasks = [AchievementTask("b", seed=s, budget=5) for s in range(2)]
    env = _ScriptedEnv({2: ("a",)})
    res = evaluate_chain(lambda: env, tasks, lambda t: _GoalLog(), chain)
    assert res["successes"] == [False, False] and res["success_rate"] == 0.0
    assert res["stage_rates"] == {"a": 1.0, "b": 0.0}


def test_chain_reports_progress_after_each_world() -> None:
    """progress(done, total, world): called after every world, so long runs show how far they got."""
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b"))]
    tasks = [AchievementTask("b", seed=s, budget=5) for s in range(3)]
    seen: list[tuple[int, int, int]] = []
    evaluate_chain(lambda: _ScriptedEnv({2: ("a",)}), tasks, lambda t: _GoalLog(), chain,
                   progress=lambda done, total, world: seen.append((done, total, world["length"])))  # fmt: skip
    assert [(d, t) for d, t, _ in seen] == [(1, 3), (2, 3), (3, 3)] and all(n == 5 for _, _, n in seen)


class _InventoryEnv(_ScriptedEnv):
    """Also scripts the wood count per step: {step: wood}."""

    def __init__(self, script: dict[int, tuple[str, ...]], wood: dict[int, int]) -> None:
        super().__init__(script)
        self.wood, self.now = wood, 0

    def reset(self, seed: int) -> np.ndarray:
        self.now = 0
        return super().reset(seed)

    def step(self, action: int) -> StepResult:
        result = super().step(action)
        self.now = self.wood.get(self.t, self.now)
        return result

    def inventory(self) -> dict[str, int]:
        return {"wood": self.now}


def test_count_stage_waits_for_enough_items() -> None:
    """A table needs 2 wood: chase wood until there are 2, then the table (which uses them up)."""
    wood_frames = [np.full((2, 2, 3), 0, np.uint8)]
    chain = [
        ChainStage("collect_wood", wood_frames, item="wood", count=2),
        ChainStage("place_table", [wood_frames[0] + 1]),
    ]
    log = _GoalLog()
    env = _InventoryEnv({1: ("collect_wood",), 5: ("place_table",)}, wood={1: 1, 3: 2, 5: 0})
    res = evaluate_chain(lambda: env, [AchievementTask("place_table", seed=0, budget=10)], lambda t: log, chain,
                         counts=lambda e: e.inventory())  # fmt: skip
    assert log.goals == [0, 1]  # wood until step 3 (2 wood), not at the first wood (step 1)
    assert res["successes"] == [True] and res["reached"] == [["collect_wood:2", "place_table"]]
    assert res["stage_rates"] == {"collect_wood:2": 1.0, "place_table": 1.0}


def test_count_stage_already_met_is_skipped_and_needs_counts() -> None:
    frames = [np.zeros((2, 2, 3), np.uint8)]
    chain = [ChainStage("collect_wood", frames, item="wood", count=1), ChainStage("place_table", [frames[0] + 1])]
    env = _InventoryEnv({}, wood={1: 3})
    log = _GoalLog()
    evaluate_chain(lambda: env, [AchievementTask("place_table", seed=0, budget=2)], lambda t: log, chain,
                   counts=lambda e: e.inventory())  # fmt: skip
    assert log.goals == [0, 1]
    with pytest.raises(ValueError):
        evaluate_chain(lambda: env, [AchievementTask("place_table", seed=0, budget=2)], lambda t: log, chain)


class _DyingEnv(_ScriptedEnv):
    """Ends the episode (the player dies) at step `death`."""

    def __init__(self, script: dict[int, tuple[str, ...]], death: int) -> None:
        super().__init__(script)
        self.death = death

    def step(self, action: int) -> StepResult:
        r = super().step(action)
        return StepResult(r.frame, r.reward, self.t >= self.death, r.info, r.achievements)


def test_chain_reports_episode_length_and_how_it_ended() -> None:
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b"))]
    tasks = [AchievementTask("b", seed=0, budget=10)]
    died = evaluate_chain(lambda: _DyingEnv({2: ("a",)}, death=4), tasks, lambda t: _GoalLog(), chain)
    assert died["lengths"] == [4] and died["ended"] == ["done"]
    won = evaluate_chain(lambda: _DyingEnv({2: ("a",), 3: ("b",)}, death=9), tasks, lambda t: _GoalLog(), chain)
    assert won["lengths"] == [3] and won["ended"] == ["success"]
    out = evaluate_chain(lambda: _DyingEnv({}, death=99), tasks, lambda t: _GoalLog(), chain)
    assert out["lengths"] == [10] and out["ended"] == ["budget"]


def test_chain_records_the_end_state_with_a_probe() -> None:
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b"))]
    tasks = [AchievementTask("b", seed=0, budget=10)]
    env = _DyingEnv({}, death=3)
    res = evaluate_chain(lambda: env, tasks, lambda t: _GoalLog(), chain, probe=lambda e: {"t": e.t})
    assert res["end_state"] == [{"t": 3}]
    assert evaluate_chain(lambda: env, tasks, lambda t: _GoalLog(), chain)["end_state"] == [None]
