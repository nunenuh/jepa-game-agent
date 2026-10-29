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
    assert res["reached"] == [["b", "a", "c"]] and res["stage_rates"] == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_chain_reports_how_far_each_world_got() -> None:
    chain = [(n, [np.full((2, 2, 3), i, np.uint8)]) for i, n in enumerate(("a", "b"))]
    tasks = [AchievementTask("b", seed=s, budget=5) for s in range(2)]
    env = _ScriptedEnv({2: ("a",)})
    res = evaluate_chain(lambda: env, tasks, lambda t: _GoalLog(), chain)
    assert res["successes"] == [False, False] and res["success_rate"] == 0.0
    assert res["stage_rates"] == {"a": 1.0, "b": 0.0}
