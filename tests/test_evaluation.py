import numpy as np
import pytest

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.fake import GridWorld
from jepa_game_agent.evaluation import (
    AchievementTask,
    RandomAgent,
    ReplayAgent,
    StillAgent,
    evaluate_achievement,
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
