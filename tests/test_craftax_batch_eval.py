"""Batched Craftax evaluation must agree with the one-game evaluator. Needs the craftax extra."""

import pytest

pytest.importorskip("craftax")

from jepa_game_agent.data.craftax_batch import collect_craftax_batch
from jepa_game_agent.envs.craftax import CraftaxEnv
from jepa_game_agent.evaluation import (
    AchievementTask,
    ReplayAgent,
    StillAgent,
    evaluate_reach,
    make_reach_tasks,
)
from jepa_game_agent.evaluation.craftax_batch_eval import batch_achievement, batch_reach, usable

pytestmark = pytest.mark.craftax
NAME = "Craftax-Classic-Pixels-v1"


@pytest.fixture(scope="module")
def tasks():
    eps = collect_craftax_batch(NAME, seeds=list(range(20, 28)), max_steps=60, chunk=60)
    return make_reach_tasks(eps, k=5, n=12, seed=0)  # no trivial filter: some tasks are trivial on purpose


@pytest.fixture(scope="module")
def single(tasks):
    env = CraftaxEnv(NAME)
    return {
        "still": evaluate_reach(lambda: env, tasks, lambda t: StillAgent()),
        "replay": evaluate_reach(lambda: env, tasks, lambda t: ReplayAgent(t.goal_actions)),
    }


def test_start_and_goal_states_match_the_one_game_evaluator(tasks, single) -> None:
    res = batch_reach(NAME, tasks, agent="still")
    assert [tuple(k) for k in res["start_keys"]] == [tuple(k) for k in single["still"]["start_keys"]]
    assert [tuple(k) for k in res["goal_keys"]] == [tuple(k) for k in single["still"]["goal_keys"]]
    assert res["evaluator"] == "batch" and res["budget_factor"] == 2 and res["tasks"] == len(tasks)


def test_still_and_replay_agree_with_the_one_game_evaluator(tasks, single) -> None:
    assert batch_reach(NAME, tasks, agent="still")["successes"] == single["still"]["successes"]
    replay = batch_reach(NAME, tasks, agent="replay")
    assert replay["success_rate"] == 1.0 and replay["successes"] == single["replay"]["successes"]


def test_random_is_reproducible(tasks) -> None:
    a = batch_reach(NAME, tasks, agent="random", seed=0)
    b = batch_reach(NAME, tasks, agent="random", seed=0)
    assert a["successes"] == b["successes"] and 0.0 <= a["success_rate"] <= 1.0
    with pytest.raises(ValueError):
        batch_reach(NAME, tasks, agent="dance")


def test_usable_matches_the_state_keys(tasks, single) -> None:
    expected = [s != g for s, g in zip(single["still"]["start_keys"], single["still"]["goal_keys"], strict=True)]
    assert usable(NAME, tasks) == expected
    assert not all(expected) and any(expected)  # the fixture has both kinds


def test_batch_achievement() -> None:
    tasks = [AchievementTask("collect_wood", seed=s, budget=150) for s in range(16)]
    still = batch_achievement(NAME, tasks, agent="still")
    assert still["success_rate"] == 0.0 and still["steps_to_success"] == [None] * 16
    a = batch_achievement(NAME, tasks, agent="random", seed=0)
    b = batch_achievement(NAME, tasks, agent="random", seed=0)
    assert a["successes"] == b["successes"] and a["success_rate"] > 0  # random play finds wood sometimes
    assert all((s is None) == (not ok) for s, ok in zip(a["steps_to_success"], a["successes"], strict=True))


def test_cli_baselines_uses_the_batch_evaluator_for_craftax(tmp_path, capsys) -> None:
    import json

    from jepa_game_agent.cli import main
    from jepa_game_agent.data.episodes import save_episodes

    eps = collect_craftax_batch(NAME, seeds=list(range(40, 52)), max_steps=80, chunk=80)
    save_episodes(tmp_path / "test.npz", eps)
    out = tmp_path / "b.json"
    main(
        [
            "baselines",
            "--data",
            str(tmp_path / "test.npz"),
            "--env",
            NAME,
            "--k",
            "5",
            "--tasks",
            "10",
            "--achievement",
            "collect_wood",
            "--resets",
            "8",
            "--budget",
            "100",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text())
    r = res["reach"]["k=5"]
    assert r["replay"]["evaluator"] == "batch" and r["replay"]["success_rate"] == 1.0
    assert r["still"]["success_rate"] == 0.0  # trivial tasks skipped by the batched check
    assert res["achievement"]["collect_wood"]["still"]["evaluator"] == "batch"
    assert res["config"]["evaluator"] == "batch"


def test_usable_skips_tasks_the_replayed_player_doesnt_survive() -> None:
    """Both evaluators stop when a game ends, so a task is unusable if the replayed player is dead at the
    start (found on Colab: GPU-collected games drift from their replays and die earlier) or dies before
    the goal step."""
    from jepa_game_agent.evaluation import ReachTask

    ep = collect_craftax_batch(NAME, seeds=[3], max_steps=2000, chunk=500)[0]  # random player: dies
    acts = ep.actions.tolist()
    n = len(acts)
    starts_dead = ReachTask(3, n, tuple(acts), (2, 3, 2, 3, 2))
    dies_on_the_way = ReachTask(3, n - 2, tuple(acts[: n - 2]), (*acts[n - 2 :], 2, 3, 2))
    starts_dead_one_step = ReachTask(3, n, tuple(acts), (1,))  # only the "dead at the start" rule catches k=1
    flags = usable(NAME, [starts_dead, dies_on_the_way, starts_dead_one_step])
    assert flags == [False, False, False]
