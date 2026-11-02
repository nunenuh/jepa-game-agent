"""Own sense of progress: the status bar and the agent's own action decide 'stage done' (docs/DESIGN_RECIPE.md)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jepa_game_agent.evaluation import AchievementTask, evaluate_chain
from jepa_game_agent.evaluation.runner import ChainStage
from jepa_game_agent.evaluation.stages import StageTracker
from jepa_game_agent.models.progress import ProgressHead
from jepa_game_agent.training.progress_trainer import load_progress, replay_inventory, save_progress, train_progress

PLACE = 8
pic = [np.zeros((2, 2, 3), np.uint8)]
CHAIN = [ChainStage("collect_wood", pic, "wood", 2), ChainStage("place_table", pic),
         ChainStage("collect_wood", pic, "wood", 1), ChainStage("make_wood_pickaxe", pic)]  # fmt: skip


def tracker(readings: list[dict[str, float]], hold: int = 2) -> StageTracker:
    it = iter(readings)
    return StageTracker(CHAIN, lambda frame: next(it), {"place_table": PLACE}, hold=hold)


def test_counted_stage_is_done_once_the_reading_holds() -> None:
    t = tracker([{"wood": 0}, {"wood": 2.1}, {"wood": 1.9}, {"wood": 2.0}])
    t.start(pic[0])
    assert [t.update(pic[0], 5) for _ in range(3)] == [0, 1, 1]  # 2 readings in a row


def test_one_misread_does_not_move_the_stage() -> None:
    t = tracker([{"wood": 0}, {"wood": 2}, {"wood": 0}, {"wood": 0}])
    t.start(pic[0])
    assert [t.update(pic[0], 5) for _ in range(3)] == [0, 0, 0]


def test_a_table_is_placed_when_the_agent_tried_and_its_wood_dropped() -> None:
    wood = [2, 2, 2, 2, 2, 0, 0, 0]
    t = tracker([{"wood": w} for w in wood])
    t.start(pic[0])
    assert t.update(pic[0], 5) == 0
    assert t.update(pic[0], 5) == 1  # wood x 2 done, the table stage begins
    assert t.update(pic[0], PLACE) == 1  # tried, no room: wood unchanged
    assert t.update(pic[0], 5) == 1
    assert t.update(pic[0], PLACE) == 1  # wood fell: seen once
    assert t.update(pic[0], 5) == 2  # and it holds


def test_a_wood_drop_without_trying_is_no_table() -> None:
    wood = [2, 2, 2, 0, 0, 0]
    t = tracker([{"wood": w} for w in wood])
    t.start(pic[0])
    assert [t.update(pic[0], 5) for _ in range(5)] == [0, 1, 1, 1, 1]  # a misread or a craft: not this stage


def test_stages_it_cannot_sense_are_refused() -> None:
    with pytest.raises(ValueError, match="wake_up"):
        StageTracker([ChainStage("wake_up", pic)], lambda f: {}, {})
    with pytest.raises(ValueError, match="place_table"):
        StageTracker([ChainStage("place_table", pic)], lambda f: {}, {})


class _Env:
    """Wood from action 5, a table (costs 2) from action 8. The game's truth, for scoring only."""

    def __init__(self) -> None:
        self.wood, self.table = 0, False

    def reset(self, seed: int) -> np.ndarray:
        self.wood, self.table = 0, False
        return pic[0]

    def step(self, action: int) -> object:
        self.wood += action == 5
        if action == PLACE and self.wood >= 2:
            self.wood, self.table = self.wood - 2, True
        return SimpleNamespace(frame=pic[0], done=False, achievements=("place_table",) if self.table else ())


class _Script:
    def __init__(self) -> None:
        self.goals = 0
        self.seen: list[int] = []

    def reset(self, frames: list[np.ndarray]) -> None:
        self.goals += 1

    def act(self, frame: np.ndarray) -> int:
        return {2: PLACE}.get(self.goals, 5)  # the agent only knows the stage it was given


def test_the_agent_follows_its_own_sense_and_the_game_only_scores() -> None:
    env, agent = _Env(), _Script()
    chain = [ChainStage("collect_wood", pic, "wood", 2), ChainStage("place_table", pic)]
    own = lambda t: StageTracker(chain, lambda f: {"wood": env.wood}, {"place_table": PLACE})  # noqa: E731
    res = evaluate_chain(lambda: env, [AchievementTask("place_table", seed=0, budget=12)], lambda t: agent, chain,
                         counts=lambda e: {"wood": e.wood}, own_stage=own)  # type: ignore[attr-defined,arg-type,return-value]  # fmt: skip
    assert res["successes"] == [True]
    assert agent.goals == 2  # moved on by its own reading
    world = res["stage_timing"][0]
    assert world["true"][0] == 2 and world["own"][0] == 3  # wood x 2: the game at step 2, the agent 1 step later


def test_a_wrong_sense_shows_in_the_score() -> None:
    env, agent = _Env(), _Script()
    chain = [ChainStage("collect_wood", pic, "wood", 2), ChainStage("place_table", pic)]
    own = lambda t: StageTracker(chain, lambda f: {"wood": 0}, {"place_table": PLACE})  # noqa: E731  # blind
    res = evaluate_chain(lambda: env, [AchievementTask("place_table", seed=0, budget=6)], lambda t: agent, chain,
                         counts=lambda e: {"wood": e.wood}, own_stage=own)  # type: ignore[attr-defined,arg-type,return-value]  # fmt: skip
    assert res["successes"] == [False] and agent.goals == 1 and res["stage_timing"][0]["own"] == []


def test_head_reads_only_the_status_bar_tokens_of_a_spatial_latent() -> None:
    head = ProgressHead(latent_dim=9 * 9 * 4, items=("wood", "stone"), grid=9, token_dim=4)
    z = torch.randn(1, 9 * 9 * 4)
    far = z.clone().view(1, 9, 9, 4)
    far[0, :7] += 5.0  # the world view changes
    assert head(z).shape == (1, 2) and torch.allclose(head(z), head(far.reshape(1, -1)))
    bar = z.clone().view(1, 9, 9, 4)
    bar[0, 8] += 5.0  # the status bar changes
    assert not torch.allclose(head(z), head(bar.reshape(1, -1)))


def test_head_learns_counts_saves_and_loads(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    lat = [rng.normal(size=(60, 8)).astype(np.float32) for _ in range(4)]
    counts = [np.round(2 + z[:, :1]).astype(np.float32) for z in lat]
    head = ProgressHead(latent_dim=8, items=("wood",), hidden=32)
    history = train_progress(head, lat, counts, steps=300, batch_size=32)
    assert history[-1]["loss"] < 0.3 * history[0]["loss"]
    save_progress(tmp_path / "p.pt", head, history)
    again, _ = load_progress(tmp_path / "p.pt")
    assert again.items == ("wood",) and torch.allclose(again(torch.from_numpy(lat[0])), head(torch.from_numpy(lat[0])))
    assert head.read(torch.from_numpy(lat[0][0]))["wood"] == pytest.approx(
        float(head(torch.from_numpy(lat[0][:1]))[0, 0])
    )
    with pytest.raises(ValueError):
        train_progress(head, lat, counts[:1], steps=1)


def test_replay_inventory_pairs_each_replayed_frame_with_its_counts() -> None:
    from jepa_game_agent.data.episodes import Episode

    class E(_Env):
        def inventory(self) -> dict[str, int]:
            return {"wood": self.wood, "stone": 0}

        def step(self, action: int) -> object:
            super().step(action)
            return SimpleNamespace(frame=np.full((2, 2, 3), self.wood, np.uint8), done=self.wood >= 2, achievements=())

    ep = Episode(np.zeros((5, 2, 2, 3), np.uint8), np.full(4, 5, np.int64), np.zeros(4, np.float32), seed=1)
    frames, rows = replay_inventory(E(), ep, ("wood", "stone"))
    assert frames.shape == (3, 2, 2, 3) and rows[:, 0].tolist() == [0, 1, 2] and rows.shape == (3, 2)


def test_cli_train_progress_and_plan_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import jepa_game_agent.cli as cli
    from jepa_game_agent.data.episodes import load_episodes

    data, wm, head, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "p.pt", tmp_path / "a.json"
    cli.main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    cli.main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
              "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    shape = load_episodes(data)[0].frames.shape[1:]

    class Env:
        num_actions = 5
        frame_shape = shape

        def __init__(self) -> None:
            self.items = 0

        def reset(self, seed: int | None = None) -> np.ndarray:
            self.items = 0
            return np.zeros(shape, np.uint8)

        def step(self, action: int) -> object:
            self.items += action == 1
            return SimpleNamespace(frame=np.full(shape, min(self.items * 20, 255), np.uint8), done=False,
                                   achievements=("collect_item",) if self.items else (), reward=0.0)  # fmt: skip

        def inventory(self) -> dict[str, int]:
            return {"item": self.items}

    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: Env())
    capsys.readouterr()
    cli.main(["train-progress", "--data", str(data), "--bottom", str(wm), "--env", "fake", "--episodes", "5",
              "--steps", "5", "--batch-size", "8", "--split", "all", "--checkpoint", str(head)])  # fmt: skip
    summary = json.loads(capsys.readouterr().out)
    assert head.exists() and "item" in summary["right"] and "item" in summary["right_when_held"]
    cli.main(["plan-achievement", "--chain", "--progress", str(head), "--checkpoint", str(wm), "--goal-data", str(data),
              "--achievement", "collect_item", "--resets", "1", "--budget", "3", "--goal-frames", "2", "--horizon", "2",
              "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    assert res["config"]["progress"] == str(head) and "stage_timing" in res["chain"]["jepa"]
