"""The recipe, learned from recorded games: order, counts and a filter for habits (docs/DESIGN_RECIPE.md)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.planning.recipe import GameTrace, learn_recipe, load_recipe, replay_trace, save_recipe


def play(rng: np.random.Generator, habit: bool, steps: int = 60) -> GameTrace:
    """A made-up game with a known recipe: a table costs 2 wood; a pickaxe costs 1 wood next to a table; stone needs
    the pickaxe. habit: the player always wakes up and picks a sapling first, though nothing needs them."""
    inv = {"wood": 0, "stone": 0, "sapling": 0, "wood_pickaxe": 0}
    table = False
    unlocks: dict[str, int] = {}
    history = [dict(inv)]
    for t in range(steps):
        legal = ["collect_wood", "collect_wood", "wake_up", "collect_sapling"]
        legal += ["place_table"] if inv["wood"] >= 2 and not table else []
        legal += ["make_wood_pickaxe"] if table and inv["wood"] >= 1 and not inv["wood_pickaxe"] else []
        legal += ["collect_stone"] if inv["wood_pickaxe"] else []
        act = {0: "wake_up", 1: "collect_sapling"}.get(t) if habit else None
        act = act or str(rng.choice(legal))
        unlocks.setdefault(act, t)
        if act == "place_table":
            inv["wood"] -= 2
            table = True
        elif act == "make_wood_pickaxe":
            inv["wood"] -= 1
            inv["wood_pickaxe"] = 1
        elif act.startswith("collect_"):
            inv[act.removeprefix("collect_")] += 1
        history.append(dict(inv))
    return GameTrace(unlocks, history)


def games(habit: bool, n: int = 60, seed: int = 0) -> list[GameTrace]:
    rng = np.random.default_rng(seed)
    return [play(rng, habit) for _ in range(n)]


HAND_WRITTEN = ("collect_wood:2", "place_table", "collect_wood:1", "make_wood_pickaxe", "collect_stone")


def test_the_recipe_comes_out_the_same_as_the_hand_written_one() -> None:
    r = learn_recipe(games(habit=True), games(habit=False, seed=1), "collect_stone")
    assert r.stages == HAND_WRITTEN
    assert r.needs["place_table"] == {"wood": 2} and r.costs["place_table"] == {"wood": 2}
    assert r.needs["make_wood_pickaxe"] == {"wood": 1} and r.needs["collect_stone"] == {"wood_pickaxe": 1}
    assert r.support["collect_stone"] > 0


def test_a_habit_looks_like_a_step_until_exploring_games_show_it_is_not() -> None:
    r = learn_recipe(games(habit=True), games(habit=False, seed=1), "place_table")
    assert r.stages == ("collect_wood:2", "place_table")
    assert not {"wake_up", "collect_sapling"} & {s.partition(":")[0] for s in r.stages}
    assert ("wake_up", "place_table") in r.dropped  # seen, tested, thrown out: reported


def test_a_link_that_exploring_games_cannot_test_is_reported_not_trusted() -> None:
    rng = np.random.default_rng(1)
    short = [play(rng, habit=False, steps=3) for _ in range(20)]  # too short to ever make a pickaxe
    r = learn_recipe(games(habit=True), short, "make_wood_pickaxe")
    assert ("place_table", "make_wood_pickaxe") in r.unverified
    assert "place_table" not in r.stages  # so the recipe comes out short, and the report says why


def test_too_little_data_is_refused_not_guessed() -> None:
    few = games(habit=True, n=2)
    with pytest.raises(ValueError, match="collect_stone"):
        learn_recipe(few, games(habit=False, n=2, seed=1), "collect_stone", min_games=5)
    with pytest.raises(ValueError, match="never"):
        learn_recipe(games(habit=True), games(habit=False, seed=1), "collect_diamond")
    with pytest.raises(ValueError, match="exploring"):
        learn_recipe(games(habit=True), [], "collect_stone")


def test_replay_trace_keeps_first_unlocks_and_the_inventory_before_and_after() -> None:
    class Env:
        def __init__(self) -> None:
            self.t = 0

        def reset(self, seed: int) -> np.ndarray:
            self.t = 0
            return np.zeros((2, 2, 3), np.uint8)

        def step(self, action: int) -> object:
            self.t += 1
            got = ("collect_wood",) if self.t >= 2 else ()  # the game keeps reporting what is unlocked
            return SimpleNamespace(frame=np.zeros((2, 2, 3), np.uint8), done=self.t >= 3, achievements=got)

        def inventory(self) -> dict[str, int]:
            return {"wood": int(self.t >= 2)}

    ep = Episode(np.zeros((6, 2, 2, 3), np.uint8), np.zeros(5, np.int64), np.zeros(5, np.float32), seed=1)
    tr = replay_trace(Env(), ep)
    assert tr.unlocks == {"collect_wood": 1}  # step index: inventory[1] is before it, inventory[2] after
    assert [i["wood"] for i in tr.inventory] == [0, 0, 1, 1]  # stopped when the replayed game ended


def test_recipe_saves_and_loads(tmp_path: Path) -> None:
    r = learn_recipe(games(habit=True), games(habit=False, seed=1), "collect_stone")
    save_recipe(tmp_path / "r.json", r)
    assert load_recipe(tmp_path / "r.json") == r
    assert json.loads((tmp_path / "r.json").read_text())["stages"] == list(HAND_WRITTEN)


def test_cli_learn_recipe_then_plan_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import jepa_game_agent.cli as cli

    data, wm, recipe, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "r.json", tmp_path / "a.json"
    cli.main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    cli.main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
              "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip

    class Env:
        def __init__(self) -> None:
            self.items = 0

        def reset(self, seed: int) -> np.ndarray:
            self.items = 0
            return np.zeros((2, 2, 3), np.uint8)

        def step(self, action: int) -> object:
            self.items += 1
            return SimpleNamespace(frame=np.zeros((2, 2, 3), np.uint8), done=False, achievements=("collect_item",))

        def inventory(self) -> dict[str, int]:
            return {"item": self.items}

    real = cli.make_env
    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: Env())
    capsys.readouterr()
    cli.main(["learn-recipe", "--data", str(data), "--explore-data", str(data), "--env", "fake", "--split", "all",
              "--target", "collect_item", "--out", str(recipe)])  # fmt: skip
    assert json.loads(capsys.readouterr().out)["stages"] == ["collect_item"]
    monkeypatch.setattr(cli, "make_env", real)
    cli.main(["plan-achievement", "--chain", "--checkpoint", str(wm), "--goal-data", str(data), "--recipe", str(recipe),
              "--resets", "1", "--budget", "2", "--goal-frames", "2", "--horizon", "2", "--samples", "8",
              "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    assert res["chain"]["jepa"]["chain"] == ["collect_item"] and res["config"]["recipe"] == str(recipe)
    with pytest.raises(ValueError, match="achievement"):
        cli.main(["plan-achievement", "--chain", "--checkpoint", str(wm), "--goal-data", str(data),
                  "--out", str(out)])  # fmt: skip
