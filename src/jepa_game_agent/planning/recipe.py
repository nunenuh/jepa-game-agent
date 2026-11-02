"""The recipe, learned from recorded games: what must come before what, and how much it takes.

Labels (achievements, inventory) are used here offline only; nothing from the game reaches the agent while it
plays. A true prerequisite never fails, so one game without it disproves it. Habits of the player that recorded
the games pass that test too, so an order link must also survive games with random actions mixed in; one those
games can't test is reported, not trusted. See docs/DESIGN_RECIPE.md."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jepa_game_agent.data.episodes import Episode


@dataclass(frozen=True)
class GameTrace:
    unlocks: Mapping[str, int]  # achievement -> the step it first unlocked: inventory[t] before, inventory[t + 1] after
    inventory: Sequence[Mapping[str, int]]


@dataclass(frozen=True)
class Recipe:
    target: str
    stages: tuple[str, ...]  # as plan-achievement takes them, e.g. collect_wood:2, place_table
    needs: dict[str, dict[str, int]]  # the fewest of each item ever held at the unlock
    costs: dict[str, dict[str, int]]  # the least of each item ever lost at the unlock
    support: dict[str, int]  # how many games each number rests on
    links: tuple[tuple[str, str], ...]  # order links kept (not counting those items explain)
    dropped: tuple[tuple[str, str], ...]  # links disproved by the exploring games: habits
    unverified: tuple[tuple[str, str], ...]  # links the exploring games couldn't test: not used


def replay_trace(env: Any, episode: Episode) -> GameTrace:
    """Replay a recorded game for its labels, taken from the replay itself (Craftax replays can drift from the
    recording), stopping if the replayed game ends. Needs an env with `inventory()`."""
    env.reset(episode.seed)
    unlocks: dict[str, int] = {}
    inventory = [dict(env.inventory())]
    for t, a in enumerate(episode.actions):
        res = env.step(int(a))
        inventory.append(dict(env.inventory()))
        for name in res.achievements:
            unlocks.setdefault(name, t)
        if res.done:
            break
    return GameTrace(unlocks, inventory)


def _before(games: Sequence[GameTrace], a: str, b: str) -> bool:
    """In every game that unlocked b, a unlocked earlier."""
    return all(a in g.unlocks and g.unlocks[a] < g.unlocks[b] for g in games if b in g.unlocks)


def _item_of(name: str) -> str | None:
    for prefix in ("collect_", "make_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return None


def _needs_and_costs(games: Sequence[GameTrace], name: str) -> tuple[dict[str, int], dict[str, int]]:
    held = [(g.inventory[g.unlocks[name]], g.inventory[g.unlocks[name] + 1]) for g in games if name in g.unlocks]
    items = {i for before, _ in held for i in before}
    needs = {i: min(before.get(i, 0) for before, _ in held) for i in items}
    costs = {i: min(max(0, before.get(i, 0) - after.get(i, 0)) for before, after in held) for i in items}
    return {i: n for i, n in needs.items() if n > 0}, {i: n for i, n in costs.items() if n > 0}


def learn_recipe(games: Sequence[GameTrace], exploring: Sequence[GameTrace], target: str, min_games: int = 5) -> Recipe:
    """games: as recorded. exploring: with random actions mixed in, to tell a habit from a prerequisite."""
    if not exploring:
        raise ValueError("need exploring games (random actions mixed in) to tell habits from prerequisites")
    every = [*games, *exploring]
    names = sorted({n for g in every for n in g.unlocks})
    if target not in names:
        raise ValueError(f"{target!r} never unlocks in these games")
    support = {n: sum(n in g.unlocks for g in every) for n in names}
    needs, costs = {}, {}
    for n in names:
        needs[n], costs[n] = _needs_and_costs(every, n)
    producer = {_item_of(n): n for n in names if _item_of(n)}
    made_by = {n: {producer[i] for i in needs[n] if i in producer and producer[i] != n} for n in names}

    kept: set[tuple[str, str]] = set()
    dropped, unverified = set(), set()
    for a in names:
        for b in names:
            if a == b or a in made_by[b] or not any(b in g.unlocks for g in games) or not _before(games, a, b):
                continue
            if sum(b in g.unlocks for g in exploring) < min_games:
                unverified.add((a, b))
            elif _before(exploring, a, b):
                kept.add((a, b))
            else:
                dropped.add((a, b))

    after = {n: made_by[n] | {a for a, b in kept if b == n} for n in names}  # n's direct prerequisites

    def reaches(start: str, goal: str) -> bool:
        return any(p == goal or reaches(p, goal) for p in after[start])

    # minimal graph: a link another path already implies adds nothing
    links = {(a, b) for a, b in kept if not any(p != a and reaches(p, a) for p in after[b])}

    stages: list[str] = []
    have: dict[str, int] = {}
    done: set[str] = set()

    def plan(name: str) -> None:
        if name in done:
            return
        done.add(name)
        if support[name] < min_games:
            raise ValueError(f"{name!r} unlocks in only {support[name]} games, need {min_games}")
        for a in sorted(a for a, b in links if b == name):  # places and tools first: they may spend items
            plan(a)
        for item, n in sorted(needs[name].items()):
            maker = producer.get(item)
            if have.get(item, 0) >= n or maker is None:
                continue
            if maker.startswith("collect_"):
                stages.append(f"{maker}:{n}")
                have[item] = n
            else:
                plan(maker)
        stages.append(name)
        for item, n in costs[name].items():
            have[item] = have.get(item, 0) - n
        if (product := _item_of(name)) is not None:
            have[product] = have.get(product, 0) + 1

    plan(target)
    return Recipe(target, tuple(stages), needs, costs, support, tuple(sorted(links)), tuple(sorted(dropped)),
                  tuple(sorted(unverified)))  # fmt: skip


def save_recipe(path: str | Path, recipe: Recipe) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(recipe), indent=1))


def load_recipe(path: str | Path) -> Recipe:
    d = json.loads(Path(path).read_text())
    pairs = {k: tuple(tuple(p) for p in d[k]) for k in ("links", "dropped", "unverified")}
    return Recipe(d["target"], tuple(d["stages"]), d["needs"], d["costs"], d["support"], **pairs)
