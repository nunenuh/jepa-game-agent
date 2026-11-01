"""Memory: a map of the tile tokens the agent has seen, at positions it works out itself (docs/DESIGN_MEMORY.md)."""

import numpy as np
import pytest
import torch

from jepa_game_agent.memory.tilemap import TileMap, path_distance

D, G = 16, 9  # token size, token grid (the top 7 rows are the 7 x 9 world view, the player at (3, 4))


def world(seed: int = 0, size: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(size, size, D)).astype(np.float32)


def view(w: np.ndarray, pos: tuple[int, int]) -> torch.Tensor:
    """The flattened 9 x 9 token grid the encoder would give with the player at world tile pos."""
    grid = np.zeros((G, G, D), np.float32)
    grid[:7] = w[pos[0] - 3 : pos[0] + 4, pos[1] - 4 : pos[1] + 5]
    grid[7:] = 1.0  # status bar
    return torch.from_numpy(grid.reshape(-1))


MOVE = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # LEFT, RIGHT, UP, DOWN as (row, col)


def test_position_follows_moves_and_notices_blocked_ones() -> None:
    w, true = world(), (20, 20)
    m = TileMap(token_dim=D, grid=G)
    m.observe(view(w, true), action=None)
    rng = np.random.default_rng(1)
    for _ in range(60):
        a = int(rng.integers(0, 6))  # moves, plus NOOP (0) and DO (5) that never move
        if a in MOVE and rng.random() < 0.7:  # 30% of moves are blocked: the player only turns
            true = (true[0] + MOVE[a][0], true[1] + MOVE[a][1])
        m.observe(view(w, true), action=a)
        assert m.pos == (true[0] - 20, true[1] - 20)  # relative to where the world began


def test_flat_ground_counts_a_move_as_taken() -> None:
    """On ground that looks the same shifted or not, a move is assumed to have happened."""
    flat = np.ones((40, 40, D), np.float32)
    m = TileMap(token_dim=D, grid=G)
    m.observe(view(flat, (20, 20)), None)
    m.observe(view(flat, (20, 19)), action=1)
    assert m.pos == (0, -1)


def test_recall_finds_the_nearest_remembered_match_out_of_view() -> None:
    w = world()
    tree = np.full(D, 5.0, np.float32)
    w[20, 10] = tree  # far left of the start, out of view once the agent walks right
    w[20, 2] = tree  # further away
    m = TileMap(token_dim=D, grid=G)
    pos = (20, 12)
    m.observe(view(w, pos), None)
    assert m.recall(torch.from_numpy(tree), min_sim=0.95) is None  # in view: the goal picture handles it
    for _ in range(8):  # walk right until the tree leaves the view
        pos = (pos[0], pos[1] + 1)
        m.observe(view(w, pos), action=2)
    assert m.recall(torch.from_numpy(tree), min_sim=0.95) == (0, -2)  # world (20, 10), start was (20, 12)
    assert m.recall(-torch.from_numpy(tree), min_sim=0.95) is None  # never seen


def test_key_token_is_the_least_ordinary_tile_next_to_the_player() -> None:
    m = TileMap(token_dim=D, grid=G)
    goal = np.zeros((G, G, D), np.float32)
    goal[3, 5] = 7.0  # right of the player: a tree among plain tiles
    key = m.key(torch.from_numpy(goal.reshape(-1)))
    assert torch.allclose(key, torch.full((D,), 7.0))


def test_path_distance_sums_how_far_each_imagined_step_leaves_the_target() -> None:
    seqs = torch.tensor([[2, 2, 2], [1, 1, 1], [5, 0, 2]])  # right x3, left x3, no move, no move, right
    d = path_distance(seqs, pos=(0, 0), target=(0, 3))
    assert d.tolist() == [2 + 1 + 0, 4 + 5 + 6, 3 + 3 + 2]


def test_bad_input_is_rejected() -> None:
    m = TileMap(token_dim=D, grid=G)
    with pytest.raises(ValueError):
        m.observe(torch.zeros(10), None)
    with pytest.raises(ValueError):
        TileMap(token_dim=D, grid=5)  # the 7 x 9 view doesn't fit a 5 x 5 grid


class _WalkEnv:
    """A 40 x 40 world of random tile tokens; trees (block 1) where token dim 0 is very high. The frame only
    carries the player's position (pixel 0), so the fake encoder can show the right 9 x 9 token view."""

    num_actions = 6

    def __init__(self, w: np.ndarray, walls: set[tuple[int, int]]) -> None:
        self.w, self.walls, self.p = w, walls, (20, 20)

    def _frame(self) -> np.ndarray:
        f = np.zeros((4, 4, 3), np.uint8)
        f[0, 0, :2] = self.p
        return f

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.p = (20, 20)
        return self._frame()

    def step(self, action: int) -> object:
        from types import SimpleNamespace

        if action in MOVE:
            nxt = (self.p[0] + MOVE[action][0], self.p[1] + MOVE[action][1])
            self.p = self.p if nxt in self.walls else nxt
        return SimpleNamespace(frame=self._frame(), done=False)

    def position(self) -> tuple[int, int]:
        return self.p

    def block_map(self) -> np.ndarray:
        return (self.w[..., 0] > 3).astype(np.int64)


def test_memory_check_scores_position_shift_and_recall_against_the_game() -> None:
    from jepa_game_agent.data.episodes import Episode
    from jepa_game_agent.evaluation.memory_check import check_memory

    w = world()
    w[..., 0] = np.clip(w[..., 0], -3, 3)
    w[20, 24, 0] = 9.0  # one tree at the right edge of the first view, left behind on the walk back
    env = _WalkEnv(w, walls={(20, 23)})
    acts = np.array([2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.int64)  # right into a wall, back left
    ep = Episode(np.zeros((len(acts) + 1, 4, 4, 3), np.uint8), acts, np.zeros(len(acts), np.float32), seed=0)

    def encode(frames: np.ndarray) -> torch.Tensor:
        return torch.stack([view(w, (int(f[0, 0, 0]), int(f[0, 0, 1]))) for f in frames])

    goal = np.zeros((G, G, D), np.float32)
    goal[3, 3] = w[20, 24]  # the goal picture: the tree next to the player
    res = check_memory(env, [ep], encode, token_dim=D, grid=G, key_goal=torch.from_numpy(goal.reshape(-1)),
                       want=1, sim=0.95)  # fmt: skip
    assert res["shift_accuracy"] == 1.0 and res["position_error_median"] == 0
    assert res["recall_fired"] > 0 and res["recall_precision"] == 1.0


def test_cli_plan_with_memory(tmp_path: object) -> None:
    import json
    from pathlib import Path

    from jepa_game_agent.cli import main

    tmp = Path(str(tmp_path))
    data, wm, out = tmp / "eps.npz", tmp / "wm.pt", tmp / "a.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--grid", "9", "--token-dim", "2", "--width", "8",
          "--window", "4", "--steps", "1", "--batch-size", "2", "--checkpoint", str(wm)])  # fmt: skip
    main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
          "--memory", "2", "--resets", "1", "--budget", "3", "--goal-frames", "2", "--horizon", "2",
          "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    assert json.loads(out.read_text())["config"]["memory"] == 2.0


def test_cli_memory_check(tmp_path: object, monkeypatch: pytest.MonkeyPatch,
                          capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    import json
    from pathlib import Path

    import jepa_game_agent.cli as cli
    from jepa_game_agent.data.episodes import load_episodes

    tmp = Path(str(tmp_path))
    data, wm, out = tmp / "eps.npz", tmp / "wm.pt", tmp / "m.json"
    cli.main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    cli.main(["train", "--data", str(data), "--num-actions", "5", "--grid", "9", "--token-dim", "2", "--width", "8",
              "--window", "4", "--steps", "1", "--batch-size", "2", "--checkpoint", str(wm)])  # fmt: skip
    shape = load_episodes(data)[0].frames.shape[1:]

    class Env(_WalkEnv):
        def _frame(self) -> np.ndarray:
            return np.full(shape, self.p[1], np.uint8)

        @staticmethod
        def block_id(name: str) -> int:
            return {"TREE": 1}[name]

    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: Env(world(), walls=set()))
    capsys.readouterr()
    cli.main(["memory-check", "--checkpoint", str(wm), "--data", str(data), "--env", "fake", "--episodes", "2",
              "--goal-data", str(data), "--achievement", "collect_item", "--block", "TREE",
              "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    assert res["episodes"] == 2 and {"shift_accuracy", "position_error_median", "recall_precision"} <= set(res)
    assert json.loads(capsys.readouterr().out) == res


def test_key_token_from_several_goal_pictures_follows_the_tree_whichever_side_it_is_on() -> None:
    """Goal pictures face the tree from different sides: averaging the pictures first would blur it away, so
    each picture gives its own key tile and those are averaged."""
    m = TileMap(token_dim=D, grid=G)
    goals = np.zeros((2, G, G, D), np.float32)
    goals[0, 3, 5] = 7.0  # tree to the right
    goals[1, 2, 4] = 7.0  # tree above
    key = m.key(torch.from_numpy(goals.reshape(2, -1)))
    assert torch.allclose(key, torch.full((D,), 7.0))


def test_the_players_own_tile_is_never_remembered() -> None:
    """The player sprite covers its tile: writing it would leave a trail of 'player' tiles across the map
    (which recall then mistakes for things)."""
    m = TileMap(token_dim=D, grid=G)
    m.observe(view(world(), (20, 20)), None)
    assert len(m) == 62 and (0, 0) not in m._tiles
