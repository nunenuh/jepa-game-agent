"""Threat sense: a tiny head shared by every tile token says 'zombie on this tile?' (docs/AGENT_ARCHITECTURE.md)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.threat import ThreatHead, around
from jepa_game_agent.training.threat_trainer import (
    load_threat,
    replay_threats,
    save_threat,
    train_threat,
    view_to_grid,
)


def test_view_grid_goes_into_the_top_rows_of_the_token_grid() -> None:
    view = np.zeros((7, 9), bool)
    view[3, 5] = True
    g = view_to_grid(view, grid=9)
    assert g.shape == (81,) and g[3 * 9 + 5] == 1 and g.sum() == 1  # status-bar rows 7-8 stay 0


def test_around_lists_the_8_tiles_next_to_the_player() -> None:
    idx = around(grid=9, center=(3, 4))
    assert sorted(idx) == sorted(r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5) if (r, c) != (3, 4))


class _ZombieEnv:
    """A zombie walks in from the right: at step t it is (4 - t) tiles right of the player; game ends at `end`."""

    def __init__(self, end: int = 100) -> None:
        self.t, self.end = 0, end

    def reset(self, seed: int) -> np.ndarray:
        self.t = 0
        return np.zeros((2, 2, 3), np.uint8)

    def step(self, action: int) -> object:
        self.t += 1
        return SimpleNamespace(frame=np.full((2, 2, 3), self.t, np.uint8), done=self.t >= self.end)

    def zombie_tiles(self) -> np.ndarray:
        g = np.zeros((7, 9), bool)
        c = 4 + max(4 - self.t, 1)
        g[3, min(c, 8)] = True
        return g


def test_replay_threats_pairs_each_replayed_frame_with_its_zombie_map() -> None:
    ep = Episode(np.zeros((4, 2, 2, 3), np.uint8), np.zeros(3, np.int64), np.zeros(3, np.float32), seed=1)
    frames, maps = replay_threats(_ZombieEnv(end=2), ep, grid=9)
    assert frames.shape == (3, 2, 2, 3) and maps.shape == (3, 81)
    assert [int(np.argmax(m)) for m in maps] == [3 * 9 + 8, 3 * 9 + 7, 3 * 9 + 6]


def test_threat_head_is_shared_by_every_tile() -> None:
    head = ThreatHead(token_dim=4).eval()
    z = torch.randn(2, 5, 4).repeat(1, 1, 3)  # three identical tokens per latent
    out = head(z)
    assert out.shape == (2, 5, 3) and torch.allclose(out[..., 0], out[..., 2])


def test_threat_head_learns_which_tiles_hold_a_zombie(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    lat = [rng.normal(size=(50, 6 * 4)).astype(np.float32) for _ in range(4)]
    maps = [(x.reshape(50, 6, 4)[..., 0] > 0.5).astype(np.float32) for x in lat]  # zombie <=> token dim 0 high
    head = ThreatHead(token_dim=4)
    history = train_threat(head, lat, maps, steps=300, batch_size=32, seed=0)
    assert history[-1]["loss"] < 0.5 * history[0]["loss"]
    probe = torch.zeros(1, 6 * 4)
    probe[0, 0] = 3.0  # tile 0: dim 0 high
    p = torch.sigmoid(head(probe))[0]
    assert p[0] > 0.8 and p[1] < 0.3
    save_threat(tmp_path / "t.pt", head, history)
    again, _ = load_threat(tmp_path / "t.pt")
    assert torch.allclose(again(probe), head(probe))
    with pytest.raises(ValueError):
        train_threat(head, lat, maps[:1], steps=1)


def test_rank_auc_matches_the_pairwise_definition() -> None:
    from jepa_game_agent.training.threat_trainer import rank_auc

    scores, labels = np.array([0.1, 0.4, 0.35, 0.8, 0.4]), np.array([0, 0, 1, 1, 1])
    assert abs(rank_auc(scores, labels) - (1 + 0 + 2 + 1 + 0.5) / 6) < 1e-9  # ties count half


def test_cli_train_threat_and_plan_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    import jepa_game_agent.cli as cli
    from jepa_game_agent.data.episodes import load_episodes

    data, wm, head, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "t.pt", tmp_path / "a.json"
    cli.main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    cli.main(["train", "--data", str(data), "--num-actions", "5", "--grid", "4", "--token-dim", "4", "--width", "16",
              "--window", "4", "--steps", "1", "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    shape = load_episodes(data)[0].frames.shape[1:]

    class Env(_ZombieEnv):
        def reset(self, seed: int) -> np.ndarray:
            super().reset(seed)
            return np.zeros(shape, np.uint8)

        def step(self, action: int) -> object:
            res = super().step(action)
            return SimpleNamespace(frame=np.zeros(shape, np.uint8), done=res.done)  # type: ignore[attr-defined]

        def zombie_tiles(self) -> np.ndarray:
            g = np.zeros((3, 3), bool)
            g[1, 2] = self.t % 2 == 0
            return g

    real_make_env = cli.make_env
    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: Env())
    capsys.readouterr()
    cli.main(["train-threat", "--data", str(data), "--bottom", str(wm), "--env", "fake", "--episodes", "5",
              "--steps", "5", "--batch-size", "8", "--checkpoint", str(head)])  # fmt: skip
    summary = json.loads(capsys.readouterr().out)
    assert head.exists() and {"tile_auc", "adjacent_auc", "adjacent_share"} <= set(summary)
    monkeypatch.setattr(cli, "make_env", real_make_env)
    cli.main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
              "--threat", str(head), "--threat-weight", "2.0", "--resets", "1", "--budget", "3", "--goal-frames", "2",
              "--horizon", "2", "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    cfg = json.loads(out.read_text())["config"]
    assert cfg["threat"] == str(head) and cfg["threat_weight"] == 2.0
