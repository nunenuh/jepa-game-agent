"""Body sense: read health, food, drink, energy off the latent (labels from replaying recorded games)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.readout import VITALS, Readout
from jepa_game_agent.training.readout_trainer import (
    danger_labels,
    load_readout,
    replay_vitals,
    save_readout,
    train_readout,
)


class _VitalEnv:
    """Health = 9 - number of steps taken (one per step), food = 5: a stand-in for Craftax's vitals.
    The frame shows the step count; the game ends after `end` steps."""

    def __init__(self, end: int = 100, shape: tuple[int, ...] = (2, 2, 3)) -> None:
        self.t, self.end, self.shape = 0, end, shape

    def reset(self, seed: int) -> np.ndarray:
        self.t = 0
        return np.zeros(self.shape, np.uint8)

    def step(self, action: int) -> object:
        from types import SimpleNamespace

        self.t += 1
        return SimpleNamespace(frame=np.full(self.shape, self.t, np.uint8), done=self.t >= self.end)

    def vitals(self) -> dict[str, object]:
        return {"health": 9 - self.t, "food": 5, "drink": 4, "energy": 3, "sleeping": False}


def test_replay_vitals_gives_one_row_per_replayed_frame() -> None:
    """Labels go with the replayed frames, not the recorded ones (a replay can drift from the recording)."""
    ep = Episode(np.zeros((4, 2, 2, 3), np.uint8), np.zeros(3, np.int64), np.zeros(3, np.float32), seed=7)
    frames, v = replay_vitals(_VitalEnv(), ep)
    assert v.shape == (4, len(VITALS)) and v[:, 0].tolist() == [9, 8, 7, 6] and v[0, 1] == 5
    assert frames.shape == (4, 2, 2, 3) and frames[:, 0, 0, 0].tolist() == [0, 1, 2, 3]


def test_replay_vitals_stops_when_the_replayed_game_ends() -> None:
    ep = Episode(np.zeros((6, 2, 2, 3), np.uint8), np.zeros(5, np.int64), np.zeros(5, np.float32), seed=7)
    frames, v = replay_vitals(_VitalEnv(end=2), ep)
    assert len(frames) == len(v) == 3 and v[:, 0].tolist() == [9, 8, 7]


def test_danger_is_the_health_lost_within_the_next_steps() -> None:
    """The critic's target: how much health the player loses in the next `ahead` steps (never negative)."""
    health = np.array([9, 9, 8, 6, 6, 7, 0], np.float32)
    vitals = np.stack([health, health * 0, health * 0, health * 0], -1)
    d = danger_labels(vitals, ahead=2)
    assert d.shape == (7, 1) and d[:, 0].tolist() == [1, 3, 2, 0, 6, 7, 0]


def test_a_critic_head_has_one_output_and_reloads(tmp_path: Path) -> None:
    head = Readout(latent_dim=4, outputs=1)
    assert head(torch.zeros(3, 4)).shape == (3, 1)
    save_readout(tmp_path / "c.pt", head, [])
    again, _ = load_readout(tmp_path / "c.pt")
    assert again(torch.zeros(1, 4)).shape == (1, 1)


def test_readout_learns_vitals_from_latents(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    z = [rng.normal(size=(60, 4)).astype(np.float32) for _ in range(6)]
    labels = [np.stack([5 + 2 * x[:, 0], 5 - x[:, 1], 4 + 0 * x[:, 0], 3 + x[:, 2]], -1) for x in z]
    head = Readout(latent_dim=4)
    history = train_readout(head, z, labels, steps=400, batch_size=64, seed=0)
    assert history[-1]["loss"] < 0.1 * history[0]["loss"]
    pred = head(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert pred.shape == (1, len(VITALS)) and abs(pred[0, 0].item() - 7.0) < 0.5
    save_readout(tmp_path / "r.pt", head, history)
    again, ckpt = load_readout(tmp_path / "r.pt")
    assert torch.allclose(again(torch.zeros(1, 4)), head(torch.zeros(1, 4))) and ckpt["history"] == history
    with pytest.raises(ValueError):
        train_readout(head, z, labels[:1], steps=1)


def test_cli_train_readout_and_plan_with_survival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    import jepa_game_agent.cli as cli

    data, wm, head, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "r.pt", tmp_path / "a.json"
    cli.main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    cli.main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4",
              "--steps", "1", "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    real_make_env = cli.make_env
    from jepa_game_agent.data.episodes import load_episodes

    shape = load_episodes(data)[0].frames.shape[1:]
    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: _VitalEnv(shape=shape))
    capsys.readouterr()
    cli.main(["train-readout", "--data", str(data), "--bottom", str(wm), "--env", "fake", "--episodes", "5",
              "--steps", "5", "--batch-size", "8", "--checkpoint", str(head)])  # fmt: skip
    summary = json.loads(capsys.readouterr().out)
    assert head.exists() and set(summary["r2"]) == {"health", "food", "drink", "energy"}
    critic = tmp_path / "c.pt"
    cli.main(["train-readout", "--data", str(data), "--bottom", str(wm), "--env", "fake", "--episodes", "5",
              "--steps", "5", "--batch-size", "8", "--danger", "3", "--checkpoint", str(critic)])  # fmt: skip
    assert set(json.loads(capsys.readouterr().out)["r2"]) == {"danger"}
    monkeypatch.setattr(cli, "make_env", real_make_env)
    cli.main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
              "--readout", str(head), "--survival", "2.0", "--readout-model", str(wm),
              "--critic", str(critic), "--danger", "1.5", "--resets", "1", "--budget", "3",
              "--goal-frames", "2",
              "--horizon", "2", "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    cfg = json.loads(out.read_text())["config"]
    assert cfg["readout"] == str(head) and cfg["survival"] == 2.0 and cfg["readout_model"] == str(wm)
    assert cfg["critic"] == str(critic) and cfg["danger"] == 1.5
