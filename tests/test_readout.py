"""Body sense: read health, food, drink, energy off the latent (labels from replaying recorded games)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.readout import VITALS, Readout
from jepa_game_agent.training.readout_trainer import load_readout, replay_vitals, save_readout, train_readout


class _VitalEnv:
    """Health = 9 - number of steps taken (one per step), food = 5: a stand-in for Craftax's vitals."""

    def __init__(self) -> None:
        self.t = 0

    def reset(self, seed: int) -> np.ndarray:
        self.t = 0
        return np.zeros((2, 2, 3), np.uint8)

    def step(self, action: int) -> object:
        self.t += 1
        return None

    def vitals(self) -> dict[str, object]:
        return {"health": 9 - self.t, "food": 5, "drink": 4, "energy": 3, "sleeping": False}


def test_replay_vitals_gives_one_row_per_frame() -> None:
    ep = Episode(np.zeros((4, 2, 2, 3), np.uint8), np.zeros(3, np.int64), np.zeros(3, np.float32), seed=7)
    v = replay_vitals(_VitalEnv(), ep)
    assert v.shape == (4, len(VITALS)) and v[:, 0].tolist() == [9, 8, 7, 6] and v[0, 1] == 5


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
    monkeypatch.setattr(cli, "make_env", lambda name, seed=0: _VitalEnv())
    capsys.readouterr()
    cli.main(["train-readout", "--data", str(data), "--bottom", str(wm), "--env", "fake", "--episodes", "5",
              "--steps", "5", "--batch-size", "8", "--checkpoint", str(head)])  # fmt: skip
    summary = json.loads(capsys.readouterr().out)
    assert head.exists() and set(summary["r2"]) == {"health", "food", "drink", "energy"}
    monkeypatch.setattr(cli, "make_env", real_make_env)
    cli.main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
              "--readout", str(head), "--survival", "2.0", "--resets", "1", "--budget", "3", "--goal-frames", "2",
              "--horizon", "2", "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    cfg = json.loads(out.read_text())["config"]
    assert cfg["readout"] == str(head) and cfg["survival"] == 2.0
