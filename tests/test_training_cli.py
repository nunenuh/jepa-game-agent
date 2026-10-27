import json
from pathlib import Path

import pytest

from jepa_game_agent.cli import main
from jepa_game_agent.config import ModelConfig, TrainConfig
from jepa_game_agent.data.episodes import Episode, WindowDataset
from jepa_game_agent.models import WorldModel
from jepa_game_agent.training import load_checkpoint, train


def test_training_lowers_prediction_loss(episodes: list[Episode], small_cfg: ModelConfig) -> None:
    hist = train(WorldModel(small_cfg), WindowDataset(episodes, 4), TrainConfig(steps=60, batch_size=16, lr=1e-3))
    first = sum(h["prediction"] for h in hist[:5]) / 5
    last = sum(h["prediction"] for h in hist[-5:]) / 5
    assert len(hist) == 60 and last < first


def test_training_resumes_from_checkpoint(tmp_path: Path, episodes: list[Episode], small_cfg: ModelConfig) -> None:
    ckpt, data = tmp_path / "wm.pt", WindowDataset(episodes, 4)
    train(WorldModel(small_cfg), data, TrainConfig(steps=10, batch_size=16, checkpoint_every=5), checkpoint=ckpt)
    hist = train(WorldModel(small_cfg), data, TrainConfig(steps=15, batch_size=16, checkpoint_every=5), checkpoint=ckpt)
    assert [h["step"] for h in hist] == list(range(1, 16))
    model, meta = load_checkpoint(ckpt)
    assert meta["step"] == 15 and model.cfg == small_cfg


def test_cli_collect_then_train(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, ckpt = tmp_path / "eps.npz", tmp_path / "wm.pt"
    main(["collect", "--episodes", "3", "--max-steps", "20", "--out", str(data)])
    collected = json.loads(capsys.readouterr().out)
    assert collected["transitions"] == 60 and data.exists()
    main(
        [
            "train",
            "--data",
            str(data),
            "--num-actions",
            "5",
            "--latent-dim",
            "32",
            "--window",
            "4",
            "--steps",
            "4",
            "--batch-size",
            "8",
            "--checkpoint",
            str(ckpt),
            "--checkpoint-every",
            "2",
        ]
    )
    assert json.loads(capsys.readouterr().out)["steps"] == 4 and ckpt.exists()
