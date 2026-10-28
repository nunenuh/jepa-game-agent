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


def test_cli_collect_reports_a_data_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["collect", "--episodes", "2", "--max-steps", "10", "--out", str(tmp_path / "g.npz")])
    card = json.loads(capsys.readouterr().out)
    assert card["collector"] == "single" and card["achievements"] == {}
    assert card["length"] == {"min": 10, "mean": 10.0, "max": 10}


def test_cli_baselines_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, out = tmp_path / "g.npz", tmp_path / "baselines.json"
    main(["collect", "--episodes", "4", "--max-steps", "40", "--out", str(data)])
    capsys.readouterr()
    main(
        [
            "baselines",
            "--data",
            str(data),
            "--env",
            "gridworld",
            "--k",
            "3",
            "5",
            "--tasks",
            "8",
            "--achievement",
            "collect_item",
            "--resets",
            "4",
            "--budget",
            "30",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text())
    assert set(res["reach"]) == {"k=3", "k=5"}
    for k in ("k=3", "k=5"):
        assert set(res["reach"][k]) == {"random", "still", "replay"}
        assert res["reach"][k]["replay"]["success_rate"] == 1.0
        assert res["reach"][k]["still"]["success_rate"] == 0.0  # trivial tasks (goal == start) are skipped
    assert set(res["achievement"]["collect_item"]) == {"random", "still"}
    assert res["config"]["tasks"] == 8 and "commit" in res
    summary = json.loads(capsys.readouterr().out)
    assert summary["reach"]["k=3"]["replay"] == 1.0


def test_cli_train_reads_many_files_and_keeps_one_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    main(["collect", "--episodes", "2", "--max-steps", "10", "--seed", "0", "--out", str(a)])  # seeds 0 (test), 1 (val)
    main(["collect", "--episodes", "2", "--max-steps", "10", "--seed", "2", "--out", str(b)])  # seeds 2, 3 (train)
    capsys.readouterr()
    args = ["train", "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1", "--batch-size", "4"]
    main([*args, "--data", str(a), str(b), "--split", "train"])
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 2 and out["transitions"] == 20 and out["split"] == "train"
    main([*args, "--data", str(a), str(b)])  # default: every episode
    assert json.loads(capsys.readouterr().out)["episodes"] == 4


def test_cli_plan_scores_a_checkpoint_on_the_baseline_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, ckpt, out, base = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "plan.json", tmp_path / "b.json"
    main(["collect", "--episodes", "6", "--max-steps", "30", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "3",
          "--batch-size", "8", "--checkpoint", str(ckpt)])  # fmt: skip
    main(["baselines", "--data", str(data), "--k", "3", "--tasks", "5", "--out", str(base)])
    capsys.readouterr()
    main(["plan", "--checkpoint", str(ckpt), "--data", str(data), "--k", "3", "--tasks", "5", "--horizon", "3",
          "--samples", "16", "--elites", "4", "--iterations", "2", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    jepa, replay = res["reach"]["k=3"]["jepa"], json.loads(base.read_text())["reach"]["k=3"]["replay"]
    assert jepa["tasks"] == 5 and 0.0 <= jepa["success_rate"] <= 1.0
    assert jepa["goal_keys"] == replay["goal_keys"]  # the very tasks the baselines were scored on
    assert res["config"]["planner"]["horizon"] == 3 and res["config"]["checkpoint"] == str(ckpt)
    assert json.loads(capsys.readouterr().out) == {"k=3": jepa["success_rate"]}
