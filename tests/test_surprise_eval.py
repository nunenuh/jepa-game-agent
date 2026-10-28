"""Surprise test: is the model more surprised by an impossible jump than by what really happened next?"""

import json
from pathlib import Path

import pytest

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.evaluation.surprise_eval import roc_auc, surprise_auc
from test_jepa_agent import exact_model  # latent = the agent's position, predictor = the exact moves


def test_roc_auc() -> None:
    assert roc_auc([0.0, 0.1], [0.5, 0.9]) == 1.0  # every impossible step scores higher
    assert roc_auc([0.5, 0.9], [0.0, 0.1]) == 0.0
    assert roc_auc([1.0, 1.0], [1.0, 1.0]) == 0.5  # ties count half
    with pytest.raises(ValueError):
        roc_auc([], [1.0])


def test_a_perfect_model_is_only_surprised_by_jumps(episodes: list[Episode]) -> None:
    res = surprise_auc(exact_model(), episodes, window=4, n=60, min_gap=8, seed=0)
    assert res["auc"] > 0.9 and res["pairs"] == 60
    assert res["mean_normal"] < 1e-6 < res["mean_jump"]  # real next frames are predicted exactly


def test_surprise_auc_needs_long_enough_episodes(episodes: list[Episode]) -> None:
    with pytest.raises(ValueError):
        surprise_auc(exact_model(), episodes, window=4, n=10, min_gap=1000)


def test_cli_surprise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, ckpt, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "s.json"
    main(["collect", "--episodes", "4", "--max-steps", "40", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "2",
          "--batch-size", "8", "--checkpoint", str(ckpt)])  # fmt: skip
    capsys.readouterr()
    main(["surprise", "--checkpoint", str(ckpt), "--data", str(data), "--window", "4", "--n", "20", "--min-gap", "10",
          "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    assert 0.0 <= res["auc"] <= 1.0 and res["config"]["min_gap"] == 10
    assert json.loads(capsys.readouterr().out)["auc"] == res["auc"]
