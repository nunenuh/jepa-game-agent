"""Linear probes: how much of the game state (position, facing, inventory) can be read off the latent."""

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_game_agent.data.episodes import random_policy
from jepa_game_agent.envs.fake import GridWorld
from jepa_game_agent.evaluation.probes import collect_states, ridge_r2, run_probes
from test_jepa_agent import exact_model  # a model whose latent is the agent's position


def make_env() -> GridWorld:
    return GridWorld(size=4, cell=4, max_steps=40, seed=0)


def test_ridge_r2_is_high_for_a_linear_target_and_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 8))
    y = np.stack([z @ rng.normal(size=8), rng.normal(size=400), np.ones(400)], 1)
    r2 = ridge_r2(z[:300], y[:300], z[300:], y[300:])
    assert r2[0] > 0.99 and r2[1] < 0.1 and np.isnan(r2[2])  # a constant target has no R²


def test_collect_states_records_every_frame_with_its_state_key() -> None:
    frames, keys, episode = collect_states(make_env(), random_policy(5), seeds=[0, 1], max_steps=10)
    assert frames.shape == (22, 16, 16, 3) and keys.shape == (22, 2) and episode.tolist() == [0] * 11 + [1] * 11
    env = make_env()
    env.reset(seed=1)
    assert tuple(keys[11]) == env.state_key()  # first frame of the second episode


def test_a_latent_that_is_the_position_probes_perfectly() -> None:
    res = run_probes(exact_model(), make_env(), random_policy(5), seeds=list(range(10)), max_steps=40)
    assert res["r2"][0] > 0.99 and res["r2"][1] > 0.99
    assert res["train_frames"] + res["test_frames"] == 10 * 41 and res["test_episodes"] == 2


def test_cli_probe_writes_r2_per_state_field(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, ckpt, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "probe.json"
    main(["collect", "--episodes", "4", "--max-steps", "20", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "2",
          "--batch-size", "8", "--checkpoint", str(ckpt)])  # fmt: skip
    capsys.readouterr()
    main(["probe", "--checkpoint", str(ckpt), "--episodes", "10", "--max-steps", "20", "--out", str(out)])
    res = json.loads(out.read_text())
    assert len(res["r2"]) == 2 and res["config"]["episodes"] == 10
    assert json.loads(capsys.readouterr().out) == res["r2"]
