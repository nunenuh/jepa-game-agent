"""Achievement goals as pictures: frames right after the achievement unlocked in recorded games."""

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.evaluation.goals import achievement_goal_frames, unlock_steps


def episode(rewards: list[float], achievements: tuple[str, ...]) -> Episode:
    t = len(rewards)
    frames = np.arange(t + 1, dtype=np.uint8)[:, None, None, None].repeat(2, 1).repeat(2, 2).repeat(3, 3)
    return Episode(frames, np.zeros(t, np.int64), np.asarray(rewards, np.float32), 0, achievements)


def test_unlock_steps_match_unit_rewards_to_achievements_in_order() -> None:
    ep = episode([0, 1.0, -0.1, 0, 0.9, 0.1, 1.0], ("collect_wood", "place_table", "make_wood_pickaxe"))
    assert unlock_steps(ep) == {"collect_wood": 1, "place_table": 4, "make_wood_pickaxe": 6}  # 0.9 = 1 - health


def test_unlock_steps_refuse_episodes_with_fewer_unlock_rewards_than_achievements() -> None:
    with pytest.raises(ValueError):
        unlock_steps(episode([0, 0.1], ("collect_wood",)))
    assert unlock_steps(episode([1.0, 1.0], ("collect_item",))) == {"collect_item": 0}  # repeatable: first one


def test_an_unlock_with_heavy_damage_still_counts() -> None:
    """Waking up is often caused by a zombie hit while asleep (-0.7 health reward): reward 1 - 0.7 = 0.3."""
    ep = episode([0.1, 0.3, -0.2, 1.1], ("wake_up", "collect_wood"))
    assert unlock_steps(ep) == {"wake_up": 1, "collect_wood": 3}  # +0.1 alone is only health regeneration


def test_goal_frames_skip_episodes_whose_unlocks_dont_add_up() -> None:
    eps = [episode([0, 0.1], ("collect_wood",)), episode([1.0, 0], ("collect_wood",))]
    assert [int(f[0, 0, 0]) for f in achievement_goal_frames(eps, "collect_wood")] == [1]


def test_goal_frames_are_the_frames_right_after_the_unlock() -> None:
    eps = [episode([0, 1.0, 0], ("collect_wood",)), episode([1.0, 0, 1.0], ("collect_wood", "place_table"))]
    frames = achievement_goal_frames(eps, "collect_wood", n=10, seed=0)
    assert sorted(int(f[0, 0, 0]) for f in frames) == [1, 2]  # frame t+1 after the step-t unlock
    assert len(achievement_goal_frames(eps, "collect_wood", n=1, seed=0)) == 1
    with pytest.raises(ValueError):
        achievement_goal_frames(eps, "collect_diamond", n=5)


def test_cli_plan_achievement_on_gridworld(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, wm, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "ach.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "2",
          "--batch-size", "8", "--hud-rows", "4", "--hud-dim", "4", "--checkpoint", str(wm)])  # fmt: skip
    capsys.readouterr()
    main(["plan-achievement", "--goal-on", "hud", "--checkpoint", str(wm), "--goal-data", str(data),
          "--achievement", "collect_item",
          "--resets", "3", "--budget", "10", "--goal-frames", "4", "--horizon", "2", "--samples", "8", "--elites",
          "2", "--iterations", "1", "--hud-rows", "4", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    r = res["achievement"]["collect_item"]["jepa"]
    assert r["tasks"] == 3 and 0 <= r["success_rate"] <= 1 and res["config"]["goal_frames"] == 4
    assert res["config"]["hud_rows"] == 4 and res["config"]["goal_on"] == "hud"
    assert json.loads(capsys.readouterr().out) == {"collect_item": r["success_rate"]}


def test_cli_goal_on_hud_needs_a_hud_model(tmp_path: Path) -> None:
    from jepa_game_agent.cli import main

    data, wm = tmp_path / "eps.npz", tmp_path / "wm.pt"
    main(["collect", "--episodes", "2", "--max-steps", "20", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    with pytest.raises(ValueError):
        main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
              "--goal-on", "hud", "--out", str(tmp_path / "a.json")])  # fmt: skip


def test_cli_plan_achievement_chain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, wm, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "chain.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    capsys.readouterr()
    main(["plan-achievement", "--chain", "--checkpoint", str(wm), "--goal-data", str(data),
          "--achievement", "collect_item", "--resets", "2", "--budget", "5", "--goal-frames", "2", "--horizon", "2",
          "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    r = res["chain"]["jepa"]
    assert r["tasks"] == 2 and r["chain"] == ["collect_item"] and res["config"]["chain"] is True
    assert json.loads(capsys.readouterr().out) == r["stage_rates"]


def test_cli_train_prior_then_plan_with_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, wm, prior, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "prior.pt", tmp_path / "a.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    main(["train-prior", "--data", str(data), "--bottom", str(wm), "--steps", "5", "--batch-size", "8",
          "--checkpoint", str(prior)])  # fmt: skip
    assert prior.exists()
    capsys.readouterr()
    main(["plan-achievement", "--checkpoint", str(wm), "--goal-data", str(data), "--achievement", "collect_item",
          "--prior", str(prior), "--prior-mix", "0.2", "--resets", "2", "--budget", "5", "--goal-frames", "2",
          "--horizon", "2", "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    cfg = json.loads(out.read_text())["config"]
    assert cfg["prior"] == str(prior) and cfg["prior_mix"] == 0.2


def test_goal_weights_balance_view_and_inventory() -> None:
    """mix: the view's and the inventory's dims each add up to one unit (inventory times hud_weight)."""
    import torch

    from jepa_game_agent.cli import goal_weights
    from jepa_game_agent.config import ModelConfig

    cfg = ModelConfig(num_actions=5, latent_dim=16, hud_rows=4, hud_dim=4)
    assert goal_weights(cfg, "all") is None
    hud = goal_weights(cfg, "hud")
    assert hud is not None and hud[:12].sum() == 0 and hud[12:].sum() == 4
    mix = goal_weights(cfg, "mix", hud_weight=2.0)
    assert mix is not None and torch.allclose(mix[:12].sum(), torch.tensor(1.0))
    assert torch.allclose(mix[12:].sum(), torch.tensor(2.0))
    with pytest.raises(ValueError):
        goal_weights(ModelConfig(num_actions=5, latent_dim=16), "mix")
