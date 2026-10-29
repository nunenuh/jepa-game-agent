"""Does the world model know what an action does? Rank the true action among all actions' predictions."""

import torch

from jepa_game_agent.config import ModelConfig
from jepa_game_agent.evaluation.action_effects import rank_actions
from jepa_game_agent.models import WorldModel

STEP = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])


class Exact(torch.nn.Module):
    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return latents + STEP[actions]


def model() -> WorldModel:
    wm = WorldModel(ModelConfig(num_actions=5, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1))
    wm.predictor = Exact()  # type: ignore[assignment]
    return wm.eval()


def test_true_action_ranks_first_when_the_model_is_right() -> None:
    z_now = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    z_next = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # actions 4 and 2 really happened
    res = rank_actions(model(), z_now, z_next, torch.tensor([4, 2]), goal=torch.tensor([3.0, 0.0]))
    assert res["next_rank"] == [1, 1] and res["next_top1"] == 1.0
    assert res["goal_rank"][0] == 1 and res["goal_rank"][1] > 1  # only action 4 heads for the goal at (3, 0)


def test_wrong_model_ranks_the_true_action_low() -> None:
    z_now = torch.tensor([[0.0, 0.0]])
    z_next = torch.tensor([[0.0, 5.0]])  # far up: action 2 gets closest, action 1 (down) is the worst guess
    res = rank_actions(model(), z_now, z_next, torch.tensor([1]))
    assert res["next_rank"] == [5] and res["goal_rank"] is None
    assert res["mean_next_rank"] == 5.0


def test_cli_action_effects(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    import json

    from jepa_game_agent.cli import main

    data, wm, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "fx.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    capsys.readouterr()
    main(["action-effects", "--checkpoint", str(wm), "--data", str(data), "--goal-data", str(data),
          "--achievement", "collect_item", "--control", "20", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    item = res["achievement"]["collect_item"]
    assert item["n"] >= 1 and 1 <= item["mean_next_rank"] <= 5 and item["goal_rank"] is not None
    assert res["control"]["n"] == 20 and res["control"]["goal_rank"] is None
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"collect_item", "control"}


def test_goal_can_differ_per_transition() -> None:
    z_now = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    z_next = torch.tensor([[1.0, 0.0], [6.0, 5.0]])
    goals = z_now + torch.tensor([1.0, 0.0])  # "one step down" from wherever each one is
    res = rank_actions(model(), z_now, z_next, torch.tensor([4, 4]), goal=goals)
    assert res["goal_rank"] == [1, 1]


def test_cli_action_effects_with_change_goals(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    import json

    from jepa_game_agent.cli import main

    data, wm, out = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "fx.json"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    main(["action-effects", "--goal-as", "change", "--checkpoint", str(wm), "--data", str(data), "--goal-data",
          str(data), "--achievement", "collect_item", "--control", "5", "--out", str(out)])  # fmt: skip
    res = json.loads(out.read_text())
    assert res["config"]["goal_as"] == "change" and res["achievement"]["collect_item"]["goal_rank"] is not None
