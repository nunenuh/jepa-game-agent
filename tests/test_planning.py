import pytest
import torch

from jepa_game_agent.config import ModelConfig, PlannerConfig
from jepa_game_agent.models import WorldModel
from jepa_game_agent.planning import categorical_cem, plan_to_goal


def test_cem_finds_the_known_best_sequence() -> None:
    target = torch.tensor([3, 1, 4, 1, 0, 2])
    cfg = PlannerConfig(horizon=6, samples=200, elites=20, iterations=8, seed=0)
    seq, cost = categorical_cem(lambda s: (s != target).float().sum(-1), num_actions=5, cfg=cfg)
    assert torch.equal(seq, target) and cost == 0.0


def test_cem_rejects_bad_cost_shape() -> None:
    cfg = PlannerConfig(horizon=2, samples=8, elites=2, iterations=1)
    try:
        categorical_cem(lambda s: torch.zeros(3), num_actions=3, cfg=cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_plan_to_goal_returns_valid_actions(small_cfg: ModelConfig) -> None:
    wm = WorldModel(small_cfg).eval()
    cfg = PlannerConfig(horizon=4, samples=32, elites=4, iterations=2)
    seq = plan_to_goal(wm, torch.randn(32), torch.randn(32), cfg)
    assert seq.shape == (4,) and int(seq.min()) >= 0 and int(seq.max()) < 5


def test_plan_beats_random_with_an_exact_model() -> None:
    """With a hand-made model whose latent is a position that each action moves, CEM reaches the goal."""
    cfg_m = ModelConfig(num_actions=3, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1)
    wm = WorldModel(cfg_m).eval()
    step = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    class Exact(torch.nn.Module):
        def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return latents + step[actions]

    wm.predictor = Exact()  # type: ignore[assignment]
    seq = plan_to_goal(wm, torch.zeros(2), torch.tensor([3.0, 2.0]), PlannerConfig(horizon=5, seed=0))
    final = step[seq].sum(0)
    assert torch.allclose(final, torch.tensor([3.0, 2.0]))


def test_plan_heads_for_the_goal_from_the_first_step() -> None:
    """Agents play only the first planned action, so plans must reach the goal early, not just end there."""
    cfg_m = ModelConfig(num_actions=5, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1)
    wm = WorldModel(cfg_m).eval()
    step = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])

    class Exact(torch.nn.Module):
        def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return latents + step[actions]

    wm.predictor = Exact()  # type: ignore[assignment]
    cfg = PlannerConfig(horizon=6, samples=128, elites=16, iterations=4, seed=0)
    assert plan_to_goal(wm, torch.tensor([2.0, 2.0]), torch.tensor([2.0, 1.0]), cfg)[0].item() == 1  # LEFT
    assert plan_to_goal(wm, torch.tensor([2.0, 1.0]), torch.tensor([2.0, 1.0]), cfg)[0].item() == 0  # stay


def test_weights_make_the_planner_ignore_unweighted_latent_parts() -> None:
    """weights (D,) scale each latent dimension's cost: weight 0 means 'don't care where this ends up'."""
    cfg_m = ModelConfig(num_actions=5, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1)
    wm = WorldModel(cfg_m).eval()
    step = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])

    class Exact(torch.nn.Module):
        def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return latents + step[actions]

    wm.predictor = Exact()  # type: ignore[assignment]
    cfg = PlannerConfig(horizon=6, samples=128, elites=16, iterations=4, seed=0)
    now, goal = torch.tensor([0.0, 0.0]), torch.tensor([3.0, 3.0])
    seq = plan_to_goal(wm, now, goal, cfg, weights=torch.tensor([0.0, 1.0]))
    assert torch.equal(seq[:3], torch.tensor([2, 2, 2]))  # only the second coordinate counts: go there first
    seq = plan_to_goal(wm, now, goal, cfg, weights=torch.tensor([1.0, 0.0]))
    assert torch.equal(seq[:3], torch.tensor([4, 4, 4]))


def test_cem_init_lets_a_rare_sequence_be_found() -> None:
    """Only [4, 4, 4] has low cost; 4 uniform samples almost never hit it, a prior on action 4 does."""
    target = torch.tensor([4, 4, 4])
    cfg = PlannerConfig(horizon=3, samples=4, elites=1, iterations=1, seed=0)
    cost = lambda s: (s != target).float().sum(-1)  # noqa: E731
    _, uniform_cost = categorical_cem(cost, num_actions=5, cfg=cfg)
    init = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.96])
    seq, prior_cost = categorical_cem(cost, num_actions=5, cfg=cfg, init=init)
    assert uniform_cost > 0 and prior_cost == 0.0 and torch.equal(seq, target)


def test_cem_rejects_bad_init() -> None:
    cfg = PlannerConfig(horizon=2, samples=8, elites=2, iterations=1)
    with pytest.raises(ValueError):
        categorical_cem(lambda s: torch.zeros(8), num_actions=3, cfg=cfg, init=torch.ones(4) / 4)


def test_extra_cost_steers_the_plan_away_from_danger() -> None:
    """Goal (3, 3) from (0, 0); an extra cost on going down (dim 0) makes the plan go right first."""
    cfg_m = ModelConfig(num_actions=5, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1)
    wm = WorldModel(cfg_m).eval()
    step = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])

    class Exact(torch.nn.Module):
        def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return latents + step[actions]

    wm.predictor = Exact()  # type: ignore[assignment]
    cfg = PlannerConfig(horizon=6, samples=128, elites=16, iterations=4, seed=0)
    danger = lambda path: 10.0 * path[..., 0].clamp(min=0).sum(-1)  # noqa: E731
    seq = plan_to_goal(wm, torch.tensor([0.0, 0.0]), torch.tensor([3.0, 3.0]), cfg, extra_cost=danger)
    assert torch.equal(seq[:3], torch.tensor([2, 2, 2]))
