"""The JEPA planning agent: encode the goal, plan with CEM in latent space, play the first action."""

import numpy as np
import pytest
import torch

from jepa_game_agent.config import ModelConfig, PlannerConfig
from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.envs.fake import GridWorld
from jepa_game_agent.evaluation import JepaAgent, RandomAgent, evaluate_reach, make_reach_tasks
from jepa_game_agent.models import WorldModel

CELL = 4
STEP = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])  # GridWorld's MOVES


class Position(torch.nn.Module):
    """Reads the yellow agent's (row, col) from the pixels: a perfect encoder for GridWorld."""

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lead, (c, h, w) = frames.shape[:-3], frames.shape[-3:]
        f = frames.reshape(-1, c, h, w)
        mask = ((f[:, 0] > 0.9) & (f[:, 1] > 0.7) & (f[:, 2] < 0.1)).float()
        rows = torch.arange(h, dtype=torch.float32).view(1, h, 1) // CELL
        cols = torch.arange(w, dtype=torch.float32).view(1, 1, w) // CELL
        n = mask.sum((1, 2)).clamp(min=1)
        z = torch.stack([(mask * rows).sum((1, 2)) / n, (mask * cols).sum((1, 2)) / n], -1)
        return z.reshape(*lead, 2)


class Moves(torch.nn.Module):
    """Each action moves the position exactly as GridWorld does, walls included (4 x 4 grid)."""

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (latents + STEP[actions]).clamp(0, 3)


def exact_model() -> WorldModel:
    wm = WorldModel(ModelConfig(num_actions=5, latent_dim=2, context=8, predictor_layers=1, predictor_heads=1))
    wm.encoder, wm.predictor = Position(), Moves()  # type: ignore[assignment]
    return wm.eval()


def make_env() -> GridWorld:
    return GridWorld(size=4, cell=CELL, max_steps=200, seed=0)


def test_with_a_perfect_model_the_agent_solves_reach_tasks(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=20, seed=3, env=make_env())
    cfg = PlannerConfig(horizon=6, samples=128, elites=16, iterations=4)
    jepa = evaluate_reach(make_env, tasks, lambda t: JepaAgent(exact_model(), cfg))
    rand = evaluate_reach(make_env, tasks, lambda t: RandomAgent(5, seed=0))
    assert jepa["success_rate"] == 1.0 and jepa["success_rate"] > rand["success_rate"]


def test_goal_is_the_mean_latent_of_the_goal_frames() -> None:
    env = make_env()
    env.reset(seed=0)
    a, b = env.set_agent(0, 0), env.set_agent(2, 2)
    agent = JepaAgent(exact_model(), PlannerConfig(horizon=2, samples=8, elites=2, iterations=1))
    agent.reset([a, b])
    assert torch.allclose(agent.goal, torch.tensor([1.0, 1.0]))


def test_act_needs_a_goal_and_returns_a_valid_action() -> None:
    env = make_env()
    frame = env.reset(seed=0)
    agent = JepaAgent(exact_model(), PlannerConfig(horizon=2, samples=8, elites=2, iterations=1))
    with pytest.raises(RuntimeError):
        agent.act(frame)
    agent.reset([frame])
    action = agent.act(frame)
    assert isinstance(action, int) and 0 <= action < 5
    with pytest.raises(ValueError):
        agent.reset([])
    assert np.array_equal(frame, env.render())  # the agent never changes the frame it sees


def test_hud_goal_keeps_the_current_view_and_takes_only_the_goal_frames_bottom_rows() -> None:
    env = make_env()
    env.reset(seed=0)
    goal = env.set_agent(3, 2)  # agent in the bottom grid row: the 4 bottom pixel rows play the "HUD"
    agent = JepaAgent(exact_model(), PlannerConfig(horizon=2, samples=8, elites=2, iterations=1), hud_rows=CELL)
    agent.reset([goal])
    agent.act(env.set_agent(0, 0))
    # composite = current map (agent at 0,0) + goal HUD (agent at 3,2): the encoder sees both, averages them
    assert torch.allclose(agent.goal, torch.tensor([1.5, 1.0]))
    agent.act(env.set_agent(1, 1))  # the goal follows the current view every step
    assert torch.allclose(agent.goal, torch.tensor([2.0, 1.5]))


def test_hud_rows_must_fit_the_frame() -> None:
    env = make_env()
    frame = env.reset(seed=0)
    agent = JepaAgent(exact_model(), PlannerConfig(horizon=2, samples=8, elites=2, iterations=1), hud_rows=99)
    agent.reset([frame])
    with pytest.raises(ValueError):
        agent.act(frame)


def test_weights_reach_the_planner() -> None:
    env = make_env()
    env.reset(seed=0)
    cfg = PlannerConfig(horizon=6, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, weights=torch.tensor([0.0, 1.0]))
    agent.reset([env.set_agent(3, 3)])
    assert agent.act(env.set_agent(0, 0)) == 2  # only the column counts: move right first


def test_prior_seeds_the_planner() -> None:
    """With one sample and one iteration, the planner plays whatever the prior proposes."""
    from jepa_game_agent.models.prior import ActionPrior

    class Right(ActionPrior):
        def probs(self, z: torch.Tensor, mix: float = 0.0) -> torch.Tensor:
            p = torch.full((*z.shape[:-1], 5), 1e-4)
            p[..., 2] = 1.0
            return p / p.sum(-1, keepdim=True)

    env = make_env()
    env.reset(seed=0)
    cfg = PlannerConfig(horizon=3, samples=1, elites=1, iterations=1)
    agent = JepaAgent(exact_model(), cfg, prior=Right(2, 5))
    agent.reset([env.set_agent(3, 0)])  # goal is straight down; the prior says right
    assert agent.act(env.set_agent(0, 0)) == 2


def test_change_goal_asks_for_the_same_change_from_wherever_the_agent_is() -> None:
    """Goal pairs (before, after) show one step right; from (2, 1) the goal is (2, 2): move right, not to (0, 1)."""
    env = make_env()
    env.reset(seed=0)
    pair = np.stack([env.set_agent(0, 0), env.set_agent(0, 1)])
    cfg = PlannerConfig(horizon=4, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, goal_as="change")
    agent.reset([pair])
    assert agent.act(env.set_agent(2, 1)) == 2
    assert agent.goal is not None and agent.goal.tolist() == [2.0, 2.0]
    assert JepaAgent(exact_model(), cfg).act is not None
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), cfg, goal_as="smell")
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), cfg, goal_as="change", hud_rows=2)
    with pytest.raises(ValueError):
        agent.reset([env.set_agent(0, 0)])  # change goals need (2, H, W, C) pairs


def test_both_goals_travel_by_picture_and_switch_to_the_change_when_it_is_reachable() -> None:
    """Pair (0,0) -> (0,1): the picture is (0, 1), the change is "one step right"."""
    env = make_env()
    env.reset(seed=0)
    pair = np.stack([env.set_agent(0, 0), env.set_agent(0, 1)])
    cfg = PlannerConfig(horizon=4, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, goal_as="both", reach=0.5)
    agent.reset([pair])
    assert agent.goal is not None and agent.goal.tolist() == [0.0, 1.0]  # the picture goal
    assert agent.act(env.set_agent(2, 1)) == 2  # one step right is possible here: do the change
    assert agent.act(env.set_agent(3, 3)) in (1, 3)  # wall on the right: no change possible, travel to (0, 1)


def test_blend_goal_is_the_mean_of_picture_and_change() -> None:
    env = make_env()
    env.reset(seed=0)
    pair = np.stack([env.set_agent(0, 0), env.set_agent(0, 1)])
    agent = JepaAgent(exact_model(), PlannerConfig(horizon=2, samples=8, elites=2, iterations=1), goal_as="blend")
    agent.reset([pair])
    agent.act(env.set_agent(2, 1))
    assert agent.goal is not None and agent.goal.tolist() == [1.0, 1.5]  # ((0, 1) + (2, 2)) / 2
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), PlannerConfig(), goal_as="both", reach=0.0)


def test_both_can_travel_to_the_before_picture() -> None:
    """travel='before': walk to where the change starts (e.g. facing a tree), not to how it looks afterwards."""
    env = make_env()
    env.reset(seed=0)
    pair = np.stack([env.set_agent(0, 0), env.set_agent(0, 1)])
    cfg = PlannerConfig(horizon=4, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, goal_as="both", travel="before")
    agent.reset([pair])
    assert agent.goal is not None and agent.goal.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), cfg, goal_as="both", travel="sideways")


def test_survival_cost_avoids_futures_that_lose_health() -> None:
    """Read-out health falls as the row grows (going down hurts): the agent heads right to (0, 3) first."""
    from jepa_game_agent.models.readout import Readout

    class Hurts(Readout):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            health = 9 - 3 * z[..., 0]
            return torch.stack([health, health * 0 + 9, health * 0 + 9, health * 0 + 9], -1)

    env = make_env()
    env.reset(seed=0)
    cfg = PlannerConfig(horizon=4, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, readout=Hurts(2), survival=5.0)
    agent.reset([env.set_agent(3, 3)])
    assert agent.act(env.set_agent(0, 0)) == 2
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), cfg, survival=1.0)  # survival needs a readout


def test_survival_cost_can_read_health_off_a_second_world_model() -> None:
    """critic: another world model imagines the same action sequences; the readout reads its latents.
    The critic's latent is (col, row), so the same Hurts readout now says going right hurts: head down first."""
    from jepa_game_agent.models.readout import Readout

    class Hurts(Readout):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            health = 9 - 3 * z[..., 0]
            return torch.stack([health, health * 0 + 9, health * 0 + 9, health * 0 + 9], -1)

    class Flipped(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            if len(args) == 1:
                return self.inner(args[0]).flip(-1)
            return self.inner(args[0].flip(-1), args[1]).flip(-1)

    critic = exact_model()
    critic.encoder, critic.predictor = Flipped(Position()), Flipped(Moves())  # type: ignore[assignment]
    env = make_env()
    env.reset(seed=0)
    cfg = PlannerConfig(horizon=4, samples=128, elites=16, iterations=4)
    agent = JepaAgent(exact_model(), cfg, readout=Hurts(2), survival=5.0, critic=critic)
    agent.reset([env.set_agent(3, 3)])
    assert agent.act(env.set_agent(0, 0)) == 4
    with pytest.raises(ValueError):
        JepaAgent(exact_model(), cfg, critic=critic)  # a critic is only read through a readout
