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
