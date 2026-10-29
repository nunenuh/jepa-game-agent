"""Hierarchical agent: the top picks a subgoal one chunk ahead, the bottom walks to it."""

import pytest
import torch

from jepa_game_agent.config import PlannerConfig, TopConfig
from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.evaluation import HierarchicalAgent, evaluate_reach, make_reach_tasks
from jepa_game_agent.models.top import TopModel
from jepa_game_agent.planning.hierarchy import plan_chunks_to_goal
from test_jepa_agent import STEP, exact_model, make_env

STRIDE = 4


class ExactTop(TopModel):
    """Applies the chunk's moves one by one, walls included: a perfect 4-step model of the 4 x 4 GridWorld."""

    def forward(self, latents: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        z = latents
        for i in range(chunks.size(-1)):
            z = (z + STEP[chunks[..., i]]).clamp(0, 3)
        return z


def exact_top() -> TopModel:
    return ExactTop(TopConfig(num_actions=5, latent_dim=2, stride=STRIDE, context=4, layers=1, heads=1)).eval()


TOP = PlannerConfig(horizon=3, samples=128, elites=16, iterations=4)  # 3 chunks = 12 game steps
BOTTOM = PlannerConfig(horizon=4, samples=64, elites=8, iterations=3)


def test_top_plan_heads_for_the_goal_and_returns_the_first_subgoal() -> None:
    actions, subgoal = plan_chunks_to_goal(exact_top(), torch.tensor([0.0, 0.0]), torch.tensor([3.0, 3.0]), TOP)
    assert actions.shape == (TOP.horizon * STRIDE,)
    assert torch.allclose(subgoal, exact_top().rollout(torch.zeros(1, 2), actions[:STRIDE].view(1, 1, -1))[0, 0])
    assert (torch.tensor([3.0, 3.0]) - subgoal).abs().sum() <= 2  # one chunk of 4 moves covers 4 of the 6 cells


def test_with_perfect_models_the_hierarchical_agent_solves_reach_tasks(episodes: list[Episode]) -> None:
    tasks = make_reach_tasks(episodes, k=5, n=20, seed=3, env=make_env())
    res = evaluate_reach(make_env, tasks, lambda t: HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM))
    assert res["success_rate"] == 1.0


def test_the_top_replans_once_per_chunk() -> None:
    env = make_env()
    frame = env.reset(seed=0)
    agent = HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM)
    agent.reset([env.set_agent(3, 3)])
    env.set_agent(0, 0)
    frame = env.render()
    subgoals = []
    for _ in range(2 * STRIDE):
        frame = env.step(agent.act(frame)).frame
        subgoals.append(agent.subgoal.clone())  # type: ignore[union-attr]
    assert all(torch.equal(subgoals[i], subgoals[0]) for i in range(STRIDE))  # held for one chunk
    assert agent.top_plans == 2


def test_act_needs_a_goal() -> None:
    agent = HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM)
    with pytest.raises(RuntimeError):
        agent.act(make_env().reset(seed=0))
    with pytest.raises(ValueError):
        agent.reset([])


def test_a_subgoal_that_makes_no_progress_is_skipped() -> None:
    """Near the goal the top may propose a subgoal no closer than where we are; then walk to the goal itself."""

    class Lazy(ExactTop):
        def forward(self, latents: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
            return latents  # imagines that nothing ever moves: every subgoal = here

    lazy = Lazy(TopConfig(num_actions=5, latent_dim=2, stride=STRIDE, context=4, layers=1, heads=1)).eval()
    env = make_env()
    env.reset(seed=0)
    agent = HierarchicalAgent(exact_model(), lazy, TOP, BOTTOM)
    agent.reset([env.set_agent(2, 1)])
    frame = env.set_agent(3, 1)
    env.step(agent.act(frame))
    assert env.state_key() == (2, 1) and torch.equal(agent.subgoal, agent.goal)  # type: ignore[arg-type]


def test_snapped_subgoals_are_real_latents_from_the_bank() -> None:
    env = make_env()
    env.reset(seed=0)
    bank = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])  # the goal (3, 3) itself is not in it
    agent = HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM, bank=bank)
    agent.reset([env.set_agent(3, 3)])
    agent.act(env.set_agent(0, 0))
    assert agent.subgoal is not None and torch.equal(agent.subgoal, torch.tensor([2.0, 2.0]))


def test_bank_must_match_the_latent_size() -> None:
    with pytest.raises(ValueError):
        HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM, bank=torch.zeros(3, 5))


def test_hierarchical_weights_reach_both_levels() -> None:
    env = make_env()
    env.reset(seed=0)
    agent = HierarchicalAgent(exact_model(), exact_top(), TOP, BOTTOM, weights=torch.tensor([0.0, 1.0]))
    agent.reset([env.set_agent(3, 3)])
    assert agent.act(env.set_agent(0, 0)) == 2
    assert agent.subgoal is not None and agent.subgoal[1] >= 2  # the top heads for the goal column
