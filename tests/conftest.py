import pytest
import torch

from jepa_game_agent.config import ModelConfig
from jepa_game_agent.data.episodes import collect_episode, random_policy
from jepa_game_agent.envs.fake import GridWorld


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(size=4, cell=4, max_steps=30, seed=0)


@pytest.fixture
def episodes(env: GridWorld):
    return [collect_episode(env, random_policy(env.num_actions), max_steps=30, seed=s) for s in range(4)]


@pytest.fixture
def small_cfg() -> ModelConfig:
    return ModelConfig(num_actions=5, latent_dim=32, context=8, predictor_layers=1, predictor_heads=2)
