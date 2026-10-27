import pytest

from jepa_game_agent.config import ModelConfig, PlannerConfig, TrainConfig


def test_defaults_are_valid() -> None:
    ModelConfig(num_actions=17)
    TrainConfig()
    PlannerConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_actions": 0},
        {"num_actions": 5, "latent_dim": 30, "predictor_heads": 4},
        {"num_actions": 5, "sigreg_weight": -1.0},
    ],
)
def test_bad_model_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)  # type: ignore[arg-type]


def test_bad_train_and_planner_config() -> None:
    with pytest.raises(ValueError):
        TrainConfig(lr=0)
    with pytest.raises(ValueError):
        PlannerConfig(samples=8, elites=16)
    with pytest.raises(ValueError):
        PlannerConfig(smoothing=0)
