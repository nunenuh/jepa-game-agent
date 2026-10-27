"""Game worlds behind one small interface."""

from jepa_game_agent.envs.base import GameEnv, StepResult
from jepa_game_agent.envs.fake import GridWorld

__all__ = ["GameEnv", "GridWorld", "StepResult", "make_env"]


def make_env(name: str, seed: int = 0) -> GameEnv:
    """`gridworld` (built in, for tests and debugging) or a Craftax name such as `Craftax-Classic-Pixels-v1`."""
    if name == "gridworld":
        return GridWorld(seed=seed)
    if name.startswith("Craftax"):
        from jepa_game_agent.envs.craftax import CraftaxEnv

        return CraftaxEnv(name, seed=seed)
    raise ValueError(f"unknown env {name!r}")
