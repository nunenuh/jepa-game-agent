"""Craftax (JAX) behind the GameEnv interface. Needs the optional extra: `pip install -e ".[craftax]"`."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from jepa_game_agent.envs.base import Frame, StepResult


class CraftaxEnv:
    """One Craftax environment, stepped one action at a time (for collecting data and playing).

    Pixel names only: `Craftax-Classic-Pixels-v1` (63x63x3, 17 actions) or `Craftax-Pixels-v1`
    (130x110x3, 43 actions). Frames come back as uint8 to keep datasets small.

    No auto-reset: Craftax's auto-reset swaps in a new world on the step the player dies, so the
    episode's last frame would show a different world. After `done`, call `reset`. The key schedule
    (split once per reset and per step) is shared with `data.craftax_batch`, so episodes collected
    there replay here exactly.
    """

    def __init__(self, name: str, seed: int = 0) -> None:
        if "Pixels" not in name:
            raise ValueError("only pixel observations are supported, e.g. Craftax-Classic-Pixels-v1")
        import jax
        from craftax.craftax_env import make_craftax_env_from_name

        self._jax = jax
        self._env = make_craftax_env_from_name(name, auto_reset=False)
        self._params = self._env.default_params
        self.num_actions = int(self._env.action_space(self._params).n)
        self._key = jax.random.PRNGKey(seed)
        self._step = jax.jit(self._env.step)
        frame = self.reset(seed)
        self.frame_shape = (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]))

    def reset(self, seed: int | None = None) -> Frame:
        if seed is not None:
            self._key = self._jax.random.PRNGKey(seed)
        self._key, sub = self._jax.random.split(self._key)
        obs, self._state = self._env.reset(sub, self._params)
        return self._to_frame(obs)

    def step(self, action: int) -> StepResult:
        self._key, sub = self._jax.random.split(self._key)
        obs, self._state, reward, done, info = self._step(sub, self._state, action, self._params)
        flags = {k.split("/", 1)[1]: float(v) for k, v in info.items() if k.startswith("Achievements/")}
        unlocked = tuple(k for k, v in flags.items() if v > 0)
        return StepResult(self._to_frame(obs), float(reward), bool(done), flags, unlocked)

    def state_key(self) -> tuple[int, ...]:
        """Player position, facing direction and the 12 inventory counts."""
        st = self._state
        inv = [int(getattr(st.inventory, f.name)) for f in dataclasses.fields(st.inventory)]
        return (*(int(x) for x in np.asarray(st.player_position)), int(st.player_direction), *inv)

    @property
    def state(self) -> Any:
        """The full Craftax state (map, player, inventory, ...), e.g. to stage impossible events."""
        return self._state

    def set_state(self, state: Any) -> Frame:
        """Replace the state (for example `env.state.replace(player_position=...)`) and return its frame."""
        self._state = state
        return self._to_frame(self._env.get_obs(state))

    @staticmethod
    def _to_frame(obs: Any) -> Frame:
        frame: Frame = (np.asarray(obs) * 255).round().astype(np.uint8)
        return frame
