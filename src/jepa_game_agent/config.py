"""Configuration dataclasses. Every field is validated on creation, so a bad config fails at once."""

from __future__ import annotations

from dataclasses import dataclass


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class ModelConfig:
    """World model: CNN encoder + action-conditioned causal transformer predictor."""

    num_actions: int
    latent_dim: int = 128
    channels: int = 3
    context: int = 16  # longest history the predictor sees
    predictor_layers: int = 4
    predictor_heads: int = 4
    sigreg_weight: float = 0.09  # LeWM's Push-T setting
    inverse_weight: float = 0.0  # > 0 adds an inverse-dynamics head: guess a_t from (z_t, z_t+1)
    hud_rows: int = 0  # > 0: the bottom hud_rows pixel rows (Craftax-Classic's inventory: 14) get their own encoder
    hud_dim: int = 32  # ...whose output is the last hud_dim latent dims
    grid: int = 0  # > 0: v4 spatial latent, grid x grid tokens (one per tile; Craftax-Classic: 9)
    token_dim: int = 32  # ...of token_dim numbers each; latent_dim must be grid * grid * token_dim
    width: int = 128  # spatial predictor's transformer width

    def __post_init__(self) -> None:
        for name in ("num_actions", "latent_dim", "channels", "context", "predictor_layers", "predictor_heads"):
            _positive(name, getattr(self, name))
        if self.grid < 0:
            raise ValueError("grid must be >= 0")
        if self.grid:
            _positive("token_dim", self.token_dim)
            _positive("width", self.width)
            if self.latent_dim != self.grid * self.grid * self.token_dim:
                raise ValueError("latent_dim must be grid * grid * token_dim")
            if self.hud_rows:
                raise ValueError("grid and hud_rows can't be combined: the tiles already cover the status bar")
            if self.width % self.predictor_heads:
                raise ValueError("width must be divisible by predictor_heads")
        elif self.latent_dim % self.predictor_heads:
            raise ValueError("latent_dim must be divisible by predictor_heads")
        if self.sigreg_weight < 0:
            raise ValueError("sigreg_weight must be >= 0")
        if self.inverse_weight < 0:
            raise ValueError("inverse_weight must be >= 0")
        if self.hud_rows < 0:
            raise ValueError("hud_rows must be >= 0")
        if self.hud_rows and not 0 < self.hud_dim < self.latent_dim:
            raise ValueError("hud_dim must be between 0 and latent_dim")


@dataclass(frozen=True)
class TopConfig:
    """Top level of the hierarchy: a causal predictor where one step = `stride` game steps (an action chunk)."""

    num_actions: int
    latent_dim: int = 128  # must match the bottom encoder
    stride: int = 4
    context: int = 8
    layers: int = 4
    heads: int = 4
    width: int = 0  # > 0: transformer width, mapped to and from the latent (v4's 2592 numbers); 0: = latent_dim

    def __post_init__(self) -> None:
        for name in ("num_actions", "latent_dim", "stride", "context", "layers", "heads"):
            _positive(name, getattr(self, name))
        if self.width < 0:
            raise ValueError("width must be >= 0")
        if (self.width or self.latent_dim) % self.heads:
            raise ValueError("the transformer width (width, or latent_dim) must be divisible by heads")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10_000
    batch_size: int = 64
    window: int = 8  # transitions per training sample
    lr: float = 3e-4
    weight_decay: float = 0.05
    checkpoint_every: int = 500
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "window", "lr", "checkpoint_every"):
            _positive(name, getattr(self, name))


@dataclass(frozen=True)
class PlannerConfig:
    """Cross-entropy method over discrete action sequences."""

    horizon: int = 8
    samples: int = 256
    elites: int = 32
    iterations: int = 6
    smoothing: float = 0.7  # weight of the elite frequencies in each update
    seed: int = 0
    half: bool = False  # imagine futures in bfloat16 (faster on a GPU); costs still compared in float32

    def __post_init__(self) -> None:
        for name in ("horizon", "samples", "elites", "iterations"):
            _positive(name, getattr(self, name))
        if self.elites > self.samples:
            raise ValueError("elites must be <= samples")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
