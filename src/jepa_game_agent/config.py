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

    def __post_init__(self) -> None:
        for name in ("num_actions", "latent_dim", "channels", "context", "predictor_layers", "predictor_heads"):
            _positive(name, getattr(self, name))
        if self.latent_dim % self.predictor_heads:
            raise ValueError("latent_dim must be divisible by predictor_heads")
        if self.sigreg_weight < 0:
            raise ValueError("sigreg_weight must be >= 0")


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

    def __post_init__(self) -> None:
        for name in ("horizon", "samples", "elites", "iterations"):
            _positive(name, getattr(self, name))
        if self.elites > self.samples:
            raise ValueError("elites must be <= samples")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
