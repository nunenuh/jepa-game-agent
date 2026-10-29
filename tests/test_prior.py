"""Action prior: which actions players press in a state, learned from recorded games, to seed the planner."""

from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_game_agent.models.prior import ActionPrior
from jepa_game_agent.training.prior_trainer import load_prior, save_prior, train_prior


def test_probs_are_a_distribution_with_a_floor() -> None:
    prior = ActionPrior(latent_dim=4, num_actions=5)
    p = prior.probs(torch.randn(3, 4), mix=0.25)
    assert p.shape == (3, 5) and torch.allclose(p.sum(-1), torch.ones(3))
    assert (p >= 0.25 / 5 - 1e-6).all()  # every action stays possible
    with pytest.raises(ValueError):
        prior.probs(torch.randn(3, 4), mix=1.5)


def test_prior_learns_which_action_follows_a_state(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    latents = [rng.normal(size=(51, 4)).astype(np.float32) for _ in range(8)]
    actions = [(z[:-1, 0] > 0).astype(np.int64) * 3 for z in latents]  # players press 3 when z[0] > 0, else 0
    prior = ActionPrior(latent_dim=4, num_actions=5)
    history = train_prior(prior, latents, actions, steps=300, batch_size=64, seed=0)
    assert history[-1]["loss"] < history[0]["loss"]
    z = torch.tensor([[2.0, 0, 0, 0], [-2.0, 0, 0, 0]])
    assert prior.probs(z, mix=0.0).argmax(-1).tolist() == [3, 0]
    save_prior(tmp_path / "p.pt", prior, history)
    again, ckpt = load_prior(tmp_path / "p.pt")
    assert torch.allclose(again.probs(z, mix=0.0), prior.probs(z, mix=0.0)) and ckpt["history"] == history


def test_train_prior_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        train_prior(ActionPrior(4, 5), [np.zeros((5, 4), np.float32)], [np.zeros(5, np.int64)], steps=1)
