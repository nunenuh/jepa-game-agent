import pytest
import torch

from jepa_game_agent.config import ModelConfig
from jepa_game_agent.models import Encoder, Predictor, WorldModel, sigreg


def test_encoder_any_frame_size() -> None:
    enc = Encoder(channels=3, latent_dim=32)
    for h, w in [(16, 16), (63, 63), (130, 110)]:
        assert enc(torch.rand(2, 5, 3, h, w)).shape == (2, 5, 32)


def test_predictor_is_causal() -> None:
    pred = Predictor(num_actions=5, latent_dim=32, context=8, layers=2, heads=2).eval()
    z, a = torch.randn(1, 6, 32), torch.randint(5, (1, 6))
    out = pred(z, a)
    z2, a2 = z.clone(), a.clone()
    z2[:, 4:], a2[:, 4:] = torch.randn(1, 2, 32), (a[:, 4:] + 1) % 5
    out2 = pred(z2, a2)
    assert torch.allclose(out[:, :4], out2[:, :4], atol=1e-5)  # the future can't change the past
    assert not torch.allclose(out[:, 4:], out2[:, 4:])


def test_predictor_checks_input() -> None:
    pred = Predictor(num_actions=5, latent_dim=32, context=4, layers=1, heads=2)
    with pytest.raises(ValueError):
        pred(torch.randn(1, 5, 32), torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        pred(torch.randn(1, 3, 32), torch.zeros(1, 2, dtype=torch.long))


def test_sigreg_tells_gaussian_from_collapsed() -> None:
    gen = torch.Generator().manual_seed(0)
    gauss = sigreg(torch.randn(512, 16, generator=gen), generator=torch.Generator().manual_seed(1))
    collapsed = sigreg(torch.ones(512, 16) * 3, generator=torch.Generator().manual_seed(1))
    assert gauss < 0.1 * collapsed
    with pytest.raises(ValueError):
        sigreg(torch.randn(1, 16))


def test_world_model_loss_and_shapes(small_cfg: ModelConfig) -> None:
    wm = WorldModel(small_cfg)
    frames, actions = torch.rand(2, 5, 3, 16, 16), torch.randint(5, (2, 4))
    losses = wm.loss(frames, actions)
    assert losses.total.ndim == 0 and losses.total > 0
    losses.total.backward()
    assert all(p.grad is not None for p in wm.parameters() if p.requires_grad)
    assert wm.surprise(frames, actions).shape == (2, 4)


def test_rollout_shapes_and_history(small_cfg: ModelConfig) -> None:
    wm = WorldModel(small_cfg).eval()
    start, acts = torch.randn(3, 32), torch.randint(5, (3, 12))  # longer than the context: windowed
    assert wm.rollout(start, acts).shape == (3, 12, 32)
    hist, hist_a = torch.randn(3, 4, 32), torch.randint(5, (3, 4))
    assert wm.rollout(start, acts[:, :2], hist, hist_a).shape == (3, 2, 32)
    with pytest.raises(ValueError):
        wm.rollout(start, acts, history=hist)


def test_inverse_dynamics_is_off_by_default_so_old_checkpoints_still_load(small_cfg: ModelConfig) -> None:
    wm = WorldModel(small_cfg)
    assert wm.inverse is None and not any(k.startswith("inverse") for k in wm.state_dict())
    losses = wm.loss(torch.rand(2, 5, 3, 16, 16), torch.randint(5, (2, 4)))
    assert losses.inverse.item() == 0.0


def test_inverse_dynamics_adds_an_action_classification_loss(small_cfg: ModelConfig) -> None:
    from dataclasses import replace

    cfg = replace(small_cfg, inverse_weight=0.5)
    wm = WorldModel(cfg)
    frames, actions = torch.rand(2, 5, 3, 16, 16), torch.randint(5, (2, 4))
    losses = wm.loss(frames, actions)
    expected = losses.prediction + cfg.sigreg_weight * losses.sigreg + 0.5 * losses.inverse
    assert losses.inverse > 0 and torch.allclose(losses.total, expected)
    losses.total.backward()
    assert wm.inverse is not None and all(p.grad is not None for p in wm.inverse.parameters())
    with pytest.raises(ValueError):
        replace(small_cfg, inverse_weight=-1.0)


def test_inverse_dynamics_learns_which_move_was_made(episodes, small_cfg: ModelConfig) -> None:  # type: ignore[no-untyped-def]
    """GridWorld moves are visible, so the action between two frames can be learned (it's ~chance at init)."""
    from dataclasses import replace

    from jepa_game_agent.config import TrainConfig
    from jepa_game_agent.data.episodes import WindowDataset
    from jepa_game_agent.training import train

    wm = WorldModel(replace(small_cfg, inverse_weight=1.0))
    hist = train(wm, WindowDataset(episodes, 4), TrainConfig(steps=200, batch_size=16, lr=1e-3))
    first = sum(h["inverse"] for h in hist[:5]) / 5
    last = sum(h["inverse"] for h in hist[-5:]) / 5
    assert last < 0.5 * first
