"""v4: a spatial latent, one token per game tile (docs/DESIGN_V4.md)."""

import pytest
import torch

from jepa_game_agent.config import ModelConfig
from jepa_game_agent.models import Encoder, Predictor, WorldModel
from jepa_game_agent.models.spatial import SpatialEncoder, SpatialPredictor


def spatial_cfg(**kw: object) -> ModelConfig:
    base: dict[str, object] = dict(num_actions=5, grid=4, token_dim=8, latent_dim=4 * 4 * 8, context=4,
                                   predictor_layers=1, predictor_heads=2, width=16)  # fmt: skip
    return ModelConfig(**(base | kw))  # type: ignore[arg-type]


def test_spatial_encoder_gives_one_token_per_tile_flattened() -> None:
    enc = SpatialEncoder(channels=3, grid=9, token_dim=8)
    assert enc(torch.rand(2, 5, 3, 63, 63)).shape == (2, 5, 9 * 9 * 8)


def test_a_token_only_sees_its_own_tile_and_neighbours() -> None:
    """Changing the bottom-right tile leaves the top-left tile's token unchanged."""
    enc = SpatialEncoder(channels=3, grid=9, token_dim=8).eval()
    x = torch.rand(1, 3, 63, 63)
    y = x.clone()
    y[..., 56:, 56:] = torch.rand(1, 3, 7, 7)
    a, b = enc(x).view(9, 9, 8), enc(y).view(9, 9, 8)
    assert torch.allclose(a[0, 0], b[0, 0], atol=1e-6) and not torch.allclose(a[8, 8], b[8, 8])


def test_spatial_predictor_never_looks_ahead_in_time_but_sees_the_whole_frame() -> None:
    pred = SpatialPredictor(num_actions=5, tokens=16, token_dim=8, context=4, layers=2, heads=2, width=16).eval()
    z, a = torch.randn(1, 4, 16 * 8), torch.randint(5, (1, 4))
    out = pred(z, a)
    assert out.shape == (1, 4, 16 * 8)
    later = z.clone()
    later[:, 3] = torch.randn(16 * 8)
    out2 = pred(later, a)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)  # the future can't change the past
    one_tile = z.clone()
    one_tile[:, 1, :8] += 1.0  # change only tile 0 at time 1
    out3 = pred(one_tile, a).view(1, 4, 16, 8)
    assert not torch.allclose(out3[:, 1, 15], out.view(1, 4, 16, 8)[:, 1, 15])  # tile 15 noticed it
    with pytest.raises(ValueError):
        pred(torch.randn(1, 5, 16 * 8), torch.zeros(1, 5, dtype=torch.long))


def test_spatial_world_model_trains_and_imagines() -> None:
    wm = WorldModel(spatial_cfg(inverse_weight=0.5))
    assert isinstance(wm.encoder, SpatialEncoder) and isinstance(wm.predictor, SpatialPredictor)
    frames, actions = torch.rand(2, 4, 3, 16, 16), torch.randint(5, (2, 3))
    losses = wm.loss(frames, actions)
    losses.total.backward()
    assert losses.total > 0 and losses.inverse > 0
    assert all(p.grad is not None for p in wm.parameters() if p.requires_grad)
    wm.eval()
    z = wm.encode(frames[:, 0])
    assert z.shape == (2, 128) and wm.rollout(z, torch.randint(5, (2, 6))).shape == (2, 6, 128)
    assert wm.surprise(frames, actions).shape == (2, 3)


def test_spatial_config_is_validated_and_off_by_default(small_cfg: ModelConfig) -> None:
    wm = WorldModel(small_cfg)
    assert small_cfg.grid == 0 and isinstance(wm.encoder, Encoder) and isinstance(wm.predictor, Predictor)
    with pytest.raises(ValueError):
        spatial_cfg(latent_dim=100)  # must be grid * grid * token_dim
    with pytest.raises(ValueError):
        spatial_cfg(hud_rows=14)  # the tiles already cover the status bar


def test_inverse_head_stays_small_for_a_big_latent() -> None:
    wm = WorldModel(spatial_cfg(grid=9, latent_dim=9 * 9 * 32, token_dim=32, inverse_weight=0.5))
    assert wm.inverse is not None and sum(p.numel() for p in wm.inverse.parameters()) < 2_000_000
