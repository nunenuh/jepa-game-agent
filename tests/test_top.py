"""The top level of the hierarchy: one imagined step = `stride` game steps, on frozen bottom latents."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_game_agent.config import TopConfig, TrainConfig
from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.models.top import TopModel
from jepa_game_agent.training.top_trainer import TopWindowDataset, encode_episodes, load_top, train_top
from test_jepa_agent import exact_model  # bottom whose latent is the agent's (row, col)


def top_cfg(**kw: object) -> TopConfig:
    return TopConfig(**{"num_actions": 5, "latent_dim": 2, "stride": 4, "context": 4, "layers": 1, "heads": 1} | kw)


def test_top_config_is_validated() -> None:
    with pytest.raises(ValueError):
        top_cfg(stride=0)
    with pytest.raises(ValueError):
        top_cfg(latent_dim=3, heads=2)


def test_top_model_shapes_and_order_of_actions_matters() -> None:
    top = TopModel(top_cfg(latent_dim=8, heads=2)).eval()
    z, chunks = torch.randn(3, 4, 8), torch.randint(5, (3, 4, 4))
    assert top(z, chunks).shape == (3, 4, 8)
    assert top.rollout(torch.randn(3, 8), torch.randint(5, (3, 6, 4))).shape == (3, 6, 8)  # longer than context
    swapped = chunks.clone()
    swapped[:, :, [0, 1]] = chunks[:, :, [1, 0]]
    differs = chunks[:, :, 0] != chunks[:, :, 1]
    assert differs.any() and not torch.allclose(top(z, chunks), top(z, swapped))


def test_a_narrow_top_model_works_on_a_wide_latent() -> None:
    """width: the top's transformer runs at `width` and maps back to the latent (v4's 2592 numbers), so it stays
    small; width 0 keeps the old layout (transformer as wide as the latent) so old checkpoints load."""
    wide = TopModel(top_cfg(latent_dim=2592, heads=4, width=64)).eval()
    assert sum(p.numel() for p in wide.parameters()) < 1_000_000
    z, chunks = torch.randn(2, 3, 2592), torch.randint(5, (2, 3, 4))
    assert wide(z, chunks).shape == (2, 3, 2592) and wide.rollout(z[:, 0], chunks).shape == (2, 3, 2592)
    old = TopModel(top_cfg(latent_dim=8, heads=2))
    assert old.cfg.width == 0 and not hasattr(old, "inp")
    with pytest.raises(ValueError):
        top_cfg(latent_dim=8, heads=3, width=10)  # width must be divisible by heads


def test_window_dataset_lines_up_latents_and_action_chunks() -> None:
    lat = [np.arange(30, dtype=np.float32)[:, None].repeat(2, 1)]  # latent t == [t, t]
    acts = [np.arange(29) % 5]
    data = TopWindowDataset(lat, acts, stride=4, steps=3)
    z, chunks = data[0]
    assert z[:, 0].tolist() == [0, 4, 8, 12]  # steps + 1 latents, stride apart
    assert chunks.tolist() == [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1]]
    assert len(data) == 30 - 12  # every start with room for 3 jumps of 4
    with pytest.raises(ValueError):
        TopWindowDataset(lat, acts, stride=10, steps=3)


def test_encode_episodes_uses_the_frozen_bottom_encoder(episodes: list[Episode]) -> None:
    lat = encode_episodes(exact_model(), episodes[:2])
    assert [len(z) for z in lat] == [len(e.frames) for e in episodes[:2]] and lat[0].shape[1] == 2


def test_top_learns_four_step_jumps_and_resumes(tmp_path: Path, episodes: list[Episode]) -> None:
    cfg, ckpt = top_cfg(latent_dim=8, heads=2), tmp_path / "top.pt"
    lat = encode_episodes(exact_model(), episodes)
    data = TopWindowDataset([np.repeat(z, 4, 1) for z in lat], [e.actions for e in episodes], 4, 3)  # D = 8
    hist = train_top(TopModel(cfg), data, TrainConfig(steps=150, batch_size=16, lr=3e-3, checkpoint_every=50), ckpt)
    first, last = np.mean([h["loss"] for h in hist[:5]]), np.mean([h["loss"] for h in hist[-5:]])
    assert last < 0.3 * first
    more = train_top(TopModel(cfg), data, TrainConfig(steps=160, batch_size=16, lr=3e-3, checkpoint_every=50), ckpt)
    assert [h["step"] for h in more] == list(range(1, 161))
    top, meta = load_top(ckpt)
    assert top.cfg == cfg and meta["step"] == 160


def test_cli_train_top(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, wm, top = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "top.pt"
    main(["collect", "--episodes", "4", "--max-steps", "30", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "2",
          "--batch-size", "8", "--checkpoint", str(wm)])  # fmt: skip
    capsys.readouterr()
    main(["train-top", "--data", str(data), "--bottom", str(wm), "--stride", "4", "--window", "3", "--steps", "3",
          "--batch-size", "8", "--checkpoint", str(top), "--seed", "1"])  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["steps"] == 3 and out["stride"] == 4 and top.exists()
    model, _ = load_top(top)
    assert model.cfg.latent_dim == 16 and model.cfg.num_actions == 5
