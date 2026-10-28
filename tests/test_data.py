from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_game_agent.data.episodes import Episode, WindowDataset, load_episodes, save_episodes


def test_collected_episode_shapes(episodes: list[Episode]) -> None:
    e = episodes[0]
    assert len(e) == 30 and e.frames.shape == (31, 16, 16, 3) and e.actions.dtype == np.int64


def test_episode_validation() -> None:
    with pytest.raises(ValueError):
        Episode(np.zeros((3, 4, 4, 3), np.uint8), np.zeros(3, np.int64), np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        Episode(np.zeros((3, 4, 4, 3), np.float32), np.zeros(2, np.int64), np.zeros(2, np.float32))  # type: ignore[arg-type]


def test_save_load_roundtrip(tmp_path: Path, episodes: list[Episode]) -> None:
    path = tmp_path / "eps.npz"
    save_episodes(path, episodes)
    back = load_episodes(path)
    assert len(back) == len(episodes)
    for a, b in zip(episodes, back, strict=True):
        assert np.array_equal(a.frames, b.frames) and np.array_equal(a.actions, b.actions)
    with pytest.raises(ValueError):
        save_episodes(path, [])


def test_save_creates_missing_folders(tmp_path: Path, episodes: list[Episode]) -> None:
    path = tmp_path / "new" / "dir" / "eps.npz"
    save_episodes(path, episodes)
    assert len(load_episodes(path)) == len(episodes)


def test_window_dataset(episodes: list[Episode]) -> None:
    ds = WindowDataset(episodes, window=5)
    assert len(ds) == 4 * (30 - 5 + 1)
    frames, actions = ds[0]
    assert frames.shape == (6, 3, 16, 16) and actions.shape == (5,)
    assert frames.min() >= 0.0 and frames.max() <= 1.0 and frames.dtype == torch.float32
    assert torch.equal(actions, torch.from_numpy(episodes[0].actions[:5]))
    with pytest.raises(ValueError):
        WindowDataset(episodes, window=31)


def test_episode_keeps_seed_and_achievements(tmp_path: Path) -> None:
    e = Episode(
        np.zeros((3, 4, 4, 3), np.uint8),
        np.zeros(2, np.int64),
        np.zeros(2, np.float32),
        seed=7,
        achievements=("collect_wood", "place_table"),
    )
    path = tmp_path / "eps.npz"
    save_episodes(path, [e, Episode(e.frames, e.actions, e.rewards)])
    a, b = load_episodes(path)
    assert a.seed == 7 and a.achievements == ("collect_wood", "place_table")
    assert b.seed == -1 and b.achievements == ()


def test_collect_episode_records_seed(episodes: list[Episode]) -> None:
    assert [e.seed for e in episodes] == [0, 1, 2, 3]


def test_split_of_follows_the_seed_rule() -> None:
    from jepa_game_agent.data.episodes import split_of

    assert [split_of(s) for s in (0, 1, 2, 19, 20, 21, 40)] == ["test", "val", "train", "train", "test", "val", "test"]
    with pytest.raises(ValueError):
        split_of(-1)  # unknown seed: can't place it
