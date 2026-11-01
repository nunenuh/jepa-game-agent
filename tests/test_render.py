"""Watch the agent: played worlds saved as episodes, rendered to a GIF with the threat head's view on top."""

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_game_agent.data.episodes import load_episodes
from jepa_game_agent.envs import GridWorld
from jepa_game_agent.envs.recording import RecordingEnv
from jepa_game_agent.evaluation.render import render_frames, save_gif


def test_render_scales_frames_and_reddens_threatened_tiles() -> None:
    frames = np.full((3, 6, 6, 3), 100, np.uint8)
    threat = np.zeros((3, 4))  # a 2 x 2 tile grid
    threat[1, 3] = 1.0  # bottom-right tile, frame 1
    out = render_frames(frames, threat=threat, scale=2)
    assert out.shape == (3, 12, 12, 3) and out.dtype == np.uint8
    hot, cold = out[1, 8, 8], out[1, 2, 2]
    assert hot[0] > cold[0] and hot[1] < cold[1]  # redder where the zombie is expected
    assert np.array_equal(out[0], render_frames(frames[:1], scale=2)[0])  # no threat: just scaled
    with pytest.raises(ValueError):
        render_frames(frames, threat=np.zeros((3, 5)))  # not a square tile grid


def test_save_gif_writes_every_frame(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "a.gif"
    save_gif(np.random.default_rng(0).integers(0, 255, (4, 8, 8, 3), dtype=np.uint8), path, fps=5)
    with Image.open(path) as gif:
        assert gif.n_frames == 4


def test_recording_env_keeps_the_first_worlds_as_episodes() -> None:
    env = RecordingEnv(GridWorld(seed=0), limit=2)
    for seed in range(3):
        env.reset(seed)
        for a in (1, 2, 4):
            env.step(a)
    eps = env.episodes()
    assert len(eps) == 2 and [e.seed for e in eps] == [0, 1]
    assert eps[0].frames.shape[0] == 4 and list(eps[0].actions) == [1, 2, 4]
    assert env.num_actions == GridWorld.num_actions  # everything else passes through


def test_cli_records_play_and_renders_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main

    data, wm, out, gif = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "a.json", tmp_path / "a.gif"
    main(["collect", "--episodes", "6", "--max-steps", "60", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--latent-dim", "16", "--window", "4", "--steps", "1",
          "--batch-size", "4", "--checkpoint", str(wm)])  # fmt: skip
    main(["plan-achievement", "--chain", "--record", "1", "--checkpoint", str(wm), "--goal-data", str(data),
          "--achievement", "collect_item", "--resets", "2", "--budget", "5", "--goal-frames", "2", "--horizon", "2",
          "--samples", "8", "--elites", "2", "--iterations", "1", "--out", str(out)])  # fmt: skip
    play = out.with_suffix(".play.npz")
    assert len(load_episodes(play)) == 1
    capsys.readouterr()
    main(["render", "--data", str(play), "--episode", "0", "--scale", "2", "--out", str(gif)])
    assert gif.exists() and json.loads(capsys.readouterr().out)["frames"] == 6


def test_cli_render_with_the_threat_view(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jepa_game_agent.cli import main
    from jepa_game_agent.models.threat import ThreatHead
    from jepa_game_agent.training.threat_trainer import save_threat

    data, wm, head, gif = tmp_path / "eps.npz", tmp_path / "wm.pt", tmp_path / "t.pt", tmp_path / "a.gif"
    main(["collect", "--episodes", "2", "--max-steps", "10", "--out", str(data)])
    main(["train", "--data", str(data), "--num-actions", "5", "--grid", "4", "--token-dim", "4", "--width", "16",
          "--window", "4", "--steps", "1", "--batch-size", "2", "--checkpoint", str(wm)])  # fmt: skip
    save_threat(head, ThreatHead(token_dim=4, grid=4), [{"step": 0, "loss": 0.0}])
    capsys.readouterr()
    main(["render", "--data", str(data), "--checkpoint", str(wm), "--threat", str(head), "--out", str(gif)])
    assert gif.exists() and json.loads(capsys.readouterr().out)["frames"] == 11
    with pytest.raises(ValueError):
        main(["render", "--data", str(data), "--threat", str(head), "--out", str(gif)])
