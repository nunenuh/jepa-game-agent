"""Turn played frames into something to watch: scaled up, with the threat head's per-tile view in red."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def render_frames(
    frames: NDArray[np.uint8], threat: NDArray[np.floating] | None = None, scale: int = 6
) -> NDArray[np.uint8]:
    """frames (T, H, W, C) -> (T, H * scale, W * scale, C). threat (T, g * g): per-tile zombie probability over
    a g x g grid of tiles covering the frame (the v4 grid), blended in as red."""
    out = frames.astype(np.float32)
    if threat is not None:
        g = math.isqrt(threat.shape[-1])
        if g * g != threat.shape[-1] or len(threat) != len(frames):
            raise ValueError("threat must be (T, g * g) for a square tile grid, one row per frame")
        h, w = frames.shape[1:3]
        rows = np.minimum(np.arange(h) * g // h, g - 1)
        cols = np.minimum(np.arange(w) * g // w, g - 1)
        alpha = 0.6 * threat.reshape(-1, g, g)[:, rows][:, :, cols][..., None]  # (T, H, W, 1)
        out = out * (1 - alpha) + np.array([255.0, 0.0, 0.0]) * alpha
    out = out.repeat(scale, axis=1).repeat(scale, axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_gif(frames: NDArray[np.uint8], path: str | Path, fps: int = 8) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
