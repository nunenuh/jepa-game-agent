"""Does the memory work? Replay recorded games and score the agent's own tile map against the game's truth
(true position and blocks are read only here, to score; the map itself never sees them)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from jepa_game_agent.data.episodes import Episode
from jepa_game_agent.memory.tilemap import MOVES, TileMap


def replay_positions(env: Any, episode: Episode) -> tuple[NDArray[np.uint8], NDArray[np.int64], list[Any]]:
    """Replayed frames (T'+1, H, W, C), true positions (T'+1, 2) and block maps, stopping if the game ends."""
    frames = [np.asarray(env.reset(episode.seed), np.uint8)]
    pos, maps = [env.position()], [env.block_map()]
    for a in episode.actions:
        res = env.step(int(a))
        frames.append(np.asarray(res.frame, np.uint8))
        pos.append(env.position())
        maps.append(env.block_map())
        if res.done:
            break
    return np.stack(frames), np.asarray(pos, np.int64), maps


def check_memory(
    env: Any,
    episodes: Sequence[Episode],
    encode: Callable[[NDArray[np.uint8]], torch.Tensor],
    token_dim: int,
    grid: int,
    key_goal: torch.Tensor | None = None,
    want: int | None = None,
    sim: float = 0.9,
) -> dict[str, Any]:
    """Shift test accuracy (moves only), position error (tiles, L1) and, with key_goal + want, how often recall
    points at a tile that really holds block `want` (e.g. a tree for the collect_wood goal picture)."""
    shift_ok = shift_n = fired = hits = 0
    errors: list[int] = []
    at_200: list[int] = []
    for ep in episodes:
        frames, pos, maps = replay_positions(env, ep)
        lat = encode(frames)
        m = TileMap(token_dim, grid)
        start = pos[0]
        for t in range(len(frames)):
            a = None if t == 0 else int(ep.actions[t - 1])
            moved = m.observe(lat[t], a)
            if a in MOVES:
                shift_n += 1
                shift_ok += moved == bool((pos[t] != pos[t - 1]).any())
            err = int(np.abs(np.asarray(m.pos) - (pos[t] - start)).sum())
            errors.append(err)
            if t == 200:
                at_200.append(err)
            if key_goal is not None and want is not None:
                tile = m.recall(m.key(key_goal), sim)
                if tile is not None:
                    fired += 1
                    r, c = start[0] + tile[0], start[1] + tile[1]
                    block = maps[t]
                    inside = 0 <= r < block.shape[0] and 0 <= c < block.shape[1]
                    hits += int(inside and block[r, c] == want)
    return {
        "episodes": len(episodes),
        "steps": len(errors),
        "shift_accuracy": shift_ok / shift_n if shift_n else None,
        "position_error_median": float(np.median(errors)) if errors else None,
        "position_error_at_200_median": float(np.median(at_200)) if at_200 else None,
        "recall_fired": fired,
        "recall_precision": hits / fired if fired else None,
    }
