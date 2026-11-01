"""A map of the tile tokens the agent has seen, kept at positions it works out from its own latents.

Pixels only: the position comes from the agent's own moves, checked by matching what it sees against the map
(stayed or moved one tile?), never from the game's state. See docs/DESIGN_MEMORY.md."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # Craftax LEFT, RIGHT, UP, DOWN as (row, col)


class TileMap:
    """World tile (row, col), relative to where the world began -> the latest token seen there."""

    def __init__(self, token_dim: int, grid: int = 9, view: tuple[int, int] = (7, 9)) -> None:
        if view[0] > grid or view[1] > grid:
            raise ValueError(f"a {view} view does not fit a {grid} x {grid} token grid")
        self.token_dim, self.grid, self.view = token_dim, grid, view
        self.center = (view[0] // 2, view[1] // 2)  # the player's tile in the view
        self.pos = (0, 0)
        self._tiles: dict[tuple[int, int], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._tiles)

    def _grid(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.numel() != self.grid * self.grid * self.token_dim:
            raise ValueError(f"latent must hold {self.grid} x {self.grid} tokens of {self.token_dim}")
        g = latent.detach().float().cpu().reshape(self.grid, self.grid, self.token_dim)
        return g[: self.view[0], : self.view[1]]  # the world view; the status bar rows are not places

    def _cells(self, at: tuple[int, int]) -> list[tuple[int, int, tuple[int, int]]]:
        """View cells and their world tiles; not the player's own tile (the sprite covers what is there)."""
        cy, cx = self.center
        return [(r, c, (at[0] + r - cy, at[1] + c - cx)) for r in range(self.view[0]) for c in range(self.view[1])
                if (r, c) != (cy, cx)]  # fmt: skip

    def _match(self, seen: torch.Tensor, at: tuple[int, int]) -> float | None:
        """Mean cosine similarity between what is seen and the map, if the player stood at `at`."""
        pairs = [(seen[r, c], self._tiles[w]) for r, c, w in self._cells(at) if w in self._tiles]
        if not pairs:
            return None
        a, b = torch.stack([p for p, _ in pairs]), torch.stack([q for _, q in pairs])
        return float(F.cosine_similarity(a, b, dim=-1).mean())

    def observe(self, latent: torch.Tensor, action: int | None) -> bool:
        """Update the position after `action` (None: first frame) and write the view onto the map.
        Returns whether the player moved. A tie (flat ground) counts as a move."""
        seen = self._grid(latent)
        moved = False
        if action in MOVES:
            dy, dx = MOVES[action]
            there = (self.pos[0] + dy, self.pos[1] + dx)
            stay, go = self._match(seen, self.pos), self._match(seen, there)
            if go is not None and (stay is None or go >= stay - 1e-6):
                self.pos, moved = there, True
        for r, c, w in self._cells(self.pos):
            self._tiles[w] = seen[r, c].clone()
        return moved

    def in_view(self, tile: tuple[int, int]) -> bool:
        cy, cx = self.center
        r, c = tile[0] - self.pos[0] + cy, tile[1] - self.pos[1] + cx
        return 0 <= r < self.view[0] and 0 <= c < self.view[1]

    def recall(self, token: torch.Tensor, min_sim: float) -> tuple[int, int] | None:
        """The nearest remembered tile whose token is at least `min_sim` (cosine) like `token`; None if none is,
        or if one is in view right now (then the goal picture already shows the way)."""
        if not self._tiles:
            return None
        where = list(self._tiles)
        sims = F.cosine_similarity(torch.stack([self._tiles[w] for w in where]), token.float().cpu()[None], dim=-1)
        hits = [w for w, s in zip(where, sims.tolist(), strict=True) if s >= min_sim]
        if not hits or any(self.in_view(w) for w in hits):
            return None
        return min(hits, key=lambda w: abs(w[0] - self.pos[0]) + abs(w[1] - self.pos[1]))

    def key(self, goal: torch.Tensor) -> torch.Tensor:
        """The goal pictures' tile that matters: in each picture (goal (D,) or (N, D)), of the 4 tiles next to the
        player, the one least like ordinary ground (furthest from the map's average token; the pictures' own
        average while the map is empty); averaged over pictures. Picking per picture keeps a tree seen from
        different sides from blurring away."""
        g = goal.detach().float().cpu().reshape(-1, self.grid, self.grid, self.token_dim)
        tiles = list(self._tiles.values())
        ground = torch.stack(tiles).mean(0) if tiles else g.reshape(-1, self.token_dim).mean(0)
        cy, cx = self.center
        near = torch.stack([g[:, cy + dy, cx + dx] for dy, dx in MOVES.values()], 1)  # (N, 4, d)
        pick = (near - ground).norm(dim=-1).argmax(1)
        return near[torch.arange(len(near)), pick].mean(0)


def path_distance(seqs: torch.Tensor, pos: tuple[int, int], target: tuple[int, int]) -> torch.Tensor:
    """Imagined action sequences (S, H) -> (S,): Manhattan distance to `target` after each step, summed. Moves
    are assumed to succeed (the planner's imagination doesn't know about walls yet)."""
    step = torch.zeros(*seqs.shape, 2, dtype=torch.long, device=seqs.device)
    for a, (dy, dx) in MOVES.items():
        step[seqs == a] = torch.tensor([dy, dx], device=seqs.device)
    where = step.cumsum(1) + torch.tensor(pos, device=seqs.device)
    gap = (where - torch.tensor(target, device=seqs.device)).abs().sum(-1)
    return gap.sum(-1).float()
