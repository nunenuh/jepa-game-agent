"""A fixed-size memory of latents with metadata; lookup by cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class Memory:
    step: int
    info: dict[str, float] = field(default_factory=dict)


class LatentMemory:
    """Ring buffer: when full, the oldest memory is overwritten."""

    def __init__(self, dim: int, capacity: int) -> None:
        if dim < 1 or capacity < 1:
            raise ValueError("dim and capacity must be >= 1")
        self.dim, self.capacity = dim, capacity
        self._z = torch.zeros(capacity, dim)
        self._meta: list[Memory | None] = [None] * capacity
        self._next, self._size = 0, 0

    def __len__(self) -> int:
        return self._size

    def add(self, z: torch.Tensor, step: int, info: dict[str, float] | None = None) -> None:
        if z.shape != (self.dim,):
            raise ValueError(f"z must have shape ({self.dim},)")
        self._z[self._next] = z.detach().float().cpu()
        self._meta[self._next] = Memory(step, dict(info or {}))
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def nearest(self, query: torch.Tensor, k: int = 1) -> list[tuple[float, Memory]]:
        """The k most similar memories, most similar first, as (cosine similarity, memory)."""
        if self._size == 0:
            return []
        sims = F.cosine_similarity(self._z[: self._size], query.detach().float().cpu().unsqueeze(0), dim=-1)
        top = sims.topk(min(k, self._size))
        out = []
        for s, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            meta = self._meta[i]
            assert meta is not None
            out.append((s, meta))
        return out
