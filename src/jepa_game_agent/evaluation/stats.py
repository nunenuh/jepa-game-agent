"""Statistics for the success bar: Wilson intervals for one success rate, McNemar's test for two agents on
the same tasks (paired)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (by default) Wilson score interval for successes / n; well behaved at 0 and n."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact two-sided McNemar p-value: do a and b differ on the tasks where exactly one of them succeeds?"""
    if len(a) != len(b):
        raise ValueError("a and b must cover the same tasks")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return float(min(1.0, 2 * tail))
