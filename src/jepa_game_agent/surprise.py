"""Turn raw prediction errors into 'how unusual is this?' scores."""

from __future__ import annotations

import math


class SurpriseMeter:
    """Running mean/variance (exponential) of prediction error; score = how many std above normal."""

    def __init__(self, decay: float = 0.99, warmup: int = 20) -> None:
        if not 0 < decay < 1 or warmup < 1:
            raise ValueError("decay in (0, 1) and warmup >= 1 required")
        self.decay, self.warmup = decay, warmup
        self.mean, self.var, self.count = 0.0, 0.0, 0

    def update(self, error: float) -> float:
        """Score this error against what came before (0 during warm-up), then learn from it."""
        if not math.isfinite(error) or error < 0:
            raise ValueError("error must be a finite number >= 0")
        score = 0.0
        if self.count >= self.warmup:
            score = (error - self.mean) / math.sqrt(self.var + 1e-12)
        if self.count == 0:
            self.mean = error
        else:
            delta = error - self.mean
            self.mean += (1 - self.decay) * delta
            self.var = self.decay * (self.var + (1 - self.decay) * delta * delta)
        self.count += 1
        return score
