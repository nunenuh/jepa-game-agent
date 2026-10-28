"""Statistics for comparing agents on the same tasks (phase 3 success bar)."""

import pytest

from jepa_game_agent.evaluation.stats import mcnemar, wilson


def test_wilson_interval() -> None:
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.05 < hi < 0.08  # 0 of 50: still up to ~7% plausible
    lo, hi = wilson(3, 50)
    assert 0.0 < lo < 0.06 < hi  # 3 of 50: the lower bound clears 0
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
    with pytest.raises(ValueError):
        wilson(5, 0)


def test_mcnemar_exact_paired_test() -> None:
    a = [True] * 10 + [False] * 90
    assert mcnemar(a, a) == 1.0  # identical results: no evidence of a difference
    b = [False] * 100
    assert mcnemar(a, b) == pytest.approx(2 * 0.5**10)  # 10 discordant pairs, all one way
    assert mcnemar(a, b) == mcnemar(b, a)
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])
