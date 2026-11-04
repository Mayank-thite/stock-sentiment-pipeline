import math

import pandas as pd

from pipeline.analysis import hit_rate, pearson_ci


def test_pearson_ci_contains_r_and_narrows_with_n():
    lo, hi = pearson_ci(0.1, 100)
    assert lo < 0.1 < hi
    lo2, hi2 = pearson_ci(0.1, 10_000)
    assert (hi2 - lo2) < (hi - lo)


def test_pearson_ci_small_n_is_nan():
    lo, hi = pearson_ci(0.5, 3)
    assert math.isnan(lo) and math.isnan(hi)


def test_hit_rate_and_baseline():
    fact = pd.DataFrame(
        {
            "day_label": ["positive", "positive", "negative", "negative", "neutral"],
            "avg_compound": [0.5, 0.3, -0.4, -0.2, 0.0],
            "ret_next": [0.01, -0.02, -0.01, 0.03, 0.05],
        }
    )
    h = hit_rate(fact, "ret_next")
    assert h["n"] == 4  # neutral day is excluded
    assert h["hits"] == 2
    assert h["hit_rate"] == 0.5
    # half the days are positive, half the returns are up -> coin flip baseline
    assert h["baseline_if_independent"] == 0.5
