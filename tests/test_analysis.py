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


def _fake_fact(n_dates=150, n_tickers=12, seed=1):
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2021-01-04", periods=n_dates, freq="B"), n_tickers)
    df = pd.DataFrame({"trade_date": dates, "ticker": np.tile([f"T{i}" for i in range(n_tickers)], n_dates)})
    df["avg_compound"] = rng.uniform(-1, 1, len(df))
    for k in ["prev", "same", "next", "2d"]:
        df[f"ret_{k}"] = rng.normal(0, 0.02, len(df)) + (0.01 * df["avg_compound"] if k == "same" else 0)
        df[f"abn_{k}"] = df[f"ret_{k}"] - df.groupby("trade_date")[f"ret_{k}"].transform("mean")
    df["day_label"] = pd.cut(df["avg_compound"], [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])
    df["day_label"] = df["day_label"].astype(str)
    df["n_headlines"] = 1
    return df


def test_analyse_end_to_end_on_fake_data():
    from pipeline.analysis import analyse

    fact = _fake_fact()
    heads = fact[["ticker", "trade_date", "avg_compound", "day_label"]].rename(
        columns={"avg_compound": "compound", "day_label": "sentiment_label"})
    heads["headline_id"] = range(len(heads))
    res = analyse(fact, heads)
    assert res["windows"]["same"]["raw"]["t_date"] > 5  # planted effect
    assert abs(res["windows"]["next"]["raw"]["t_date"]) < 3
    pt = res["per_ticker"]["ret_next"]
    assert pt["n_tickers"] == 12
    assert pt["n_holm_below_0_05"] <= pt["n_p_below_0_05"]
    assert "loughran_mcdonald" not in res  # no dictionary passed
