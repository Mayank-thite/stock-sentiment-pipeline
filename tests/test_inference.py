import numpy as np
import pandas as pd
import pytest

from pipeline.inference import adjust_pvalues, clustered_mean_diff, clustered_slope, fama_macbeth


def _panel(n_dates=200, n_tickers=30, beta=0.0, common=0.0, seed=0):
    """Fake panel. common adds a date effect to x and a separate date effect to y,
    like market wide news days and market wide moves. x and y are unrelated unless beta > 0."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_tickers)
    tickers = np.tile([f"T{i}" for i in range(n_tickers)], n_dates)
    shock_x = np.repeat(rng.normal(size=n_dates), n_tickers)
    shock_y = np.repeat(rng.normal(size=n_dates), n_tickers)
    x = rng.normal(size=len(dates)) + common * shock_x
    y = beta * x + rng.normal(size=len(dates)) + common * shock_y
    return pd.DataFrame({"trade_date": dates, "ticker": tickers, "x": x, "y": y})


def test_slope_equals_pearson_r():
    df = _panel(beta=0.3)
    out = clustered_slope(df, "x", "y")
    assert out["r"] == pytest.approx(np.corrcoef(df["x"], df["y"])[0, 1], abs=1e-12)


def test_date_clustering_widens_errors_when_dates_have_their_own_shocks():
    df = _panel(common=1.5)
    out = clustered_slope(df, "x", "y")
    # both sides move together within a date, so 30 rows per date are worth far fewer than 30
    assert out["se_date"] > 2 * out["se_iid"]


def test_fama_macbeth_finds_the_real_slope_through_date_shocks():
    df = _panel(beta=0.2, common=1.5)
    fm = fama_macbeth(df, "x", "y", min_obs=10)
    assert fm["n_dates"] == 200
    assert fm["mean_slope"] == pytest.approx(0.2, abs=0.03)
    assert fm["t_newey_west"] > 5
    assert fama_macbeth(_panel(common=1.5), "x", "y")["p_newey_west"] > 0.01


def test_fama_macbeth_skips_thin_dates():
    df = _panel(n_tickers=5)
    assert fama_macbeth(df, "x", "y", min_obs=10)["n_dates"] == 0


def test_clustered_mean_diff_matches_group_means():
    df = _panel()
    flag = df["x"] > 0
    out = clustered_mean_diff(df, flag, "y")
    assert out["difference"] == pytest.approx(df.loc[flag, "y"].mean() - df.loc[~flag, "y"].mean())


def test_adjust_pvalues():
    adj = adjust_pvalues(pd.Series([0.01, 0.02, 0.04, 0.5]))
    assert list(adj["p_holm"].round(4)) == [0.04, 0.06, 0.08, 0.5]
    assert (adj["p_bh"] >= pd.Series([0.01, 0.02, 0.04, 0.5])).all()
