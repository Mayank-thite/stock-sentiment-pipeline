"""Tests that do not pretend ticker days are independent.

All tickers trade on the same market, so on a big news day (a Fed decision,
a CPI print) most of them move together and most of their headlines turn the
same way. Plain t tests count each of those ticker days as a separate piece of
evidence. The functions here fix that two ways:

  clustered_slope  pooled OLS of standardised return on standardised sentiment,
                   with standard errors clustered by date (and by date and
                   ticker). With both sides standardised the slope is the
                   Pearson r, so the CI reads as a CI for r.
  fama_macbeth     one cross sectional regression per date, then a t test on
                   the time series of daily slopes (Newey-West errors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def _fit_summary(fit, name: str) -> dict:
    lo, hi = fit.conf_int()[1]
    return {
        f"se_{name}": float(fit.bse[1]),
        f"t_{name}": float(fit.tvalues[1]),
        f"p_{name}": float(fit.pvalues[1]),
        f"ci95_{name}": [float(lo), float(hi)],
    }


def clustered_slope(
    df: pd.DataFrame, x: str, y: str, date_col: str = "trade_date", ticker_col: str = "ticker"
) -> dict:
    """Slope of z(y) on z(x) with iid, date clustered and date plus ticker clustered errors.

    The slope equals the Pearson r of x and y. Only the standard errors change.
    """
    d = df[[x, y, date_col, ticker_col]].dropna()
    xs, ys = _zscore(d[x]).to_numpy(), _zscore(d[y]).to_numpy()
    model = sm.OLS(ys, sm.add_constant(xs))
    dates = pd.factorize(d[date_col])[0]
    tickers = pd.factorize(d[ticker_col])[0]
    out = {"n": int(len(d)), "n_dates": int(len(np.unique(dates))), "r": float(model.fit().params[1])}
    out |= _fit_summary(model.fit(), "iid")
    out |= _fit_summary(model.fit(cov_type="cluster", cov_kwds={"groups": dates}), "date")
    out |= _fit_summary(
        model.fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([dates, tickers])}), "date_ticker"
    )
    return out


def clustered_mean_se(values: pd.Series, dates: pd.Series) -> float:
    """Standard error of a mean when rows on the same date are correlated."""
    ok = values.notna()
    if ok.sum() < 2:
        return float("nan")
    fit = sm.OLS(values[ok].to_numpy(), np.ones(int(ok.sum()))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dates[ok])[0]}
    )
    return float(fit.bse[0])


def clustered_mean_diff(df: pd.DataFrame, flag: pd.Series, y: str, date_col: str = "trade_date") -> dict:
    """Mean of y where flag is True minus where it is False, with date clustered errors.

    Same idea as a two sample t test, but a whole date counts as one cluster.
    """
    d = df.assign(_flag=flag.astype(float))[[y, "_flag", date_col]].dropna()
    model = sm.OLS(d[y].to_numpy(), sm.add_constant(d["_flag"].to_numpy()))
    iid = model.fit()
    cl = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[date_col])[0]})
    return {
        "difference": float(iid.params[1]),
        "t_iid": float(iid.tvalues[1]),
        "t_date": float(cl.tvalues[1]),
        "p_date": float(cl.pvalues[1]),
        "ci95_date": [float(v) for v in cl.conf_int()[1]],
        "n": int(len(d)),
    }


def fama_macbeth(
    df: pd.DataFrame, x: str, y: str, date_col: str = "trade_date", min_obs: int = 10, nw_lags: int = 5
) -> dict:
    """Average of daily cross sectional slopes of y on x.

    Each date gives one slope, so market wide moves cancel out within the day
    and the test has one observation per date, not per ticker day. The t stat
    uses Newey-West errors in case slopes are correlated across nearby days.
    """
    d = df[[x, y, date_col]].dropna()
    slopes = []
    for _, g in d.groupby(date_col):
        if len(g) < min_obs or g[x].std(ddof=1) == 0:
            continue
        slopes.append(np.polyfit(g[x].to_numpy(), g[y].to_numpy(), 1)[0])
    s = np.asarray(slopes)
    if len(s) < 3:
        return {"n_dates": int(len(s))}
    fit = sm.OLS(s, np.ones_like(s)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return {
        "n_dates": int(len(s)),
        "min_tickers_per_date": min_obs,
        "mean_slope": float(s.mean()),
        "t_simple": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))),
        "t_newey_west": float(fit.tvalues[0]),
        "p_newey_west": float(fit.pvalues[0]),
        "share_positive_slopes": float((s > 0).mean()),
    }


def adjust_pvalues(p: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg adjusted p-values for a family of tests."""
    p = p.to_numpy()
    return pd.DataFrame(
        {
            "p_holm": multipletests(p, alpha=alpha, method="holm")[1],
            "p_bh": multipletests(p, alpha=alpha, method="fdr_bh")[1],
        }
    )
