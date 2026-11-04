"""Does headline sentiment line up with returns?

Reads the fact table built by the SQL step and writes:
  outputs/results.json   every number the README quotes
  outputs/results.md     the same numbers as Markdown tables
  outputs/monthly.csv    monthly sentiment series (used by charts and dashboard)

Ticker days on the same date are not independent, so the headline tests use
standard errors clustered by date and Fama-MacBeth daily regressions
(pipeline/inference.py). The plain Pearson CIs are kept only for comparison.

Usage:
  python -m pipeline.analysis [--sample]
"""
from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from pipeline import lexicon
from pipeline.config import FM_MIN_TICKERS_PER_DAY, FM_NEWEY_WEST_LAGS, ROOT, get_paths, setup_logging
from pipeline.inference import (
    adjust_pvalues,
    clustered_mean_diff,
    clustered_mean_se,
    clustered_slope,
    fama_macbeth,
)

log = logging.getLogger(__name__)

WINDOWS = {
    "prev": "Day before (t-1)",
    "same": "News day (t)",
    "next": "Day after (t+1)",
    "2d": "t and t+1 together",
}
MIN_DAYS_PER_TICKER = 100

# Words that usually mean the headline is reporting a price move that already
# happened ("Nvidia stock soars"). Used to check how much of the news day link
# is headlines describing the move rather than predicting it.
MOVE_WORDS = (
    r"\b(?:soar|soars|soared|jump|jumps|jumped|surge|surges|surged|rally|rallies|rallied|"
    r"plunge|plunges|plunged|tumble|tumbles|tumbled|sink|sinks|sank|slide|slides|slid|"
    r"drop|drops|dropped|fall|falls|fell|rise|rises|rose|gain|gains|gained|climb|climbs|climbed|"
    r"slump|slumps|slumped|down|up)\b"
)


def pearson_ci(r: float, n: int, z_crit: float = 1.959964) -> tuple[float, float]:
    """95% CI for a Pearson r via the Fisher z transform. Assumes independent rows."""
    if n <= 3 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    z = math.atanh(max(min(r, 0.999999), -0.999999))
    se = 1 / math.sqrt(n - 3)
    return (math.tanh(z - z_crit * se), math.tanh(z + z_crit * se))


def correlate(x: pd.Series, y: pd.Series) -> dict:
    """Pearson and Spearman with the naive (independent rows) CI."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    n = int(ok.sum())
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    lo, hi = pearson_ci(r, n)
    return {
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "pearson_ci95": [lo, hi],
        "spearman_rho": float(rho),
        "spearman_p": float(p_s),
    }


def bucket_table(fact: pd.DataFrame) -> list[dict]:
    rows = []
    for lab in ["negative", "neutral", "positive"]:
        g = fact[fact["day_label"] == lab]
        row = {"day_label": lab, "n_days": int(len(g))}
        for col in ["ret_same", "ret_next", "ret_2d", "abn_same", "abn_next"]:
            v = g[col].dropna()
            row[f"mean_{col}"] = float(v.mean())
            row[f"se_{col}"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
            row[f"se_date_{col}"] = clustered_mean_se(g[col], g["trade_date"])
            row[f"n_{col}"] = int(len(v))
        rows.append(row)
    return rows


def pos_vs_neg(fact: pd.DataFrame, col: str) -> dict:
    """Positive days minus negative days: Welch t (naive) and date clustered t."""
    d = fact[fact["day_label"].isin(["positive", "negative"])]
    a = d.loc[d["day_label"] == "positive", col].dropna()
    b = d.loc[d["day_label"] == "negative", col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    cl = clustered_mean_diff(d, d["day_label"] == "positive", col)
    return {
        "mean_positive": float(a.mean()),
        "mean_negative": float(b.mean()),
        "difference": float(a.mean() - b.mean()),
        "welch_t": float(t),
        "welch_p": float(p),
        "t_date_clustered": cl["t_date"],
        "p_date_clustered": cl["p_date"],
        "ci95_date_clustered": cl["ci95_date"],
        "n_positive": int(len(a)),
        "n_negative": int(len(b)),
    }


def hit_rate(fact: pd.DataFrame, col: str) -> dict:
    """Share of non neutral days where the sentiment sign matches the return sign."""
    d = fact[(fact["day_label"] != "neutral") & fact[col].notna() & (fact[col] != 0)]
    hits = int((np.sign(d["avg_compound"]) == np.sign(d[col])).sum())
    n = int(len(d))
    test = stats.binomtest(hits, n, 0.5) if n else None
    # What a coin with the same base rates would score: positive days are more
    # common than negative days and stocks rise more often than they fall, so
    # 0.5 is the wrong yardstick.
    p_pos = float((d["avg_compound"] > 0).mean()) if n else float("nan")
    p_up = float((d[col] > 0).mean()) if n else float("nan")
    baseline = p_pos * p_up + (1 - p_pos) * (1 - p_up)
    return {"n": n, "hits": hits, "hit_rate": hits / n if n else float("nan"),
            "binom_p_vs_half": float(test.pvalue) if test else float("nan"),
            "share_positive_days": p_pos, "share_up_days": p_up,
            "baseline_if_independent": baseline}


def by_year(fact: pd.DataFrame) -> list[dict]:
    """Correlation per calendar year, with a date clustered t for the news day."""
    rows = []
    for year, g in fact.groupby(fact["trade_date"].dt.year):
        cl = clustered_slope(g, "avg_compound", "ret_same")
        row = {"year": int(year), "n": int(len(g)), "t_ret_same_date": cl["t_date"]}
        for col in ["ret_same", "ret_next", "abn_next"]:
            ok = g[col].notna()
            row[f"r_{col}"] = float(stats.pearsonr(g.loc[ok, "avg_compound"], g.loc[ok, col])[0])
        rows.append(row)
    return rows


def per_ticker(fact: pd.DataFrame, col: str) -> dict:
    """One time series correlation per ticker, then Holm and BH corrections across tickers."""
    rows = []
    for tic, g in fact.groupby("ticker"):
        g = g[g[col].notna()]
        if len(g) < MIN_DAYS_PER_TICKER:
            continue
        r, p = stats.pearsonr(g["avg_compound"], g[col])
        rows.append({"ticker": tic, "n": int(len(g)), "r": float(r), "p": float(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_tickers": 0, "rows": []}
    df = pd.concat([df, adjust_pvalues(df["p"])], axis=1)
    return {
        "min_days": MIN_DAYS_PER_TICKER,
        "n_tickers": int(len(df)),
        "median_r": float(df["r"].median()),
        "share_positive_r": float((df["r"] > 0).mean()),
        "n_p_below_0_05": int((df["p"] < 0.05).sum()),
        "expected_by_chance": float(0.05 * len(df)),
        "n_holm_below_0_05": int((df["p_holm"] < 0.05).sum()),
        "n_bh_below_0_05": int((df["p_bh"] < 0.05).sum()),
        "rows": df.sort_values("r").to_dict("records"),
    }


def monthly(headlines: pd.DataFrame) -> pd.DataFrame:
    h = headlines.copy()
    h["month"] = pd.to_datetime(h["trade_date"]).dt.to_period("M").astype(str)
    m = h.groupby("month").agg(
        n_headlines=("headline_id", "count"),
        avg_compound=("compound", "mean"),
        share_positive=("sentiment_label", lambda s: (s == "positive").mean()),
        share_negative=("sentiment_label", lambda s: (s == "negative").mean()),
    )
    return m.reset_index()


def move_word_check(fact: pd.DataFrame, headlines: pd.DataFrame) -> dict:
    """News day link with and without days that have a move word headline."""
    h = headlines.copy()
    h["has_move_word"] = h["headline"].str.contains(MOVE_WORDS, case=False, regex=True, na=False)
    flagged = h.groupby(["ticker", "trade_date"])["has_move_word"].any().rename("day_has_move_word").reset_index()
    f = fact.merge(flagged, on=["ticker", "trade_date"], how="left")
    clean = f[f["day_has_move_word"].eq(False)]
    return {
        "share_headlines_with_move_word": float(h["has_move_word"].mean()),
        "days_without_move_word": int(len(clean)),
        "ret_same_without_move_word": clustered_slope(clean, "avg_compound", "ret_same"),
        "abn_same_without_move_word": clustered_slope(clean, "avg_compound", "abn_same"),
        "ret_next_without_move_word": clustered_slope(clean, "avg_compound", "ret_next"),
    }


def window_tests(fact: pd.DataFrame, x: str) -> dict:
    """Clustered slope for raw and market adjusted returns, plus Fama-MacBeth, per window."""
    out = {}
    for k in WINDOWS:
        out[k] = {
            "raw": clustered_slope(fact, x, f"ret_{k}"),
            "abnormal": clustered_slope(fact, x, f"abn_{k}"),
            "fama_macbeth": fama_macbeth(fact, x, f"ret_{k}", min_obs=FM_MIN_TICKERS_PER_DAY,
                                         nw_lags=FM_NEWEY_WEST_LAGS),
        }
    return out


def lm_comparison(fact: pd.DataFrame, headlines: pd.DataFrame, lists: lexicon.WordLists) -> dict:
    """Same tests as VADER, with the Loughran-McDonald tone in its place."""
    h = lexicon.add_lm_scores(headlines, lists)
    daily = h.groupby(["ticker", "trade_date"], as_index=False)["lm_tone"].mean()
    f = fact.merge(daily, on=["ticker", "trade_date"], how="inner")
    return {
        "agreement": lexicon.label_agreement(h),
        "headline_labels": {k: int(v) for k, v in h["lm_label"].value_counts().items()},
        "windows": window_tests(f, "lm_tone"),
    }


def analyse(fact: pd.DataFrame, headlines: pd.DataFrame, lists: lexicon.WordLists | None = None) -> dict:
    fact = fact.copy()
    headlines = headlines.copy()
    fact["trade_date"] = pd.to_datetime(fact["trade_date"])
    headlines["trade_date"] = pd.to_datetime(headlines["trade_date"])
    label_counts = headlines["sentiment_label"].value_counts().to_dict()
    res = {
        "sample": {
            "fact_rows": int(len(fact)),
            "tickers": int(fact["ticker"].nunique()),
            "trading_dates": int(fact["trade_date"].nunique()),
            "median_tickers_per_date": float(fact.groupby("trade_date").size().median()),
            "headlines_in_fact": int(fact["n_headlines"].sum()),
            "headlines_matched": int(len(headlines)),
            "first_date": fact["trade_date"].min().strftime("%Y-%m-%d"),
            "last_date": fact["trade_date"].max().strftime("%Y-%m-%d"),
            "mean_headlines_per_row": float(fact["n_headlines"].mean()),
            "headline_labels": {k: int(v) for k, v in label_counts.items()},
            "mean_compound_headline": float(headlines["compound"].mean()),
        },
        "correlations_naive": {},
        "correlations_abnormal_naive": {},
    }
    for k in WINDOWS:
        res["correlations_naive"][f"ret_{k}"] = correlate(fact["avg_compound"], fact[f"ret_{k}"])
        res["correlations_abnormal_naive"][f"abn_{k}"] = correlate(fact["avg_compound"], fact[f"abn_{k}"])
    res["windows"] = window_tests(fact, "avg_compound")
    res["buckets"] = bucket_table(fact)
    res["pos_vs_neg"] = {c: pos_vs_neg(fact, c) for c in ["ret_same", "ret_next", "ret_2d", "abn_same", "abn_next"]}
    res["hit_rate"] = {c: hit_rate(fact, c) for c in ["ret_same", "ret_next"]}
    res["per_ticker"] = {c: per_ticker(fact, c) for c in ["ret_same", "ret_next"]}
    res["by_year"] = by_year(fact)
    if "headline" in headlines.columns:
        res["move_words"] = move_word_check(fact, headlines)
    res["phrase_check"] = lexicon.phrase_check(lists)
    if lists is not None and "headline" in headlines.columns:
        res["loughran_mcdonald"] = lm_comparison(fact, headlines, lists)
    return res


def fmt(x: float | None, d: int = 4) -> str:
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fmt_p(p: float) -> str:
    return "<0.0001" if np.isfinite(p) and p < 1e-4 else fmt(p, 4)


def _ci(c: list[float]) -> str:
    return f"[{fmt(c[0])}, {fmt(c[1])}]"


def _window_table(windows: dict) -> list[str]:
    lines = [
        "| Window | Return | n | r | 95% CI, date clustered | t naive | t date clustered | t date and ticker |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, w in windows.items():
        for kind in ["raw", "abnormal"]:
            c = w[kind]
            lines.append(
                f"| {WINDOWS[k]} | {kind} | {c['n']} | {fmt(c['r'])} | {_ci(c['ci95_date'])} | "
                f"{fmt(c['t_iid'], 2)} | {fmt(c['t_date'], 2)} | {fmt(c['t_date_ticker'], 2)} |"
            )
    lines += ["", "Fama-MacBeth: one cross sectional slope per date (return on compound), "
              "then a Newey-West t on the daily slopes.", "",
              "| Window | Dates | Mean slope (pct points per unit compound) | t simple | t Newey-West | p |",
              "|---|---|---|---|---|---|"]
    for k, w in windows.items():
        fm = w["fama_macbeth"]
        if "mean_slope" in fm:
            lines.append(f"| {WINDOWS[k]} | {fm['n_dates']} | {fmt(100 * fm['mean_slope'], 3)} | "
                         f"{fmt(fm['t_simple'], 2)} | {fmt(fm['t_newey_west'], 2)} | {fmt_p(fm['p_newey_west'])} |")
    return lines


def to_markdown(res: dict) -> str:
    s = res["sample"]
    lines = [
        "# Results",
        "",
        "Generated by `python -m pipeline.analysis`. Do not edit by hand.",
        "",
        f"- Ticker days with news and returns: {s['fact_rows']}",
        f"- Tickers: {s['tickers']}",
        f"- Trading dates: {s['trading_dates']} (median {fmt(s['median_tickers_per_date'], 0)} tickers per date)",
        f"- Headlines in those rows: {s['headlines_in_fact']} of {s['headlines_matched']} matched to a trading day",
        f"- Date range: {s['first_date']} to {s['last_date']}",
        f"- Headline labels: {s['headline_labels']}",
        f"- Raw headlines scored: {s.get('raw_headlines')}",
        f"- Share of raw timestamps at exactly 00:00:00 (no time of day): {fmt(s.get('share_timestamp_midnight'), 4)}",
        "",
        "## Sentiment vs returns, with errors that allow for same date correlation",
        "",
        "r is the slope of standardised return on standardised daily VADER compound, which equals the Pearson r.",
        "",
    ]
    lines += _window_table(res["windows"])
    lines += ["", "## Naive Pearson and Spearman (treat every ticker day as independent)", "",
              "| Window | n | Pearson r | 95% CI naive | p naive | Spearman rho |", "|---|---|---|---|---|---|"]
    for block in [res["correlations_naive"], res["correlations_abnormal_naive"]]:
        for col, c in block.items():
            lines.append(f"| {col} | {c['n']} | {fmt(c['pearson_r'])} | {_ci(c['pearson_ci95'])} | "
                         f"{fmt_p(c['pearson_p'])} | {fmt(c['spearman_rho'])} |")
    lines += ["", "## Mean return by day label (percent)", "",
              "| Day label | Days | News day | Day after | t and t+1 | Day after, abnormal |", "|---|---|---|---|---|---|"]
    for b in res["buckets"]:
        lines.append(
            f"| {b['day_label']} | {b['n_days']} | {fmt(100 * b['mean_ret_same'], 3)} | "
            f"{fmt(100 * b['mean_ret_next'], 3)} | {fmt(100 * b['mean_ret_2d'], 3)} | {fmt(100 * b['mean_abn_next'], 3)} |"
        )
    lines += ["", "## Positive days minus negative days", "",
              "| Return | Difference (pct points) | 95% CI, date clustered | Welch t | t date clustered | p date clustered |",
              "|---|---|---|---|---|---|"]
    for col, t in res["pos_vs_neg"].items():
        ci = [100 * v for v in t["ci95_date_clustered"]]
        lines.append(
            f"| {col} | {fmt(100 * t['difference'], 3)} | [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] | {fmt(t['welch_t'], 2)} | "
            f"{fmt(t['t_date_clustered'], 2)} | {fmt_p(t['p_date_clustered'])} |"
        )
    lines += ["", "## Direction hit rate (non neutral days)", "",
              "Baseline is the hit rate you would expect if sentiment and returns were unrelated, "
              "given how often days are positive and how often stocks rise.", "",
              "| Return | n | Hit rate | Baseline if unrelated |", "|---|---|---|---|"]
    for col, h in res["hit_rate"].items():
        lines.append(f"| {col} | {h['n']} | {fmt(h['hit_rate'], 4)} | {fmt(h['baseline_if_independent'], 4)} |")
    lines += ["", "## By year", "",
              "| Year | n | r news day | t news day, date clustered | r day after | r day after, abnormal |",
              "|---|---|---|---|---|---|"]
    for r in res["by_year"]:
        lines.append(f"| {r['year']} | {r['n']} | {fmt(r['r_ret_same'])} | {fmt(r['t_ret_same_date'], 2)} | "
                     f"{fmt(r['r_ret_next'])} | {fmt(r['r_abn_next'])} |")
    for col, pt in res["per_ticker"].items():
        if not pt["n_tickers"]:
            continue
        lines += [
            "",
            f"## Per ticker correlation with {col} (tickers with at least {pt['min_days']} days)",
            "",
            f"- Tickers: {pt['n_tickers']}",
            f"- Median r: {fmt(pt['median_r'])}",
            f"- Share with r above 0: {fmt(pt['share_positive_r'], 3)}",
            f"- p below 0.05 before correction: {pt['n_p_below_0_05']} (chance alone gives about "
            f"{fmt(pt['expected_by_chance'], 1)})",
            f"- Below 0.05 after Holm: {pt['n_holm_below_0_05']}",
            f"- Below 0.05 after Benjamini-Hochberg: {pt['n_bh_below_0_05']}",
        ]
    mw = res.get("move_words")
    if mw:
        a, b = mw["ret_same_without_move_word"], mw["ret_next_without_move_word"]
        lines += [
            "",
            "## Headlines that describe a price move",
            "",
            f"- Share of headlines with a move word (soar, jump, fall, up, down and so on): "
            f"{fmt(mw['share_headlines_with_move_word'], 4)}",
            f"- Ticker days with no move word headline: {mw['days_without_move_word']}",
            f"- News day r on those days: {fmt(a['r'])} {_ci(a['ci95_date'])} date clustered, t = {fmt(a['t_date'], 2)}",
            f"- Day after r on those days: {fmt(b['r'])} {_ci(b['ci95_date'])} date clustered, t = {fmt(b['t_date'], 2)}",
        ]
    pc = res["phrase_check"]
    lines += ["", f"## Finance phrase check ({pc['n']} hand labelled phrases)", "",
              f"- VADER accuracy: {fmt(pc['vader_accuracy'], 3)}",
              f"- Loughran-McDonald accuracy: {fmt(pc.get('lm_accuracy'), 3)}", "",
              "| Phrase | My label | VADER | LM |", "|---|---|---|---|"]
    for r in pc["rows"]:
        lines.append(f"| {r['phrase']} | {r['expected']} | {r['vader']} | {r.get('lm', 'n/a')} |")
    lm = res.get("loughran_mcdonald")
    if lm:
        ag = lm["agreement"]
        lines += ["", "## Loughran-McDonald instead of VADER", "",
                  f"- Same label as VADER: {fmt(ag['share_same_label'], 4)}",
                  f"- Correlation of VADER compound with LM tone, per headline: {fmt(ag['pearson_compound_vs_lm_tone'], 4)}",
                  f"- Share of headlines LM calls neutral (no LM word): {fmt(ag['share_lm_neutral'], 4)}",
                  f"- LM headline labels: {lm['headline_labels']}", ""]
        lines += _window_table(lm["windows"])
    return "\n".join(lines) + "\n"


def run(sample: bool) -> dict:
    paths = get_paths(sample)
    fact = pd.read_parquet(paths.work / "fact_sentiment_daily.parquet")
    headlines = pd.read_parquet(paths.work / "headline_trading_day.parquet")
    res = analyse(fact, headlines, lexicon.load_word_lists())
    raw = pd.read_parquet(paths.warehouse / "headline_scores", columns=["published_at"])
    res["sample"]["raw_headlines"] = int(len(raw))
    res["sample"]["share_timestamp_midnight"] = float(raw["published_at"].str.endswith("00:00:00").mean())
    res["run"] = {"dataset": "sample" if sample else "full"}
    out = ROOT / "outputs" / ("sample" if sample else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "results.md").write_text(to_markdown(res))
    monthly(headlines).to_csv(out / "monthly.csv", index=False)
    w = res["windows"]
    log.info(
        "n=%d  r(t)=%.4f t_date=%.2f  r(t+1)=%.4f t_date=%.2f  -> %s",
        res["sample"]["fact_rows"], w["same"]["raw"]["r"], w["same"]["raw"]["t_date"],
        w["next"]["raw"]["r"], w["next"]["raw"]["t_date"], out / "results.md",
    )
    return res


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args(argv)
    run(args.sample)


if __name__ == "__main__":
    main()
