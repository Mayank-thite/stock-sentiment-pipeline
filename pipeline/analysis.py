"""Does headline sentiment line up with returns?

Reads the fact table built by the SQL step and writes:
  outputs/results.json   every number the README quotes
  outputs/results.md     the same numbers as Markdown tables
  outputs/monthly.csv    monthly sentiment series (used by charts and dashboard)

Usage:
  python -m pipeline.analysis [--sample]
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.config import ROOT, get_paths

WINDOWS = [
    ("ret_prev", "Day before (t-1)"),
    ("ret_same", "News day (t)"),
    ("ret_next", "Day after (t+1)"),
]
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


def pearson_ci(r: float, n: int, z_crit: float = 1.959964):
    """95% confidence interval for a Pearson r via the Fisher z transform."""
    if n <= 3 or not np.isfinite(r):
        return (float("nan"), float("nan"))
    z = math.atanh(max(min(r, 0.999999), -0.999999))
    se = 1 / math.sqrt(n - 3)
    return (math.tanh(z - z_crit * se), math.tanh(z + z_crit * se))


def correlate(x: pd.Series, y: pd.Series) -> dict:
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


def bucket_table(fact: pd.DataFrame) -> list:
    rows = []
    for lab in ["negative", "neutral", "positive"]:
        g = fact[fact["day_label"] == lab]
        row = {"day_label": lab, "n_days": int(len(g))}
        for col in ["ret_same", "ret_next", "abn_same", "abn_next"]:
            v = g[col].dropna()
            row[f"mean_{col}"] = float(v.mean())
            row[f"se_{col}"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
            row[f"n_{col}"] = int(len(v))
        rows.append(row)
    return rows


def pos_vs_neg(fact: pd.DataFrame, col: str) -> dict:
    a = fact.loc[fact["day_label"] == "positive", col].dropna()
    b = fact.loc[fact["day_label"] == "negative", col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_positive": float(a.mean()),
        "mean_negative": float(b.mean()),
        "difference": float(a.mean() - b.mean()),
        "welch_t": float(t),
        "p_value": float(p),
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


def by_year(fact: pd.DataFrame) -> list:
    """Correlation per calendar year, a rough check that one period is not driving the pooled result."""
    rows = []
    for year, g in fact.groupby(fact["trade_date"].dt.year):
        row = {"year": int(year), "n": int(len(g))}
        for col in ["ret_same", "ret_next", "abn_next"]:
            ok = g[col].notna()
            row[f"r_{col}"] = float(stats.pearsonr(g.loc[ok, "avg_compound"], g.loc[ok, col])[0])
        rows.append(row)
    return rows


def per_ticker(fact: pd.DataFrame, col: str = "ret_next") -> dict:
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
    return {
        "min_days": MIN_DAYS_PER_TICKER,
        "n_tickers": int(len(df)),
        "median_r": float(df["r"].median()),
        "share_positive_r": float((df["r"] > 0).mean()),
        "n_p_below_0_05": int((df["p"] < 0.05).sum()),
        "rows": df.sort_values("r").to_dict("records"),
    }


def monthly(fact: pd.DataFrame, headlines: pd.DataFrame) -> pd.DataFrame:
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
    """News day correlation with and without days that have a move word headline."""
    h = headlines.copy()
    h["has_move_word"] = h["headline"].str.contains(MOVE_WORDS, case=False, regex=True, na=False)
    flagged = h.groupby(["ticker", "trade_date"])["has_move_word"].any().rename("day_has_move_word").reset_index()
    flagged["trade_date"] = pd.to_datetime(flagged["trade_date"])
    f = fact.merge(flagged, on=["ticker", "trade_date"], how="left")
    clean = f[f["day_has_move_word"] == False]  # noqa: E712
    return {
        "share_headlines_with_move_word": float(h["has_move_word"].mean()),
        "days_without_move_word": int(len(clean)),
        "ret_same_without_move_word": correlate(clean["avg_compound"], clean["ret_same"]),
        "ret_next_without_move_word": correlate(clean["avg_compound"], clean["ret_next"]),
    }


def analyse(fact: pd.DataFrame, headlines: pd.DataFrame) -> dict:
    fact = fact.copy()
    fact["trade_date"] = pd.to_datetime(fact["trade_date"])
    label_counts = headlines["sentiment_label"].value_counts().to_dict()
    res = {
        "sample": {
            "fact_rows": int(len(fact)),
            "tickers": int(fact["ticker"].nunique()),
            "headlines_in_fact": int(fact["n_headlines"].sum()),
            "first_date": fact["trade_date"].min().strftime("%Y-%m-%d"),
            "last_date": fact["trade_date"].max().strftime("%Y-%m-%d"),
            "mean_headlines_per_row": float(fact["n_headlines"].mean()),
            "headline_labels": {k: int(v) for k, v in label_counts.items()},
            "mean_compound_headline": float(headlines["compound"].mean()),
        },
        "correlations": {},
        "correlations_abnormal": {},
    }
    for col, _ in WINDOWS:
        res["correlations"][col] = correlate(fact["avg_compound"], fact[col])
        abn = col.replace("ret_", "abn_")
        res["correlations_abnormal"][abn] = correlate(fact["avg_compound"], fact[abn])
    res["buckets"] = bucket_table(fact)
    res["pos_vs_neg"] = {c: pos_vs_neg(fact, c) for c in ["ret_same", "ret_next", "abn_next"]}
    res["hit_rate"] = {c: hit_rate(fact, c) for c in ["ret_same", "ret_next"]}
    res["per_ticker_next"] = per_ticker(fact, "ret_next")
    res["by_year"] = by_year(fact)
    if "headline" in headlines.columns:
        res["move_words"] = move_word_check(fact, headlines)
    return res


def fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fmt_p(p):
    return "<0.0001" if np.isfinite(p) and p < 1e-4 else fmt(p, 4)


def to_markdown(res: dict) -> str:
    s = res["sample"]
    lines = [
        "# Results",
        "",
        "Generated by `python -m pipeline.analysis`. Do not edit by hand.",
        "",
        f"- Ticker days with news and returns: {s['fact_rows']}",
        f"- Tickers: {s['tickers']}",
        f"- Headlines in those rows: {s['headlines_in_fact']}",
        f"- Date range: {s['first_date']} to {s['last_date']}",
        f"- Headline labels: {s['headline_labels']}",
        f"- Raw headlines scored: {s.get('raw_headlines')}",
        f"- Share of raw timestamps at exactly 00:00:00 (no time of day): {fmt(s.get('share_timestamp_midnight'), 4)}",
        "",
        "## Correlation of daily average sentiment with returns",
        "",
        "| Window | Return | n | Pearson r | 95% CI | p | Spearman rho | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, block in [("raw", res["correlations"]), ("abnormal", res["correlations_abnormal"])]:
        for col, c in block.items():
            lines.append(
                f"| {col} | {kind} | {c['n']} | {fmt(c['pearson_r'])} | "
                f"[{fmt(c['pearson_ci95'][0])}, {fmt(c['pearson_ci95'][1])}] | {fmt_p(c['pearson_p'])} | "
                f"{fmt(c['spearman_rho'])} | {fmt_p(c['spearman_p'])} |"
            )
    lines += [
        "",
        "## Mean return by day label (percent)",
        "",
        "| Day label | Days | News day | Day after | Day after, abnormal |",
        "|---|---|---|---|---|",
    ]
    for b in res["buckets"]:
        lines.append(
            f"| {b['day_label']} | {b['n_days']} | {fmt(100 * b['mean_ret_same'], 3)} | "
            f"{fmt(100 * b['mean_ret_next'], 3)} | {fmt(100 * b['mean_abn_next'], 3)} |"
        )
    lines += ["", "## Positive days minus negative days (Welch t test)", "",
              "| Return | Difference (pct points) | t | p | n pos | n neg |", "|---|---|---|---|---|---|"]
    for col, t in res["pos_vs_neg"].items():
        lines.append(
            f"| {col} | {fmt(100 * t['difference'], 3)} | {fmt(t['welch_t'], 2)} | {fmt_p(t['p_value'])} | "
            f"{t['n_positive']} | {t['n_negative']} |"
        )
    lines += ["", "## Direction hit rate (non neutral days)", "",
              "Baseline is the hit rate you would expect if sentiment and returns were unrelated, "
              "given how often days are positive and how often stocks rise.", "",
              "| Return | n | Hit rate | Baseline if unrelated | Binomial p vs 0.5 |", "|---|---|---|---|---|"]
    for col, h in res["hit_rate"].items():
        lines.append(f"| {col} | {h['n']} | {fmt(h['hit_rate'], 4)} | {fmt(h['baseline_if_independent'], 4)} | "
                     f"{fmt_p(h['binom_p_vs_half'])} |")
    lines += ["", "## Pearson r by year", "", "| Year | n | News day | Day after | Day after, abnormal |", "|---|---|---|---|---|"]
    for r in res["by_year"]:
        lines.append(f"| {r['year']} | {r['n']} | {fmt(r['r_ret_same'])} | {fmt(r['r_ret_next'])} | {fmt(r['r_abn_next'])} |")
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
            f"- News day r on those days: {fmt(a['pearson_r'])} "
            f"[{fmt(a['pearson_ci95'][0])}, {fmt(a['pearson_ci95'][1])}], n={a['n']}",
            f"- Day after r on those days: {fmt(b['pearson_r'])} "
            f"[{fmt(b['pearson_ci95'][0])}, {fmt(b['pearson_ci95'][1])}], n={b['n']}",
        ]
    pt = res["per_ticker_next"]
    if pt["n_tickers"]:
        lines += [
            "",
            f"## Per ticker correlation with next day return (tickers with at least {pt['min_days']} days)",
            "",
            f"- Tickers: {pt['n_tickers']}",
            f"- Median r: {fmt(pt['median_r'])}",
            f"- Share with r above 0: {fmt(pt['share_positive_r'], 3)}",
            f"- Tickers with p below 0.05: {pt['n_p_below_0_05']}",
        ]
    return "\n".join(lines) + "\n"


def run(sample: bool) -> dict:
    paths = get_paths(sample)
    fact = pd.read_parquet(paths.work / "fact_sentiment_daily.parquet")
    headlines = pd.read_parquet(paths.work / "headline_trading_day.parquet")
    res = analyse(fact, headlines)
    raw = pd.read_parquet(paths.warehouse / "headline_scores", columns=["published_at"])
    res["sample"]["raw_headlines"] = int(len(raw))
    res["sample"]["share_timestamp_midnight"] = float(raw["published_at"].str.endswith("00:00:00").mean())
    res["run"] = {"dataset": "sample" if sample else "full"}
    out = ROOT / "outputs" / ("sample" if sample else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "results.md").write_text(to_markdown(res))
    monthly(fact, headlines).to_csv(out / "monthly.csv", index=False)
    c = res["correlations"]
    print(
        f"analysis: n={res['sample']['fact_rows']} "
        f"r(t-1)={c['ret_prev']['pearson_r']:.4f} r(t)={c['ret_same']['pearson_r']:.4f} "
        f"r(t+1)={c['ret_next']['pearson_r']:.4f}  -> {out / 'results.md'}"
    )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args(argv)
    run(args.sample)


if __name__ == "__main__":
    main()
