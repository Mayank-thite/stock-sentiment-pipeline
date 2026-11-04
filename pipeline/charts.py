"""PNG charts for the README, built from outputs/results.json and monthly.csv.

Same two colours as the dashboard: warm amber for the series I care most about
(return minus the market) and cool blue for the second one. The shades are a
bit deeper than on the dark dashboard so they hold up on a white README.

Usage:
  python -m pipeline.charts [--sample]
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.config import ROOT, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#FFFFFF"
INK = "#0B0D15"
INK_2 = "#4A5163"
GRID = "#E6E8EE"
AXIS = "#B8BDCA"
AMBER = "#C47A12"
COOL = "#2F78C8"

WINDOW_LABELS = {"prev": "Day before\n(t-1)", "same": "News day\n(t)", "next": "Day after\n(t+1)",
                 "2d": "t and t+1\ntogether"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def _style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=26 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=10.5, color=INK_2)


def _fig(w: float = 9, h: float = 5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, out) -> None:
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _grouped(ax, groups: list[str], series: list[tuple], fmt: str) -> None:
    """series: (name, colour, values, lows, highs). Direct labels on the first group, no legend."""
    w = 0.38
    for i, (_name, color, vals, lo, hi) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, vals, width=w - 0.05, color=color)
        ax.errorbar(xs, vals, yerr=[[v - a for v, a in zip(vals, lo)], [b - v for v, b in zip(vals, hi)]],
                    fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        for x, v, top, bot in zip(xs, vals, hi, lo):
            y = top if v >= 0 else bot
            ax.annotate(fmt.format(v), (x, y), xytext=(0, 4 if v >= 0 else -14), textcoords="offset points",
                        ha="center", fontsize=9.5, color=INK_2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.margins(y=0.12)  # room for the value labels
    ax.set_xticks(range(len(groups)), groups)


def _series_key(ax, items: list[tuple[str, str]]) -> None:
    """A one line colour key written as text under the axis, instead of a legend box."""
    x = 0.0
    for name, color in items:
        ax.text(x, -0.2, "■ " + name, transform=ax.transAxes, color=color, fontsize=10.5, fontweight="bold")
        x += 0.06 + 0.0105 * len(name)


def chart_correlations(res: dict, out) -> None:
    fig, ax = _fig()
    keys = list(WINDOW_LABELS)
    series = []
    for kind, name, color in [("raw", "Raw return", COOL), ("abnormal", "Return minus market average", AMBER)]:
        c = [res["windows"][k][kind] for k in keys]
        series.append((name, color, [x["r"] for x in c], [x["ci95_date"][0] for x in c],
                       [x["ci95_date"][1] for x in c]))
    _grouped(ax, [WINDOW_LABELS[k] for k in keys], series, "{:.3f}")
    ax.set_ylabel("Pearson r (sentiment vs return)", color=INK_2)
    same = res["windows"]["same"]["abnormal"]
    nxt = res["windows"]["next"]["abnormal"]
    title = "Headline tone moves with the same day return, not the next day"
    if not (same["ci95_date"][0] > 0 and nxt["ci95_date"][0] <= 0 <= nxt["ci95_date"][1]):
        title = "Correlation of headline tone with returns, by window"
    _style(ax, title, f"{same['n']:,} ticker days. Whiskers are 95% CIs with errors clustered by date.")
    _series_key(ax, [(s[0], s[1]) for s in series])
    _save(fig, out)


def chart_buckets(res: dict, out) -> None:
    fig, ax = _fig()
    b = res["buckets"]
    series = []
    for col, name, color in [("ret_same", "News day (t)", AMBER), ("ret_next", "Day after (t+1)", COOL)]:
        m = [100 * x[f"mean_{col}"] for x in b]
        se = [100 * 1.96 * x[f"se_date_{col}"] for x in b]
        series.append((name, color, m, [v - e for v, e in zip(m, se)], [v + e for v, e in zip(m, se)]))
    _grouped(ax, [f"{x['day_label']}\n{x['n_days']:,} days" for x in b], series, "{:.2f}%")
    ax.set_ylabel("Mean daily return (%)", color=INK_2)
    gap = res["pos_vs_neg"]["ret_same"]["difference"] * 100
    _style(ax, f"Positive news days beat negative ones by {gap:.2f} percentage points on the day",
           "Day label from the average VADER score of that day's headlines. 95% CIs clustered by date.")
    _series_key(ax, [(s[0], s[1]) for s in series])
    _save(fig, out)


def chart_monthly(monthly: pd.DataFrame, out) -> None:
    fig, ax = _fig(9, 4.2)
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["month"])
    ax.plot(m["month"], m["avg_compound"], color=AMBER, linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylim(min(0, m["avg_compound"].min()) * 1.1, m["avg_compound"].max() * 1.15)
    ax.set_ylabel("Mean VADER compound", color=INK_2)
    above = int((m["avg_compound"] > 0).sum())
    title = (f"Headlines leaned positive in all {len(m)} months" if above == len(m)
             else f"Headlines leaned positive in {above} of {len(m)} months")
    _style(ax, title, "Average VADER compound per month, all tickers pooled. 0 is neutral.")
    _save(fig, out)


def chart_per_ticker(res: dict, out) -> None:
    pt = res["per_ticker"]["ret_next"]
    fig, ax = _fig(9, 4.6)
    rs = [r["r"] for r in pt["rows"]]
    ax.hist(rs, bins=20, color=COOL, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("Pearson r, sentiment vs next day return", color=INK_2)
    ax.set_ylabel("Tickers", color=INK_2)
    holm = pt["n_holm_below_0_05"]
    title = ("No single ticker shows a next day link after correction" if holm == 0
             else f"{holm} of {pt['n_tickers']} tickers keep a next day link after correction")
    _style(ax, title,
           f"{pt['n_tickers']} tickers with {pt['min_days']}+ news days. {pt['n_p_below_0_05']} have raw p < 0.05, "
           f"chance alone gives about {pt['expected_by_chance']:.0f}. Holm: {holm}.")
    _save(fig, out)


def run(sample: bool) -> None:
    src = ROOT / "outputs" / ("sample" if sample else "")
    res = json.loads((src / "results.json").read_text())
    monthly = pd.read_csv(src / "monthly.csv")
    img = ROOT / "docs" / "img" / ("sample" if sample else "")
    img.mkdir(parents=True, exist_ok=True)
    chart_correlations(res, img / "correlation_by_window.png")
    chart_buckets(res, img / "returns_by_label.png")
    chart_monthly(monthly, img / "monthly_sentiment.png")
    if res["per_ticker"]["ret_next"]["n_tickers"]:
        chart_per_ticker(res, img / "per_ticker_correlation.png")
    log.info("wrote PNGs to %s", img)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    run(ap.parse_args(argv).sample)


if __name__ == "__main__":
    main()
