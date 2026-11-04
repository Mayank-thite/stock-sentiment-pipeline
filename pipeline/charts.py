"""PNG charts for the README, built from outputs/results.json and monthly.csv.

Usage:
  python -m pipeline.charts [--sample]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.config import ROOT  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

WINDOW_LABELS = {"prev": "Day before\n(t-1)", "same": "News day\n(t)", "next": "Day after\n(t+1)"}


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK_2)


def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def chart_correlations(res, out):
    fig, ax = _fig()
    keys = ["prev", "same", "next"]
    x = range(len(keys))
    w = 0.36
    for i, (block, name, color) in enumerate(
        [("correlations", "Raw return", BLUE), ("correlations_abnormal", "Return minus market average", ORANGE)]
    ):
        prefix = "ret_" if block == "correlations" else "abn_"
        rs = [res[block][prefix + k]["pearson_r"] for k in keys]
        lo = [r - res[block][prefix + k]["pearson_ci95"][0] for r, k in zip(rs, keys)]
        hi = [res[block][prefix + k]["pearson_ci95"][1] - r for r, k in zip(rs, keys)]
        xs = [xi + (i - 0.5) * w for xi in x]
        ax.bar(xs, rs, width=w - 0.04, color=color, label=name, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, rs, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(list(x), [WINDOW_LABELS[k] for k in keys])
    ax.set_ylabel("Pearson r", color=INK_2)
    n = res["correlations"]["ret_same"]["n"]
    _style(ax, "Sentiment vs return correlation, with 95% CI", f"{n:,} ticker days. Bars near zero mean no linear link.")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_buckets(res, out):
    fig, ax = _fig()
    labels = [b["day_label"] for b in res["buckets"]]
    x = range(len(labels))
    w = 0.36
    for i, (col, name, color) in enumerate([("ret_same", "News day (t)", BLUE), ("ret_next", "Day after (t+1)", ORANGE)]):
        means = [100 * b[f"mean_{col}"] for b in res["buckets"]]
        ci = [100 * 1.96 * b[f"se_{col}"] for b in res["buckets"]]
        xs = [xi + (i - 0.5) * w for xi in x]
        ax.bar(xs, means, width=w - 0.04, color=color, label=name, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, means, yerr=ci, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(list(x), [f"{lab}\n{b['n_days']:,} days" for lab, b in zip(labels, res["buckets"])])
    ax.set_ylabel("Mean daily return (%)", color=INK_2)
    _style(ax, "Mean return by day sentiment label, with 95% CI", "Label from the average VADER compound score of that day's headlines.")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_monthly(monthly, out):
    fig, ax = _fig()
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["month"])
    ax.plot(m["month"], m["avg_compound"], color=BLUE, linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("Mean VADER compound", color=INK_2)
    _style(ax, "Average headline sentiment by month", "All tickers pooled. Above zero means headlines lean positive.")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_per_ticker(res, out):
    pt = res["per_ticker_next"]
    fig, ax = _fig()
    rs = [r["r"] for r in pt["rows"]]
    ax.hist(rs, bins=20, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("Pearson r, sentiment vs next day return", color=INK_2)
    ax.set_ylabel("Tickers", color=INK_2)
    _style(ax, "Per ticker correlation with next day return",
           f"{pt['n_tickers']} tickers with at least {pt['min_days']} news days. Median r = {pt['median_r']:.3f}.")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def run(sample: bool):
    src = ROOT / "outputs" / ("sample" if sample else "")
    res = json.loads((src / "results.json").read_text())
    monthly = pd.read_csv(src / "monthly.csv")
    img = ROOT / "docs" / "img" / ("sample" if sample else "")
    img.mkdir(parents=True, exist_ok=True)
    chart_correlations(res, img / "correlation_by_window.png")
    chart_buckets(res, img / "returns_by_label.png")
    chart_monthly(monthly, img / "monthly_sentiment.png")
    if res["per_ticker_next"]["n_tickers"]:
        chart_per_ticker(res, img / "per_ticker_correlation.png")
    print(f"charts: wrote PNGs to {img}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    run(ap.parse_args(argv).sample)


if __name__ == "__main__":
    main()
