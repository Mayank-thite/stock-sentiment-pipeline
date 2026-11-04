"""Build docs/dashboard.html: one self contained page, inline SVG, no CDN.

Usage:
  python -m pipeline.dashboard [--sample]
"""
import argparse
import math
import html
import json

import pandas as pd

from pipeline.config import ROOT

W, H = 480, 300
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 16, 48


def _scale(lo, hi, a, b):
    return lambda v: a + (v - lo) / (hi - lo) * (b - a) if hi != lo else (a + b) / 2


def _yaxis(lo, hi, y, ticks, fmt):
    out = []
    for t in ticks:
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    return out


def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks, t = [start], start
    while t < hi - step * 1e-9:
        t += step
        ticks.append(round(t, 10))
    return ticks


def _p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def grouped_bars(groups, series, value_fmt, axis_fmt):
    """groups: list of labels. series: list of (name, css_class, values, lo, hi)."""
    vals = [v for s in series for v in list(s[3]) + list(s[4])] + [0]
    ticks = _nice_ticks(min(vals), max(vals))
    lo, hi = min(ticks), max(ticks)
    y = _scale(lo, hi, H - PAD_B, PAD_T)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img">'] + _yaxis(lo, hi, y, ticks, axis_fmt)
    gw = (W - PAD_L - PAD_R) / len(groups)
    bw = min(44, gw / (len(series) + 1.2))
    for gi, g in enumerate(groups):
        cx = PAD_L + gw * (gi + 0.5)
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{H - PAD_B + 20}" text-anchor="middle">{html.escape(g)}</text>')
        for si, (name, cls, values, los, his) in enumerate(series):
            x = cx + (si - len(series) / 2) * (bw + 2)
            v = values[gi]
            y0, y1 = y(0), y(v)
            top, h = min(y0, y1), max(abs(y1 - y0), 1)
            tip = html.escape(f"{name}, {g}: {value_fmt(v)} (95% CI {value_fmt(los[gi])} to {value_fmt(his[gi])})")
            parts.append(f'<g class="mark" data-tip="{tip}"><rect class="{cls}" x="{x:.1f}" y="{top:.1f}" '
                         f'width="{bw:.1f}" height="{h:.1f}" rx="3"/>'
                         f'<line class="whisker" x1="{x + bw / 2:.1f}" x2="{x + bw / 2:.1f}" '
                         f'y1="{y(los[gi]):.1f}" y2="{y(his[gi]):.1f}"/>'
                         f'<rect class="hit" x="{x - 4:.1f}" y="{PAD_T}" width="{bw + 8:.1f}" height="{H - PAD_T - PAD_B}"/></g>')
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(xs, values, tips, axis_fmt):
    global W, H
    saved = (W, H)
    W, H = 960, 280
    try:
        return _line_chart(xs, values, tips, axis_fmt)
    finally:
        W, H = saved


def _line_chart(xs, values, tips, axis_fmt):
    ticks = _nice_ticks(min(min(values), 0), max(values))
    lo, hi = min(ticks), max(ticks)
    y = _scale(lo, hi, H - PAD_B, PAD_T)
    x = _scale(0, len(xs) - 1, PAD_L + 6, W - PAD_R - 6)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img">'] + _yaxis(lo, hi, y, ticks, axis_fmt)
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline class="line s1" points="{pts}"/>')
    for i, lab in enumerate(xs):
        if lab.endswith("-01"):
            parts.append(f'<text class="tick" x="{x(i):.1f}" y="{H - PAD_B + 20}" text-anchor="middle">{lab[:4]}</text>')
    step = (x(1) - x(0)) if len(xs) > 1 else 20
    for i, v in enumerate(values):
        parts.append(f'<g class="mark" data-tip="{html.escape(tips[i])}"><circle class="dot s1" cx="{x(i):.1f}" '
                     f'cy="{y(v):.1f}" r="4"/><rect class="hit" x="{x(i) - step / 2:.1f}" y="{PAD_T}" '
                     f'width="{step:.1f}" height="{H - PAD_T - PAD_B}"/></g>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--border:rgba(11,11,11,.10);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--border:rgba(255,255,255,.10);color-scheme:dark}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--border:rgba(255,255,255,.10);color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:17px;margin:0 0 4px}p{margin:0 0 12px;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}
.tile,.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
.tile .v{font-size:28px;font-weight:600}.tile .l{color:var(--ink2);font-size:13px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-bottom:16px}
@media (max-width:520px){.grid2{grid-template-columns:1fr}}
svg{width:100%;height:auto;display:block;margin-top:8px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.whisker{stroke:var(--ink2);stroke-width:1.5}
.line{fill:none;stroke:var(--s1);stroke-width:2}.dot{stroke:var(--surface);stroke-width:2;opacity:0}
.mark:hover .dot{opacity:1}.hit{fill:transparent}.mark:hover rect.s1,.mark:hover rect.s2{opacity:.8}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin-top:6px;flex-wrap:wrap}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
background:var(--c)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;max-width:280px}
footer{margin-top:24px;font-size:13px;color:var(--muted)}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.mark').forEach(m=>{
 m.addEventListener('mousemove',e=>{tip.textContent=m.dataset.tip;tip.style.display='block';
  const x=Math.min(e.clientX+12,window.innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+12)+'px';});
 m.addEventListener('mouseleave',()=>{tip.style.display='none';});
});
"""


def build(res: dict, monthly: pd.DataFrame) -> str:
    s = res["sample"]
    c, ca = res["correlations"], res["correlations_abnormal"]
    keys = ["prev", "same", "next"]
    groups = ["Day before (t-1)", "News day (t)", "Day after (t+1)"]
    corr = grouped_bars(
        groups,
        [("Raw return", "s1", [c["ret_" + k]["pearson_r"] for k in keys],
          [c["ret_" + k]["pearson_ci95"][0] for k in keys], [c["ret_" + k]["pearson_ci95"][1] for k in keys]),
         ("Minus market average", "s2", [ca["abn_" + k]["pearson_r"] for k in keys],
          [ca["abn_" + k]["pearson_ci95"][0] for k in keys], [ca["abn_" + k]["pearson_ci95"][1] for k in keys])],
        lambda v: f"r = {v:.3f}", lambda v: f"{v:.3f}",
    )
    b = res["buckets"]
    bars = grouped_bars(
        [f"{x['day_label']} ({x['n_days']:,})" for x in b],
        [(name, cls, [100 * x[f"mean_{col}"] for x in b],
          [100 * (x[f"mean_{col}"] - 1.96 * x[f"se_{col}"]) for x in b],
          [100 * (x[f"mean_{col}"] + 1.96 * x[f"se_{col}"]) for x in b])
         for name, cls, col in [("News day", "s1", "ret_same"), ("Day after", "s2", "ret_next")]],
        lambda v: f"{v:.3f}%", lambda v: f"{v:.2f}%",
    )
    line = line_chart(
        list(monthly["month"]), list(monthly["avg_compound"]),
        [f"{m}: mean compound {v:.3f}, {n:,} headlines" for m, v, n in
         zip(monthly["month"], monthly["avg_compound"], monthly["n_headlines"])],
        lambda v: f"{v:.2f}",
    )
    rows = []
    for kind, block in [("Raw", c), ("Minus market", ca)]:
        for k, g in zip(keys, groups):
            d = block[("ret_" if kind == "Raw" else "abn_") + k]
            rows.append(f"<tr><td>{g}, {kind.lower()}</td><td>{d['n']:,}</td><td>{d['pearson_r']:.4f}</td>"
                        f"<td>{d['pearson_ci95'][0]:.4f} to {d['pearson_ci95'][1]:.4f}</td><td>{_p(d['pearson_p'])}</td>"
                        f"<td>{d['spearman_rho']:.4f}</td></tr>")
    hr = res["hit_rate"]["ret_next"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stock Headline Sentiment</title><style>{CSS}</style></head>
<body><main>
<h1>Stock headline sentiment vs returns</h1>
<p>NASDAQ-100 headlines from FNSPID, scored with VADER, joined to daily adjusted closes.
{s['first_date']} to {s['last_date']}. Generated by <code>pipeline/dashboard.py</code>.</p>
<div class="tiles">
<div class="tile"><div class="v">{s['fact_rows']:,}</div><div class="l">ticker days with news and prices</div></div>
<div class="tile"><div class="v">{s['tickers']}</div><div class="l">tickers</div></div>
<div class="tile"><div class="v">{s['headlines_in_fact']:,}</div><div class="l">headlines in those days</div></div>
<div class="tile"><div class="v">{c['ret_next']['pearson_r']:.3f}</div><div class="l">Pearson r, sentiment vs next day return</div></div>
<div class="tile"><div class="v">{100 * hr['hit_rate']:.1f}%</div><div class="l">direction hit rate, next day ({hr['n']:,} non neutral days)</div></div>
</div>
<div class="grid2">
<div class="card"><h2>Correlation by return window</h2><p>Pearson r with 95% CI. Near zero means no linear link.</p>
<div class="legend"><span style="--c:var(--s1)">Raw return</span><span style="--c:var(--s2)">Minus market average</span></div>{corr}</div>
<div class="card"><h2>Mean return by day label</h2><p>Day label from the average VADER score of that day's headlines.</p>
<div class="legend"><span style="--c:var(--s1)">News day</span><span style="--c:var(--s2)">Day after</span></div>{bars}</div>
</div>
<div class="card"><h2>Average headline sentiment by month</h2><p>All tickers pooled. Hover a month for the count.</p>{line}</div>
<div class="card" style="margin-top:16px"><h2>Numbers</h2><div class="scroll"><table>
<tr><th>Window</th><th>n</th><th>Pearson r</th><th>95% CI</th><th>p</th><th>Spearman rho</th></tr>
{''.join(rows)}</table></div></div>
<footer>Data: FNSPID (CC BY-NC 4.0) via Hugging Face. Prices via benstaf/nasdaq_2013_2023. Correlation is not a trading signal.</footer>
</main><div id="tip"></div><script>{JS}</script></body></html>
"""


def run(sample: bool):
    src = ROOT / "outputs" / ("sample" if sample else "")
    res = json.loads((src / "results.json").read_text())
    monthly = pd.read_csv(src / "monthly.csv")
    out = ROOT / "docs" / ("dashboard_sample.html" if sample else "dashboard.html")
    out.write_text(build(res, monthly), encoding="utf-8")
    print(f"dashboard: wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    run(ap.parse_args(argv).sample)


if __name__ == "__main__":
    main()
