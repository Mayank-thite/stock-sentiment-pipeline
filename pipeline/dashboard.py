"""Build docs/dashboard.html: one self contained page, inline SVG, no external JS.

Styled to match my portfolio site: dark by default, light when the browser
asks for it, Geist for text and Geist Mono for labels and numbers.

Usage:
  python -m pipeline.dashboard [--sample]
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math

import pandas as pd

from pipeline.config import ROOT, setup_logging

log = logging.getLogger(__name__)

PAD_L, PAD_R, PAD_T, PAD_B = 52, 12, 22, 46
WINDOWS = [("prev", "t-1"), ("same", "t"), ("next", "t+1"), ("2d", "t and t+1")]
SOURCE_URL = "https://huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"
PRICES_URL = "https://huggingface.co/datasets/benstaf/nasdaq_2013_2023"
REPO_URL = "https://github.com/mayank-thite/stock-sentiment-pipeline"


def _scale(lo: float, hi: float, a: float, b: float):
    return lambda v: a + (v - lo) / (hi - lo) * (b - a) if hi != lo else (a + b) / 2


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values that cover lo to hi."""
    span = (hi - lo) or abs(hi) or 1.0
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    ticks = [round(t, 10)]
    while t < hi - step * 1e-9:
        t += step
        ticks.append(round(t, 10))
    return ticks


def _yaxis(w: int, y, ticks: list[float], fmt) -> list[str]:
    out = []
    for t in ticks:
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{w - PAD_R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt(t) if t else "0"}</text>')
    return out


def grouped_bars(groups: list[str], series: list[tuple], value_fmt, axis_fmt, label: str,
                 w: int = 520, h: int = 300) -> str:
    """Bars from zero with CI whiskers and a value label on each bar.

    series: list of (name, css_class, values, lows, highs).
    """
    vals = [v for s in series for v in list(s[3]) + list(s[4])] + [0]
    pad = 0.12 * (max(vals) - min(vals))  # room for the value labels above and below the whiskers
    ticks = nice_ticks(min(vals) - (pad if min(vals) < 0 else 0), max(vals) + pad)
    y = _scale(min(ticks), max(ticks), h - PAD_B, PAD_T)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}">']
    parts += _yaxis(w, y, ticks, axis_fmt)
    gw = (w - PAD_L - PAD_R) / len(groups)
    bw = min(40, gw / (len(series) + 1))
    for gi, g in enumerate(groups):
        cx = PAD_L + gw * (gi + 0.5)
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{h - PAD_B + 20}" text-anchor="middle">{html.escape(g)}</text>')
        for si, (name, cls, values, los, his) in enumerate(series):
            x = cx + (si - len(series) / 2) * (bw + 3)
            v = values[gi]
            top, height = min(y(0), y(v)), max(abs(y(v) - y(0)), 1)
            tip = f"{name}, {g}: {value_fmt(v)} (95% CI {value_fmt(los[gi])} to {value_fmt(his[gi])})"
            ly = y(his[gi]) - 6 if v >= 0 else y(los[gi]) + 14
            parts.append(
                f'<g class="mark" tabindex="0" data-tip="{html.escape(tip)}" aria-label="{html.escape(tip)}">'
                f'<rect class="{cls}" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{height:.1f}" rx="2"/>'
                f'<line class="whisker" x1="{x + bw / 2:.1f}" x2="{x + bw / 2:.1f}" '
                f'y1="{y(los[gi]):.1f}" y2="{y(his[gi]):.1f}"/>'
                f'<text class="val" x="{x + bw / 2:.1f}" y="{ly:.1f}" text-anchor="middle">{value_fmt(v, short=True)}</text>'
                f"</g>"
            )
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{w - PAD_R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(months: list[str], values: list[float], counts: list[int], w: int = 1000, h: int = 260) -> str:
    ticks = nice_ticks(min(min(values), 0), max(values))
    y = _scale(min(ticks), max(ticks), h - PAD_B, PAD_T)
    x = _scale(0, len(months) - 1, PAD_L + 6, w - PAD_R - 6)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Average headline sentiment by month">']
    parts += _yaxis(w, y, ticks, lambda v: f"{v:.2f}")
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{w - PAD_R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline class="line" points="{pts}"/>')
    for i, m in enumerate(months):
        if m.endswith("-01"):
            parts.append(f'<text class="tick" x="{x(i):.1f}" y="{h - PAD_B + 20}" text-anchor="middle">{m[:4]}</text>')
    step = x(1) - x(0) if len(months) > 1 else 20
    for i, (m, v, n) in enumerate(zip(months, values, counts)):
        tip = html.escape(f"{m}: mean compound {v:.3f}, {n:,} headlines")
        parts.append(f'<g class="mark" data-tip="{tip}"><circle class="dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5"/>'
                     f'<rect class="hit" x="{x(i) - step / 2:.1f}" y="{PAD_T}" width="{step:.1f}" '
                     f'height="{h - PAD_T - PAD_B}"/></g>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#04050A;--ink:#ECEEF3;--text-2:#A4ABBD;--body:#C9CEDA;--line:rgba(220,226,240,.12);--panel:#0B0D15;
--accent:#FFC77A;--cool:#9CCBFF;color-scheme:dark}
@media (prefers-color-scheme:light){:root{--bg:#FAFAFC;--ink:#0B0D15;--text-2:#4A5163;--body:#2A2F3D;
--line:rgba(11,13,21,.12);--panel:#FFFFFF;--accent:#A8650C;--cool:#1F63B0;color-scheme:light}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--body);font:300 16px/1.6 "Geist",system-ui,-apple-system,"Segoe UI",sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:1080px;margin:0 auto;padding:56px 16px 64px}
.mono,.eyebrow,.tile .l,.tick,.val,th,.key,footer{font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:0 0 16px}
h1{font-weight:400;font-size:clamp(26px,4.2vw,40px);line-height:1.2;color:var(--ink);margin:0 0 16px;letter-spacing:-.01em;max-width:24ch}
h2{font-weight:500;font-size:17px;color:var(--ink);margin:0 0 4px}
p{margin:0 0 12px}.lede{max-width:68ch;color:var(--body)}
a{color:var(--ink);text-underline-offset:3px;text-decoration-color:var(--line)}a:hover{text-decoration-color:var(--accent)}
a:focus-visible,.mark:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:40px 0}
.tile,.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px}
.tile .v{font-family:"Geist Mono",ui-monospace,monospace;font-size:30px;font-weight:400;color:var(--ink);line-height:1.2}
.tile .v.hl{color:var(--accent)}
.tile .l{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-2);margin-top:8px}
.tile .s{font-size:13px;color:var(--text-2);margin-top:4px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:12px}
.card p{color:var(--text-2);font-size:14px}
.key{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:8px 0 0}
.key .a{color:var(--accent)}.key .c{color:var(--cool)}
svg{width:100%;height:auto;display:block;margin-top:8px;overflow:visible}
.grid{stroke:var(--line);stroke-width:1}.axis{stroke:var(--text-2);stroke-width:1;opacity:.6}
.tick{fill:var(--text-2);font-size:11px}.val{fill:var(--body);font-size:10.5px}
.s1{fill:var(--accent)}.s2{fill:var(--cool)}.whisker{stroke:var(--ink);stroke-width:1.2;opacity:.75}
.line{fill:none;stroke:var(--accent);stroke-width:2}.dot{fill:var(--accent);opacity:0}
.mark:hover .dot,.mark:focus .dot{opacity:1}.hit{fill:transparent}.mark{outline:none}
.mark:hover rect.s1,.mark:hover rect.s2,.mark:focus rect.s1,.mark:focus rect.s2{opacity:.8}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;
color:var(--text-2);font-weight:400}td{color:var(--body)}.scroll{overflow-x:auto}
.notes{margin-top:12px}.notes ul{margin:8px 0 0;padding-left:18px}.notes li{margin:6px 0;color:var(--body);font-size:14px}
#tip{position:fixed;pointer-events:none;background:var(--panel);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:6px 10px;font:12px/1.4 "Geist Mono",ui-monospace,monospace;display:none;max-width:280px;z-index:2}
footer{margin-top:40px;font-size:12px;color:var(--text-2);line-height:1.8}
@media (max-width:860px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2{grid-template-columns:1fr}}
.narrow{display:none}
@media (max-width:560px){.wide{display:none}.narrow{display:block}.tick{font-size:15px}.val{font-size:14px}}
@media (max-width:420px){main{padding-top:36px}.tile{padding:16px}.tile .v{font-size:24px}.card{padding:16px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
const tip=document.getElementById('tip');
function show(m,x,y){tip.textContent=m.dataset.tip;tip.style.display='block';
 tip.style.left=Math.max(8,Math.min(x+12,window.innerWidth-tip.offsetWidth-8))+'px';tip.style.top=(y+12)+'px';}
document.querySelectorAll('.mark').forEach(m=>{
 m.addEventListener('mousemove',e=>show(m,e.clientX,e.clientY));
 m.addEventListener('mouseleave',()=>{tip.style.display='none';});
 m.addEventListener('focus',()=>{const r=m.getBoundingClientRect();show(m,r.left,r.top);});
 m.addEventListener('blur',()=>{tip.style.display='none';});
});
"""


def _r(v: float, short: bool = False) -> str:
    return f"{v:.3f}" if short else f"r = {v:.3f}"


def _pct(v: float, short: bool = False) -> str:
    return f"{v:.2f}%"


def build(res: dict, monthly: pd.DataFrame) -> str:
    s = res["sample"]
    win = res["windows"]
    same, nxt = win["same"]["abnormal"], win["next"]["abnormal"]
    gap = res["pos_vs_neg"]["ret_same"]
    fm = win["same"]["fama_macbeth"]
    holds = same["ci95_date"][0] > 0 and nxt["ci95_date"][0] <= 0 <= nxt["ci95_date"][1]
    if holds:
        title = "Headline tone moves with the same day return, and tells you nothing about the next day"
        lede = ("On the news day, a more positive headline lines up with a higher return than the rest of the market. "
                "The day after, the link is gone.")
    else:
        title = "Headline tone and daily returns, by return window"
        lede = "On this data the news day link is not clearly different from zero."
    fm_note = ""
    if "mean_slope" in fm:
        fm_note = (f"<li>Fama-MacBeth daily regressions give a mean news day slope of {100 * fm['mean_slope']:.3f} "
                   f"pp of return per unit of sentiment, Newey-West t = {fm['t_newey_west']:.2f}, over "
                   f"{fm['n_dates']:,} dates.</li>")

    corr = grouped_bars(
        [lab for _, lab in WINDOWS],
        [(name, cls, [win[k][kind]["r"] for k, _ in WINDOWS],
          [win[k][kind]["ci95_date"][0] for k, _ in WINDOWS], [win[k][kind]["ci95_date"][1] for k, _ in WINDOWS])
         for name, cls, kind in [("Minus market", "s1", "abnormal"), ("Raw return", "s2", "raw")]],
        _r, lambda v: f"{v:.2f}", "Correlation of daily sentiment with returns by window",
    )
    b = res["buckets"]
    bars = grouped_bars(
        [x["day_label"] for x in b],
        [(name, cls, [100 * x[f"mean_{col}"] for x in b],
          [100 * (x[f"mean_{col}"] - 1.96 * x[f"se_date_{col}"]) for x in b],
          [100 * (x[f"mean_{col}"] + 1.96 * x[f"se_date_{col}"]) for x in b])
         for name, cls, col in [("News day", "s1", "ret_same"), ("Day after", "s2", "ret_next")]],
        _pct, lambda v: f"{v:.2f}%", "Mean return by day sentiment label",
    )
    m_args = (list(monthly["month"]), list(monthly["avg_compound"]), list(monthly["n_headlines"]))
    # A wide version for desktop and a narrow one for phones, so the axis labels stay readable on both.
    line = (f'<div class="wide">{line_chart(*m_args)}</div>'
            f'<div class="narrow">{line_chart(*m_args, w=420, h=240)}</div>')

    rows = []
    for k, lab in WINDOWS:
        for kind, kname in [("abnormal", "minus market"), ("raw", "raw")]:
            d = win[k][kind]
            rows.append(f"<tr><td>{lab}, {kname}</td><td>{d['n']:,}</td><td>{d['r']:.4f}</td>"
                        f"<td>{d['ci95_date'][0]:.4f} to {d['ci95_date'][1]:.4f}</td>"
                        f"<td>{d['t_iid']:.2f}</td><td>{d['t_date']:.2f}</td></tr>")
    pt = res["per_ticker"]["ret_next"]
    pt_note = ""
    if pt["n_tickers"]:
        pt_note = (f"<li>Per ticker, {pt['n_p_below_0_05']} of {pt['n_tickers']} tickers have a next day p below 0.05. "
                   f"Chance alone gives about {pt['expected_by_chance']:.0f}. After a Holm correction, "
                   f"{pt['n_holm_below_0_05']} remain.</li>")
    lm = res.get("loughran_mcdonald")
    lm_note = ""
    if lm:
        lms, lmn = lm["windows"]["same"]["abnormal"], lm["windows"]["next"]["abnormal"]
        lm_note = (f"<li>Swapping VADER for the Loughran-McDonald finance word lists gives the same shape: "
                   f"r = {lms['r']:.3f} on the news day (t = {lms['t_date']:.2f}) and {lmn['r']:.3f} the day after "
                   f"(t = {lmn['t_date']:.2f}), both net of the market.</li>")
    sample_note = ""
    if res.get("run", {}).get("dataset") == "sample":
        sample_note = '<p class="eyebrow" style="color:var(--accent)">Sample run, 8 tickers, 2023 only</p>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Headline Sentiment vs Returns</title>
<meta name="description" content="NASDAQ-100 headline sentiment against daily returns, 2019 to 2023.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><main>
{sample_note}<p class="eyebrow">Stock sentiment pipeline &middot; NASDAQ-100 &middot; {s['first_date']} to {s['last_date']}</p>
<h1>{title}</h1>
<p class="lede">I scored {s['raw_headlines']:,} NASDAQ-100 headlines with VADER and joined them to daily adjusted closes.
{lede} All errors below are clustered by date, because stocks on the same day move together.</p>

<div class="tiles">
<div class="tile"><div class="v hl">{same['r']:.3f}</div><div class="l">r, news day, net of market</div>
<div class="s">95% CI {same['ci95_date'][0]:.3f} to {same['ci95_date'][1]:.3f}</div></div>
<div class="tile"><div class="v">{nxt['r']:.3f}</div><div class="l">r, day after, net of market</div>
<div class="s">95% CI {nxt['ci95_date'][0]:.3f} to {nxt['ci95_date'][1]:.3f}</div></div>
<div class="tile"><div class="v">{100 * gap['difference']:.2f} pp</div><div class="l">positive minus negative days</div>
<div class="s">news day return, t = {gap['t_date_clustered']:.2f}</div></div>
<div class="tile"><div class="v">{s['fact_rows']:,}</div><div class="l">ticker days with news</div>
<div class="s">{s['tickers']} tickers, {s['trading_dates']:,} trading days</div></div>
</div>

<div class="grid2">
<div class="card"><h2>Correlation by return window</h2>
<p>Pearson r of daily sentiment with return. Whiskers are 95% CIs clustered by date.</p>
<div class="key"><span class="a">&#9632; minus market</span><span class="c">&#9632; raw return</span></div>{corr}</div>
<div class="card"><h2>Mean return by day label</h2>
<p>Label from the average VADER score of that day's headlines. 95% CIs clustered by date.</p>
<div class="key"><span class="a">&#9632; news day</span><span class="c">&#9632; day after</span></div>{bars}</div>
</div>

<div class="card"><h2>Average headline sentiment by month</h2>
<p>All tickers pooled. Above zero means headlines lean positive. Hover a month for the count.</p>{line}</div>

<div class="card" style="margin-top:12px"><h2>Numbers</h2>
<p>t naive treats every ticker day as independent. t clustered treats each date as one cluster.</p>
<div class="scroll"><table>
<tr><th>Window</th><th>n</th><th>r</th><th>95% CI, clustered</th><th>t naive</th><th>t clustered</th></tr>
{''.join(rows)}</table></div></div>

<div class="card notes"><h2>Caveats</h2><ul>
<li>{100 * s['share_timestamp_midnight']:.2f}% of timestamps are exactly midnight, so I only know the date.
A headline dated t may have come out before the open or after the close. That is why "t and t+1" is shown too.</li>
<li>The same day link does not mean news moves prices. Many headlines describe a move that already happened.</li>
{fm_note}
{pt_note}
{lm_note}
<li>Not a trading signal. No costs, no out of sample test.</li>
</ul></div>

<footer>Data as of {s['last_date']}, the last trading day in the dataset. Data: <a href="{SOURCE_URL}">FNSPID NASDAQ-100 subset</a> (CC BY-NC 4.0) and
<a href="{PRICES_URL}">benstaf/nasdaq_2013_2023</a> prices, via Hugging Face.
Built by <code>pipeline/dashboard.py</code> in <a href="{REPO_URL}">stock-sentiment-pipeline</a>.</footer>
</main><div id="tip" role="status"></div><script>{JS}</script></body></html>
"""


def run(sample: bool) -> None:
    src = ROOT / "outputs" / ("sample" if sample else "")
    res = json.loads((src / "results.json").read_text())
    monthly = pd.read_csv(src / "monthly.csv")
    out = ROOT / "docs" / ("dashboard_sample.html" if sample else "dashboard.html")
    out.write_text(build(res, monthly), encoding="utf-8")
    log.info("wrote %s", out)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    run(ap.parse_args(argv).sample)


if __name__ == "__main__":
    main()
