"""Optional: fetch daily prices from a public API instead of the static file.

Source: the keyless Yahoo Finance chart endpoint
(query1.finance.yahoo.com/v8/finance/chart/<ticker>). It rate limits (HTTP 429)
without warning, so every request has retries with exponential backoff, and
each ticker's response is cached under data/cache/prices_api/. If a ticker
still fails, its rows come from the static Hugging Face file, so the output is
always complete.

Output: data/raw/prices_api.csv, same columns as the static file
(date, tic, open, high, low, close, volume). close is the adjusted close,
which matches how the static file stores it.

Usage:
  python -m pipeline.ingest_prices [--tickers AAPL MSFT] [--refresh] [--check]

--check compares daily returns from the API with the static file and writes
outputs/price_check.json.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import ROOT, setup_logging
from pipeline.ingest import load_prices

log = logging.getLogger(__name__)

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
CACHE = ROOT / "data" / "cache" / "prices_api"
STATIC = ROOT / "data" / "raw" / "prices_2019_2023.csv"
OUT = ROOT / "data" / "raw" / "prices_api.csv"
FALLBACK = ROOT / "data" / "raw" / "prices_api_fallback.json"
START = "2018-12-01"
END = "2024-01-31"
HEADERS = {"User-Agent": "Mozilla/5.0 (stock-sentiment-pipeline research script)"}


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_json(ticker: str, retries: int = 5, backoff: float = 2.0, session=None) -> dict:
    """GET one ticker with retries on 429, 5xx and network errors."""
    session = session or requests.Session()
    params = {"period1": _epoch(START), "period2": _epoch(END), "interval": "1d", "events": "div,splits"}
    last, tries = None, 0
    for attempt in range(retries):
        tries += 1
        try:
            r = session.get(URL.format(ticker=ticker), params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:  # network blip, try again
            last = str(e)
        time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"{ticker}: giving up after {tries} tries ({last})")


def parse_chart(ticker: str, payload: dict) -> pd.DataFrame:
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    # Timestamps are the market open in New York; convert to the exchange date.
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    dates = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "tic": ticker, "open": q["open"], "high": q["high"],
                       "low": q["low"], "close": adj, "volume": q["volume"]})
    return df.dropna(subset=["close"])


def get_ticker(ticker: str, refresh: bool, session) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{ticker}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        payload = fetch_json(ticker, session=session)
        cached.write_text(json.dumps(payload))
        time.sleep(0.5)  # be polite
    return parse_chart(ticker, payload)


def run(tickers=None, refresh=False):
    static = pd.read_csv(STATIC, usecols=["date", "tic", "open", "high", "low", "close", "volume"])
    tickers = tickers or sorted(static["tic"].unique())
    session = requests.Session()
    frames, fell_back = [], []
    for t in tickers:
        try:
            frames.append(get_ticker(t, refresh, session))
        except Exception as e:  # keep going, use the static rows for this one
            log.warning(f"{e}; using static rows")
            fell_back.append(t)
            frames.append(static[static["tic"] == t])
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= static["date"].min()) & (df["date"] <= static["date"].max())]
    df.to_csv(OUT, index=False)
    FALLBACK.write_text(json.dumps(fell_back))
    log.info(f"{len(df)} rows for {len(tickers)} tickers -> {OUT} (static fallback: {fell_back})")
    return df, fell_back


def check(api_csv: Path = OUT) -> dict:
    """How closely do API returns match the static file's returns?"""
    a, s = load_prices(api_csv), load_prices(STATIC)
    fell_back = json.loads(FALLBACK.read_text()) if FALLBACK.exists() else []
    a = a[~a["ticker"].isin(fell_back)]  # only compare rows that really came from the API
    for d in (a, s):
        d["ret"] = d.groupby("ticker")["close"].pct_change()
    m = a.merge(s, on=["ticker", "trade_date"], suffixes=("_api", "_static")).dropna(subset=["ret_api", "ret_static"])
    diff = (m["ret_api"] - m["ret_static"]).abs()
    out = {
        "matched_rows": int(len(m)),
        "tickers": int(m["ticker"].nunique()),
        "tickers_on_static_fallback": fell_back,
        "return_correlation": float(m["ret_api"].corr(m["ret_static"])),
        "median_abs_diff": float(diff.median()),
        "share_abs_diff_below_1bp": float((diff < 1e-4).mean()),
    }
    path = ROOT / "outputs" / "price_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("price check %s written to %s", out, path)
    return out


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    run(args.tickers, args.refresh)
    if args.check:
        check()


if __name__ == "__main__":
    main()
