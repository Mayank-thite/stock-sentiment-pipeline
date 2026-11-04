"""Load the raw news and price files into one standard shape.

News columns:   headline_id, ticker, published_at, headline, publisher
Price columns:  trade_date, ticker, open, high, low, close, volume
"""
from pathlib import Path

import pandas as pd

NEWS_COLUMNS = ["headline_id", "ticker", "published_at", "headline", "publisher"]
PRICE_COLUMNS = ["trade_date", "ticker", "open", "high", "low", "close", "volume"]


def load_news(path: Path) -> pd.DataFrame:
    """Read FNSPID style news (parquet) or the sample CSV."""
    path = Path(path)
    if path.suffix == ".parquet":
        raw = pd.read_parquet(
            path, columns=["Date", "Article_title", "Stock_symbol", "Publisher"]
        )
    else:
        raw = pd.read_csv(path)
    if "Article_title" in raw.columns:
        raw = raw.rename(
            columns={
                "Date": "published_at",
                "Article_title": "headline",
                "Stock_symbol": "ticker",
                "Publisher": "publisher",
            }
        )
    raw["published_at"] = raw["published_at"].astype(str).str.replace(" UTC", "", regex=False)
    raw["publisher"] = raw["publisher"].fillna("").astype(str)
    raw = raw.sort_values(["published_at", "ticker"], kind="stable").reset_index(drop=True)
    if "headline_id" not in raw.columns:
        raw["headline_id"] = range(1, len(raw) + 1)
    return raw[NEWS_COLUMNS]


def load_prices(path: Path) -> pd.DataFrame:
    """Read the FinRL style price CSV, the API cache CSV or the sample CSV."""
    raw = pd.read_csv(path)
    raw = raw.rename(columns={"date": "trade_date", "tic": "ticker"})
    out = raw[PRICE_COLUMNS].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
