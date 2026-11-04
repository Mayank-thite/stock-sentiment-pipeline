"""Export a small star schema for Power BI.

  fact_sentiment_daily  one row per ticker per trading day with news
  dim_ticker            one row per ticker
  dim_date              one row per calendar day in range

Written as CSV and Parquet to outputs/powerbi/.

Usage:
  python -m pipeline.export_powerbi [--sample]
"""
import argparse

import pandas as pd

from pipeline.config import ROOT, get_paths


def build_star(fact: pd.DataFrame):
    fact = fact.copy()
    fact["trade_date"] = pd.to_datetime(fact["trade_date"])
    fact["date_key"] = fact["trade_date"].dt.strftime("%Y%m%d").astype(int)
    fact = fact[
        ["date_key", "ticker", "n_headlines", "avg_compound", "n_positive", "n_negative", "n_neutral",
         "day_label", "close", "ret_prev", "ret_same", "ret_next", "abn_prev", "abn_same", "abn_next"]
    ]

    dim_ticker = (
        fact.groupby("ticker")
        .agg(news_days=("date_key", "count"), headlines=("n_headlines", "sum"),
             first_date_key=("date_key", "min"), last_date_key=("date_key", "max"))
        .reset_index()
    )
    dim_ticker["index_membership"] = "NASDAQ-100 (FNSPID subset)"

    days = pd.date_range(
        pd.to_datetime(str(fact["date_key"].min())), pd.to_datetime(str(fact["date_key"].max())), freq="D"
    )
    dim_date = pd.DataFrame({"date": days})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date["month"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.strftime("%b")
    dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    dim_date["is_weekend"] = dim_date["date"].dt.dayofweek >= 5
    dim_date["date"] = dim_date["date"].dt.date
    dim_date = dim_date[["date_key", "date", "year", "quarter", "month", "month_name", "year_month",
                         "day_of_week", "is_weekend"]]
    return fact, dim_ticker, dim_date


def run(sample: bool):
    paths = get_paths(sample)
    fact = pd.read_parquet(paths.work / "fact_sentiment_daily.parquet")
    out = ROOT / "outputs" / "powerbi" / ("sample" if sample else "")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in zip(["fact_sentiment_daily", "dim_ticker", "dim_date"], build_star(fact)):
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.8g")
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"export_powerbi: {name} {len(df)} rows")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    run(ap.parse_args(argv).sample)


if __name__ == "__main__":
    main()
