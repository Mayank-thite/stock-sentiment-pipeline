"""Run the HiveQL files in sql/ with Spark SQL, in file name order.

Stages the price CSV as Parquet first, then runs DDL, cleaning, returns,
the headline to trading day mapping, and the daily fact table. Writes the
fact table to <work>/fact_sentiment_daily.parquet for the analysis step.

Usage:
  python -m pipeline.batch_sql [--sample]
"""
import argparse
import re
from pathlib import Path

from pipeline.config import LABEL_THRESHOLD, ROOT, get_paths, spark_session
from pipeline.ingest import load_prices

SQL_DIR = ROOT / "sql"


def split_statements(text: str):
    """Split a SQL file into statements. Drops -- comments and blank parts."""
    no_comments = "\n".join(re.sub(r"--.*$", "", line) for line in text.splitlines())
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def render(text: str, variables: dict) -> str:
    for key, value in variables.items():
        text = text.replace("${" + key + "}", str(value))
    return text


def stage_prices(spark, prices_csv: Path, warehouse: Path) -> int:
    pdf = load_prices(prices_csv)
    spark.createDataFrame(pdf).write.mode("overwrite").parquet(str(warehouse / "stg_prices"))
    return len(pdf)


def run_sql_files(spark, variables: dict, sql_dir: Path = SQL_DIR) -> None:
    for f in sorted(sql_dir.glob("*.sql")):
        for stmt in split_statements(render(f.read_text(), variables)):
            spark.sql(stmt)
        print(f"batch_sql: ran {f.name}")


def run(sample: bool, prices_csv: Path = None, spark=None):
    paths = get_paths(sample)
    spark = spark or spark_session("hive-sql", metastore=paths.work / "metastore")
    n_prices = stage_prices(spark, Path(prices_csv or paths.prices), paths.warehouse)
    print(f"batch_sql: staged {n_prices} price rows")
    run_sql_files(spark, {"WAREHOUSE": paths.warehouse.as_posix(), "THRESHOLD": LABEL_THRESHOLD})
    fact = spark.table("sentiment.fact_sentiment_daily").orderBy("ticker", "trade_date")
    out = paths.work / "fact_sentiment_daily.parquet"
    fact.toPandas().to_parquet(out, index=False)
    headlines = spark.table("sentiment.headline_trading_day").toPandas()
    headlines.to_parquet(paths.work / "headline_trading_day.parquet", index=False)
    counts = {
        "raw_headlines": spark.table("sentiment.raw_headline_scores").count(),
        "clean_headlines": spark.table("sentiment.clean_headlines").count(),
        "headlines_matched_to_trading_day": len(headlines),
        "fact_rows": fact.count(),
    }
    print(f"batch_sql: {counts}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--prices", help="price CSV to use instead of the default")
    args = ap.parse_args(argv)
    run(args.sample, args.prices)


if __name__ == "__main__":
    main()
