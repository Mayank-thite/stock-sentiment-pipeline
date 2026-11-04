"""Run the whole pipeline end to end.

  python -m pipeline.run --local            full data, no Docker (file stream source)
  python -m pipeline.run --local --sample   small sample in data/sample, about a minute
  python -m pipeline.run --kafka            full data through Kafka (docker compose up -d first)
  add --prices api to use prices fetched by pipeline.ingest_prices

Steps: producer -> Spark streaming scoring -> HiveQL cleaning and joins ->
analysis -> charts -> Power BI export -> HTML dashboard.
"""
import argparse
import subprocess
import sys
import time

from pipeline import analysis, batch_sql, charts, dashboard, export_powerbi, producer, stream_score
from pipeline.config import KAFKA_BOOTSTRAP, ROOT, get_paths, spark_session


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="no Kafka; stream through a local folder (default)")
    mode.add_argument("--kafka", action="store_true", help="stream through Kafka")
    ap.add_argument("--sample", action="store_true", help="use the small sample in data/sample")
    ap.add_argument("--prices", choices=["static", "api"], default="static")
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = ap.parse_args(argv)

    t0 = time.time()
    paths = get_paths(args.sample)
    if not args.sample and not (paths.news.exists() and paths.prices.exists()):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True)
    prices = None
    if args.prices == "api":
        prices = ROOT / "data" / "raw" / "prices_api.csv"
        if not prices.exists():
            from pipeline import ingest_prices

            ingest_prices.run()

    sink = "kafka" if args.kafka else "files"
    argv_common = ["--sample"] if args.sample else []
    producer.main(["--sink", sink, "--bootstrap", args.bootstrap] + argv_common)

    spark = spark_session("stock-sentiment-pipeline", kafka=args.kafka, metastore=paths.work / "metastore")
    stream_score.run(sink, args.sample, args.bootstrap, spark=spark)
    batch_sql.run(args.sample, prices, spark=spark)
    spark.stop()

    analysis.run(args.sample)
    charts.run(args.sample)
    export_powerbi.run(args.sample)
    dashboard.run(args.sample)
    print(f"run: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
