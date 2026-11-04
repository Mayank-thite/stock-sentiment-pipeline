"""Replay historical headlines as a stream.

Two sinks:
  kafka  sends one JSON message per headline to the `headlines` topic
         (`headlines_sample` with --sample).
  files  writes JSON lines micro-batches into a landing folder. Spark reads that
         folder as a streaming file source, so the pipeline runs without Docker.

Usage:
  python -m pipeline.producer --sink kafka [--sample] [--rate 500]
  python -m pipeline.producer --sink files [--sample]
"""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from pipeline.config import KAFKA_BOOTSTRAP, get_paths, kafka_topic, setup_logging
from pipeline.ingest import load_news

log = logging.getLogger(__name__)


def to_messages(df: pd.DataFrame):
    for row in df.itertuples(index=False):
        yield {
            "headline_id": int(row.headline_id),
            "ticker": str(row.ticker),
            "published_at": str(row.published_at),
            "headline": str(row.headline),
            "publisher": str(row.publisher),
        }


def replay_to_files(df: pd.DataFrame, landing: Path, batch_size: int = 5000) -> int:
    """Write headlines in publish order as numbered JSON lines files."""
    if landing.exists():
        shutil.rmtree(landing)
    landing.mkdir(parents=True)
    batch, n_files, n = [], 0, 0
    for msg in to_messages(df):
        batch.append(json.dumps(msg))
        n += 1
        if len(batch) == batch_size:
            _write_batch(landing, n_files, batch)
            batch, n_files = [], n_files + 1
    if batch:
        _write_batch(landing, n_files, batch)
    return n


def _write_batch(landing: Path, idx: int, lines) -> None:
    tmp = landing / f".part-{idx:05d}.json.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.rename(landing / f"part-{idx:05d}.json")  # atomic, so Spark never reads half a file


def replay_to_kafka(df: pd.DataFrame, bootstrap: str, topic: str, rate: float) -> int:
    """Send headlines to Kafka. rate is messages per second, 0 means no limit."""
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=20,
    )
    n = 0
    start = time.time()
    for msg in to_messages(df):
        producer.send(topic, key=msg["ticker"], value=msg)
        n += 1
        if rate and n % 100 == 0:
            ahead = n / rate - (time.time() - start)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush()
    producer.close()
    return n


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sink", choices=["kafka", "files"], default="files")
    ap.add_argument("--sample", action="store_true", help="use data/sample instead of data/raw")
    ap.add_argument("--rate", type=float, default=0, help="Kafka messages per second (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0, help="only send the first N headlines")
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = ap.parse_args(argv)

    paths = get_paths(args.sample)
    df = load_news(paths.news)
    if args.limit:
        df = df.head(args.limit)
    if args.sink == "kafka":
        topic = kafka_topic(args.sample)
        n = replay_to_kafka(df, args.bootstrap, topic, args.rate)
        log.info(f"sent {n} headlines to Kafka topic '{topic}' at {args.bootstrap}")
    else:
        n = replay_to_files(df, paths.landing)
        log.info(f"wrote {n} headlines to {paths.landing}")
    return n


if __name__ == "__main__":
    main()
