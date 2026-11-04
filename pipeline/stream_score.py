"""Spark Structured Streaming job: read headlines, score them, write Parquet.

Source is either Kafka (topic `headlines`, or `headlines_sample`) or the local landing folder the
producer writes to. Both use trigger(availableNow=True), so the job processes
everything that is waiting and then stops. Drop that trigger for an always-on
job.

Usage:
  python -m pipeline.stream_score --source files [--sample]
  python -m pipeline.stream_score --source kafka [--sample]
"""
import argparse
import shutil

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline.config import KAFKA_BOOTSTRAP, get_paths, kafka_topic, spark_session

HEADLINE_SCHEMA = T.StructType(
    [
        T.StructField("headline_id", T.LongType()),
        T.StructField("ticker", T.StringType()),
        T.StructField("published_at", T.StringType()),
        T.StructField("headline", T.StringType()),
        T.StructField("publisher", T.StringType()),
    ]
)

SCORE_SCHEMA = T.StructType(
    [
        T.StructField("compound", T.DoubleType()),
        T.StructField("pos", T.DoubleType()),
        T.StructField("neu", T.DoubleType()),
        T.StructField("neg", T.DoubleType()),
    ]
)


def _score(text):
    from pipeline.sentiment import score_headline

    return score_headline(text)


score_udf = F.udf(_score, SCORE_SCHEMA)


def read_stream(spark: SparkSession, source: str, paths, bootstrap: str, topic: str = None) -> DataFrame:
    if source == "kafka":
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", bootstrap)
            .option("subscribe", topic or kafka_topic(False))
            .option("startingOffsets", "earliest")
            .load()
        )
        return raw.select(
            F.from_json(F.col("value").cast("string"), HEADLINE_SCHEMA).alias("m")
        ).select("m.*")
    return (
        spark.readStream.schema(HEADLINE_SCHEMA)
        .option("maxFilesPerTrigger", 4)
        .json(str(paths.landing))
    )


def add_scores(df: DataFrame) -> DataFrame:
    """Score each headline. Kept separate so tests can run it on a batch frame."""
    return (
        df.withColumn("s", score_udf(F.col("headline")))
        .select(
            "headline_id",
            "ticker",
            "published_at",
            "headline",
            "publisher",
            F.col("s.compound").alias("compound"),
            F.col("s.pos").alias("pos"),
            F.col("s.neu").alias("neu"),
            F.col("s.neg").alias("neg"),
        )
        .withColumn("scored_at", F.current_timestamp())
    )


def run(source: str, sample: bool, bootstrap: str = KAFKA_BOOTSTRAP, spark=None) -> int:
    paths = get_paths(sample)
    out = paths.warehouse / "headline_scores"
    # Fresh run each time so reruns are reproducible.
    for p in (out, paths.checkpoint):
        if p.exists():
            shutil.rmtree(p)
    spark = spark or spark_session("score-headlines", kafka=(source == "kafka"))
    query = (
        add_scores(read_stream(spark, source, paths, bootstrap, kafka_topic(sample)))
        .writeStream.format("parquet")
        .option("path", str(out))
        .option("checkpointLocation", str(paths.checkpoint))
        .outputMode("append")
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    n = spark.read.parquet(str(out)).count()
    print(f"stream_score: scored {n} headlines from {source} into {out}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["kafka", "files"], default="files")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = ap.parse_args(argv)
    run(args.source, args.sample, args.bootstrap)


if __name__ == "__main__":
    main()
