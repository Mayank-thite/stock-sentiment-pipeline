"""Paths and settings shared by every step."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KAFKA_TOPIC = "headlines"
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_SPARK_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"

# A headline counts as positive or negative when VADER compound is past this.
LABEL_THRESHOLD = 0.05


@dataclass(frozen=True)
class Paths:
    news: Path
    prices: Path
    work: Path

    @property
    def landing(self) -> Path:
        """Folder the local producer writes JSON micro-batches into."""
        return self.work / "stream" / "landing"

    @property
    def checkpoint(self) -> Path:
        return self.work / "stream" / "checkpoint"

    @property
    def warehouse(self) -> Path:
        return self.work / "warehouse"


def get_paths(sample: bool) -> Paths:
    if sample:
        return Paths(
            news=ROOT / "data" / "sample" / "headlines_sample.csv",
            prices=ROOT / "data" / "sample" / "prices_sample.csv",
            work=ROOT / "data" / "work_sample",
        )
    return Paths(
        news=ROOT / "data" / "raw" / "news.parquet",
        prices=ROOT / "data" / "raw" / "prices_2019_2023.csv",
        work=ROOT / "data" / "work",
    )


def spark_session(app_name: str, kafka: bool = False, metastore: Path = None):
    """Local Spark session. Workers use the same Python as the driver.

    Hive support (a local Derby metastore) is always on so the HiveQL DDL in
    sql/ runs as written. kafka=True pulls the Spark Kafka connector.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
    )
    metastore = Path(metastore or ROOT / "data" / "metastore")
    metastore.mkdir(parents=True, exist_ok=True)
    # Managed Hive tables live next to their metastore, so sample and full runs never collide.
    builder = builder.config("spark.sql.warehouse.dir", str(metastore.parent / "spark-warehouse"))
    os.environ.setdefault("DERBY_SYSTEM_HOME", str(metastore))
    builder = builder.config(
        "spark.hadoop.javax.jdo.option.ConnectionURL",
        f"jdbc:derby:;databaseName={metastore / 'metastore_db'};create=true",
    ).config("spark.driver.extraJavaOptions", f"-Dderby.system.home={metastore}")
    builder = builder.enableHiveSupport()
    if kafka:
        builder = builder.config("spark.jars.packages", KAFKA_SPARK_PACKAGE)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
