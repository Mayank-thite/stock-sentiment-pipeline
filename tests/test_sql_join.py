"""Run the real sql/ files on a tiny hand made dataset and check the joins."""

import pytest
from pyspark.sql import types as T

from pipeline.batch_sql import render, run_sql_files, split_statements
from pipeline.stream_score import add_scores

# Trading days: Thu 5th, Fri 6th, Mon 9th, Tue 10th of January 2023.
PRICES = [
    ("2023-01-05", "AAA", 100.0), ("2023-01-06", "AAA", 110.0),
    ("2023-01-09", "AAA", 99.0), ("2023-01-10", "AAA", 99.0),
    ("2023-01-05", "BBB", 50.0), ("2023-01-06", "BBB", 50.0),
    ("2023-01-09", "BBB", 55.0), ("2023-01-10", "BBB", 55.0),
]

HEADLINES = [
    # id, ticker, published_at, headline
    (1, "AAA", "2023-01-06 00:00:00", "AAA posts record profit, great quarter"),
    (2, "AAA", "2023-01-06 00:00:00", "  AAA posts record   profit, great quarter "),  # duplicate after cleaning
    (3, "AAA", "2023-01-07 00:00:00", "AAA faces terrible lawsuit"),  # Saturday -> Monday
    (4, "aaa", "2023-01-08 00:00:00", "AAA hit by fraud probe"),  # Sunday -> Monday, lower case ticker
    (5, "BBB", "2023-01-09 00:00:00", "BBB holds annual meeting"),
    (6, "BBB", "2023-01-09 00:00:00", ""),  # empty, dropped
    (7, "ZZZ", "2023-01-09 00:00:00", "ZZZ wins big contract"),  # no prices, not in fact
]


@pytest.fixture(scope="module")
def built(spark, tmp_path_factory):
    wh = tmp_path_factory.mktemp("wh")
    schema = T.StructType([
        T.StructField("headline_id", T.LongType()), T.StructField("ticker", T.StringType()),
        T.StructField("published_at", T.StringType()), T.StructField("headline", T.StringType()),
        T.StructField("publisher", T.StringType()),
    ])
    raw = spark.createDataFrame([(i, t, p, h, "") for i, t, p, h in HEADLINES], schema)
    add_scores(raw).write.parquet(str(wh / "headline_scores"))
    prices = spark.createDataFrame(
        [(d, t, c, c, c, c, 1000.0) for d, t, c in PRICES],
        "trade_date string, ticker string, open double, high double, low double, close double, volume double",
    )
    prices.write.parquet(str(wh / "stg_prices"))
    run_sql_files(spark, {"WAREHOUSE": wh.as_posix(), "THRESHOLD": 0.05})
    return spark


def test_split_and_render():
    sql = "-- comment\nSELECT '${X}';\n\nSELECT 2; -- trailing\n"
    assert split_statements(render(sql, {"X": "a"})) == ["SELECT 'a'", "SELECT 2"]


def test_cleaning_drops_duplicates_and_empties(built):
    ids = sorted(r.headline_id for r in built.table("sentiment.clean_headlines").collect())
    assert ids == [1, 3, 4, 5, 7]


def test_returns(built):
    r = {(x.ticker, str(x.trade_date)): x for x in built.table("sentiment.daily_returns").collect()}
    a = r[("AAA", "2023-01-06")]
    assert a.ret_same == pytest.approx(0.10)
    assert a.ret_next == pytest.approx(-0.10)
    assert r[("AAA", "2023-01-09")].ret_prev == pytest.approx(0.10)
    # two day window from close(t-1) to close(t+1): 100 -> 99
    assert a.ret_2d == pytest.approx(99 / 100 - 1)
    assert r[("AAA", "2023-01-05")].ret_2d is None
    assert r[("AAA", "2023-01-05")].ret_same is None
    assert r[("AAA", "2023-01-10")].ret_next is None
    # abnormal = own return minus the average across tickers that day
    assert a.abn_same == pytest.approx(0.10 - 0.05)


def test_weekend_headlines_map_to_monday(built):
    m = {x.headline_id: str(x.trade_date) for x in built.table("sentiment.headline_trading_day").collect()}
    assert m[1] == "2023-01-06"
    assert m[3] == "2023-01-09"
    assert m[4] == "2023-01-09"


def test_fact_table(built):
    rows = {(x.ticker, str(x.trade_date)): x for x in built.table("sentiment.fact_sentiment_daily").collect()}
    assert set(rows) == {("AAA", "2023-01-06"), ("AAA", "2023-01-09"), ("BBB", "2023-01-09")}
    mon = rows[("AAA", "2023-01-09")]
    assert mon.n_headlines == 2
    assert mon.n_negative == 2
    assert mon.day_label == "negative"
    assert mon.ret_same == pytest.approx(99 / 110 - 1)
    assert rows[("AAA", "2023-01-06")].day_label == "positive"
    assert rows[("BBB", "2023-01-09")].ret_same == pytest.approx(0.10)
