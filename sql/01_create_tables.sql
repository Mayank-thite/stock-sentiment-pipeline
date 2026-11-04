-- HiveQL DDL. Runs as is in Hive, or in Spark SQL with Hive support.
-- ${WAREHOUSE} is replaced by the runner (pipeline/batch_sql.py).

CREATE DATABASE IF NOT EXISTS sentiment;

-- Scored headlines written by the streaming job.
DROP TABLE IF EXISTS sentiment.raw_headline_scores;
CREATE EXTERNAL TABLE sentiment.raw_headline_scores (
    headline_id   BIGINT,
    ticker        STRING,
    published_at  STRING,
    headline      STRING,
    publisher     STRING,
    compound      DOUBLE,
    pos           DOUBLE,
    neu           DOUBLE,
    neg           DOUBLE,
    scored_at     TIMESTAMP
)
STORED AS PARQUET
LOCATION '${WAREHOUSE}/headline_scores';

-- Daily prices, staged as Parquet by the runner. close is split and dividend adjusted.
DROP TABLE IF EXISTS sentiment.raw_prices;
CREATE EXTERNAL TABLE sentiment.raw_prices (
    trade_date  STRING,
    ticker      STRING,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE
)
STORED AS PARQUET
LOCATION '${WAREHOUSE}/stg_prices';
