-- One row per ticker per trading day that had at least one headline,
-- joined to that ticker's returns. This is the fact table for analysis and Power BI.

DROP TABLE IF EXISTS sentiment.fact_sentiment_daily;
CREATE TABLE sentiment.fact_sentiment_daily STORED AS PARQUET AS
WITH daily AS (
    SELECT
        ticker,
        trade_date,
        count(*)                                                     AS n_headlines,
        avg(compound)                                                AS avg_compound,
        sum(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS n_positive,
        sum(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS n_negative,
        sum(CASE WHEN sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
    FROM sentiment.headline_trading_day
    GROUP BY ticker, trade_date
)
SELECT
    d.ticker,
    d.trade_date,
    d.n_headlines,
    d.avg_compound,
    d.n_positive,
    d.n_negative,
    d.n_neutral,
    CASE
        WHEN d.avg_compound >= ${THRESHOLD}  THEN 'positive'
        WHEN d.avg_compound <= -${THRESHOLD} THEN 'negative'
        ELSE 'neutral'
    END AS day_label,
    r.close,
    r.ret_prev,
    r.ret_same,
    r.ret_next,
    r.abn_prev,
    r.abn_same,
    r.abn_next
FROM daily d
JOIN sentiment.daily_returns r
  ON r.ticker = d.ticker
 AND r.trade_date = d.trade_date;
