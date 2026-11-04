-- Daily simple returns per ticker, plus the same return minus the equal
-- weighted average of all tickers that day ("abnormal" return).
--   ret_prev = close(t-1) / close(t-2) - 1   (the day before the news day)
--   ret_same = close(t)   / close(t-1) - 1   (the news day)
--   ret_next = close(t+1) / close(t)   - 1   (the day after)
--   ret_2d   = close(t+1) / close(t-1) - 1   (news day and day after together)
-- The timestamps have dates but no times, so a headline dated t may have come
-- out before the open (move on t) or after the close (move on t+1). ret_2d
-- covers both cases.

DROP TABLE IF EXISTS sentiment.daily_returns;
CREATE TABLE sentiment.daily_returns STORED AS PARQUET AS
WITH p AS (
    SELECT DISTINCT ticker, to_date(trade_date) AS trade_date, close
    FROM sentiment.raw_prices
    WHERE close IS NOT NULL AND close > 0
),
lagged AS (
    SELECT
        ticker,
        trade_date,
        close,
        lag(close, 1)  OVER w AS close_lag1,
        lag(close, 2)  OVER w AS close_lag2,
        lead(close, 1) OVER w AS close_lead1
    FROM p
    WINDOW w AS (PARTITION BY ticker ORDER BY trade_date)
),
rets AS (
    SELECT
        ticker,
        trade_date,
        close,
        close_lag1 / close_lag2 - 1  AS ret_prev,
        close / close_lag1 - 1       AS ret_same,
        close_lead1 / close - 1      AS ret_next,
        close_lead1 / close_lag1 - 1 AS ret_2d
    FROM lagged
)
SELECT
    r.*,
    r.ret_prev - avg(r.ret_prev) OVER (PARTITION BY r.trade_date) AS abn_prev,
    r.ret_same - avg(r.ret_same) OVER (PARTITION BY r.trade_date) AS abn_same,
    r.ret_next - avg(r.ret_next) OVER (PARTITION BY r.trade_date) AS abn_next,
    r.ret_2d   - avg(r.ret_2d)   OVER (PARTITION BY r.trade_date) AS abn_2d
FROM rets r;
