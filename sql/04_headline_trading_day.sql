-- Map each headline to the first trading day on or after its publish date.
-- Most FNSPID timestamps are midnight UTC with no time of day, so a headline
-- dated Saturday lands on Monday, and a weekday headline lands on that day.

DROP TABLE IF EXISTS sentiment.headline_trading_day;
CREATE TABLE sentiment.headline_trading_day STORED AS PARQUET AS
WITH calendar AS (
    SELECT DISTINCT trade_date FROM sentiment.daily_returns
)
SELECT
    h.headline_id,
    h.ticker,
    h.pub_date,
    h.headline,
    min(c.trade_date) AS trade_date,
    h.compound,
    h.sentiment_label
FROM sentiment.clean_headlines h
JOIN calendar c
  ON c.trade_date >= h.pub_date
 AND c.trade_date <  date_add(h.pub_date, 7)
GROUP BY h.headline_id, h.ticker, h.pub_date, h.headline, h.compound, h.sentiment_label;
