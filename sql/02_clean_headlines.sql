-- Clean headlines: trim, collapse whitespace, drop empties, and remove
-- duplicates (same ticker, same day, same text after lower casing).

DROP TABLE IF EXISTS sentiment.clean_headlines;
CREATE TABLE sentiment.clean_headlines STORED AS PARQUET AS
WITH base AS (
    SELECT
        headline_id,
        upper(trim(ticker))                                AS ticker,
        to_date(substr(published_at, 1, 10))               AS pub_date,
        trim(regexp_replace(headline, '\\s+', ' '))        AS headline,
        publisher,
        compound, pos, neu, neg
    FROM sentiment.raw_headline_scores
    WHERE headline IS NOT NULL
      AND length(trim(headline)) > 0
      AND ticker IS NOT NULL
      AND to_date(substr(published_at, 1, 10)) IS NOT NULL
),
ranked AS (
    SELECT
        b.*,
        row_number() OVER (
            PARTITION BY ticker, pub_date, lower(headline)
            ORDER BY headline_id
        ) AS rn
    FROM base b
)
SELECT
    headline_id, ticker, pub_date, headline, publisher,
    compound, pos, neu, neg,
    CASE
        WHEN compound >= ${THRESHOLD}  THEN 'positive'
        WHEN compound <= -${THRESHOLD} THEN 'negative'
        ELSE 'neutral'
    END AS sentiment_label
FROM ranked
WHERE rn = 1;
