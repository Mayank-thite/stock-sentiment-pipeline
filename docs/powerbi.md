# Loading the data into Power BI

I can't build a .pbix file on Linux, so the repo ships the model inputs instead.
Everything is in `outputs/powerbi/`.

| File | Grain | Key |
|---|---|---|
| `fact_sentiment_daily.parquet` | one ticker on one trading day with at least one headline | `date_key`, `ticker` |
| `dim_ticker.csv` / `.parquet` | one ticker | `ticker` |
| `dim_date.csv` / `.parquet` | one calendar day | `date_key` (yyyymmdd) |
| `measures.dax` | DAX measures | |

The fact table is committed as Parquet only, to keep the repo small. Run
`make powerbi` after a pipeline run to also write `fact_sentiment_daily.csv`.

## Steps

1. Power BI Desktop, Get data, Parquet (or Text/CSV). Load the three tables.
2. Model view. Create two relationships, both many to one, single direction:
   - `fact_sentiment_daily[date_key]` to `dim_date[date_key]`
   - `fact_sentiment_daily[ticker]` to `dim_ticker[ticker]`
3. Mark `dim_date` as a date table, using `dim_date[date]`.
4. Set the format of `ret_*` and `abn_*` columns to percentage.
5. New measure, paste each measure from `measures.dax`. Put them in a
   display folder called Measures if you like.

## A simple report layout

- Cards: News Days, Headlines, Avg Sentiment, Corr Sentiment Next Day,
  Direction Hit Rate Next Day.
- Line chart: Avg Sentiment by `dim_date[year_month]`.
- Clustered column chart: Avg Return Same Day and Avg Return Next Day by
  `fact_sentiment_daily[day_label]`.
- Table: `dim_ticker[ticker]` with News Days, Avg Sentiment and
  Corr Sentiment Next Day, sorted by the correlation.
- Slicers: `dim_date[year]`, `dim_ticker[ticker]`.

The correlation measures follow the filter context, so slicing by year or
ticker gives the same numbers as `outputs/results.md` for that slice. With
no filters, Corr Sentiment Next Day should match the Pearson r for
`ret_next` in `outputs/results.md`.
