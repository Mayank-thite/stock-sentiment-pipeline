"""Headline sentiment scoring with VADER.

VADER is a rule based lexicon model. It is fast on CPU and needs no training,
which keeps the pipeline cheap to run. It was built for social media text, not
finance, so it misses words like "beat", "downgrade" or "recall". The README
compares it with the Loughran-McDonald finance lexicon (pipeline/lexicon.py).

One fix I do make: VADER scores "share" and "shares" as positive (+1.2), as in
"thanks for sharing". In headlines it is the noun, so "Tesla shares fall" came
out positive. About 4% of headlines contain the word, so I drop both entries.
"""
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pipeline.config import LABEL_THRESHOLD

# Lexicon entries that mean something else in market news.
NEUTRAL_IN_FINANCE = ("share", "shares")


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    for word in NEUTRAL_IN_FINANCE:
        analyzer.lexicon.pop(word, None)
    return analyzer


def score_headline(text: str | None) -> dict[str, float]:
    """Return VADER scores for one headline. Empty or missing text scores 0."""
    if text is None or not str(text).strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    s = _analyzer().polarity_scores(str(text))
    return {k: float(s[k]) for k in ("compound", "pos", "neu", "neg")}


def label(compound: float, threshold: float = LABEL_THRESHOLD) -> str:
    """Map a compound score to positive, negative or neutral."""
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"
