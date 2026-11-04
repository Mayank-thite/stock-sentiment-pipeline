"""Headline sentiment scoring with VADER.

VADER is a rule based lexicon model. It is fast on CPU and needs no training,
which keeps the pipeline cheap to run. It was built for social media text, not
finance, so it misses domain words like "beat" or "downgrade". That is a known
limitation and is called out in the README.
"""
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pipeline.config import LABEL_THRESHOLD


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_headline(text) -> dict:
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
