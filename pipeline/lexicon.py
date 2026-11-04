"""Loughran-McDonald finance lexicon, used as a second opinion next to VADER.

VADER was built for social media. Loughran and McDonald built their word lists
from 10-K filings, so words like "liability", "tax" or "cost" are not negative
there, and "downgrade" or "recall" are. I score each headline as

    lm_tone = (positive words - negative words) / (positive words + negative words)

and 0 when it has none, so it runs from -1 to 1 like VADER compound.

The dictionary is free for academic use only, so it is not in the repo.
scripts/download_data.py fetches it into data/raw/. Every step that uses it
skips cleanly when the file is missing (CI and the sample run).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import LM_DICTIONARY, ROOT
from pipeline.sentiment import label, score_headline

log = logging.getLogger(__name__)

PHRASES = ROOT / "data" / "checks" / "finance_phrases.csv"
_TOKEN = re.compile(r"[A-Z]+(?:'[A-Z]+)?")


@dataclass(frozen=True)
class WordLists:
    positive: frozenset[str]
    negative: frozenset[str]


def parse_master_dictionary(df: pd.DataFrame) -> WordLists:
    """Pull the positive and negative lists out of the LM master dictionary.

    The Positive and Negative columns hold the year a word was added. A
    negative year means it was removed later, so only values above 0 count.
    """
    words = df["Word"].astype(str).str.upper()
    return WordLists(
        positive=frozenset(words[df["Positive"] > 0]),
        negative=frozenset(words[df["Negative"] > 0]),
    )


@lru_cache(maxsize=1)
def load_word_lists(path: Path = LM_DICTIONARY) -> WordLists | None:
    if not Path(path).exists():
        log.info("Loughran-McDonald dictionary not found at %s, skipping the LM comparison", path)
        return None
    df = pd.read_csv(path, usecols=["Word", "Positive", "Negative"], keep_default_na=False)
    return parse_master_dictionary(df)


def lm_tone(text: str | None, lists: WordLists) -> float:
    """(pos - neg) / (pos + neg) over LM words in the text, 0 when there are none."""
    tokens = _TOKEN.findall(str(text or "").upper())
    pos = sum(t in lists.positive for t in tokens)
    neg = sum(t in lists.negative for t in tokens)
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


def tone_label(tone: float) -> str:
    if tone > 0:
        return "positive"
    if tone < 0:
        return "negative"
    return "neutral"


def add_lm_scores(headlines: pd.DataFrame, lists: WordLists) -> pd.DataFrame:
    out = headlines.copy()
    out["lm_tone"] = out["headline"].map(lambda t: lm_tone(t, lists))
    out["lm_label"] = out["lm_tone"].map(tone_label)
    return out


def phrase_check(lists: WordLists | None, path: Path = PHRASES) -> dict:
    """Score a small set of finance phrases that I labelled by hand.

    It is 30 phrases, so it is a smell test, not a benchmark. The point is to
    see which finance words each lexicon gets wrong.
    """
    df = pd.read_csv(path)
    df["vader"] = df["phrase"].map(lambda t: label(score_headline(t)["compound"]))
    out = {"n": int(len(df)), "vader_accuracy": float((df["vader"] == df["expected"]).mean())}
    if lists is not None:
        df["lm"] = df["phrase"].map(lambda t: tone_label(lm_tone(t, lists)))
        out["lm_accuracy"] = float((df["lm"] == df["expected"]).mean())
    out["rows"] = df.to_dict("records")
    return out


def label_agreement(headlines: pd.DataFrame) -> dict:
    """How often VADER and LM give the same label on the real headlines."""
    tab = pd.crosstab(headlines["sentiment_label"], headlines["lm_label"])
    return {
        "share_same_label": float((headlines["sentiment_label"] == headlines["lm_label"]).mean()),
        "pearson_compound_vs_lm_tone": float(np.corrcoef(headlines["compound"], headlines["lm_tone"])[0, 1]),
        "share_lm_neutral": float((headlines["lm_label"] == "neutral").mean()),
        "crosstab_vader_rows_lm_cols": {r: {c: int(tab.loc[r, c]) for c in tab.columns} for r in tab.index},
    }
