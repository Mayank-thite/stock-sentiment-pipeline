import pandas as pd

from pipeline.lexicon import WordLists, lm_tone, parse_master_dictionary, phrase_check, tone_label

LISTS = WordLists(positive=frozenset({"GAIN", "STRONG"}), negative=frozenset({"LOSS", "RECALL", "DOWNGRADE"}))


def test_parse_master_dictionary_skips_removed_words():
    df = pd.DataFrame({"Word": ["gain", "loss", "old"], "Positive": [2009, 0, 0], "Negative": [0, 2009, -2020]})
    lists = parse_master_dictionary(df)
    assert lists.positive == {"GAIN"}
    assert lists.negative == {"LOSS"}


def test_lm_tone():
    assert lm_tone("Strong gain for Apple", LISTS) == 1.0
    assert lm_tone("Ford recall after downgrade", LISTS) == -1.0
    assert lm_tone("Strong quarter despite loss", LISTS) == 0.0
    assert lm_tone("Apple holds meeting", LISTS) == 0.0
    assert lm_tone(None, LISTS) == 0.0
    assert tone_label(0.5) == "positive" and tone_label(-0.1) == "negative" and tone_label(0) == "neutral"


def test_phrase_check_runs_without_the_dictionary():
    out = phrase_check(None)
    assert out["n"] == 30
    assert 0 <= out["vader_accuracy"] <= 1
    assert "lm_accuracy" not in out
