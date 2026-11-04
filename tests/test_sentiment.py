from pipeline.sentiment import label, score_headline


def test_positive_and_negative_headlines():
    assert score_headline("Apple posts record profit and strong growth")["compound"] > 0.05
    assert score_headline("Tesla shares crash after fraud lawsuit")["compound"] < -0.05


def test_empty_and_missing_text_score_zero():
    for text in (None, "", "   "):
        s = score_headline(text)
        assert s["compound"] == 0.0 and s["neu"] == 1.0


def test_scores_are_bounded():
    s = score_headline("great great great amazing wonderful best ever")
    assert -1.0 <= s["compound"] <= 1.0
    assert abs(s["pos"] + s["neu"] + s["neg"] - 1.0) < 0.01


def test_label_thresholds():
    assert label(0.05) == "positive"
    assert label(-0.05) == "negative"
    assert label(0.049) == "neutral"
    assert label(0.0) == "neutral"


def test_shares_is_not_positive():
    # plain VADER scores this positive because of "shares"
    assert score_headline("Tesla shares fall after delivery numbers")["compound"] <= 0
    assert score_headline("Apple shares")["compound"] == 0.0
