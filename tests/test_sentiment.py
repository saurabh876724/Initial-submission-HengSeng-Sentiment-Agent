import pandas as pd
import pytest

from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    aggregate_sentiment,
    engagement_weight,
    sentiment_by_source,
    sentiment_trend,
)


class FakePipeline:
    """Rule-based stand-in for the HF pipeline: keyword -> probabilities."""

    def __call__(self, text):
        t = text.lower()
        if "crash" in t or "loss" in t:
            probs = {"negative": 0.8, "neutral": 0.1, "positive": 0.1}
        elif "rally" in t or "beat" in t or "upgrade" in t:
            probs = {"positive": 0.75, "neutral": 0.2, "negative": 0.05}
        else:
            probs = {"neutral": 0.7, "positive": 0.2, "negative": 0.1}
        return [[{"label": k, "score": v} for k, v in probs.items()]]


@pytest.fixture
def analyzer():
    return SentimentAnalyzer(pipeline=FakePipeline())


def test_score_positive(analyzer):
    label, score = analyzer.score_text("Tencent beats estimates, shares rally")
    assert label == "positive"
    assert score == pytest.approx(0.70, abs=0.01)


def test_score_negative(analyzer):
    label, score = analyzer.score_text("Big loss after crash warning")
    assert label == "negative"
    assert score == pytest.approx(-0.70, abs=0.01)


def test_score_neutral_and_range(analyzer):
    label, score = analyzer.score_text("The company held its annual meeting")
    assert label == "neutral"
    assert -1.0 <= score <= 1.0


def test_score_empty_text(analyzer):
    assert analyzer.score_text("") == ("neutral", 0.0)
    assert analyzer.score_text(None) == ("neutral", 0.0)


def test_annotate_adds_columns(analyzer):
    df = pd.DataFrame({"text": ["rally today", "crash feared"], "ticker": ["0700.HK"] * 2})
    out = analyzer.annotate(df)
    assert list(out["sentiment_label"]) == ["positive", "negative"]
    assert out["sentiment_score"].iloc[0] > 0 > out["sentiment_score"].iloc[1]


def test_annotate_empty_df(analyzer):
    out = analyzer.annotate(pd.DataFrame(columns=["text"]))
    assert "sentiment_score" in out.columns and out.empty


def annotated_posts():
    rows = [
        {"ticker": "0700.HK", "source": "lihkg", "text": "rally", "sentiment_label": "positive", "sentiment_score": 0.7, "likes": 10, "replies": 2, "reposts": 0, "author_followers": 100, "trading_date": "2026-11-09"},
        {"ticker": "0700.HK", "source": "lihkg", "text": "crash", "sentiment_label": "negative", "sentiment_score": -0.8, "likes": 0, "replies": 0, "reposts": 0, "author_followers": None, "trading_date": "2026-11-09"},
        {"ticker": "0700.HK", "source": "telegram", "text": "meeting", "sentiment_label": "neutral", "sentiment_score": 0.1, "likes": 5, "replies": 0, "reposts": 1, "author_followers": 50, "trading_date": "2026-11-10"},
        {"ticker": "005930.KS", "source": "naver", "text": "upgrade", "sentiment_label": "positive", "sentiment_score": 0.6, "likes": 1, "replies": 0, "reposts": 0, "author_followers": None, "trading_date": "2026-11-10"},
    ]
    return pd.DataFrame(rows)


def test_aggregate_sentiment():
    agg = aggregate_sentiment(annotated_posts()).set_index("ticker")
    hk = agg.loc["0700.HK"]
    assert hk["mentions"] == 3
    assert hk["pos_pct"] == pytest.approx(33.33, abs=0.1)
    assert hk["avg_sentiment"] == pytest.approx(0.0, abs=0.01)  # (0.7 - 0.8 + 0.1)/3
    assert -1 <= hk["engagement_weighted_sentiment"] <= 1
    assert agg.loc["005930.KS", "mentions"] == 1


def test_aggregate_empty():
    out = aggregate_sentiment(pd.DataFrame())
    assert out.empty and "avg_sentiment" in out.columns


def test_engagement_weight_sublinear():
    low = engagement_weight(pd.Series({"likes": 1, "replies": 0, "reposts": 0, "author_followers": None}))
    high = engagement_weight(pd.Series({"likes": 1_000_000, "replies": 0, "reposts": 0, "author_followers": None}))
    assert high > low
    assert high < 20 * low  # log-scaled, viral post cannot dominate


def test_engagement_weight_handles_missing():
    w = engagement_weight(pd.Series({"likes": None, "replies": None, "reposts": None, "author_followers": None}))
    assert w == 1.0


def test_sentiment_by_source():
    out = sentiment_by_source(annotated_posts())
    lihkg = out[(out["source"] == "lihkg") & (out["ticker"] == "0700.HK")]
    assert lihkg["mentions"].iloc[0] == 2
    assert lihkg["avg_sentiment"].iloc[0] == pytest.approx(-0.05, abs=0.001)


def test_sentiment_trend():
    out = sentiment_trend(annotated_posts())
    hk_1109 = out[(out["ticker"] == "0700.HK") & (out["date"] == "2026-11-09")]
    assert hk_1109["mentions"].iloc[0] == 2


@pytest.mark.network
def test_real_finbert_scores():
    """Live FinBERT check — downloads model on first run. pytest -m network"""
    real = SentimentAnalyzer()
    label, score = real.score_text("Company reports record profits and upgrades guidance")
    assert label == "positive" and score > 0.2
    label, score = real.score_text("Shares plunge on widening losses and lawsuit")
    assert label == "negative" and score < -0.2
