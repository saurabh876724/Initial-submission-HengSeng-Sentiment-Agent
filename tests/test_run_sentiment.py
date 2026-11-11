import pandas as pd
import pytest

from config.settings import Asset, Universe
from src.ingestion.base import make_post
from src.sentiment import run_sentiment
from tests.test_sentiment import FakePipeline
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

HK = Asset("0700.HK", "Tencent", "騰訊", "HKEX", None, ["Tencent"])
UNI = Universe(10, 1, {}, [HK], {})


def test_annotate_all_end_to_end_with_fake_model(monkeypatch, tmp_path):
    post = make_post("lihkg", "0700.HK", "Tencent rally continues after earnings beat",
                     pd.Timestamp.now(tz="UTC").isoformat(), "https://l/1", likes=3)
    news_row = {
        "source": "exa", "ticker": "0700.HK", "title": "Tencent beats estimates",
        "content": "Strong quarter with rally in shares.", "url": "https://n/1",
        "published_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "collected_at": "x",
    }
    monkeypatch.setattr(run_sentiment, "load_from_sqlite",
                        lambda table, **kw: pd.DataFrame([post]) if table == "social_posts" else pd.DataFrame([news_row]))
    replaced = {}
    monkeypatch.setattr(run_sentiment, "replace_table", lambda df, table, **kw: replaced.setdefault(table, df))
    monkeypatch.setattr(run_sentiment, "save_parquet", lambda df, path: None)
    monkeypatch.setattr(run_sentiment, "PROCESSED_DIR", tmp_path)

    out = run_sentiment.annotate_all(UNI, analyzer=SentimentAnalyzer(pipeline=FakePipeline()))

    assert len(out["posts"]) == 1
    assert out["posts"]["sentiment_label"].iloc[0] == "positive"
    assert out["posts"]["trading_date"].notna().iloc[0]
    assert len(out["news"]) == 1
    assert out["news"]["sentiment_label"].iloc[0] == "positive"
    assert set(replaced) == {"posts_annotated", "news_annotated"}
    agg = out["social_agg"]
    assert agg["mentions"].iloc[0] == 1
    assert agg["avg_sentiment"].iloc[0] > 0


def test_annotate_all_handles_empty_store(monkeypatch):
    monkeypatch.setattr(run_sentiment, "load_from_sqlite", lambda table, **kw: pd.DataFrame())
    monkeypatch.setattr(run_sentiment, "replace_table", lambda df, table, **kw: 0)
    monkeypatch.setattr(run_sentiment, "save_parquet", lambda df, path: None)
    out = run_sentiment.annotate_all(UNI, analyzer=SentimentAnalyzer(pipeline=FakePipeline()))
    assert out["posts"].empty and out["social_agg"].empty
