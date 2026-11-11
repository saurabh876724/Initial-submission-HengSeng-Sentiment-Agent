from unittest.mock import MagicMock

import pytest

from config.settings import Asset, Universe
from src.ingestion import pipeline

HK = Asset("0700.HK", "Tencent Holdings", "騰訊控股", "HKEX", None, ["Tencent"])
KR = Asset("005930.KS", "Samsung Electronics", "삼성전자", "KRX", "005930", ["Samsung"])

TINY_UNIVERSE = Universe(
    social_lookback_days=10,
    history_years=1,
    exchanges={},
    assets=[HK, KR],
    apify_actors={},
)


def sample_post(source, ticker, text):
    from src.ingestion.base import make_post

    return make_post(source, ticker, text, "2026-11-05T10:00:00Z", f"https://x/{source}/{text}")


@pytest.fixture
def capture(monkeypatch, tmp_path):
    """Redirect sqlite saves + raw dir; capture saved (df, table) pairs."""
    saved = []
    monkeypatch.setattr(pipeline, "save_to_sqlite", lambda df, table, **kw: saved.append((df, table)) or len(df))
    monkeypatch.setattr(pipeline, "RAW_DIR", tmp_path)
    return saved, tmp_path


def test_collect_social_happy_path(capture, monkeypatch):
    saved, raw_dir = capture
    runner = MagicMock(available=True)
    monkeypatch.setattr(
        pipeline.lihkg, "collect", lambda asset, r, days=10, actor_id=None: [sample_post("lihkg", asset.ticker, "bull post")]
    )
    monkeypatch.setattr(
        pipeline.telegram, "collect", lambda asset, r, days=10, actor_id=None: [sample_post("telegram", asset.ticker, "tg post")]
    )
    monkeypatch.setattr(pipeline.naver, "collect", lambda asset, r, days=10, actor_id=None: [sample_post("naver", asset.ticker, "kr post")])
    monkeypatch.setattr(pipeline.daum, "collect", lambda asset, r, days=10, actor_id=None: [])

    df = pipeline.collect_social(TINY_UNIVERSE, runner=runner)

    # HK asset: lihkg+telegram; KR asset: naver (+daum empty) = 3 posts
    assert len(df) == 3
    tables = [t for _, t in saved]
    assert "social_posts" in tables and "collection_status" in tables
    assert list(raw_dir.glob("social_lihkg_0700_HK_*.json"))


def test_collect_social_isolates_source_failure(capture, monkeypatch):
    saved, _ = capture
    runner = MagicMock(available=True)

    def boom(asset, r, days=10, actor_id=None):
        raise RuntimeError("actor exploded")

    monkeypatch.setattr(pipeline.lihkg, "collect", boom)
    monkeypatch.setattr(pipeline.telegram, "collect", lambda asset, r, days=10, actor_id=None: [sample_post("telegram", asset.ticker, "ok")])
    monkeypatch.setattr(pipeline.naver, "collect", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.daum, "collect", lambda *a, **kw: [])

    df = pipeline.collect_social(TINY_UNIVERSE, runner=runner)
    assert len(df) == 1  # telegram still delivered

    status_df = next(d for d, t in saved if t == "collection_status")
    lihkg_rows = status_df[(status_df["source"] == "lihkg")]
    assert not lihkg_rows["ok"].any()
    assert "exploded" in lihkg_rows["error"].iloc[0]


def test_collect_news_exa_primary(capture):
    saved, raw_dir = capture
    exa_record = {
        "source": "exa", "ticker": "0700.HK", "title": "T", "content": "C",
        "url": "https://n/1", "published_at_utc": "2026-11-08T00:00:00+00:00",
        "collected_at": "2026-11-11T00:00:00+00:00",
    }
    tavily_record = {
        "source": "tavily", "ticker": "005930.KS", "title": "S", "content": "C",
        "url": "https://n/2", "published_at_utc": "2026-11-09T00:00:00+00:00",
        "collected_at": "2026-11-11T00:00:00+00:00",
    }
    exa = MagicMock(available=True)
    exa.collect.side_effect = lambda asset, days=10: [dict(exa_record)] if asset.ticker == "0700.HK" else []
    tavily = MagicMock(available=True)
    tavily.collect.side_effect = lambda asset, days=10: [dict(tavily_record)]

    df = pipeline.collect_news(TINY_UNIVERSE, exa=exa, tavily=tavily)
    assert len(df) == 2
    # tavily only used as fallback for the KR asset where exa returned nothing
    assert tavily.collect.call_count == 1
    assert list(raw_dir.glob("news_0700_HK_*.json"))
    assert list(raw_dir.glob("news_005930_KS_*.json"))


def test_collect_news_records_missing_key(capture):
    saved, _ = capture
    exa = MagicMock(available=False)
    exa.collect.return_value = []
    tavily = MagicMock(available=False)
    tavily.collect.return_value = []

    df = pipeline.collect_news(TINY_UNIVERSE, exa=exa, tavily=tavily)
    assert df.empty
    status_df = next(d for d, t in saved if t == "collection_status")
    assert not status_df["ok"].any()
    assert "EXA_API_KEY" in status_df["error"].iloc[0]
