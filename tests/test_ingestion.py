from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import Asset, RetrySettings
from src.ingestion import daum, lihkg, naver, telegram
from src.ingestion.apify_client import ApifyRunner, ApifySourceUnavailable
from src.ingestion.exa_news import ExaNewsCollector
from src.ingestion.tavily_news import TavilyNewsCollector

HK_ASSET = Asset(
    ticker="0700.HK",
    company_name="Tencent Holdings",
    local_name="騰訊控股",
    exchange="HKEX",
    naver_code=None,
    search_keywords=["騰訊", "Tencent", "0700"],
)
KR_ASSET = Asset(
    ticker="005930.KS",
    company_name="Samsung Electronics",
    local_name="삼성전자",
    exchange="KRX",
    naver_code="005930",
    search_keywords=["삼성전자", "Samsung Electronics"],
)

LIHKG_ITEM = {
    "title": "Tencent hits new high",
    "content": "0700 breaking out, buy buy buy",
    "createdAt": "2026-11-05T10:00:00+08:00",
    "url": "https://lihkg.com/thread/123",
    "likeCount": 42,
    "replyCount": 7,
    "userName": "hk_trader",
}


def fake_apify_client(items, status="SUCCEEDED"):
    client = MagicMock()
    client.actor.return_value.call.return_value = {"status": status, "defaultDatasetId": "ds1"}
    client.dataset.return_value.list_items.return_value = SimpleNamespace(items=items, total=len(items))
    return client


def make_runner(client=None, token="test-token", retries=1):
    return ApifyRunner(
        token=token,
        retry=RetrySettings(max_retries=retries, backoff_base_seconds=0.01),
        client_factory=lambda t: client,
    )


# ---------- ApifyRunner ----------

def test_runner_unavailable_without_token():
    runner = make_runner(client=None, token=None)
    assert not runner.available
    with pytest.raises(ApifySourceUnavailable):
        runner.run_actor("a~b", {})


def test_runner_returns_items():
    runner = make_runner(fake_apify_client([LIHKG_ITEM]))
    items = runner.run_actor("claude_code_reviewer~lihkg-scraper-en", {"mode": "search"})
    assert items == [LIHKG_ITEM]


def test_runner_failed_status_raises():
    runner = make_runner(fake_apify_client([], status="FAILED"))
    with pytest.raises(ApifySourceUnavailable):
        runner.run_actor("a~b", {})


def test_runner_transient_errors_return_empty_after_retries():
    client = MagicMock()
    client.actor.side_effect = ConnectionError("boom")
    runner = make_runner(client, retries=2)
    assert runner.run_actor("a~b", {}) == []
    assert client.actor.call_count == 2


# ---------- adapters ----------

def test_lihkg_normalizes_posts():
    runner = make_runner(fake_apify_client([LIHKG_ITEM, {"content": ""}, {"nope": 1}]))
    posts = lihkg.collect(HK_ASSET, runner, days=10)
    assert len(posts) == 1
    p = posts[0]
    assert p["source"] == "lihkg"
    assert p["ticker"] == "0700.HK"
    assert p["text"].startswith("Tencent hits new high")
    assert p["posted_at_utc"].endswith("+00:00")  # normalized to UTC
    assert p["likes"] == 42 and p["replies"] == 7
    assert p["url"] == "https://lihkg.com/thread/123"


def test_lihkg_input_uses_search_mode_and_relative_window():
    inp = lihkg.build_input(HK_ASSET, days=10)
    assert inp["mode"] == "search"
    assert inp["dateFrom"] == "10d"
    assert "騰訊" in inp["keywords"]


def test_telegram_input_discovery_and_date():
    inp = telegram.build_input(HK_ASSET, days=10)
    assert inp["discoverChannels"] is True
    assert inp["keywords"] == HK_ASSET.search_keywords
    assert len(inp["dateFrom"]) == 10  # YYYY-MM-DD


def test_telegram_normalizes_posts():
    item = {"text": "$Tencent accumulating", "date": 1762300000, "views": 5000, "channel": "hk_stocks"}
    runner = make_runner(fake_apify_client([item]))
    posts = telegram.collect(HK_ASSET, runner)
    assert len(posts) == 1
    assert posts[0]["views"] == 5000
    assert posts[0]["posted_at_utc"].startswith("2025-11-04T23:46:40")


def test_naver_skips_non_krx_asset():
    runner = make_runner(fake_apify_client([{"content": "x"}]))
    assert naver.collect(HK_ASSET, runner) == []


def test_naver_uses_krx_code():
    assert naver.build_input(KR_ASSET)["stocks"] == ["005930"]
    item = {"content": "삼성전자 간다", "regTime": "2026-11-06 09:30:00", "nickName": "개미1"}
    runner = make_runner(fake_apify_client([item]))
    posts = naver.collect(KR_ASSET, runner)
    assert len(posts) == 1 and posts[0]["author"] == "개미1"


def test_daum_builds_search_urls():
    inp = daum.build_input(KR_ASSET)
    urls = [u["url"] for u in inp["startUrls"]]
    assert any("search.daum.net" in u for u in urls)
    item = {"title": "삼성전자 주가 전망", "description": "반등 예상된다", "url": "https://blog.daum.net/x/1"}
    runner = make_runner(fake_apify_client([item]))
    posts = daum.collect(KR_ASSET, runner)
    assert len(posts) == 1
    assert "삼성전자" in posts[0]["text"]


# ---------- news collectors ----------

def test_exa_collects_and_dedupes():
    resp = SimpleNamespace(
        results=[
            SimpleNamespace(title="Tencent beats estimates", url="https://news/1", text="Strong quarter.", published_date="2026-11-08T00:00:00Z"),
            SimpleNamespace(title="dup", url="https://news/1", text="Same url.", published_date="2026-11-08T00:00:00Z"),
        ]
    )
    client = MagicMock()
    client.search_and_contents.return_value = resp
    exa = ExaNewsCollector(api_key="k", client_factory=lambda key: client)
    records = exa.collect(HK_ASSET, days=10)
    urls = [r["url"] for r in records]
    assert len(set(urls)) == len(urls)  # deduped across queries
    assert records[0]["published_at_utc"].endswith("+00:00")


def test_exa_unavailable_without_key():
    exa = ExaNewsCollector(api_key=None)
    assert not exa.available
    assert exa.collect(HK_ASSET) == []


def test_exa_client_exception_returns_empty():
    client = MagicMock()
    client.search_and_contents.side_effect = RuntimeError("api down")
    exa = ExaNewsCollector(api_key="k", client_factory=lambda key: client)
    assert exa.collect(HK_ASSET) == []


def test_tavily_collects():
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {"title": "Samsung news", "url": "https://n/1", "content": "Chip demand up.", "published_date": "2026-11-09"},
            {"title": "no url"},
        ]
    }
    tavily = TavilyNewsCollector(api_key="k", client_factory=lambda key: client)
    records = tavily.collect(KR_ASSET)
    assert len(records) == 1
    assert records[0]["source"] == "tavily"


def test_tavily_unavailable_without_key():
    tavily = TavilyNewsCollector(api_key=None)
    assert not tavily.available
    assert tavily.collect(KR_ASSET) == []
