"""Daum adapter — Apify actor `rainminer~daum-search-scraper`.

Documented limitation (spec §4: never assume an actor exists):
No dedicated Daum *Finance forum* actor exists on the Apify store as of
2026-11-11. This adapter uses the Daum/Kakao unified search scraper with
stock-related queries, which surfaces Daum blog/cafe/news posts — a
best-effort independent second Korean source. Output items are search
results (title + snippet + url), not full forum threads.
"""
from __future__ import annotations

from urllib.parse import quote

from config.settings import Asset
from src.ingestion.apify_client import ApifyRunner
from src.ingestion.base import normalize_item
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "daum_search"
ACTOR_ID = "rainminer~daum-search-scraper"
DEFAULT_MAX_ITEMS = 30
QUERY_SUFFIX = "주가"  # "stock price" — steers search toward finance content

FIELD_MAP = {
    "text": ["description", "snippet", "content", "text", "body"],
    "title": ["title", "name"],
    "timestamp": ["date", "publishedDate", "createdAt", "timestamp", "datetime"],
    "url": ["url", "link", "href"],
    "author": ["author", "publisher", "source", "channel"],
}


def build_input(asset: Asset, days: int = 10, max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    queries = [f"{asset.local_name} {QUERY_SUFFIX}", f"{asset.company_name} stock"]
    start_urls = [
        {"url": f"https://search.daum.net/search?w=tot&q={quote(q)}"} for q in queries
    ]
    return {
        "startUrls": start_urls,
        "maxItems": max_items,
        "proxyConfiguration": {"useApifyProxy": True},
    }


def collect(
    asset: Asset,
    runner: ApifyRunner,
    days: int = 10,
    max_items: int = DEFAULT_MAX_ITEMS,
    actor_id: str = ACTOR_ID,
) -> list[dict]:
    if not runner.available:
        log.warning("daum: Apify token missing — skipping %s", asset.ticker)
        return []
    try:
        items = runner.run_actor(actor_id, build_input(asset, days, max_items))
    except Exception as exc:  # noqa: BLE001
        log.error("daum collection failed for %s: %s", asset.ticker, exc)
        return []
    posts = [p for p in (normalize_item(SOURCE, asset.ticker, it, FIELD_MAP) for it in items) if p]
    log.info("daum: %d/%d items normalized for %s", len(posts), len(items), asset.ticker)
    return posts
