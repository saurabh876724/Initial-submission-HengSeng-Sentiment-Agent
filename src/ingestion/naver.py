"""Naver Finance stock discussion board (종목토론실) adapter —
Apify actor `hgservices~naver-stock-discussion-scraper`.

Actor input schema verified against the Apify store on 2026-11-11:
{"stocks": ["005930"], "maxResults": 100} — takes raw Korean ticker codes.
Only applicable to KRX assets (naver_code present).
"""
from __future__ import annotations

from config.settings import Asset
from src.ingestion.apify_client import ApifyRunner
from src.ingestion.base import normalize_item
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "naver_stock_discussion"
ACTOR_ID = "hgservices~naver-stock-discussion-scraper"
DEFAULT_MAX_RESULTS = 200

FIELD_MAP = {
    "text": ["content", "text", "body", "message"],
    "title": ["title", "subject"],
    "timestamp": ["createdAt", "regTime", "registeredAt", "date", "writeTime", "timestamp"],
    "url": ["url", "link"],
    "likes": ["likeCount", "likes", "recommend", "sympathy"],
    "replies": ["replyCount", "replies", "commentCount"],
    "views": ["viewCount", "views", "readCount"],
    "author": ["nickName", "nickname", "author", "userName", "writerId"],
}


def build_input(asset: Asset, max_results: int = DEFAULT_MAX_RESULTS) -> dict:
    return {"stocks": [asset.naver_code], "maxResults": max_results}


def collect(
    asset: Asset,
    runner: ApifyRunner,
    days: int = 10,
    max_results: int = DEFAULT_MAX_RESULTS,
    actor_id: str = ACTOR_ID,
) -> list[dict]:
    """`days` is accepted for adapter-interface parity; the actor returns the
    most recent posts, and window filtering happens in preprocessing."""
    if not runner.available:
        log.warning("naver: Apify token missing — skipping %s", asset.ticker)
        return []
    if not asset.naver_code:
        log.debug("naver: %s has no KRX code — skipping", asset.ticker)
        return []
    try:
        items = runner.run_actor(actor_id, build_input(asset, max_results))
    except Exception as exc:  # noqa: BLE001
        log.error("naver collection failed for %s: %s", asset.ticker, exc)
        return []
    posts = [p for p in (normalize_item(SOURCE, asset.ticker, it, FIELD_MAP) for it in items) if p]
    log.info("naver: %d/%d items normalized for %s", len(posts), len(items), asset.ticker)
    return posts
