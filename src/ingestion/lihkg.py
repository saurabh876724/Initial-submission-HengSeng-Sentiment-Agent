"""LIHKG (Hong Kong forum) adapter — Apify actor `claude_code_reviewer~lihkg-scraper-en`.

Uses the actor's keyword-search mode with a relative date window (e.g. "10d").
Actor input schema verified against the Apify store build on 2026-11-11:
mode=search, keywords, dateFrom (relative "10d"), maxItems, maxReplies.
"""
from __future__ import annotations

from config.settings import Asset
from src.ingestion.apify_client import ApifyRunner
from src.ingestion.base import normalize_item
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "lihkg"
ACTOR_ID = "claude_code_reviewer~lihkg-scraper-en"
DEFAULT_MAX_ITEMS = 50
DEFAULT_MAX_REPLIES = 20

FIELD_MAP = {
    "text": ["text", "content", "message", "body", "reply"],
    "title": ["title", "threadTitle"],
    "timestamp": ["createdAt", "createTime", "date", "time", "timestamp", "replyTime"],
    "url": ["url", "threadUrl", "link"],
    "likes": ["likeCount", "likes", "positiveVoteCount", "upvotes"],
    "replies": ["replyCount", "replies", "noOfReplies", "totalReply"],
    "views": ["viewCount", "views", "noOfPageView"],
    "author": ["author", "userName", "nickname", "user"],
}


def build_input(asset: Asset, days: int = 10, max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    return {
        "mode": "search",
        "keywords": asset.search_keywords,
        "dateFrom": f"{days}d",
        "maxItems": max_items,
        "maxReplies": DEFAULT_MAX_REPLIES,
    }


def collect(
    asset: Asset,
    runner: ApifyRunner,
    days: int = 10,
    max_items: int = DEFAULT_MAX_ITEMS,
    actor_id: str = ACTOR_ID,
) -> list[dict]:
    if not runner.available:
        log.warning("lihkg: Apify token missing — skipping %s", asset.ticker)
        return []
    try:
        items = runner.run_actor(actor_id, build_input(asset, days, max_items))
    except Exception as exc:  # noqa: BLE001
        log.error("lihkg collection failed for %s: %s", asset.ticker, exc)
        return []
    posts = [p for p in (normalize_item(SOURCE, asset.ticker, it, FIELD_MAP) for it in items) if p]
    log.info("lihkg: %d/%d items normalized for %s", len(posts), len(items), asset.ticker)
    return posts
