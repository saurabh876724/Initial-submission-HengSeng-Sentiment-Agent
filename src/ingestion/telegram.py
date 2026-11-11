"""Telegram public-channel adapter — Apify actor `george.the.developer~telegram-channel-scraper`.

Actor input schema verified against the Apify store build on 2026-11-11:
channels, keywords (case-insensitive message filter), discoverChannels
(searches up to 10 public channels per keyword via Google/DuckDuckGo),
maxMessages, includeMedia, dateFrom/dateTo (YYYY-MM-DD).

Keyword search + channel discovery is used instead of a fixed channel list
so collection works without knowing relevant channels in advance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.settings import Asset
from src.ingestion.apify_client import ApifyRunner
from src.ingestion.base import normalize_item
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "telegram"
ACTOR_ID = "george.the.developer~telegram-channel-scraper"
DEFAULT_MAX_MESSAGES = 200

FIELD_MAP = {
    "text": ["text", "message", "content"],
    "timestamp": ["date", "createdAt", "timestamp", "datetime"],
    "url": ["url", "link", "messageUrl"],
    "views": ["views", "viewCount", "view_count"],
    "reposts": ["forwards", "reposts", "shareCount"],
    "likes": ["reactions", "likes", "reactionCount"],
    "author": ["channel", "channelName", "author", "sender"],
    "author_followers": ["subscribers", "members", "channelFollowers"],
}


def build_input(asset: Asset, days: int = 10, max_messages: int = DEFAULT_MAX_MESSAGES) -> dict:
    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return {
        "channels": [],
        "keywords": asset.search_keywords,
        "discoverChannels": True,
        "maxMessages": max_messages,
        "includeMedia": False,
        "dateFrom": date_from,
    }


def collect(
    asset: Asset,
    runner: ApifyRunner,
    days: int = 10,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    actor_id: str = ACTOR_ID,
) -> list[dict]:
    if not runner.available:
        log.warning("telegram: Apify token missing — skipping %s", asset.ticker)
        return []
    try:
        items = runner.run_actor(actor_id, build_input(asset, days, max_messages))
    except Exception as exc:  # noqa: BLE001
        log.error("telegram collection failed for %s: %s", asset.ticker, exc)
        return []
    posts = [p for p in (normalize_item(SOURCE, asset.ticker, it, FIELD_MAP) for it in items) if p]
    log.info("telegram: %d/%d items normalized for %s", len(posts), len(items), asset.ticker)
    return posts
