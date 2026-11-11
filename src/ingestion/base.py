"""Common record schema and normalization helpers shared by all adapters."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

POST_FIELDS = [
    "source",
    "ticker",
    "text",
    "posted_at_utc",
    "url",
    "likes",
    "replies",
    "reposts",
    "views",
    "author",
    "author_followers",
    "collected_at",
    "post_hash",
]

NEWS_FIELDS = [
    "source",
    "ticker",
    "title",
    "content",
    "url",
    "published_at_utc",
    "collected_at",
]


def to_utc_iso(value: Any) -> str | None:
    """Normalize a timestamp (ISO string, epoch s/ms, datetime) to UTC ISO-8601."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        # epoch milliseconds vs seconds heuristic
        ts = value / 1000.0 if value > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    parsed = pd.to_datetime(str(value), errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().isoformat()


def _first_int(item: dict, candidates: list[str]) -> int | None:
    for key in candidates:
        if key in item and item[key] is not None:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                continue
    return None


def _first_str(item: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def make_post(
    source: str,
    ticker: str,
    text: Any,
    timestamp: Any = None,
    url: Any = None,
    likes: Any = None,
    replies: Any = None,
    reposts: Any = None,
    views: Any = None,
    author: Any = None,
    author_followers: Any = None,
) -> dict | None:
    """Build a normalized post record. Returns None when there is no text."""
    from src.utils.storage import post_hash

    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "source": source,
        "ticker": ticker,
        "text": text.strip(),
        "posted_at_utc": to_utc_iso(timestamp),
        "url": str(url).strip() if url else None,
        "likes": likes,
        "replies": replies,
        "reposts": reposts,
        "views": views,
        "author": author,
        "author_followers": author_followers,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "post_hash": post_hash(text, str(url) if url else None),
    }


def normalize_item(source: str, ticker: str, item: dict, field_map: dict[str, list[str]]) -> dict | None:
    """Map an arbitrary actor output item onto the common post schema using
    ordered candidate field names (actors differ; fields may be absent)."""
    text = _first_str(item, field_map.get("text", ["text", "content", "message", "body"]))
    title = _first_str(item, field_map.get("title", ["title"]))
    if title and text and title not in text:
        text = f"{title} — {text}"
    elif title and not text:
        text = title
    return make_post(
        source=source,
        ticker=ticker,
        text=text,
        timestamp=normalize_timestamp(item, field_map.get("timestamp", ["date", "createdAt", "timestamp", "time"])),
        url=_first_str(item, field_map.get("url", ["url", "link"])),
        likes=_first_int(item, field_map.get("likes", ["likeCount", "likes", "positiveVoteCount"])),
        replies=_first_int(item, field_map.get("replies", ["replyCount", "replies", "noOfReplies", "comments"])),
        reposts=_first_int(item, field_map.get("reposts", ["repostCount", "reposts", "shares", "retweets"])),
        views=_first_int(item, field_map.get("views", ["views", "viewCount", "view_count"])),
        author=_first_str(item, field_map.get("author", ["author", "userName", "nickName", "user"])),
        author_followers=_first_int(item, field_map.get("author_followers", ["authorFollowers", "followers", "subscriberCount"])),
    )


def normalize_timestamp(item: dict, candidates: list[str]) -> Any:
    for key in candidates:
        if item.get(key) is not None:
            return item[key]
    return None


def posts_to_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=POST_FIELDS)
    return df.drop_duplicates(subset=["source", "post_hash"])
