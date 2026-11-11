"""Text and dataframe cleaning for collected posts/news.

No fabrication: rows without usable text are dropped and logged, never
imputed with placeholder content.
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

ENGAGEMENT_COLS = ["likes", "replies", "reposts", "views", "author_followers"]


def clean_text(text: str | None, keep_urls: bool = False) -> str:
    """Normalize a single post/article text."""
    if not isinstance(text, str):
        return ""
    text = _CONTROL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    if not keep_urls:
        text = _URL.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean_posts(df: pd.DataFrame, lookback_days: int = 10, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Clean a normalized posts dataframe:
    - drop empty/missing text
    - dedupe on (source, post_hash)
    - parse posted_at_utc; drop posts older than the lookback window
      (posts without a timestamp are KEPT but flagged timestamp_missing=True —
      they still carry sentiment signal)
    - coerce engagement columns to nullable numeric
    """
    if df.empty:
        return df
    out = df.copy()
    out["text"] = out["text"].map(clean_text)
    before = len(out)
    out = out[out["text"].str.len() > 0]
    dropped = before - len(out)
    out = out.drop_duplicates(subset=["source", "post_hash"])

    now = now or pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(out["posted_at_utc"], errors="coerce", utc=True, format="ISO8601")
    out["posted_at_utc"] = parsed
    out["timestamp_missing"] = parsed.isna()
    cutoff = now - pd.Timedelta(days=lookback_days)
    stale = parsed.notna() & (parsed < cutoff)
    out = out[~stale]

    for col in ENGAGEMENT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    log.info(
        "cleaning: %d posts kept (%d empty-text dropped, %d stale dropped, %d missing timestamps)",
        len(out), dropped, int(stale.sum()), int(out["timestamp_missing"].sum()),
    )
    return out.reset_index(drop=True)


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a normalized news dataframe: dedupe on url, clean title/content."""
    if df.empty:
        return df
    out = df.copy()
    out["title"] = out["title"].map(lambda t: clean_text(t, keep_urls=False))
    out["content"] = out["content"].map(lambda t: clean_text(t, keep_urls=False))
    out = out[(out["title"].str.len() > 0) | (out["content"].str.len() > 0)]
    out = out.drop_duplicates(subset=["source", "url"])
    out["published_at_utc"] = pd.to_datetime(out["published_at_utc"], errors="coerce", utc=True, format="ISO8601")
    log.info("cleaning: %d news articles kept", len(out))
    return out.reset_index(drop=True)
