"""Financial news ingestion via the Exa API.

Per asset: search recent news by company name and ticker, normalize to a
common news record (UTC timestamps), dedupe by URL. Missing API key ->
graceful skip with a warning. Results are never fabricated: if the API
returns nothing, the collector returns an empty list.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import Asset, api_keys, retry_settings
from src.ingestion.base import NEWS_FIELDS, to_utc_iso
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "exa"
MAX_CONTENT_CHARS = 2000


def _make_news(ticker: str, title: str, content: str, url: str, published: Any) -> dict:
    return {
        "source": SOURCE,
        "ticker": ticker,
        "title": (title or "").strip()[:500],
        "content": (content or "").strip()[:MAX_CONTENT_CHARS],
        "url": (url or "").strip(),
        "published_at_utc": to_utc_iso(published),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


class ExaNewsCollector:
    def __init__(self, api_key: str | None = None, client_factory: Any = None):
        self.api_key = api_key if api_key is not None else api_keys.exa
        self._client_factory = client_factory
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self.api_key)
            else:
                from exa_py import Exa

                self._client = Exa(self.api_key)
        return self._client

    def collect(self, asset: Asset, days: int = 10, max_results: int = 10) -> list[dict]:
        if not self.available:
            log.warning("exa: EXA_API_KEY not set — skipping news for %s", asset.ticker)
            return []
        start = datetime.now(timezone.utc) - timedelta(days=days)
        queries = [
            f"{asset.company_name} stock financial news",
            f"{asset.ticker} shares market",
        ]
        records: list[dict] = []
        seen_urls: set[str] = set()
        client = self._get_client()
        for query in queries:
            for attempt in range(1, retry_settings.max_retries + 1):
                try:
                    resp = client.search_and_contents(
                        query,
                        category="news",
                        num_results=max_results,
                        start_published_date=start,
                        text={"max_characters": MAX_CONTENT_CHARS},
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning("exa: query %r attempt %d failed: %s", query, attempt, exc)
                    if attempt == retry_settings.max_retries:
                        resp = None
            if resp is None:
                continue
            for r in getattr(resp, "results", None) or []:
                url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else None)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                get = (lambda k, r=r: r.get(k)) if isinstance(r, dict) else (lambda k, r=r: getattr(r, k, None))
                records.append(
                    _make_news(asset.ticker, get("title"), get("text") or get("summary"), url, get("published_date"))
                )
        log.info("exa: %d news articles for %s", len(records), asset.ticker)
        return records


def news_to_dataframe(records: list[dict]):
    import pandas as pd

    df = pd.DataFrame(records, columns=NEWS_FIELDS)
    return df.drop_duplicates(subset=["source", "url"])
