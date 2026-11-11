"""Optional secondary news source: Tavily search API.

Used only as a fallback/complement to Exa (spec §5). Same normalization
and graceful-degradation behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config.settings import Asset, api_keys
from src.ingestion.base import NEWS_FIELDS, to_utc_iso
from src.utils.logger import get_logger

log = get_logger(__name__)

SOURCE = "tavily"


class TavilyNewsCollector:
    def __init__(self, api_key: str | None = None, client_factory: Any = None):
        self.api_key = api_key if api_key is not None else api_keys.tavily
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
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def collect(self, asset: Asset, days: int = 10, max_results: int = 5) -> list[dict]:
        if not self.available:
            log.info("tavily: TAVILY_API_KEY not set — optional source skipped")
            return []
        try:
            resp = self._get_client().search(
                query=f"{asset.company_name} {asset.ticker} stock news",
                topic="news",
                days=days,
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("tavily: search failed for %s: %s", asset.ticker, exc)
            return []
        records = []
        for r in resp.get("results", []):
            url = r.get("url")
            if not url:
                continue
            records.append(
                {
                    "source": SOURCE,
                    "ticker": asset.ticker,
                    "title": (r.get("title") or "").strip()[:500],
                    "content": (r.get("content") or "").strip()[:2000],
                    "url": url.strip(),
                    "published_at_utc": to_utc_iso(r.get("published_date")),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        log.info("tavily: %d news articles for %s", len(records), asset.ticker)
        return records


def news_to_dataframe(records: list[dict]):
    import pandas as pd

    df = pd.DataFrame(records, columns=NEWS_FIELDS)
    return df.drop_duplicates(subset=["source", "url"])
