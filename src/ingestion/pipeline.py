"""Collection orchestrator: social posts (Apify adapters) + news (Exa/Tavily).

Design rules honored here:
- each source is independent; one failure never aborts the run
- every run writes raw JSON evidence to data/raw/ and a status row to the
  `collection_status` SQLite table (used by the dashboard's Data Status view)
- posts are deduped on (source, post_hash); news on (source, url)
- nothing is fabricated: sources without credentials or results produce
  empty frames plus an explanatory status row
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config.settings import RAW_DIR, Universe, load_universe
from src.ingestion import daum, lihkg, naver, telegram
from src.ingestion.apify_client import ApifyRunner
from src.ingestion.base import POST_FIELDS, posts_to_dataframe
from src.ingestion.exa_news import ExaNewsCollector
from src.ingestion.exa_news import news_to_dataframe as exa_news_df
from src.ingestion.tavily_news import TavilyNewsCollector
from src.utils.logger import get_logger
from src.utils.storage import save_to_sqlite

log = get_logger(__name__)

SOCIAL_SOURCES_BY_EXCHANGE = {
    "HKEX": ["lihkg", "telegram"],
    "KRX": ["naver_stock_discussion", "daum_search"],
}
_ADAPTERS = {
    "lihkg": lihkg,
    "telegram": telegram,
    "naver_stock_discussion": naver,
    "daum_search": daum,
}


def _status_row(source: str, ticker: str, items: int, new_items: int, ok: bool, error: str | None) -> dict:
    return {
        "source": source,
        "ticker": ticker,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "new_items": new_items,
        "ok": ok,
        "error": error,
    }


def _save_raw(records: list[dict], source: str, ticker: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_ticker = ticker.replace(".", "_")
    path = RAW_DIR / f"social_{source}_{safe_ticker}_{stamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)


def collect_social(
    universe: Universe | None = None,
    days: int | None = None,
    runner: ApifyRunner | None = None,
) -> pd.DataFrame:
    universe = universe or load_universe()
    days = days or universe.social_lookback_days
    runner = runner or ApifyRunner()
    all_posts: list[dict] = []
    statuses: list[dict] = []

    for asset in universe.assets:
        sources = SOCIAL_SOURCES_BY_EXCHANGE.get(asset.exchange, [])
        for source in sources:
            adapter = _ADAPTERS[source]
            actor_id = universe.apify_actors.get(source, adapter.ACTOR_ID)
            try:
                posts = adapter.collect(asset, runner, days=days, actor_id=actor_id)
                error = None
                ok = True
            except Exception as exc:  # noqa: BLE001 — adapter isolation
                posts, ok, error = [], False, str(exc)
                log.error("social source %s failed for %s: %s", source, asset.ticker, exc)
            if posts:
                _save_raw(posts, source, asset.ticker)
            new = save_to_sqlite(
                pd.DataFrame(posts, columns=POST_FIELDS),
                "social_posts",
                dedup_keys=["source", "post_hash"],
            ) if posts else 0
            statuses.append(_status_row(source, asset.ticker, len(posts), new, ok, error))
            all_posts.extend(posts)

    save_to_sqlite(pd.DataFrame(statuses), "collection_status")
    df = posts_to_dataframe(all_posts)
    log.info("social collection complete: %d unique posts this run", len(df))
    return df


def collect_news(
    universe: Universe | None = None,
    days: int | None = None,
    exa: ExaNewsCollector | None = None,
    tavily: TavilyNewsCollector | None = None,
) -> pd.DataFrame:
    universe = universe or load_universe()
    days = days or universe.social_lookback_days
    exa = exa or ExaNewsCollector()
    tavily = tavily or TavilyNewsCollector()
    all_news: list[dict] = []
    statuses: list[dict] = []

    for asset in universe.assets:
        articles, ok, error = [], True, None
        try:
            if exa.available:
                articles = exa.collect(asset, days=days)
            else:
                error = "EXA_API_KEY not set"
                ok = False
        except Exception as exc:  # noqa: BLE001
            ok, error = False, str(exc)
            log.error("exa news failed for %s: %s", asset.ticker, exc)

        if not articles and tavily.available:
            try:
                fallback = tavily.collect(asset, days=days)
                log.info("tavily fallback: %d articles for %s", len(fallback), asset.ticker)
                articles.extend(fallback)
            except Exception as exc:  # noqa: BLE001
                log.error("tavily fallback failed for %s: %s", asset.ticker, exc)

        if articles:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = RAW_DIR / f"news_{asset.ticker.replace('.', '_')}_{stamp}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(articles, fh, ensure_ascii=False, indent=1)
            new = save_to_sqlite(
                exa_news_df(articles), "news_articles", dedup_keys=["source", "url"]
            )
        else:
            new = 0
        statuses.append(_status_row("news", asset.ticker, len(articles), new, ok, error))
        all_news.extend(articles)

    save_to_sqlite(pd.DataFrame(statuses), "collection_status")
    df = exa_news_df(all_news)
    log.info("news collection complete: %d unique articles this run", len(df))
    return df


def collect_all(universe: Universe | None = None, days: int | None = None) -> dict[str, pd.DataFrame]:
    from src.ingestion.prices import collect_prices

    universe = universe or load_universe()
    prices = collect_prices(universe)
    return {
        "prices": pd.concat(prices.values(), ignore_index=True) if prices else pd.DataFrame(),
        "social": collect_social(universe, days),
        "news": collect_news(universe, days),
    }
