"""Sentiment stage orchestration: clean -> align -> FinBERT annotate -> aggregate.

Reads raw collected data from SQLite (`social_posts`, `news_articles`),
writes derived tables (`posts_annotated`, `news_annotated`) and parquet
aggregates consumed by features, analytics, dashboard and reports.

Derived tables are fully recomputed and replaced each run — they are
projections of raw data, never the source of truth.
"""
from __future__ import annotations

import pandas as pd

from config.settings import PROCESSED_DIR, Universe, load_universe
from src.preprocessing.cleaning import clean_news, clean_posts
from src.preprocessing.timestamp_alignment import align_posts
from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    aggregate_sentiment,
    sentiment_by_source,
    sentiment_trend,
)
from src.utils.logger import get_logger
from src.utils.storage import load_from_sqlite, replace_table, save_parquet

log = get_logger(__name__)


def annotate_all(
    universe: Universe | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> dict[str, pd.DataFrame]:
    universe = universe or load_universe()
    analyzer = analyzer or SentimentAnalyzer()
    lookback = universe.social_lookback_days

    posts = clean_posts(load_from_sqlite("social_posts"), lookback_days=lookback)
    posts = align_posts(posts, universe)
    if not posts.empty:
        posts = analyzer.annotate(posts, text_col="text")
    replace_table(posts, "posts_annotated")

    news = clean_news(load_from_sqlite("news_articles"))
    if not news.empty:
        news["text"] = news["title"].fillna("") + ". " + news["content"].fillna("")
        news = align_posts(news.rename(columns={"published_at_utc": "posted_at_utc"}), universe)
        news = analyzer.annotate(news, text_col="text")
    replace_table(news, "news_annotated")

    social_agg = aggregate_sentiment(posts)
    news_agg = aggregate_sentiment(news) if not news.empty else pd.DataFrame()
    by_source = sentiment_by_source(posts)
    trend = sentiment_trend(posts)

    save_parquet(social_agg, PROCESSED_DIR / "sentiment_social_agg.parquet")
    save_parquet(news_agg, PROCESSED_DIR / "sentiment_news_agg.parquet")
    save_parquet(by_source, PROCESSED_DIR / "sentiment_by_source.parquet")
    save_parquet(trend, PROCESSED_DIR / "sentiment_trend.parquet")

    log.info(
        "sentiment stage: %d posts (avg score across assets: %s), %d news articles",
        len(posts),
        social_agg["avg_sentiment"].mean().round(4) if not social_agg.empty else "n/a",
        len(news),
    )
    return {
        "posts": posts,
        "news": news,
        "social_agg": social_agg,
        "news_agg": news_agg,
        "by_source": by_source,
        "trend": trend,
    }
