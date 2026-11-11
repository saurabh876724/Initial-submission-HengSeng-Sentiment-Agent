"""Financial sentiment analysis with FinBERT (ProsusAI/finbert).

Scores each post/article:
- label: positive / negative / neutral
- score: p(positive) - p(negative), normalized to [-1, +1]

Aggregations produced per asset:
- average sentiment, pos/neg/neu percentages, total mentions
- engagement-weighted sentiment (weights are influence hints, NOT truth:
  a liked post is not automatically a reliable post — spec §7)
- source-specific sentiment and daily sentiment trend

The model pipeline is injectable so unit tests never download weights.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import SENTIMENT_MODEL
from src.utils.logger import get_logger

log = get_logger(__name__)

LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
MAX_CHARS = 1500  # FinBERT truncates at 512 tokens anyway


class SentimentAnalyzer:
    def __init__(self, model_name: str | None = None, pipeline: Any = None):
        self.model_name = model_name or SENTIMENT_MODEL
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading FinBERT (%s) on %s — first run downloads ~440MB", self.model_name, device)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if device == "cuda" else -1,
                top_k=None,
            )
        return self._pipeline

    def score_text(self, text: str) -> tuple[str, float]:
        """Return (label, score in [-1, 1]) for one text."""
        if not isinstance(text, str) or not text.strip():
            return "neutral", 0.0
        results = self.pipeline(text[:MAX_CHARS])
        probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for item in results[0] if isinstance(results[0], list) else results:
            label = str(item.get("label", "")).lower()
            if label in probs:
                probs[label] = float(item.get("score", 0.0))
        score = probs["positive"] - probs["negative"]
        label = max(probs, key=probs.get)
        return label, round(score, 4)

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        return [self.score_text(t) for t in texts]

    def annotate(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Add sentiment_label + sentiment_score columns."""
        out = df.copy()
        if out.empty:
            out["sentiment_label"] = pd.Series(dtype="object")
            out["sentiment_score"] = pd.Series(dtype="float64")
            return out
        scored = [self.score_text(t) for t in out[text_col].fillna("").astype(str)]
        out["sentiment_label"] = [s[0] for s in scored]
        out["sentiment_score"] = [s[1] for s in scored]
        return out


def engagement_weight(row: pd.Series) -> float:
    """Weight = 1 + log-scaled engagement + small author-influence term.

    Deliberately sub-linear (log) so viral posts cannot dominate the
    aggregate, and documented as a heuristic — engagement is not proof
    of reliability.
    """
    eng = sum(
        float(row.get(c) or 0.0)
        for c in ("likes", "replies", "reposts")
        if pd.notna(row.get(c))
    )
    followers = row.get("author_followers")
    influence = 0.5 * np.log1p(float(followers)) if followers is not None and pd.notna(followers) else 0.0
    return 1.0 + np.log1p(eng) + influence


def aggregate_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker sentiment summary from annotated posts."""
    if posts.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "mentions", "avg_sentiment", "pos_pct", "neg_pct", "neu_pct",
                "engagement_weighted_sentiment",
            ]
        )
    rows = []
    df = posts.copy()
    df["weight"] = df.apply(engagement_weight, axis=1)
    for ticker, g in df.groupby("ticker"):
        counts = g["sentiment_label"].value_counts()
        n = len(g)
        w_sum = g["weight"].sum()
        rows.append(
            {
                "ticker": ticker,
                "mentions": n,
                "avg_sentiment": round(float(g["sentiment_score"].mean()), 4),
                "pos_pct": round(100 * counts.get("positive", 0) / n, 2),
                "neg_pct": round(100 * counts.get("negative", 0) / n, 2),
                "neu_pct": round(100 * counts.get("neutral", 0) / n, 2),
                "engagement_weighted_sentiment": round(float((g["weight"] * g["sentiment_score"]).sum() / w_sum), 4) if w_sum else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sentiment_by_source(posts: pd.DataFrame) -> pd.DataFrame:
    """Source-specific sentiment kept available for inspection (spec §7)."""
    if posts.empty:
        return pd.DataFrame(columns=["source", "ticker", "mentions", "avg_sentiment"])
    g = posts.groupby(["source", "ticker"]).agg(
        mentions=("sentiment_score", "size"),
        avg_sentiment=("sentiment_score", "mean"),
    ).reset_index()
    g["avg_sentiment"] = g["avg_sentiment"].round(4)
    return g


def sentiment_trend(posts: pd.DataFrame) -> pd.DataFrame:
    """Daily sentiment trend per ticker, keyed on trading_date when present."""
    if posts.empty:
        return pd.DataFrame(columns=["ticker", "date", "mentions", "avg_sentiment"])
    df = posts.copy()
    date_col = "trading_date" if "trading_date" in df.columns else "posted_at_utc"
    df = df.dropna(subset=[date_col])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "mentions", "avg_sentiment"])
    df["date"] = df[date_col].astype(str).str[:10]
    g = df.groupby(["ticker", "date"]).agg(
        mentions=("sentiment_score", "size"),
        avg_sentiment=("sentiment_score", "mean"),
    ).reset_index()
    g["avg_sentiment"] = g["avg_sentiment"].round(4)
    return g
