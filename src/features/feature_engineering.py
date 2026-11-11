"""Time-aware feature engineering.

Leakage rules (enforced and tested):
- Every rolling/lagged feature at session D uses only data from sessions <= D.
- Sentiment features for session D come from posts attributed to D by
  `timestamp_alignment` (i.e. published before D's close) — never later.
- The regression target is the NEXT session's close (D+1), created with
  shift(-1); the most recent row therefore has no target and is used for
  inference only.
- When no sentiment exists for a session, features default to neutral (0
  score, 0 mentions) and `has_sentiment` is False. We never fabricate
  sentiment history; models are also trained price-only for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

PRICE_FEATURES = [
    "close_prev",
    "ret_1d",
    "ret_1d_lag1",
    "ret_1d_lag2",
    "ret_1d_lag3",
    "ret_5d",
    "volatility_10d",
    "sma_20",
    "sma_50",
    "sma_ratio_20_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "volume_ratio_20d",
]
SENTIMENT_FEATURES = [
    "social_sentiment",
    "social_mentions",
    "engagement_total",
    "sentiment_momentum",
    "news_sentiment",
    "news_mentions",
    "has_sentiment",
]
FEATURE_COLS = PRICE_FEATURES + SENTIMENT_FEATURES
TARGET_COL = "next_close"


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # pure-gain window -> 100; flat window -> 50; warmup NaN -> 50
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi.fillna(50.0).clip(0, 100)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: one ticker's bars with columns date, open, high, low, close, volume."""
    df = prices.sort_values("date").reset_index(drop=True).copy()
    close, high, low = df["close"], df["high"], df["low"]

    df["close_prev"] = close.shift(1)
    df["ret_1d"] = close.pct_change()
    for lag in (1, 2, 3):
        df[f"ret_1d_lag{lag}"] = df["ret_1d"].shift(lag)
    df["ret_5d"] = close.pct_change(5)
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio_20_50"] = df["sma_20"] / df["sma_50"]
    df["rsi_14"] = compute_rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["atr_14"] = compute_atr(high, low, close)
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


def daily_sentiment(posts: pd.DataFrame, score_col: str = "sentiment_score") -> pd.DataFrame:
    """Aggregate annotated posts/news to one row per (ticker, trading_date)."""
    if posts is None or posts.empty or "trading_date" not in posts.columns:
        return pd.DataFrame(columns=["ticker", "trading_date", "score", "mentions", "engagement"])
    df = posts.dropna(subset=["trading_date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "trading_date", "score", "mentions", "engagement"])
    df["_eng"] = df.get("likes", 0).fillna(0) + df.get("replies", 0).fillna(0) + df.get("reposts", 0).fillna(0)
    g = df.groupby(["ticker", "trading_date"]).agg(
        score=(score_col, "mean"),
        mentions=(score_col, "size"),
        engagement=("_eng", "sum"),
    ).reset_index()
    return g


def build_dataset(
    prices: pd.DataFrame,
    social_posts: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix + target for ONE ticker.

    Returns all sessions with complete features. The final row has a NaN
    target (inference row). `date` stays as the session key.
    """
    df = add_technical_features(prices)

    social = daily_sentiment(social_posts)
    news_daily = daily_sentiment(news)

    if not social.empty:
        social = social.rename(
            columns={"score": "social_sentiment", "mentions": "social_mentions", "engagement": "engagement_total", "trading_date": "date"}
        )
        df = df.merge(social[["ticker", "date", "social_sentiment", "social_mentions", "engagement_total"]], on=["ticker", "date"], how="left")
    else:
        df["social_sentiment"] = 0.0
        df["social_mentions"] = 0
        df["engagement_total"] = 0.0

    if not news_daily.empty:
        news_daily = news_daily.rename(
            columns={"score": "news_sentiment", "mentions": "news_mentions", "trading_date": "date"}
        )
        df = df.merge(news_daily[["ticker", "date", "news_sentiment", "news_mentions"]], on=["ticker", "date"], how="left")
    else:
        df["news_sentiment"] = 0.0
        df["news_mentions"] = 0

    has_social = df["social_mentions"].fillna(0) > 0
    has_news = df["news_mentions"].fillna(0) > 0
    df["has_sentiment"] = (has_social | has_news).astype(int)
    df["social_sentiment"] = df["social_sentiment"].fillna(0.0)
    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
    df["social_mentions"] = df["social_mentions"].fillna(0).astype(int)
    df["news_mentions"] = df["news_mentions"].fillna(0).astype(int)
    df["engagement_total"] = df["engagement_total"].fillna(0.0)
    df["sentiment_momentum"] = df["social_sentiment"].diff().fillna(0.0)

    df[TARGET_COL] = df["close"].shift(-1)

    # keep only rows where all features are computable
    complete = df.dropna(subset=PRICE_FEATURES).index
    df = df.loc[complete].reset_index(drop=True)
    log.info(
        "dataset for %s: %d rows (%d with sentiment), target coverage %.1f%%",
        df["ticker"].iloc[0] if len(df) else "?",
        len(df),
        int(df["has_sentiment"].sum()),
        100 * df[TARGET_COL].notna().mean() if len(df) else 0.0,
    )
    return df


def split_dataset(df: pd.DataFrame, feature_set: list[str] | None = None):
    """Return (X, y, inference_row) dropping the target-less final row from training data."""
    feature_set = feature_set or FEATURE_COLS
    train_pool = df[df[TARGET_COL].notna()]
    X = train_pool[feature_set]
    y = train_pool[TARGET_COL]
    inference_row = df.iloc[[-1]][feature_set] if len(df) else None
    return X, y, inference_row
