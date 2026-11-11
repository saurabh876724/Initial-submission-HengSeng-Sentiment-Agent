"""Per-asset analytics: price state, technicals, sentiment, direction,
support/resistance, and the model's next-period prediction.

Support & resistance method (transparent, documented):
  Primary  — classic pivot points from the latest session:
             P = (H + L + C) / 3;  R1 = 2P - L;  S1 = 2P - H;
             R2 = P + (H - L);     S2 = P - (H - L)
  Secondary — 20-session swing high/low as confirmation levels.

Direction rule (transparent, documented):
  technical = sign(SMA20 - SMA50) + sign(MACD histogram), scaled to [-1, 1]
  sentiment = engagement-weighted social score blended with news score
  composite = 0.5 * technical + 0.5 * sentiment
  bullish > +0.15, bearish < -0.15, else sideways.

Everything here is an analytical signal for research/evaluation —
NOT financial advice. The output row carries that disclaimer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, PROCESSED_DIR, Universe, load_universe
from src.features.feature_engineering import compute_atr, compute_rsi
from src.utils.logger import get_logger
from src.utils.storage import load_from_sqlite, load_parquet

log = get_logger(__name__)

DISCLAIMER = "Analytical signal for research purposes only — not financial advice."
DIRECTION_THRESHOLD = 0.15


def pivot_levels(high: float, low: float, close: float) -> dict:
    p = (high + low + close) / 3
    return {
        "pivot": round(p, 4),
        "resistance_1": round(2 * p - low, 4),
        "resistance_2": round(p + (high - low), 4),
        "support_1": round(2 * p - high, 4),
        "support_2": round(p - (high - low), 4),
    }


def technical_snapshot(prices: pd.DataFrame) -> dict:
    """Latest-session technical state from a ticker's price history."""
    df = prices.sort_values("date").reset_index(drop=True)
    close, high, low = df["close"], df["high"], df["low"]
    last = df.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26)
    macd_hist = (macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1]
    rets = close.pct_change()
    levels = pivot_levels(float(last["high"]), float(last["low"]), float(last["close"]))
    return {
        "date": str(last["date"]),
        "price": float(last["close"]),
        "sma_20": round(float(sma20), 4),
        "sma_50": round(float(sma50), 4),
        "rsi_14": round(float(compute_rsi(close).iloc[-1]), 2),
        "macd_hist": round(float(macd_hist), 4),
        "atr_14": round(float(compute_atr(high, low, close).iloc[-1]), 4),
        "volatility_20d_annualized": round(float(rets.tail(20).std() * np.sqrt(252)), 4),
        "swing_high_20d": float(high.tail(20).max()),
        "swing_low_20d": float(low.tail(20).min()),
        **levels,
    }


def direction_from(technical: dict, social_score: float, news_score: float) -> tuple[str, float]:
    tech = (np.sign(technical["sma_20"] - technical["sma_50"]) + np.sign(technical["macd_hist"])) / 2
    sentiment = 0.6 * social_score + 0.4 * news_score
    composite = 0.5 * tech + 0.5 * sentiment
    if composite > DIRECTION_THRESHOLD:
        label = "bullish"
    elif composite < -DIRECTION_THRESHOLD:
        label = "bearish"
    else:
        label = "sideways"
    return label, round(float(composite), 4)


def analyze_asset(
    asset,
    prices: pd.DataFrame,
    social_agg: pd.DataFrame,
    news_agg: pd.DataFrame,
    prediction: dict | None = None,
) -> dict:
    tech = technical_snapshot(prices)
    s_row = social_agg[social_agg["ticker"] == asset.ticker] if not social_agg.empty else pd.DataFrame()
    n_row = news_agg[news_agg["ticker"] == asset.ticker] if not news_agg.empty else pd.DataFrame()
    social_score = float(s_row["engagement_weighted_sentiment"].iloc[0]) if not s_row.empty else 0.0
    mentions = int(s_row["mentions"].iloc[0]) if not s_row.empty else 0
    news_score = float(n_row["avg_sentiment"].iloc[0]) if not n_row.empty else 0.0
    direction, composite = direction_from(tech, social_score, news_score)
    return {
        "ticker": asset.ticker,
        "company_name": asset.company_name,
        "exchange": asset.exchange,
        "price": tech["price"],
        "price_date": tech["date"],
        "social_sentiment": round(social_score, 4),
        "social_mentions": mentions,
        "pos_pct": float(s_row["pos_pct"].iloc[0]) if not s_row.empty else None,
        "neg_pct": float(s_row["neg_pct"].iloc[0]) if not s_row.empty else None,
        "news_sentiment": round(news_score, 4),
        "direction": direction,
        "composite_signal": composite,
        "predicted_next_close": prediction.get("predicted_next_close") if prediction else None,
        "predicted_pct_change": prediction.get("predicted_pct_change") if prediction else None,
        "prediction_model": prediction.get("model") if prediction else None,
        "disclaimer": DISCLAIMER,
        **{k: tech[k] for k in (
            "sma_20", "sma_50", "rsi_14", "macd_hist", "atr_14",
            "volatility_20d_annualized", "pivot", "resistance_1", "resistance_2",
            "support_1", "support_2", "swing_high_20d", "swing_low_20d",
        )},
    }


def _load_predictions() -> dict[str, dict]:
    path = MODELS_DIR / "predictions_latest.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return {r["ticker"]: r for r in json.load(fh)}
    return {}


def analyze_universe(universe: Universe | None = None) -> pd.DataFrame:
    universe = universe or load_universe()
    prices = load_from_sqlite("prices")
    social_agg = load_parquet(PROCESSED_DIR / "sentiment_social_agg.parquet")
    if social_agg is None:
        social_agg = pd.DataFrame()
    news_agg = load_parquet(PROCESSED_DIR / "sentiment_news_agg.parquet")
    if news_agg is None:
        news_agg = pd.DataFrame()
    predictions = _load_predictions()

    rows = []
    for asset in universe.assets:
        p = prices[prices["ticker"] == asset.ticker] if not prices.empty else pd.DataFrame()
        if p.empty:
            log.warning("analytics: no prices for %s — skipped", asset.ticker)
            continue
        rows.append(analyze_asset(asset, p, social_agg, news_agg, predictions.get(asset.ticker)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(PROCESSED_DIR / "stock_analysis.csv", index=False)
    return out
