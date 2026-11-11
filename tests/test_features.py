import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FEATURE_COLS,
    TARGET_COL,
    add_technical_features,
    build_dataset,
    compute_atr,
    compute_rsi,
    daily_sentiment,
    split_dataset,
)


def synth_prices(n=120, start_price=100.0, seed=7, ticker="0700.HK"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.015, n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates.strftime("%Y-%m-%d"),
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        }
    )


def test_rsi_bounds_and_uptrend():
    up = pd.Series(np.linspace(100, 200, 60))
    rsi = compute_rsi(up)
    assert rsi.between(0, 100).all()
    assert rsi.iloc[-1] > 70
    down = pd.Series(np.linspace(200, 100, 60))
    assert compute_rsi(down).iloc[-1] < 30


def test_atr_positive():
    p = synth_prices(60)
    atr = compute_atr(p["high"], p["low"], p["close"])
    assert (atr.dropna() > 0).all()


def test_target_is_next_close():
    df = build_dataset(synth_prices(120))
    closes = df["close"].tolist()
    for i in range(len(df) - 1):
        assert df.loc[i, TARGET_COL] == pytest.approx(closes[i + 1])
    assert pd.isna(df.loc[len(df) - 1, TARGET_COL])


def test_no_lookahead_future_changes_do_not_affect_past_features():
    base = synth_prices(120, seed=3)
    full = add_technical_features(base)
    truncated = add_technical_features(base.iloc[:80].copy())
    # perturb the "future" portion of the full series
    perturbed = base.copy()
    perturbed.loc[perturbed.index >= 80, "close"] *= 1.5
    perturbed_feats = add_technical_features(perturbed)
    cols = ["ret_1d", "sma_20", "rsi_14", "macd", "volatility_10d"]
    pd.testing.assert_frame_equal(
        truncated[cols].iloc[60:80].reset_index(drop=True),
        perturbed_feats[cols].iloc[60:80].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        full[cols].iloc[60:80].reset_index(drop=True),
        truncated[cols].iloc[60:80].reset_index(drop=True),
    )


def test_sentiment_merge_and_defaults():
    prices = synth_prices(120)
    session = prices["date"].iloc[-5]
    posts = pd.DataFrame(
        [
            {"ticker": "0700.HK", "trading_date": session, "sentiment_score": 0.6, "likes": 5, "replies": 1, "reposts": 0},
            {"ticker": "0700.HK", "trading_date": session, "sentiment_score": -0.2, "likes": 0, "replies": 0, "reposts": 0},
        ]
    )
    df = build_dataset(prices, social_posts=posts)
    row = df[df["date"] == session].iloc[0]
    assert row["social_sentiment"] == pytest.approx(0.2)
    assert row["social_mentions"] == 2
    assert row["has_sentiment"] == 1
    other = df[df["date"] != session].iloc[-1]
    assert other["social_sentiment"] == 0.0
    assert other["has_sentiment"] == 0


def test_daily_sentiment_empty_inputs():
    out = daily_sentiment(pd.DataFrame())
    assert out.empty


def test_split_dataset_shapes():
    df = build_dataset(synth_prices(120))
    X, y, inference_row = split_dataset(df)
    assert list(X.columns) == FEATURE_COLS
    assert y.notna().all()
    assert len(X) == len(y)
    assert inference_row is not None and len(inference_row) == 1
    # training rows exclude the final inference row
    assert len(X) == len(df) - 1


def test_dataset_without_sentiment_still_builds():
    df = build_dataset(synth_prices(120))
    assert not df.empty
    assert (df["has_sentiment"] == 0).all()
    assert df[TARGET_COL].notna().sum() == len(df) - 1
