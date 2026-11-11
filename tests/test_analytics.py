import pandas as pd
import pytest

from config.settings import Asset
from src.analytics import stock_analysis as sa
from tests.test_features import synth_prices

HK = Asset("0700.HK", "Tencent", "騰訊", "HKEX", None, ["Tencent"])


def test_pivot_levels_math():
    lv = sa.pivot_levels(high=110, low=90, close=100)
    assert lv["pivot"] == 100
    assert lv["resistance_1"] == 110
    assert lv["support_1"] == 90
    assert lv["resistance_2"] == 120
    assert lv["support_2"] == 80


def test_technical_snapshot_fields():
    tech = sa.technical_snapshot(synth_prices(120))
    for key in ("price", "sma_20", "sma_50", "rsi_14", "macd_hist", "atr_14",
                "volatility_20d_annualized", "pivot", "resistance_1", "support_1",
                "swing_high_20d", "swing_low_20d"):
        assert key in tech
    assert 0 <= tech["rsi_14"] <= 100
    assert tech["swing_high_20d"] >= tech["swing_low_20d"]


def test_direction_bullish_bearish_sideways():
    up_tech = {"sma_20": 110, "sma_50": 100, "macd_hist": 0.5}
    down_tech = {"sma_20": 90, "sma_50": 100, "macd_hist": -0.5}
    flat_tech = {"sma_20": 100, "sma_50": 100, "macd_hist": 0.0}
    assert sa.direction_from(up_tech, 0.6, 0.4)[0] == "bullish"
    assert sa.direction_from(down_tech, -0.6, -0.4)[0] == "bearish"
    assert sa.direction_from(flat_tech, 0.0, 0.0)[0] == "sideways"
    label, composite = sa.direction_from(up_tech, 0.6, 0.4)
    assert -1 <= composite <= 1


def test_analyze_asset_full_record():
    prices = synth_prices(120)
    social = pd.DataFrame([{"ticker": "0700.HK", "mentions": 25, "avg_sentiment": 0.3,
                            "pos_pct": 60.0, "neg_pct": 20.0, "neu_pct": 20.0,
                            "engagement_weighted_sentiment": 0.35}])
    news = pd.DataFrame([{"ticker": "0700.HK", "mentions": 8, "avg_sentiment": 0.2,
                          "pos_pct": 50, "neg_pct": 25, "neu_pct": 25,
                          "engagement_weighted_sentiment": 0.2}])
    pred = {"predicted_next_close": 105.5, "predicted_pct_change": 0.7, "model": "xgboost"}
    row = sa.analyze_asset(HK, prices, social, news, pred)
    assert row["ticker"] == "0700.HK"
    assert row["social_mentions"] == 25
    assert row["social_sentiment"] == 0.35
    assert row["predicted_next_close"] == 105.5
    assert row["disclaimer"] == sa.DISCLAIMER
    assert row["direction"] in ("bullish", "bearish", "sideways")


def test_analyze_asset_without_sentiment_or_prediction():
    row = sa.analyze_asset(HK, synth_prices(120), pd.DataFrame(), pd.DataFrame(), None)
    assert row["social_mentions"] == 0
    assert row["social_sentiment"] == 0.0
    assert row["predicted_next_close"] is None


def test_analyze_universe_persists_csv(monkeypatch, tmp_path):
    prices = synth_prices(120)
    monkeypatch.setattr(sa, "load_from_sqlite", lambda table, **kw: prices)
    monkeypatch.setattr(sa, "load_parquet", lambda path: None)
    monkeypatch.setattr(sa, "_load_predictions", lambda: {})
    monkeypatch.setattr(sa, "PROCESSED_DIR", tmp_path)
    out = sa.analyze_universe()
    assert len(out) == 1
    assert out["ticker"].iloc[0] == "0700.HK"
    assert (tmp_path / "stock_analysis.csv").exists()
