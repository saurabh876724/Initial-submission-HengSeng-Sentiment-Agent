import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")

ANALYSIS_ROWS = [
    {"ticker": "0700.HK", "company_name": "Tencent", "exchange": "HKEX", "price": 417.8,
     "price_date": "2026-11-10", "social_sentiment": 0.35, "social_mentions": 40,
     "pos_pct": 60, "neg_pct": 20, "news_sentiment": 0.2, "direction": "bullish",
     "composite_signal": 0.4, "predicted_next_close": 420.1, "predicted_pct_change": 0.55,
     "prediction_model": "xgboost", "disclaimer": "x", "sma_20": 410.0, "sma_50": 400.0,
     "rsi_14": 58.0, "macd_hist": 1.2, "atr_14": 6.5, "volatility_20d_annualized": 0.32,
     "pivot": 418.0, "resistance_1": 425.0, "resistance_2": 432.0, "support_1": 411.0,
     "support_2": 404.0, "swing_high_20d": 430.0, "swing_low_20d": 395.0},
    {"ticker": "005930.KS", "company_name": "Samsung", "exchange": "KRX", "price": 198900.0,
     "price_date": "2026-11-10", "social_sentiment": -0.1, "social_mentions": 12,
     "pos_pct": 30, "neg_pct": 45, "news_sentiment": 0.05, "direction": "sideways",
     "composite_signal": 0.02, "predicted_next_close": 199500.0, "predicted_pct_change": 0.3,
     "prediction_model": "random_forest", "disclaimer": "x", "sma_20": 197000.0,
     "sma_50": 195000.0, "rsi_14": 51.0, "macd_hist": 100.0, "atr_14": 3200.0,
     "volatility_20d_annualized": 0.28, "pivot": 199000.0, "resistance_1": 202000.0,
     "resistance_2": 205000.0, "support_1": 196000.0, "support_2": 193000.0,
     "swing_high_20d": 205000.0, "swing_low_20d": 190000.0},
]


def prices_df():
    dates = pd.date_range("2026-06-01", periods=100, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "ticker": "0700.HK", "date": dates,
        "open": 400.0, "high": 420.0, "low": 390.0,
        "close": [400 + i * 0.2 for i in range(100)],
        "adj_close": 400.0, "volume": 1_000_000,
    })


def posts_df():
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame([
        {"source": "lihkg", "ticker": "0700.HK", "text": "rally continues",
         "posted_at_utc": (now - pd.Timedelta(days=1)).isoformat(),
         "sentiment_label": "positive", "sentiment_score": 0.6,
         "likes": 5, "replies": 1, "reposts": 0, "views": None,
         "url": "https://l/1", "trading_date": "2026-11-10"},
    ])


@pytest.fixture
def patch_data(monkeypatch, tmp_path):
    import streamlit as st

    import config.settings as settings
    import src.utils.storage as storage

    st.cache_data.clear()

    csv = tmp_path / "stock_analysis.csv"
    pd.DataFrame(ANALYSIS_ROWS).to_csv(csv, index=False)
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)

    preds_dir = tmp_path / "models"
    preds_dir.mkdir()
    (preds_dir / "predictions_latest.json").write_text(
        json.dumps([{
            "ticker": "0700.HK", "as_of_date": "2026-11-10", "last_close": 417.8,
            "predicted_next_close": 420.1, "predicted_pct_change": 0.55,
            "interval_95_low": 410.0, "interval_95_high": 430.0, "model": "xgboost",
        }]), encoding="utf-8")
    monkeypatch.setattr(settings, "MODELS_DIR", preds_dir)

    def fake_sqlite(table, where=None, params=(), **kw):
        return {
            "prices": prices_df(),
            "posts_annotated": posts_df(),
            "news_annotated": pd.DataFrame(),
            "collection_status": pd.DataFrame([{
                "source": "lihkg", "ticker": "0700.HK", "finished_at": "2026-11-11T00:00:00+00:00",
                "items": 10, "new_items": 8, "ok": 1, "error": None,
            }]),
        }.get(table, pd.DataFrame())

    monkeypatch.setattr(storage, "load_from_sqlite", fake_sqlite)
    monkeypatch.setattr(storage, "load_parquet", lambda path: None)


def test_dashboard_empty_state_shows_instructions(monkeypatch, tmp_path):
    import streamlit as st

    import config.settings as settings
    import src.utils.storage as storage

    st.cache_data.clear()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(settings, "PROCESSED_DIR", empty_dir)
    monkeypatch.setattr(settings, "MODELS_DIR", empty_dir / "models")
    monkeypatch.setattr(storage, "load_from_sqlite", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(storage, "load_parquet", lambda *a, **kw: None)

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("python main.py" in w.value for w in at.warning)


def test_dashboard_renders_with_data(patch_data):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    # metrics rendered on overview
    assert len(at.metric) > 0
