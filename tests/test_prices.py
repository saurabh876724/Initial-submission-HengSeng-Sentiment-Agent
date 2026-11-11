from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion import prices as prices_mod


def fake_ohlcv(n=30, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Adj Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=idx,
    )


def test_fetch_history_normalizes_columns():
    with patch.object(prices_mod.yf, "download", return_value=fake_ohlcv()):
        df = prices_mod.fetch_history("0700.HK", years=3)
    assert list(df.columns) == ["ticker", "date"] + prices_mod.REQUIRED_COLUMNS
    assert len(df) == 30
    assert df["ticker"].unique() == ["0700.HK"]
    assert df["date"].iloc[0] == "2026-01-01"


def test_fetch_history_handles_multiindex_columns():
    raw = fake_ohlcv(5)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["0700.HK"]])
    with patch.object(prices_mod.yf, "download", return_value=raw):
        df = prices_mod.fetch_history("0700.HK")
    assert len(df) == 5


def test_fetch_history_empty_on_no_data():
    with patch.object(prices_mod.yf, "download", return_value=pd.DataFrame()):
        assert prices_mod.fetch_history("BAD.SYMBOL").empty


def test_fetch_history_empty_on_exception():
    with patch.object(prices_mod.yf, "download", side_effect=RuntimeError("network down")):
        assert prices_mod.fetch_history("0700.HK").empty


def test_collect_prices_persists_and_continues_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(prices_mod, "PRICES_DIR", tmp_path)

    saved_tables = []
    monkeypatch.setattr(
        prices_mod, "save_to_sqlite", lambda df, table, **kw: saved_tables.append(table) or len(df)
    )

    class TinyAsset:
        def __init__(self, ticker):
            self.ticker = ticker

    class TinyUniverse:
        history_years = 1
        assets = [TinyAsset("0700.HK"), TinyAsset("BAD.SYMBOL")]

    def fake_download(ticker, **kwargs):
        if ticker == "BAD.SYMBOL":
            return pd.DataFrame()
        return fake_ohlcv(10)

    with patch.object(prices_mod.yf, "download", side_effect=fake_download):
        results = prices_mod.collect_prices(TinyUniverse())

    assert set(results) == {"0700.HK"}
    assert (tmp_path / "0700_HK.parquet").exists()
    assert saved_tables == ["prices"]


@pytest.mark.network
def test_real_yfinance_single_ticker():
    """Live check against Yahoo Finance. Run with: pytest -m network"""
    df = prices_mod.fetch_history("0700.HK", years=1)
    assert not df.empty
    assert df["close"].iloc[-1] > 0
