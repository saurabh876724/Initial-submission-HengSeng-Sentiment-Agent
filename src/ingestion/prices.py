"""Historical OHLCV price ingestion via yfinance.

Stores one parquet file per ticker (data/historical_prices/) and mirrors
everything into the SQLite `prices` table.

Honesty note: yfinance provides real end-of-day bars. If a symbol is
delisted/unavailable the failure is logged and the rest continue —
no data is ever fabricated to fill gaps.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from config.settings import PRICES_DIR, Universe, load_universe
from src.utils.logger import get_logger
from src.utils.storage import save_parquet, save_to_sqlite

log = get_logger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def fetch_history(ticker: str, years: int = 3) -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker. Returns empty df on failure."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network errors, bad symbols
        log.error("price fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        log.warning("no price data returned for %s (symbol may be invalid)", ticker)
        return pd.DataFrame()

    # yfinance may return MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date.astype(str)

    adj = df.get("adj close", df.get("adj_close"))
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": df["date"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "adj_close": (adj if adj is not None else df["close"]).astype(float),
            "volume": df["volume"].astype("int64"),
        }
    )
    out = out.dropna(subset=["close"])
    log.info("fetched %d price bars for %s (%s .. %s)", len(out), ticker, out["date"].min(), out["date"].max())
    return out


def collect_prices(universe: Universe | None = None) -> dict[str, pd.DataFrame]:
    """Fetch, persist (parquet + sqlite) and return prices for all assets."""
    universe = universe or load_universe()
    results: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for asset in universe.assets:
        df = fetch_history(asset.ticker, universe.history_years)
        if df.empty:
            failures.append(asset.ticker)
            continue
        save_parquet(df, PRICES_DIR / f"{asset.ticker.replace('.', '_')}.parquet")
        save_to_sqlite(df, "prices", dedup_keys=["ticker", "date"])
        results[asset.ticker] = df
    if failures:
        log.warning("price collection failed for: %s", ", ".join(failures))
    return results


if __name__ == "__main__":
    collect_prices()
