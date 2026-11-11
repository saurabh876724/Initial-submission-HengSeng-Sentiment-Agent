"""Align social/news timestamps to exchange trading sessions (UTC-safe).

Attribution rule (documented to prevent look-ahead bias):
    A post published on trading day D *before* the market close is fully
    known at D's close, so it informs the D -> D+1 prediction and is
    attributed to session D. Posts published after the close, or on
    weekends/holidays, are attributed to the NEXT trading session.

Sentiment for session D therefore only ever uses information available
at D's close. Calendars come from the `exchange_calendars` package
(XHKG for HKEX, XKRX for KRX) which encode real holidays.
"""
from __future__ import annotations

from datetime import time

import pandas as pd

from config.settings import Universe, load_universe
from src.utils.logger import get_logger

log = get_logger(__name__)

CALENDAR_IDS = {"HKEX": "XHKG", "KRX": "XKRX"}
TIMEZONES = {"HKEX": "Asia/Hong_Kong", "KRX": "Asia/Seoul"}
# Local close times with a small buffer for auction/late prints
CLOSE_TIMES_LOCAL = {"HKEX": time(16, 10), "KRX": time(15, 40)}

_calendar_cache: dict[str, object] = {}


def get_calendar(exchange: str):
    cal_id = CALENDAR_IDS[exchange]
    if cal_id not in _calendar_cache:
        import exchange_calendars as xcals

        _calendar_cache[cal_id] = xcals.get_calendar(cal_id)
    return _calendar_cache[cal_id]


def trading_date_for(ts_utc: pd.Timestamp, exchange: str) -> pd.Timestamp | None:
    """Return the session date a UTC timestamp is attributed to (see module docstring)."""
    if ts_utc is None or pd.isna(ts_utc):
        return None
    cal = get_calendar(exchange)
    local = ts_utc.tz_convert(TIMEZONES[exchange])
    candidate = pd.Timestamp(local.date())

    before_close = local.time() < CLOSE_TIMES_LOCAL[exchange]
    if before_close and cal.is_session(candidate):
        return candidate
    if before_close:
        # weekend/holiday morning -> next session
        return cal.date_to_session(candidate, direction="next")
    # after close -> next session strictly after candidate
    nxt = candidate + pd.Timedelta(days=1)
    return cal.date_to_session(nxt, direction="next")


def align_posts(df: pd.DataFrame, universe: Universe | None = None) -> pd.DataFrame:
    """Add a `trading_date` column to a cleaned posts/news dataframe."""
    universe = universe or load_universe()
    if df.empty:
        out = df.copy()
        out["trading_date"] = pd.Series(dtype="object")
        return out
    exchange_of = {a.ticker: a.exchange for a in universe.assets}
    out = df.copy()
    dates = []
    for _, row in out.iterrows():
        exchange = exchange_of.get(row["ticker"])
        ts = row.get("posted_at_utc", row.get("published_at_utc"))
        if exchange is None or ts is None or pd.isna(ts):
            dates.append(None)
            continue
        session = trading_date_for(pd.Timestamp(ts).tz_convert("UTC"), exchange)
        dates.append(session.date().isoformat() if session is not None else None)
    out["trading_date"] = dates
    unresolved = out["trading_date"].isna().sum()
    if unresolved:
        log.warning("timestamp alignment: %d rows could not be aligned (missing timestamp)", unresolved)
    return out
