import pandas as pd
import pytest

from config.settings import Asset, Universe
from src.ingestion.base import make_post
from src.preprocessing import cleaning, timestamp_alignment as ta

HK = Asset("0700.HK", "Tencent", "騰訊", "HKEX", None, ["Tencent"])
KR = Asset("005930.KS", "Samsung", "삼성전자", "KRX", "005930", ["Samsung"])
UNI = Universe(10, 1, {}, [HK, KR], {})

# Reference sessions resolved dynamically (holiday-safe)
CAL_HK = ta.get_calendar("HKEX")
CAL_KR = ta.get_calendar("KRX")
MONDAY = CAL_HK.date_to_session(pd.Timestamp("2026-11-09"))
NEXT_AFTER_MONDAY = CAL_HK.next_session(MONDAY)


def posts_frame(rows):
    df = pd.DataFrame(rows)
    for col in ["likes", "replies", "reposts", "views", "author_followers"]:
        if col not in df.columns:
            df[col] = None
    return df


# ---------- clean_text ----------

def test_clean_text_strips_html_urls_whitespace():
    raw = "<p>Buy  $TENCENT</p>   https://t.co/abc\n\nboom  "
    assert cleaning.clean_text(raw) == "Buy $TENCENT boom"


def test_clean_text_handles_non_string():
    assert cleaning.clean_text(None) == ""
    assert cleaning.clean_text(123) == ""


def test_clean_text_keeps_url_when_asked():
    assert "https://x.com" in cleaning.clean_text("see https://x.com now", keep_urls=True)


# ---------- clean_posts ----------

def test_clean_posts_drops_empty_and_dedupes():
    p1 = make_post("lihkg", "0700.HK", "good news", "2026-11-10T02:00:00Z", "https://l/1")
    p2 = make_post("lihkg", "0700.HK", "good news", "2026-11-10T02:00:00Z", "https://l/1")  # dup
    p3 = {**p1, "text": "   ", "url": "https://l/9", "post_hash": "emptytext"}  # empty after clean
    df = posts_frame([p1, p2, p3])
    out = cleaning.clean_posts(df, lookback_days=10, now=pd.Timestamp("2026-11-11", tz="UTC"))
    assert len(out) == 1


def test_clean_posts_drops_stale_keeps_missing_timestamps():
    fresh = make_post("lihkg", "0700.HK", "fresh", "2026-11-10T02:00:00Z", "https://l/2")
    stale = make_post("lihkg", "0700.HK", "stale", "2026-10-01T02:00:00Z", "https://l/3")
    no_ts = make_post("lihkg", "0700.HK", "no timestamp", None, None)
    df = posts_frame([fresh, stale, no_ts])
    out = cleaning.clean_posts(df, lookback_days=10, now=pd.Timestamp("2026-11-11", tz="UTC"))
    assert len(out) == 2
    assert out.loc[out["text"] == "no timestamp", "timestamp_missing"].iloc[0]


def test_clean_posts_engagement_numeric():
    p = make_post("tg", "0700.HK", "x", "2026-11-10T02:00:00Z", "https://l/4", likes="42", views="bad")
    out = cleaning.clean_posts(posts_frame([p]), now=pd.Timestamp("2026-11-11", tz="UTC"))
    assert out["likes"].iloc[0] == 42
    assert pd.isna(out["views"].iloc[0])


def test_clean_news_dedupe_and_drop_empty():
    base = {"source": "exa", "ticker": "0700.HK", "published_at_utc": "2026-11-10T00:00:00Z", "collected_at": "x"}
    df = pd.DataFrame(
        [
            {**base, "title": "A", "content": "c", "url": "u1"},
            {**base, "title": "A copy", "content": "c", "url": "u1"},
            {**base, "title": "", "content": "", "url": "u2"},
        ]
    )
    out = cleaning.clean_news(df)
    assert len(out) == 1
    assert out["url"].iloc[0] == "u1"


# ---------- timestamp alignment ----------

def test_hk_morning_post_maps_to_same_session():
    ts = pd.Timestamp(f"{MONDAY.date()} 09:00", tz="Asia/Hong_Kong").tz_convert("UTC")
    assert ta.trading_date_for(ts, "HKEX") == MONDAY


def test_hk_after_close_maps_to_next_session():
    ts = pd.Timestamp(f"{MONDAY.date()} 17:00", tz="Asia/Hong_Kong").tz_convert("UTC")
    assert ta.trading_date_for(ts, "HKEX") == NEXT_AFTER_MONDAY


def test_weekend_maps_to_next_session():
    saturday = MONDAY - pd.Timedelta(days=2)
    while CAL_HK.is_session(saturday):
        saturday -= pd.Timedelta(days=1)
    ts = pd.Timestamp(f"{saturday.date()} 12:00", tz="Asia/Hong_Kong").tz_convert("UTC")
    session = ta.trading_date_for(ts, "HKEX")
    assert session >= MONDAY or CAL_HK.is_session(session)


def test_kr_session_alignment():
    kr_monday = CAL_KR.date_to_session(pd.Timestamp("2026-11-09"))
    ts = pd.Timestamp(f"{kr_monday.date()} 10:00", tz="Asia/Seoul").tz_convert("UTC")
    assert ta.trading_date_for(ts, "KRX") == kr_monday


def test_none_timestamp_returns_none():
    assert ta.trading_date_for(None, "HKEX") is None
    assert ta.trading_date_for(pd.NaT, "HKEX") is None


def test_align_posts_adds_trading_date_column():
    ts = pd.Timestamp(f"{MONDAY.date()} 09:00", tz="Asia/Hong_Kong").tz_convert("UTC")
    p = make_post("lihkg", "0700.HK", "morning post", ts.isoformat(), "https://l/5")
    p_no_ts = make_post("naver", "005930.KS", "no ts", None, None)
    df = posts_frame([p, p_no_ts])
    df["posted_at_utc"] = pd.to_datetime(df["posted_at_utc"], errors="coerce", utc=True)
    out = ta.align_posts(df, UNI)
    assert out.loc[0, "trading_date"] == MONDAY.date().isoformat()
    assert out.loc[1, "trading_date"] is None or pd.isna(out.loc[1, "trading_date"])
