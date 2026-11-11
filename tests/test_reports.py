import pandas as pd
import pytest

from config.settings import Asset, Universe
from src.analytics import report_builder as rb

HK1 = Asset("0700.HK", "Tencent", "騰訊", "HKEX", None, ["Tencent"])
HK2 = Asset("9988.HK", "Alibaba", "阿里巴巴", "HKEX", None, ["Alibaba"])
UNI = Universe(10, 1, {"HKEX": {"name": "Hong Kong Stock Exchange"}}, [HK1, HK2], {})

ANALYSIS = pd.DataFrame(
    [
        {"ticker": "0700.HK", "company_name": "Tencent", "exchange": "HKEX", "price": 417.8,
         "social_sentiment": 0.35, "social_mentions": 40, "news_sentiment": 0.2,
         "direction": "bullish", "composite_signal": 0.4, "predicted_next_close": 420.1,
         "predicted_pct_change": 0.55, "pos_pct": 60, "neg_pct": 20},
        {"ticker": "9988.HK", "company_name": "Alibaba", "exchange": "HKEX", "price": 101.6,
         "social_sentiment": -0.22, "social_mentions": 15, "news_sentiment": -0.1,
         "direction": "bearish", "composite_signal": -0.3, "predicted_next_close": None,
         "predicted_pct_change": None, "pos_pct": 20, "neg_pct": 55},
    ]
)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(rb, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(rb, "load_from_sqlite", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(rb, "load_parquet", lambda *a, **kw: None)
    return tmp_path


def test_build_exchange_report(isolated):
    out = rb.build_exchange_report("HKEX", UNI, analysis=ANALYSIS)
    assert out is not None and out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Market overview" in html
    assert "0700.HK" in html and "9988.HK" in html
    assert "not financial advice" in html.lower()
    assert "Data-quality limitations" in html
    assert (isolated / "HKEX_overview.png").exists()


def test_build_all_reports_skips_empty_exchange(isolated):
    paths = rb.build_all_reports(UNI, analysis=ANALYSIS)
    assert len(paths) == 1


def test_model_table_marks_untrained(isolated):
    tbl = rb._model_table(UNI, ["0700.HK"])
    assert tbl["model"].iloc[0] == "not trained"


def test_status_table_empty(isolated):
    tbl = rb._status_table(["0700.HK"])
    assert tbl.empty


# ---------- CLI ----------

def test_main_all_expands_to_five_stages():
    import main as main_mod

    assert main_mod.MODES[:-1] == ["collect", "sentiment", "train", "predict", "report"]


def test_main_requires_mode():
    import main as main_mod

    with pytest.raises(SystemExit):
        main_mod.main([])
