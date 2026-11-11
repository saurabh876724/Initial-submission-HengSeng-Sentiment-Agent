from config.settings import APIKeys, load_universe


def test_universe_has_ten_assets():
    u = load_universe()
    assert len(u.assets) == 10
    assert len(u.by_exchange("HKEX")) == 5
    assert len(u.by_exchange("KRX")) == 5


def test_universe_tickers():
    u = load_universe()
    tickers = {a.ticker for a in u.assets}
    assert "0700.HK" in tickers
    assert "005930.KS" in tickers
    assert u.get("9988.HK").company_name == "Alibaba Group"
    assert u.get("NOPE") is None


def test_krx_assets_have_naver_codes():
    u = load_universe()
    for a in u.by_exchange("KRX"):
        assert a.naver_code and a.naver_code.isdigit()
    for a in u.by_exchange("HKEX"):
        assert a.naver_code is None


def test_all_assets_have_keywords():
    u = load_universe()
    for a in u.assets:
        assert len(a.search_keywords) >= 2


def test_apify_actors_configured():
    u = load_universe()
    for key in ("lihkg", "telegram", "naver_stock_discussion", "daum_search"):
        assert "~" in u.apify_actors[key]


def test_api_keys_graceful_when_missing(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    keys = APIKeys()
    assert keys.apify is None
    assert not keys.available("apify")
    assert "apify" in keys.missing()


def test_placeholder_values_treated_as_missing(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "your_apify_token_here")
    monkeypatch.setenv("EXA_API_KEY", "  ")
    monkeypatch.setenv("TAVILY_API_KEY", "real-key-123")
    keys = APIKeys()
    assert keys.apify is None
    assert keys.exa is None
    assert keys.tavily == "real-key-123"
