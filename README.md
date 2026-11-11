# HengSeng Sentiment Price Prediction Agent

AI-powered financial sentiment analysis and next-period stock price prediction for
**HKEX** (Hong Kong) and **KRX** (South Korea) equities. Built as an internship
assessment for CrowdWisdomTrading / TZURONI LTD.

The pipeline collects social-media posts (Apify), financial news (Exa, optional
Tavily fallback) and historical prices (Yahoo Finance), scores sentiment with
**FinBERT**, engineers leakage-free time-series features, trains and compares
**Linear Regression / Random Forest / XGBoost** with chronological + walk-forward
validation, predicts the next trading day's close, and serves everything through a
**Streamlit** dashboard and per-exchange visual reports.

> **Disclaimer:** every directional signal, score and prediction produced by this
> project is an analytical research output — **not financial advice**.

---

## System architecture

```
                       ┌─────────────────────────────────────────────┐
                       │                 main.py CLI                 │
                       │  collect → sentiment → train → predict →    │
                       │  report            (or --mode all)          │
                       └───────┬─────────────┬─────────────┬─────────┘
                               │             │             │
   ┌───────────────────────────▼──┐   ┌──────▼───────┐  ┌──▼─────────────────┐
   │ src/ingestion                │   │ src/         │  │ src/prediction     │
   │  apify_client (retry/backoff)│   │ sentiment    │  │  train / evaluate  │
   │  lihkg · telegram (HKEX)     │   │  FinBERT     │  │  predict           │
   │  naver · daum  (KRX)         │   │  aggregates  │  │ src/features       │
   │  exa_news · tavily_news      │   │ src/         │  │  no look-ahead     │
   │  prices (yfinance)           │   │ preprocessing│  │ src/analytics      │
   └──────────────┬───────────────┘   │  clean+align │  │  S/R, signals      │
                  │                   └──────┬───────┘  │  report_builder    │
                  ▼                          ▼          └──┬─────────────────┘
        ┌──────────────────────────────────────────────┐   │
        │ SQLite (data/database) + Parquet (data/)     │◄──┘
        │  prices · social_posts · news_articles ·     │
        │  posts_annotated · predictions · status      │
        └───────────────────┬──────────────────────────┘
                            ▼
              ┌───────────────────────────┐      ┌──────────────────────┐
              │ dashboard/app.py          │      │ reports/             │
              │ Streamlit (sections A–F)  │      │ HKEX/KRX HTML + PNG  │
              └───────────────────────────┘      └──────────────────────┘
```

## Features

- **10 assets, 2 exchanges** — configurable via `config/assets.yaml` (no hardcoded tickers)
- **2 independent social sources per exchange**:
  - HKEX: LIHKG forum + Telegram public channels
  - KRX: Naver Finance 종목토론실 + Daum search (see limitation below)
- **News ingestion**: Exa API (primary), Tavily (optional fallback)
- **FinBERT financial sentiment** — normalized score in [-1, +1], engagement-weighted
  aggregation (log-scaled so viral posts can't dominate), source-specific metrics
- **Technical features**: SMA20/50, RSI, MACD, ATR, rolling volatility, volume ratio
- **Leakage-safe alignment**: posts attributed to exchange sessions (XHKG/XKRX
  calendars) — a post only informs the session whose close comes *after* it
- **3 models + previous-close baseline**, chronological split and walk-forward
  validation, MAE/RMSE/MAPE/directional accuracy
- **Pivot-point support/resistance** + transparent bullish/bearish/sideways rule
- **Streamlit dashboard** (6 sections, filters, CSV export) and HTML/PNG reports
- **Graceful degradation**: missing API keys or a dead source never crash the run;
  every failure is logged and surfaced in the Data Status view

## Technologies

Python 3.11 · Apify API · Exa API · Tavily API · pandas/NumPy · FinBERT
(transformers + torch) · scikit-learn · XGBoost · yfinance · exchange-calendars ·
Streamlit · Plotly · Matplotlib · SQLite + Parquet · Pytest

## Installation

```bash
# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux/macOS:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Environment configuration

```bash
cp .env.example .env     # then edit .env and paste your real keys
```

`.env` is gitignored — never commit it. Keys are read via environment variables
only; nothing is hardcoded.

### API account setup

| Service | Signup | Free tier | Key → variable |
|---|---|---|---|
| Apify | https://console.apify.com | $5 platform credit/month | Account → Integrations → **APIFY_API_TOKEN** |
| Exa | https://dashboard.exa.ai | free credits on signup | Dashboard → **EXA_API_KEY** |
| Tavily (optional) | https://app.tavily.com | 1,000 searches/month | Dashboard → **TAVILY_API_KEY** |

If a key is missing the corresponding collector logs a warning and skips —
price collection, training and prediction still work.

## Running the pipeline

```bash
python main.py --mode collect     # prices + social (Apify) + news (Exa/Tavily)
python main.py --mode sentiment   # clean, align, FinBERT-annotate, aggregate
python main.py --mode train       # train/compare models per ticker
python main.py --mode predict     # next-period predictions + analytics
python main.py --mode report      # reports/HKEX_report.html, KRX_report.html, PNGs
python main.py --mode all         # every stage in order
python main.py --mode collect --days 10   # override lookback window
```

First `sentiment` run downloads FinBERT weights (~440 MB) from Hugging Face.

### Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501 — sections: Market Overview · Stock Ranking ·
Stock Analysis · Price Prediction · Model Performance · Data Status. All tables
have CSV export buttons.

### Tests

```bash
pytest -v              # offline suite (mocked APIs, no network)
pytest -m network -v   # live checks: real yfinance + real FinBERT scoring
```

## Apify actors (verified 2026-11-11 against the public Apify store)

| Source | Actor | Notes |
|---|---|---|
| LIHKG | `claude_code_reviewer~lihkg-scraper-en` | keyword-search mode, relative date window (`dateFrom: "10d"`) |
| Telegram | `george.the.developer~telegram-channel-scraper` | keyword filter + `discoverChannels` finds public channels per keyword |
| Naver 토론실 | `hgservices~naver-stock-discussion-scraper` | input `{"stocks": ["005930"], "maxResults": N}` |
| Daum | `rainminer~daum-search-scraper` | **see limitation below** |

Actor IDs are configurable in `config/assets.yaml` (`apify_actors:`).

## Known limitations (honest by design)

1. **No dedicated Daum Finance forum actor exists on Apify** (verified by store
   search on 2026-11-11). The KRX second source uses the Daum unified *search*
   scraper, which surfaces Daum blog/cafe/news results for stock queries — search
   snippets, not full forum threads. `ardent_fork~daum-news` is a documented
   complement candidate.
2. **10 days of social data cannot statistically validate sentiment-driven
   prediction.** The system trains primarily on price features, includes sentiment
   features where sessions have coverage, and reports the measured difference
   between `price_only` and `with_sentiment` model variants in
   `models/<ticker>/meta.json`. No predictive power of sentiment is claimed.
3. **Historical social sentiment does not exist in this project** — it is never
   fabricated. Sentiment coverage is limited to the collection window.
4. Output field names per actor are normalized defensively (candidate field
   lists); if an actor changes its output schema, adapters log the mismatch and
   return fewer/zero posts instead of crashing.
5. The 95% prediction interval is a normal approximation from out-of-sample
   residual std — not a conformal or bootstrap interval.
6. LG Energy Solution (373220.KS) listed in Jan 2022 — shorter history than other
   assets.

## Project structure

```
HengSeng_Sentiment_Agent/
├── config/            settings.py (env, paths) · assets.yaml (universe, actors)
├── data/              raw/ processed/ historical_prices/ database/ logs/
├── src/
│   ├── ingestion/     apify_client · lihkg · telegram · naver · daum ·
│   │                  exa_news · tavily_news · prices · pipeline
│   ├── preprocessing/ cleaning · timestamp_alignment (XHKG/XKRX calendars)
│   ├── sentiment/     sentiment_analyzer (FinBERT) · run_sentiment
│   ├── features/      feature_engineering (leakage-safe)
│   ├── prediction/    train · evaluate · predict
│   ├── analytics/     stock_analysis (S/R, signals) · report_builder
│   └── utils/         logger · storage (SQLite + Parquet, dedup)
├── dashboard/app.py   Streamlit dashboard (sections A–F)
├── reports/           generated HTML + PNG exchange reports
├── models/            per-ticker best_model.joblib + meta.json
├── tests/             pytest suite (offline mocks + `-m network` live checks)
├── main.py            CLI entry point
├── requirements.txt · .env.example · .gitignore · pytest.ini
```

## Example outputs

- `models/test_summary.csv` — per-ticker out-of-sample metrics for the selected model
- `models/0700_HK/meta.json` — validation of all 6 candidates, test metrics,
  walk-forward metrics, baseline comparison, sentiment coverage count
- `reports/HKEX_report.html` — interactive exchange report (rankings by
  mentions/sentiment/direction, predictions, model evaluation, data quality notes)
- `data/processed/stock_analysis.csv` — full analytics row per asset (price,
  sentiment, S/R levels, technicals, prediction, disclaimer)

## Reproducibility checklist for another developer

1. Python 3.11+, `pip install -r requirements.txt`
2. `cp .env.example .env`, add Apify + Exa keys
3. `python main.py --mode all`
4. `pytest -v` (offline) and optionally `pytest -m network`
5. `streamlit run dashboard/app.py`
