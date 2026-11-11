"""Central application settings.

All configuration is loaded from environment variables (.env) and
config/assets.yaml. No credentials are hardcoded anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRICES_DIR = DATA_DIR / "historical_prices"
DATABASE_DIR = DATA_DIR / "database"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, PRICES_DIR, DATABASE_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "market_data.sqlite"

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")

_PLACEHOLDER_PREFIXES = ("your_", "changeme", "placeholder", "replace_", "<", "xxx")


def _clean_secret(value: str | None) -> str | None:
    """Treat empty strings and .env.example placeholders as missing keys,
    so the pipeline degrades gracefully instead of sending junk to APIs."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower().startswith(_PLACEHOLDER_PREFIXES):
        return None
    return value


@dataclass
class APIKeys:
    """API credentials. Missing keys degrade gracefully: the affected
    collector is skipped with a logged warning instead of crashing."""

    apify: str | None = field(default_factory=lambda: _clean_secret(os.getenv("APIFY_API_TOKEN")))
    exa: str | None = field(default_factory=lambda: _clean_secret(os.getenv("EXA_API_KEY")))
    tavily: str | None = field(default_factory=lambda: _clean_secret(os.getenv("TAVILY_API_KEY")))

    def available(self, name: str) -> bool:
        return bool(getattr(self, name, None))

    def missing(self) -> list[str]:
        return [n for n in ("apify", "exa", "tavily") if not self.available(n)]


@dataclass
class RetrySettings:
    max_retries: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    backoff_base_seconds: float = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT", "60"))


@dataclass
class Asset:
    ticker: str
    company_name: str
    local_name: str
    exchange: str
    naver_code: str | None
    search_keywords: list[str]


@dataclass
class Universe:
    social_lookback_days: int
    history_years: int
    exchanges: dict
    assets: list[Asset]
    apify_actors: dict[str, str]

    def by_exchange(self, exchange: str) -> list[Asset]:
        return [a for a in self.assets if a.exchange == exchange]

    def get(self, ticker: str) -> Asset | None:
        return next((a for a in self.assets if a.ticker == ticker), None)


def load_universe(path: Path | None = None) -> Universe:
    """Load the asset universe from YAML (configurable, not hardcoded)."""
    path = path or PROJECT_ROOT / "config" / "assets.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    assets = [
        Asset(
            ticker=a["ticker"],
            company_name=a["company_name"],
            local_name=a.get("local_name", a["company_name"]),
            exchange=a["exchange"],
            naver_code=a.get("naver_code"),
            search_keywords=list(a.get("search_keywords", [a["company_name"]])),
        )
        for a in raw["assets"]
    ]
    return Universe(
        social_lookback_days=int(raw.get("social_lookback_days", 10)),
        history_years=int(raw.get("history_years", 3)),
        exchanges=raw.get("exchanges", {}),
        assets=assets,
        apify_actors=raw.get("apify_actors", {}),
    )


api_keys = APIKeys()
retry_settings = RetrySettings()
