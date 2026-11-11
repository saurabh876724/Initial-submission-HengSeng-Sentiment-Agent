"""Thin, resilient wrapper around the Apify API.

Responsibilities:
- graceful degradation when APIFY_API_TOKEN is missing (log + return empty)
- retries with exponential backoff on transient failures
- run an actor, wait, and return dataset items as plain dicts

The four social-source adapters (lihkg, telegram, naver, daum) each build
their own actor input and normalize the output into a common post record,
so one unavailable source never breaks the pipeline.
"""
from __future__ import annotations

import time
from typing import Any

from config.settings import RetrySettings, api_keys, retry_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class ApifySourceUnavailable(RuntimeError):
    """Raised internally when Apify cannot be used; caught by callers."""


class ApifyRunner:
    def __init__(
        self,
        token: str | None = None,
        retry: RetrySettings | None = None,
        client_factory: Any = None,
    ):
        self.token = token if token is not None else api_keys.apify
        self.retry = retry or retry_settings
        self._client_factory = client_factory  # injectable for tests
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _get_client(self):
        if self._client is None:
            if not self.available:
                raise ApifySourceUnavailable(
                    "APIFY_API_TOKEN is not set — skipping Apify collection. "
                    "Copy .env.example to .env and add your token."
                )
            if self._client_factory is not None:
                self._client = self._client_factory(self.token)
            else:
                from apify_client import ApifyClient

                self._client = ApifyClient(self.token)
        return self._client

    def run_actor(self, actor_id: str, run_input: dict, timeout_secs: int = 600) -> list[dict]:
        """Run an Apify actor and return its dataset items.

        Retries transient failures with exponential backoff. Returns []
        (with a logged error) when the actor is unavailable or fails
        permanently — collection for other sources continues.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.retry.max_retries + 1):
            try:
                client = self._get_client()
                log.info("apify: starting actor %s (attempt %d)", actor_id, attempt)
                run = client.actor(actor_id).call(run_input=run_input, timeout=timeout_secs)
                if run is None:
                    raise ApifySourceUnavailable(f"actor {actor_id} returned no run object")
                status = run.get("status")
                if status != "SUCCEEDED":
                    raise ApifySourceUnavailable(f"actor {actor_id} run status: {status}")
                dataset_id = run.get("defaultDatasetId")
                items: list[dict] = []
                offset = 0
                page = 1000
                while True:  # pagination over dataset items
                    batch = client.dataset(dataset_id).list_items(offset=offset, limit=page)
                    chunk = batch.items if hasattr(batch, "items") else batch.get("items", [])
                    items.extend(chunk)
                    total = batch.total if hasattr(batch, "total") else batch.get("total", len(items))
                    offset += len(chunk)
                    if not chunk or offset >= total:
                        break
                log.info("apify: actor %s returned %d items", actor_id, len(items))
                return items
            except ApifySourceUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 — transient network/API errors
                last_error = exc
                wait = self.retry.backoff_base_seconds ** attempt
                log.warning(
                    "apify: actor %s attempt %d failed (%s); retrying in %.0fs",
                    actor_id, attempt, exc, wait,
                )
                time.sleep(wait)
        log.error("apify: actor %s failed permanently: %s", actor_id, last_error)
        return []
