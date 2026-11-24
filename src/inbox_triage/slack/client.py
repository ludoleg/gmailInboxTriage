"""Slack Incoming Webhook delivery client with exponential backoff."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SlackWebhookClient:
    """Posts Block Kit payloads to a Slack Incoming Webhook URL."""

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not webhook_url:
            raise ValueError("SLACK_WEBHOOK_URL must be configured to deliver Slack updates.")
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = http_client

    def post_payloads(self, payloads: list[dict[str, Any]]) -> None:
        """Send one or more sequential Slack webhook payloads."""
        client = self._http_client or httpx.Client(timeout=self.timeout)
        owns_client = self._http_client is None
        try:
            for idx, payload in enumerate(payloads, start=1):
                self._post_single(client, payload, idx, len(payloads))
        finally:
            if owns_client:
                client.close()

    def _post_single(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
        index: int,
        total: int,
    ) -> None:
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            resp = client.post(self.webhook_url, json=payload)
            if 200 <= resp.status_code < 300:
                logger.info("Delivered Slack digest part %d/%d.", index, total)
                return
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                logger.warning(
                    "Slack webhook returned %d (attempt %d/%d); retrying in %.1fs...",
                    resp.status_code,
                    attempt,
                    self.max_retries,
                    retry_after,
                )
                time.sleep(retry_after)
                backoff *= 2.0
                continue
            raise RuntimeError(
                f"Slack webhook failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
