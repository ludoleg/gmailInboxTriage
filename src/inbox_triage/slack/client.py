"""Slack delivery clients: Incoming Webhook (push-only fallback) and Bot Web API (interactive)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api"


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


class SlackBotClient:
    """Posts, updates, and replies to interactive Block Kit messages via Slack Web API."""

    def __init__(
        self,
        bot_token: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not bot_token:
            raise ValueError("SLACK_BOT_TOKEN must be configured for SlackBotClient.")
        self.bot_token = bot_token
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = http_client

    def _api_post(self, method: str, body: dict[str, Any]) -> dict[str, Any]:
        client = self._http_client or httpx.Client(timeout=self.timeout)
        owns_client = self._http_client is None
        url = f"{SLACK_API_BASE}/{method}"
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        backoff = 1.0
        try:
            for attempt in range(1, self.max_retries + 1):
                resp = client.post(url, json=body, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    logger.warning(
                        "Slack API %s returned HTTP %d (attempt %d/%d); retrying in %.1fs...",
                        method,
                        resp.status_code,
                        attempt,
                        self.max_retries,
                        retry_after,
                    )
                    time.sleep(retry_after)
                    backoff *= 2.0
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Slack API {method} failed with HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                if not data.get("ok"):
                    raise RuntimeError(f"Slack API {method} error: {data.get('error', 'unknown')}")
                return data
            raise RuntimeError(f"Slack API {method} exhausted retries.")
        finally:
            if owns_client:
                client.close()

    def post_payloads(self, channel_id: str, payloads: list[dict[str, Any]]) -> list[str]:
        """Post one or more Block Kit payloads to `channel_id`; returns message `ts` values."""
        timestamps: list[str] = []
        for idx, payload in enumerate(payloads, start=1):
            body = {"channel": channel_id, **payload, "unfurl_links": False, "unfurl_media": False}
            data = self._api_post("chat.postMessage", body)
            ts = str(data.get("ts") or "")
            timestamps.append(ts)
            logger.info("Posted interactive Slack digest part %d/%d (ts=%s).", idx, len(payloads), ts)
        return timestamps

    def post_thread_reply(
        self,
        channel_id: str,
        thread_ts: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> str:
        """Post a reply inside an existing Slack message thread."""
        body: dict[str, Any] = {
            "channel": channel_id,
            "thread_ts": thread_ts,
            "text": text,
            "unfurl_links": False,
        }
        if blocks:
            body["blocks"] = blocks
        data = self._api_post("chat.postMessage", body)
        return str(data.get("ts") or "")

    def update_message(
        self,
        channel_id: str,
        ts: str,
        text: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        """Update a Slack message's Block Kit blocks in place."""
        self._api_post(
            "chat.update",
            {"channel": channel_id, "ts": ts, "text": text, "blocks": blocks},
        )

    def post_response_url(self, response_url: str, payload: dict[str, Any]) -> None:
        """Send an ephemeral or in-channel follow-up to a Slack interaction response_url."""
        client = self._http_client or httpx.Client(timeout=self.timeout)
        owns_client = self._http_client is None
        try:
            resp = client.post(response_url, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Slack response_url failed with HTTP {resp.status_code}: {resp.text[:300]}"
                )
        finally:
            if owns_client:
                client.close()
