"""Slack request signing (HMAC-SHA256) and user allowlist verification."""

from __future__ import annotations

import hashlib
import hmac
import time


def verify_slack_signature(
    *,
    signing_secret: str,
    timestamp: str | None,
    signature: str | None,
    raw_body: bytes,
    now: float | None = None,
    tolerance_seconds: int = 300,
) -> bool:
    """Verify Slack X-Slack-Signature (v0=HMAC-SHA256) with 5-minute replay protection."""
    if not signing_secret or not timestamp or not signature:
        return False

    try:
        req_ts = int(timestamp)
    except ValueError:
        return False

    current_ts = time.time() if now is None else now
    if abs(current_ts - req_ts) > tolerance_seconds:
        return False

    basestring = f"v0:{req_ts}:".encode() + raw_body
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        basestring,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def is_allowed_slack_user(user_id: str | None, allowed_user_ids: set[str]) -> bool:
    """Return True if allowlist is empty or user_id is explicitly permitted."""
    if not allowed_user_ids:
        return True
    return bool(user_id and user_id in allowed_user_ids)
