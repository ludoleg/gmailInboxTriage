"""Unit tests for Slack HMAC-SHA256 request verification and user allowlist checks."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from inbox_triage.slack.security import is_allowed_slack_user, verify_slack_signature


def _sign(secret: str, timestamp: int | str, body: bytes) -> str:
    basestring = f"v0:{timestamp}:".encode() + body
    return "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()


class TestVerifySlackSignature:
    def test_valid_hmac_sha256_signature_accepted(self) -> None:
        secret = "slack-signing-secret-123"
        ts = 1776254400
        body = b"token=abc&command=%2Ftriage&text=48h"
        sig = _sign(secret, ts, body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts + 45),
            )
            is True
        )

    def test_exact_tolerance_boundary_is_accepted(self) -> None:
        secret = "slack-signing-secret-123"
        ts = 1776254400
        body = b"payload=%7B%7D"
        sig = _sign(secret, ts, body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts + 300),
            )
            is True
        )
        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts - 300),
            )
            is True
        )

    def test_tampered_body_rejected(self) -> None:
        secret = "slack-signing-secret-123"
        ts = 1776254400
        original_body = b"command=%2Ftriage&text=24h"
        tampered_body = b"command=%2Ftriage&text=168h"
        sig = _sign(secret, ts, original_body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=tampered_body,
                now=float(ts + 5),
            )
            is False
        )

    def test_wrong_secret_rejected(self) -> None:
        ts = 1776254400
        body = b"payload=%7B%22type%22%3A%22block_actions%22%7D"
        sig = _sign("attacker-secret", ts, body)

        assert (
            verify_slack_signature(
                signing_secret="real-secret",
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts),
            )
            is False
        )

    @pytest.mark.parametrize(
        ("secret", "timestamp", "signature"),
        [
            ("", "1776254400", "v0=deadbeef"),
            ("real-secret", None, "v0=deadbeef"),
            ("real-secret", "", "v0=deadbeef"),
            ("real-secret", "1776254400", None),
            ("real-secret", "1776254400", ""),
        ],
    )
    def test_missing_secret_or_headers_rejected(
        self,
        secret: str,
        timestamp: str | None,
        signature: str | None,
    ) -> None:
        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=timestamp,
                signature=signature,
                raw_body=b"body",
                now=1776254400.0,
            )
            is False
        )

    def test_expired_timestamp_too_old_rejected(self) -> None:
        secret = "slack-signing-secret-123"
        ts = 1776254400
        body = b"command=%2Ftriage"
        sig = _sign(secret, ts, body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts + 301),
            )
            is False
        )

    def test_expired_timestamp_too_far_in_future_rejected(self) -> None:
        secret = "slack-signing-secret-123"
        ts = 1776254400
        body = b"command=%2Ftriage"
        sig = _sign(secret, ts, body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=str(ts),
                signature=sig,
                raw_body=body,
                now=float(ts - 301),
            )
            is False
        )

    @pytest.mark.parametrize("bad_ts", ["not-an-int", "1776254400.5", "NaN", "v0:123"])
    def test_non_integer_timestamp_rejected(self, bad_ts: str) -> None:
        secret = "slack-signing-secret-123"
        body = b"command=%2Ftriage"
        sig = _sign(secret, bad_ts, body)

        assert (
            verify_slack_signature(
                signing_secret=secret,
                timestamp=bad_ts,
                signature=sig,
                raw_body=body,
                now=1776254400.0,
            )
            is False
        )


class TestIsAllowedSlackUser:
    def test_empty_allowlist_allows_any_user(self) -> None:
        assert is_allowed_slack_user("U123456", set()) is True
        assert is_allowed_slack_user("", set()) is True
        assert is_allowed_slack_user(None, set()) is True

    def test_non_empty_allowlist_restricts_strictly(self) -> None:
        allowed = {"U01OWNER", "U02ADMIN"}

        assert is_allowed_slack_user("U01OWNER", allowed) is True
        assert is_allowed_slack_user("U02ADMIN", allowed) is True
        assert is_allowed_slack_user("U99INTRUDER", allowed) is False
        assert is_allowed_slack_user("u01owner", allowed) is False
        assert is_allowed_slack_user("", allowed) is False
        assert is_allowed_slack_user(None, allowed) is False
