"""Unit tests for Slack Block Kit formatting, chunking, truncation, and webhook retry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from inbox_triage.models import (
    CloudEscalationResult,
    DeepThreadAnalysis,
    ExecutiveBriefing,
    ModelCallStats,
    MultimodalFinding,
    TechExtraction,
    TriageCategory,
    TriagedThread,
    TriageRunReport,
)
from inbox_triage.slack.client import SlackWebhookClient
from inbox_triage.slack.formatter import (
    MAX_BLOCKS_PER_MESSAGE,
    MAX_HEADER_CHARS,
    MAX_SECTION_TEXT_CHARS,
    build_slack_payloads,
)
from tests.conftest import make_email_thread


def _all_block_text(payloads: list[dict]) -> str:
    chunks: list[str] = []
    for p in payloads:
        for b in p["blocks"]:
            if "text" in b:
                chunks.append(b["text"]["text"])
            for el in b.get("elements", []):
                if "text" in el:
                    chunks.append(el["text"])
    return "\n".join(chunks)


def test_inbox_zero_card_uses_pacific_time() -> None:
    # 12:30 UTC in April (PDT = UTC-7) -> 5:30 AM PDT
    ts = datetime(2026, 4, 15, 12, 30, tzinfo=UTC)
    report = TriageRunReport(
        run_id="zero-1",
        started_at=ts,
        completed_at=ts,
        timezone="America/Los_Angeles",
        total_threads_fetched=0,
    )

    payloads = build_slack_payloads(report)

    assert len(payloads) == 1
    text = _all_block_text(payloads)
    assert "5:30 AM PDT" in text
    assert "Inbox Zero!" in text
    assert "0 GPU switches · 0 model calls" in text


def test_full_report_sections_deep_links_cloud_badge_and_pacific_time() -> None:
    started = datetime(2026, 1, 15, 13, 30, tzinfo=UTC)  # 5:30 AM PST (standard time)
    completed = started + timedelta(seconds=12.4)

    urgent_item = TriagedThread(
        thread=make_email_thread(
            thread_id="t-urg",
            subject="Urgent: A & B <Contract>",
            sender="CEO <Boss>",
            received_at=started,
        ),
        category=TriageCategory.URGENT,
        confidence=0.96,
        summary="Sign updated contract before noon.",
        cloud_model_used="gemini-4-neon",
        cloud_escalation=CloudEscalationResult(
            refined_category=TriageCategory.URGENT,
            executive_summary="Sign updated contract before noon.",
            action_items=["Review section 4", "Countersign PDF"],
            deadline="Today 12:00 PM PT",
            risk_assessment="Auto-renewal clause triggers at noon.",
            suggested_next_step="Email legal to confirm redlines.",
        ),
        multimodal_findings=[
            MultimodalFinding(filename="sig.png", summary="Signature page", urgency_signal="high")
        ],
    )
    action_item = TriagedThread(
        thread=make_email_thread(thread_id="t-act", subject="Schedule Q2 sync", received_at=started),
        category=TriageCategory.ACTION_REQUIRED,
        confidence=0.90,
        summary="Pick a meeting time.",
        deep_analysis=DeepThreadAnalysis(
            refined_category=TriageCategory.ACTION_REQUIRED,
            confidence=0.90,
            summary="Pick a meeting time.",
            action_items=["Reply with availability"],
            deadline="Tomorrow",
            suggested_reply_points=["10 AM PT works"],
        ),
    )
    tech_item = TriagedThread(
        thread=make_email_thread(thread_id="t-tech", subject="CI failed", received_at=started),
        category=TriageCategory.TECHNICAL_ALERT,
        confidence=0.95,
        summary="Tests failed.",
        tech_extraction=TechExtraction(
            source_system="GitHub",
            severity="high",
            repo_or_service="acme/api",
            error_or_event="test_idempotency failed",
        ),
    )
    fyi_item = TriagedThread(
        thread=make_email_thread(thread_id="t-fyi", subject="Office closed Friday", received_at=started),
        category=TriageCategory.FYI,
        confidence=0.88,
        summary="Holiday notice.",
    )
    promo_item = TriagedThread(
        thread=make_email_thread(thread_id="t-promo", subject="Big Sale", received_at=started),
        category=TriageCategory.PROMO_NOISE,
        confidence=0.99,
        summary="Promo",
    )

    report = TriageRunReport(
        run_id="full-1",
        started_at=started,
        completed_at=completed,
        timezone="America/Los_Angeles",
        total_threads_fetched=5,
        triaged_threads=[urgent_item, action_item, tech_item, fyi_item, promo_item],
        briefing=ExecutiveBriefing(
            headline="Contract deadline at noon + CI failure",
            executive_summary="Prioritize the contract renewal and check CI.",
            cross_thread_connections=["Contract and Q2 sync both involve Partner X"],
        ),
        call_stats=[
            ModelCallStats(backend="inkling", model="inkling-2-mini", stage="stage1_triage"),
            ModelCallStats(backend="cloud", model="gemini-4-neon", stage="stage4_cloud"),
        ],
        cloud_escalations_count=1,
        cloud_blocked_by_denylist=2,
    )

    payloads = build_slack_payloads(report, mention_on_urgent="<@U12345>")
    assert len(payloads) == 1

    text = _all_block_text(payloads)
    assert "5:30 AM PST" in text
    assert "<@U12345>" in text
    assert "https://mail.google.com/mail/u/0/#inbox/t-urg" in text
    assert "Urgent: A &amp; B &lt;Contract&gt;" in text
    assert "☁️ gemini-4-neon" in text
    assert "Auto-renewal clause triggers at noon." in text
    assert "sig.png: Signature page" in text
    assert "https://mail.google.com/mail/u/0/#inbox/t-act" in text
    assert "10 AM PT works" in text
    assert "`[HIGH]` *GitHub*" in text
    assert "Office closed Friday" in text
    assert "Filtered silently: 0 newsletter(s), 1 promo/noise thread(s)." in text
    assert "Privacy-blocked: 2" in text
    assert "gemini-4-neon" in text and "inkling-2-mini" in text


def test_chunking_splits_payloads_exceeding_50_blocks_and_truncates_long_text() -> None:
    ts = datetime(2026, 4, 15, 12, 30, tzinfo=UTC)
    huge_summary = "A" * 4000
    items = [
        TriagedThread(
            thread=make_email_thread(thread_id=f"t-{i}", subject=f"Urgent item #{i}", received_at=ts),
            category=TriageCategory.URGENT,
            confidence=0.95,
            summary=huge_summary if i == 0 else f"Summary {i}",
        )
        for i in range(55)
    ]
    report = TriageRunReport(
        run_id="chunk-1",
        started_at=ts,
        completed_at=ts,
        timezone="America/Los_Angeles",
        total_threads_fetched=len(items),
        triaged_threads=items,
    )

    payloads = build_slack_payloads(report)

    assert len(payloads) >= 2
    for payload in payloads:
        assert len(payload["blocks"]) <= MAX_BLOCKS_PER_MESSAGE
        for block in payload["blocks"]:
            if block["type"] == "header":
                assert len(block["text"]["text"]) <= MAX_HEADER_CHARS
            elif block["type"] == "section":
                assert len(block["text"]["text"]) <= MAX_SECTION_TEXT_CHARS


# --- SlackWebhookClient retry tests ---


def test_slack_webhook_client_retries_on_429_and_503(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("inbox_triage.slack.client.time.sleep", lambda s: sleeps.append(s))

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="rate_limited")
        if attempts == 2:
            return httpx.Response(503, text="service_unavailable")
        return httpx.Response(200, text="ok")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    slack = SlackWebhookClient(
        webhook_url="https://hooks.slack.com/services/T/B/X",
        max_retries=3,
        http_client=http_client,
    )
    slack.post_payloads([{"text": "hi", "blocks": []}])

    assert attempts == 3
    assert sleeps == [2.0, 2.0]


def test_slack_webhook_client_raises_immediately_on_400_and_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("inbox_triage.slack.client.time.sleep", lambda _: None)

    with pytest.raises(ValueError, match="SLACK_WEBHOOK_URL"):
        SlackWebhookClient("")

    # 400 Bad Request -> no retries
    calls_400 = 0

    def handler_400(request: httpx.Request) -> httpx.Response:
        nonlocal calls_400
        calls_400 += 1
        return httpx.Response(400, text="invalid_blocks")

    slack_400 = SlackWebhookClient(
        webhook_url="https://hooks.slack.com/services/T/B/X",
        max_retries=3,
        http_client=httpx.Client(transport=httpx.MockTransport(handler_400)),
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        slack_400.post_payloads([{"text": "bad", "blocks": []}])
    assert calls_400 == 1

    # 503 exhausted -> raises after max_retries
    calls_503 = 0

    def handler_503(request: httpx.Request) -> httpx.Response:
        nonlocal calls_503
        calls_503 += 1
        return httpx.Response(503, text="down")

    slack_503 = SlackWebhookClient(
        webhook_url="https://hooks.slack.com/services/T/B/X",
        max_retries=3,
        http_client=httpx.Client(transport=httpx.MockTransport(handler_503)),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        slack_503.post_payloads([{"text": "retry", "blocks": []}])
    assert calls_503 == 3
