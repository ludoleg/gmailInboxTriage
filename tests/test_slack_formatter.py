"""Unit tests for Slack Block Kit formatting and SlackWebhookClient retries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest

from inbox_triage.models import (
    DeepThreadAnalysis,
    ExecutiveBriefing,
    HighStakesReview,
    ModelCallStats,
    TechExtraction,
    TriageCategory,
    TriagedThread,
    TriageRunReport,
)
from inbox_triage.slack.client import SlackWebhookClient
from inbox_triage.slack.formatter import MAX_BLOCKS_PER_MESSAGE, build_slack_payloads
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


class TestBuildSlackPayloads:
    def test_inbox_zero_card(self) -> None:
        # 2026-04-15 12:30 UTC == 5:30 AM PDT
        started = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        report = TriageRunReport(
            run_id="zero-1",
            started_at=started,
            completed_at=started + timedelta(seconds=1),
            timezone="America/Los_Angeles",
            total_threads_fetched=0,
        )

        payloads = build_slack_payloads(report)

        assert len(payloads) == 1
        assert "Inbox Zero" in payloads[0]["text"]
        text = _all_block_text(payloads)
        assert "5:30 AM PDT" in text
        assert "Inbox Zero!" in text
        assert "0 model calls · GCP Cloud Run" in text

    def test_full_digest_formatting_badges_links_attachments_and_footer(self) -> None:
        started = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        completed = started + timedelta(seconds=14.6)

        urgent_item = TriagedThread(
            thread=make_email_thread(
                thread_id="thr-u1",
                subject="Legal & Contract <Urgent>",
                sender="Legal <Team>",
                received_at=datetime(2026, 4, 15, 11, 15, tzinfo=UTC),  # 4:15 AM PDT
            ),
            category=TriageCategory.URGENT,
            confidence=0.98,
            summary="Sign MSA amendment before noon.",
            high_stakes_review=HighStakesReview(
                refined_category=TriageCategory.URGENT,
                executive_summary="Sign MSA amendment before noon.",
                action_items=["Review indemnity clause", "Countersign DocuSign"],
                deadline="Today 12:00 PM PT",
                risk_assessment="Auto-renewal penalty if missed.",
                suggested_next_step="Approve clause 4 and sign.",
            ),
            deep_review_model="gemini-4-neon",
        )
        action_item = TriagedThread(
            thread=make_email_thread(
                thread_id="thr-a1",
                subject="Expense receipt review",
                sender="Finance Ops",
                received_at=datetime(2026, 4, 15, 10, 0, tzinfo=UTC),
            ),
            category=TriageCategory.ACTION_REQUIRED,
            confidence=0.91,
            summary="Confirm $480 hosting charge.",
            deep_analysis=DeepThreadAnalysis(
                refined_category=TriageCategory.ACTION_REQUIRED,
                confidence=0.91,
                summary="Confirm $480 hosting charge.",
                action_items=["Reply approved"],
                deadline="Tomorrow",
                attachment_findings=["receipt.png confirms $480.00 charge", "statement.pdf matches #INV-9921"],
                suggested_reply_points=["Approved for payment"],
            ),
        )
        tech_item = TriagedThread(
            thread=make_email_thread(thread_id="thr-t1", subject="CI Failed"),
            category=TriageCategory.TECHNICAL_ALERT,
            confidence=0.96,
            summary="CI failed",
            tech_extraction=TechExtraction(
                source_system="GitHub Actions",
                severity="critical",
                repo_or_service="acme/api",
                error_or_event="test_idempotency failed",
            ),
        )
        fyi_items = [
            TriagedThread(
                thread=make_email_thread(thread_id=f"thr-f{i}", subject=f"FYI #{i}", sender=f"Sender {i}"),
                category=TriageCategory.FYI,
                confidence=0.85,
                summary=f"FYI summary {i}",
            )
            for i in range(10)
        ]
        promo_item = TriagedThread(
            thread=make_email_thread(thread_id="thr-p1", subject="Sale!"),
            category=TriageCategory.PROMO_NOISE,
            confidence=0.99,
            summary="Promo",
        )

        report = TriageRunReport(
            run_id="run-full",
            started_at=started,
            completed_at=completed,
            timezone="America/Los_Angeles",
            total_threads_fetched=14,
            triaged_threads=[urgent_item, action_item, tech_item, *fyi_items, promo_item],
            briefing=ExecutiveBriefing(
                headline="Contract deadline at noon + critical CI failure.",
                executive_summary="Handle the MSA first, then unblock CI.",
                cross_thread_connections=["CloudHost invoice relates to Finance receipt review."],
            ),
            call_stats=[
                ModelCallStats(
                    backend="vertexai",
                    model="gemini-3.5-flash-lite",
                    stage="stage1",
                    prompt_tokens=1000,
                    completion_tokens=200,
                ),
                ModelCallStats(
                    backend="vertexai",
                    model="gemini-4-carbon",
                    stage="stage2",
                    prompt_tokens=2500,
                    completion_tokens=500,
                ),
                ModelCallStats(
                    backend="vertexai",
                    model="gemini-4-neon",
                    stage="stage3",
                    prompt_tokens=3000,
                    completion_tokens=800,
                ),
            ],
            deep_reviews_count=1,
        )

        payloads = build_slack_payloads(report, mention_on_urgent="<!here>")
        text = _all_block_text(payloads)

        # Mention on urgent & Pacific Time
        assert "<!here>" in text
        assert "5:30 AM PDT" in text
        assert "4:15 AM PDT" in text

        # Gmail deep links & Slack escaping
        assert "https://mail.google.com/mail/u/0/#inbox/thr-u1|Legal &amp; Contract &lt;Urgent&gt;" in text
        assert "Legal &lt;Team&gt;" in text

        # Neon review badge & high-stakes review fields
        assert "`✨ gemini-4-neon`" in text
        assert "Auto-renewal penalty if missed." in text
        assert "Approve clause 4 and sign." in text

        # Attachment findings & reply points from Stage 2 deep analysis
        assert "receipt.png confirms $480.00 charge" in text
        assert "Approved for payment" in text

        # Cross-thread connection & tech alert
        assert "CloudHost invoice relates to Finance receipt review." in text
        assert "`[CRITICAL]` *GitHub Actions*" in text

        # FYI capped at 8 with overflow indicator
        assert "…and 2 more FYI thread(s)." in text

        # Silent filter count & telemetry footer (1000+200+2500+500+3000+800 = 8,000 tokens)
        assert "0 newsletter(s), 1 promo/noise thread(s)" in text
        assert "Models: gemini-3.5-flash-lite, gemini-4-carbon, gemini-4-neon" in text
        assert "Tokens: 8,000" in text
        assert "Deep reviews: 1" in text
        assert "Runtime: 14.6s" in text

    def test_chunks_payloads_when_exceeding_50_blocks(self) -> None:
        started = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        # 65 urgent cards -> guaranteed > 50 blocks (will require 2 payloads)
        urgent_items = [
            TriagedThread(
                thread=make_email_thread(thread_id=f"thr-u-{i}", subject=f"Urgent #{i}"),
                category=TriageCategory.URGENT,
                confidence=0.95,
                summary=f"Summary {i}",
            )
            for i in range(65)
        ]
        report = TriageRunReport(
            run_id="run-big",
            started_at=started,
            completed_at=started + timedelta(seconds=5),
            timezone="America/Los_Angeles",
            total_threads_fetched=len(urgent_items),
            triaged_threads=urgent_items,
        )

        payloads = build_slack_payloads(report)

        assert len(payloads) == 2
        assert all(len(p["blocks"]) <= MAX_BLOCKS_PER_MESSAGE for p in payloads)
        assert sum(len(p["blocks"]) for p in payloads) > MAX_BLOCKS_PER_MESSAGE


class TestSlackWebhookClient:
    def test_empty_webhook_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="SLACK_WEBHOOK_URL"):
            SlackWebhookClient("")

    def test_retries_on_429_and_503_then_succeeds(self) -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "1.5"}, text="rate_limited"),
            httpx.Response(503, text="service_unavailable"),
            httpx.Response(200, text="ok"),
        ]
        calls: list[httpx.Request] = []

        def _transport_handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return responses.pop(0)

        http_client = httpx.Client(transport=httpx.MockTransport(_transport_handler))
        client = SlackWebhookClient(
            "https://hooks.slack.com/services/T000/B000/TEST",
            max_retries=3,
            http_client=http_client,
        )

        with patch("inbox_triage.slack.client.time.sleep") as mock_sleep:
            client.post_payloads([{"text": "hi", "blocks": []}])

        assert len(calls) == 3
        # Attempt 1 (429): Retry-After=1.5; Attempt 2 (503): backoff doubled from 1.0 to 2.0
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.5, 2.0]
        http_client.close()

    def test_raises_immediately_on_non_retryable_400(self) -> None:
        calls = 0

        def _transport_handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, text="invalid_blocks")

        http_client = httpx.Client(transport=httpx.MockTransport(_transport_handler))
        client = SlackWebhookClient(
            "https://hooks.slack.com/services/T000/B000/TEST",
            max_retries=3,
            http_client=http_client,
        )

        with patch("inbox_triage.slack.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="HTTP 400: invalid_blocks"):
                client.post_payloads([{"text": "bad", "blocks": []}])

        assert calls == 1
        mock_sleep.assert_not_called()
        http_client.close()

    def test_raises_after_exhausting_503_retries(self) -> None:
        http_client = httpx.Client(
            transport=httpx.MockTransport(lambda _r: httpx.Response(503, text="down"))
        )
        client = SlackWebhookClient(
            "https://hooks.slack.com/services/T000/B000/TEST",
            max_retries=2,
            http_client=http_client,
        )

        with patch("inbox_triage.slack.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="HTTP 503: down"):
                client.post_payloads([{"text": "hi", "blocks": []}])

        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0]
        http_client.close()
