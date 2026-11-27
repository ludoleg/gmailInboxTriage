"""Unit tests for Slack Block Kit formatting, interactive actions, and Slack clients."""

from __future__ import annotations

import json
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
from inbox_triage.slack.client import SlackBotClient, SlackWebhookClient
from inbox_triage.slack.formatter import (
    ACTION_ARCHIVE,
    ACTION_DRAFT_REPLY,
    ACTION_MUTE_SENDER,
    ACTION_TASK_EVENT,
    MAX_BLOCKS_PER_MESSAGE,
    build_slack_payloads,
    mark_card_action_in_blocks,
)
from tests.conftest import make_email_thread


def _all_block_text(payloads: list[dict]) -> str:
    chunks: list[str] = []
    for p in payloads:
        for b in p["blocks"]:
            if "text" in b:
                chunks.append(b["text"]["text"])
            for el in b.get("elements", []):
                txt = el.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
                elif isinstance(txt, dict) and isinstance(txt.get("text"), str):
                    chunks.append(txt["text"])
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
        assert "🧹 Filtered silently: 1 promo/noise." in text
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

    def test_interactive_mode_adds_four_buttons_per_priority_and_tech_card(self) -> None:
        started = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        long_subject = "Very Important Subject " + ("X" * 300)
        urgent = TriagedThread(
            thread=make_email_thread(
                thread_id="thr-u1",
                subject=long_subject,
                sender_email="ceo@acme.example",
            ),
            category=TriageCategory.URGENT,
            confidence=0.98,
            summary="Urgent summary",
        )
        action = TriagedThread(
            thread=make_email_thread(
                thread_id="thr-a1",
                subject="Approve Q2 budget",
                sender_email="finance@acme.example",
            ),
            category=TriageCategory.ACTION_REQUIRED,
            confidence=0.91,
            summary="Action summary",
        )
        tech = TriagedThread(
            thread=make_email_thread(
                thread_id="thr-t1",
                subject="CI failed on main",
                sender_email="ci@github.com",
            ),
            category=TriageCategory.TECHNICAL_ALERT,
            confidence=0.95,
            summary="CI summary",
        )
        fyi = TriagedThread(
            thread=make_email_thread(thread_id="thr-f1", subject="Office snacks"),
            category=TriageCategory.FYI,
            confidence=0.88,
            summary="FYI summary",
        )
        report = TriageRunReport(
            run_id="run-int",
            started_at=started,
            completed_at=started + timedelta(seconds=3),
            timezone="America/Los_Angeles",
            total_threads_fetched=4,
            muted_skipped_count=3,
            triaged_threads=[urgent, action, tech, fyi],
        )

        # Non-interactive omits actions blocks
        plain_payloads = build_slack_payloads(report, interactive=False)
        plain_actions = [
            b for p in plain_payloads for b in p["blocks"] if b.get("type") == "actions"
        ]
        assert plain_actions == []

        # Interactive emits 4 buttons per Urgent, Action, and Tech card (NOT FYI)
        payloads = build_slack_payloads(report, interactive=True)
        action_blocks = [b for p in payloads for b in p["blocks"] if b.get("type") == "actions"]
        assert [b["block_id"] for b in action_blocks] == [
            "actions_thr-u1",
            "actions_thr-a1",
            "actions_thr-t1",
        ]

        expected_action_ids = [
            ACTION_ARCHIVE,
            ACTION_DRAFT_REPLY,
            ACTION_TASK_EVENT,
            ACTION_MUTE_SENDER,
        ]
        for blk in action_blocks:
            elems = blk["elements"]
            assert len(elems) == 4
            assert [e["action_id"] for e in elems] == expected_action_ids
            for el in elems:
                val = el["value"]
                assert len(val) <= 2000
                parsed = json.loads(val)
                assert set(parsed.keys()) == {"tid", "sender", "subj"}
                assert len(parsed["subj"]) <= 80

        u1_val = json.loads(action_blocks[0]["elements"][0]["value"])
        assert u1_val["tid"] == "thr-u1"
        assert u1_val["sender"] == "ceo@acme.example"
        assert u1_val["subj"] == long_subject[:80]

        # Muted skipped count reflected in both counts header and filtered context footer
        text = _all_block_text(payloads)
        assert "3 Filtered" in text
        assert "3 muted sender(s)" in text

    def test_inbox_zero_shows_muted_skipped_count_when_nonzero(self) -> None:
        started = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        report = TriageRunReport(
            run_id="zero-muted",
            started_at=started,
            completed_at=started + timedelta(seconds=1),
            timezone="America/Los_Angeles",
            total_threads_fetched=0,
            muted_skipped_count=5,
        )
        payloads = build_slack_payloads(report, interactive=True)
        text = _all_block_text(payloads)
        assert "Muted skipped: 5" in text


class TestMarkCardActionInBlocks:
    def _sample_blocks(self, thread_id: str = "thr-1") -> list[dict]:
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Urgent card"}},
            {
                "type": "actions",
                "block_id": f"actions_{thread_id}",
                "elements": [{"type": "button", "action_id": ACTION_ARCHIVE, "value": "{}"}],
            },
            {"type": "divider"},
        ]

    def test_preserves_actions_and_inserts_status_badge(self) -> None:
        updated = mark_card_action_in_blocks(
            self._sample_blocks("thr-1"),
            "thr-1",
            "📝 Draft saved",
            remove_actions=False,
        )

        block_ids = [b.get("block_id") for b in updated]
        assert "actions_thr-1" in block_ids
        assert block_ids.count("status_thr-1") == 1
        status_blk = next(b for b in updated if b.get("block_id") == "status_thr-1")
        assert status_blk["elements"][0]["text"] == "📝 Draft saved"

    def test_removes_actions_when_requested(self) -> None:
        updated = mark_card_action_in_blocks(
            self._sample_blocks("thr-1"),
            "thr-1",
            "✅ Archived & marked read",
            remove_actions=True,
        )

        block_ids = [b.get("block_id") for b in updated]
        assert "actions_thr-1" not in block_ids
        assert block_ids.count("status_thr-1") == 1
        status_blk = next(b for b in updated if b.get("block_id") == "status_thr-1")
        assert status_blk["elements"][0]["text"] == "✅ Archived & marked read"

    def test_combines_multiple_badges_and_deduplicates_identical_badge(self) -> None:
        step1 = mark_card_action_in_blocks(
            self._sample_blocks("thr-1"),
            "thr-1",
            "📝 Draft saved",
            remove_actions=False,
        )
        step2 = mark_card_action_in_blocks(
            step1,
            "thr-1",
            "📅 Added to Calendar",
            remove_actions=False,
        )
        step3 = mark_card_action_in_blocks(
            step2,
            "thr-1",
            "✅ Archived & marked read",
            remove_actions=True,
        )
        step4 = mark_card_action_in_blocks(
            step3,
            "thr-1",
            "✅ Archived & marked read",
            remove_actions=True,
        )

        # Every intermediate and final state must have exactly ONE status_thr-1 block (unique block_id)
        for state in (step1, step2, step3, step4):
            status_blocks = [b for b in state if b.get("block_id") == "status_thr-1"]
            assert len(status_blocks) == 1

        step2_text = next(b for b in step2 if b.get("block_id") == "status_thr-1")["elements"][0]["text"]
        assert "📝 Draft saved" in step2_text
        assert "📅 Added to Calendar" in step2_text

        final_text = next(b for b in step4 if b.get("block_id") == "status_thr-1")["elements"][0]["text"]
        assert "📝 Draft saved" in final_text
        assert "📅 Added to Calendar" in final_text
        assert "✅ Archived & marked read" in final_text
        assert not any(b.get("block_id") == "actions_thr-1" for b in step4)

    def test_appends_status_when_actions_block_missing(self) -> None:
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Plain card"}}]
        updated = mark_card_action_in_blocks(blocks, "thr-9", "✅ Archived", remove_actions=True)

        assert updated[-1]["block_id"] == "status_thr-9"
        assert updated[-1]["elements"][0]["text"] == "✅ Archived"


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


class TestSlackBotClient:
    def test_empty_bot_token_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
            SlackBotClient("")

    def test_post_payloads_update_message_and_thread_reply(self) -> None:
        captured: list[tuple[str, str, dict]] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(
                (
                    str(request.url),
                    request.headers.get("Authorization", ""),
                    json.loads(request.content.decode("utf-8")),
                )
            )
            if str(request.url).endswith("/chat.postMessage"):
                return httpx.Response(200, json={"ok": True, "ts": f"1776254400.000{len(captured)}"})
            if str(request.url).endswith("/chat.update"):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(404)

        http_client = httpx.Client(transport=httpx.MockTransport(_handler))
        bot = SlackBotClient("xoxb-secret-token", http_client=http_client)

        ts_list = bot.post_payloads("C01TRIAGE", [{"text": "Digest", "blocks": [{"type": "divider"}]}])
        assert ts_list == ["1776254400.0001"]

        bot.update_message("C01TRIAGE", "1776254400.0001", "Updated", [{"type": "divider"}])
        reply_ts = bot.post_thread_reply("C01TRIAGE", "1776254400.0001", "Draft ready!")
        assert reply_ts == "1776254400.0003"

        assert len(captured) == 3
        assert captured[0][0] == "https://slack.com/api/chat.postMessage"
        assert captured[0][1] == "Bearer xoxb-secret-token"
        assert captured[0][2]["channel"] == "C01TRIAGE"
        assert captured[0][2]["unfurl_links"] is False
        assert captured[0][2]["unfurl_media"] is False

        assert captured[1][0] == "https://slack.com/api/chat.update"
        assert captured[1][2] == {
            "channel": "C01TRIAGE",
            "ts": "1776254400.0001",
            "text": "Updated",
            "blocks": [{"type": "divider"}],
        }

        assert captured[2][0] == "https://slack.com/api/chat.postMessage"
        assert captured[2][2]["thread_ts"] == "1776254400.0001"
        assert captured[2][2]["text"] == "Draft ready!"
        http_client.close()

    def test_raises_runtime_error_when_slack_returns_ok_false(self) -> None:
        http_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
            )
        )
        bot = SlackBotClient("xoxb-token", http_client=http_client)

        with pytest.raises(RuntimeError, match="channel_not_found"):
            bot.post_payloads("C_BAD", [{"text": "hi", "blocks": []}])
        http_client.close()

    def test_retries_on_429_then_succeeds(self) -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "1.25"}, text="ratelimited"),
            httpx.Response(200, json={"ok": True, "ts": "1776254400.9999"}),
        ]
        calls = 0

        def _handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return responses.pop(0)

        http_client = httpx.Client(transport=httpx.MockTransport(_handler))
        bot = SlackBotClient("xoxb-token", max_retries=3, http_client=http_client)

        with patch("inbox_triage.slack.client.time.sleep") as mock_sleep:
            ts = bot.post_thread_reply("C01", "1776254400.0001", "Reply")

        assert ts == "1776254400.9999"
        assert calls == 2
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.25]
        http_client.close()

    def test_post_response_url_success_and_error(self) -> None:
        statuses = [200, 500]
        http_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(statuses.pop(0), text="resp")
            )
        )
        bot = SlackBotClient("xoxb-token", http_client=http_client)

        bot.post_response_url("https://hooks.slack.com/actions/1", {"text": "ack"})
        with pytest.raises(RuntimeError, match="HTTP 500"):
            bot.post_response_url("https://hooks.slack.com/actions/1", {"text": "fail"})
        http_client.close()
