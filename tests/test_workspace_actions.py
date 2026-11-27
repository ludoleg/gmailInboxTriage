"""Unit tests for Gmail write actions, GoogleWorkspaceClient, and ActionService."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from email import message_from_bytes
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inbox_triage.config import Settings
from inbox_triage.gmail.client import GmailClient
from inbox_triage.gmail.workspace import GoogleWorkspaceClient
from inbox_triage.models import (
    ModelCallStats,
    ReplyDraftResult,
    TaskOrEventSpec,
)
from inbox_triage.pipeline.actions import ActionService
from inbox_triage.state import SQLiteStateStore
from tests.conftest import make_email_thread


def _dummy_stats(model: str, stage: str) -> ModelCallStats:
    return ModelCallStats(
        backend="gemini_api",
        model=model,
        stage=stage,
        prompt_tokens=120,
        completion_tokens=60,
        latency_ms=85.0,
    )


class TestGmailClientWriteActions:
    def test_archive_and_mark_read_removes_inbox_and_unread_labels(
        self, test_settings: Settings
    ) -> None:
        service = MagicMock()
        client = GmailClient(test_settings, service=service)

        client.archive_and_mark_read("thr-archive-1")

        service.users.return_value.threads.return_value.modify.assert_called_once_with(
            userId="me",
            id="thr-archive-1",
            body={"removeLabelIds": ["INBOX", "UNREAD"]},
        )
        service.users.return_value.threads.return_value.modify.return_value.execute.assert_called_once()

    def test_create_reply_draft_builds_rfc2822_headers_and_never_sends(
        self, test_settings: Settings
    ) -> None:
        service = MagicMock()
        drafts_api = service.users.return_value.drafts.return_value
        drafts_api.create.return_value.execute.return_value = {
            "id": "r-draft-99",
            "message": {"id": "m-draft-99", "threadId": "thr-reply-1"},
        }

        client = GmailClient(test_settings, service=service)
        draft_id, draft_url = client.create_reply_draft(
            thread_id="thr-reply-1",
            to_email="maya.lin@partner.example",
            subject="Q2 Roadmap Review",
            body_text="Hi Maya,\n\n10:00 AM PT tomorrow works great!\n\nBest,",
            in_reply_to="<msg-105@partner.example>",
        )

        assert draft_id == "r-draft-99"
        assert draft_url == "https://mail.google.com/mail/u/0/#drafts?compose=m-draft-99"

        drafts_api.create.assert_called_once()
        kwargs = drafts_api.create.call_args.kwargs
        assert kwargs["userId"] == "me"
        msg_body = kwargs["body"]["message"]
        assert msg_body["threadId"] == "thr-reply-1"

        decoded_bytes = base64.urlsafe_b64decode(msg_body["raw"].encode("ascii"))
        parsed = message_from_bytes(decoded_bytes)
        assert parsed["To"] == "maya.lin@partner.example"
        assert parsed["Subject"] == "Re: Q2 Roadmap Review"
        assert parsed["In-Reply-To"] == "<msg-105@partner.example>"
        assert parsed["References"] == "<msg-105@partner.example>"
        assert "10:00 AM PT tomorrow works great!" in parsed.get_payload(decode=True).decode("utf-8")

        # Crucial safety invariant: must NEVER auto-send emails or drafts
        drafts_api.send.assert_not_called()
        service.users.return_value.messages.return_value.send.assert_not_called()

    @pytest.mark.parametrize("existing_subject", ["Re: Q2 Review", "RE: Q2 Review", "re: Q2 Review"])
    def test_create_reply_draft_avoids_double_re_prefix_and_omits_optional_headers(
        self,
        test_settings: Settings,
        existing_subject: str,
    ) -> None:
        service = MagicMock()
        drafts_api = service.users.return_value.drafts.return_value
        drafts_api.create.return_value.execute.return_value = {"id": "r-draft-2"}

        client = GmailClient(test_settings, service=service)
        draft_id, draft_url = client.create_reply_draft(
            thread_id="thr-reply-2",
            to_email="alice@example.com",
            subject=existing_subject,
            body_text="Sounds good!",
            in_reply_to=None,
        )

        assert draft_id == "r-draft-2"
        assert draft_url == "https://mail.google.com/mail/u/0/#drafts?compose=r-draft-2"

        raw = drafts_api.create.call_args.kwargs["body"]["message"]["raw"]
        parsed = message_from_bytes(base64.urlsafe_b64decode(raw.encode("ascii")))
        assert parsed["Subject"] == existing_subject
        assert parsed["In-Reply-To"] is None
        assert parsed["References"] is None
        drafts_api.send.assert_not_called()
        service.users.return_value.messages.return_value.send.assert_not_called()


class TestGoogleWorkspaceClient:
    def test_create_calendar_event_with_explicit_iso_times_and_location(
        self, test_settings: Settings
    ) -> None:
        cal_service = MagicMock()
        cal_service.events.return_value.insert.return_value.execute.return_value = {
            "id": "evt-123",
            "htmlLink": "https://calendar.google.com/calendar/event?eid=evt-123",
        }
        client = GoogleWorkspaceClient(test_settings, calendar_service=cal_service)

        spec = TaskOrEventSpec(
            kind="calendar_event",
            title="Q2 Roadmap Review",
            notes="Confirm Q2 goals with Maya.",
            start_iso="2026-04-16T10:00:00-07:00",
            end_iso="2026-04-16T11:30:00-07:00",
            location="https://meet.google.com/abc-defg-hij",
            timezone="America/Los_Angeles",
        )
        event_id, html_link = client.create_calendar_event(spec)

        assert event_id == "evt-123"
        assert html_link == "https://calendar.google.com/calendar/event?eid=evt-123"

        insert_kwargs = cal_service.events.return_value.insert.call_args.kwargs
        assert insert_kwargs["calendarId"] == "primary"
        body = insert_kwargs["body"]
        assert body["summary"] == "Q2 Roadmap Review"
        assert body["description"] == "Confirm Q2 goals with Maya."
        assert body["location"] == "https://meet.google.com/abc-defg-hij"
        assert body["start"] == {
            "dateTime": "2026-04-16T10:00:00-07:00",
            "timeZone": "America/Los_Angeles",
        }
        assert body["end"] == {
            "dateTime": "2026-04-16T11:30:00-07:00",
            "timeZone": "America/Los_Angeles",
        }

    def test_create_calendar_event_defaults_end_to_plus_one_hour_when_omitted(
        self, test_settings: Settings
    ) -> None:
        cal_service = MagicMock()
        cal_service.events.return_value.insert.return_value.execute.return_value = {"id": "evt-456"}
        client = GoogleWorkspaceClient(test_settings, calendar_service=cal_service)

        spec = TaskOrEventSpec(
            kind="calendar_event",
            title="Dentist Appointment",
            notes="Bring insurance card.",
            start_iso="2026-04-17T14:00:00-07:00",
            end_iso=None,
            location=None,
            timezone="America/Los_Angeles",
        )
        event_id, html_link = client.create_calendar_event(spec)

        assert event_id == "evt-456"
        assert html_link == "https://calendar.google.com/"

        body = cal_service.events.return_value.insert.call_args.kwargs["body"]
        assert "location" not in body
        assert body["start"]["dateTime"] == "2026-04-17T14:00:00-07:00"
        assert body["end"]["dateTime"] == "2026-04-17T15:00:00-07:00"

    def test_create_calendar_event_fixes_non_positive_duration_and_naive_datetime(
        self, test_settings: Settings
    ) -> None:
        cal_service = MagicMock()
        cal_service.events.return_value.insert.return_value.execute.return_value = {
            "id": "evt-789",
            "htmlLink": "https://calendar.google.com/event/789",
        }
        client = GoogleWorkspaceClient(test_settings, calendar_service=cal_service)

        spec = TaskOrEventSpec(
            kind="calendar_event",
            title="Flight Check-in",
            start_iso="2026-04-18T09:00:00",
            end_iso="2026-04-18T08:30:00",
            timezone="America/Los_Angeles",
        )
        client.create_calendar_event(spec)

        body = cal_service.events.return_value.insert.call_args.kwargs["body"]
        start_dt = datetime.fromisoformat(body["start"]["dateTime"])
        end_dt = datetime.fromisoformat(body["end"]["dateTime"])
        assert start_dt.tzinfo is not None
        assert end_dt - start_dt == timedelta(hours=1)

    def test_create_google_task_with_due_date_formats_rfc3339(
        self, test_settings: Settings
    ) -> None:
        tasks_service = MagicMock()
        tasks_service.tasks.return_value.insert.return_value.execute.return_value = {
            "id": "task-101"
        }
        client = GoogleWorkspaceClient(test_settings, tasks_service=tasks_service)

        spec = TaskOrEventSpec(
            kind="google_task",
            title="Pay CloudHost invoice #INV-9921",
            notes="Amount: $480.00",
            due_date="2026-04-20",
        )
        task_id, tasks_url = client.create_google_task(spec)

        assert task_id == "task-101"
        assert tasks_url == "https://tasks.google.com/"
        tasks_service.tasks.return_value.insert.assert_called_once_with(
            tasklist="@default",
            body={
                "title": "Pay CloudHost invoice #INV-9921",
                "notes": "Amount: $480.00",
                "due": "2026-04-20T00:00:00.000Z",
            },
        )

    def test_create_google_task_without_due_date_omits_due_field(
        self, test_settings: Settings
    ) -> None:
        tasks_service = MagicMock()
        tasks_service.tasks.return_value.insert.return_value.execute.return_value = {
            "id": "task-102"
        }
        client = GoogleWorkspaceClient(test_settings, tasks_service=tasks_service)

        spec = TaskOrEventSpec(
            kind="google_task",
            title="Review Q2 hiring plan",
            notes="Share comments in doc.",
            due_date=None,
        )
        task_id, tasks_url = client.create_google_task(spec)

        assert task_id == "task-102"
        assert tasks_url == "https://tasks.google.com/"
        tasks_service.tasks.return_value.insert.assert_called_once_with(
            tasklist="@default",
            body={
                "title": "Review Q2 hiring plan",
                "notes": "Share comments in doc.",
            },
        )


class TestActionService:
    def test_handle_archive(self, test_settings: Settings, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        gmail = MagicMock(spec=GmailClient)
        workspace = MagicMock(spec=GoogleWorkspaceClient)

        svc = ActionService(settings=test_settings, store=store, gmail=gmail, workspace=workspace)
        outcome = svc.handle_archive("thr-abc")

        gmail.archive_and_mark_read.assert_called_once_with("thr-abc")
        assert outcome.action == "archive"
        assert "Archived" in outcome.status_badge
        assert outcome.resource_id == "thr-abc"
        assert "https://mail.google.com/mail/u/0/#inbox/thr-abc" in outcome.slack_message

    def test_handle_mute_normalizes_and_persists_sender(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        gmail = MagicMock(spec=GmailClient)
        workspace = MagicMock(spec=GoogleWorkspaceClient)

        svc = ActionService(settings=test_settings, store=store, gmail=gmail, workspace=workspace)
        outcome = svc.handle_mute("  Promo@Deals.EXAMPLE  ")

        assert store.get_muted_senders() == {"promo@deals.example"}
        assert outcome.action == "mute"
        assert outcome.resource_id == "promo@deals.example"
        assert "promo@deals.example" in outcome.status_badge
        assert "/triage unmute promo@deals.example" in outcome.slack_message

    def test_handle_draft_reply_uses_gemini_4_neon_and_saves_draft(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        thread = make_email_thread(
            thread_id="thr-draft-1",
            rfc_message_id="<orig-42@partner.example>",
            subject="Contract Renewal Terms",
            sender="Maya Lin",
            sender_email="maya.lin@partner.example",
            body_text="Could you confirm the renewal terms by Friday?",
        )

        gmail = MagicMock(spec=GmailClient)
        gmail.get_thread.return_value = thread
        gmail.create_reply_draft.return_value = (
            "draft-77",
            "https://mail.google.com/mail/u/0/#drafts?compose=msg-77",
        )

        router = MagicMock()
        router.generate_structured.return_value = (
            ReplyDraftResult(
                subject="Re: Contract Renewal Terms",
                body_text="Hi Maya,\n\nThanks for sending this over. We will review by [CONFIRM DATE].\n\nBest,",
                tone_note="Concise and professional",
                open_questions=["Confirm exact sign-off date before sending"],
            ),
            _dummy_stats("gemini-4-neon", "action_draft_reply"),
        )

        svc = ActionService(
            settings=test_settings,
            store=store,
            gmail=gmail,
            workspace=MagicMock(spec=GoogleWorkspaceClient),
            router=router,
        )
        outcome = svc.handle_draft_reply("thr-draft-1")

        gmail.get_thread.assert_called_once_with("thr-draft-1")
        router.generate_structured.assert_called_once()
        router_kwargs = router.generate_structured.call_args.kwargs
        assert router_kwargs["model"] == "gemini-4-neon"
        assert router_kwargs["stage"] == "action_draft_reply"
        assert router_kwargs["response_schema"] is ReplyDraftResult

        gmail.create_reply_draft.assert_called_once_with(
            thread_id="thr-draft-1",
            to_email="maya.lin@partner.example",
            subject="Re: Contract Renewal Terms",
            body_text="Hi Maya,\n\nThanks for sending this over. We will review by [CONFIRM DATE].\n\nBest,",
            in_reply_to="<orig-42@partner.example>",
        )

        assert outcome.action == "draft_reply"
        assert outcome.status_badge == "📝 Draft saved"
        assert outcome.resource_id == "draft-77"
        assert outcome.link_url == "https://mail.google.com/mail/u/0/#drafts?compose=msg-77"
        assert "not sent" in outcome.slack_message
        assert "Open Draft in Gmail" in outcome.slack_message
        assert "Confirm exact sign-off date before sending" in outcome.slack_message

    def test_handle_create_task_or_event_calendar_event_via_gemini_4_carbon(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        thread = make_email_thread(
            thread_id="thr-cal-1",
            subject="Flight Confirmation SFO -> JFK",
            body_text="Your flight departs Apr 22 at 8:00 AM PDT.",
        )

        gmail = MagicMock(spec=GmailClient)
        gmail.get_thread.return_value = thread

        workspace = MagicMock(spec=GoogleWorkspaceClient)
        workspace.create_calendar_event.return_value = (
            "evt-900",
            "https://calendar.google.com/event?eid=evt-900",
        )

        router = MagicMock()
        router.generate_structured.return_value = (
            TaskOrEventSpec(
                kind="calendar_event",
                title="Flight SFO -> JFK",
                notes="Confirmation #XYZ123",
                start_iso="2026-04-22T08:00:00-07:00",
                end_iso="2026-04-22T16:30:00-04:00",
                location="SFO Terminal 2",
                timezone="America/Los_Angeles",
            ),
            _dummy_stats("gemini-4-carbon", "action_task_or_event"),
        )

        svc = ActionService(
            settings=test_settings,
            store=store,
            gmail=gmail,
            workspace=workspace,
            router=router,
        )
        outcome = svc.handle_create_task_or_event("thr-cal-1")

        router_kwargs = router.generate_structured.call_args.kwargs
        assert router_kwargs["model"] == "gemini-4-carbon"
        assert router_kwargs["stage"] == "action_task_or_event"
        assert router_kwargs["response_schema"] is TaskOrEventSpec

        workspace.create_calendar_event.assert_called_once()
        workspace.create_google_task.assert_not_called()
        passed_spec: TaskOrEventSpec = workspace.create_calendar_event.call_args.args[0]
        assert thread.gmail_url in passed_spec.notes

        assert outcome.action == "calendar_event"
        assert outcome.status_badge == "📅 Added to Calendar"
        assert outcome.resource_id == "evt-900"
        assert outcome.link_url == "https://calendar.google.com/event?eid=evt-900"
        assert "Flight SFO -> JFK" in outcome.slack_message
        assert "SFO Terminal 2" in outcome.slack_message

    def test_handle_create_task_or_event_google_task_via_gemini_4_carbon(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        thread = make_email_thread(
            thread_id="thr-task-1",
            subject="Please approve vendor invoice",
            body_text="Approve invoice #8812 by April 25.",
        )

        gmail = MagicMock(spec=GmailClient)
        gmail.get_thread.return_value = thread

        workspace = MagicMock(spec=GoogleWorkspaceClient)
        workspace.create_google_task.return_value = ("task-333", "https://tasks.google.com/")

        router = MagicMock()
        router.generate_structured.return_value = (
            TaskOrEventSpec(
                kind="google_task",
                title="Approve vendor invoice #8812",
                notes="Check line items before approving.",
                due_date="2026-04-25",
            ),
            _dummy_stats("gemini-4-carbon", "action_task_or_event"),
        )

        svc = ActionService(
            settings=test_settings,
            store=store,
            gmail=gmail,
            workspace=workspace,
            router=router,
        )
        outcome = svc.handle_create_task_or_event("thr-task-1")

        router_kwargs = router.generate_structured.call_args.kwargs
        assert router_kwargs["model"] == "gemini-4-carbon"
        assert router_kwargs["stage"] == "action_task_or_event"
        assert router_kwargs["response_schema"] is TaskOrEventSpec

        workspace.create_google_task.assert_called_once()
        workspace.create_calendar_event.assert_not_called()
        passed_spec: TaskOrEventSpec = workspace.create_google_task.call_args.args[0]
        assert thread.gmail_url in passed_spec.notes

        assert outcome.action == "google_task"
        assert outcome.status_badge == "☑️ Added to Tasks"
        assert outcome.resource_id == "task-333"
        assert outcome.link_url == "https://tasks.google.com/"
        assert "Approve vendor invoice #8812" in outcome.slack_message
        assert "2026-04-25" in outcome.slack_message
