"""Unit tests for the Phase 2 FastAPI server (/healthz, /slack/*, /internal/execute-action)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlencode

from fastapi.testclient import TestClient

from inbox_triage.config import Settings
from inbox_triage.pipeline.actions import ActionOutcome
from inbox_triage.server import InternalActionPayload, create_app
from inbox_triage.slack.formatter import (
    ACTION_ARCHIVE,
    ACTION_DRAFT_REPLY,
    ACTION_MUTE_SENDER,
    ACTION_TASK_EVENT,
)
from inbox_triage.state import SQLiteStateStore

SIGNING_SECRET = "test-slack-signing-secret"
INTERNAL_SECRET = "test-internal-task-secret"


def _slack_headers(secret: str, raw_body: bytes, *, ts: int | None = None) -> dict[str, str]:
    req_ts = int(time.time()) if ts is None else ts
    basestring = f"v0:{req_ts}:".encode() + raw_body
    sig = "v0=" + hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Slack-Request-Timestamp": str(req_ts),
        "X-Slack-Signature": sig,
    }


def _interaction_body(
    *,
    action_id: str,
    thread_id: str = "thr-101",
    sender_email: str = "sender@example.com",
    subject: str = "Important Subject",
    user_id: str = "U01ALLOWED",
    channel_id: str = "C01TRIAGE",
    message_ts: str = "1776254400.000100",
    response_url: str = "https://hooks.slack.com/actions/T000/123/abc",
    blocks: list[dict[str, Any]] | None = None,
) -> bytes:
    default_blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "Card text"}},
        {
            "type": "actions",
            "block_id": f"actions_{thread_id}",
            "elements": [{"type": "button", "action_id": action_id, "value": "{}"}],
        },
    ]
    payload = {
        "type": "block_actions",
        "user": {"id": user_id},
        "channel": {"id": channel_id},
        "response_url": response_url,
        "message": {
            "ts": message_ts,
            "text": "Morning Inbox Triage",
            "blocks": default_blocks if blocks is None else blocks,
        },
        "actions": [
            {
                "action_id": action_id,
                "value": json.dumps(
                    {"tid": thread_id, "sender": sender_email, "subj": subject},
                    separators=(",", ":"),
                ),
            }
        ],
    }
    return urlencode({"payload": json.dumps(payload)}).encode("utf-8")


def _command_body(text: str, *, user_id: str = "U01ALLOWED") -> bytes:
    return urlencode({"command": "/triage", "text": text, "user_id": user_id}).encode("utf-8")


def _server_settings(test_settings: Settings, *, allowed_users: str = "") -> Settings:
    return test_settings.model_copy(
        update={
            "slack_bot_token": "xoxb-test-token",
            "slack_channel_id": "C01TRIAGE",
            "slack_signing_secret": SIGNING_SECRET,
            "slack_allowed_user_ids": allowed_users,
            "internal_task_secret": INTERNAL_SECRET,
            "cloud_tasks_queue": None,
            "cloud_run_service_url": None,
        }
    )


class TestHealthz:
    def test_healthz_returns_ok(self, test_settings: Settings, tmp_path: Path) -> None:
        cfg = _server_settings(test_settings)
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=MagicMock(),
            slack_bot=MagicMock(),
        )
        client = TestClient(app)

        resp = client.get("/healthz")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSlackInteractions:
    def test_rejects_invalid_signature_with_401(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=MagicMock(),
        )
        client = TestClient(app)

        body = _interaction_body(action_id=ACTION_ARCHIVE)
        resp = client.post(
            "/slack/interactions",
            content=body,
            headers=_slack_headers("wrong-secret", body),
        )

        assert resp.status_code == 401
        actions.handle_archive.assert_not_called()

    def test_blocks_disallowed_user(self, test_settings: Settings, tmp_path: Path) -> None:
        cfg = _server_settings(test_settings, allowed_users="U01OWNER,U02ADMIN")
        actions = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=MagicMock(),
        )
        client = TestClient(app)

        body = _interaction_body(action_id=ACTION_ARCHIVE, user_id="U99STRANGER")
        resp = client.post(
            "/slack/interactions",
            content=body,
            headers=_slack_headers(SIGNING_SECRET, body),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["response_type"] == "ephemeral"
        assert "not authorized" in data["text"].lower()
        actions.handle_archive.assert_not_called()

    def test_archive_executes_synchronously_and_updates_blocks(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        actions.handle_archive.return_value = ActionOutcome(
            action="archive",
            status_badge="✅ Archived & marked read",
            slack_message="✅ Archived and marked read.",
            resource_id="thr-arch-1",
        )
        bot = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
        )
        client = TestClient(app)

        body = _interaction_body(action_id=ACTION_ARCHIVE, thread_id="thr-arch-1")
        resp = client.post(
            "/slack/interactions",
            content=body,
            headers=_slack_headers(SIGNING_SECRET, body),
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        actions.handle_archive.assert_called_once_with("thr-arch-1")
        bot.update_message.assert_called_once()
        upd_args = bot.update_message.call_args.args
        assert upd_args[0] == "C01TRIAGE"
        assert upd_args[1] == "1776254400.000100"
        updated_blocks = upd_args[3]
        assert not any(b.get("block_id") == "actions_thr-arch-1" for b in updated_blocks)
        assert any(b.get("block_id") == "status_thr-arch-1" for b in updated_blocks)

    def test_mute_sender_executes_synchronously_updates_blocks_and_replies(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        actions.handle_mute.return_value = ActionOutcome(
            action="mute",
            status_badge="🔇 Muted `spam@noisy.example`",
            slack_message="🔇 Future emails from `spam@noisy.example` will be filtered.",
            resource_id="spam@noisy.example",
        )
        bot = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
        )
        client = TestClient(app)

        body = _interaction_body(
            action_id=ACTION_MUTE_SENDER,
            thread_id="thr-mute-1",
            sender_email="spam@noisy.example",
        )
        resp = client.post(
            "/slack/interactions",
            content=body,
            headers=_slack_headers(SIGNING_SECRET, body),
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        actions.handle_mute.assert_called_once_with("spam@noisy.example")
        bot.update_message.assert_called_once()
        bot.post_thread_reply.assert_called_once_with(
            "C01TRIAGE",
            "1776254400.000100",
            "🔇 Future emails from `spam@noisy.example` will be filtered.",
        )

    def test_draft_reply_and_task_event_dispatch_via_task_dispatcher(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        bot = MagicMock()
        dispatched: list[InternalActionPayload] = []

        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
            task_dispatcher=dispatched.append,
        )
        client = TestClient(app)

        for action_id, tid in ((ACTION_DRAFT_REPLY, "thr-d1"), (ACTION_TASK_EVENT, "thr-t1")):
            body = _interaction_body(action_id=action_id, thread_id=tid, subject="Urgent Follow-up")
            resp = client.post(
                "/slack/interactions",
                content=body,
                headers=_slack_headers(SIGNING_SECRET, body),
            )
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}

        assert [(p.action_id, p.thread_id) for p in dispatched] == [
            (ACTION_DRAFT_REPLY, "thr-d1"),
            (ACTION_TASK_EVENT, "thr-t1"),
        ]
        # Must NOT run heavy LLM actions inline when dispatched
        actions.handle_draft_reply.assert_not_called()
        actions.handle_create_task_or_event.assert_not_called()
        # Ephemeral acks sent via response_url
        assert bot.post_response_url.call_count == 2
        first_ack = bot.post_response_url.call_args_list[0].args[1]["text"]
        second_ack = bot.post_response_url.call_args_list[1].args[1]["text"]
        assert "gemini-4-neon" in first_ack
        assert "gemini-4-carbon" in second_ack

    def test_draft_reply_falls_back_to_background_tasks_locally(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        actions.handle_draft_reply.return_value = ActionOutcome(
            action="draft_reply",
            status_badge="📝 Draft saved",
            slack_message="📝 Reply draft saved.",
            link_url="https://mail.google.com/mail/u/0/#drafts?compose=d1",
            resource_id="d1",
        )
        bot = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
            task_dispatcher=None,
        )
        client = TestClient(app)

        body = _interaction_body(action_id=ACTION_DRAFT_REPLY, thread_id="thr-bg-1")
        resp = client.post(
            "/slack/interactions",
            content=body,
            headers=_slack_headers(SIGNING_SECRET, body),
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        actions.handle_draft_reply.assert_called_once_with("thr-bg-1")
        bot.post_thread_reply.assert_called_once_with(
            channel_id="C01TRIAGE",
            thread_ts="1776254400.000100",
            text="📝 Reply draft saved.",
        )
        bot.update_message.assert_called_once()


class TestSlackCommands:
    def test_rejects_invalid_signature_with_401(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        trigger = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=MagicMock(),
            slack_bot=MagicMock(),
            job_trigger=trigger,
        )
        client = TestClient(app)

        body = _command_body("")
        resp = client.post(
            "/slack/commands",
            content=body,
            headers=_slack_headers("bad-secret", body),
        )

        assert resp.status_code == 401
        trigger.assert_not_called()

    def test_default_triage_and_hours_override_trigger_job(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        triggered_hours: list[int | None] = []
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=MagicMock(),
            slack_bot=MagicMock(),
            job_trigger=lambda h: (triggered_hours.append(h), "op-1")[1],
        )
        client = TestClient(app)

        # /triage (default watermark)
        b1 = _command_body("")
        r1 = client.post("/slack/commands", content=b1, headers=_slack_headers(SIGNING_SECRET, b1))
        assert r1.status_code == 200
        assert r1.json()["response_type"] == "ephemeral"
        assert "Triggered Cloud Run inbox triage" in r1.json()["text"]

        # /triage 48h
        b2 = _command_body("48h")
        r2 = client.post("/slack/commands", content=b2, headers=_slack_headers(SIGNING_SECRET, b2))
        assert r2.status_code == 200
        assert "(last 48h)" in r2.json()["text"]

        # /triage 999h -> clamped to 168h
        b3 = _command_body("999h")
        r3 = client.post("/slack/commands", content=b3, headers=_slack_headers(SIGNING_SECRET, b3))
        assert r3.status_code == 200
        assert "(last 168h)" in r3.json()["text"]

        assert triggered_hours == [None, 48, 168]

    def test_muted_and_unmute_subcommands(self, test_settings: Settings, tmp_path: Path) -> None:
        cfg = _server_settings(test_settings)
        store = SQLiteStateStore(tmp_path / "state.db")
        trigger = MagicMock()
        app = create_app(
            settings=cfg,
            store=store,
            action_service=MagicMock(),
            slack_bot=MagicMock(),
            job_trigger=trigger,
        )
        client = TestClient(app)

        # Initially empty
        b_empty = _command_body("muted")
        r_empty = client.post(
            "/slack/commands", content=b_empty, headers=_slack_headers(SIGNING_SECRET, b_empty)
        )
        assert r_empty.status_code == 200
        assert "No senders are currently muted" in r_empty.json()["text"]

        # Populate mute list and list again
        store.mute_sender("promo@shop.example", reason="noise")
        b_list = _command_body("muted")
        r_list = client.post(
            "/slack/commands", content=b_list, headers=_slack_headers(SIGNING_SECRET, b_list)
        )
        assert r_list.status_code == 200
        assert "promo@shop.example" in r_list.json()["text"]

        # Unmute missing arg -> usage hint
        b_usage = _command_body("unmute")
        r_usage = client.post(
            "/slack/commands", content=b_usage, headers=_slack_headers(SIGNING_SECRET, b_usage)
        )
        assert "Usage:" in r_usage.json()["text"]

        # Unmute existing sender
        b_unmute = _command_body("unmute PROMO@shop.example")
        r_unmute = client.post(
            "/slack/commands", content=b_unmute, headers=_slack_headers(SIGNING_SECRET, b_unmute)
        )
        assert r_unmute.status_code == 200
        assert "Unmuted `promo@shop.example`" in r_unmute.json()["text"]
        assert store.get_muted_senders() == set()

        # Unmute absent sender
        b_absent = _command_body("unmute nobody@shop.example")
        r_absent = client.post(
            "/slack/commands", content=b_absent, headers=_slack_headers(SIGNING_SECRET, b_absent)
        )
        assert "was not in the muted list" in r_absent.json()["text"]
        trigger.assert_not_called()

    def test_help_and_unknown_subcommand(self, test_settings: Settings, tmp_path: Path) -> None:
        cfg = _server_settings(test_settings)
        trigger = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=MagicMock(),
            slack_bot=MagicMock(),
            job_trigger=trigger,
        )
        client = TestClient(app)

        b_help = _command_body("help")
        r_help = client.post(
            "/slack/commands", content=b_help, headers=_slack_headers(SIGNING_SECRET, b_help)
        )
        assert r_help.status_code == 200
        assert "/triage 48h" in r_help.json()["text"]
        assert "/triage muted" in r_help.json()["text"]

        b_unknown = _command_body("foobar")
        r_unknown = client.post(
            "/slack/commands", content=b_unknown, headers=_slack_headers(SIGNING_SECRET, b_unknown)
        )
        assert r_unknown.status_code == 200
        assert "Unknown option" in r_unknown.json()["text"]
        trigger.assert_not_called()


class TestInternalExecuteAction:
    def test_rejects_missing_or_wrong_secret_with_403(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=MagicMock(),
        )
        client = TestClient(app)

        payload = {
            "action_id": ACTION_DRAFT_REPLY,
            "thread_id": "thr-1",
            "channel_id": "C01TRIAGE",
            "message_ts": "1776254400.000100",
        }

        r_missing = client.post("/internal/execute-action", json=payload)
        assert r_missing.status_code == 403

        r_wrong = client.post(
            "/internal/execute-action",
            json=payload,
            headers={"X-Internal-Task-Secret": "wrong-secret"},
        )
        assert r_wrong.status_code == 403
        actions.handle_draft_reply.assert_not_called()

    def test_executes_draft_reply_and_posts_thread_reply_plus_block_update(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        actions.handle_draft_reply.return_value = ActionOutcome(
            action="draft_reply",
            status_badge="📝 Draft saved",
            slack_message="📝 *Reply draft saved to Gmail Drafts*",
            link_url="https://mail.google.com/mail/u/0/#drafts?compose=m1",
            resource_id="draft-1",
        )
        bot = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
        )
        client = TestClient(app)

        payload = {
            "action_id": ACTION_DRAFT_REPLY,
            "thread_id": "thr-draft-9",
            "channel_id": "C01TRIAGE",
            "message_ts": "1776254400.000200",
            "message_text": "Morning Inbox Triage",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "Urgent card"}},
                {"type": "actions", "block_id": "actions_thr-draft-9", "elements": []},
            ],
        }
        resp = client.post(
            "/internal/execute-action",
            json=payload,
            headers={"X-Internal-Task-Secret": INTERNAL_SECRET},
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "status": "completed",
            "action_id": ACTION_DRAFT_REPLY,
            "thread_id": "thr-draft-9",
        }
        actions.handle_draft_reply.assert_called_once_with("thr-draft-9")
        bot.post_thread_reply.assert_called_once_with(
            channel_id="C01TRIAGE",
            thread_ts="1776254400.000200",
            text="📝 *Reply draft saved to Gmail Drafts*",
        )
        bot.update_message.assert_called_once()
        upd_kwargs = bot.update_message.call_args.kwargs
        assert upd_kwargs["channel_id"] == "C01TRIAGE"
        assert upd_kwargs["ts"] == "1776254400.000200"
        # Actions block is preserved for non-destructive draft action
        assert any(b.get("block_id") == "actions_thr-draft-9" for b in upd_kwargs["blocks"])
        assert any(b.get("block_id") == "status_thr-draft-9" for b in upd_kwargs["blocks"])

    def test_executes_task_event_when_authorized(
        self, test_settings: Settings, tmp_path: Path
    ) -> None:
        cfg = _server_settings(test_settings)
        actions = MagicMock()
        actions.handle_create_task_or_event.return_value = ActionOutcome(
            action="calendar_event",
            status_badge="📅 Added to Calendar",
            slack_message="📅 *Google Calendar event created*",
            link_url="https://calendar.google.com/event/1",
            resource_id="evt-1",
        )
        bot = MagicMock()
        app = create_app(
            settings=cfg,
            store=SQLiteStateStore(tmp_path / "state.db"),
            action_service=actions,
            slack_bot=bot,
        )
        client = TestClient(app)

        payload = {
            "action_id": ACTION_TASK_EVENT,
            "thread_id": "thr-cal-5",
            "channel_id": "C01TRIAGE",
            "message_ts": "1776254400.000300",
            "blocks": [
                {"type": "actions", "block_id": "actions_thr-cal-5", "elements": []},
            ],
        }
        resp = client.post(
            "/internal/execute-action",
            json=payload,
            headers={"X-Internal-Task-Secret": INTERNAL_SECRET},
        )

        assert resp.status_code == 200
        assert resp.json()["action_id"] == ACTION_TASK_EVENT
        actions.handle_create_task_or_event.assert_called_once_with("thr-cal-5")
        bot.post_thread_reply.assert_called_once_with(
            channel_id="C01TRIAGE",
            thread_ts="1776254400.000300",
            text="📅 *Google Calendar event created*",
        )
        bot.update_message.assert_called_once()
