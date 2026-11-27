"""FastAPI server for Phase 2 Slack interactivity (/slack/interactions, /slack/commands, Cloud Tasks worker)."""

from __future__ import annotations

import hmac
import json
import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from inbox_triage.config import Settings
from inbox_triage.gmail.client import GmailClient
from inbox_triage.gmail.workspace import GoogleWorkspaceClient
from inbox_triage.pipeline.actions import ActionService
from inbox_triage.slack.client import SlackBotClient
from inbox_triage.slack.formatter import (
    ACTION_ARCHIVE,
    ACTION_DRAFT_REPLY,
    ACTION_MUTE_SENDER,
    ACTION_TASK_EVENT,
    mark_card_action_in_blocks,
)
from inbox_triage.slack.security import is_allowed_slack_user, verify_slack_signature
from inbox_triage.state import StateStore, create_state_store

logger = logging.getLogger(__name__)

_HOURS_RE = re.compile(r"^(\d{1,3})h?$", re.IGNORECASE)
_SLACK_MAILTO_RE = re.compile(r"^<mailto:([^|>]+)(?:\|[^>]+)?>$", re.IGNORECASE)


def _clean_slack_email(raw: str) -> str:
    """Strip Slack's `<mailto:user@domain|user@domain>` auto-linking wrapper if present."""
    token = raw.strip()
    match = _SLACK_MAILTO_RE.match(token)
    if match:
        token = match.group(1)
    return token.strip().lower()


class InternalActionPayload(BaseModel):
    action_id: str
    thread_id: str
    sender_email: str = ""
    subject: str = ""
    channel_id: str
    message_ts: str
    message_text: str = ""
    blocks: list[dict[str, Any]] = []


def _default_cloud_tasks_dispatcher(settings: Settings, payload: InternalActionPayload) -> None:
    """Enqueue an async action to Cloud Tasks targeting /internal/execute-action."""
    from google.cloud import tasks_v2

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(
        settings.gcp_project_id,
        settings.gcp_region,
        settings.cloud_tasks_queue or "triage-actions",
    )
    target_url = f"{(settings.cloud_run_service_url or '').rstrip('/')}/internal/execute-action"
    headers = {"Content-Type": "application/json"}
    if settings.internal_task_secret:
        headers["X-Internal-Task-Secret"] = settings.internal_task_secret

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": target_url,
            "headers": headers,
            "body": payload.model_dump_json().encode("utf-8"),
        }
    }
    client.create_task(parent=parent, task=task)
    logger.info("Enqueued Cloud Task for %s (thread=%s).", payload.action_id, payload.thread_id)


def _default_cloud_run_job_trigger(settings: Settings, since_hours: int | None) -> str:
    """Trigger the Cloud Run Job asynchronously via Cloud Run Jobs API."""
    from google.cloud import run_v2

    client = run_v2.JobsClient()
    job_name = (
        f"projects/{settings.gcp_project_id}/locations/{settings.gcp_region}"
        f"/jobs/{settings.cloud_run_job_name}"
    )
    overrides = None
    if since_hours is not None:
        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    args=["run", "--since-hours", str(since_hours), "--ignore-watermark"]
                )
            ]
        )
    req = run_v2.RunJobRequest(name=job_name, overrides=overrides)
    op = client.run_job(request=req)
    op_name = getattr(op, "operation", None)
    return str(getattr(op_name, "name", "triggered"))


def create_app(
    *,
    settings: Settings | None = None,
    store: StateStore | None = None,
    action_service: ActionService | None = None,
    slack_bot: SlackBotClient | None = None,
    task_dispatcher: Callable[[InternalActionPayload], None] | None = None,
    job_trigger: Callable[[int | None], str] | None = None,
) -> FastAPI:
    """Build the FastAPI application with injectable dependencies for clean testing."""
    cfg = settings or Settings()
    state_store = store or create_state_store(cfg)
    actions = action_service or ActionService(
        settings=cfg,
        store=state_store,
        gmail=GmailClient(cfg),
        workspace=GoogleWorkspaceClient(cfg),
    )
    bot = slack_bot or (SlackBotClient(cfg.slack_bot_token) if cfg.slack_bot_token else None)

    app = FastAPI(title="Gmail Inbox Triage Slack Bot", version="0.3.0")

    def _require_slack_auth(raw_body: bytes, timestamp: str | None, signature: str | None) -> None:
        if not cfg.slack_signing_secret:
            raise HTTPException(status_code=500, detail="SLACK_SIGNING_SECRET is not configured.")
        if not verify_slack_signature(
            signing_secret=cfg.slack_signing_secret,
            timestamp=timestamp,
            signature=signature,
            raw_body=raw_body,
        ):
            raise HTTPException(status_code=401, detail="Invalid Slack signature.")

    def _execute_heavy_action(payload: InternalActionPayload) -> None:
        try:
            if payload.action_id == ACTION_DRAFT_REPLY:
                outcome = actions.handle_draft_reply(payload.thread_id)
            elif payload.action_id == ACTION_TASK_EVENT:
                outcome = actions.handle_create_task_or_event(payload.thread_id)
            else:
                logger.warning("Unknown async action_id: %s", payload.action_id)
                return

            if bot is not None:
                bot.post_thread_reply(
                    channel_id=payload.channel_id,
                    thread_ts=payload.message_ts,
                    text=outcome.slack_message,
                )
                if payload.blocks:
                    new_blocks = mark_card_action_in_blocks(
                        payload.blocks,
                        payload.thread_id,
                        outcome.status_badge,
                        remove_actions=False,
                    )
                    bot.update_message(
                        channel_id=payload.channel_id,
                        ts=payload.message_ts,
                        text=payload.message_text or "Morning Inbox Triage",
                        blocks=new_blocks,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Async action %s failed for thread %s", payload.action_id, payload.thread_id)
            if bot is not None:
                try:
                    bot.post_thread_reply(
                        channel_id=payload.channel_id,
                        thread_ts=payload.message_ts,
                        text=f"❌ Action `{payload.action_id}` failed for thread `{payload.thread_id}`: {exc}",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to post error reply to Slack.")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/slack/interactions")
    async def slack_interactions(
        request: Request,
        background_tasks: BackgroundTasks,
        x_slack_request_timestamp: str | None = Header(default=None),
        x_slack_signature: str | None = Header(default=None),
    ) -> dict[str, Any]:
        raw_body = await request.body()
        _require_slack_auth(raw_body, x_slack_request_timestamp, x_slack_signature)

        form = parse_qs(raw_body.decode("utf-8"))
        raw_payload = (form.get("payload") or [""])[0]
        if not raw_payload:
            raise HTTPException(status_code=400, detail="Missing Slack payload.")

        try:
            data = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Malformed Slack payload JSON.") from exc
        user_id = str((data.get("user") or {}).get("id") or "")
        if not is_allowed_slack_user(user_id, cfg.allowed_slack_user_set):
            return {"response_type": "ephemeral", "text": "⛔ You are not authorized to run triage actions."}

        action_list = data.get("actions") or []
        if not action_list:
            return {"ok": True}

        act = action_list[0]
        action_id = str(act.get("action_id") or "")
        btn_val = json.loads(str(act.get("value") or "{}"))
        thread_id = str(btn_val.get("tid") or "")
        sender_email = str(btn_val.get("sender") or "")
        subject = str(btn_val.get("subj") or "")

        channel_id = str((data.get("channel") or {}).get("id") or cfg.slack_channel_id or "")
        message = data.get("message") or {}
        message_ts = str(message.get("ts") or "")
        message_text = str(message.get("text") or "Morning Inbox Triage")
        blocks = list(message.get("blocks") or [])
        response_url = str(data.get("response_url") or "")

        # Fast synchronous actions (<500ms)
        if action_id == ACTION_ARCHIVE:
            outcome = actions.handle_archive(thread_id)
            if bot is not None and channel_id and message_ts and blocks:
                new_blocks = mark_card_action_in_blocks(
                    blocks, thread_id, outcome.status_badge, remove_actions=True
                )
                bot.update_message(channel_id, message_ts, message_text, new_blocks)
            return {"ok": True, "text": outcome.slack_message}

        if action_id == ACTION_MUTE_SENDER:
            outcome = actions.handle_mute(sender_email)
            if bot is not None and channel_id and message_ts and blocks:
                new_blocks = mark_card_action_in_blocks(
                    blocks, thread_id, outcome.status_badge, remove_actions=True
                )
                bot.update_message(channel_id, message_ts, message_text, new_blocks)
            if bot is not None and channel_id and message_ts:
                bot.post_thread_reply(channel_id, message_ts, outcome.slack_message)
            return {"ok": True, "text": outcome.slack_message}

        # Heavy LLM/API actions -> Cloud Tasks (or local BackgroundTasks) for <300ms Slack ack
        if action_id in (ACTION_DRAFT_REPLY, ACTION_TASK_EVENT):
            task_payload = InternalActionPayload(
                action_id=action_id,
                thread_id=thread_id,
                sender_email=sender_email,
                subject=subject,
                channel_id=channel_id,
                message_ts=message_ts,
                message_text=message_text,
                blocks=blocks,
            )
            if task_dispatcher is not None:
                task_dispatcher(task_payload)
            elif cfg.cloud_tasks_queue and cfg.cloud_run_service_url:
                _default_cloud_tasks_dispatcher(cfg, task_payload)
            else:
                background_tasks.add_task(_execute_heavy_action, task_payload)

            ack_msg = (
                f"⏳ Drafting reply with `{cfg.model_stage3_reasoning}` for *{subject or thread_id}*… I'll post the Gmail Draft link in this thread shortly."
                if action_id == ACTION_DRAFT_REPLY
                else f"⏳ Extracting Calendar Event / Google Task with `{cfg.model_stage2_workhorse}` for *{subject or thread_id}*… I'll reply in this thread shortly."
            )
            if bot is not None and response_url:
                background_tasks.add_task(
                    bot.post_response_url,
                    response_url,
                    {"response_type": "ephemeral", "replace_original": False, "text": ack_msg},
                )
            return {"ok": True}

        return {"ok": True}

    @app.post("/slack/commands")
    async def slack_commands(
        request: Request,
        x_slack_request_timestamp: str | None = Header(default=None),
        x_slack_signature: str | None = Header(default=None),
    ) -> dict[str, Any]:
        raw_body = await request.body()
        _require_slack_auth(raw_body, x_slack_request_timestamp, x_slack_signature)

        form = parse_qs(raw_body.decode("utf-8"))
        user_id = (form.get("user_id") or [""])[0]
        if not is_allowed_slack_user(user_id, cfg.allowed_slack_user_set):
            return {"response_type": "ephemeral", "text": "⛔ You are not authorized to run `/triage`."}

        text = ((form.get("text") or [""])[0]).strip()
        parts = text.split()
        sub = parts[0].lower() if parts else ""

        if sub == "help":
            return {
                "response_type": "ephemeral",
                "text": (
                    "*`/triage` commands:*\n"
                    "• `/triage` — trigger an inbox triage using the latest watermark\n"
                    "• `/triage 48h` — triage unread threads from the last N hours (1–168)\n"
                    "• `/triage muted` — list muted senders\n"
                    "• `/triage unmute <email>` — remove a sender from the mute list"
                ),
            }

        if sub == "muted":
            muted = state_store.list_muted_senders()
            if not muted:
                return {"response_type": "ephemeral", "text": "No senders are currently muted."}
            lines = ["*🔇 Muted Senders:*"] + [f"• `{m.sender_email}`" for m in muted[:30]]
            return {"response_type": "ephemeral", "text": "\n".join(lines)}

        if sub == "unmute":
            if len(parts) < 2:
                return {"response_type": "ephemeral", "text": "Usage: `/triage unmute sender@example.com`"}
            target = _clean_slack_email(parts[1])
            removed = state_store.unmute_sender(target)
            msg = (
                f"🔊 Unmuted `{target}` — future emails will be triaged normally."
                if removed
                else f"`{target}` was not in the muted list."
            )
            return {"response_type": "ephemeral", "text": msg}

        since_hours: int | None = None
        if sub:
            match = _HOURS_RE.match(sub)
            if not match:
                return {
                    "response_type": "ephemeral",
                    "text": f"Unknown option `{text}`. Try `/triage`, `/triage 48h`, `/triage muted`, or `/triage help`.",
                }
            since_hours = max(1, min(168, int(match.group(1))))

        trigger_fn = job_trigger or (lambda h: _default_cloud_run_job_trigger(cfg, h))
        trigger_fn(since_hours)
        window_note = f" (last {since_hours}h)" if since_hours else ""
        return {
            "response_type": "ephemeral",
            "text": f"🔄 Triggered Cloud Run inbox triage{window_note} — results will post to this channel shortly!",
        }

    @app.post("/internal/execute-action")
    def internal_execute_action(
        payload: InternalActionPayload,
        x_internal_task_secret: str | None = Header(default=None),
    ) -> dict[str, Any]:
        expected = cfg.internal_task_secret
        if not expected or not x_internal_task_secret or not hmac.compare_digest(
            expected, x_internal_task_secret
        ):
            raise HTTPException(status_code=403, detail="Forbidden internal task request.")

        _execute_heavy_action(payload)
        return {"status": "completed", "action_id": payload.action_id, "thread_id": payload.thread_id}

    return app
