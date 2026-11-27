"""Phase 2 interactive action handlers (Archive, Mute, Draft Reply, Calendar Event / Google Task)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from inbox_triage.config import Settings
from inbox_triage.gmail.client import GmailClient
from inbox_triage.gmail.workspace import GoogleWorkspaceClient
from inbox_triage.llm.router import GenAIRouter
from inbox_triage.models import ReplyDraftResult, TaskOrEventSpec
from inbox_triage.pipeline import prompts
from inbox_triage.pipeline.orchestrator import _build_multimodal_contents, _format_thread_text
from inbox_triage.state import StateStore

logger = logging.getLogger(__name__)


@dataclass
class ActionOutcome:
    """Structured result returned by an interactive Slack action."""

    action: str
    status_badge: str
    slack_message: str
    link_url: str | None = None
    resource_id: str | None = None


class ActionService:
    """Executes the four interactive Slack actions against Gmail, Workspace, Firestore, and Gemini."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: StateStore,
        gmail: GmailClient,
        workspace: GoogleWorkspaceClient,
        router: GenAIRouter | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.gmail = gmail
        self.workspace = workspace
        self._router = router

    @property
    def router(self) -> GenAIRouter:
        if self._router is None:
            self._router = GenAIRouter(self.settings)
        return self._router

    def handle_archive(self, thread_id: str) -> ActionOutcome:
        """Archive a Gmail thread and mark it read."""
        self.gmail.archive_and_mark_read(thread_id)
        return ActionOutcome(
            action="archive",
            status_badge="✅ Archived & marked read",
            slack_message=f"✅ Archived and marked read (<https://mail.google.com/mail/u/0/#inbox/{thread_id}|view thread>).",
            resource_id=thread_id,
        )

    def handle_mute(self, sender_email: str, reason: str | None = None) -> ActionOutcome:
        """Add sender_email to the persistent muted-senders list."""
        norm = sender_email.strip().lower()
        self.store.mute_sender(norm, reason=reason or "Muted via Slack button")
        return ActionOutcome(
            action="mute",
            status_badge=f"🔇 Muted `{norm}`",
            slack_message=(
                f"🔇 Future emails from `{norm}` will be filtered automatically. "
                f"Undo anytime with `/triage unmute {norm}`."
            ),
            resource_id=norm,
        )

    def handle_draft_reply(self, thread_id: str) -> ActionOutcome:
        """Use gemini-4-neon to generate a reply draft and save it to Gmail Drafts (never sends)."""
        thread = self.gmail.get_thread(thread_id)
        prompt_text = (
            f"Current local time: {datetime.now(self.settings.tz).isoformat()}\n\n"
            + _format_thread_text(thread, max_body_chars=30000)
        )
        contents = _build_multimodal_contents(thread, prompt_text)

        draft_res, _stats = self.router.generate_structured(
            model=self.settings.model_stage3_reasoning,
            stage="action_draft_reply",
            system_prompt=prompts.DRAFT_REPLY_SYSTEM_PROMPT,
            contents=contents,
            response_schema=ReplyDraftResult,
            temperature=0.2,
            max_output_tokens=8192,
        )

        draft_id, draft_url = self.gmail.create_reply_draft(
            thread_id=thread.thread_id,
            to_email=thread.sender_email,
            subject=draft_res.subject or thread.subject,
            body_text=draft_res.body_text,
            in_reply_to=thread.rfc_message_id,
        )

        preview = draft_res.body_text.strip()
        if len(preview) > 900:
            preview = preview[:899].rstrip() + "…"

        lines = [
            f"📝 *Reply draft saved to Gmail Drafts* (not sent) — <{draft_url}|Open Draft in Gmail>",
            f"*To:* `{thread.sender_email}`  ·  *Subject:* {draft_res.subject}",
            f"*Tone:* {draft_res.tone_note}",
            f"```{preview}```",
        ]
        if draft_res.open_questions:
            qs = "\n".join(f"• {q}" for q in draft_res.open_questions[:4])
            lines.append(f"*Double-check before sending:*\n{qs}")

        return ActionOutcome(
            action="draft_reply",
            status_badge="📝 Draft saved",
            slack_message="\n".join(lines),
            link_url=draft_url,
            resource_id=draft_id,
        )

    def handle_create_task_or_event(self, thread_id: str) -> ActionOutcome:
        """Use gemini-4-carbon to extract a Calendar Event or Google Task and create it."""
        thread = self.gmail.get_thread(thread_id)
        now_local = datetime.now(self.settings.tz)
        prompt_text = (
            f"Current local time ({self.settings.timezone}): {now_local.isoformat()}\n"
            f"Default timezone: {self.settings.timezone}\n"
            f"Gmail thread URL: {thread.gmail_url}\n\n"
            + _format_thread_text(thread, max_body_chars=24000)
        )
        contents = _build_multimodal_contents(thread, prompt_text)

        spec, _stats = self.router.generate_structured(
            model=self.settings.model_stage2_workhorse,
            stage="action_task_or_event",
            system_prompt=prompts.TASK_OR_EVENT_SYSTEM_PROMPT,
            contents=contents,
            response_schema=TaskOrEventSpec,
            temperature=0.1,
            max_output_tokens=4096,
        )

        # Ensure thread link is attached in notes
        if thread.gmail_url not in spec.notes:
            spec.notes = f"{spec.notes}\n\nEmail thread: {thread.gmail_url}".strip()

        if spec.kind == "calendar_event":
            event_id, link = self.workspace.create_calendar_event(spec)
            when = f"{spec.start_iso or 'TBD'} → {spec.end_iso or '+1h'}"
            loc = f"\n*Location:* {spec.location}" if spec.location else ""
            msg = (
                f"📅 *Google Calendar event created* — <{link}|Open in Calendar>\n"
                f"*{spec.title}*\n*When:* `{when}`{loc}"
            )
            return ActionOutcome(
                action="calendar_event",
                status_badge="📅 Added to Calendar",
                slack_message=msg,
                link_url=link,
                resource_id=event_id,
            )

        task_id, link = self.workspace.create_google_task(spec)
        due = f"\n*Due:* `{spec.due_date}`" if spec.due_date else ""
        msg = (
            f"☑️ *Google Task created* — <{link}|Open Google Tasks>\n"
            f"*{spec.title}*{due}"
        )
        return ActionOutcome(
            action="google_task",
            status_badge="☑️ Added to Tasks",
            slack_message=msg,
            link_url=link,
            resource_id=task_id,
        )
