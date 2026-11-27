"""Domain & structured-output Pydantic models for the GCP 3-tier triage & interactive actions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class TriageCategory(StrEnum):
    URGENT = "URGENT"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    TECHNICAL_ALERT = "TECHNICAL_ALERT"
    FYI = "FYI"
    NEWSLETTER = "NEWSLETTER"
    PROMO_NOISE = "PROMO_NOISE"


class AttachmentKind(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    OTHER = "other"


class EmailAttachment(BaseModel):
    filename: str
    mime_type: str
    kind: AttachmentKind
    size_bytes: int
    data_bytes: bytes | None = Field(default=None, exclude=True)
    note: str | None = None


class EmailThread(BaseModel):
    thread_id: str
    last_message_id: str
    subject: str
    sender: str
    sender_email: str
    sender_domain: str
    received_at: datetime
    message_count: int = 1
    rfc_message_id: str | None = None
    snippet: str = ""
    body_text: str = ""
    has_list_unsubscribe: bool = False
    is_auto_submitted: bool = False
    labels: list[str] = Field(default_factory=list)
    attachments: list[EmailAttachment] = Field(default_factory=list)

    @property
    def multimodal_attachments(self) -> list[EmailAttachment]:
        return [
            a
            for a in self.attachments
            if a.kind in (AttachmentKind.IMAGE, AttachmentKind.AUDIO, AttachmentKind.PDF)
            and a.data_bytes
        ]

    @property
    def gmail_url(self) -> str:
        return f"https://mail.google.com/mail/u/0/#inbox/{self.thread_id}"


# --- Structured LLM Output Schemas ---


class Stage1Classification(BaseModel):
    category: TriageCategory = Field(description="Primary triage category for the email thread.")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence from 0.0 to 1.0.")
    one_line_summary: str = Field(description="Single concise sentence summarizing the thread.")
    needs_tech_extract: bool = Field(
        default=False,
        description="True if this is a CI/CD, GitHub, monitoring, or structured automated alert.",
    )
    needs_deep_analysis: bool = Field(
        default=False,
        description="True if this requires detailed action-item, attachment, deadline, or reply extraction.",
    )
    reason: str = Field(description="Brief justification for the chosen category.")


class TechExtraction(BaseModel):
    source_system: str = Field(description="Originating system (e.g., GitHub, Sentry, AWS, Stripe).")
    severity: Literal["critical", "high", "medium", "low", "info"] = Field(
        default="medium",
        description="Operational severity of the alert or event.",
    )
    repo_or_service: str = Field(default="", description="Repository, service, or vendor name.")
    error_or_event: str = Field(description="Concise summary of the failure, PR, or notification.")
    action_url: str | None = Field(default=None, description="Primary link to act on if present.")
    deadline_or_amount: str | None = Field(
        default=None,
        description="Due date, invoice amount, or SLA deadline if present.",
    )


class DeepThreadAnalysis(BaseModel):
    refined_category: TriageCategory = Field(description="Confirmed or refined triage category.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this analysis.")
    summary: str = Field(description="Clear 1-2 sentence summary for a busy executive.")
    action_items: list[str] = Field(
        default_factory=list,
        description="Concrete actions the user needs to take.",
    )
    deadline: str | None = Field(default=None, description="Explicit or implied deadline, if any.")
    attachment_findings: list[str] = Field(
        default_factory=list,
        description="Key facts extracted from any attached images, audio, or PDFs.",
    )
    suggested_reply_points: list[str] = Field(
        default_factory=list,
        description="Short bullet points for how to respond, if a reply is needed.",
    )
    deep_review_recommended: bool = Field(
        default=False,
        description="True only if high-stakes ambiguity or complex legal/financial nuance warrants Stage 3 Neon review.",
    )
    deep_review_reason: str | None = Field(
        default=None,
        description="Why Stage 3 frontier review is recommended, if applicable.",
    )


class ExecutiveBriefing(BaseModel):
    headline: str = Field(description="One-line morning overview headline.")
    executive_summary: str = Field(
        description="2-3 sentence synthesis of the most important themes this morning.",
    )
    top_priority_thread_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of the top 3-5 thread_ids the user should tackle first.",
    )
    cross_thread_connections: list[str] = Field(
        default_factory=list,
        description="Notable connections or dependencies across separate threads, if any.",
    )
    deep_review_thread_ids: list[str] = Field(
        default_factory=list,
        description="Thread IDs that genuinely warrant Stage 3 HighStakesReview.",
    )


class HighStakesReview(BaseModel):
    refined_category: TriageCategory = Field(description="Final high-confidence triage category.")
    executive_summary: str = Field(description="Authoritative 2-sentence summary.")
    action_items: list[str] = Field(default_factory=list, description="Concrete next actions.")
    deadline: str | None = Field(default=None, description="Exact deadline if present.")
    risk_assessment: str = Field(description="Key risks, commitments, or nuances to watch.")
    suggested_next_step: str = Field(description="Single clearest next step for the user.")


# --- Phase 2 Interactive Action Schemas ---


class ReplyDraftResult(BaseModel):
    subject: str = Field(description="Email subject line (typically starting with 'Re: ').")
    body_text: str = Field(description="Plain-text reply body ready to save as a Gmail Draft.")
    tone_note: str = Field(description="Short note describing the reply tone and intent.")
    open_questions: list[str] = Field(
        default_factory=list,
        description="Any details the user should double-check before clicking Send in Gmail.",
    )


class TaskOrEventSpec(BaseModel):
    kind: Literal["calendar_event", "google_task"] = Field(
        description="Use 'calendar_event' when the email describes a scheduled event with a time window (meeting, flight, hotel, appointment); otherwise 'google_task'."
    )
    title: str = Field(description="Concise event or task title.")
    notes: str = Field(default="", description="Relevant context, confirmation numbers, or action details.")
    start_iso: str | None = Field(
        default=None,
        description="ISO-8601 start datetime with timezone offset for calendar_event (e.g. 2026-12-13T15:00:00-08:00).",
    )
    end_iso: str | None = Field(
        default=None,
        description="ISO-8601 end datetime with timezone offset for calendar_event.",
    )
    due_date: str | None = Field(
        default=None,
        description="Optional YYYY-MM-DD due date for google_task.",
    )
    location: str | None = Field(default=None, description="Physical location or meeting link if present.")
    timezone: str = Field(default="America/Los_Angeles", description="IANA timezone for calendar events.")


class MutedSender(BaseModel):
    sender_email: str
    muted_at: datetime
    reason: str | None = None


# --- Aggregated Pipeline Artifacts ---


class ModelCallStats(BaseModel):
    backend: Literal["vertexai", "gemini_api"]
    model: str
    stage: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class TriagedThread(BaseModel):
    thread: EmailThread
    category: TriageCategory
    confidence: float
    summary: str
    stage1: Stage1Classification | None = None
    tech_extraction: TechExtraction | None = None
    deep_analysis: DeepThreadAnalysis | None = None
    high_stakes_review: HighStakesReview | None = None
    deep_review_model: str | None = None
    error_note: str | None = None


class TriageRunReport(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: datetime
    timezone: str
    total_threads_fetched: int
    muted_skipped_count: int = 0
    triaged_threads: list[TriagedThread] = Field(default_factory=list)
    briefing: ExecutiveBriefing | None = None
    call_stats: list[ModelCallStats] = Field(default_factory=list)
    deep_reviews_count: int = 0

    def by_category(self, category: TriageCategory) -> list[TriagedThread]:
        return [t for t in self.triaged_threads if t.category == category]

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.call_stats)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.call_stats)
