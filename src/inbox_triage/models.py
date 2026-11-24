"""Domain & structured-output Pydantic models for the triage pipeline."""

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
    extracted_text: str | None = None
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
    snippet: str = ""
    body_text: str = ""
    has_list_unsubscribe: bool = False
    is_auto_submitted: bool = False
    labels: list[str] = Field(default_factory=list)
    attachments: list[EmailAttachment] = Field(default_factory=list)

    @property
    def has_multimodal_attachments(self) -> bool:
        return any(
            a.kind in (AttachmentKind.IMAGE, AttachmentKind.AUDIO) and a.data_bytes
            for a in self.attachments
        )

    @property
    def gmail_url(self) -> str:
        return f"https://mail.google.com/mail/u/0/#inbox/{self.thread_id}"

    @property
    def total_char_length(self) -> int:
        pdf_chars = sum(len(a.extracted_text or "") for a in self.attachments)
        return len(self.subject) + len(self.body_text) + pdf_chars


# --- Structured LLM Output Schemas ---


class MultimodalFinding(BaseModel):
    filename: str = Field(description="Attachment filename.")
    summary: str = Field(description="Concise description or transcription of the attachment.")
    extracted_key_facts: list[str] = Field(
        default_factory=list,
        description="Key facts such as invoice amounts, dates, error codes, or action requests.",
    )
    urgency_signal: Literal["none", "normal", "high"] = Field(
        default="none",
        description="Whether the attachment content indicates urgency.",
    )


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
        description="True if this requires detailed action-item, deadline, or reply-point extraction.",
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
    suggested_reply_points: list[str] = Field(
        default_factory=list,
        description="Short bullet points for how to respond, if a reply is needed.",
    )
    escalation_recommended: bool = Field(
        default=False,
        description="True only if high-stakes ambiguity or complex legal/financial nuance needs frontier review.",
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Why frontier cloud escalation is recommended, if applicable.",
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
    cloud_escalation_thread_ids: list[str] = Field(
        default_factory=list,
        description="Thread IDs that genuinely warrant Cloud Gemini 4 escalation.",
    )


class CloudEscalationResult(BaseModel):
    refined_category: TriageCategory = Field(description="Final high-confidence triage category.")
    executive_summary: str = Field(description="Authoritative 2-sentence summary.")
    action_items: list[str] = Field(default_factory=list, description="Concrete next actions.")
    deadline: str | None = Field(default=None, description="Exact deadline if present.")
    risk_assessment: str = Field(description="Key risks, commitments, or nuances to watch.")
    suggested_next_step: str = Field(description="Single clearest next step for the user.")


# --- Aggregated Pipeline Artifacts ---


class ModelCallStats(BaseModel):
    backend: Literal["inkling", "ollama", "cloud"]
    model: str
    stage: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False


class TriagedThread(BaseModel):
    thread: EmailThread
    category: TriageCategory
    confidence: float
    summary: str
    stage1: Stage1Classification | None = None
    multimodal_findings: list[MultimodalFinding] = Field(default_factory=list)
    tech_extraction: TechExtraction | None = None
    deep_analysis: DeepThreadAnalysis | None = None
    cloud_escalation: CloudEscalationResult | None = None
    cloud_model_used: str | None = None
    error_note: str | None = None


class TriageRunReport(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: datetime
    timezone: str
    total_threads_fetched: int
    triaged_threads: list[TriagedThread] = Field(default_factory=list)
    briefing: ExecutiveBriefing | None = None
    call_stats: list[ModelCallStats] = Field(default_factory=list)
    used_inkling: bool = False
    used_ollama_multimodal: bool = False
    used_ollama_fallback: bool = False
    cloud_escalations_count: int = 0
    cloud_blocked_by_denylist: int = 0

    def by_category(self, category: TriageCategory) -> list[TriagedThread]:
        return [t for t in self.triaged_threads if t.category == category]
