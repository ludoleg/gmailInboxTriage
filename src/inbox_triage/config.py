"""Application configuration for GCP Cloud Run Jobs/Services and local development."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the serverless GCP Gmail inbox triage & interactive Slack bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Locale & Scheduling
    timezone: str = Field(default="America/Los_Angeles", alias="TIMEZONE")

    # GCP & GenAI Backend
    gcp_project_id: str = Field(default="gmail-inbox-triage-515705", alias="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-west1", alias="GCP_REGION")
    use_vertexai: bool = Field(default=False, alias="USE_VERTEXAI")
    vertex_location: str = Field(default="global", alias="VERTEX_LOCATION")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    # Cloud Run & Cloud Tasks (Phase 2 Interactive Bot)
    cloud_run_job_name: str = Field(default="gmail-inbox-triage", alias="CLOUD_RUN_JOB_NAME")
    cloud_run_service_url: str | None = Field(default=None, alias="CLOUD_RUN_SERVICE_URL")
    cloud_tasks_queue: str | None = Field(default=None, alias="CLOUD_TASKS_QUEUE")
    internal_task_secret: str | None = Field(default=None, alias="INTERNAL_TASK_SECRET")

    # 3-Tier Strength-Matched Model Funnel
    model_stage1_fast: str = Field(default="gemini-3.5-flash-lite", alias="MODEL_STAGE1_FAST")
    model_stage2_workhorse: str = Field(default="gemini-4-carbon", alias="MODEL_STAGE2_WORKHORSE")
    model_stage3_reasoning: str = Field(default="gemini-4-neon", alias="MODEL_STAGE3_REASONING")
    max_concurrency: int = Field(default=6, ge=1, le=20, alias="MAX_CONCURRENCY")
    deep_review_confidence_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0, alias="DEEP_REVIEW_CONFIDENCE_THRESHOLD"
    )
    max_deep_reviews_per_run: int = Field(default=5, ge=0, alias="MAX_DEEP_REVIEWS_PER_RUN")

    # State Persistence
    state_backend: Literal["firestore", "sqlite"] = Field(default="firestore", alias="STATE_BACKEND")
    firestore_database: str = Field(default="(default)", alias="FIRESTORE_DATABASE")
    state_db_path: Path = Field(
        default=Path.home() / ".local" / "share" / "gmail-inbox-triage" / "state.db",
        alias="STATE_DB_PATH",
    )

    # Google Workspace OAuth (gmail.modify + calendar.events + tasks)
    gmail_token_json: str | None = Field(default=None, alias="GMAIL_TOKEN_JSON")
    gmail_credentials_path: Path = Field(
        default=Path("/home/ludo/projects/gmailInboxTriage/credentials.json"),
        alias="GMAIL_CREDENTIALS_PATH",
    )
    gmail_token_path: Path = Field(
        default=Path.home() / ".local" / "share" / "gmail-inbox-triage" / "token.json",
        alias="GMAIL_TOKEN_PATH",
    )
    gmail_query_base: str = Field(default="in:inbox is:unread", alias="GMAIL_QUERY_BASE")
    gmail_max_threads: int = Field(default=60, alias="GMAIL_MAX_THREADS")
    default_lookback_hours: int = Field(default=24, alias="DEFAULT_LOOKBACK_HOURS")

    # Slack Delivery & Interactivity
    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_channel_id: str | None = Field(default=None, alias="SLACK_CHANNEL_ID")
    slack_signing_secret: str | None = Field(default=None, alias="SLACK_SIGNING_SECRET")
    slack_allowed_user_ids: str = Field(default="", alias="SLACK_ALLOWED_USER_IDS")
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")
    slack_mention_on_urgent: str | None = Field(default=None, alias="SLACK_MENTION_ON_URGENT")

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        ZoneInfo(v)
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def allowed_slack_user_set(self) -> set[str]:
        return {u.strip() for u in self.slack_allowed_user_ids.split(",") if u.strip()}

    @property
    def has_slack_bot(self) -> bool:
        return bool(self.slack_bot_token and self.slack_channel_id)
