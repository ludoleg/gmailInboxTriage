"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the Gmail inbox triage service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Locale & Scheduling
    timezone: str = Field(default="America/Los_Angeles", alias="TIMEZONE")

    # Primary Local Engine: Inkling-2-Mini (sm_61 CUDA server)
    inkling_enabled: bool = Field(default=True, alias="INKLING_ENABLED")
    inkling_base_url: str = Field(default="http://127.0.0.1:8001", alias="INKLING_BASE_URL")
    model_inkling: str = Field(default="inkling-2-mini", alias="MODEL_INKLING")
    inkling_ctl_path: Path = Field(
        default=Path("/home/ludo/.local/bin/inkling-ctl"),
        alias="INKLING_CTL_PATH",
    )
    inkling_context_limit: int = Field(default=32768, alias="INKLING_CONTEXT_LIMIT")
    inkling_overflow_chars: int = Field(default=84000, alias="INKLING_OVERFLOW_CHARS")
    gpu_restore_policy: Literal["restore", "keep_inkling", "idle"] = Field(
        default="restore",
        alias="GPU_RESTORE_POLICY",
    )

    # Conditional Multimodal & Fallback Engine: Ollama
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    model_multimodal: str = Field(default="gemma4:12b", alias="MODEL_MULTIMODAL")
    model_fallback_triage: str = Field(default="gemma4:e2b", alias="MODEL_FALLBACK_TRIAGE")
    model_fallback_coder: str = Field(default="qwen2.5-coder:3b", alias="MODEL_FALLBACK_CODER")
    model_fallback_synthesis: str = Field(default="bonsai-2:27b", alias="MODEL_FALLBACK_SYNTHESIS")

    # Selective Cloud Escalation: Google Gemini 4 (via google-genai)
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    model_cloud_fast: str = Field(default="gemini-4-carbon", alias="MODEL_CLOUD_FAST")
    model_cloud_deep: str = Field(default="gemini-4-neon", alias="MODEL_CLOUD_DEEP")
    cloud_escalation_enabled: bool = Field(default=True, alias="CLOUD_ESCALATION_ENABLED")
    cloud_confidence_threshold: float = Field(default=0.70, alias="CLOUD_CONFIDENCE_THRESHOLD")
    max_cloud_escalations_per_run: int = Field(default=5, alias="MAX_CLOUD_ESCALATIONS_PER_RUN")
    cloud_denylist_domains: str = Field(default="", alias="CLOUD_DENYLIST_DOMAINS")

    # Gmail Read-Only OAuth
    gmail_credentials_path: Path = Field(
        default=Path("/home/ludo/gmailInboxTriage/credentials.json"),
        alias="GMAIL_CREDENTIALS_PATH",
    )
    gmail_token_path: Path = Field(
        default=Path.home() / ".local" / "share" / "gmail-inbox-triage" / "token.json",
        alias="GMAIL_TOKEN_PATH",
    )
    gmail_query_base: str = Field(default="in:inbox is:unread", alias="GMAIL_QUERY_BASE")
    gmail_max_threads: int = Field(default=60, alias="GMAIL_MAX_THREADS")
    default_lookback_hours: int = Field(default=24, alias="DEFAULT_LOOKBACK_HOURS")

    # Slack Incoming Webhook
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")
    slack_mention_on_urgent: str | None = Field(default=None, alias="SLACK_MENTION_ON_URGENT")

    # Persistence
    state_db_path: Path = Field(
        default=Path.home() / ".local" / "share" / "gmail-inbox-triage" / "state.db",
        alias="STATE_DB_PATH",
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        ZoneInfo(v)  # raises ZoneInfoNotFoundError if invalid
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def denylist_domain_set(self) -> set[str]:
        return {
            d.strip().lower().lstrip("@")
            for d in self.cloud_denylist_domains.split(",")
            if d.strip()
        }
