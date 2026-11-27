"""Google Calendar & Google Tasks client reusing the project's Workspace OAuth credentials."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from googleapiclient.discovery import build

from inbox_triage.config import Settings
from inbox_triage.gmail.auth import get_gmail_credentials
from inbox_triage.models import TaskOrEventSpec

logger = logging.getLogger(__name__)


class GoogleWorkspaceClient:
    """Creates Google Calendar events and Google Tasks from structured TaskOrEventSpec objects."""

    def __init__(
        self,
        settings: Settings,
        calendar_service: Any | None = None,
        tasks_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self._calendar_service = calendar_service
        self._tasks_service = tasks_service

    def _creds(self) -> Any:
        return get_gmail_credentials(
            self.settings.gmail_token_path,
            token_json=self.settings.gmail_token_json,
        )

    @property
    def calendar_service(self) -> Any:
        if self._calendar_service is None:
            self._calendar_service = build("calendar", "v3", credentials=self._creds(), cache_discovery=False)
        return self._calendar_service

    @property
    def tasks_service(self) -> Any:
        if self._tasks_service is None:
            self._tasks_service = build("tasks", "v1", credentials=self._creds(), cache_discovery=False)
        return self._tasks_service

    def create_calendar_event(self, spec: TaskOrEventSpec) -> tuple[str, str]:
        """Create a Google Calendar event on the primary calendar; returns (event_id, html_link)."""
        tz_name = spec.timezone or self.settings.timezone
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            tz_name = self.settings.timezone
            tz = ZoneInfo(tz_name)

        if spec.start_iso:
            start_dt = datetime.fromisoformat(spec.start_iso)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
        else:
            start_dt = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        if spec.end_iso:
            end_dt = datetime.fromisoformat(spec.end_iso)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(hours=1)
        else:
            end_dt = start_dt + timedelta(hours=1)

        body: dict[str, Any] = {
            "summary": spec.title,
            "description": spec.notes,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": tz_name},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": tz_name},
        }
        if spec.location:
            body["location"] = spec.location

        created = (
            self.calendar_service.events()
            .insert(calendarId="primary", body=body)
            .execute()
        )
        event_id = str(created.get("id") or "")
        html_link = str(created.get("htmlLink") or "https://calendar.google.com/")
        logger.info("Created Google Calendar event %s (%s).", event_id, spec.title)
        return event_id, html_link

    def create_google_task(self, spec: TaskOrEventSpec) -> tuple[str, str]:
        """Create a Google Task in the user's @default tasklist; returns (task_id, tasks_url)."""
        body: dict[str, Any] = {
            "title": spec.title,
            "notes": spec.notes,
        }
        if spec.due_date:
            # Google Tasks API expects RFC 3339 timestamp with UTC Z suffix
            clean_date = spec.due_date.strip()[:10]
            body["due"] = f"{clean_date}T00:00:00.000Z"

        created = (
            self.tasks_service.tasks()
            .insert(tasklist="@default", body=body)
            .execute()
        )
        task_id = str(created.get("id") or "")
        tasks_url = "https://tasks.google.com/"
        logger.info("Created Google Task %s (%s).", task_id, spec.title)
        return task_id, tasks_url
