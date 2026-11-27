"""Google Workspace OAuth2 authentication supporting Secret Manager JSON env and local token file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"
TASKS_SCOPE = "https://www.googleapis.com/auth/tasks"

# Minimal Phase 2 scopes: read/archive/draft Gmail (cannot permanently delete), create Calendar events & Tasks.
SCOPES = [
    GMAIL_MODIFY_SCOPE,
    CALENDAR_EVENTS_SCOPE,
    TASKS_SCOPE,
]


def _write_token_secure(token_path: Path, token_json: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token_json)
    os.chmod(token_path, 0o600)


def run_oauth_flow(
    credentials_path: Path,
    token_path: Path,
    *,
    port: int = 8085,
    open_browser: bool | None = None,
) -> Credentials:
    """Run one-time OAuth2 installed-app authorization flow and persist token.json."""
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Google OAuth client secrets not found at {credentials_path}. "
            "Download Desktop App credentials.json from Google Cloud Console."
        )
    if open_browser is None:
        open_browser = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    creds = flow.run_local_server(port=port, open_browser=open_browser)
    _write_token_secure(token_path, creds.to_json())
    return creds


def get_gmail_credentials(
    token_path: Path,
    *,
    token_json: str | None = None,
) -> Credentials:
    """Load OAuth2 credentials from Secret Manager env (GMAIL_TOKEN_JSON) or local token.json.

    When loaded from `token_json`, expired access tokens refresh in memory without writing to disk.
    Does not force a strict scope subset during load so existing read-only tokens continue working
    for read-only runs until the user re-runs `triage auth` for Phase 2 write scopes.
    """
    from_env = bool(token_json and token_json.strip())
    if from_env:
        info = json.loads(token_json.strip())  # type: ignore[union-attr]
        creds = Credentials.from_authorized_user_info(info)
    else:
        if not token_path.exists():
            raise FileNotFoundError(
                f"Gmail OAuth token not found at {token_path} and GMAIL_TOKEN_JSON is unset. "
                "Run `uv run triage auth` first or inject GMAIL_TOKEN_JSON from Secret Manager."
            )
        creds = Credentials.from_authorized_user_file(str(token_path))

    if (not creds.valid or creds.expired) and creds.refresh_token:
        creds.refresh(Request())
        if not from_env:
            _write_token_secure(token_path, creds.to_json())

    if not creds.valid:
        raise RuntimeError(
            "Gmail OAuth token is invalid and cannot be refreshed. "
            "Re-run `uv run triage auth` and update the Secret Manager secret."
        )
    return creds
