"""Gmail OAuth2 authentication strictly scoped to gmail.readonly."""

from __future__ import annotations

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = [GMAIL_READONLY_SCOPE]


def _write_token_secure(token_path: Path, token_json: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token_json)


def run_oauth_flow(
    credentials_path: Path,
    token_path: Path,
    *,
    port: int = 8085,
    open_browser: bool | None = None,
) -> Credentials:
    """Run OAuth2 installed-app authorization flow and persist token.json.

    On headless hosts (no DISPLAY/WAYLAND_DISPLAY), automatically sets open_browser=False
    and binds a fixed localhost port so the user can authorize via SSH port forwarding.
    """
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


def get_gmail_credentials(token_path: Path) -> Credentials:
    """Load persisted OAuth2 credentials and refresh headlessly if expired."""
    if not token_path.exists():
        raise FileNotFoundError(
            f"Gmail OAuth token not found at {token_path}. Run `uv run triage auth` first."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _write_token_secure(token_path, creds.to_json())
    if not creds.valid:
        raise RuntimeError(
            f"Gmail OAuth token at {token_path} is invalid and cannot be refreshed. "
            "Re-run `uv run triage auth`."
        )
    return creds
