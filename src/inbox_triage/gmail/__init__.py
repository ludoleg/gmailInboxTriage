"""Google Workspace OAuth2 auth, Gmail client (read/archive/drafts), and Calendar/Tasks client."""

from inbox_triage.gmail.auth import (
    CALENDAR_EVENTS_SCOPE,
    GMAIL_MODIFY_SCOPE,
    GMAIL_READONLY_SCOPE,
    SCOPES,
    TASKS_SCOPE,
    get_gmail_credentials,
    run_oauth_flow,
)
from inbox_triage.gmail.client import GmailClient, clean_html_to_text, parse_gmail_thread
from inbox_triage.gmail.workspace import GoogleWorkspaceClient

__all__ = [
    "CALENDAR_EVENTS_SCOPE",
    "GMAIL_MODIFY_SCOPE",
    "GMAIL_READONLY_SCOPE",
    "SCOPES",
    "TASKS_SCOPE",
    "GmailClient",
    "GoogleWorkspaceClient",
    "clean_html_to_text",
    "get_gmail_credentials",
    "parse_gmail_thread",
    "run_oauth_flow",
]
