"""Read-only Gmail API authentication and thread ingestion."""

from inbox_triage.gmail.auth import GMAIL_READONLY_SCOPE, get_gmail_credentials, run_oauth_flow
from inbox_triage.gmail.client import GmailClient, clean_html_to_text, parse_gmail_thread

__all__ = [
    "GMAIL_READONLY_SCOPE",
    "GmailClient",
    "clean_html_to_text",
    "get_gmail_credentials",
    "parse_gmail_thread",
    "run_oauth_flow",
]
