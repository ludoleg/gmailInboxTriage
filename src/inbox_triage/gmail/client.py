"""Read-only Gmail thread fetching with native Gemini multimodal attachment pass-through."""

from __future__ import annotations

import base64
import logging
import re
from datetime import UTC, datetime
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup
from googleapiclient.discovery import build

from inbox_triage.config import Settings
from inbox_triage.gmail.auth import get_gmail_credentials
from inbox_triage.models import AttachmentKind, EmailAttachment, EmailThread

logger = logging.getLogger(__name__)

MAX_ATTACHMENTS_PER_THREAD = 3
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_PDF_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_BODY_CHARS_PER_MESSAGE = 16000

IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
}

_WHITESPACE_RE = re.compile(r"\n{3,}")


def decode_urlsafe_b64(data: str) -> bytes:
    """Decode Gmail URL-safe base64 payload (handling missing padding)."""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def clean_html_to_text(html: str) -> str:
    """Strip scripts, styles, and tracking markup from HTML emails into clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return _WHITESPACE_RE.sub("\n\n", cleaned).strip()


def extract_sender_parts(raw_from: str) -> tuple[str, str, str]:
    """Return (display_or_raw, email_address, domain) from a From header."""
    display, addr = parseaddr(raw_from or "")
    addr_lower = addr.strip().lower()
    domain = addr_lower.split("@", 1)[1] if "@" in addr_lower else ""
    sender_label = display.strip() or addr_lower or raw_from.strip() or "Unknown"
    return sender_label, addr_lower, domain


def _header_map(headers: list[dict[str, str]]) -> dict[str, str]:
    return {h["name"].lower(): h.get("value", "") for h in headers if "name" in h}


def _parse_received_at(headers: dict[str, str], internal_date_ms: str | None) -> datetime:
    if internal_date_ms:
        try:
            return datetime.fromtimestamp(int(internal_date_ms) / 1000.0, tz=UTC)
        except ValueError:
            pass
    if "date" in headers:
        try:
            dt = parsedate_to_datetime(headers["date"])
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC)


def _walk_parts(
    payload: dict[str, Any],
    plain_chunks: list[str],
    html_chunks: list[str],
    attachment_refs: list[dict[str, Any]],
) -> None:
    mime_type = (payload.get("mimeType") or "").lower()
    filename = (payload.get("filename") or "").strip()
    body = payload.get("body") or {}

    if filename and (body.get("attachmentId") or body.get("data")):
        attachment_refs.append(
            {
                "filename": filename,
                "mime_type": mime_type or "application/octet-stream",
                "size": int(body.get("size") or 0),
                "attachment_id": body.get("attachmentId"),
                "data": body.get("data"),
            }
        )
    elif mime_type == "text/plain" and body.get("data"):
        plain_chunks.append(decode_urlsafe_b64(body["data"]).decode("utf-8", errors="replace"))
    elif mime_type == "text/html" and body.get("data"):
        html_chunks.append(decode_urlsafe_b64(body["data"]).decode("utf-8", errors="replace"))

    for child in payload.get("parts") or []:
        _walk_parts(child, plain_chunks, html_chunks, attachment_refs)


def _canonical_mime(mime: str, kind: AttachmentKind) -> str:
    if kind == AttachmentKind.PDF:
        return "application/pdf"
    if mime == "image/jpg":
        return "image/jpeg"
    if mime == "audio/x-wav":
        return "audio/wav"
    if mime == "audio/mp3":
        return "audio/mpeg"
    return mime


def parse_gmail_thread(
    raw_thread: dict[str, Any],
    attachment_fetcher: Any | None = None,
) -> EmailThread:
    """Convert a raw Gmail API thread resource into a normalized EmailThread.

    Image, audio, and PDF attachments are kept as raw bytes for native Gemini multimodal input.
    """
    thread_id = raw_thread["id"]
    messages = raw_thread.get("messages") or []
    if not messages:
        raise ValueError(f"Gmail thread {thread_id} has no messages.")

    latest_msg = messages[-1]
    latest_headers = _header_map((latest_msg.get("payload") or {}).get("headers") or [])
    first_headers = _header_map((messages[0].get("payload") or {}).get("headers") or [])

    subject = latest_headers.get("subject") or first_headers.get("subject") or "(No Subject)"
    sender_label, sender_email, sender_domain = extract_sender_parts(
        latest_headers.get("from") or first_headers.get("from") or ""
    )
    received_at = _parse_received_at(latest_headers, latest_msg.get("internalDate"))

    has_unsub = False
    is_auto = False
    labels: set[str] = set()
    body_sections: list[str] = []
    attachments: list[EmailAttachment] = []

    for msg in messages:
        labels.update(msg.get("labelIds") or [])
        hdrs = _header_map((msg.get("payload") or {}).get("headers") or [])
        if "list-unsubscribe" in hdrs:
            has_unsub = True
        auto_val = hdrs.get("auto-submitted", "").strip().lower()
        if auto_val and auto_val != "no":
            is_auto = True

        plain_chunks: list[str] = []
        html_chunks: list[str] = []
        att_refs: list[dict[str, Any]] = []
        _walk_parts(msg.get("payload") or {}, plain_chunks, html_chunks, att_refs)

        if plain_chunks:
            msg_text = "\n".join(plain_chunks).strip()
        elif html_chunks:
            msg_text = clean_html_to_text("\n".join(html_chunks))
        else:
            msg_text = (msg.get("snippet") or "").strip()

        if len(msg_text) > MAX_BODY_CHARS_PER_MESSAGE:
            msg_text = msg_text[:MAX_BODY_CHARS_PER_MESSAGE] + "\n...[message truncated]"

        msg_sender = hdrs.get("from") or sender_label
        msg_date = hdrs.get("date") or ""
        if len(messages) > 1:
            body_sections.append(f"--- From: {msg_sender} ({msg_date}) ---\n{msg_text}")
        else:
            body_sections.append(msg_text)

        for ref in att_refs:
            if len(attachments) >= MAX_ATTACHMENTS_PER_THREAD:
                break
            mime = ref["mime_type"]
            size = ref["size"]
            fname = ref["filename"]

            if mime in IMAGE_MIME_TYPES:
                kind = AttachmentKind.IMAGE
                cap = MAX_IMAGE_BYTES
            elif mime in AUDIO_MIME_TYPES:
                kind = AttachmentKind.AUDIO
                cap = MAX_AUDIO_BYTES
            elif mime == "application/pdf" or fname.lower().endswith(".pdf"):
                kind = AttachmentKind.PDF
                cap = MAX_PDF_BYTES
            else:
                attachments.append(
                    EmailAttachment(
                        filename=fname,
                        mime_type=mime,
                        kind=AttachmentKind.OTHER,
                        size_bytes=size,
                        note="Non-analyzed attachment type",
                    )
                )
                continue

            if size > cap:
                attachments.append(
                    EmailAttachment(
                        filename=fname,
                        mime_type=_canonical_mime(mime, kind),
                        kind=kind,
                        size_bytes=size,
                        note=f"Attachment exceeds {cap // (1024 * 1024)} MB cap",
                    )
                )
                continue

            raw_bytes: bytes | None = None
            if ref.get("data"):
                raw_bytes = decode_urlsafe_b64(ref["data"])
            elif ref.get("attachment_id") and attachment_fetcher is not None:
                raw_bytes = attachment_fetcher(msg["id"], ref["attachment_id"])

            if not raw_bytes:
                attachments.append(
                    EmailAttachment(
                        filename=fname,
                        mime_type=_canonical_mime(mime, kind),
                        kind=kind,
                        size_bytes=size,
                        note="Attachment payload unavailable",
                    )
                )
                continue

            attachments.append(
                EmailAttachment(
                    filename=fname,
                    mime_type=_canonical_mime(mime, kind),
                    kind=kind,
                    size_bytes=len(raw_bytes),
                    data_bytes=raw_bytes,
                )
            )

    return EmailThread(
        thread_id=thread_id,
        last_message_id=latest_msg["id"],
        subject=subject,
        sender=sender_label,
        sender_email=sender_email,
        sender_domain=sender_domain,
        received_at=received_at,
        message_count=len(messages),
        snippet=latest_msg.get("snippet") or "",
        body_text="\n\n".join(s for s in body_sections if s).strip(),
        has_list_unsubscribe=has_unsub,
        is_auto_submitted=is_auto,
        labels=sorted(labels),
        attachments=attachments,
    )


class GmailClient:
    """Read-only Gmail API client using gmail.readonly scope."""

    def __init__(self, settings: Settings, service: Any | None = None) -> None:
        self.settings = settings
        self._service = service

    @property
    def service(self) -> Any:
        if self._service is None:
            creds = get_gmail_credentials(
                self.settings.gmail_token_path,
                token_json=self.settings.gmail_token_json,
            )
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    def _fetch_attachment_bytes(self, message_id: str, attachment_id: str) -> bytes | None:
        resp = (
            self.service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        data = resp.get("data")
        return decode_urlsafe_b64(data) if data else None

    def fetch_unread_threads(
        self,
        after_epoch: int,
        max_threads: int | None = None,
    ) -> list[EmailThread]:
        """Fetch unread inbox threads received after `after_epoch`."""
        limit = max_threads or self.settings.gmail_max_threads
        query = f"{self.settings.gmail_query_base} after:{after_epoch}".strip()
        logger.info("Querying Gmail with q=%r (max=%d)", query, limit)

        thread_ids: list[str] = []
        page_token: str | None = None
        while len(thread_ids) < limit:
            resp = (
                self.service.users()
                .threads()
                .list(
                    userId="me",
                    q=query,
                    maxResults=min(50, limit - len(thread_ids)),
                    pageToken=page_token,
                )
                .execute()
            )
            for t in resp.get("threads") or []:
                thread_ids.append(t["id"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        threads: list[EmailThread] = []
        for tid in thread_ids[:limit]:
            raw = (
                self.service.users()
                .threads()
                .get(userId="me", id=tid, format="full")
                .execute()
            )
            threads.append(parse_gmail_thread(raw, attachment_fetcher=self._fetch_attachment_bytes))
        return threads
