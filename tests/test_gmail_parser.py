"""Unit tests for Gmail MIME/HTML/PDF/attachment parsing and SQLite StateStore."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from inbox_triage.gmail.client import (
    MAX_ATTACHMENTS_PER_THREAD,
    MAX_IMAGE_BYTES,
    clean_html_to_text,
    decode_urlsafe_b64,
    extract_pdf_text,
    extract_sender_parts,
    parse_gmail_thread,
)
from inbox_triage.models import AttachmentKind, TriageCategory, TriagedThread, TriageRunReport
from inbox_triage.state import StateStore
from tests.conftest import make_email_thread, make_valid_pdf_bytes


def _b64(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# --- Helper functions ---


def test_decode_urlsafe_b64_handles_missing_padding() -> None:
    encoded = _b64("hello world!")  # stripped '=' padding
    assert decode_urlsafe_b64(encoded) == b"hello world!"


def test_clean_html_to_text_strips_scripts_styles_and_svg() -> None:
    html = """
    <html>
      <head><title>Ignored Title</title><style>.x { color: red; }</style></head>
      <body>
        <script>alert('xss')</script>
        <svg><path d="M0 0"/></svg>
        <noscript>Enable JS</noscript>
        <h1>Action Required</h1>
        <p>Please approve <b>Budget Q2</b> by Friday.</p>
      </body>
    </html>
    """
    cleaned = clean_html_to_text(html)
    assert "Ignored Title" not in cleaned
    assert "color: red" not in cleaned
    assert "alert" not in cleaned
    assert "Enable JS" not in cleaned
    assert "Action Required" in cleaned
    assert "Budget Q2" in cleaned


@pytest.mark.parametrize(
    ("raw", "expected_label", "expected_email", "expected_domain"),
    [
        ("Alice Smith <Alice.Smith@Example.COM>", "Alice Smith", "alice.smith@example.com", "example.com"),
        ("bob@corp.io", "bob@corp.io", "bob@corp.io", "corp.io"),
        ('"Ops, Team" <ops@sub.acme.org>', "Ops, Team", "ops@sub.acme.org", "sub.acme.org"),
        ("", "Unknown", "", ""),
    ],
)
def test_extract_sender_parts(
    raw: str, expected_label: str, expected_email: str, expected_domain: str
) -> None:
    label, email, domain = extract_sender_parts(raw)
    assert label == expected_label
    assert email == expected_email
    assert domain == expected_domain


def test_extract_pdf_text_reads_real_pdf(tiny_pdf_bytes: bytes) -> None:
    extracted = extract_pdf_text(tiny_pdf_bytes)
    assert "Invoice #42" in extracted
    assert "$1,250.00" in extracted


# --- parse_gmail_thread ---


def test_parse_gmail_thread_multi_message_headers_and_html_fallback() -> None:
    raw_thread = {
        "id": "thread-abc",
        "messages": [
            {
                "id": "msg-1",
                "internalDate": "1744718400000",  # 2025-04-15T12:00:00Z
                "labelIds": ["INBOX", "UNREAD"],
                "snippet": "First snippet",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [
                        {"name": "Subject", "value": "Quarterly Update"},
                        {"name": "From", "value": "Newsletter Bot <news@digest.example>"},
                        {"name": "List-Unsubscribe", "value": "<mailto:unsub@digest.example>"},
                        {"name": "Auto-Submitted", "value": "auto-generated"},
                        {"name": "Date", "value": "Tue, 15 Apr 2025 12:00:00 +0000"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/html",
                            "body": {"data": _b64("<p>Hello <b>Reader</b></p><script>bad()</script>")},
                        }
                    ],
                },
            },
            {
                "id": "msg-2",
                "internalDate": "1744722000000",  # 1 hour later
                "labelIds": ["INBOX", "IMPORTANT"],
                "snippet": "Follow-up snippet",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "Jane Doe <jane@digest.example>"},
                        {"name": "Auto-Submitted", "value": "no"},
                        {"name": "Date", "value": "Tue, 15 Apr 2025 13:00:00 +0000"},
                    ],
                    "body": {"data": _b64("Follow-up plain text body.")},
                },
            },
        ],
    }

    parsed = parse_gmail_thread(raw_thread)

    assert parsed.thread_id == "thread-abc"
    assert parsed.last_message_id == "msg-2"
    assert parsed.subject == "Quarterly Update"
    assert parsed.sender == "Jane Doe"
    assert parsed.sender_email == "jane@digest.example"
    assert parsed.sender_domain == "digest.example"
    assert parsed.message_count == 2
    assert parsed.has_list_unsubscribe is True
    assert parsed.is_auto_submitted is True
    assert parsed.labels == ["IMPORTANT", "INBOX", "UNREAD"]
    assert "Hello" in parsed.body_text
    assert "bad()" not in parsed.body_text
    assert "Follow-up plain text body." in parsed.body_text


def test_parse_gmail_thread_pdf_image_audio_and_caps(
    tiny_png_bytes: bytes,
    tiny_wav_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = make_valid_pdf_bytes("Contract renewal due May 1")

    def fake_fetcher(msg_id: str, att_id: str) -> bytes | None:
        return {
            "att-pdf": pdf_bytes,
            "att-img": tiny_png_bytes,
            "att-wav": tiny_wav_bytes,
        }.get(att_id)

    # Simulate no ffmpeg so WAV passthrough path is exercised cleanly
    monkeypatch.setattr("inbox_triage.gmail.client.shutil.which", lambda _: None)

    raw_thread = {
        "id": "thread-media",
        "messages": [
            {
                "id": "msg-media-1",
                "internalDate": "1744718400000",
                "labelIds": ["INBOX", "UNREAD"],
                "snippet": "See attachments",
                "payload": {
                    "mimeType": "multipart/mixed",
                    "headers": [
                        {"name": "Subject", "value": "Invoice + Voicemail"},
                        {"name": "From", "value": "Client <client@acme.example>"},
                    ],
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64("Attached files.")}},
                        {
                            "filename": "contract.pdf",
                            "mimeType": "application/pdf",
                            "body": {"attachmentId": "att-pdf", "size": len(pdf_bytes)},
                        },
                        {
                            "filename": "receipt.png",
                            "mimeType": "image/png",
                            "body": {"attachmentId": "att-img", "size": len(tiny_png_bytes)},
                        },
                        {
                            "filename": "voicemail.wav",
                            "mimeType": "audio/wav",
                            "body": {"attachmentId": "att-wav", "size": len(tiny_wav_bytes)},
                        },
                        # 4th attachment exceeds MAX_ATTACHMENTS_PER_THREAD (3)
                        {
                            "filename": "extra.png",
                            "mimeType": "image/png",
                            "body": {"attachmentId": "att-img", "size": len(tiny_png_bytes)},
                        },
                    ],
                },
            }
        ],
    }

    parsed = parse_gmail_thread(raw_thread, attachment_fetcher=fake_fetcher)

    assert len(parsed.attachments) == MAX_ATTACHMENTS_PER_THREAD
    assert parsed.has_multimodal_attachments is True

    by_name = {a.filename: a for a in parsed.attachments}
    assert by_name["contract.pdf"].kind == AttachmentKind.PDF
    assert "Contract renewal due May 1" in (by_name["contract.pdf"].extracted_text or "")
    assert by_name["receipt.png"].kind == AttachmentKind.IMAGE
    assert by_name["receipt.png"].data_bytes == tiny_png_bytes
    assert by_name["voicemail.wav"].kind == AttachmentKind.AUDIO
    assert by_name["voicemail.wav"].data_bytes == tiny_wav_bytes
    assert "extra.png" not in by_name


def test_parse_gmail_thread_skips_oversized_image_and_marks_other() -> None:
    raw_thread = {
        "id": "thread-caps",
        "messages": [
            {
                "id": "msg-caps-1",
                "internalDate": "1744718400000",
                "payload": {
                    "mimeType": "multipart/mixed",
                    "headers": [{"name": "Subject", "value": "Big files"}],
                    "parts": [
                        {
                            "filename": "huge.png",
                            "mimeType": "image/png",
                            "body": {"attachmentId": "att-huge", "size": MAX_IMAGE_BYTES + 1024},
                        },
                        {
                            "filename": "archive.zip",
                            "mimeType": "application/zip",
                            "body": {"attachmentId": "att-zip", "size": 5000},
                        },
                    ],
                },
            }
        ],
    }

    parsed = parse_gmail_thread(raw_thread)
    assert parsed.has_multimodal_attachments is False
    assert len(parsed.attachments) == 2
    assert parsed.attachments[0].data_bytes is None
    assert "exceeds" in (parsed.attachments[0].note or "")
    assert parsed.attachments[1].kind == AttachmentKind.OTHER
    assert parsed.attachments[1].data_bytes is None


def test_parse_gmail_thread_empty_messages_raises() -> None:
    with pytest.raises(ValueError, match="has no messages"):
        parse_gmail_thread({"id": "empty-1", "messages": []})


# --- StateStore SQLite watermark & idempotency ---


def test_state_store_watermark_and_thread_idempotency(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "state.db"
    store = StateStore(db_path)
    assert db_path.exists()

    # Fresh store falls back to ~default_lookback_hours ago
    now_epoch = int(datetime.now(UTC).timestamp())
    initial_wm = store.get_watermark_epoch(default_lookback_hours=12)
    assert abs((now_epoch - 12 * 3600) - initial_wm) < 10
    assert store.is_already_processed("t-1", "m-1") is False

    # Record a run with thread t-1 at message m-1
    started = datetime(2026, 4, 15, 12, 30, tzinfo=UTC)
    completed = started + timedelta(seconds=14)
    thread_v1 = make_email_thread(thread_id="t-1", last_message_id="m-1")
    report_1 = TriageRunReport(
        run_id="run-001",
        started_at=started,
        completed_at=completed,
        timezone="America/Los_Angeles",
        total_threads_fetched=1,
        triaged_threads=[
            TriagedThread(
                thread=thread_v1,
                category=TriageCategory.ACTION_REQUIRED,
                confidence=0.9,
                summary="Reply needed",
            )
        ],
        used_inkling=True,
    )
    store.record_run(report_1)

    # Watermark returns started_at - 300s (5-minute overlap)
    assert store.get_watermark_epoch() == int(started.timestamp()) - 300

    # Same (thread_id, last_message_id) -> already processed
    assert store.is_already_processed("t-1", "m-1") is True
    # New reply in same thread (m-2) -> NOT already processed (re-triaged!)
    assert store.is_already_processed("t-1", "m-2") is False
    # Different thread -> NOT already processed
    assert store.is_already_processed("t-2", "m-1") is False

    # Record second run with updated message m-2
    started_2 = started + timedelta(hours=24)
    thread_v2 = make_email_thread(thread_id="t-1", last_message_id="m-2")
    report_2 = TriageRunReport(
        run_id="run-002",
        started_at=started_2,
        completed_at=started_2 + timedelta(seconds=10),
        timezone="America/Los_Angeles",
        total_threads_fetched=1,
        triaged_threads=[
            TriagedThread(
                thread=thread_v2,
                category=TriageCategory.URGENT,
                confidence=0.95,
                summary="Urgent follow-up",
            )
        ],
        used_inkling=True,
    )
    store.record_run(report_2)

    assert store.get_watermark_epoch() == int(started_2.timestamp()) - 300
    assert store.is_already_processed("t-1", "m-2") is True
    assert store.is_already_processed("t-1", "m-1") is False
