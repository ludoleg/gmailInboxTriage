"""Shared fixtures and synthetic data builders for inbox_triage GCP serverless tests."""

from __future__ import annotations

import io
import struct
import wave
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inbox_triage.config import Settings
from inbox_triage.models import AttachmentKind, EmailAttachment, EmailThread


def make_valid_png_bytes(width: int = 32, height: int = 32) -> bytes:
    """Create a minimal valid RGB PNG image in pure Python (no Pillow required)."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            if 4 <= y <= 10 and 4 <= x <= width - 5:
                raw_rows.extend((30, 30, 30))
            else:
                raw_rows.extend((250, 250, 250))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def make_valid_wav_bytes(duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid 16 kHz mono 16-bit PCM RIFF WAV buffer."""
    buf = io.BytesIO()
    n_frames = int(duration_s * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def make_valid_pdf_bytes(text: str = "Invoice #42 Total: $1,250.00 due Friday") -> bytes:
    """Build a minimal valid single-page PDF with Helvetica text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("latin-1")

    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"endstream\nendobj\n"
        ),
    ]

    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj in objects:
        offsets.append(len(buf))
        buf.extend(obj)

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    buf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(buf)


def make_email_thread(
    *,
    thread_id: str = "t-001",
    last_message_id: str = "m-001",
    rfc_message_id: str | None = None,
    subject: str = "Test Subject",
    sender: str = "Alice Example",
    sender_email: str = "alice@example.com",
    sender_domain: str | None = None,
    received_at: datetime | None = None,
    message_count: int = 1,
    snippet: str = "Short snippet",
    body_text: str = "Hello, please review this item.",
    has_list_unsubscribe: bool = False,
    is_auto_submitted: bool = False,
    labels: list[str] | None = None,
    attachments: list[EmailAttachment] | None = None,
) -> EmailThread:
    domain = (
        sender_domain
        if sender_domain is not None
        else (sender_email.split("@", 1)[1] if "@" in sender_email else "")
    )
    return EmailThread(
        thread_id=thread_id,
        last_message_id=last_message_id,
        rfc_message_id=rfc_message_id,
        subject=subject,
        sender=sender,
        sender_email=sender_email,
        sender_domain=domain,
        received_at=received_at or datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
        message_count=message_count,
        snippet=snippet,
        body_text=body_text,
        has_list_unsubscribe=has_list_unsubscribe,
        is_auto_submitted=is_auto_submitted,
        labels=labels if labels is not None else ["INBOX", "UNREAD"],
        attachments=attachments if attachments is not None else [],
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Isolated Settings instance backed by tmp_path SQLite (ignores host .env)."""
    return Settings(
        _env_file=None,
        TIMEZONE="America/Los_Angeles",
        GCP_PROJECT_ID="test-gcp-project",
        USE_VERTEXAI=False,
        VERTEX_LOCATION="global",
        GEMINI_API_KEY="test-gemini-key",
        MODEL_STAGE1_FAST="gemini-3.5-flash-lite",
        MODEL_STAGE2_WORKHORSE="gemini-4-carbon",
        MODEL_STAGE3_REASONING="gemini-4-neon",
        MAX_CONCURRENCY=4,
        DEEP_REVIEW_CONFIDENCE_THRESHOLD=0.70,
        MAX_DEEP_REVIEWS_PER_RUN=5,
        STATE_BACKEND="sqlite",
        FIRESTORE_DATABASE="(default)",
        STATE_DB_PATH=tmp_path / "state.db",
        GMAIL_TOKEN_JSON=None,
        GMAIL_CREDENTIALS_PATH=tmp_path / "credentials.json",
        GMAIL_TOKEN_PATH=tmp_path / "token.json",
        SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T000/B000/XXXX",
        SLACK_MENTION_ON_URGENT=None,
    )


@pytest.fixture
def tiny_png_bytes() -> bytes:
    return make_valid_png_bytes()


@pytest.fixture
def tiny_wav_bytes() -> bytes:
    return make_valid_wav_bytes()


@pytest.fixture
def tiny_pdf_bytes() -> bytes:
    return make_valid_pdf_bytes()


@pytest.fixture
def synthetic_inbox_threads(
    tiny_png_bytes: bytes,
    tiny_wav_bytes: bytes,
    tiny_pdf_bytes: bytes,
) -> list[EmailThread]:
    """Realistic synthetic emails covering urgent+multimodal, CI alert, promo, newsletter, and action."""
    base_time = datetime(2026, 4, 15, 11, 30, tzinfo=UTC)
    return [
        make_email_thread(
            thread_id="thr-urgent-receipt",
            last_message_id="msg-101",
            subject="URGENT: Production hosting invoice overdue — service suspension at 5 PM PT today",
            sender="Billing Escalations",
            sender_email="billing@cloudhost.example",
            received_at=base_time,
            snippet="Your invoice #INV-9921 ($480.00) is past due. Please confirm payment today.",
            body_text=(
                "Hi,\n\n"
                "Your production hosting invoice #INV-9921 for $480.00 is past due. "
                "Please review the attached receipt image, voicemail, and PDF statement "
                "before 5:00 PM Pacific today to avoid cluster suspension.\n\n"
                "Thanks,\nCloudHost Billing"
            ),
            attachments=[
                EmailAttachment(
                    filename="receipt.png",
                    mime_type="image/png",
                    kind=AttachmentKind.IMAGE,
                    size_bytes=len(tiny_png_bytes),
                    data_bytes=tiny_png_bytes,
                ),
                EmailAttachment(
                    filename="voicemail.wav",
                    mime_type="audio/wav",
                    kind=AttachmentKind.AUDIO,
                    size_bytes=len(tiny_wav_bytes),
                    data_bytes=tiny_wav_bytes,
                ),
                EmailAttachment(
                    filename="statement.pdf",
                    mime_type="application/pdf",
                    kind=AttachmentKind.PDF,
                    size_bytes=len(tiny_pdf_bytes),
                    data_bytes=tiny_pdf_bytes,
                ),
            ],
        ),
        make_email_thread(
            thread_id="thr-ci-failure",
            last_message_id="msg-102",
            subject="[acme/payments-api] Run failed: CI / test-suite (main)",
            sender="GitHub Actions",
            sender_email="notifications@github.com",
            received_at=base_time,
            is_auto_submitted=True,
            snippet="Workflow CI failed on branch main in acme/payments-api.",
            body_text=(
                "Repository: acme/payments-api\n"
                "Workflow: CI / test-suite\n"
                "Branch: main\n"
                "Commit: a1b2c3d Fix stripe webhook retry\n"
                "Error: AssertionError in tests/test_webhooks.py::test_idempotency\n"
                "View run: https://github.com/acme/payments-api/actions/runs/987654321\n"
            ),
        ),
        make_email_thread(
            thread_id="thr-promo-sale",
            last_message_id="msg-103",
            subject="Flash Sale! 40% OFF all ergonomic chairs this weekend only!",
            sender="ErgoDeals Marketing",
            sender_email="promo@ergodeals.example",
            received_at=base_time,
            has_list_unsubscribe=True,
            snippet="Save 40% on standing desks and chairs. Use code SAVE40.",
            body_text=(
                "Limited time offer! Get 40% off sitewide with code SAVE40. "
                "Click here to shop now or unsubscribe at the bottom."
            ),
        ),
        make_email_thread(
            thread_id="thr-newsletter",
            last_message_id="msg-104",
            subject="Weekly Cloud Digest #88",
            sender="Cloud Weekly",
            sender_email="digest@cloudweekly.example",
            received_at=base_time,
            has_list_unsubscribe=True,
            snippet="Top serverless articles this week.",
            body_text="Here are the top 5 serverless articles this week.",
        ),
        make_email_thread(
            thread_id="thr-action-meeting",
            last_message_id="msg-105",
            subject="Confirm Q2 roadmap review slot tomorrow?",
            sender="Maya Lin",
            sender_email="maya.lin@partner.example",
            received_at=base_time,
            snippet="Can you confirm whether 10 AM or 2 PM PT tomorrow works for the Q2 review?",
            body_text=(
                "Hi,\n\n"
                "Could you let me know by end of day whether 10:00 AM PT or 2:00 PM PT "
                "tomorrow works better for our Q2 roadmap review? I'll send the calendar invite "
                "once you confirm.\n\nBest,\nMaya"
            ),
        ),
    ]
