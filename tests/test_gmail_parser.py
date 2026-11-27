"""Unit tests for Gmail parsing, native multimodal attachment pass-through, and OAuth loading."""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inbox_triage.config import Settings
from inbox_triage.gmail.auth import (
    CALENDAR_EVENTS_SCOPE,
    GMAIL_MODIFY_SCOPE,
    SCOPES,
    TASKS_SCOPE,
    get_gmail_credentials,
)
from inbox_triage.gmail.client import (
    MAX_ATTACHMENTS_PER_THREAD,
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    GmailClient,
    clean_html_to_text,
    decode_urlsafe_b64,
    extract_sender_parts,
    parse_gmail_thread,
)
from inbox_triage.models import AttachmentKind


def _b64(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _make_raw_message(
    *,
    msg_id: str = "m1",
    subject: str = "Hello",
    sender: str = "Alice Example <alice@example.com>",
    date: str = "Wed, 15 Apr 2026 12:00:00 +0000",
    internal_date_ms: str = "1776254400000",
    snippet: str = "Fallback snippet",
    label_ids: list[str] | None = None,
    extra_headers: list[dict[str, str]] | None = None,
    parts: list[dict[str, Any]] | None = None,
    body_mime: str = "text/plain",
    body_text: str | None = "Plain body text.",
) -> dict[str, Any]:
    headers = [
        {"name": "Subject", "value": subject},
        {"name": "From", "value": sender},
        {"name": "Date", "value": date},
    ]
    if extra_headers:
        headers.extend(extra_headers)

    payload: dict[str, Any] = {"headers": headers}
    if parts is not None:
        payload["mimeType"] = "multipart/mixed"
        payload["parts"] = parts
    else:
        payload["mimeType"] = body_mime
        payload["body"] = {"data": _b64(body_text)} if body_text is not None else {}

    return {
        "id": msg_id,
        "internalDate": internal_date_ms,
        "snippet": snippet,
        "labelIds": label_ids if label_ids is not None else ["INBOX", "UNREAD"],
        "payload": payload,
    }


class TestGmailHelpers:
    def test_decode_urlsafe_b64_handles_unpadded_input(self) -> None:
        encoded = base64.urlsafe_b64encode(b"hello world!").decode("ascii").rstrip("=")
        assert decode_urlsafe_b64(encoded) == b"hello world!"

    def test_clean_html_to_text_strips_non_content_tags(self) -> None:
        html = (
            "<html><head><title>Hidden</title><style>.a{color:red}</style></head>"
            "<body><script>alert(1)</script><svg><path d='M0 0'/></svg>"
            "<noscript>Enable JS</noscript>"
            "<h1>Invoice Due</h1><p>Please pay <b>$480</b> today.</p></body></html>"
        )
        cleaned = clean_html_to_text(html)
        assert "Invoice Due" in cleaned
        assert "Please pay" in cleaned
        assert "$480" in cleaned
        assert "alert(1)" not in cleaned
        assert "color:red" not in cleaned
        assert "Hidden" not in cleaned
        assert "Enable JS" not in cleaned

    @pytest.mark.parametrize(
        ("raw_from", "expected"),
        [
            ("Alice Example <ALICE@Example.COM>", ("Alice Example", "alice@example.com", "example.com")),
            ("bob@sub.domain.org", ("bob@sub.domain.org", "bob@sub.domain.org", "sub.domain.org")),
            ("", ("Unknown", "", "")),
            ('"Display Only" <>', ("Display Only", "", "")),
        ],
    )
    def test_extract_sender_parts(self, raw_from: str, expected: tuple[str, str, str]) -> None:
        assert extract_sender_parts(raw_from) == expected


class TestParseGmailThread:
    def test_empty_messages_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="has no messages"):
            parse_gmail_thread({"id": "thr-empty", "messages": []})

    def test_html_fallback_and_headers_detection(self) -> None:
        raw = {
            "id": "thr-promo",
            "messages": [
                _make_raw_message(
                    msg_id="m1",
                    subject="Weekly Sale",
                    sender="Deals <promo@shop.example>",
                    body_mime="text/html",
                    body_text="<html><body><p>Big <b>50% off</b> deal!</p></body></html>",
                    extra_headers=[
                        {"name": "List-Unsubscribe", "value": "<mailto:unsub@shop.example>"},
                        {"name": "Auto-Submitted", "value": "auto-generated"},
                    ],
                )
            ],
        }
        thread = parse_gmail_thread(raw)
        assert thread.thread_id == "thr-promo"
        assert thread.last_message_id == "m1"
        assert thread.has_list_unsubscribe is True
        assert thread.is_auto_submitted is True
        assert "50% off" in thread.body_text
        assert "<b>" not in thread.body_text

    def test_auto_submitted_no_is_false(self) -> None:
        raw = {
            "id": "thr-human",
            "messages": [
                _make_raw_message(
                    extra_headers=[{"name": "Auto-Submitted", "value": "no"}],
                )
            ],
        }
        thread = parse_gmail_thread(raw)
        assert thread.is_auto_submitted is False
        assert thread.has_list_unsubscribe is False

    def test_multi_message_thread_concatenation_and_labels(self) -> None:
        raw = {
            "id": "thr-multi",
            "messages": [
                _make_raw_message(
                    msg_id="m1",
                    subject="Project Kickoff",
                    sender="Alice <alice@example.com>",
                    body_text="First message.",
                    label_ids=["INBOX"],
                ),
                _make_raw_message(
                    msg_id="m2",
                    subject="Re: Project Kickoff",
                    sender="Bob <bob@example.com>",
                    body_text="Second reply.",
                    label_ids=["INBOX", "UNREAD", "IMPORTANT"],
                ),
            ],
        }
        thread = parse_gmail_thread(raw)
        assert thread.message_count == 2
        assert thread.last_message_id == "m2"
        assert thread.sender == "Bob"
        assert thread.sender_email == "bob@example.com"
        assert thread.labels == ["IMPORTANT", "INBOX", "UNREAD"]
        assert "--- From: Alice <alice@example.com>" in thread.body_text
        assert "First message." in thread.body_text
        assert "Second reply." in thread.body_text

    def test_native_multimodal_passthrough_and_canonical_mimes(
        self,
        tiny_png_bytes: bytes,
        tiny_wav_bytes: bytes,
        tiny_pdf_bytes: bytes,
    ) -> None:
        fetched_calls: list[tuple[str, str]] = []

        def _fetcher(msg_id: str, att_id: str) -> bytes | None:
            fetched_calls.append((msg_id, att_id))
            mapping = {
                "att-wav": tiny_wav_bytes,
                "att-pdf": tiny_pdf_bytes,
            }
            return mapping.get(att_id)

        raw = {
            "id": "thr-mm",
            "messages": [
                _make_raw_message(
                    msg_id="m-mm",
                    parts=[
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _b64("See attached photo, voicemail, and PDF.")},
                        },
                        {
                            "filename": "receipt.jpg",
                            "mimeType": "image/jpg",
                            "body": {"size": len(tiny_png_bytes), "data": _b64(tiny_png_bytes)},
                        },
                        {
                            "filename": "voicemail.wav",
                            "mimeType": "audio/x-wav",
                            "body": {"size": len(tiny_wav_bytes), "attachmentId": "att-wav"},
                        },
                        {
                            "filename": "statement.PDF",
                            "mimeType": "application/octet-stream",
                            "body": {"size": len(tiny_pdf_bytes), "attachmentId": "att-pdf"},
                        },
                    ],
                )
            ],
        }

        thread = parse_gmail_thread(raw, attachment_fetcher=_fetcher)

        assert fetched_calls == [("m-mm", "att-wav"), ("m-mm", "att-pdf")]
        assert len(thread.attachments) == 3
        assert len(thread.multimodal_attachments) == 3

        img, aud, pdf = thread.attachments
        assert img.kind == AttachmentKind.IMAGE
        assert img.mime_type == "image/jpeg"
        assert img.data_bytes == tiny_png_bytes

        assert aud.kind == AttachmentKind.AUDIO
        assert aud.mime_type == "audio/wav"
        assert aud.data_bytes == tiny_wav_bytes

        assert pdf.kind == AttachmentKind.PDF
        assert pdf.mime_type == "application/pdf"
        assert pdf.data_bytes == tiny_pdf_bytes

    def test_mp3_canonical_mime_normalization(self) -> None:
        mp3_stub = b"ID3\x03\x00\x00\x00\x00\x00\x00fake-mp3"
        raw = {
            "id": "thr-mp3",
            "messages": [
                _make_raw_message(
                    msg_id="m-mp3",
                    parts=[
                        {"mimeType": "text/plain", "body": {"data": _b64("Audio note")}},
                        {
                            "filename": "note.mp3",
                            "mimeType": "audio/mp3",
                            "body": {"size": len(mp3_stub), "data": _b64(mp3_stub)},
                        },
                    ],
                )
            ],
        }
        thread = parse_gmail_thread(raw)
        assert len(thread.attachments) == 1
        assert thread.attachments[0].mime_type == "audio/mpeg"
        assert thread.attachments[0].kind == AttachmentKind.AUDIO
        assert thread.attachments[0].data_bytes == mp3_stub

    def test_attachment_size_caps_and_other_type(self) -> None:
        raw = {
            "id": "thr-caps",
            "messages": [
                _make_raw_message(
                    msg_id="m-caps",
                    parts=[
                        {"mimeType": "text/plain", "body": {"data": _b64("Huge attachments")}},
                        {
                            "filename": "huge.png",
                            "mimeType": "image/png",
                            "body": {"size": MAX_IMAGE_BYTES + 1, "attachmentId": "a1"},
                        },
                        {
                            "filename": "huge.wav",
                            "mimeType": "audio/wav",
                            "body": {"size": MAX_AUDIO_BYTES + 1, "attachmentId": "a2"},
                        },
                        {
                            "filename": "huge.pdf",
                            "mimeType": "application/pdf",
                            "body": {"size": MAX_PDF_BYTES + 1, "attachmentId": "a3"},
                        },
                    ],
                )
            ],
        }
        fetcher = MagicMock()
        thread = parse_gmail_thread(raw, attachment_fetcher=fetcher)

        fetcher.assert_not_called()
        assert len(thread.attachments) == 3
        assert thread.multimodal_attachments == []
        for att in thread.attachments:
            assert att.data_bytes is None
            assert att.note is not None and "cap" in att.note

    def test_attachment_count_cap_and_non_analyzed_type(self, tiny_png_bytes: bytes) -> None:
        parts: list[dict[str, Any]] = [
            {"mimeType": "text/plain", "body": {"data": _b64("Many files")}},
            {
                "filename": "archive.zip",
                "mimeType": "application/zip",
                "body": {"size": 100, "data": _b64(b"PK\x03\x04")},
            },
        ]
        for idx in range(5):
            parts.append(
                {
                    "filename": f"img-{idx}.png",
                    "mimeType": "image/png",
                    "body": {"size": len(tiny_png_bytes), "data": _b64(tiny_png_bytes)},
                }
            )

        thread = parse_gmail_thread({"id": "thr-many", "messages": [_make_raw_message(parts=parts)]})
        assert len(thread.attachments) == MAX_ATTACHMENTS_PER_THREAD
        assert thread.attachments[0].kind == AttachmentKind.OTHER
        assert thread.attachments[0].data_bytes is None
        assert thread.attachments[0].note == "Non-analyzed attachment type"
        assert len(thread.multimodal_attachments) == MAX_ATTACHMENTS_PER_THREAD - 1

    def test_extracts_rfc_message_id_from_latest_message(self) -> None:
        raw = {
            "id": "thr-rfc",
            "messages": [
                _make_raw_message(
                    msg_id="m1",
                    extra_headers=[{"name": "Message-ID", "value": "<first@example.com>"}],
                ),
                _make_raw_message(
                    msg_id="m2",
                    extra_headers=[{"name": "Message-Id", "value": "<latest-123@example.com>"}],
                ),
            ],
        }
        thread = parse_gmail_thread(raw)
        assert thread.rfc_message_id == "<latest-123@example.com>"

    def test_rfc_message_id_is_none_when_header_absent(self) -> None:
        raw = {"id": "thr-no-rfc", "messages": [_make_raw_message(msg_id="m1")]}
        thread = parse_gmail_thread(raw)
        assert thread.rfc_message_id is None


class TestGmailCredentials:
    def test_phase2_scopes_include_gmail_modify_calendar_and_tasks(self) -> None:
        assert GMAIL_MODIFY_SCOPE in SCOPES
        assert CALENDAR_EVENTS_SCOPE in SCOPES
        assert TASKS_SCOPE in SCOPES

    def test_loads_from_env_json_and_refreshes_in_memory_without_writing_disk(
        self, tmp_path: Path
    ) -> None:
        token_path = tmp_path / "token.json"
        token_json = json.dumps(
            {
                "token": "old-access-token",
                "refresh_token": "refresh-xyz",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "cid",
                "client_secret": "csec",
                "scopes": SCOPES,
            }
        )
        fake_creds = MagicMock()
        fake_creds.valid = False
        fake_creds.expired = True
        fake_creds.refresh_token = "refresh-xyz"

        def _do_refresh(_request: Any) -> None:
            fake_creds.valid = True
            fake_creds.expired = False

        fake_creds.refresh.side_effect = _do_refresh

        with patch(
            "inbox_triage.gmail.auth.Credentials.from_authorized_user_info",
            return_value=fake_creds,
        ) as mock_from_info:
            creds = get_gmail_credentials(token_path, token_json=token_json)

        mock_from_info.assert_called_once_with(json.loads(token_json))
        fake_creds.refresh.assert_called_once()
        assert creds is fake_creds
        # Crucial for Cloud Run Jobs: must NEVER write to disk when loaded from GMAIL_TOKEN_JSON
        assert not token_path.exists()

    def test_loads_from_local_file_and_persists_refreshed_token(self, tmp_path: Path) -> None:
        token_path = tmp_path / "token.json"
        token_path.write_text('{"token": "stale"}', encoding="utf-8")
        token_path.chmod(0o600)

        fake_creds = MagicMock()
        fake_creds.valid = False
        fake_creds.expired = True
        fake_creds.refresh_token = "refresh-local"
        fake_creds.to_json.return_value = '{"token": "fresh-local"}'

        def _do_refresh(_request: Any) -> None:
            fake_creds.valid = True
            fake_creds.expired = False

        fake_creds.refresh.side_effect = _do_refresh

        with patch(
            "inbox_triage.gmail.auth.Credentials.from_authorized_user_file",
            return_value=fake_creds,
        ) as mock_from_file:
            creds = get_gmail_credentials(token_path, token_json=None)

        mock_from_file.assert_called_once_with(str(token_path))
        fake_creds.refresh.assert_called_once()
        assert creds is fake_creds
        assert token_path.read_text(encoding="utf-8") == '{"token": "fresh-local"}'
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    def test_missing_token_file_and_env_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="GMAIL_TOKEN_JSON"):
            get_gmail_credentials(tmp_path / "missing.json", token_json=None)

    def test_invalid_unrefreshable_token_raises_runtime_error(self, tmp_path: Path) -> None:
        fake_creds = MagicMock()
        fake_creds.valid = False
        fake_creds.expired = True
        fake_creds.refresh_token = None

        with patch(
            "inbox_triage.gmail.auth.Credentials.from_authorized_user_info",
            return_value=fake_creds,
        ):
            with pytest.raises(RuntimeError, match="cannot be refreshed"):
                get_gmail_credentials(tmp_path / "token.json", token_json='{"token": "bad"}')


class TestGmailClient:
    def test_fetch_unread_threads_paginates_and_respects_limit(
        self, test_settings: Settings
    ) -> None:
        service = MagicMock()
        threads_api = service.users.return_value.threads.return_value

        threads_api.list.return_value.execute.side_effect = [
            {"threads": [{"id": "t1"}, {"id": "t2"}], "nextPageToken": "page-2"},
            {"threads": [{"id": "t3"}]},
        ]
        threads_api.get.return_value.execute.side_effect = [
            {"id": "t1", "messages": [_make_raw_message(msg_id="m1", subject="One")]},
            {"id": "t2", "messages": [_make_raw_message(msg_id="m2", subject="Two")]},
        ]

        client = GmailClient(test_settings, service=service)
        result = client.fetch_unread_threads(after_epoch=1700000000, max_threads=2)

        assert [t.thread_id for t in result] == ["t1", "t2"]
        # Stopped after hitting max_threads=2 without fetching page-2
        assert threads_api.list.call_count == 1
        assert threads_api.list.call_args.kwargs["q"] == "in:inbox is:unread after:1700000000"
