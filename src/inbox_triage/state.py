"""Dual-backend state persistence: Firestore (Cloud Run) and SQLite (local dev/tests)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from inbox_triage.config import Settings
from inbox_triage.models import MutedSender, TriageRunReport


class StateStore(Protocol):
    """Contract for watermark tracking, processed-thread idempotency, and muted senders."""

    def get_watermark_epoch(self, default_lookback_hours: int = 24) -> int: ...

    def is_already_processed(self, thread_id: str, last_message_id: str) -> bool: ...

    def record_run(self, report: TriageRunReport) -> None: ...

    def mute_sender(self, sender_email: str, reason: str | None = None) -> None: ...

    def unmute_sender(self, sender_email: str) -> bool: ...

    def get_muted_senders(self) -> set[str]: ...

    def list_muted_senders(self) -> list[MutedSender]: ...


def _normalize_email(email: str) -> str:
    return email.strip().lower()


class SQLiteStateStore:
    """Lightweight SQLite store for local development and unit tests."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    watermark_epoch INTEGER NOT NULL,
                    total_threads INTEGER NOT NULL,
                    deep_reviews INTEGER NOT NULL,
                    stats_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_threads (
                    thread_id TEXT PRIMARY KEY,
                    last_message_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    run_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS muted_senders (
                    sender_email TEXT PRIMARY KEY,
                    muted_at TEXT NOT NULL,
                    reason TEXT
                );
                """
            )

    def get_watermark_epoch(self, default_lookback_hours: int = 24) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT watermark_epoch FROM runs ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        if row is not None:
            return max(0, int(row["watermark_epoch"]) - 300)
        fallback = datetime.now(UTC) - timedelta(hours=default_lookback_hours)
        return int(fallback.timestamp())

    def is_already_processed(self, thread_id: str, last_message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_message_id FROM processed_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return row is not None and row["last_message_id"] == last_message_id

    def record_run(self, report: TriageRunReport) -> None:
        watermark_epoch = int(report.started_at.astimezone(UTC).timestamp())
        stats_json = json.dumps([s.model_dump(mode="json") for s in report.call_stats])
        now_iso = report.completed_at.astimezone(UTC).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, started_at, completed_at, watermark_epoch,
                    total_threads, deep_reviews, stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.started_at.astimezone(UTC).isoformat(),
                    now_iso,
                    watermark_epoch,
                    report.total_threads_fetched,
                    report.deep_reviews_count,
                    stats_json,
                ),
            )
            for item in report.triaged_threads:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO processed_threads (
                        thread_id, last_message_id, category, processed_at, run_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        item.thread.thread_id,
                        item.thread.last_message_id,
                        item.category.value,
                        now_iso,
                        report.run_id,
                    ),
                )

    def mute_sender(self, sender_email: str, reason: str | None = None) -> None:
        norm = _normalize_email(sender_email)
        if not norm:
            raise ValueError("sender_email must not be empty.")
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO muted_senders (sender_email, muted_at, reason)
                VALUES (?, ?, ?)
                """,
                (norm, now_iso, reason),
            )

    def unmute_sender(self, sender_email: str) -> bool:
        norm = _normalize_email(sender_email)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM muted_senders WHERE sender_email = ?", (norm,))
            return cur.rowcount > 0

    def get_muted_senders(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT sender_email FROM muted_senders").fetchall()
        return {str(r["sender_email"]).lower() for r in rows}

    def list_muted_senders(self) -> list[MutedSender]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sender_email, muted_at, reason FROM muted_senders ORDER BY muted_at DESC"
            ).fetchall()
        return [
            MutedSender(
                sender_email=r["sender_email"],
                muted_at=datetime.fromisoformat(r["muted_at"]),
                reason=r["reason"],
            )
            for r in rows
        ]


class FirestoreStateStore:
    """Serverless GCP Firestore store for Cloud Run Jobs & Services."""

    RUNS_COLLECTION = "triage_runs"
    THREADS_COLLECTION = "triage_processed_threads"
    MUTED_COLLECTION = "triage_muted_senders"

    def __init__(
        self,
        project_id: str,
        database: str = "(default)",
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._db = client
        else:
            from google.cloud import firestore

            self._db = firestore.Client(project=project_id, database=database)

    def get_watermark_epoch(self, default_lookback_hours: int = 24) -> int:
        from google.cloud.firestore_v1 import Query

        docs = list(
            self._db.collection(self.RUNS_COLLECTION)
            .order_by("completed_at", direction=Query.DESCENDING)
            .limit(1)
            .stream()
        )
        if docs:
            data = docs[0].to_dict() or {}
            if "watermark_epoch" in data:
                return max(0, int(data["watermark_epoch"]) - 300)
        fallback = datetime.now(UTC) - timedelta(hours=default_lookback_hours)
        return int(fallback.timestamp())

    def is_already_processed(self, thread_id: str, last_message_id: str) -> bool:
        doc = self._db.collection(self.THREADS_COLLECTION).document(thread_id).get()
        if not doc.exists:
            return False
        data = doc.to_dict() or {}
        return data.get("last_message_id") == last_message_id

    def record_run(self, report: TriageRunReport) -> None:
        watermark_epoch = int(report.started_at.astimezone(UTC).timestamp())
        now_iso = report.completed_at.astimezone(UTC).isoformat()

        batch = self._db.batch()
        run_ref = self._db.collection(self.RUNS_COLLECTION).document(report.run_id)
        batch.set(
            run_ref,
            {
                "run_id": report.run_id,
                "started_at": report.started_at.astimezone(UTC).isoformat(),
                "completed_at": now_iso,
                "watermark_epoch": watermark_epoch,
                "total_threads": report.total_threads_fetched,
                "muted_skipped": report.muted_skipped_count,
                "deep_reviews": report.deep_reviews_count,
                "total_prompt_tokens": report.total_prompt_tokens,
                "total_completion_tokens": report.total_completion_tokens,
                "stats": [s.model_dump(mode="json") for s in report.call_stats],
            },
        )
        for item in report.triaged_threads:
            t_ref = self._db.collection(self.THREADS_COLLECTION).document(item.thread.thread_id)
            batch.set(
                t_ref,
                {
                    "thread_id": item.thread.thread_id,
                    "last_message_id": item.thread.last_message_id,
                    "category": item.category.value,
                    "processed_at": now_iso,
                    "run_id": report.run_id,
                },
            )
        batch.commit()

    def mute_sender(self, sender_email: str, reason: str | None = None) -> None:
        norm = _normalize_email(sender_email)
        if not norm:
            raise ValueError("sender_email must not be empty.")
        self._db.collection(self.MUTED_COLLECTION).document(norm).set(
            {
                "sender_email": norm,
                "muted_at": datetime.now(UTC).isoformat(),
                "reason": reason,
            }
        )

    def unmute_sender(self, sender_email: str) -> bool:
        norm = _normalize_email(sender_email)
        ref = self._db.collection(self.MUTED_COLLECTION).document(norm)
        doc = ref.get()
        if not doc.exists:
            return False
        ref.delete()
        return True

    def get_muted_senders(self) -> set[str]:
        docs = self._db.collection(self.MUTED_COLLECTION).stream()
        out: set[str] = set()
        for doc in docs:
            data = doc.to_dict() or {}
            email = data.get("sender_email") or getattr(doc, "id", "")
            if email:
                out.add(str(email).lower())
        return out

    def list_muted_senders(self) -> list[MutedSender]:
        docs = self._db.collection(self.MUTED_COLLECTION).stream()
        items: list[MutedSender] = []
        for doc in docs:
            data = doc.to_dict() or {}
            email = data.get("sender_email") or getattr(doc, "id", "")
            muted_at_raw = data.get("muted_at")
            muted_at = (
                datetime.fromisoformat(muted_at_raw)
                if isinstance(muted_at_raw, str)
                else datetime.now(UTC)
            )
            if email:
                items.append(
                    MutedSender(
                        sender_email=str(email).lower(),
                        muted_at=muted_at,
                        reason=data.get("reason"),
                    )
                )
        items.sort(key=lambda m: m.muted_at, reverse=True)
        return items


def create_state_store(settings: Settings, *, force_sqlite: bool = False) -> StateStore:
    """Instantiate the configured StateStore backend (Firestore or SQLite)."""
    if force_sqlite or settings.state_backend == "sqlite":
        return SQLiteStateStore(settings.state_db_path)
    return FirestoreStateStore(
        project_id=settings.gcp_project_id,
        database=settings.firestore_database,
    )
