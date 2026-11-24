"""Dual-backend state persistence: Firestore (Cloud Run) and SQLite (local dev/tests)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from inbox_triage.config import Settings
from inbox_triage.models import TriageRunReport


class StateStore(Protocol):
    """Contract for watermark tracking and processed-thread idempotency."""

    def get_watermark_epoch(self, default_lookback_hours: int = 24) -> int: ...

    def is_already_processed(self, thread_id: str, last_message_id: str) -> bool: ...

    def record_run(self, report: TriageRunReport) -> None: ...


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


class FirestoreStateStore:
    """Serverless GCP Firestore store for Cloud Run Jobs."""

    RUNS_COLLECTION = "triage_runs"
    THREADS_COLLECTION = "triage_processed_threads"

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


def create_state_store(settings: Settings, *, force_sqlite: bool = False) -> StateStore:
    """Instantiate the configured StateStore backend (Firestore or SQLite)."""
    if force_sqlite or settings.state_backend == "sqlite":
        return SQLiteStateStore(settings.state_db_path)
    return FirestoreStateStore(
        project_id=settings.gcp_project_id,
        database=settings.firestore_database,
    )
