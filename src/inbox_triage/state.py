"""SQLite persistence for watermarks, thread idempotency, and run history."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from inbox_triage.models import TriageRunReport


class StateStore:
    """Lightweight SQLite store for triage run watermarks and deduplication."""

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
                    cloud_escalations INTEGER NOT NULL,
                    used_inkling INTEGER NOT NULL,
                    used_ollama_multimodal INTEGER NOT NULL,
                    used_ollama_fallback INTEGER NOT NULL,
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
        """Return the Unix epoch timestamp of the last successful run (with a 5-min overlap)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT watermark_epoch FROM runs ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        if row is not None:
            # 5-minute overlap prevents boundary misses; processed_threads handles deduplication.
            return max(0, int(row["watermark_epoch"]) - 300)
        fallback = datetime.now(UTC) - timedelta(hours=default_lookback_hours)
        return int(fallback.timestamp())

    def is_already_processed(self, thread_id: str, last_message_id: str) -> bool:
        """Return True if this exact thread state (latest message ID) was already triaged."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_message_id FROM processed_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return row is not None and row["last_message_id"] == last_message_id

    def record_run(self, report: TriageRunReport) -> None:
        """Persist run telemetry, advance watermark, and mark processed threads."""
        watermark_epoch = int(report.started_at.astimezone(UTC).timestamp())
        stats_json = json.dumps([s.model_dump(mode="json") for s in report.call_stats])
        now_iso = report.completed_at.astimezone(UTC).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, started_at, completed_at, watermark_epoch,
                    total_threads, cloud_escalations, used_inkling,
                    used_ollama_multimodal, used_ollama_fallback, stats_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.started_at.astimezone(UTC).isoformat(),
                    now_iso,
                    watermark_epoch,
                    report.total_threads_fetched,
                    report.cloud_escalations_count,
                    int(report.used_inkling),
                    int(report.used_ollama_multimodal),
                    int(report.used_ollama_fallback),
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
