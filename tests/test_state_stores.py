"""Unit tests for SQLiteStateStore, FirestoreStateStore, and create_state_store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.cloud.firestore_v1 import Query

from inbox_triage.config import Settings
from inbox_triage.models import (
    ModelCallStats,
    TriageCategory,
    TriagedThread,
    TriageRunReport,
)
from inbox_triage.state import FirestoreStateStore, SQLiteStateStore, create_state_store
from tests.conftest import make_email_thread


def _make_report(
    *,
    run_id: str = "run-001",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    threads: list[TriagedThread] | None = None,
    deep_reviews_count: int = 1,
) -> TriageRunReport:
    start = started_at or datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
    end = completed_at or (start + timedelta(seconds=18))
    items = threads if threads is not None else [
        TriagedThread(
            thread=make_email_thread(thread_id="thr-1", last_message_id="msg-1a"),
            category=TriageCategory.URGENT,
            confidence=0.95,
            summary="Urgent item",
        ),
        TriagedThread(
            thread=make_email_thread(thread_id="thr-2", last_message_id="msg-2a"),
            category=TriageCategory.FYI,
            confidence=0.85,
            summary="FYI item",
        ),
    ]
    return TriageRunReport(
        run_id=run_id,
        started_at=start,
        completed_at=end,
        timezone="America/Los_Angeles",
        total_threads_fetched=len(items),
        triaged_threads=items,
        call_stats=[
            ModelCallStats(
                backend="vertexai",
                model="gemini-3.5-flash-lite",
                stage="stage1_fast_triage",
                prompt_tokens=300,
                completion_tokens=75,
                latency_ms=140.2,
            ),
            ModelCallStats(
                backend="vertexai",
                model="gemini-4-neon",
                stage="stage3_executive_synthesis",
                prompt_tokens=800,
                completion_tokens=220,
                latency_ms=650.5,
            ),
        ],
        deep_reviews_count=deep_reviews_count,
    )


class TestSQLiteStateStore:
    def test_default_lookback_when_empty(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "nested" / "state.db")
        now_epoch = int(datetime.now(UTC).timestamp())
        wm = store.get_watermark_epoch(default_lookback_hours=12)

        expected = now_epoch - 12 * 3600
        assert abs(wm - expected) <= 5

    def test_watermark_five_minute_overlap_and_latest_run(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")

        t1 = datetime(2026, 4, 14, 12, 30, 0, tzinfo=UTC)
        t2 = datetime(2026, 4, 15, 12, 30, 0, tzinfo=UTC)
        store.record_run(_make_report(run_id="run-1", started_at=t1, completed_at=t1 + timedelta(seconds=10)))
        store.record_run(_make_report(run_id="run-2", started_at=t2, completed_at=t2 + timedelta(seconds=10)))

        wm = store.get_watermark_epoch()
        assert wm == int(t2.timestamp()) - 300

    def test_idempotency_and_updated_thread_retriage(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        assert store.is_already_processed("thr-1", "msg-1a") is False

        store.record_run(_make_report(run_id="run-1"))
        assert store.is_already_processed("thr-1", "msg-1a") is True
        assert store.is_already_processed("thr-2", "msg-2a") is True
        # New reply arrives in thr-1 -> different last_message_id -> must re-triage
        assert store.is_already_processed("thr-1", "msg-1b") is False
        assert store.is_already_processed("thr-unknown", "msg-x") is False

        # After re-triaging thr-1 with msg-1b, updated message id is recorded
        updated_report = _make_report(
            run_id="run-2",
            threads=[
                TriagedThread(
                    thread=make_email_thread(thread_id="thr-1", last_message_id="msg-1b"),
                    category=TriageCategory.ACTION_REQUIRED,
                    confidence=0.92,
                    summary="Updated reply",
                )
            ],
        )
        store.record_run(updated_report)
        assert store.is_already_processed("thr-1", "msg-1b") is True
        assert store.is_already_processed("thr-1", "msg-1a") is False

    def test_watermark_clamped_at_zero(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        epoch_start = datetime(1970, 1, 1, 0, 1, 0, tzinfo=UTC)
        store.record_run(
            _make_report(
                run_id="run-epoch",
                started_at=epoch_start,
                completed_at=epoch_start + timedelta(seconds=5),
            )
        )
        assert store.get_watermark_epoch() == 0


class TestFirestoreStateStore:
    def test_get_watermark_epoch_queries_descending_with_5min_overlap(self) -> None:
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {"watermark_epoch": 1776256200}

        runs_col = mock_db.collection.return_value
        ordered = runs_col.order_by.return_value
        limited = ordered.limit.return_value
        limited.stream.return_value = [mock_doc]

        store = FirestoreStateStore(project_id="proj-1", database="(default)", client=mock_db)
        wm = store.get_watermark_epoch(default_lookback_hours=24)

        mock_db.collection.assert_called_once_with("triage_runs")
        runs_col.order_by.assert_called_once_with("completed_at", direction=Query.DESCENDING)
        ordered.limit.assert_called_once_with(1)
        assert wm == 1776256200 - 300

    def test_get_watermark_epoch_fallback_when_no_runs(self) -> None:
        mock_db = MagicMock()
        mock_db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = []

        store = FirestoreStateStore(project_id="proj-1", client=mock_db)
        now_epoch = int(datetime.now(UTC).timestamp())
        wm = store.get_watermark_epoch(default_lookback_hours=6)

        assert abs(wm - (now_epoch - 6 * 3600)) <= 5

    def test_is_already_processed_checks_existence_and_last_message_id(self) -> None:
        mock_db = MagicMock()
        threads_col = mock_db.collection.return_value
        doc_ref = threads_col.document.return_value

        # Case 1: document does not exist
        doc_ref.get.return_value = SimpleNamespace(exists=False, to_dict=lambda: None)
        store = FirestoreStateStore(project_id="proj-1", client=mock_db)
        assert store.is_already_processed("thr-1", "msg-1a") is False
        mock_db.collection.assert_called_with("triage_processed_threads")
        threads_col.document.assert_called_with("thr-1")

        # Case 2: exists with matching last_message_id
        doc_ref.get.return_value = SimpleNamespace(
            exists=True,
            to_dict=lambda: {"thread_id": "thr-1", "last_message_id": "msg-1a"},
        )
        assert store.is_already_processed("thr-1", "msg-1a") is True

        # Case 3: exists with older last_message_id (updated thread -> re-triage)
        assert store.is_already_processed("thr-1", "msg-1b") is False

    def test_record_run_performs_atomic_batch_writes(self) -> None:
        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch

        col_refs: dict[str, MagicMock] = {}

        def _collection(name: str) -> MagicMock:
            if name not in col_refs:
                col = MagicMock(name=f"col:{name}")
                col.document.side_effect = lambda doc_id, n=name: f"{n}/{doc_id}"
                col_refs[name] = col
            return col_refs[name]

        mock_db.collection.side_effect = _collection

        report = _make_report(run_id="run-fs-42", deep_reviews_count=2)
        store = FirestoreStateStore(project_id="proj-1", client=mock_db)
        store.record_run(report)

        mock_db.batch.assert_called_once()
        # 1 run document + 2 processed thread documents = 3 batch.set calls
        assert mock_batch.set.call_count == 3

        run_call_ref, run_payload = mock_batch.set.call_args_list[0].args
        assert run_call_ref == "triage_runs/run-fs-42"
        assert run_payload["run_id"] == "run-fs-42"
        assert run_payload["watermark_epoch"] == int(report.started_at.astimezone(UTC).timestamp())
        assert run_payload["total_threads"] == 2
        assert run_payload["deep_reviews"] == 2
        assert run_payload["total_prompt_tokens"] == 1100
        assert run_payload["total_completion_tokens"] == 295
        assert len(run_payload["stats"]) == 2

        t1_ref, t1_payload = mock_batch.set.call_args_list[1].args
        assert t1_ref == "triage_processed_threads/thr-1"
        assert t1_payload["thread_id"] == "thr-1"
        assert t1_payload["last_message_id"] == "msg-1a"
        assert t1_payload["category"] == "URGENT"
        assert t1_payload["run_id"] == "run-fs-42"

        t2_ref, t2_payload = mock_batch.set.call_args_list[2].args
        assert t2_ref == "triage_processed_threads/thr-2"
        assert t2_payload["last_message_id"] == "msg-2a"
        assert t2_payload["category"] == "FYI"

        mock_batch.commit.assert_called_once()


class TestCreateStateStoreFactory:
    def test_returns_sqlite_when_configured(self, test_settings: Settings) -> None:
        store = create_state_store(test_settings)
        assert isinstance(store, SQLiteStateStore)
        assert store.db_path == test_settings.state_db_path

    def test_force_sqlite_overrides_firestore_setting(self, test_settings: Settings) -> None:
        fs_settings = test_settings.model_copy(update={"state_backend": "firestore"})
        store = create_state_store(fs_settings, force_sqlite=True)
        assert isinstance(store, SQLiteStateStore)

    def test_returns_firestore_when_configured(self, test_settings: Settings) -> None:
        fs_settings = test_settings.model_copy(
            update={
                "state_backend": "firestore",
                "gcp_project_id": "gmail-inbox-triage-515705",
                "firestore_database": "triage-db",
            }
        )
        fake_client = MagicMock()
        with patch("google.cloud.firestore.Client", return_value=fake_client) as mock_fs:
            store = create_state_store(fs_settings)

        assert isinstance(store, FirestoreStateStore)
        mock_fs.assert_called_once_with(
            project="gmail-inbox-triage-515705",
            database="triage-db",
        )
