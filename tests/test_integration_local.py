"""Live local model integration test against real Ollama + inkling-ctl + inkling_engine."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from inbox_triage.config import Settings
from inbox_triage.llm.router import ModelRouter
from inbox_triage.models import EmailThread, ExecutiveBriefing, TriageCategory
from inbox_triage.pipeline.orchestrator import TriageOrchestrator
from inbox_triage.slack.formatter import MAX_BLOCKS_PER_MESSAGE, build_slack_payloads
from inbox_triage.state import StateStore


def _is_inkling_service_active() -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", "inkling-server.service"],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "active"


@pytest.mark.integration
def test_live_local_hybrid_pipeline_ollama_multimodal_to_inkling_and_restore(
    tmp_path: Path,
    synthetic_inbox_threads: list[EmailThread],
) -> None:
    """End-to-end local test: gemma4:12b (Ollama) -> inkling-2-mini (Stages 1-3) -> GPU restore."""
    inkling_ctl = Path("/home/ludo/.local/bin/inkling-ctl")
    if not inkling_ctl.exists():
        pytest.skip(f"inkling-ctl not found at {inkling_ctl}")

    pre_test_active = _is_inkling_service_active()

    settings = Settings(
        _env_file=None,
        STATE_DB_PATH=tmp_path / "state.db",
        INKLING_CTL_PATH=inkling_ctl,
        INKLING_BASE_URL="http://127.0.0.1:8001",
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        GPU_RESTORE_POLICY="restore",
        CLOUD_ESCALATION_ENABLED=False,
        GEMINI_API_KEY=None,
    )

    router = ModelRouter(settings=settings)
    orchestrator = TriageOrchestrator(settings=settings, router=router)

    report = orchestrator.run(synthetic_inbox_threads, allow_cloud=False, allow_inkling=True)

    # 1. Verify GPU state was restored to exact pre-test state
    post_test_active = _is_inkling_service_active()
    assert post_test_active == pre_test_active, (
        f"inkling-server.service state changed: before={pre_test_active}, after={post_test_active}"
    )

    # 2. Verify hybrid execution path flags
    assert report.total_threads_fetched == 4
    assert report.used_ollama_multimodal is True
    assert report.used_inkling is True
    assert report.used_ollama_fallback is False
    assert report.cloud_escalations_count == 0
    assert len(report.triaged_threads) == 4

    # 3. Verify call telemetry spans both Ollama multimodal and Inkling stages
    backends_used = {s.backend for s in report.call_stats}
    assert backends_used == {"ollama", "inkling"}
    models_used = {s.model for s in report.call_stats}
    assert settings.model_multimodal in models_used
    assert settings.model_inkling in models_used

    # 4. Verify per-thread structured outputs
    by_tid = {t.thread.thread_id: t for t in report.triaged_threads}

    receipt = by_tid["thr-urgent-receipt"]
    assert receipt.error_note is None
    assert len(receipt.multimodal_findings) == 1
    assert receipt.multimodal_findings[0].summary.strip() != ""
    assert receipt.category in (TriageCategory.URGENT, TriageCategory.ACTION_REQUIRED)

    ci_alert = by_tid["thr-ci-failure"]
    assert ci_alert.error_note is None
    assert ci_alert.category in (TriageCategory.TECHNICAL_ALERT, TriageCategory.URGENT)

    promo = by_tid["thr-promo-sale"]
    assert promo.error_note is None
    assert promo.category in (TriageCategory.PROMO_NOISE, TriageCategory.NEWSLETTER)

    meeting = by_tid["thr-action-meeting"]
    assert meeting.error_note is None
    assert meeting.category in (TriageCategory.ACTION_REQUIRED, TriageCategory.URGENT)

    # 5. Verify Stage 3 ExecutiveBriefing
    assert isinstance(report.briefing, ExecutiveBriefing)
    assert report.briefing.headline.strip() != ""
    assert report.briefing.executive_summary.strip() != ""

    # 6. Verify Slack Block Kit payload generation & StateStore persistence
    payloads = build_slack_payloads(report)
    assert len(payloads) >= 1
    for payload in payloads:
        assert payload["text"]
        assert 1 <= len(payload["blocks"]) <= MAX_BLOCKS_PER_MESSAGE

    store = StateStore(settings.state_db_path)
    store.record_run(report)
    for t in synthetic_inbox_threads:
        assert store.is_already_processed(t.thread_id, t.last_message_id) is True
