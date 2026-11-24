"""Unit tests for TriageOrchestrator multi-model routing, fallback, and cloud guardrails."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from inbox_triage.config import Settings
from inbox_triage.models import (
    AttachmentKind,
    CloudEscalationResult,
    DeepThreadAnalysis,
    EmailAttachment,
    ExecutiveBriefing,
    ModelCallStats,
    MultimodalFinding,
    Stage1Classification,
    TechExtraction,
    TriageCategory,
)
from inbox_triage.pipeline.orchestrator import TriageOrchestrator
from tests.conftest import make_email_thread


def _make_fake_lease(*, acquire_inkling_result: bool = True) -> MagicMock:
    lease = MagicMock()
    lease.__enter__.return_value = lease
    lease.__exit__.return_value = None
    lease.acquire_inkling.return_value = acquire_inkling_result
    return lease


def _stats(backend: str, model: str, stage: str) -> ModelCallStats:
    return ModelCallStats(
        backend=backend,  # type: ignore[arg-type]
        model=model,
        stage=stage,
        prompt_tokens=100,
        completion_tokens=40,
        latency_ms=120.0,
    )


def test_inbox_zero_short_circuit_zero_gpu_switches_zero_llm_calls(test_settings: Settings) -> None:
    router = MagicMock()
    lease = _make_fake_lease()
    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)

    report = orchestrator.run([])

    assert report.total_threads_fetched == 0
    assert report.triaged_threads == []
    assert report.used_inkling is False
    assert report.used_ollama_multimodal is False
    assert report.used_ollama_fallback is False
    assert lease.__enter__.call_count == 0
    assert lease.acquire_ollama.call_count == 0
    assert lease.acquire_inkling.call_count == 0
    assert router.generate_structured.call_count == 0


def test_text_only_inbox_skips_ollama_and_runs_inkling_stages_1_to_3(test_settings: Settings) -> None:
    threads = [
        make_email_thread(thread_id="t-ci", subject="[repo] CI failed", sender_email="ci@github.com"),
        make_email_thread(thread_id="t-action", subject="Approve Q2 budget", sender_email="cfo@acme.example"),
        make_email_thread(thread_id="t-promo", subject="50% off shoes!", has_list_unsubscribe=True),
    ]
    calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> tuple[Any, ModelCallStats]:
        calls.append(kwargs)
        stage = kwargs["stage"]
        contents = str(kwargs["contents"])
        if stage == "stage1_triage":
            if "t-ci" in contents:
                return (
                    Stage1Classification(
                        category=TriageCategory.TECHNICAL_ALERT,
                        confidence=0.95,
                        one_line_summary="CI failed on main",
                        needs_tech_extract=True,
                        needs_deep_analysis=False,
                        reason="CI alert",
                    ),
                    _stats(kwargs["backend"], kwargs["model"], stage),
                )
            if "t-action" in contents:
                return (
                    Stage1Classification(
                        category=TriageCategory.ACTION_REQUIRED,
                        confidence=0.92,
                        one_line_summary="Approve Q2 budget",
                        needs_tech_extract=False,
                        needs_deep_analysis=True,
                        reason="Requires approval",
                    ),
                    _stats(kwargs["backend"], kwargs["model"], stage),
                )
            return (
                Stage1Classification(
                    category=TriageCategory.PROMO_NOISE,
                    confidence=0.99,
                    one_line_summary="Shoe sale promo",
                    needs_tech_extract=False,
                    needs_deep_analysis=False,
                    reason="Marketing",
                ),
                _stats(kwargs["backend"], kwargs["model"], stage),
            )
        if stage == "stage2_tech_extract":
            return (
                TechExtraction(
                    source_system="GitHub",
                    severity="high",
                    repo_or_service="acme/repo",
                    error_or_event="pytest failed",
                ),
                _stats(kwargs["backend"], kwargs["model"], stage),
            )
        if stage == "stage2_deep_analysis":
            return (
                DeepThreadAnalysis(
                    refined_category=TriageCategory.ACTION_REQUIRED,
                    confidence=0.93,
                    summary="CFO asks to approve Q2 budget by Friday.",
                    action_items=["Approve Q2 budget"],
                    deadline="Friday",
                ),
                _stats(kwargs["backend"], kwargs["model"], stage),
            )
        if stage == "stage3_executive_synthesis":
            return (
                ExecutiveBriefing(
                    headline="1 CI failure and 1 budget approval",
                    executive_summary="Review CI failure and approve Q2 budget.",
                    top_priority_thread_ids=["t-action", "t-ci"],
                ),
                _stats(kwargs["backend"], kwargs["model"], stage),
            )
        raise AssertionError(f"Unexpected stage: {stage}")

    router = MagicMock()
    router.cloud_client = None
    router.generate_structured.side_effect = fake_generate
    lease = _make_fake_lease(acquire_inkling_result=True)

    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)
    report = orchestrator.run(threads, allow_cloud=False)

    assert report.used_ollama_multimodal is False
    assert report.used_inkling is True
    assert report.used_ollama_fallback is False
    lease.acquire_ollama.assert_not_called()
    lease.acquire_inkling.assert_called_once()

    # Verify reasoning_effort progression: none -> low -> medium
    s1_calls = [c for c in calls if c["stage"] == "stage1_triage"]
    s2_calls = [c for c in calls if c["stage"].startswith("stage2_")]
    s3_calls = [c for c in calls if c["stage"] == "stage3_executive_synthesis"]

    assert len(s1_calls) == 3
    assert all(c["backend"] == "inkling" and c["reasoning_effort"] == "none" for c in s1_calls)
    assert len(s2_calls) == 2  # promo was dropped before Stage 2
    assert all(c["backend"] == "inkling" and c["reasoning_effort"] == "low" for c in s2_calls)
    assert len(s3_calls) == 1
    assert s3_calls[0]["backend"] == "inkling"
    assert s3_calls[0]["reasoning_effort"] == "medium"
    assert "t-promo" not in str(s3_calls[0]["contents"])


def test_multimodal_inbox_runs_phase_a_gemma4_12b_then_unloads_then_inkling(
    test_settings: Settings, tiny_png_bytes: bytes, tiny_wav_bytes: bytes
) -> None:
    mm_thread = make_email_thread(
        thread_id="t-mm",
        subject="Receipt and voicemail",
        attachments=[
            EmailAttachment(
                filename="receipt.png",
                mime_type="image/png",
                kind=AttachmentKind.IMAGE,
                size_bytes=len(tiny_png_bytes),
                data_bytes=tiny_png_bytes,
            ),
            EmailAttachment(
                filename="vm.wav",
                mime_type="audio/wav",
                kind=AttachmentKind.AUDIO,
                size_bytes=len(tiny_wav_bytes),
                data_bytes=tiny_wav_bytes,
            ),
        ],
    )
    calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> tuple[Any, ModelCallStats]:
        calls.append(kwargs)
        stage = kwargs["stage"]
        if stage == "phase_a_multimodal":
            fname = "receipt.png" if "receipt.png" in str(kwargs["contents"][0]) else "vm.wav"
            return (
                MultimodalFinding(
                    filename=fname,
                    summary=f"Extracted from {fname}",
                    extracted_key_facts=["$480 due"],
                    urgency_signal="high",
                ),
                _stats("ollama", kwargs["model"], stage),
            )
        if stage == "stage1_triage":
            return (
                Stage1Classification(
                    category=TriageCategory.URGENT,
                    confidence=0.95,
                    one_line_summary="Urgent receipt + voicemail",
                    needs_deep_analysis=True,
                    reason="Urgent attachment",
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        if stage == "stage2_deep_analysis":
            return (
                DeepThreadAnalysis(
                    refined_category=TriageCategory.URGENT,
                    confidence=0.95,
                    summary="Pay $480 today.",
                    action_items=["Pay invoice"],
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        return (
            ExecutiveBriefing(headline="1 urgent item", executive_summary="Pay $480 today."),
            _stats("inkling", kwargs["model"], stage),
        )

    router = MagicMock()
    router.cloud_client = None
    router.generate_structured.side_effect = fake_generate
    lease = _make_fake_lease(acquire_inkling_result=True)

    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)
    report = orchestrator.run([mm_thread], allow_cloud=False)

    assert report.used_ollama_multimodal is True
    assert report.used_inkling is True
    lease.acquire_ollama.assert_called_once()
    router.unload_ollama_model.assert_called_once_with(test_settings.model_multimodal)
    lease.acquire_inkling.assert_called_once()

    mm_calls = [c for c in calls if c["stage"] == "phase_a_multimodal"]
    assert len(mm_calls) == 2
    # Image uses think=True, audio uses think=False
    assert mm_calls[0]["model"] == "gemma4:12b"
    assert mm_calls[0]["think"] is True
    assert mm_calls[1]["model"] == "gemma4:12b"
    assert mm_calls[1]["think"] is False

    # Stage 1 prompt receives multimodal findings
    s1_call = next(c for c in calls if c["stage"] == "stage1_triage")
    assert "Extracted from receipt.png" in str(s1_call["contents"])
    assert "Extracted from vm.wav" in str(s1_call["contents"])
    assert len(report.triaged_threads[0].multimodal_findings) == 2


def test_inkling_unavailable_falls_back_to_ollama_models_with_unloads(test_settings: Settings) -> None:
    threads = [
        make_email_thread(thread_id="t-ci", subject="CI alert"),
        make_email_thread(thread_id="t-urgent", subject="Urgent request"),
    ]
    calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> tuple[Any, ModelCallStats]:
        calls.append(kwargs)
        stage = kwargs["stage"]
        if stage == "stage1_triage":
            if "t-ci" in str(kwargs["contents"]):
                return (
                    Stage1Classification(
                        category=TriageCategory.TECHNICAL_ALERT,
                        confidence=0.9,
                        one_line_summary="CI failed",
                        needs_tech_extract=True,
                        reason="CI",
                    ),
                    _stats("ollama", kwargs["model"], stage),
                )
            return (
                Stage1Classification(
                    category=TriageCategory.URGENT,
                    confidence=0.9,
                    one_line_summary="Urgent item",
                    needs_deep_analysis=True,
                    reason="Urgent",
                ),
                _stats("ollama", kwargs["model"], stage),
            )
        if stage == "stage2_tech_extract":
            return (
                TechExtraction(source_system="GitHub", severity="high", error_or_event="Build failed"),
                _stats("ollama", kwargs["model"], stage),
            )
        if stage == "stage2_deep_analysis":
            return (
                DeepThreadAnalysis(
                    refined_category=TriageCategory.URGENT,
                    confidence=0.92,
                    summary="Act immediately.",
                ),
                _stats("ollama", kwargs["model"], stage),
            )
        return (
            ExecutiveBriefing(headline="Fallback brief", executive_summary="2 items."),
            _stats("ollama", kwargs["model"], stage),
        )

    router = MagicMock()
    router.cloud_client = None
    router.generate_structured.side_effect = fake_generate
    lease = _make_fake_lease(acquire_inkling_result=False)

    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)
    report = orchestrator.run(threads, allow_cloud=False)

    assert report.used_inkling is False
    assert report.used_ollama_fallback is True
    lease.acquire_ollama.assert_called_once()

    stage_models = [(c["stage"], c["model"]) for c in calls]
    assert ("stage1_triage", "gemma4:e2b") in stage_models
    assert ("stage2_tech_extract", "qwen2.5-coder:3b") in stage_models
    assert ("stage2_deep_analysis", "gemma4:12b") in stage_models
    assert ("stage3_executive_synthesis", "bonsai-2:27b") in stage_models

    # All 4 fallback models unloaded sequentially between stages
    unloaded = [call.args[0] for call in router.unload_ollama_model.call_args_list]
    assert unloaded == ["gemma4:e2b", "qwen2.5-coder:3b", "gemma4:12b", "bonsai-2:27b"]


def test_phase_c_cloud_routing_denylist_and_budget_cap(test_settings: Settings) -> None:
    test_settings.inkling_overflow_chars = 100
    test_settings.max_cloud_escalations_per_run = 2
    test_settings.cloud_denylist_domains = "bank.example"

    threads = [
        # 1. Denylisted domain + low confidence -> MUST be blocked
        make_email_thread(
            thread_id="t-denied",
            subject="Wire transfer approval",
            sender_email="alerts@bank.example",
        ),
        # 2. Low confidence URGENT -> gemini-4-neon
        make_email_thread(
            thread_id="t-lowconf",
            subject="Ambiguous contract term",
            sender_email="legal@partner.example",
        ),
        # 3. Oversized FYI (> 100 chars) -> gemini-4-carbon
        make_email_thread(
            thread_id="t-oversized",
            subject="Huge log dump",
            sender_email="ops@acme.example",
            body_text="X" * 200,
        ),
        # 4. Another flagged thread -> skipped because max_cloud_escalations_per_run=2
        make_email_thread(
            thread_id="t-overcap",
            subject="Another complex thread",
            sender_email="exec@partner.example",
        ),
    ]

    cloud_calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> tuple[Any, ModelCallStats]:
        stage = kwargs["stage"]
        contents = str(kwargs["contents"])
        if stage == "stage1_triage":
            if "t-oversized" in contents:
                return (
                    Stage1Classification(
                        category=TriageCategory.FYI,
                        confidence=0.90,
                        one_line_summary="Huge log",
                        reason="FYI",
                    ),
                    _stats("inkling", kwargs["model"], stage),
                )
            return (
                Stage1Classification(
                    category=TriageCategory.URGENT,
                    confidence=0.55,  # below 0.70 threshold
                    one_line_summary="Needs review",
                    needs_deep_analysis=True,
                    reason="Uncertain",
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        if stage == "stage2_deep_analysis":
            return (
                DeepThreadAnalysis(
                    refined_category=TriageCategory.URGENT,
                    confidence=0.55,
                    summary="Ambiguous",
                    escalation_recommended=True,
                    escalation_reason="High stakes",
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        if stage == "stage3_executive_synthesis":
            return (
                ExecutiveBriefing(
                    headline="Needs cloud check",
                    executive_summary="Several uncertain threads.",
                    cloud_escalation_thread_ids=["t-overcap"],
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        if kwargs["backend"] == "cloud":
            cloud_calls.append(kwargs)
            return (
                CloudEscalationResult(
                    refined_category=TriageCategory.URGENT,
                    executive_summary="Authoritative cloud summary.",
                    action_items=["Follow up"],
                    risk_assessment="Moderate risk",
                    suggested_next_step="Reply today",
                ),
                _stats("cloud", kwargs["model"], stage),
            )
        raise AssertionError(f"Unexpected call: {kwargs}")

    router = MagicMock()
    router.cloud_client = MagicMock()
    router.generate_structured.side_effect = fake_generate
    lease = _make_fake_lease(acquire_inkling_result=True)

    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)
    report = orchestrator.run(threads, allow_cloud=True)

    assert report.cloud_blocked_by_denylist == 1
    assert report.cloud_escalations_count == 2
    assert len(cloud_calls) == 2

    # First escalated: t-lowconf -> gemini-4-neon; second: t-oversized -> gemini-4-carbon
    by_tid = {t.thread.thread_id: t for t in report.triaged_threads}
    assert by_tid["t-denied"].cloud_model_used is None
    assert by_tid["t-lowconf"].cloud_model_used == "gemini-4-neon"
    assert by_tid["t-oversized"].cloud_model_used == "gemini-4-carbon"
    assert by_tid["t-overcap"].cloud_model_used is None  # capped at 2


def test_per_thread_fault_isolation_when_stage1_fails(test_settings: Settings) -> None:
    threads = [
        make_email_thread(thread_id="t-bad", subject="Broken thread", snippet="Broken snippet"),
        make_email_thread(thread_id="t-good", subject="Good thread"),
    ]

    def fake_generate(**kwargs: Any) -> tuple[Any, ModelCallStats]:
        stage = kwargs["stage"]
        contents = str(kwargs["contents"])
        if stage == "stage1_triage":
            if "t-bad" in contents:
                raise RuntimeError("Simulated JSON failure")
            return (
                Stage1Classification(
                    category=TriageCategory.FYI,
                    confidence=0.9,
                    one_line_summary="All good",
                    reason="Normal",
                ),
                _stats("inkling", kwargs["model"], stage),
            )
        return (
            ExecutiveBriefing(headline="Partial ok", executive_summary="Good thread survived."),
            _stats("inkling", kwargs["model"], stage),
        )

    router = MagicMock()
    router.cloud_client = None
    router.generate_structured.side_effect = fake_generate
    lease = _make_fake_lease(acquire_inkling_result=True)

    orchestrator = TriageOrchestrator(settings=test_settings, router=router, gpu_lease=lease)
    report = orchestrator.run(threads, allow_cloud=False)

    assert len(report.triaged_threads) == 2
    by_tid = {t.thread.thread_id: t for t in report.triaged_threads}
    assert by_tid["t-bad"].category == TriageCategory.FYI
    assert by_tid["t-bad"].confidence == 0.0
    assert by_tid["t-bad"].error_note is not None
    assert "Simulated JSON failure" in by_tid["t-bad"].error_note
    assert by_tid["t-good"].summary == "All good"
    assert by_tid["t-good"].error_note is None
