"""Unit tests for the concurrent 3-tier TriageOrchestrator pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from google.genai import types
from pydantic import BaseModel

from inbox_triage.config import Settings
from inbox_triage.models import (
    AttachmentKind,
    DeepThreadAnalysis,
    EmailAttachment,
    EmailThread,
    ExecutiveBriefing,
    HighStakesReview,
    ModelCallStats,
    Stage1Classification,
    TechExtraction,
    TriageCategory,
)
from inbox_triage.pipeline.orchestrator import TriageOrchestrator
from tests.conftest import make_email_thread


@dataclass
class RecordedCall:
    model: str
    stage: str
    contents: Any
    response_schema: type[BaseModel]
    temperature: float
    max_output_tokens: int


class FakeGenAIRouter:
    """Thread-safe stub implementing GenAIRouter.generate_structured for pipeline tests."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self.calls: list[RecordedCall] = []

    def generate_structured(
        self,
        *,
        model: str,
        stage: str,
        system_prompt: str,
        contents: Any,
        response_schema: type[BaseModel],
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        max_retries: int = 3,
    ) -> tuple[Any, ModelCallStats]:
        call = RecordedCall(
            model=model,
            stage=stage,
            contents=contents,
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        with self._lock:
            self.calls.append(call)
        result = self._handler(call)
        stats = ModelCallStats(
            backend="vertexai",
            model=model,
            stage=stage,
            prompt_tokens=150,
            completion_tokens=60,
            latency_ms=42.0,
        )
        return result, stats


def _extract_prompt_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        return "\n".join(c for c in contents if isinstance(c, str))
    return str(contents)


class TestTriageOrchestrator:
    def test_inbox_zero_short_circuits_with_zero_llm_calls(self, test_settings: Settings) -> None:
        router = FakeGenAIRouter(handler=lambda _c: (_ for _ in ()).throw(AssertionError("Must not call LLM")))
        orchestrator = TriageOrchestrator(settings=test_settings, router=router)  # type: ignore[arg-type]

        report = orchestrator.run([])

        assert router.calls == []
        assert report.total_threads_fetched == 0
        assert report.triaged_threads == []
        assert report.briefing is None
        assert report.deep_reviews_count == 0
        assert report.call_stats == []

    def test_full_three_tier_funnel_with_multimodal_parts_and_noise_filtering(
        self,
        test_settings: Settings,
        synthetic_inbox_threads: list[EmailThread],
        tiny_png_bytes: bytes,
        tiny_wav_bytes: bytes,
        tiny_pdf_bytes: bytes,
    ) -> None:
        def _handler(call: RecordedCall) -> BaseModel:
            text = _extract_prompt_text(call.contents)

            if call.stage == "stage1_fast_triage":
                if "thr-urgent-receipt" in text:
                    return Stage1Classification(
                        category=TriageCategory.URGENT,
                        confidence=0.95,
                        one_line_summary="Overdue hosting invoice due at 5 PM PT.",
                        needs_tech_extract=False,
                        needs_deep_analysis=True,
                        reason="Service suspension deadline today.",
                    )
                if "thr-ci-failure" in text:
                    return Stage1Classification(
                        category=TriageCategory.TECHNICAL_ALERT,
                        confidence=0.98,
                        one_line_summary="CI failed on acme/payments-api main.",
                        needs_tech_extract=True,
                        needs_deep_analysis=False,
                        reason="GitHub Actions failure notification.",
                    )
                if "thr-promo-sale" in text:
                    return Stage1Classification(
                        category=TriageCategory.PROMO_NOISE,
                        confidence=0.99,
                        one_line_summary="40% off ergonomic chairs sale.",
                        reason="Promotional marketing email.",
                    )
                if "thr-newsletter" in text:
                    return Stage1Classification(
                        category=TriageCategory.NEWSLETTER,
                        confidence=0.97,
                        one_line_summary="Weekly Cloud Digest #88.",
                        reason="Subscribed weekly newsletter.",
                    )
                return Stage1Classification(
                    category=TriageCategory.ACTION_REQUIRED,
                    confidence=0.62,  # Below 0.70 threshold -> triggers Stage 3 HighStakesReview
                    one_line_summary="Confirm Q2 roadmap review time tomorrow.",
                    needs_deep_analysis=True,
                    reason="Partner requests time confirmation.",
                )

            if call.stage == "stage2_tech_extract":
                return TechExtraction(
                    source_system="GitHub Actions",
                    severity="high",
                    repo_or_service="acme/payments-api",
                    error_or_event="AssertionError in test_idempotency on main",
                    action_url="https://github.com/acme/payments-api/actions/runs/987654321",
                )

            if call.stage == "stage2_deep_multimodal":
                if "thr-urgent-receipt" in text:
                    return DeepThreadAnalysis(
                        refined_category=TriageCategory.URGENT,
                        confidence=0.94,
                        summary="Pay $480.00 CloudHost invoice #INV-9921 before 5 PM PT today.",
                        action_items=["Update payment method on CloudHost", "Confirm invoice #INV-9921"],
                        deadline="Today 5:00 PM PT",
                        attachment_findings=[
                            "receipt.png shows $480.00 balance",
                            "voicemail.wav confirms 5 PM cutoff",
                            "statement.pdf lists invoice #INV-9921",
                        ],
                        suggested_reply_points=["Payment updated, please confirm receipt"],
                        deep_review_recommended=True,
                        deep_review_reason="Production suspension risk.",
                    )
                return DeepThreadAnalysis(
                    refined_category=TriageCategory.ACTION_REQUIRED,
                    confidence=0.65,
                    summary="Confirm 10 AM or 2 PM PT tomorrow for Q2 roadmap review.",
                    action_items=["Reply to Maya with preferred slot"],
                    deadline="End of day",
                )

            if call.stage == "stage3_executive_synthesis":
                return ExecutiveBriefing(
                    headline="Hosting suspension risk today + broken CI on payments-api.",
                    executive_summary="Resolve the overdue CloudHost invoice first, then fix CI and reply to Maya.",
                    top_priority_thread_ids=["thr-urgent-receipt", "thr-ci-failure", "thr-action-meeting"],
                    cross_thread_connections=[],
                    deep_review_thread_ids=["thr-urgent-receipt"],
                )

            if call.stage == "stage3_high_stakes_review":
                if "thr-urgent-receipt" in text:
                    return HighStakesReview(
                        refined_category=TriageCategory.URGENT,
                        executive_summary="Immediate payment of $480.00 required before 5 PM PT to prevent cluster suspension.",
                        action_items=["Pay invoice #INV-9921 immediately"],
                        deadline="Today 5:00 PM PT",
                        risk_assessment="Production outage if unpaid by 5 PM PT.",
                        suggested_next_step="Log into CloudHost billing portal and settle #INV-9921 now.",
                    )
                return HighStakesReview(
                    refined_category=TriageCategory.ACTION_REQUIRED,
                    executive_summary="Confirm tomorrow's Q2 roadmap review slot with Maya.",
                    action_items=["Pick 10 AM or 2 PM PT"],
                    deadline="End of day",
                    risk_assessment="Scheduling delay with partner.",
                    suggested_next_step="Reply confirming 10 AM PT.",
                )

            raise AssertionError(f"Unexpected stage: {call.stage}")

        router = FakeGenAIRouter(handler=_handler)
        orchestrator = TriageOrchestrator(settings=test_settings, router=router)  # type: ignore[arg-type]

        report = orchestrator.run(synthetic_inbox_threads)

        # Input order is preserved despite concurrent Stage 1 execution
        assert [t.thread.thread_id for t in report.triaged_threads] == [
            t.thread_id for t in synthetic_inbox_threads
        ]

        # Stage 1: all 5 threads classified by gemini-3.5-flash-lite
        s1_calls = [c for c in router.calls if c.stage == "stage1_fast_triage"]
        assert len(s1_calls) == 5
        assert all(c.model == "gemini-3.5-flash-lite" for c in s1_calls)

        # Stage 2: noise (promo + newsletter) dropped; only CI gets tech_extract; urgent + action get deep_multimodal
        tech_calls = [c for c in router.calls if c.stage == "stage2_tech_extract"]
        deep_calls = [c for c in router.calls if c.stage == "stage2_deep_multimodal"]
        assert len(tech_calls) == 1
        assert tech_calls[0].model == "gemini-4-carbon"
        assert "thr-ci-failure" in _extract_prompt_text(tech_calls[0].contents)

        assert len(deep_calls) == 2
        assert all(c.model == "gemini-4-carbon" for c in deep_calls)
        for c in deep_calls:
            txt = _extract_prompt_text(c.contents)
            assert "thr-promo-sale" not in txt
            assert "thr-newsletter" not in txt

        # Verify native multimodal types.Part passed for image, audio, AND PDF on urgent thread
        urgent_deep_call = next(c for c in deep_calls if "thr-urgent-receipt" in _extract_prompt_text(c.contents))
        assert isinstance(urgent_deep_call.contents, list)
        parts = [p for p in urgent_deep_call.contents if isinstance(p, types.Part)]
        assert len(parts) == 3
        part_map = {p.inline_data.mime_type: p.inline_data.data for p in parts}  # type: ignore[union-attr]
        assert part_map == {
            "image/png": tiny_png_bytes,
            "audio/wav": tiny_wav_bytes,
            "application/pdf": tiny_pdf_bytes,
        }

        # Stage 3: gemini-4-neon with max_output_tokens=8192 for both synthesis and high-stakes review
        synth_calls = [c for c in router.calls if c.stage == "stage3_executive_synthesis"]
        assert len(synth_calls) == 1
        assert synth_calls[0].model == "gemini-4-neon"
        assert synth_calls[0].max_output_tokens == 8192

        review_calls = [c for c in router.calls if c.stage == "stage3_high_stakes_review"]
        assert len(review_calls) == 2  # urgent (flagged) + action (confidence 0.65 < 0.70)
        assert all(c.model == "gemini-4-neon" for c in review_calls)
        assert all(c.max_output_tokens == 8192 for c in review_calls)

        # Verify Stage 3 HighStakesReview also passes multimodal Parts for urgent thread
        urgent_review_call = next(
            c for c in review_calls if "thr-urgent-receipt" in _extract_prompt_text(c.contents)
        )
        assert len([p for p in urgent_review_call.contents if isinstance(p, types.Part)]) == 3

        assert report.deep_reviews_count == 2
        by_id = {t.thread.thread_id: t for t in report.triaged_threads}
        assert by_id["thr-urgent-receipt"].deep_review_model == "gemini-4-neon"
        assert by_id["thr-urgent-receipt"].high_stakes_review is not None
        assert by_id["thr-urgent-receipt"].confidence >= 0.95
        assert by_id["thr-ci-failure"].tech_extraction is not None
        assert by_id["thr-promo-sale"].deep_analysis is None
        assert by_id["thr-newsletter"].deep_analysis is None
        assert report.briefing is not None
        assert "Hosting suspension" in report.briefing.headline

    def test_fyi_thread_with_multimodal_attachment_still_gets_stage2_deep_analysis(
        self, test_settings: Settings, tiny_pdf_bytes: bytes
    ) -> None:
        thread = make_email_thread(
            thread_id="thr-fyi-pdf",
            subject="Updated Benefit Summary PDF",
            attachments=[
                EmailAttachment(
                    filename="benefits.pdf",
                    mime_type="application/pdf",
                    kind=AttachmentKind.PDF,
                    size_bytes=len(tiny_pdf_bytes),
                    data_bytes=tiny_pdf_bytes,
                )
            ],
        )

        def _handler(call: RecordedCall) -> BaseModel:
            if call.stage == "stage1_fast_triage":
                return Stage1Classification(
                    category=TriageCategory.FYI,
                    confidence=0.90,
                    one_line_summary="Benefits summary attached.",
                    needs_deep_analysis=False,
                    reason="Informational HR update.",
                )
            if call.stage == "stage2_deep_multimodal":
                return DeepThreadAnalysis(
                    refined_category=TriageCategory.FYI,
                    confidence=0.92,
                    summary="Updated 2026 benefits summary PDF.",
                    attachment_findings=["Deductible unchanged"],
                )
            return ExecutiveBriefing(
                headline="One FYI update.",
                executive_summary="HR sent updated benefits PDF.",
            )

        router = FakeGenAIRouter(handler=_handler)
        report = TriageOrchestrator(settings=test_settings, router=router).run([thread])  # type: ignore[arg-type]

        deep_calls = [c for c in router.calls if c.stage == "stage2_deep_multimodal"]
        assert len(deep_calls) == 1
        assert any(isinstance(p, types.Part) for p in deep_calls[0].contents)
        assert report.triaged_threads[0].deep_analysis is not None

    def test_all_noise_threads_skip_stage2_and_stage3_llm_calls(
        self, test_settings: Settings
    ) -> None:
        threads = [
            make_email_thread(thread_id="t-promo", subject="Buy now!"),
            make_email_thread(thread_id="t-news", subject="Weekly digest"),
        ]

        def _handler(call: RecordedCall) -> BaseModel:
            assert call.stage == "stage1_fast_triage"
            cat = TriageCategory.PROMO_NOISE if "t-promo" in str(call.contents) else TriageCategory.NEWSLETTER
            return Stage1Classification(
                category=cat,
                confidence=0.98,
                one_line_summary="Noise",
                reason="Noise",
            )

        router = FakeGenAIRouter(handler=_handler)
        report = TriageOrchestrator(settings=test_settings, router=router).run(threads)  # type: ignore[arg-type]

        assert len(router.calls) == 2
        assert report.briefing is not None
        assert "Quiet inbox" in report.briefing.headline
        assert report.deep_reviews_count == 0

    def test_enforces_max_deep_reviews_per_run_cap(self, test_settings: Settings) -> None:
        capped_settings = test_settings.model_copy(update={"max_deep_reviews_per_run": 2})
        threads = [make_email_thread(thread_id=f"thr-{i}", subject=f"Urgent {i}") for i in range(4)]

        def _handler(call: RecordedCall) -> BaseModel:
            if call.stage == "stage1_fast_triage":
                return Stage1Classification(
                    category=TriageCategory.URGENT,
                    confidence=0.50,
                    one_line_summary="Ambiguous urgent email",
                    needs_deep_analysis=True,
                    reason="Low confidence",
                )
            if call.stage == "stage2_deep_multimodal":
                return DeepThreadAnalysis(
                    refined_category=TriageCategory.URGENT,
                    confidence=0.55,
                    summary="Still ambiguous",
                    deep_review_recommended=True,
                    deep_review_reason="Needs Neon",
                )
            if call.stage == "stage3_executive_synthesis":
                return ExecutiveBriefing(
                    headline="Four urgent threads.",
                    executive_summary="All four need review.",
                    deep_review_thread_ids=[f"thr-{i}" for i in range(4)],
                )
            return HighStakesReview(
                refined_category=TriageCategory.URGENT,
                executive_summary="Reviewed by Neon.",
                risk_assessment="High",
                suggested_next_step="Respond promptly.",
            )

        router = FakeGenAIRouter(handler=_handler)
        report = TriageOrchestrator(settings=capped_settings, router=router).run(threads)  # type: ignore[arg-type]

        review_calls = [c for c in router.calls if c.stage == "stage3_high_stakes_review"]
        assert len(review_calls) == 2
        assert report.deep_reviews_count == 2

    def test_per_thread_fault_isolation_across_stages(self, test_settings: Settings) -> None:
        threads = [
            make_email_thread(thread_id="thr-s1-boom", subject="Malformed"),
            make_email_thread(thread_id="thr-s2-boom", subject="Stage 2 fails"),
            make_email_thread(thread_id="thr-healthy", subject="Healthy urgent"),
        ]

        def _handler(call: RecordedCall) -> BaseModel:
            text = _extract_prompt_text(call.contents)
            if call.stage == "stage1_fast_triage":
                if "thr-s1-boom" in text:
                    raise RuntimeError("Transient Stage 1 failure")
                return Stage1Classification(
                    category=TriageCategory.URGENT,
                    confidence=0.60,
                    one_line_summary="Urgent item",
                    needs_deep_analysis=True,
                    reason="Urgent",
                )
            if call.stage == "stage2_deep_multimodal":
                if "thr-s2-boom" in text:
                    raise RuntimeError("Stage 2 exploded")
                return DeepThreadAnalysis(
                    refined_category=TriageCategory.URGENT,
                    confidence=0.92,
                    summary="Healthy urgent summary",
                    deep_review_recommended=True,
                )
            if call.stage == "stage3_executive_synthesis":
                raise RuntimeError("Stage 3 synthesis down")
            if call.stage == "stage3_high_stakes_review":
                if "thr-s2-boom" in text:
                    raise RuntimeError("Stage 3 review failed for s2-boom")
                return HighStakesReview(
                    refined_category=TriageCategory.URGENT,
                    executive_summary="Healthy Neon review",
                    risk_assessment="Manageable",
                    suggested_next_step="Reply today.",
                )
            raise AssertionError(f"Unexpected stage {call.stage}")

        router = FakeGenAIRouter(handler=_handler)
        report = TriageOrchestrator(settings=test_settings, router=router).run(threads)  # type: ignore[arg-type]

        by_id = {t.thread.thread_id: t for t in report.triaged_threads}

        # Stage 1 failure degraded cleanly to FYI with error_note
        assert by_id["thr-s1-boom"].category == TriageCategory.FYI
        assert by_id["thr-s1-boom"].confidence == 0.0
        assert by_id["thr-s1-boom"].error_note is not None
        assert "Transient Stage 1 failure" in by_id["thr-s1-boom"].error_note

        # Stage 2 failure preserved Stage 1 summary/category without crashing
        assert by_id["thr-s2-boom"].category == TriageCategory.URGENT
        assert by_id["thr-s2-boom"].deep_analysis is None
        assert by_id["thr-s2-boom"].high_stakes_review is None

        # Healthy thread completed all 3 stages including HighStakesReview
        assert by_id["thr-healthy"].deep_analysis is not None
        assert by_id["thr-healthy"].high_stakes_review is not None
        assert by_id["thr-healthy"].deep_review_model == "gemini-4-neon"
        assert report.deep_reviews_count == 1

        # Stage 3 synthesis failure produced graceful fallback briefing
        assert report.briefing is not None
        assert "3 substantive email(s)" in report.briefing.headline
