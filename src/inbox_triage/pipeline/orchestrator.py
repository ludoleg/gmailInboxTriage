"""Concurrent 3-tier GCP Gemini inbox triage orchestrator (Flash-Lite -> Carbon -> Neon)."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from google.genai import types

from inbox_triage.config import Settings
from inbox_triage.llm.router import GenAIRouter
from inbox_triage.models import (
    DeepThreadAnalysis,
    EmailThread,
    ExecutiveBriefing,
    HighStakesReview,
    ModelCallStats,
    Stage1Classification,
    TechExtraction,
    TriageCategory,
    TriagedThread,
    TriageRunReport,
)
from inbox_triage.pipeline import prompts

logger = logging.getLogger(__name__)

NOISE_CATEGORIES = {TriageCategory.PROMO_NOISE, TriageCategory.NEWSLETTER}
HIGH_PRIORITY_CATEGORIES = {TriageCategory.URGENT, TriageCategory.ACTION_REQUIRED}


def _format_thread_text(thread: EmailThread, max_body_chars: int = 16000) -> str:
    """Render an EmailThread into a clean prompt text block."""
    body = thread.body_text
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "\n...[body truncated]"

    parts = [
        f"Thread ID: {thread.thread_id}",
        f"Subject: {thread.subject}",
        f"From: {thread.sender} <{thread.sender_email}>",
        f"Received (UTC): {thread.received_at.isoformat()}",
        f"Messages in thread: {thread.message_count}",
        f"Headers: list_unsubscribe={thread.has_list_unsubscribe}, auto_submitted={thread.is_auto_submitted}",
    ]
    if thread.attachments:
        att_desc = ", ".join(
            f"{a.filename} ({a.kind.value}, {a.size_bytes}B"
            + (f", note={a.note}" if a.note else "")
            + ")"
            for a in thread.attachments
        )
        parts.append(f"Attachments: {att_desc}")

    parts.append(f"\nBody:\n{body}")
    return "\n".join(parts)


def _build_multimodal_contents(thread: EmailThread, prompt_text: str) -> Any:
    """Return text-only string or multimodal [text, Part, ...] list for native Gemini input."""
    media = thread.multimodal_attachments
    if not media:
        return prompt_text
    contents: list[Any] = [prompt_text]
    for att in media:
        if att.data_bytes:
            contents.append(f"\n[Attached {att.kind.value}: {att.filename}]")
            contents.append(types.Part.from_bytes(data=att.data_bytes, mime_type=att.mime_type))
    return contents


class TriageOrchestrator:
    """Runs the concurrent 3-tier GCP Gemini funnel:
    Stage 1: gemini-3.5-flash-lite (bulk classification & noise filtering)
    Stage 2: gemini-4-carbon (concurrent tech extraction + native multimodal deep analysis)
    Stage 3: gemini-4-neon (cross-thread executive briefing + targeted HighStakesReview)
    """

    def __init__(self, settings: Settings, router: GenAIRouter) -> None:
        self.settings = settings
        self.router = router

    def run(self, threads: list[EmailThread]) -> TriageRunReport:
        """Execute the full 3-tier pipeline on the provided unread threads."""
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(UTC)
        stats_lock = threading.Lock()
        call_stats: list[ModelCallStats] = []

        def record_stat(stat: ModelCallStats) -> None:
            with stats_lock:
                call_stats.append(stat)

        # Stage 0 short-circuit: zero LLM calls when inbox has no new unread threads
        if not threads:
            return TriageRunReport(
                run_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                timezone=self.settings.timezone,
                total_threads_fetched=0,
            )

        # --- Stage 1 (Concurrent): Fast bulk triage with Flash-Lite ---
        logger.info(
            "Stage 1: Classifying %d thread(s) concurrently with %s.",
            len(threads),
            self.settings.model_stage1_fast,
        )
        triaged_by_idx: dict[int, TriagedThread] = {}
        with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as pool:
            futures = {
                pool.submit(self._classify_thread, thread, record_stat): idx
                for idx, thread in enumerate(threads)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                triaged_by_idx[idx] = fut.result()

        triaged = [triaged_by_idx[i] for i in range(len(threads))]

        # --- Stage 2 (Concurrent): Tech extraction & native multimodal deep analysis (Carbon) ---
        tech_items = [
            item
            for item in triaged
            if item.category not in NOISE_CATEGORIES
            and (
                item.category == TriageCategory.TECHNICAL_ALERT
                or (item.stage1 is not None and item.stage1.needs_tech_extract)
            )
        ]
        deep_items = [
            item
            for item in triaged
            if item.category not in NOISE_CATEGORIES
            and (
                item.category in HIGH_PRIORITY_CATEGORIES
                or (item.stage1 is not None and item.stage1.needs_deep_analysis)
                or bool(item.thread.multimodal_attachments)
            )
        ]

        if tech_items or deep_items:
            logger.info(
                "Stage 2: Running %d tech extract(s) and %d deep/multimodal analysis call(s) on %s.",
                len(tech_items),
                len(deep_items),
                self.settings.model_stage2_workhorse,
            )
            with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as pool:
                stage2_futures = [
                    pool.submit(self._extract_tech, item, record_stat) for item in tech_items
                ] + [
                    pool.submit(self._analyze_deep, item, record_stat) for item in deep_items
                ]
                for fut in as_completed(stage2_futures):
                    fut.result()

        # --- Stage 3: Executive synthesis & targeted HighStakesReview (Neon) ---
        briefing = self._run_executive_synthesis(triaged, record_stat)
        deep_reviews_count = self._run_high_stakes_reviews(triaged, briefing, record_stat)

        return TriageRunReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            timezone=self.settings.timezone,
            total_threads_fetched=len(threads),
            triaged_threads=triaged,
            briefing=briefing,
            call_stats=call_stats,
            deep_reviews_count=deep_reviews_count,
        )

    def _classify_thread(
        self,
        thread: EmailThread,
        record_stat: Any,
    ) -> TriagedThread:
        prompt = _format_thread_text(thread, max_body_chars=6000)
        try:
            s1, stats = self.router.generate_structured(
                model=self.settings.model_stage1_fast,
                stage="stage1_fast_triage",
                system_prompt=prompts.STAGE1_FAST_TRIAGE_SYSTEM_PROMPT,
                contents=prompt,
                response_schema=Stage1Classification,
                temperature=0.0,
                max_output_tokens=1024,
            )
            record_stat(stats)
            return TriagedThread(
                thread=thread,
                category=s1.category,
                confidence=s1.confidence,
                summary=s1.one_line_summary,
                stage1=s1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Stage 1 failed for thread %s: %s", thread.thread_id, exc)
            return TriagedThread(
                thread=thread,
                category=TriageCategory.FYI,
                confidence=0.0,
                summary=thread.snippet or thread.subject,
                error_note=f"Stage 1 classification failed: {exc}",
            )

    def _extract_tech(self, item: TriagedThread, record_stat: Any) -> None:
        prompt = _format_thread_text(item.thread, max_body_chars=12000)
        try:
            tech, stats = self.router.generate_structured(
                model=self.settings.model_stage2_workhorse,
                stage="stage2_tech_extract",
                system_prompt=prompts.STAGE2_TECH_SYSTEM_PROMPT,
                contents=prompt,
                response_schema=TechExtraction,
                temperature=0.0,
                max_output_tokens=4096,
            )
            record_stat(stats)
            item.tech_extraction = tech
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 tech extraction failed for %s: %s", item.thread.thread_id, exc)

    def _analyze_deep(self, item: TriagedThread, record_stat: Any) -> None:
        prompt_text = _format_thread_text(item.thread, max_body_chars=24000)
        contents = _build_multimodal_contents(item.thread, prompt_text)
        try:
            deep, stats = self.router.generate_structured(
                model=self.settings.model_stage2_workhorse,
                stage="stage2_deep_multimodal",
                system_prompt=prompts.STAGE2_DEEP_MULTIMODAL_SYSTEM_PROMPT,
                contents=contents,
                response_schema=DeepThreadAnalysis,
                temperature=0.1,
                max_output_tokens=4096,
            )
            record_stat(stats)
            item.deep_analysis = deep
            item.category = deep.refined_category
            item.confidence = deep.confidence
            item.summary = deep.summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 deep analysis failed for %s: %s", item.thread.thread_id, exc)

    def _run_executive_synthesis(
        self,
        triaged: list[TriagedThread],
        record_stat: Any,
    ) -> ExecutiveBriefing:
        relevant = [t for t in triaged if t.category not in NOISE_CATEGORIES]
        if not relevant:
            return ExecutiveBriefing(
                headline="Quiet inbox — only newsletters and promotional noise overnight.",
                executive_summary=(
                    f"Filtered {len(triaged)} low-priority newsletter/promotional thread(s). "
                    "No urgent or actionable emails require your attention."
                ),
            )

        cards: list[str] = []
        for item in relevant:
            card = (
                f"- thread_id={item.thread.thread_id} | category={item.category.value} "
                f"| conf={item.confidence:.2f} | from={item.thread.sender} <{item.thread.sender_email}> "
                f"| subject={item.thread.subject}\n"
                f"  summary: {item.summary}"
            )
            if item.deep_analysis:
                if item.deep_analysis.deadline:
                    card += f"\n  deadline: {item.deep_analysis.deadline}"
                if item.deep_analysis.action_items:
                    card += "\n  actions: " + "; ".join(item.deep_analysis.action_items)
                if item.deep_analysis.attachment_findings:
                    card += "\n  attachments: " + "; ".join(item.deep_analysis.attachment_findings)
                if item.deep_analysis.deep_review_recommended:
                    card += f"\n  review_flag: {item.deep_analysis.deep_review_reason}"
            if item.tech_extraction:
                card += (
                    f"\n  tech: [{item.tech_extraction.severity}] "
                    f"{item.tech_extraction.source_system} / {item.tech_extraction.repo_or_service}: "
                    f"{item.tech_extraction.error_or_event}"
                )
            cards.append(card)

        synthesis_prompt = (
            f"Total unread threads triaged: {len(triaged)} "
            f"({len(relevant)} substantive, {len(triaged) - len(relevant)} filtered noise).\n\n"
            "Substantive threads:\n" + "\n".join(cards)
        )

        try:
            briefing, stats = self.router.generate_structured(
                model=self.settings.model_stage3_reasoning,
                stage="stage3_executive_synthesis",
                system_prompt=prompts.STAGE3_EXECUTIVE_SYSTEM_PROMPT,
                contents=synthesis_prompt,
                response_schema=ExecutiveBriefing,
                temperature=0.2,
                max_output_tokens=8192,
            )
            record_stat(stats)
            return briefing
        except Exception as exc:  # noqa: BLE001
            logger.error("Stage 3 executive synthesis failed: %s", exc)
            return ExecutiveBriefing(
                headline=f"{len(relevant)} substantive email(s) triaged this morning.",
                executive_summary="Executive synthesis encountered an error; see itemized threads below.",
            )

    def _run_high_stakes_reviews(
        self,
        triaged: list[TriagedThread],
        briefing: ExecutiveBriefing,
        record_stat: Any,
    ) -> int:
        if self.settings.max_deep_reviews_per_run <= 0:
            return 0

        flagged_ids = set(briefing.deep_review_thread_ids)
        candidates: list[TriagedThread] = []
        for item in triaged:
            if item.category in NOISE_CATEGORIES:
                continue
            needs_review = (
                item.thread.thread_id in flagged_ids
                or (item.deep_analysis is not None and item.deep_analysis.deep_review_recommended)
                or (
                    item.category in HIGH_PRIORITY_CATEGORIES
                    and item.confidence < self.settings.deep_review_confidence_threshold
                )
            )
            if needs_review:
                candidates.append(item)

        reviewed = 0
        for item in candidates[: self.settings.max_deep_reviews_per_run]:
            prompt_text = (
                f"Stage 2 category: {item.category.value} (confidence={item.confidence:.2f})\n\n"
                + _format_thread_text(item.thread, max_body_chars=60000)
            )
            contents = _build_multimodal_contents(item.thread, prompt_text)
            try:
                review, stats = self.router.generate_structured(
                    model=self.settings.model_stage3_reasoning,
                    stage="stage3_high_stakes_review",
                    system_prompt=prompts.STAGE3_HIGH_STAKES_SYSTEM_PROMPT,
                    contents=contents,
                    response_schema=HighStakesReview,
                    temperature=0.1,
                    max_output_tokens=8192,
                )
                record_stat(stats)
                item.high_stakes_review = review
                item.deep_review_model = self.settings.model_stage3_reasoning
                item.category = review.refined_category
                item.summary = review.executive_summary
                item.confidence = max(item.confidence, 0.95)
                reviewed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Stage 3 high-stakes review failed for %s: %s",
                    item.thread.thread_id,
                    exc,
                )

        return reviewed
