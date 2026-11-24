"""Inkling-primary, GPU-exclusive multi-model inbox triage orchestrator."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Literal

from google.genai import types

from inbox_triage.config import Settings
from inbox_triage.llm.gpu_lease import GpuLeaseManager
from inbox_triage.llm.router import ModelRouter
from inbox_triage.models import (
    AttachmentKind,
    CloudEscalationResult,
    DeepThreadAnalysis,
    EmailThread,
    ExecutiveBriefing,
    ModelCallStats,
    MultimodalFinding,
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


def _format_thread_for_prompt(
    thread: EmailThread,
    multimodal_findings: list[MultimodalFinding] | None = None,
    max_body_chars: int = 12000,
) -> str:
    """Render an EmailThread (plus PDF text & multimodal findings) into a compact prompt block."""
    body = thread.body_text
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "\n...[body truncated for local stage]"

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

    for att in thread.attachments:
        if att.extracted_text:
            pdf_snippet = att.extracted_text[:6000]
            parts.append(f"\n[PDF Text: {att.filename}]\n{pdf_snippet}")

    if multimodal_findings:
        parts.append("\n[Multimodal Attachment Findings]")
        for finding in multimodal_findings:
            facts = "; ".join(finding.extracted_key_facts)
            parts.append(
                f"- {finding.filename} (urgency={finding.urgency_signal}): {finding.summary}"
                + (f" | Facts: {facts}" if facts else "")
            )

    return "\n".join(parts)


class TriageOrchestrator:
    """Coordinates Stage 0 -> Conditional Phase A (Ollama MM) -> Phase B (Inkling) -> Phase C (Cloud Gemini 4)."""

    def __init__(
        self,
        settings: Settings,
        router: ModelRouter,
        gpu_lease: GpuLeaseManager | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self._gpu_lease = gpu_lease

    def _make_lease(self) -> GpuLeaseManager:
        if self._gpu_lease is not None:
            return self._gpu_lease
        return GpuLeaseManager(
            inkling_ctl_path=self.settings.inkling_ctl_path,
            inkling_base_url=self.settings.inkling_base_url,
            restore_policy=self.settings.gpu_restore_policy,
        )

    def run(
        self,
        threads: list[EmailThread],
        *,
        allow_cloud: bool = True,
        allow_inkling: bool = True,
    ) -> TriageRunReport:
        """Execute the full multi-model triage funnel on the provided unread threads."""
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(UTC)
        call_stats: list[ModelCallStats] = []

        # Stage 0 short-circuit: zero GPU switches when inbox has no new unread threads
        if not threads:
            return TriageRunReport(
                run_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                timezone=self.settings.timezone,
                total_threads_fetched=0,
            )

        mm_findings_by_tid: dict[str, list[MultimodalFinding]] = {}
        triaged: list[TriagedThread] = []
        briefing: ExecutiveBriefing | None = None
        used_ollama_mm = False
        used_inkling = False
        used_ollama_fallback = False

        with self._make_lease() as lease:
            # --- Conditional Phase A: Ollama gemma4:12b ONLY if image/audio attachments exist ---
            mm_threads = [t for t in threads if t.has_multimodal_attachments]
            if mm_threads:
                logger.info(
                    "Phase A: %d thread(s) have image/audio attachments -> running %s multimodal pre-pass.",
                    len(mm_threads),
                    self.settings.model_multimodal,
                )
                lease.acquire_ollama()
                used_ollama_mm = True
                for thread in mm_threads:
                    findings = self._run_multimodal_prepass(thread, call_stats)
                    if findings:
                        mm_findings_by_tid[thread.thread_id] = findings
                self.router.unload_ollama_model(self.settings.model_multimodal)

            # --- Primary Phase B: Inkling-2-Mini (or Ollama Fallback if unavailable) ---
            use_inkling_now = allow_inkling and self.settings.inkling_enabled and lease.acquire_inkling()
            if use_inkling_now:
                used_inkling = True
                logger.info("Phase B: Running Stages 1-3 on Inkling-2-Mini (%s).", self.settings.model_inkling)
                triaged = self._run_inkling_stages_1_and_2(threads, mm_findings_by_tid, call_stats)
                briefing = self._run_stage3_synthesis(
                    triaged=triaged,
                    backend="inkling",
                    model=self.settings.model_inkling,
                    call_stats=call_stats,
                )
            else:
                used_ollama_fallback = True
                logger.info("Phase B Fallback: Running Stages 1-3 on Ollama local models.")
                lease.acquire_ollama()
                triaged = self._run_ollama_fallback_stages_1_and_2(
                    threads, mm_findings_by_tid, call_stats
                )
                briefing = self._run_stage3_synthesis(
                    triaged=triaged,
                    backend="ollama",
                    model=self.settings.model_fallback_synthesis,
                    call_stats=call_stats,
                )
                self.router.unload_ollama_model(self.settings.model_fallback_synthesis)

        # GPU lease exited & restored before Phase C cloud network calls!
        cloud_count = 0
        blocked_count = 0
        if allow_cloud and self.settings.cloud_escalation_enabled and self.router.cloud_client is not None:
            cloud_count, blocked_count = self._run_phase_c_cloud_escalation(
                triaged=triaged,
                briefing=briefing,
                call_stats=call_stats,
            )

        return TriageRunReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            timezone=self.settings.timezone,
            total_threads_fetched=len(threads),
            triaged_threads=triaged,
            briefing=briefing,
            call_stats=call_stats,
            used_inkling=used_inkling,
            used_ollama_multimodal=used_ollama_mm,
            used_ollama_fallback=used_ollama_fallback,
            cloud_escalations_count=cloud_count,
            cloud_blocked_by_denylist=blocked_count,
        )

    def _run_multimodal_prepass(
        self,
        thread: EmailThread,
        call_stats: list[ModelCallStats],
    ) -> list[MultimodalFinding]:
        findings: list[MultimodalFinding] = []
        for att in thread.attachments:
            if att.kind not in (AttachmentKind.IMAGE, AttachmentKind.AUDIO) or not att.data_bytes:
                continue
            prompt_text = (
                f"Email Subject: {thread.subject}\n"
                f"From: {thread.sender} <{thread.sender_email}>\n"
                f"Attachment filename: {att.filename} ({att.mime_type})\n"
                "Analyze this attachment and extract key facts."
            )
            media_part = types.Part.from_bytes(data=att.data_bytes, mime_type=att.mime_type)
            # Gemma 4 audio is more stable with think=False on raw audio transcription
            think_mode = False if att.kind == AttachmentKind.AUDIO else True
            try:
                finding, stats = self.router.generate_structured(
                    backend="ollama",
                    model=self.settings.model_multimodal,
                    stage="phase_a_multimodal",
                    system_prompt=prompts.MULTIMODAL_SYSTEM_PROMPT,
                    contents=[prompt_text, media_part],
                    response_schema=MultimodalFinding,
                    temperature=0.1,
                    max_output_tokens=1024,
                    think=think_mode,
                    num_ctx=16384,
                )
                call_stats.append(stats)
                findings.append(finding)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Multimodal pre-pass failed for %s/%s: %s", thread.thread_id, att.filename, exc)
        return findings

    def _run_inkling_stages_1_and_2(
        self,
        threads: list[EmailThread],
        mm_findings: dict[str, list[MultimodalFinding]],
        call_stats: list[ModelCallStats],
    ) -> list[TriagedThread]:
        results: list[TriagedThread] = []

        # Stage 1: Bulk triage at reasoning_effort="none" (KV prefix cache hit across all threads)
        for thread in threads:
            findings = mm_findings.get(thread.thread_id, [])
            item = self._classify_single_thread(
                thread=thread,
                findings=findings,
                backend="inkling",
                model=self.settings.model_inkling,
                call_stats=call_stats,
                reasoning_effort="none",
            )
            results.append(item)

        # Stage 2: Deep extraction at reasoning_effort="low" for non-noise actionable/tech threads
        for item in results:
            if item.category in NOISE_CATEGORIES:
                continue
            if item.category == TriageCategory.TECHNICAL_ALERT or (
                item.stage1 and item.stage1.needs_tech_extract
            ):
                self._extract_tech_alert(
                    item=item,
                    backend="inkling",
                    model=self.settings.model_inkling,
                    call_stats=call_stats,
                    reasoning_effort="low",
                )
            if item.category in HIGH_PRIORITY_CATEGORIES or (
                item.stage1 and item.stage1.needs_deep_analysis
            ):
                self._analyze_deep_thread(
                    item=item,
                    backend="inkling",
                    model=self.settings.model_inkling,
                    call_stats=call_stats,
                    reasoning_effort="low",
                )

        return results

    def _run_ollama_fallback_stages_1_and_2(
        self,
        threads: list[EmailThread],
        mm_findings: dict[str, list[MultimodalFinding]],
        call_stats: list[ModelCallStats],
    ) -> list[TriagedThread]:
        results: list[TriagedThread] = []

        # Stage 1 Fallback: gemma4:e2b (fast bulk triage, think=False)
        for thread in threads:
            findings = mm_findings.get(thread.thread_id, [])
            item = self._classify_single_thread(
                thread=thread,
                findings=findings,
                backend="ollama",
                model=self.settings.model_fallback_triage,
                call_stats=call_stats,
                think=False,
                num_ctx=8192,
            )
            results.append(item)
        self.router.unload_ollama_model(self.settings.model_fallback_triage)

        # Stage 2A Fallback: qwen2.5-coder:3b (tech/CI extraction)
        tech_items = [
            item
            for item in results
            if item.category not in NOISE_CATEGORIES
            and (
                item.category == TriageCategory.TECHNICAL_ALERT
                or (item.stage1 and item.stage1.needs_tech_extract)
            )
        ]
        if tech_items:
            for item in tech_items:
                self._extract_tech_alert(
                    item=item,
                    backend="ollama",
                    model=self.settings.model_fallback_coder,
                    call_stats=call_stats,
                    num_ctx=8192,
                )
            self.router.unload_ollama_model(self.settings.model_fallback_coder)

        # Stage 2B Fallback: gemma4:12b (deep urgent/action analysis)
        deep_items = [
            item
            for item in results
            if item.category not in NOISE_CATEGORIES
            and (
                item.category in HIGH_PRIORITY_CATEGORIES
                or (item.stage1 and item.stage1.needs_deep_analysis)
            )
        ]
        if deep_items:
            for item in deep_items:
                self._analyze_deep_thread(
                    item=item,
                    backend="ollama",
                    model=self.settings.model_multimodal,
                    call_stats=call_stats,
                    think=True,
                    num_ctx=16384,
                )
            self.router.unload_ollama_model(self.settings.model_multimodal)

        return results

    def _classify_single_thread(
        self,
        *,
        thread: EmailThread,
        findings: list[MultimodalFinding],
        backend: Literal["inkling", "ollama"],
        model: str,
        call_stats: list[ModelCallStats],
        reasoning_effort: str | None = None,
        think: bool | str | None = None,
        num_ctx: int | None = None,
    ) -> TriagedThread:
        prompt = _format_thread_for_prompt(thread, findings, max_body_chars=6000)
        try:
            s1, stats = self.router.generate_structured(
                backend=backend,
                model=model,
                stage="stage1_triage",
                system_prompt=prompts.STAGE1_TRIAGE_SYSTEM_PROMPT,
                contents=prompt,
                response_schema=Stage1Classification,
                temperature=0.0,
                max_output_tokens=512,
                reasoning_effort=reasoning_effort,
                think=think,
                num_ctx=num_ctx,
            )
            call_stats.append(stats)
            return TriagedThread(
                thread=thread,
                category=s1.category,
                confidence=s1.confidence,
                summary=s1.one_line_summary,
                stage1=s1,
                multimodal_findings=findings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Stage 1 failed for thread %s: %s", thread.thread_id, exc)
            return TriagedThread(
                thread=thread,
                category=TriageCategory.FYI,
                confidence=0.0,
                summary=thread.snippet or thread.subject,
                multimodal_findings=findings,
                error_note=f"Stage 1 classification failed: {exc}",
            )

    def _extract_tech_alert(
        self,
        *,
        item: TriagedThread,
        backend: Literal["inkling", "ollama"],
        model: str,
        call_stats: list[ModelCallStats],
        reasoning_effort: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        prompt = _format_thread_for_prompt(item.thread, item.multimodal_findings, max_body_chars=10000)
        try:
            tech, stats = self.router.generate_structured(
                backend=backend,
                model=model,
                stage="stage2_tech_extract",
                system_prompt=prompts.STAGE2_TECH_SYSTEM_PROMPT,
                contents=prompt,
                response_schema=TechExtraction,
                temperature=0.0,
                max_output_tokens=768,
                reasoning_effort=reasoning_effort,
                num_ctx=num_ctx,
            )
            call_stats.append(stats)
            item.tech_extraction = tech
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 tech extraction failed for %s: %s", item.thread.thread_id, exc)

    def _analyze_deep_thread(
        self,
        *,
        item: TriagedThread,
        backend: Literal["inkling", "ollama"],
        model: str,
        call_stats: list[ModelCallStats],
        reasoning_effort: str | None = None,
        think: bool | str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        prompt = _format_thread_for_prompt(item.thread, item.multimodal_findings, max_body_chars=14000)
        try:
            deep, stats = self.router.generate_structured(
                backend=backend,
                model=model,
                stage="stage2_deep_analysis",
                system_prompt=prompts.STAGE2_DEEP_SYSTEM_PROMPT,
                contents=prompt,
                response_schema=DeepThreadAnalysis,
                temperature=0.1,
                max_output_tokens=1024,
                reasoning_effort=reasoning_effort,
                think=think,
                num_ctx=num_ctx,
            )
            call_stats.append(stats)
            item.deep_analysis = deep
            item.category = deep.refined_category
            item.confidence = deep.confidence
            item.summary = deep.summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 deep analysis failed for %s: %s", item.thread.thread_id, exc)

    def _run_stage3_synthesis(
        self,
        *,
        triaged: list[TriagedThread],
        backend: Literal["inkling", "ollama"],
        model: str,
        call_stats: list[ModelCallStats],
    ) -> ExecutiveBriefing | None:
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
                if item.deep_analysis.escalation_recommended:
                    card += f"\n  local_escalation_flag: {item.deep_analysis.escalation_reason}"
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
                backend=backend,
                model=model,
                stage="stage3_executive_synthesis",
                system_prompt=prompts.STAGE3_EXECUTIVE_SYSTEM_PROMPT,
                contents=synthesis_prompt,
                response_schema=ExecutiveBriefing,
                temperature=0.2,
                max_output_tokens=1536,
                reasoning_effort="medium" if backend == "inkling" else None,
                think="medium" if backend == "ollama" else None,
                num_ctx=32768 if backend == "ollama" else None,
            )
            call_stats.append(stats)
            return briefing
        except Exception as exc:  # noqa: BLE001
            logger.error("Stage 3 executive synthesis failed: %s", exc)
            return ExecutiveBriefing(
                headline=f"{len(relevant)} substantive email(s) triaged this morning.",
                executive_summary="Local synthesis encountered an error; see itemized threads below.",
            )

    def _run_phase_c_cloud_escalation(
        self,
        *,
        triaged: list[TriagedThread],
        briefing: ExecutiveBriefing | None,
        call_stats: list[ModelCallStats],
    ) -> tuple[int, int]:
        denylist = self.settings.denylist_domain_set
        flagged_by_briefing = set(briefing.cloud_escalation_thread_ids) if briefing else set()

        candidates: list[tuple[TriagedThread, str, str]] = []
        for item in triaged:
            if item.category in NOISE_CATEGORIES:
                continue
            is_oversized = item.thread.total_char_length > self.settings.inkling_overflow_chars
            is_deep_escalation = (
                item.thread.thread_id in flagged_by_briefing
                or (item.deep_analysis is not None and item.deep_analysis.escalation_recommended)
                or (
                    item.category in HIGH_PRIORITY_CATEGORIES
                    and item.confidence < self.settings.cloud_confidence_threshold
                )
            )
            if is_deep_escalation:
                candidates.append((item, self.settings.model_cloud_deep, "high_stakes_or_low_conf"))
            elif is_oversized:
                candidates.append((item, self.settings.model_cloud_fast, "oversized_context"))

        escalated = 0
        blocked = 0
        for item, cloud_model, reason in candidates:
            if item.thread.sender_domain in denylist:
                blocked += 1
                logger.info(
                    "Cloud escalation blocked by denylist for thread %s (%s).",
                    item.thread.thread_id,
                    item.thread.sender_domain,
                )
                continue
            if escalated >= self.settings.max_cloud_escalations_per_run:
                logger.info(
                    "Reached MAX_CLOUD_ESCALATIONS_PER_RUN (%d); skipping remaining candidates.",
                    self.settings.max_cloud_escalations_per_run,
                )
                break

            prompt = (
                f"Escalation trigger: {reason}\n"
                f"Local initial category: {item.category.value} (confidence={item.confidence:.2f})\n\n"
                + _format_thread_for_prompt(item.thread, item.multimodal_findings, max_body_chars=120000)
            )
            try:
                cloud_res, stats = self.router.generate_structured(
                    backend="cloud",
                    model=cloud_model,
                    stage=f"stage4_cloud_{reason}",
                    system_prompt=prompts.STAGE4_CLOUD_ESCALATION_SYSTEM_PROMPT,
                    contents=prompt,
                    response_schema=CloudEscalationResult,
                    temperature=0.1,
                    max_output_tokens=8192,
                )
                call_stats.append(stats)
                item.cloud_escalation = cloud_res
                item.cloud_model_used = cloud_model
                item.category = cloud_res.refined_category
                item.summary = cloud_res.executive_summary
                item.confidence = max(item.confidence, 0.95)
                escalated += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Phase C cloud escalation (%s) failed for %s: %s",
                    cloud_model,
                    item.thread.thread_id,
                    exc,
                )

        return escalated, blocked
