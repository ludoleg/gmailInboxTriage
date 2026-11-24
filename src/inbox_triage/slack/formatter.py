"""Slack Block Kit formatter for the 5:30 AM Pacific morning inbox digest."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

from inbox_triage.models import TriageCategory, TriagedThread, TriageRunReport

MAX_BLOCKS_PER_MESSAGE = 50
MAX_SECTION_TEXT_CHARS = 2900
MAX_HEADER_CHARS = 150


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _escape_slack(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _section(mrkdwn: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": _truncate(mrkdwn, MAX_SECTION_TEXT_CHARS)}}


def _header(text: str) -> dict[str, Any]:
    return {"type": "header", "text": {"type": "plain_text", "text": _truncate(text, MAX_HEADER_CHARS), "emoji": True}}


def _context(mrkdwn: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": _truncate(mrkdwn, MAX_SECTION_TEXT_CHARS)}]}


def _format_priority_card(item: TriagedThread, tz: ZoneInfo, badge: str) -> str:
    ts = item.thread.received_at.astimezone(tz).strftime("%-I:%M %p %Z")
    subj = _escape_slack(item.thread.subject)
    sender = _escape_slack(item.thread.sender)
    review_tag = f"  `✨ {item.deep_review_model}`" if item.deep_review_model else ""

    lines = [
        f"{badge} *<{item.thread.gmail_url}|{subj}>*{review_tag}",
        f"From: *{sender}* · {ts}",
        _escape_slack(item.summary),
    ]

    if item.high_stakes_review:
        hr = item.high_stakes_review
        if hr.deadline:
            lines.append(f"• *Deadline:* {_escape_slack(hr.deadline)}")
        if hr.action_items:
            lines.append("• *Actions:* " + "; ".join(_escape_slack(a) for a in hr.action_items[:3]))
        if hr.risk_assessment:
            lines.append(f"• *Risk:* {_escape_slack(hr.risk_assessment)}")
        if hr.suggested_next_step:
            lines.append(f"• *Next step:* {_escape_slack(hr.suggested_next_step)}")
    elif item.deep_analysis:
        da = item.deep_analysis
        if da.deadline:
            lines.append(f"• *Deadline:* {_escape_slack(da.deadline)}")
        if da.action_items:
            lines.append("• *Actions:* " + "; ".join(_escape_slack(a) for a in da.action_items[:3]))
        if da.suggested_reply_points:
            lines.append("• *Reply points:* " + "; ".join(_escape_slack(r) for r in da.suggested_reply_points[:2]))
        if da.attachment_findings:
            lines.append("• *Attachment(s):* " + "; ".join(_escape_slack(f) for f in da.attachment_findings[:2]))

    return "\n".join(lines)


def build_slack_payloads(
    report: TriageRunReport,
    *,
    mention_on_urgent: str | None = None,
) -> list[dict[str, Any]]:
    """Build one or more Slack Block Kit webhook payloads (chunked to <= 50 blocks each)."""
    tz = ZoneInfo(report.timezone)
    local_time = report.completed_at.astimezone(tz)
    date_label = local_time.strftime("%a, %b %-d · %-I:%M %p %Z")

    urgent = report.by_category(TriageCategory.URGENT)
    action = report.by_category(TriageCategory.ACTION_REQUIRED)
    tech = report.by_category(TriageCategory.TECHNICAL_ALERT)
    fyi = report.by_category(TriageCategory.FYI)
    newsletters = report.by_category(TriageCategory.NEWSLETTER)
    promos = report.by_category(TriageCategory.PROMO_NOISE)

    blocks: list[dict[str, Any]] = []
    blocks.append(_header(f"📬 Morning Inbox Triage — {date_label}"))

    if report.total_threads_fetched == 0:
        blocks.append(_section("✨ *Inbox Zero!* No new unread emails since the last triage run."))
        blocks.append(_context("0 model calls · GCP Cloud Run"))
        return [{"text": f"Morning Inbox Triage ({date_label}): Inbox Zero", "blocks": blocks}]

    counts_line = (
        f"*🔴 {len(urgent)} Urgent*  ·  *🟠 {len(action)} Action*  ·  "
        f"*⚙️ {len(tech)} Tech Alerts*  ·  *ℹ️ {len(fyi)} FYI*  ·  "
        f"*🧹 {len(newsletters) + len(promos)} Filtered*"
    )
    if urgent and mention_on_urgent:
        counts_line = f"{mention_on_urgent}  {counts_line}"
    blocks.append(_section(counts_line))

    if report.briefing:
        brief_text = f"*{_escape_slack(report.briefing.headline)}*\n{_escape_slack(report.briefing.executive_summary)}"
        if report.briefing.cross_thread_connections:
            conns = "\n".join(f"• {_escape_slack(c)}" for c in report.briefing.cross_thread_connections[:3])
            brief_text += f"\n\n*Cross-thread notes:*\n{conns}"
        blocks.append(_section(brief_text))

    if urgent or action:
        blocks.append({"type": "divider"})
        for item in urgent:
            blocks.append(_section(_format_priority_card(item, tz, "🔴")))
        for item in action:
            blocks.append(_section(_format_priority_card(item, tz, "🟠")))

    if tech:
        blocks.append({"type": "divider"})
        tech_lines = ["*⚙️ Technical & Automated Alerts*"]
        for item in tech[:10]:
            subj = _escape_slack(item.thread.subject)
            if item.tech_extraction:
                te = item.tech_extraction
                sev = te.severity.upper()
                detail = _escape_slack(te.error_or_event)
                src = _escape_slack(te.source_system)
                tech_lines.append(f"• `[{sev}]` *{src}* — <{item.thread.gmail_url}|{subj}>: {detail}")
            else:
                tech_lines.append(f"• <{item.thread.gmail_url}|{subj}> — {_escape_slack(item.summary)}")
        blocks.append(_section("\n".join(tech_lines)))

    if fyi:
        blocks.append({"type": "divider"})
        fyi_lines = ["*ℹ️ FYI*"]
        for item in fyi[:8]:
            subj = _escape_slack(item.thread.subject)
            sender = _escape_slack(item.thread.sender)
            fyi_lines.append(f"• *{sender}* — <{item.thread.gmail_url}|{subj}>: {_escape_slack(item.summary)}")
        if len(fyi) > 8:
            fyi_lines.append(f"_…and {len(fyi) - 8} more FYI thread(s)._")
        blocks.append(_section("\n".join(fyi_lines)))

    if newsletters or promos:
        blocks.append(
            _context(
                f"🧹 Filtered silently: {len(newsletters)} newsletter(s), {len(promos)} promo/noise thread(s)."
            )
        )

    models_used = sorted({s.model for s in report.call_stats})
    elapsed_s = (report.completed_at - report.started_at).total_seconds()
    total_tokens = report.total_prompt_tokens + report.total_completion_tokens
    footer_parts = [
        f"Models: {', '.join(models_used) if models_used else 'none'}",
        f"Tokens: {total_tokens:,}",
        f"Deep reviews: {report.deep_reviews_count}",
        f"Runtime: {elapsed_s:.1f}s",
    ]
    blocks.append(_context(" · ".join(footer_parts)))

    fallback_text = (
        f"Morning Inbox Triage ({date_label}): "
        f"{len(urgent)} Urgent, {len(action)} Action, {len(tech)} Tech Alerts"
    )

    payloads: list[dict[str, Any]] = []
    for i in range(0, len(blocks), MAX_BLOCKS_PER_MESSAGE):
        chunk = blocks[i : i + MAX_BLOCKS_PER_MESSAGE]
        payloads.append({"text": fallback_text, "blocks": chunk})
    return payloads
