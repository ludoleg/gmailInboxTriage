"""Typer CLI entry point for serverless GCP Gmail Inbox Triage & interactive Slack bot."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from inbox_triage.config import Settings
from inbox_triage.gmail.auth import get_gmail_credentials, run_oauth_flow
from inbox_triage.gmail.client import GmailClient
from inbox_triage.llm.router import GenAIRouter
from inbox_triage.pipeline.orchestrator import TriageOrchestrator
from inbox_triage.slack.client import SlackBotClient, SlackWebhookClient
from inbox_triage.slack.formatter import build_slack_payloads
from inbox_triage.state import create_state_store

app = typer.Typer(
    name="triage",
    help="Serverless GCP 5:30 AM Pacific Gmail Inbox Triage & Interactive Slack Bot.",
    no_args_is_help=True,
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@app.command()
def auth(
    port: int = typer.Option(8085, "--port", help="Localhost callback port for OAuth redirect."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not launch a local browser (default on headless hosts)."
    ),
) -> None:
    """Run one-time Google Workspace OAuth2 setup (gmail.modify, calendar.events, tasks)."""
    settings = Settings()
    console.print(f"Using client secrets: [cyan]{settings.gmail_credentials_path}[/cyan]")
    open_browser = False if no_browser else None
    run_oauth_flow(
        settings.gmail_credentials_path,
        settings.gmail_token_path,
        port=port,
        open_browser=open_browser,
    )
    console.print(f"[green]Authorized![/green] Token saved securely to {settings.gmail_token_path}")


@app.command("check-health")
def check_health(
    local_state: bool = typer.Option(False, "--local-state", help="Check local SQLite store instead of Firestore."),
) -> None:
    """Verify GenAI backend, state store, Workspace OAuth credentials, and Slack config."""
    settings = Settings()
    table = Table(title="Gmail Inbox Triage (GCP Phase 2) — Health Check")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # 1. GenAI Backend
    try:
        router = GenAIRouter(settings)
        mode_detail = (
            f"Vertex AI (project={settings.gcp_project_id}, location={settings.vertex_location})"
            if settings.use_vertexai
            else "Gemini API (GEMINI_API_KEY)"
        )
        models_detail = (
            f"{settings.model_stage1_fast} -> {settings.model_stage2_workhorse} -> {settings.model_stage3_reasoning}"
        )
        table.add_row("Google GenAI", "[green]OK[/green]", f"{router.backend_name} | {mode_detail} | {models_detail}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Google GenAI", "[red]ERROR[/red]", str(exc))

    # 2. State Store
    try:
        store = create_state_store(settings, force_sqlite=local_state)
        wm = store.get_watermark_epoch(default_lookback_hours=settings.default_lookback_hours)
        muted_n = len(store.get_muted_senders())
        backend_label = "sqlite" if (local_state or settings.state_backend == "sqlite") else "firestore"
        table.add_row("State Store", "[green]OK[/green]", f"backend={backend_label} | watermark={wm} | muted={muted_n}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("State Store", "[red]ERROR[/red]", str(exc))

    # 3. Google Workspace OAuth
    try:
        creds = get_gmail_credentials(settings.gmail_token_path, token_json=settings.gmail_token_json)
        src = "GMAIL_TOKEN_JSON env" if settings.gmail_token_json else str(settings.gmail_token_path)
        scopes = ", ".join(creds.scopes or []) or "default"
        table.add_row("Workspace OAuth", "[green]OK[/green]", f"Valid ({src}) | scopes: {scopes}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Workspace OAuth", "[red]NOT READY[/red]", str(exc))

    # 4. Slack Delivery
    if settings.has_slack_bot:
        sig_note = "signing_secret=set" if settings.slack_signing_secret else "signing_secret=MISSING"
        table.add_row("Slack Bot", "[green]INTERACTIVE[/green]", f"channel={settings.slack_channel_id} | {sig_note}")
    elif settings.slack_webhook_url:
        table.add_row("Slack Webhook", "[yellow]FALLBACK[/yellow]", "Push-only webhook (set SLACK_BOT_TOKEN + SLACK_CHANNEL_ID for buttons)")
    else:
        table.add_row("Slack Delivery", "[red]MISSING[/red]", "Set SLACK_BOT_TOKEN+SLACK_CHANNEL_ID or SLACK_WEBHOOK_URL")

    console.print(table)


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview digest locally without posting to Slack or saving watermark."),
    since_hours: int | None = typer.Option(None, "--since-hours", help="Override lookback window in hours."),
    ignore_watermark: bool = typer.Option(False, "--ignore-watermark", help="Re-triage even if thread was previously processed."),
    local_state: bool = typer.Option(False, "--local-state", help="Use local SQLite instead of Firestore."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Fetch unread Gmail threads, filter muted senders, triage across 3-tier Gemini, and post to Slack."""
    _configure_logging(verbose)
    settings = Settings()
    store = create_state_store(settings, force_sqlite=local_state)
    gmail = GmailClient(settings)
    router = GenAIRouter(settings)
    orchestrator = TriageOrchestrator(settings=settings, router=router)

    if since_hours is not None:
        after_epoch = int((datetime.now(UTC) - timedelta(hours=since_hours)).timestamp())
    else:
        after_epoch = store.get_watermark_epoch(default_lookback_hours=settings.default_lookback_hours)

    fetched = gmail.fetch_unread_threads(after_epoch=after_epoch)
    candidates = (
        fetched
        if ignore_watermark
        else [t for t in fetched if not store.is_already_processed(t.thread_id, t.last_message_id)]
    )

    muted_senders = store.get_muted_senders()
    threads = [t for t in candidates if t.sender_email.lower() not in muted_senders]
    muted_skipped = len(candidates) - len(threads)

    console.print(
        f"Fetched [bold]{len(fetched)}[/bold] unread thread(s); "
        f"[bold]{len(threads)}[/bold] to triage ([dim]{muted_skipped} muted skipped[/dim])."
    )

    report = orchestrator.run(threads)
    report.muted_skipped_count = muted_skipped

    interactive = settings.has_slack_bot
    payloads = build_slack_payloads(
        report,
        mention_on_urgent=settings.slack_mention_on_urgent,
        interactive=interactive,
    )

    if dry_run:
        console.print("\n[bold yellow]DRY RUN — Slack Payload Preview:[/bold yellow]")
        console.print_json(json.dumps(payloads, indent=2))
        return

    if settings.has_slack_bot:
        assert settings.slack_bot_token and settings.slack_channel_id
        SlackBotClient(settings.slack_bot_token).post_payloads(settings.slack_channel_id, payloads)
    elif settings.slack_webhook_url:
        SlackWebhookClient(settings.slack_webhook_url).post_payloads(payloads)
    else:
        raise typer.BadParameter(
            "Configure SLACK_BOT_TOKEN + SLACK_CHANNEL_ID (or SLACK_WEBHOOK_URL), or use --dry-run."
        )

    store.record_run(report)
    console.print(
        f"[green]Done![/green] Sent {len(payloads)} Slack message(s) "
        f"({'interactive bot' if interactive else 'webhook'}) and saved run {report.run_id}."
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address for Cloud Run Service."),
    port: int = typer.Option(8080, "--port", help="HTTP port for Cloud Run Service."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Start the FastAPI server handling Slack button interactions and /triage slash commands."""
    import uvicorn

    from inbox_triage.server import create_app

    _configure_logging(verbose)
    uvicorn.run(create_app(), host=host, port=port, log_level="debug" if verbose else "info")


@app.command()
def muted(
    local_state: bool = typer.Option(False, "--local-state", help="Inspect local SQLite instead of Firestore."),
) -> None:
    """List senders muted via Slack or CLI."""
    settings = Settings()
    store = create_state_store(settings, force_sqlite=local_state)
    items = store.list_muted_senders()
    if not items:
        console.print("No muted senders.")
        return
    for item in items:
        console.print(f"- [cyan]{item.sender_email}[/cyan] ({item.muted_at.isoformat()})")


@app.command()
def unmute(
    sender_email: str = typer.Argument(..., help="Email address to remove from the mute list."),
    local_state: bool = typer.Option(False, "--local-state", help="Use local SQLite instead of Firestore."),
) -> None:
    """Remove a sender from the muted-senders list."""
    settings = Settings()
    store = create_state_store(settings, force_sqlite=local_state)
    if store.unmute_sender(sender_email):
        console.print(f"[green]Unmuted[/green] {sender_email}")
    else:
        console.print(f"[yellow]{sender_email} was not muted.[/yellow]")


if __name__ == "__main__":
    app()
