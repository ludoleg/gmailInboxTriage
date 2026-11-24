"""Typer CLI entry point for Gmail Inbox Triage."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from inbox_triage.config import Settings
from inbox_triage.gmail.auth import get_gmail_credentials, run_oauth_flow
from inbox_triage.gmail.client import GmailClient
from inbox_triage.llm.gpu_lease import GpuLeaseManager
from inbox_triage.llm.router import ModelRouter
from inbox_triage.pipeline.orchestrator import TriageOrchestrator
from inbox_triage.slack.client import SlackWebhookClient
from inbox_triage.slack.formatter import build_slack_payloads
from inbox_triage.state import StateStore

app = typer.Typer(
    name="triage",
    help="5:30 AM Pacific Gmail Inbox Triage powered by google-genai, Inkling-2-Mini, Ollama, and Gemini 4.",
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
    """Run one-time Gmail OAuth2 setup (gmail.readonly scope)."""
    settings = Settings()
    console.print(f"Using client secrets: [cyan]{settings.gmail_credentials_path}[/cyan]")
    open_browser = False if no_browser else None
    console.print(
        f"[yellow]Headless tip:[/yellow] If running over SSH, forward port {port} from your laptop:\n"
        f"  [bold]ssh -L {port}:localhost:{port} <user>@<host>[/bold]\n"
        "Then open the Google authorization URL printed below in your laptop's browser."
    )
    run_oauth_flow(
        settings.gmail_credentials_path,
        settings.gmail_token_path,
        port=port,
        open_browser=open_browser,
    )
    console.print(f"[green]Authorized![/green] Token saved securely to {settings.gmail_token_path}")


@app.command("check-health")
def check_health() -> None:
    """Verify Ollama models, inkling-ctl / inkling-server, Gmail token, Slack webhook, and Gemini API."""
    settings = Settings()
    router = ModelRouter(settings)
    lease = GpuLeaseManager(
        inkling_ctl_path=settings.inkling_ctl_path,
        inkling_base_url=settings.inkling_base_url,
        restore_policy=settings.gpu_restore_policy,
    )

    table = Table(title="Gmail Inbox Triage — Health Check")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # 1. inkling-ctl & server
    ctl_ok = settings.inkling_ctl_path.exists()
    active = lease.is_inkling_service_active()
    healthy = lease.is_inkling_healthy(timeout=1.5)
    table.add_row(
        "Inkling-2-Mini",
        "[green]OK[/green]" if ctl_ok else "[yellow]MISSING CTL[/yellow]",
        f"ctl={settings.inkling_ctl_path} | service={'active' if active else 'inactive'} | health={'up' if healthy else 'down'}",
    )

    # 2. Ollama models
    try:
        installed = set(router.list_ollama_models())
        expected = [
            settings.model_multimodal,
            settings.model_fallback_triage,
            settings.model_fallback_coder,
            settings.model_fallback_synthesis,
        ]
        missing = [m for m in expected if m not in installed]
        status = "[green]OK[/green]" if not missing else "[yellow]PARTIAL[/yellow]"
        detail = (
            f"Installed: {', '.join(sorted(installed))}"
            + (f" | Missing: {', '.join(missing)}" if missing else "")
        )
        table.add_row("Ollama Local", status, detail)
    except httpx.HTTPError as exc:
        table.add_row("Ollama Local", "[red]DOWN[/red]", str(exc))

    # 3. Cloud Gemini 4
    if settings.gemini_api_key:
        table.add_row(
            "Cloud Gemini 4",
            "[green]CONFIGURED[/green]",
            f"fast={settings.model_cloud_fast} | deep={settings.model_cloud_deep} | enabled={settings.cloud_escalation_enabled}",
        )
    else:
        table.add_row("Cloud Gemini 4", "[yellow]NO KEY[/yellow]", "GEMINI_API_KEY not set (local-only mode)")

    # 4. Gmail OAuth token
    try:
        get_gmail_credentials(settings.gmail_token_path)
        table.add_row("Gmail OAuth", "[green]OK[/green]", f"Token valid at {settings.gmail_token_path}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Gmail OAuth", "[red]NOT READY[/red]", str(exc))

    # 5. Slack Webhook
    if settings.slack_webhook_url:
        table.add_row("Slack Webhook", "[green]CONFIGURED[/green]", "SLACK_WEBHOOK_URL set")
    else:
        table.add_row("Slack Webhook", "[yellow]MISSING[/yellow]", "Set SLACK_WEBHOOK_URL in .env")

    console.print(table)


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview digest locally without posting to Slack or saving watermark."),
    since_hours: int | None = typer.Option(None, "--since-hours", help="Override lookback window in hours."),
    no_cloud: bool = typer.Option(False, "--no-cloud", help="Disable Phase C Cloud Gemini 4 escalations."),
    no_inkling: bool = typer.Option(False, "--no-inkling", help="Skip Inkling-2-Mini and force Ollama fallback."),
    ignore_watermark: bool = typer.Option(False, "--ignore-watermark", help="Re-triage even if thread was previously processed."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Fetch unread Gmail threads, triage across local/cloud models, and post Slack digest."""
    _configure_logging(verbose)
    settings = Settings()
    store = StateStore(settings.state_db_path)
    gmail = GmailClient(settings)
    router = ModelRouter(settings)
    orchestrator = TriageOrchestrator(settings=settings, router=router)

    if since_hours is not None:
        after_epoch = int((datetime.now(UTC) - timedelta(hours=since_hours)).timestamp())
    else:
        after_epoch = store.get_watermark_epoch(default_lookback_hours=settings.default_lookback_hours)

    fetched = gmail.fetch_unread_threads(after_epoch=after_epoch)
    if ignore_watermark:
        threads = fetched
    else:
        threads = [t for t in fetched if not store.is_already_processed(t.thread_id, t.last_message_id)]

    console.print(
        f"Fetched [bold]{len(fetched)}[/bold] unread thread(s); "
        f"[bold]{len(threads)}[/bold] new/updated to triage."
    )

    report = orchestrator.run(
        threads,
        allow_cloud=not no_cloud,
        allow_inkling=not no_inkling,
    )
    payloads = build_slack_payloads(report, mention_on_urgent=settings.slack_mention_on_urgent)

    if dry_run:
        console.print("\n[bold yellow]DRY RUN — Slack Payload Preview:[/bold yellow]")
        console.print_json(json.dumps(payloads, indent=2))
        return

    if not settings.slack_webhook_url:
        raise typer.BadParameter("SLACK_WEBHOOK_URL is not set in .env (use --dry-run to preview).")

    SlackWebhookClient(settings.slack_webhook_url).post_payloads(payloads)
    store.record_run(report)
    console.print(
        f"[green]Done![/green] Sent {len(payloads)} Slack message(s) and saved run {report.run_id}."
    )


@app.command("install-timer")
def install_timer(
    timezone: str = typer.Option("America/Los_Angeles", "--timezone", help="IANA timezone for 5:30 AM schedule."),
) -> None:
    """Install and enable the 5:30 AM systemd --user service and timer."""
    project_root = Path(__file__).resolve().parents[2]
    deploy_dir = project_root / "deploy"
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)

    service_src = (deploy_dir / "gmail-triage.service").read_text(encoding="utf-8")
    timer_src = (deploy_dir / "gmail-triage.timer").read_text(encoding="utf-8")
    timer_rendered = timer_src.replace("America/Los_Angeles", timezone)

    (unit_dir / "gmail-triage.service").write_text(service_src, encoding="utf-8")
    (unit_dir / "gmail-triage.timer").write_text(timer_rendered, encoding="utf-8")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "gmail-triage.timer"], check=True)
    console.print(f"[green]Installed and enabled gmail-triage.timer[/green] (05:30 {timezone})")
    console.print("Verify with: [cyan]systemctl --user list-timers gmail-triage.timer[/cyan]")


if __name__ == "__main__":
    app()
