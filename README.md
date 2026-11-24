# Gmail Inbox Triage Service

A read-only, scheduled morning Gmail inbox triage service that runs daily at **5:30 AM Pacific Time (`America/Los_Angeles`)** and posts an actionable **Slack Block Kit** digest via an Incoming Webhook.

All LLM calls go through the official **`google-genai` Python SDK** (`from google import genai`) across three backends:
1. **Conditional Phase A — Ollama (`gemma4:12b`)**: Runs *only* when unread emails contain image or audio attachments (extracting visual descriptions & 16 kHz WAV audio transcripts). Skipped completely on text-only mornings (**zero GPU switches**).
2. **Primary Phase B — Inkling-2-Mini (`inkling_engine` on `127.0.0.1:8001`)**: Managed via `/home/ludo/.local/bin/inkling-ctl` for exclusive GTX 1080 Ti VRAM ownership. Runs Stages 1–3 in one hot pass using dynamic `reasoning_effort` (`none` → `low` → `medium`) with KV prefix reuse. Automatically falls back to Ollama (`gemma4:e2b` → `qwen2.5-coder:3b` → `bonsai-2:27b`) if Inkling is unavailable, and always restores `inkling-server.service` to its exact pre-run state without touching `~/.hermes/config.yaml`.
3. **Selective Phase C — Cloud Gemini 4 (`gemini-4-carbon` & `gemini-4-neon`)**: Invoked only for oversized threads (>28K tokens → `gemini-4-carbon`) or high-stakes/low-confidence escalations (`gemini-4-neon`), gated by `CLOUD_DENYLIST_DOMAINS` and `MAX_CLOUD_ESCALATIONS_PER_RUN`.

## Setup

### 1. Install Dependencies
```bash
cd /home/ludo/gmailInboxTriage
uv sync
```

### 2. Configure `.env`
```bash
cp .env.example .env
```
Fill in:
- `SLACK_WEBHOOK_URL`: Create an Incoming Webhook in your Slack workspace (`https://api.slack.com/apps` → Incoming Webhooks).
- `GEMINI_API_KEY`: Your Google AI Studio API key (for optional Phase C escalations to `gemini-4-carbon` / `gemini-4-neon`).
- `CLOUD_DENYLIST_DOMAINS`: Optional comma-separated domains (e.g., `bank.com,legal.org`) that must never leave localhost.

### 3. Authorize Read-Only Gmail Access (`gmail.readonly`)
1. In Google Cloud Console, enable the **Gmail API** and create an **OAuth 2.0 Client ID (Desktop app)**.
2. Download the JSON file to `/home/ludo/gmailInboxTriage/credentials.json`.
3. Run the one-time authorization flow:
```bash
uv run triage auth
```

### 4. Verify Health & Dry-Run
```bash
uv run triage check-health
uv run triage run --dry-run --since-hours 24
```

### 5. Enable the 5:30 AM Pacific `systemd --user` Timer
```bash
uv run triage install-timer
systemctl --user list-timers gmail-triage.timer
```
To ensure the 5:30 AM timer runs even when you are not actively logged into a desktop session:
```bash
loginctl enable-linger "$USER"
```
View logs from scheduled runs:
```bash
journalctl --user -u gmail-triage.service -n 100
```
