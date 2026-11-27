# Gmail Inbox Triage Service (GCP Serverless · Phase 2 Interactive Bot)

A serverless Gmail inbox triage & interactive Slack assistant on **Google Cloud Platform**:
- **5:30 AM Pacific Scheduled Triage (`Cloud Run Job` + `Cloud Scheduler`)**: Triages unread Gmail threads through a concurrent 3-tier `google-genai` funnel (**`gemini-3.5-flash-lite` → `gemini-4-carbon` → `gemini-4-neon`**) and posts interactive Block Kit cards to Slack.
- **1-Click Slack Actions (`Cloud Run Service` + `Cloud Tasks` + `Firestore`)**:
  - **✅ Archive**: Removes `INBOX` + `UNREAD` in Gmail and updates the Slack card in place.
  - **📝 Draft Reply**: Uses `gemini-4-neon` to save a ready-to-send reply in **Gmail Drafts** (never auto-sends) and posts a preview + Gmail link in the Slack thread.
  - **📅 Task / Event**: Uses `gemini-4-carbon` to create a **Google Calendar Event** or **Google Task** and replies in the thread with a direct link.
  - **🔇 Mute Sender**: Adds the sender to Firestore `triage_muted_senders` so future runs filter them with zero LLM tokens.
- **On-Demand `/triage` Slash Command**: `/triage`, `/triage 48h`, `/triage muted`, `/triage unmute <email>`.

*(Looking for the local GPU `barcalinux` edition with Inkling-2-Mini + Ollama? Check out branch `barcalinux` or tag `v0.1.0-local`.)*

## Setup & Deploy (Phase 2)

### 1. Re-Authorize Google Workspace OAuth (`gmail.modify`, `calendar.events`, `tasks`)
```bash
cd /home/ludo/projects/gmailInboxTriage
uv sync
uv run triage auth
```

### 2. Configure Your Slack App (https://api.slack.com/apps)
1. **OAuth & Permissions → Bot Token Scopes**: add `chat:write` and `commands`, install/reinstall to workspace, copy the **Bot User OAuth Token** (`xoxb-...`), and invite the bot to your channel (`/invite @Gmail Inbox Triage`).
2. **Basic Information → App Credentials**: copy **Signing Secret**.
3. Put `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`, and `SLACK_SIGNING_SECRET` (plus optional `SLACK_ALLOWED_USER_IDS`) in `.env`.

### 3. Deploy to GCP (`gmail-inbox-triage-515705`)
```bash
export SLACK_BOT_TOKEN=... SLACK_CHANNEL_ID=... SLACK_SIGNING_SECRET=... GEMINI_API_KEY=...
./deploy/gcp/deploy.sh
```
The script deploys **both** the `gmail-inbox-triage` Cloud Run Job and the `gmail-inbox-triage-bot` Cloud Run Service, and prints the exact URLs to paste into your Slack App:
- **Interactivity & Shortcuts → Request URL**: `https://<service-url>/slack/interactions`
- **Slash Commands → `/triage` Request URL**: `https://<service-url>/slack/commands`

### 4. Local Development & Testing
```bash
uv run triage check-health --local-state
uv run triage run --dry-run --local-state --since-hours 48
uv run triage serve --port 8080
uv run ruff check src tests && uv run pytest -v
```
