# Gmail Inbox Triage Service (GCP Serverless)

A read-only, serverless Gmail inbox triage service running on **Google Cloud Platform** (**Cloud Run Jobs** + **Cloud Scheduler** at **5:30 AM Pacific Time / `America/Los_Angeles`**) and posting an actionable **Slack Block Kit** digest via an Incoming Webhook.

*(Looking for the local GPU `barcalinux` edition with Inkling-2-Mini + Ollama? Check out the `barcalinux` branch or tag `v0.1.0-local`.)*

## Architecture (3-Tier Strength-Matched Gemini Funnel)

All model calls use the official **`google-genai` Python SDK** (defaulting to `GEMINI_API_KEY` via Secret Manager, with `USE_VERTEXAI=true` supported as a 1-line toggle for keyless Vertex AI IAM):

1. **Stage 0 (Gmail API + Firestore)**: Fetches unread inbox threads since the last Firestore watermark, deduplicates by `(thread_id, last_message_id)`, cleans HTML, and keeps raw image/audio/PDF bytes for native Gemini multimodal input.
2. **Stage 1 (`gemini-3.5-flash-lite`, concurrent)**: Ultra-fast bulk classification (`URGENT`, `ACTION_REQUIRED`, `TECHNICAL_ALERT`, `FYI`, `NEWSLETTER`, `PROMO_NOISE`); drops noise early.
3. **Stage 2 (`gemini-4-carbon`, concurrent, native multimodal)**: Extracts structured CI/PR/monitoring alerts and performs deep action-item, deadline, reply-point, and native attachment analysis (images, audio, and PDFs via `types.Part.from_bytes`).
4. **Stage 3 (`gemini-4-neon`, `max_output_tokens=8192`)**: Synthesizes the cross-thread morning `ExecutiveBriefing` and runs authoritative `HighStakesReview` on any flagged or low-confidence threads.

## Quickstart (Local Dev & Testing)

```bash
uv sync
cp .env.example .env
# Fill in GEMINI_API_KEY, SLACK_WEBHOOK_URL, and run one-time OAuth:
uv run triage auth

# Verify health & dry-run using local SQLite state:
uv run triage check-health --local-state
uv run triage run --dry-run --local-state --since-hours 24
```

## Deploy to GCP (`gmail-inbox-triage-515705`)

1. Ensure `gcloud` is authenticated (`gcloud auth login`) and billing is linked to `gmail-inbox-triage-515705`.
2. Export your secrets (or ensure they exist in Secret Manager) and run the idempotent deploy script:
   ```bash
   chmod +x deploy/gcp/deploy.sh
   source .env
   ./deploy/gcp/deploy.sh
   ```
   *(To use keyless Vertex AI IAM instead of `GEMINI_API_KEY`, run `USE_VERTEXAI=true ./deploy/gcp/deploy.sh`.)*
3. Trigger an immediate test execution in Cloud Run:
   ```bash
   gcloud run jobs execute gmail-inbox-triage --region us-west1 --wait
   ```
4. Once confirmed, disable the old local `barcalinux` timer so you don't receive duplicate digests:
   ```bash
   systemctl --user disable --now gmail-triage.timer
   ```

## GitHub Actions Auto-Deploy

`.github/workflows/deploy.yml` runs `ruff` and `pytest` on every push/PR. To enable Heroku-style auto-deploy on push to `main`, set repository variable `ENABLE_GCP_DEPLOY=true` and secrets `GCP_WORKLOAD_IDENTITY_PROVIDER` and `GCP_SERVICE_ACCOUNT`.
