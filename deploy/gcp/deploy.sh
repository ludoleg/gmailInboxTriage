#!/usr/bin/env bash
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

# =============================================================================
# Idempotent GCP Serverless Deployment for Gmail Inbox Triage (Phase 2)
# Deploys: Artifact Registry, Secret Manager, Firestore, Cloud Tasks Queue,
#          Cloud Run Job (5:30 AM PT), Cloud Run Service (Interactive Slack Bot),
#          and Cloud Scheduler.
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-gmail-inbox-triage-515705}"
REGION="${GCP_REGION:-us-west1}"
REPO_NAME="${ARTIFACT_REPO:-gmail-triage}"
JOB_NAME="${CLOUD_RUN_JOB:-gmail-inbox-triage}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-gmail-inbox-triage-bot}"
SCHEDULER_NAME="${SCHEDULER_JOB:-gmail-inbox-triage-daily}"
TASKS_QUEUE="${CLOUD_TASKS_QUEUE:-triage-actions}"
SA_NAME="gmail-triage-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${JOB_NAME}:latest"

USE_VERTEXAI="${USE_VERTEXAI:-false}"
TOKEN_FILE="${GMAIL_TOKEN_PATH:-${HOME}/.local/share/gmail-inbox-triage/token.json}"

echo "==> Targeting project=${PROJECT_ID}, region=${REGION}, use_vertexai=${USE_VERTEXAI}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> 1. Enabling required GCP APIs..."
APIS=(
  run.googleapis.com
  cloudscheduler.googleapis.com
  cloudtasks.googleapis.com
  secretmanager.googleapis.com
  firestore.googleapis.com
  artifactregistry.googleapis.com
  cloudbuild.googleapis.com
  calendar-json.googleapis.com
  tasks.googleapis.com
)
if [[ "${USE_VERTEXAI}" == "true" ]]; then
  APIS+=(aiplatform.googleapis.com)
fi
gcloud services enable "${APIS[@]}"

echo "==> 2. Ensuring Firestore Native database exists..."
if ! gcloud firestore databases describe --database="(default)" >/dev/null 2>&1; then
  gcloud firestore databases create --location="${REGION}" --type=firestore-native --quiet
fi

echo "==> 3. Ensuring Artifact Registry repo & Cloud Tasks queue exist..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Gmail Inbox Triage container images"
fi

if ! gcloud tasks queues describe "${TASKS_QUEUE}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud tasks queues create "${TASKS_QUEUE}" --location="${REGION}"
fi

echo "==> 4. Ensuring least-privilege Service Account exists..."
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Gmail Inbox Triage Cloud Run SA"
  echo "    Waiting for Service Account IAM propagation..."
  sleep 10
fi

ROLES=(
  roles/secretmanager.secretAccessor
  roles/datastore.user
  roles/run.invoker
  roles/run.developer
  roles/cloudtasks.enqueuer
)
if [[ "${USE_VERTEXAI}" == "true" ]]; then
  ROLES+=(roles/aiplatform.user)
fi
for role in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --quiet >/dev/null
done

upsert_secret() {
  local name="$1"
  local value="$2"
  if ! gcloud secrets describe "${name}" >/dev/null 2>&1; then
    gcloud secrets create "${name}" --replication-policy="automatic"
  fi
  printf '%s' "${value}" | gcloud secrets versions add "${name}" --data-file=- >/dev/null
  echo "    Updated secret: ${name}"
}

echo "==> 5. Syncing secrets to Secret Manager..."
if [[ -f "${TOKEN_FILE}" ]]; then
  upsert_secret "gmail-oauth-token" "$(cat "${TOKEN_FILE}")"
else
  echo "    Note: ${TOKEN_FILE} not found; assuming secret 'gmail-oauth-token' already exists."
fi

SECRET_FLAGS="GMAIL_TOKEN_JSON=gmail-oauth-token:latest"

if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  upsert_secret "slack-webhook-url" "${SLACK_WEBHOOK_URL}"
fi
if gcloud secrets describe "slack-webhook-url" >/dev/null 2>&1; then
  SECRET_FLAGS="${SECRET_FLAGS},SLACK_WEBHOOK_URL=slack-webhook-url:latest"
fi

if [[ -n "${SLACK_BOT_TOKEN:-}" ]]; then
  upsert_secret "slack-bot-token" "${SLACK_BOT_TOKEN}"
fi
if gcloud secrets describe "slack-bot-token" >/dev/null 2>&1; then
  SECRET_FLAGS="${SECRET_FLAGS},SLACK_BOT_TOKEN=slack-bot-token:latest"
fi

if [[ -n "${SLACK_SIGNING_SECRET:-}" ]]; then
  upsert_secret "slack-signing-secret" "${SLACK_SIGNING_SECRET}"
fi
if gcloud secrets describe "slack-signing-secret" >/dev/null 2>&1; then
  SECRET_FLAGS="${SECRET_FLAGS},SLACK_SIGNING_SECRET=slack-signing-secret:latest"
fi

# Auto-generate internal Cloud Tasks secret if it does not exist yet
if [[ -n "${INTERNAL_TASK_SECRET:-}" ]]; then
  upsert_secret "internal-task-secret" "${INTERNAL_TASK_SECRET}"
elif ! gcloud secrets describe "internal-task-secret" >/dev/null 2>&1; then
  upsert_secret "internal-task-secret" "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
fi
SECRET_FLAGS="${SECRET_FLAGS},INTERNAL_TASK_SECRET=internal-task-secret:latest"

if [[ "${USE_VERTEXAI}" != "true" ]]; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    upsert_secret "gemini-api-key" "${GEMINI_API_KEY}"
  fi
  if gcloud secrets describe "gemini-api-key" >/dev/null 2>&1; then
    SECRET_FLAGS="${SECRET_FLAGS},GEMINI_API_KEY=gemini-api-key:latest"
  fi
fi

echo "==> 6. Building & pushing container image via Cloud Build..."
gcloud builds submit --tag "${IMAGE_URI}" .

COMMON_ENV="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},USE_VERTEXAI=${USE_VERTEXAI},VERTEX_LOCATION=global,STATE_BACKEND=firestore,TIMEZONE=America/Los_Angeles,CLOUD_RUN_JOB_NAME=${JOB_NAME},CLOUD_TASKS_QUEUE=${TASKS_QUEUE}"
if [[ -n "${SLACK_CHANNEL_ID:-}" ]]; then
  COMMON_ENV="${COMMON_ENV},SLACK_CHANNEL_ID=${SLACK_CHANNEL_ID}"
fi
if [[ -n "${SLACK_ALLOWED_USER_IDS:-}" ]]; then
  COMMON_ENV="${COMMON_ENV},SLACK_ALLOWED_USER_IDS=${SLACK_ALLOWED_USER_IDS}"
fi

echo "==> 7. Deploying Cloud Run Job (${JOB_NAME})..."
gcloud run jobs deploy "${JOB_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --set-env-vars "${COMMON_ENV}" \
  --set-secrets "${SECRET_FLAGS}" \
  --tasks 1 \
  --max-retries 1 \
  --task-timeout 600s \
  --memory 512Mi \
  --cpu 1

echo "==> 8. Deploying scale-to-zero Cloud Run Service (${SERVICE_NAME}) for Slack interactivity..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --args "serve,--port,8080" \
  --port 8080 \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --memory 512Mi \
  --cpu 1 \
  --set-env-vars "${COMMON_ENV}" \
  --set-secrets "${SECRET_FLAGS}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"
gcloud run services update "${SERVICE_NAME}" \
  --region "${REGION}" \
  --update-env-vars "CLOUD_RUN_SERVICE_URL=${SERVICE_URL}" >/dev/null

echo "==> 9. Scheduling daily 5:30 AM Pacific Cloud Scheduler trigger..."
JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"
if gcloud scheduler jobs describe "${SCHEDULER_NAME}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --schedule="30 5 * * *" \
    --time-zone="America/Los_Angeles" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
else
  gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --schedule="30 5 * * *" \
    --time-zone="America/Los_Angeles" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
fi

echo ""
echo "=============================================================================="
echo "Phase 2 Deployment complete!"
echo "Interactive Bot URL : ${SERVICE_URL}"
echo "  Slack Interactivity Request URL : ${SERVICE_URL}/slack/interactions"
echo "  Slack /triage Slash Command URL : ${SERVICE_URL}/slack/commands"
echo ""
echo "Test scheduled job immediately:"
echo "  gcloud run jobs execute ${JOB_NAME} --region ${REGION} --wait"
echo "=============================================================================="
