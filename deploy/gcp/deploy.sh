#!/usr/bin/env bash
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

# =============================================================================
# Idempotent GCP Serverless Deployment for Gmail Inbox Triage
# Deploys: Artifact Registry, Secret Manager, Firestore, Cloud Run Job,
#          and Cloud Scheduler (daily 05:30 America/Los_Angeles).
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-gmail-inbox-triage-515705}"
REGION="${GCP_REGION:-us-west1}"
REPO_NAME="${ARTIFACT_REPO:-gmail-triage}"
JOB_NAME="${CLOUD_RUN_JOB:-gmail-inbox-triage}"
SCHEDULER_NAME="${SCHEDULER_JOB:-gmail-inbox-triage-daily}"
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
  secretmanager.googleapis.com
  firestore.googleapis.com
  artifactregistry.googleapis.com
  cloudbuild.googleapis.com
)
if [[ "${USE_VERTEXAI}" == "true" ]]; then
  APIS+=(aiplatform.googleapis.com)
fi
gcloud services enable "${APIS[@]}"

echo "==> 2. Ensuring Firestore Native database exists..."
if ! gcloud firestore databases describe --database="(default)" >/dev/null 2>&1; then
  gcloud firestore databases create --location="${REGION}" --type=firestore-native --quiet
fi

echo "==> 3. Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Gmail Inbox Triage container images"
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

if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  upsert_secret "slack-webhook-url" "${SLACK_WEBHOOK_URL}"
else
  echo "    Note: SLACK_WEBHOOK_URL env unset; assuming secret 'slack-webhook-url' already exists."
fi

SECRET_FLAGS="GMAIL_TOKEN_JSON=gmail-oauth-token:latest,SLACK_WEBHOOK_URL=slack-webhook-url:latest"
if [[ "${USE_VERTEXAI}" != "true" ]]; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    upsert_secret "gemini-api-key" "${GEMINI_API_KEY}"
  else
    echo "    Note: GEMINI_API_KEY env unset; assuming secret 'gemini-api-key' already exists."
  fi
  SECRET_FLAGS="${SECRET_FLAGS},GEMINI_API_KEY=gemini-api-key:latest"
fi

echo "==> 6. Building & pushing container image via Cloud Build..."
gcloud builds submit --tag "${IMAGE_URI}" .

echo "==> 7. Deploying Cloud Run Job (${JOB_NAME})..."
ENV_VARS="GCP_PROJECT_ID=${PROJECT_ID},USE_VERTEXAI=${USE_VERTEXAI},VERTEX_LOCATION=global,STATE_BACKEND=firestore,TIMEZONE=America/Los_Angeles"
gcloud run jobs deploy "${JOB_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --set-env-vars "${ENV_VARS}" \
  --set-secrets "${SECRET_FLAGS}" \
  --tasks 1 \
  --max-retries 1 \
  --task-timeout 600s \
  --memory 512Mi \
  --cpu 1

echo "==> 8. Scheduling daily 5:30 AM Pacific Cloud Scheduler trigger..."
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
echo "Deployment complete!"
echo "Test immediately with:"
echo "  gcloud run jobs execute ${JOB_NAME} --region ${REGION} --wait"
