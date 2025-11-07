# Invoice Insights — Monorepo (Cloud Run)

Single-container Cloud Run app:
- Frontend React (Vite) served by FastAPI
- Backend FastAPI with endpoints:
  - /upload_csv
  - /extract_invoice
  - /suggest_query
  - /confirm_execute

## Env vars (set in Cloud Run)
- GCS_BUCKET_NAME (required) — bucket to store uploads & results
- GCP_PROJECT_ID (required)
- GENAI_MODEL (optional) — default models/gpt-4o-mini
- DEMO_API_KEY (optional) — simple guard for confirm_execute

## Build & Deploy (gcloud)
1. Build and push:
```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/invoice-insights
