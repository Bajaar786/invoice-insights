                ┌─────────────────────────┐
                │ React Frontend (Cloud Run)
User (Browser)  │ Upload CSV + NL Query   │
        ───────▶│                         │
                └───────────┬─────────────┘
                            │ REST API
                            ▼
                ┌─────────────────────────┐
                │ FastAPI Backend (Cloud Run)
                │ Service Account Auth     │
                │                          │
                │ Calls Gemini 2.0 Flash   │
                │ Generates SQL            │
                └───────────┬─────────────┘
                            │
                            ▼
         ┌────────────── BigQuery ───────────────┐
         │ External Table created dynamically     │
         │ Executes generated SQL query           │
         └───────────────┬───────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │ Google Cloud Storage (GCS)  │
           │ Stores raw CSV + result CSV │
           └─────────────────────────────┘
