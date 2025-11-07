"""
Invoice Insights backend (Vertex AI version)
- Uses Vertex AI Gemini (model: gemini-2.0-flash) to suggest BigQuery SQL for invoice CSVs.
- Endpoints:
  - POST /upload_csv       -> upload CSV to GCS (returns gs:// URI)
  - POST /extract_invoice  -> extract invoice JSON -> CSV
  - POST /suggest_query    -> ask Gemini to return JSON {sql, explanation, confidence}
  - POST /confirm_execute  -> create BigQuery external table, run SQL, export CSV to GCS, return signed URL
  - GET  /healthz
"""

import os
import io
import json
import uuid
import time
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from google.cloud import storage, bigquery
from fastapi.middleware.cors import CORSMiddleware

# Import Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import pandas as pd

# ---------- Configuration ----------
GCS_BUCKET = os.environ.get("GCS_BUCKET_NAME", "")         # e.g. "my-invoice-bucket"
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")         # your GCP project id
GENAI_MODEL = os.environ.get("GENAI_MODEL", "gemini-2.0-flash")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")  # Vertex AI location

# Validate required environment variables
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET_NAME environment variable is required")
if not PROJECT_ID:
    raise RuntimeError("GCP_PROJECT_ID environment variable is required")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

storage_client = storage.Client()
bq_client = bigquery.Client()

app = FastAPI(title="Invoice Insights Engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request models ----------
class SuggestQueryRequest(BaseModel):
    natural_language_query: str
    gcs_csv_path: str
    max_rows: int = 100

class ExecRequest(BaseModel):
    sql: str
    gcs_csv_path: str
    preferred_output: Optional[str] = "csv"
    filename: Optional[str] = None

class ExtractRequest(BaseModel):
    raw_text: Optional[str] = None
    gcs_input_path: Optional[str] = None
    output_filename: Optional[str] = None

# ---------- Helpers ----------
def parse_gs_uri(gs_uri: str):
    if gs_uri.startswith("gs://"):
        _, rest = gs_uri.split("gs://", 1)
        bucket, path = rest.split("/", 1)
        return bucket, path
    if "/" in gs_uri:
        return GCS_BUCKET, gs_uri
    return GCS_BUCKET, gs_uri

def upload_bytes_to_gcs(content: bytes, bucket_name: str, blob_name: str, content_type: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type=content_type)
    return blob

def make_signed_url(blob, expiration_seconds: int = 3600):
    """
    Generate a download URL for Cloud Storage blob.
    On Cloud Run, we use public URLs since signed URLs require a private key.
    """
    try:
        # For Cloud Run environments, use public URL approach
        # This requires the bucket to have appropriate IAM permissions
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"⚠️ [DEBUG] Public URL failed, trying alternative methods: {e}")
        
        # Fallback: try to generate signed URL (might work in some environments)
        try:
            return blob.generate_signed_url(expiration=expiration_seconds)
        except Exception as e2:
            print(f"❌ [DEBUG] All URL generation methods failed: {e2}")
            # Final fallback: return the GCS URI
            return f"gs://{blob.bucket.name}/{blob.name}"

# ---------- Prompt templates ----------
EXTRACTOR_PROMPT = """
SYSTEM: You are an invoice extraction assistant. Input is raw invoice or OCR text.
YOU MUST OUTPUT JSON ONLY with keys:
invoice_id, invoice_date (YYYY-MM-DD), vendor, vendor_address,
line_items (array of {description, quantity, unit_price, tax, total_price}),
subtotal, tax_total, total_amount, currency, notes.
If a field is missing use null or empty. Numbers numeric. Dates ISO if possible.
INPUT:
\"\"\"{raw_text}\"\"\"
"""

SQL_PROMPT_TEMPLATE = """
SYSTEM: You are an expert SQL generator producing BigQuery-compatible SQL for an invoice table named `df`.
OUTPUT A SINGLE JSON OBJECT ONLY (no commentary) with keys: "sql", "explanation", "confidence".

SCHEMA:
{schema}

USER_QUESTION:
{user_question}

INSTRUCTIONS:
- Use column names exactly as provided.
- If returning rows, include LIMIT {max_rows}.
- For date filtering, assume invoice_date is ISO (YYYY-MM-DD).
- If referencing the table, the model may use table name `df`. The backend will replace `df` with the BigQuery external table reference.
- You can use standard SQL functions like GENERATE_UUID(), COUNT(), SUM(), AVG(), etc.
- Only generate SELECT or WITH queries - no DROP, DELETE, INSERT, UPDATE, CREATE, ALTER statements.
- Output exactly ONE JSON object and nothing else.
"""

# ---------- Generative API call using Vertex AI ----------
def call_gemini(prompt: str, model: Optional[str] = None, temperature: float = 0.0) -> str:
    """
    Call Gemini using Vertex AI SDK with service account authentication
    """
    model_name = model or GENAI_MODEL
    
    try:
        # Initialize the generative model
        generative_model = GenerativeModel(model_name)
        
        # Generate content with configuration
        response = generative_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Vertex AI API error: {str(e)}")
        raise RuntimeError(f"Vertex AI API error: {str(e)}")

# ---------- Endpoints ----------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        blob_name = f"uploads/{uuid.uuid4().hex}_{file.filename}"
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(file.file, content_type="text/csv")
        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        return {"gcs_uri": gs_uri, "gcs_path": blob_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/extract_invoice")
async def extract_invoice(req: ExtractRequest):
    if not (req.raw_text or req.gcs_input_path):
        raise HTTPException(status_code=400, detail="Provide raw_text or gcs_input_path")

    raw_text = req.raw_text
    if not raw_text:
        bucket, path = parse_gs_uri(req.gcs_input_path)
        blob = storage.Client().bucket(bucket).blob(path)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Input blob not found")
        raw_text = blob.download_as_text()

    prompt = EXTRACTOR_PROMPT.format(raw_text=raw_text)
    
    try:
        gen = call_gemini(prompt, temperature=0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini extraction failed: {str(e)}")

    try:
        first, last = gen.find("{"), gen.rfind("}")
        json_text = gen[first:last+1]
        parsed = json.loads(json_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}. Raw: {gen[:400]}")

    invoice_id = parsed.get("invoice_id") or f"inv_{int(time.time())}"
    line_items = parsed.get("line_items") or []

    rows = []
    if line_items:
        for li in line_items:
            rows.append({
                "invoice_id": invoice_id,
                "invoice_date": parsed.get("invoice_date"),
                "vendor": parsed.get("vendor"),
                "vendor_address": parsed.get("vendor_address"),
                "description": li.get("description"),
                "quantity": li.get("quantity"),
                "unit_price": li.get("unit_price"),
                "tax": li.get("tax"),
                "total_price": li.get("total_price"),
                "subtotal": parsed.get("subtotal"),
                "tax_total": parsed.get("tax_total"),
                "total_amount": parsed.get("total_amount"),
                "currency": parsed.get("currency"),
                "notes": parsed.get("notes")
            })
    else:
        rows.append({
            "invoice_id": invoice_id,
            "invoice_date": parsed.get("invoice_date"),
            "vendor": parsed.get("vendor"),
            "vendor_address": parsed.get("vendor_address"),
            "description": "",
            "quantity": None,
            "unit_price": None,
            "tax": parsed.get("tax_total") or 0,
            "total_price": parsed.get("total_amount"),
            "subtotal": parsed.get("subtotal"),
            "tax_total": parsed.get("tax_total"),
            "total_amount": parsed.get("total_amount"),
            "currency": parsed.get("currency"),
            "notes": parsed.get("notes")
        })

    df = pd.DataFrame(rows)
    filename_base = req.output_filename or f"extracted_{invoice_id}_{int(time.time())}"
    blob_name = f"extracted/{filename_base}.csv"
    blob = upload_bytes_to_gcs(df.to_csv(index=False).encode("utf-8"), GCS_BUCKET, blob_name, "text/csv")
    signed = make_signed_url(blob, expiration_seconds=3600)
    return {"gs_uri": f"gs://{GCS_BUCKET}/{blob_name}", "signed_url": signed, "rows": len(df)}

@app.post("/suggest_query")
async def suggest_query(req: SuggestQueryRequest):
    try:
        bucket, obj = parse_gs_uri(req.gcs_csv_path)
        tmpf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        storage.Client().bucket(bucket).blob(obj).download_to_filename(tmpf.name)
        df = pd.read_csv(tmpf.name, nrows=5)

        schema_text = ", ".join(df.columns.tolist())
        prompt = SQL_PROMPT_TEMPLATE.format(
            schema=schema_text, 
            user_question=req.natural_language_query, 
            max_rows=req.max_rows
        )
        
        gen_text = call_gemini(prompt, temperature=0.0)

        try:
            first, last = gen_text.find("{"), gen_text.rfind("}")
            json_text = gen_text[first:last+1]
            print(f"🔍 [DEBUG] Extracted JSON: {json_text}")
            
            parsed = json.loads(json_text)
            sql = parsed.get("sql", "").strip().rstrip(";")
            explanation = parsed.get("explanation", "")
            
            # FIX: Handle confidence as string, not float
            confidence_str = parsed.get("confidence", "Medium")
            # Convert string confidence to numeric if needed, but keep as string for now
            confidence_map = {"High": "High", "Medium": "Medium", "Low": "Low"}
            confidence = confidence_map.get(confidence_str, "Medium")
            
            print(f"🔍 [DEBUG] Parsed SQL: '{sql}'")
            print(f"🔍 [DEBUG] Confidence: {confidence}")
            
        except Exception as e:
            print(f"❌ [DEBUG] JSON parsing error: {e}")
            return {
                "suggested_sql": gen_text, 
                "safe": False, 
                "preview": [], 
                "estimated_rows": 0, 
                "note": f"Failed to parse model JSON: {e}"
            }

        return {
            "suggested_sql": sql,  # This should be JUST the SQL string
            "explanation": explanation, 
            "confidence": confidence,  # Now a string
            "safe": True,
            "preview": [], 
            "estimated_rows": 0
        }
        
    except Exception as e:
        print(f"❌ [DEBUG] suggest_query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query suggestion failed: {str(e)}")
        
@app.post("/confirm_execute")
async def confirm_execute(req: ExecRequest):
    # Extract clean SQL from potential JSON input
    sql_input = (req.sql or "").strip()
    
    print(f"🔍 [DEBUG] Received SQL input: '{sql_input}'")
    print(f"🔍 [DEBUG] Input length: {len(sql_input)}")
    print(f"🔍 [DEBUG] Input type: {type(sql_input)}")
    
    # Try to parse as JSON first (in case frontend sends the whole Gemini response)
    clean_sql = sql_input
    try:
        # Look for JSON pattern
        if sql_input.startswith('{') or 'sql' in sql_input.lower():
            # Try to find JSON object
            import re
            json_match = re.search(r'\{.*?"sql".*?\}', sql_input, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                clean_sql = parsed_json.get('sql', '').strip().rstrip(';')
                print(f"🔍 [DEBUG] Extracted SQL from JSON: '{clean_sql}'")
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"⚠️ [DEBUG] JSON parsing failed, using raw input: {e}")
        # If JSON parsing fails, use the raw input but try to clean it
        if 'sql' in sql_input.lower() and 'explanation' in sql_input.lower():
            # This looks like the full Gemini response, try to extract SQL more aggressively
            lines = sql_input.split('\n')
            for i, line in enumerate(lines):
                if 'SELECT' in line.upper() or 'FROM' in line.upper():
                    # Found a SQL-like line, use it
                    clean_sql = line.strip().rstrip(';')
                    break
    
    # Final cleanup
    clean_sql = clean_sql.strip().rstrip(';')
    
    # Check if SQL is empty
    if not clean_sql:
        raise HTTPException(status_code=400, detail="SQL query is empty")
    
    print(f"🔍 [DEBUG] Clean SQL to execute: '{clean_sql}'")
    
    bucket, path = parse_gs_uri(req.gcs_csv_path)
    dataset_id = f"{PROJECT_ID}.invoice_temp"
    
    try:
        bq_client.get_dataset(dataset_id)
    except Exception:
        bq_client.create_dataset(dataset_id, exists_ok=True)

    table_id = f"ext_{uuid.uuid4().hex[:8]}"
    table_ref = f"{PROJECT_ID}.invoice_temp.{table_id}"
    
    print(f"🔍 [DEBUG] Table reference: {table_ref}")

    external_config = bigquery.ExternalConfig("CSV")
    external_config.source_uris = [f"gs://{bucket}/{path}"]
    external_config.options.skip_leading_rows = 1
    external_config.autodetect = True

    table = bigquery.Table(table_ref)
    table.external_data_configuration = external_config
    table = bq_client.create_table(table, exists_ok=True)

    sql_to_run = clean_sql.replace("FROM df", f"FROM `{table_ref}`").replace("from df", f"from `{table_ref}`")
    
    print(f"🔍 [DEBUG] Final SQL to run: '{sql_to_run}'")
    
    try:
        job = bq_client.query(sql_to_run)
        result = job.result()
        print(f"✅ [DEBUG] BigQuery job completed successfully")
    except Exception as e:
        print(f"❌ [DEBUG] BigQuery error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BigQuery execution failed: {str(e)}")

    # Convert result to CSV string
    out_buf = io.StringIO()
    header = [schema.name for schema in result.schema]
    out_buf.write(",".join(header) + "\n")
    for row in result:
        values = ["" if getattr(row, f) is None else str(getattr(row, f)) for f in header]
        out_buf.write(",".join(values) + "\n")
    
    csv_data = out_buf.getvalue()
    
    # Generate a filename
    filename_base = req.filename or f"result_{int(time.time())}"
    filename = f"{filename_base}.csv"
    
    # Return CSV data directly in the response
    return {
        "csv_data": csv_data,
        "filename": filename,
        "row_count": result.total_rows if hasattr(result, 'total_rows') else 0,
        "message": "Download ready - use the csv_data field to download the file"
    }



@app.get("/healthz")
def healthz():
    return {
        "status": "ok", 
        "project": PROJECT_ID, 
        "bucket": GCS_BUCKET,
        "vertex_ai_configured": True,
        "location": LOCATION
    }

# Serve frontend static if present
static_dir = os.path.join(os.path.dirname(__file__), "frontend/dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")