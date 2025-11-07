import React from "react";

export default function App() {
  const [file, setFile] = React.useState(null);
  const [template, setTemplate] = React.useState("Vendor Name Extract");
  const [modifier, setModifier] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [gcsPath, setGcsPath] = React.useState("");
  const [suggestedSql, setSuggestedSql] = React.useState("");
  const [downloadUrl, setDownloadUrl] = React.useState("");

  // Function to download CSV from data
  const downloadCSV = (csvData, filename) => {
    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  async function uploadFile() {
    if (!file) {
      setMessage("Please choose a CSV file first.");
      return null;
    }
    setLoading(true);
    setMessage("Uploading file...");
    try {
      const form = new FormData();
      form.append("file", file);
      // IMPORTANT: On Cloud Run the backend is same domain, use relative path
      const res = await fetch("/upload_csv", { method: "POST", body: form });
      
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setGcsPath(data.gcs_uri || data.gcs_path || "");
      setMessage("Upload complete.");
      return data.gcs_uri || data.gcs_path;
    } catch (err) {
      console.error(err);
      setMessage("Upload failed: " + err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }

  function buildPrompt(templateName, modifierText) {
    const map = {
      "Vendor Name Extract": "Extract vendor names and totals.",
      "Total Amount Summary": "Summarize total invoice amounts grouped by vendor.",
      "Invoice Date Extract": "Extract invoice dates and invoice IDs."
    };
    let base = map[templateName] || templateName;
    if (modifierText && modifierText.trim().length) base += " Modifier: " + modifierText.trim();
    return base;
  }

  async function runExtractionFlow() {
    setDownloadUrl("");
    setSuggestedSql("");
    setMessage("");

    const uploadedPath = await uploadFile();
    if (!uploadedPath) return;

    setLoading(true);
    setMessage("Asking AI for suggested SQL...");
    const userQuestion = buildPrompt(template, modifier);
    try {
      const payload = { natural_language_query: userQuestion, gcs_csv_path: uploadedPath, max_rows: 100 };
      const res = await fetch("/suggest_query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSuggestedSql(data.suggested_sql || data.sql || "");
      setMessage("Suggested SQL ready. Review then Confirm Execute.");
    } catch (err) {
      console.error(err);
      setMessage("Suggestion failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  async function confirmExecute() {
    if (!suggestedSql) {
      setMessage("No suggested SQL to execute.");
      return;
    }
    if (!gcsPath) {
      setMessage("Missing GCS path for CSV.");
      return;
    }
    setLoading(true);
    setMessage("Executing confirmed SQL and generating CSV...");
    try {
      const payload = { 
        sql: suggestedSql, 
        gcs_csv_path: gcsPath, 
        preferred_output: "csv", 
        filename: "results_" + Date.now() 
      };
      const res = await fetch("/confirm_execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      
      // Handle the new response format with csv_data
      if (data.csv_data) {
        const filename = data.filename || `results_${Date.now()}.csv`;
        // Trigger download immediately
        downloadCSV(data.csv_data, filename);
        setMessage(`Execution finished. Downloaded ${filename} with ${data.row_count || 0} rows.`);
        setDownloadUrl(""); // Clear any previous URL since we're downloading directly
      } 
      // Fallback for old response format (if backend hasn't been updated yet)
      else if (data.download_url || data.signed_url) {
        const url = data.download_url || data.signed_url;
        setDownloadUrl(url);
        setMessage("Execution finished. Download ready.");
      } 
      else if (data.gcs_uri) {
        setMessage("Execution finished. Results at " + data.gcs_uri);
      } 
      else {
        setMessage("Execution finished but no download data returned.");
      }
    } catch (err) {
      console.error(err);
      setMessage("Execution failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 24, minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
      <h1 style={{ fontSize: 22 }}>AI Invoice Insights — Demo UI</h1>
      <p style={{ color: "#666" }}>Semi-auto flow: Upload CSV → AI suggests SQL → You confirm → Download CSV</p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, width: "100%", maxWidth: 1100 }}>
        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
          <h3>Upload Invoice CSV</h3>
          <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <p style={{ fontSize: 12, color: "#777" }}>CSV headers: invoice_id, invoice_date, vendor, total_amount, etc.</p>
        </div>

        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
          <h3>Prompt Template & Modifier</h3>
          <select value={template} onChange={(e) => setTemplate(e.target.value)} style={{ width: "100%", padding: 8 }}>
            <option>Vendor Name Extract</option>
            <option>Total Amount Summary</option>
            <option>Invoice Date Extract</option>
          </select>
          <input placeholder="Optional modifier (e.g. totals > 5000)" value={modifier} onChange={(e) => setModifier(e.target.value)} style={{ width: "100%", marginTop: 8, padding: 8 }} />
        </div>

        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
          <div>
            <h3>Run Extraction & Suggest</h3>
            <button onClick={runExtractionFlow} disabled={loading} style={{ padding: 10, width: "100%", background: "#0ea5e9", color: "white", borderRadius: 10 }}>
              {loading ? "Working..." : "Run & Suggest"}
            </button>
            <p style={{ fontSize: 12, color: "#777" }}>Uploads CSV, asks AI for suggested SQL, shows preview.</p>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Suggested SQL</h4>
            <pre style={{ background: "#f5f5f5", padding: 8, borderRadius: 8, height: 120, overflow: "auto" }}>{suggestedSql || "(no suggestion yet)"}</pre>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => navigator.clipboard.writeText(suggestedSql)} style={{ padding: 8 }}>Copy SQL</button>
              <button onClick={confirmExecute} disabled={!suggestedSql || loading} style={{ padding: 8, background: "#16a34a", color: "white", borderRadius: 6 }}>
                Confirm & Execute
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ width: "100%", maxWidth: 1100, border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
        <h3>Result</h3>
        <p style={{ color: "#444" }}>{message}</p>
        {/* Only show download button for old URL-based approach */}
        {downloadUrl ? (
          <a href={downloadUrl} target="_blank" rel="noreferrer" style={{ padding: 8, background: "#4f46e5", color: "white", borderRadius: 6 }}>
            Download CSV
          </a>
        ) : null}
      </div>
    </div>
  );
}



