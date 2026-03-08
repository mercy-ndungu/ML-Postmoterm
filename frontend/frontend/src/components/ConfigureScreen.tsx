import { useState } from "react";
import axios from "axios";
import { UploadResponse, PostmortemResponse } from "../types";

const API = "http://localhost:8000";

interface Props {
  uploadData: UploadResponse;
  onSuccess: (data: PostmortemResponse) => void;
}

export default function ConfigureScreen({ uploadData, onSuccess }: Props) {
  const [target, setTarget] = useState(uploadData.columns[0].name);
  const [testSize, setTestSize] = useState(0.2);
  const [apiKey, setApiKey] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyse = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.post<PostmortemResponse>(
        `${API}/api/analyze/${uploadData.session_id}`,
        { target_column: target, test_size: testSize, anthropic_api_key: apiKey }
      );
      onSuccess(res.data);
    } catch {
      setError("Analysis failed. Please check your dataset and try again.");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <div style={styles.spinner}>⚙</div>
          <h2 style={styles.loadingTitle}>Analysing your dataset...</h2>
          <p style={styles.loadingText}>Training your model, computing SHAP values, and generating your postmortem. This takes about 20–40 seconds.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.title}>Configure Your Analysis</h2>
        <p style={styles.meta}>{uploadData.n_rows} rows · {uploadData.n_cols} columns</p>

        <label style={styles.label}>What do you want to predict?</label>
        <select style={styles.select} value={target} onChange={(e) => setTarget(e.target.value)}>
          {uploadData.columns.map((col) => (
            <option key={col.name} value={col.name}>{col.name} ({col.dtype})</option>
          ))}
        </select>
        <p style={styles.hint}>This is the column your model will learn to predict.</p>

        <label style={styles.label}>How much data to use for testing</label>
        <input style={styles.input} type="number" min={0.1} max={0.4} step={0.05} value={testSize} onChange={(e) => setTestSize(parseFloat(e.target.value))} />
        <p style={styles.hint}>0.2 means 20% of your data is held back to test accuracy. The default is fine for most cases.</p>

        <button style={styles.advancedBtn} onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? "▲ Hide" : "▼ Advanced options (optional)"}
        </button>

        {showAdvanced && (
          <>
            <label style={styles.label}>Anthropic API Key (optional)</label>
            <input style={styles.input} type="password" placeholder="sk-ant-..." value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <p style={styles.hint}>Enables an AI-written plain-English explanation of your results.</p>
          </>
        )}

        {error && <p style={styles.error}>{error}</p>}

        <button style={styles.button} onClick={handleAnalyse}>
          Run Analysis →
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "2rem" },
  card: { background: "#1a1d27", borderRadius: "16px", padding: "3rem", maxWidth: "560px", width: "100%", boxShadow: "0 25px 50px rgba(0,0,0,0.5)" },
  title: { fontSize: "1.75rem", fontWeight: 700, color: "#00d2ff", marginBottom: "0.25rem" },
  meta: { color: "#64748b", fontSize: "0.9rem", marginBottom: "2rem" },
  label: { display: "block", color: "#e2e8f0", fontWeight: 600, marginBottom: "0.5rem", marginTop: "1.25rem" },
  select: { width: "100%", padding: "0.75rem", background: "#0f1117", color: "#e2e8f0", border: "1px solid #2d3748", borderRadius: "8px", fontSize: "1rem" },
  input: { width: "100%", padding: "0.75rem", background: "#0f1117", color: "#e2e8f0", border: "1px solid #2d3748", borderRadius: "8px", fontSize: "1rem" },
  hint: { color: "#64748b", fontSize: "0.8rem", marginTop: "0.35rem" },
  advancedBtn: { background: "none", border: "none", color: "#00d2ff", cursor: "pointer", fontSize: "0.9rem", marginTop: "1.5rem", marginBottom: "0.5rem", padding: 0 },
  button: { width: "100%", padding: "1rem", background: "#00d2ff", color: "#0f1117", border: "none", borderRadius: "8px", fontSize: "1rem", fontWeight: 700, cursor: "pointer", marginTop: "2rem" },
  error: { color: "#f87171", marginTop: "1rem", fontSize: "0.9rem" },
  spinner: { fontSize: "3rem", textAlign: "center", marginBottom: "1rem", animation: "spin 2s linear infinite" },
  loadingTitle: { color: "#00d2ff", textAlign: "center", marginBottom: "1rem" },
  loadingText: { color: "#94a3b8", textAlign: "center", lineHeight: 1.6 },
};