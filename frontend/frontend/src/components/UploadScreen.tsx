import { useState, useRef } from "react";
import axios from "axios";
import { UploadResponse } from "../types";

const API = "http://localhost:8000";

interface Props {
  onSuccess: (data: UploadResponse) => void;
}

export default function UploadScreen({ onSuccess }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    if (!f.name.endsWith(".csv")) {
      setError("Please upload a CSV file.");
      return;
    }
    setFile(f);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post<UploadResponse>(`${API}/api/upload`, form);
      onSuccess(res.data);
    } catch {
      setError("Upload failed. Make sure your backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>ML Postmortem</h1>
        <p style={styles.subtitle}>
          Upload your dataset and find out what your model learned.
          No coding required.
        </p>

        <div
          style={{ ...styles.dropzone, borderColor: dragging ? "#00d2ff" : "#2d3748" }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
          onClick={() => inputRef.current?.click()}
        >
          <div style={styles.uploadIcon}>☁</div>
          {file ? (
            <p style={styles.filename}>{file.name} ({(file.size / 1024).toFixed(1)} KB)</p>
          ) : (
            <p style={styles.dropText}>Drag and drop a CSV file here, or click to browse</p>
          )}
          <input ref={inputRef} type="file" accept=".csv" style={{ display: "none" }} onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }} />
        </div>

        {error && <p style={styles.error}>{error}</p>}

        <button
          style={{ ...styles.button, opacity: !file || loading ? 0.5 : 1 }}
          disabled={!file || loading}
          onClick={handleUpload}
        >
          {loading ? "Uploading..." : "Upload & Configure →"}
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "2rem" },
  card: { background: "#1a1d27", borderRadius: "16px", padding: "3rem", maxWidth: "560px", width: "100%", boxShadow: "0 25px 50px rgba(0,0,0,0.5)" },
  title: { fontSize: "2rem", fontWeight: 700, color: "#00d2ff", marginBottom: "0.5rem" },
  subtitle: { color: "#94a3b8", marginBottom: "2rem", lineHeight: 1.6 },
  dropzone: { border: "2px dashed", borderRadius: "12px", padding: "3rem 2rem", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s", marginBottom: "1.5rem" },
  uploadIcon: { fontSize: "3rem", marginBottom: "1rem" },
  dropText: { color: "#64748b", fontSize: "0.95rem" },
  filename: { color: "#00d2ff", fontWeight: 600 },
  button: { width: "100%", padding: "1rem", background: "#00d2ff", color: "#0f1117", border: "none", borderRadius: "8px", fontSize: "1rem", fontWeight: 700, cursor: "pointer" },
  error: { color: "#f87171", marginBottom: "1rem", fontSize: "0.9rem" },
};