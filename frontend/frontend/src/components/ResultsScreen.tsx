import { useState } from "react";
import { PostmortemResponse, Diagnostic } from "../types";

interface Props {
  results: PostmortemResponse;
  onStartOver: () => void;
}

export default function ResultsScreen({ results, onStartOver }: Props) {
  const [openDiag, setOpenDiag] = useState<string | null>(null);
  const { metrics, diagnostics, narrative, feature_importances,
          shap_plot_b64, correlation_plot_b64, residuals_plot_b64 } = results;

  const severityColor = (s: Diagnostic["severity"]) =>
    s === "ok" ? "#4cdd91" : s === "warning" ? "#ffb347" : "#f87171";

  const severityBg = (s: Diagnostic["severity"]) =>
    s === "ok" ? "#052e16" : s === "warning" ? "#1c1008" : "#1c0a0a";

  return (
    <div style={styles.container}>
      <div style={styles.inner}>

        <div style={styles.header}>
          <div>
            <h1 style={styles.title}>Your Model Postmortem</h1>
            <p style={styles.subtitle}>Here is what your model learned and what to do next.</p>
          </div>
          <button style={styles.startOver} onClick={onStartOver}>← Start Over</button>
        </div>

        {/* Score Cards */}
        <div style={styles.scoreRow}>
          {[
            { label: "Task Type", value: metrics.task_type },
            { label: "Training Score", value: `${(metrics.train_score * 100).toFixed(1)}%` },
            { label: "Test Score", value: `${(metrics.test_score * 100).toFixed(1)}%` },
            { label: "Metric Used", value: metrics.score_metric },
            { label: "Features", value: metrics.n_features },
            { label: "Dataset Size", value: `${metrics.n_samples} rows` },
          ].map((item) => (
            <div key={item.label} style={styles.scoreCard}>
              <p style={styles.scoreLabel}>{item.label}</p>
              <p style={styles.scoreValue}>{item.value}</p>
            </div>
          ))}
        </div>

        {/* Diagnostics */}
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Diagnostic Checks</h2>
          <p style={styles.sectionHint}>Green means good. Amber means worth reviewing. Red means action needed.</p>
          <div style={styles.badgeRow}>
            {diagnostics.map((d) => (
              <div key={d.name}>
                <button
                  style={{ ...styles.badge, background: severityBg(d.severity), borderColor: severityColor(d.severity), color: severityColor(d.severity) }}
                  onClick={() => setOpenDiag(openDiag === d.name ? null : d.name)}
                >
                  {d.severity === "ok" ? "✓" : d.severity === "warning" ? "⚠" : "✗"} {d.name}
                </button>
                {openDiag === d.name && (
                  <p style={styles.diagDetail}>{d.detail}</p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Narrative */}
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>What Your Model Learned</h2>
          <div style={styles.narrativeCard}>
            {narrative.split("\n").map((line, i) => (
              <p key={i} style={{ marginBottom: "0.75rem", lineHeight: 1.7 }}>{line}</p>
            ))}
          </div>
        </div>

        {/* Feature Importance */}
        {shap_plot_b64 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Feature Importance</h2>
            <p style={styles.sectionHint}>The longer the bar, the more this feature influenced your model's predictions.</p>
            <img src={`data:image/png;base64,${shap_plot_b64}`} style={styles.plot} alt="SHAP feature importance" />
          </div>
        )}

        {/* Correlation */}
        {correlation_plot_b64 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Feature Correlations</h2>
            <p style={styles.sectionHint}>Shows how strongly your input columns relate to each other.</p>
            <img src={`data:image/png;base64,${correlation_plot_b64}`} style={styles.plot} alt="Correlation heatmap" />
          </div>
        )}

        {/* Residuals */}
        {residuals_plot_b64 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Residual Analysis</h2>
            <p style={styles.sectionHint}>Shows the difference between what your model predicted and the actual values.</p>
            <img src={`data:image/png;base64,${residuals_plot_b64}`} style={styles.plot} alt="Residuals" />
          </div>
        )}

        <button style={styles.bottomBtn} onClick={onStartOver}>← Analyse Another Dataset</button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { minHeight: "100vh", padding: "2rem" },
  inner: { maxWidth: "860px", margin: "0 auto" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "2rem", flexWrap: "wrap", gap: "1rem" },
  title: { fontSize: "2rem", fontWeight: 700, color: "#00d2ff" },
  subtitle: { color: "#94a3b8", marginTop: "0.25rem" },
  startOver: { background: "none", border: "1px solid #2d3748", color: "#94a3b8", padding: "0.5rem 1rem", borderRadius: "8px", cursor: "pointer", fontSize: "0.9rem" },
  scoreRow: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "1rem", marginBottom: "2rem" },
  scoreCard: { background: "#1a1d27", borderRadius: "12px", padding: "1.25rem", textAlign: "center" },
  scoreLabel: { color: "#64748b", fontSize: "0.8rem", marginBottom: "0.5rem", textTransform: "uppercase", letterSpacing: "0.05em" },
  scoreValue: { color: "#00d2ff", fontSize: "1.25rem", fontWeight: 700, textTransform: "capitalize" },
  section: { background: "#1a1d27", borderRadius: "12px", padding: "1.75rem", marginBottom: "1.5rem" },
  sectionTitle: { color: "#e2e8f0", fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.4rem" },
  sectionHint: { color: "#64748b", fontSize: "0.85rem", marginBottom: "1.25rem" },
  badgeRow: { display: "flex", flexWrap: "wrap", gap: "0.75rem" },
  badge: { border: "1px solid", borderRadius: "20px", padding: "0.4rem 1rem", fontSize: "0.85rem", fontWeight: 600, cursor: "pointer", background: "none" },
  diagDetail: { marginTop: "0.5rem", color: "#94a3b8", fontSize: "0.85rem", lineHeight: 1.6, padding: "0.75rem", background: "#0f1117", borderRadius: "8px", maxWidth: "480px" },
  narrativeCard: { color: "#cbd5e1", lineHeight: 1.7, maxHeight: "320px", overflowY: "auto", paddingRight: "0.5rem" },
  plot: { width: "100%", borderRadius: "8px", marginTop: "0.5rem" },
  bottomBtn: { width: "100%", padding: "1rem", background: "#1a1d27", color: "#00d2ff", border: "1px solid #00d2ff", borderRadius: "8px", fontSize: "1rem", fontWeight: 700, cursor: "pointer", marginTop: "1rem" },
};