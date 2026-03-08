export interface Column {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
  sample_values: (string | number)[];
}

export interface UploadResponse {
  session_id: string;
  n_rows: number;
  n_cols: number;
  columns: Column[];
}

export interface FeatureImportance {
  feature: string;
  shap_mean_abs: number;
  direction: "positive" | "negative";
}

export interface Diagnostic {
  name: string;
  severity: "ok" | "warning" | "critical";
  detail: string;
}

export interface Metrics {
  task_type: string;
  train_score: number;
  test_score: number;
  score_metric: string;
  overfit_gap: number;
  n_features: number;
  n_samples: number;
}

export interface PostmortemResponse {
  status: string;
  metrics: Metrics;
  feature_importances: FeatureImportance[];
  diagnostics: Diagnostic[];
  narrative: string;
  residuals_plot_b64?: string;
  shap_plot_b64?: string;
  correlation_plot_b64?: string;
}