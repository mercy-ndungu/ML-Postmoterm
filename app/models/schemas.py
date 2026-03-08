from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AnalysisRequest(BaseModel):
    target_column: str
    task_type: Optional[str] = "auto"  # "regression", "classification", or "auto"
    test_size: Optional[float] = 0.2
    anthropic_api_key: Optional[str] = None  # passed from frontend; never stored


class FeatureImportance(BaseModel):
    feature: str
    shap_mean_abs: float
    direction: str  # "positive" or "negative"


class DiagnosticFlag(BaseModel):
    name: str
    severity: str  # "ok", "warning", "critical"
    detail: str


class MetricsSummary(BaseModel):
    task_type: str
    train_score: float
    test_score: float
    score_metric: str
    overfit_gap: float
    n_features: int
    n_samples: int


class PostmortemResponse(BaseModel):
    status: str
    metrics: MetricsSummary
    feature_importances: List[FeatureImportance]
    diagnostics: List[DiagnosticFlag]
    narrative: str  # Plain-English AI postmortem
    residuals_plot_b64: Optional[str] = None
    shap_plot_b64: Optional[str] = None
    correlation_plot_b64: Optional[str] = None
