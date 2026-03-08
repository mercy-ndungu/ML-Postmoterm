"""
ml_engine.py
Core analysis pipeline:
  1. Preprocess (encode categoricals, impute nulls)
  2. Auto-detect task type (regression vs classification)
  3. Train/test split + model fit
  4. Compute metrics
  5. Run diagnostics (overfitting, VIF, class imbalance, residuals)
  6. SHAP feature importance
  7. Generate plots as base64 strings
"""

import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ─── Colour palette ───────────────────────────────────────────────
PALETTE = {
    "bg": "#0f1117",
    "panel": "#1a1d27",
    "accent": "#00d2ff",
    "warn": "#ffb347",
    "ok": "#4cdd91",
    "text": "#e2e8f0",
    "muted": "#64748b",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["panel"],
    "axes.edgecolor": PALETTE["muted"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "text.color": PALETTE["text"],
    "grid.color": "#2d3748",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})


# ─── Helpers ──────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _detect_task(series: pd.Series) -> str:
    n_unique = series.nunique()
    if series.dtype == object or n_unique <= 10:
        return "classification"
    return "regression"


def _preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()

    # Drop columns with >50% nulls
    threshold = len(df) * 0.5
    df = df.loc[:, df.isnull().sum() < threshold]

    # Separate target
    y_raw = df.pop(target_col)

    # Encode categoricals in features
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Impute remaining nulls with median
    df = df.fillna(df.median(numeric_only=True))

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    return df, y_raw


def _encode_target(y: pd.Series, task: str):
    if task == "classification" and y.dtype == object:
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le.classes_.tolist()
    return y.values, None


def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for multicollinearity detection."""
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])
    vif_data = []
    X_arr = X.values
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X_arr, i)
        except Exception:
            v = np.nan
        vif_data.append({"feature": col, "vif": round(float(v), 2)})
    return pd.DataFrame(vif_data).sort_values("vif", ascending=False)


# ─── Main pipeline ────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, target_col: str, task_type: str = "auto", test_size: float = 0.2) -> dict:
    """
    Full analysis pipeline. Returns a dict ready to be serialised into PostmortemResponse.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # ── 1. Preprocess ──
    X, y_raw = _preprocess(df, target_col)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain after preprocessing.")

    # ── 2. Task type ──
    if task_type == "auto":
        task_type = _detect_task(y_raw)

    y, label_classes = _encode_target(y_raw, task_type)

    # ── 3. Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── 4. Model fit (two models: linear/logistic for interpretability + RF for SHAP) ──
    if task_type == "regression":
        model_interp = LinearRegression()
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_interp.fit(X_train_sc, y_train)
        model_rf.fit(X_train, y_train)

        train_pred = model_interp.predict(X_train_sc)
        test_pred = model_interp.predict(X_test_sc)
        train_score = float(r2_score(y_train, train_pred))
        test_score = float(r2_score(y_test, test_pred))
        score_metric = "R²"
        residuals = y_test - test_pred

    else:  # classification
        model_interp = LogisticRegression(max_iter=1000, random_state=42)
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_interp.fit(X_train_sc, y_train)
        model_rf.fit(X_train, y_train)

        train_pred = model_interp.predict(X_train_sc)
        test_pred = model_interp.predict(X_test_sc)
        train_score = float(accuracy_score(y_train, train_pred))
        test_score = float(accuracy_score(y_test, test_pred))
        score_metric = "Accuracy"
        residuals = None  # not meaningful for classification

    overfit_gap = round(train_score - test_score, 4)

    # ── 5. Diagnostics ──
    diagnostics = []

    # Overfitting check
    if overfit_gap > 0.15:
        diagnostics.append({
            "name": "Overfitting Detected",
            "severity": "critical",
            "detail": f"Train {score_metric} is {overfit_gap:.2%} higher than test. Your model has memorised the training data rather than learning general patterns. Consider regularisation, fewer features, or more data."
        })
    elif overfit_gap > 0.05:
        diagnostics.append({
            "name": "Mild Overfitting",
            "severity": "warning",
            "detail": f"Train {score_metric} is {overfit_gap:.2%} above test. Slight generalisation gap — worth monitoring as your dataset grows."
        })
    else:
        diagnostics.append({
            "name": "Generalisation",
            "severity": "ok",
            "detail": f"Train/test gap is only {overfit_gap:.2%}. Your model generalises well."
        })

    # Dataset size
    if len(df) < 100:
        diagnostics.append({
            "name": "Small Dataset",
            "severity": "warning",
            "detail": f"Only {len(df)} rows. ML models need more data to learn reliable patterns. Treat all results with caution."
        })
    else:
        diagnostics.append({
            "name": "Dataset Size",
            "severity": "ok",
            "detail": f"{len(df)} rows — sufficient for a baseline model."
        })

    # Null values
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) > 0:
        top_nulls = ", ".join([f"{c} ({v})" for c, v in null_cols.head(3).items()])
        diagnostics.append({
            "name": "Missing Values",
            "severity": "warning",
            "detail": f"Found null values in: {top_nulls}. These were median-imputed for training, but missing data can bias your model."
        })

    # VIF multicollinearity (regression only)
    if task_type == "regression" and X.shape[1] >= 2:
        vif_df = _compute_vif(X)
        high_vif = vif_df[vif_df["vif"] > 10]
        if not high_vif.empty:
            flagged = ", ".join(high_vif["feature"].head(3).tolist())
            diagnostics.append({
                "name": "Multicollinearity (VIF)",
                "severity": "warning",
                "detail": f"Features with VIF > 10: {flagged}. These features are highly correlated with each other, which inflates coefficient uncertainty."
            })
        else:
            diagnostics.append({
                "name": "Multicollinearity (VIF)",
                "severity": "ok",
                "detail": "No severe multicollinearity detected (all VIF < 10)."
            })

    # Class imbalance (classification only)
    if task_type == "classification":
        class_counts = pd.Series(y).value_counts(normalize=True)
        min_class_pct = float(class_counts.min())
        if min_class_pct < 0.15:
            diagnostics.append({
                "name": "Class Imbalance",
                "severity": "critical",
                "detail": f"Minority class is only {min_class_pct:.1%} of data. Accuracy will be misleading — consider SMOTE, class_weight='balanced', or F1 as your metric."
            })
        elif min_class_pct < 0.30:
            diagnostics.append({
                "name": "Mild Class Imbalance",
                "severity": "warning",
                "detail": f"Minority class is {min_class_pct:.1%} of data. Monitor F1-score alongside accuracy."
            })

    # ── 6. SHAP ──
    explainer = shap.TreeExplainer(model_rf)
    shap_sample = X_test.head(min(100, len(X_test)))
    shap_values_raw = explainer.shap_values(shap_sample)

   # Handle multiclass SHAP (3D array: classes x samples x features)
    if isinstance(shap_values_raw, list):
        # older SHAP: list of arrays, one per class
        sv = np.mean([np.abs(s) for s in shap_values_raw], axis=0)
        raw_sv = np.mean(shap_values_raw, axis=0)
    elif shap_values_raw.ndim == 3:
        # newer SHAP: 3D array (samples x features x classes)
        sv = np.mean(np.abs(shap_values_raw), axis=2)
        raw_sv = np.mean(shap_values_raw, axis=2)
    else:
        sv = np.abs(shap_values_raw)
        raw_sv = shap_values_raw

    mean_shap = np.mean(sv, axis=0)
    feature_names = list(X.columns)
    mean_shap_signed = np.mean(raw_sv, axis=0)

    feature_importances = []
    for i, fname in enumerate(feature_names):
        feature_importances.append({
            "feature": fname,
            "shap_mean_abs": round(float(mean_shap[i]), 4),
            "direction": "positive" if mean_shap_signed[i] >= 0 else "negative"
        })
    feature_importances.sort(key=lambda x: x["shap_mean_abs"], reverse=True)

    # ── 7. Plots ──
    residuals_plot_b64 = None
    if task_type == "regression" and residuals is not None:
        residuals_plot_b64 = _plot_residuals(y_test, test_pred, residuals)

    shap_plot_b64 = _plot_shap(feature_importances[:10])
    correlation_plot_b64 = _plot_correlation(X.head(500))

    return {
        "metrics": {
            "task_type": task_type,
            "train_score": round(train_score, 4),
            "test_score": round(test_score, 4),
            "score_metric": score_metric,
            "overfit_gap": overfit_gap,
            "n_features": X.shape[1],
            "n_samples": len(df),
        },
        "feature_importances": feature_importances,
        "diagnostics": diagnostics,
        "residuals_plot_b64": residuals_plot_b64,
        "shap_plot_b64": shap_plot_b64,
        "correlation_plot_b64": correlation_plot_b64,
    }


# ─── Plot helpers ─────────────────────────────────────────────────

def _plot_residuals(y_true, y_pred, residuals) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color=PALETTE["accent"], edgecolors="none", s=20)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], color=PALETTE["warn"], lw=1.5, linestyle="--", label="Perfect fit")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted", color=PALETTE["text"], fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # Residuals distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color=PALETTE["accent"], alpha=0.7, edgecolor="none")
    ax2.axvline(0, color=PALETTE["warn"], linestyle="--", lw=1.5, label="Zero line")
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", color=PALETTE["text"], fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_shap(feature_importances: list) -> str:
    names = [f["feature"] for f in feature_importances]
    values = [f["shap_mean_abs"] for f in feature_importances]
    directions = [f["direction"] for f in feature_importances]
    colors = [PALETTE["ok"] if d == "positive" else PALETTE["warn"] for d in directions]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    fig.patch.set_facecolor(PALETTE["bg"])

    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="none", height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (SHAP)", color=PALETTE["text"], fontsize=12, pad=12)
    ax.grid(True, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["ok"], label="Positive effect"),
        Patch(facecolor=PALETTE["warn"], label="Negative effect"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_correlation(X: pd.DataFrame) -> str:
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(max(6, len(X.columns) * 0.7), max(5, len(X.columns) * 0.65)))
    fig.patch.set_facecolor(PALETTE["bg"])

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=len(X.columns) <= 12,
        fmt=".2f", linewidths=0.5,
        linecolor=PALETTE["bg"],
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", color=PALETTE["text"], fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)
