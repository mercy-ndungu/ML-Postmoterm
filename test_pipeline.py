"""
test_pipeline.py
Runs a quick offline test of the ML engine without spinning up the server.
Uses the classic California Housing dataset (regression) and Iris (classification).

Run from the project root:
    python test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris

from app.services.ml_engine import run_analysis


def test_regression():
    print("\n=== REGRESSION TEST (California Housing) ===")
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1).head(500)

    results = run_analysis(df, target_col="MedHouseVal", task_type="regression")

    m = results["metrics"]
    print(f"  Train R²:   {m['train_score']:.4f}")
    print(f"  Test  R²:   {m['test_score']:.4f}")
    print(f"  Overfit gap: {m['overfit_gap']:.4f}")
    print(f"  Features:   {m['n_features']}")

    print("\n  Top 5 features by SHAP:")
    for f in results["feature_importances"][:5]:
        print(f"    {f['feature']:20s}  |SHAP|={f['shap_mean_abs']:.4f}  ({f['direction']})")

    print("\n  Diagnostics:")
    for d in results["diagnostics"]:
        print(f"    [{d['severity'].upper():8s}] {d['name']}")

    assert results["shap_plot_b64"] is not None, "SHAP plot missing"
    assert results["residuals_plot_b64"] is not None, "Residuals plot missing"
    print("  ✓ Plots generated")


def test_classification():
    print("\n=== CLASSIFICATION TEST (Iris) ===")
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("species")], axis=1)

    results = run_analysis(df, target_col="species", task_type="classification")

    m = results["metrics"]
    print(f"  Train Accuracy: {m['train_score']:.4f}")
    print(f"  Test  Accuracy: {m['test_score']:.4f}")
    print(f"  Overfit gap:    {m['overfit_gap']:.4f}")

    print("\n  Top features by SHAP:")
    for f in results["feature_importances"][:4]:
        print(f"    {f['feature']:25s}  |SHAP|={f['shap_mean_abs']:.4f}  ({f['direction']})")

    print("\n  Diagnostics:")
    for d in results["diagnostics"]:
        print(f"    [{d['severity'].upper():8s}] {d['name']}")

    assert results["shap_plot_b64"] is not None
    print("  ✓ Plots generated")


if __name__ == "__main__":
    test_regression()
    test_classification()
    print("\n✅ All tests passed.\n")
