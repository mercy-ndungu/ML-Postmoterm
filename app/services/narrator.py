"""
narrator.py
Calls the Anthropic API to generate a plain-English postmortem
from structured analysis results.
"""

import os
import httpx

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def _build_prompt(metrics: dict, feature_importances: list, diagnostics: list) -> str:
    top_features = feature_importances[:5]
    feature_lines = "\n".join(
        f"  - {f['feature']}: SHAP={f['shap_mean_abs']:.4f}, direction={f['direction']}"
        for f in top_features
    )

    diag_lines = "\n".join(
        f"  - [{d['severity'].upper()}] {d['name']}: {d['detail']}"
        for d in diagnostics
    )

    return f"""You are an expert ML engineer and educator reviewing a student's machine learning experiment.
Below are the results of their model. Write a clear, plain-English postmortem (no jargon unless explained).

Be honest, warm, and pedagogical. Structure your response in three sections:
1. **What your model learned** — summarise performance and what the top features mean
2. **Red flags to address** — explain the warnings/critical issues in plain language
3. **Your next steps** — 3 concrete, actionable improvements the student should try

Keep the total length to ~250–350 words. Do NOT repeat the raw numbers verbatim — interpret them.

=== EXPERIMENT RESULTS ===
Task type: {metrics['task_type']}
Score metric: {metrics['score_metric']}
Train score: {metrics['train_score']}
Test score: {metrics['test_score']}
Overfit gap: {metrics['overfit_gap']}
Features used: {metrics['n_features']}
Dataset size: {metrics['n_samples']} rows

Top features by SHAP importance:
{feature_lines}

Diagnostic flags:
{diag_lines}
"""


async def generate_narrative(
    metrics: dict,
    feature_importances: list,
    diagnostics: list,
    api_key: str | None = None,
) -> str:
    """
    Returns the AI-generated postmortem as a string.
    Falls back to a structured template if no API key is provided.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not key:
        return _fallback_narrative(metrics, diagnostics)

    prompt = _build_prompt(metrics, feature_importances, diagnostics)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    if response.status_code != 200:
        return _fallback_narrative(metrics, diagnostics)

    data = response.json()
    return data["content"][0]["text"]


def _fallback_narrative(metrics: dict, diagnostics: list) -> str:
    """Template-based fallback when no API key is available."""
    task = metrics["task_type"].title()
    metric = metrics["score_metric"]
    test_score = metrics["test_score"]
    gap = metrics["overfit_gap"]

    warnings = [d for d in diagnostics if d["severity"] in ("warning", "critical")]
    oks = [d for d in diagnostics if d["severity"] == "ok"]

    lines = [
        f"**{task} model trained successfully.**\n",
        f"Your model achieved a test {metric} of **{test_score:.4f}**.",
        f"The train/test gap is **{gap:.2%}** — {'this suggests overfitting; the model is memorising training data.' if gap > 0.1 else 'a healthy generalisation gap.'}",
        "",
    ]

    if warnings:
        lines.append("**Issues to address:**")
        for w in warnings:
            lines.append(f"- {w['name']}: {w['detail']}")
        lines.append("")

    lines.append("**What to do next:**")
    lines.append("1. Review the SHAP chart — focus on the top 2–3 features and ask whether they make real-world sense.")
    lines.append("2. If overfitting is flagged, try reducing the number of features or adding regularisation (Ridge/Lasso for regression).")
    lines.append("3. Collect more data if your dataset is small — even doubling it can significantly improve stability.")

    return "\n".join(lines)
