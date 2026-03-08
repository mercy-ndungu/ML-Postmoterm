from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalysisRequest, PostmortemResponse, MetricsSummary, FeatureImportance, DiagnosticFlag
from app.routers.upload import get_dataframe
from app.services.ml_engine import run_analysis
from app.services.narrator import generate_narrative

router = APIRouter()


@router.post("/analyze/{session_id}", response_model=PostmortemResponse)
async def analyze(session_id: str, request: AnalysisRequest):
    """
    Run the full ML postmortem on a previously uploaded dataset.

    - session_id: returned from POST /api/upload
    - target_column: the column you want to predict
    - task_type: "regression", "classification", or "auto" (default)
    - test_size: fraction of data held out for testing (default 0.2)
    - anthropic_api_key: optional — enables AI narrative generation
    """
    df = get_dataframe(session_id)

    try:
        results = run_analysis(
            df=df,
            target_col=request.target_column,
            task_type=request.task_type,
            test_size=request.test_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Generate AI narrative
    narrative = await generate_narrative(
        metrics=results["metrics"],
        feature_importances=results["feature_importances"],
        diagnostics=results["diagnostics"],
        api_key=request.anthropic_api_key,
    )

    return PostmortemResponse(
        status="success",
        metrics=MetricsSummary(**results["metrics"]),
        feature_importances=[FeatureImportance(**f) for f in results["feature_importances"]],
        diagnostics=[DiagnosticFlag(**d) for d in results["diagnostics"]],
        narrative=narrative,
        residuals_plot_b64=results.get("residuals_plot_b64"),
        shap_plot_b64=results.get("shap_plot_b64"),
        correlation_plot_b64=results.get("correlation_plot_b64"),
    )
