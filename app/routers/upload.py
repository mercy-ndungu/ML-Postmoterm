import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

# In-memory store for uploaded dataframes (keyed by session_id)
# In production, use Redis or a temp file store
_dataset_store: dict[str, pd.DataFrame] = {}


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file. Returns a session_id and a preview of columns + dtypes.
    Use the session_id in subsequent /analyze calls.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns and some rows.")

    # Simple session id from filename + shape
    import hashlib, time
    session_id = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()[:12]
    _dataset_store[session_id] = df

    # Build column summary
    col_info = []
    for col in df.columns:
        col_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist()
        })

    return {
        "session_id": session_id,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": col_info
    }


def get_dataframe(session_id: str) -> pd.DataFrame:
    """Helper used by the analysis router."""
    df = _dataset_store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Please upload your CSV first.")
    return df
